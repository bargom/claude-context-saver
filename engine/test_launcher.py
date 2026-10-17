"""Launcher (bin/context-saver) tests. A stub base python builds a fake venv in
a temporary data folder, so no package is downloaded and the real venv,
cache and config are never touched."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / 'bin' / 'context-saver'
ENGINE = ROOT / 'engine'
# A stand-in for python3: records each call, builds a venv folder holding a
# copy of itself, installs nothing, and prints the argv of a script run.
STUB = r'''#!/bin/sh
echo "$0 $*" >> "$STUB_LOG"
case ${1:-} in
  -c)
    if [ -n "${STUB_TOO_OLD:-}" ]; then exit 1; fi
    exit 0
    ;;
  -m)
    case $2 in
      venv) mkdir -p "$3/bin" && cp "$0" "$3/bin/python" && exit 0 ;;
      pip)
        if [ -n "${STUB_PIP_SLEEP:-}" ]; then sleep "$STUB_PIP_SLEEP"; fi
        if [ -n "${STUB_PIP_FAIL:-}" ]; then echo "ERROR: synthetic pip failure" >&2; exit 1; fi
        exit 0
        ;;
    esac
    ;;
esac
printf 'ARGV'
printf ' %s' "$@"
printf '\n'
'''


def sha12(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def age(path, seconds):
    """Sets the modification time of path to `seconds` ago."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.data = self.tmp / 'data'
        self.venv = self.data / f'venv-{sha12(ENGINE / "requirements.txt")}'
        self.mcp_stamp = self.venv / f'.mcp-{sha12(ENGINE / "requirements-mcp.txt")}'
        self.stub = self.tmp / 'stub' / 'python3'
        self.stub.parent.mkdir()
        self.stub.write_text(STUB)
        self.stub.chmod(0o755)
        self.log = self.tmp / 'stub.log'
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('CONTEXT_SAVER_', 'STUB_')) and key != 'XDG_DATA_HOME'}
        self.env.update(HOME=str(self.tmp), CONTEXT_SAVER_DATA=str(self.data),
                        CONTEXT_SAVER_BASE_PYTHON=str(self.stub), STUB_LOG=str(self.log),
                        TIKTOKEN_CACHE_DIR=str(self.tmp / 'tiktoken'), SHELL='/bin/sh',
                        CONTEXT_SAVER_CONFIG=str(self.tmp / 'config.json'),
                        CONTEXT_SAVER_STATE=str(self.tmp / 'state'))

    def tearDown(self):
        # A background setup must end before its folder is removed.
        self.wait_for(lambda: not (self.data / 'setup.lock').exists())

    def run_launcher(self, *args, env=None, input=None, timeout=30):
        return subprocess.run([str(LAUNCHER), *args], env={**self.env, **(env or {})}, input=input,
                              capture_output=True, text=True, timeout=timeout)

    @staticmethod
    def wait_for(predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(.05)
        return False

    def ready(self):
        return (self.venv / '.ready').exists() and not (self.data / 'setup.lock').exists()

    def leftovers(self):
        return sorted(path.name for path in self.data.glob('venv-*')
                      if '.partial-' in path.name or '.stale-' in path.name)

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def build(self, *args):
        result = self.run_launcher('setup', *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    # setup

    def test_setup_builds_the_venv_named_by_the_requirements_hash(self):
        result = self.build()
        self.assertEqual(result.stdout, f'context-saver: Python environment ready in {self.venv}\n')
        self.assertTrue((self.venv / '.ready').is_file())
        self.assertTrue(os.access(self.venv / 'bin' / 'python', os.X_OK))
        self.assertEqual(self.leftovers(), [])
        self.assertFalse((self.data / 'setup.lock').exists())
        self.assertFalse((self.data / 'setup-failed').exists())
        self.assertEqual(sorted(path.name for path in self.data.iterdir()), [self.venv.name])
        self.assertFalse(self.mcp_stamp.exists())
        pip = [line for line in self.calls() if ' -m pip ' in line]
        # Through the venv's own python (never a console script), with hashes.
        self.assertEqual(len(pip), 1)
        self.assertRegex(pip[0], rf'^{re.escape(str(self.venv))}\.partial-\d+/bin/python -m pip install '
                                 rf'--quiet --disable-pip-version-check --require-hashes '
                                 rf'-r {re.escape(str(ENGINE / "requirements.txt"))}$')

    def test_setup_mcp_adds_the_sdk_and_its_stamp(self):
        self.build('--mcp')
        self.assertTrue(self.mcp_stamp.is_file())
        installed = [line.rsplit(' -r ', 1)[1] for line in self.calls() if ' -m pip ' in line]
        self.assertEqual(installed, [str(ENGINE / 'requirements.txt'), str(ENGINE / 'requirements-mcp.txt')])

    def test_setup_mcp_on_a_ready_venv_installs_only_the_extra(self):
        self.build()
        self.log.unlink()
        self.build('--mcp')
        self.assertTrue(self.mcp_stamp.is_file())
        self.assertEqual([line for line in self.calls() if ' -m venv ' in line], [])
        installed = [line for line in self.calls() if ' -m pip ' in line]
        self.assertEqual(len(installed), 1)
        self.assertTrue(installed[0].startswith(f'{self.venv}/bin/python -m pip install '))
        self.assertTrue(installed[0].endswith(str(ENGINE / 'requirements-mcp.txt')))

    def test_setup_on_a_ready_venv_builds_nothing(self):
        self.build()
        self.log.unlink()
        self.build()
        self.assertEqual(self.calls(), [])

    def test_a_venv_left_without_ready_is_replaced(self):
        self.venv.mkdir(parents=True)
        (self.venv / 'half-built').write_text('')
        self.build()
        self.assertTrue((self.venv / '.ready').exists())
        self.assertFalse((self.venv / 'half-built').exists())
        self.assertEqual(self.leftovers(), [])

    def test_failed_setup_writes_setup_failed_and_keeps_no_venv(self):
        result = self.run_launcher('setup', env={'STUB_PIP_FAIL': '1'})
        self.assertEqual(result.returncode, 1)
        self.assertIn('synthetic pip failure', result.stderr)
        failed = (self.data / 'setup-failed').read_text()
        self.assertIn('could not install engine/requirements.txt', failed)
        self.assertIn('ERROR: synthetic pip failure', failed)
        self.assertFalse(self.venv.exists())
        self.assertEqual(self.leftovers(), [])
        self.assertFalse((self.data / 'setup.lock').exists())
        self.assertEqual(sorted(path.name for path in self.data.iterdir()), ['setup-failed'])
        # A later success removes the marker.
        self.build()
        self.assertFalse((self.data / 'setup-failed').exists())

    def test_too_old_python_fails_with_exit_2(self):
        result = self.run_launcher('setup', env={'STUB_TOO_OLD': '1'})
        self.assertEqual(result.returncode, 2)
        self.assertIn('needs Python 3.10 or newer', (self.data / 'setup-failed').read_text())
        self.assertFalse(self.venv.exists())

    def test_setup_prunes_unused_venvs_and_dead_partials_only(self):
        old, recent = self.data / 'venv-aaaaaaaaaaaa', self.data / 'venv-bbbbbbbbbbbb'
        dead, young = self.data / 'venv-cccccccccccc.partial-123', self.data / 'venv-dddddddddddd.partial-456'
        shared = self.data / 'venv'
        for folder in (old, recent, dead, young, shared):
            folder.mkdir(parents=True)
        for folder in (old, recent):
            (folder / '.ready').write_text('')
        age(old / '.ready', 40 * 86400)
        age(dead, 2 * 3600)
        self.build()
        self.assertEqual(sorted(path.name for path in self.data.iterdir()),
                         sorted(['venv', recent.name, young.name, self.venv.name]))

    def test_setup_waits_for_a_running_setup(self):
        lock = self.data / 'setup.lock'
        lock.mkdir(parents=True)
        started = time.monotonic()
        process = subprocess.Popen([str(LAUNCHER), 'setup'], env=self.env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(1.5)
            self.assertFalse((self.venv / '.ready').exists())
            self.assertIsNone(process.poll())
            lock.rmdir()
            stdout, stderr = process.communicate(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
        self.assertEqual(process.returncode, 0, stderr)
        self.assertGreaterEqual(time.monotonic() - started, 1.5)
        self.assertIn('ready in', stdout)
        self.assertTrue((self.venv / '.ready').exists())

    def test_a_stale_lock_is_broken(self):
        lock = self.data / 'setup.lock'
        lock.mkdir(parents=True)
        age(lock, 30 * 60)
        self.build()
        self.assertFalse(lock.exists())

    def test_usage_errors(self):
        self.assertEqual(self.run_launcher('setup', '--bogus').returncode, 2)
        self.assertEqual(self.run_launcher('ensure', '--bogus').returncode, 2)

    # ensure

    def test_ensure_builds_once_and_then_marks_the_venv_used(self):
        self.assertEqual(self.run_launcher('ensure').returncode, 0)
        self.assertTrue((self.venv / '.ready').exists())
        age(self.venv / '.ready', 10 * 86400)
        self.log.unlink()
        self.assertEqual(self.run_launcher('ensure').returncode, 0)
        self.assertEqual(self.calls(), [])
        self.assertLess(time.time() - (self.venv / '.ready').stat().st_mtime, 60)

    def test_ensure_background_starts_a_detached_setup(self):
        started = time.monotonic()
        # The caller's pipes close at once although pip takes 3 s.
        result = self.run_launcher('ensure', '--background', env={'STUB_PIP_SLEEP': '3'}, timeout=2)
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(result.returncode, 75)
        self.assertEqual(result.stdout, 'context-saver: setting up the Python environment in the background '
                                        f'(log: {self.data / "setup.log"})\n')
        self.assertTrue(self.wait_for(lambda: (self.data / 'setup.lock').exists(), 5))
        self.assertFalse((self.venv / '.ready').exists())
        # Running now: the next check reports it without starting another.
        again = self.run_launcher('ensure', '--background')
        self.assertEqual(again.returncode, 75)
        self.assertTrue(self.wait_for(self.ready))
        self.assertIn('ready in', (self.data / 'setup.log').read_text())
        self.assertEqual(len([line for line in self.calls() if ' -m venv ' in line]), 1)
        age(self.venv / '.ready', 10 * 86400)
        done = self.run_launcher('ensure', '--background')
        self.assertEqual((done.returncode, done.stdout, done.stderr), (0, '', ''))
        self.assertLess(time.time() - (self.venv / '.ready').stat().st_mtime, 60)

    def test_ensure_background_reports_a_recent_failure_then_retries(self):
        self.run_launcher('setup', env={'STUB_PIP_FAIL': '1'})
        result = self.run_launcher('ensure', '--background')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, (self.data / 'setup-failed').read_text()[-300:])
        self.assertIn('synthetic pip failure', result.stderr)
        self.assertFalse((self.data / 'setup.log').exists())
        age(self.data / 'setup-failed', 11 * 60)
        retry = self.run_launcher('ensure', '--background')
        self.assertEqual(retry.returncode, 75)
        self.assertTrue(self.wait_for(self.ready))
        self.assertFalse((self.data / 'setup-failed').exists())

    def test_ensure_background_with_a_held_lock_starts_nothing(self):
        (self.data / 'setup.lock').mkdir(parents=True)
        result = self.run_launcher('ensure', '--background')
        self.assertEqual(result.returncode, 75)
        time.sleep(.3)
        self.assertFalse((self.data / 'setup.log').exists())
        self.assertEqual(self.calls(), [])
        (self.data / 'setup.lock').rmdir()

    # actions before the venv is ready

    def test_hook_actions_exit_quietly_and_start_the_setup(self):
        for action in (('process',), ('pre-tool',), ('session-start',), ()):
            with self.subTest(action=action):
                result = self.run_launcher(*action, input='{"tool_name": "Bash"}')
                self.assertEqual((result.returncode, result.stdout, result.stderr), (0, '', ''))
        self.assertTrue(self.wait_for(self.ready))
        self.assertLessEqual(len([line for line in self.calls() if ' -m venv ' in line]), 2)

    def test_hook_actions_start_nothing_while_a_setup_runs(self):
        (self.data / 'setup.lock').mkdir(parents=True)
        result = self.run_launcher('process', input='{}')
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, '', ''))
        time.sleep(.3)
        self.assertFalse((self.data / 'setup.log').exists())
        (self.data / 'setup.lock').rmdir()

    def test_run_executes_the_command_directly(self):
        result = self.run_launcher('run', '--command', 'echo hi', '--session-id', 's1', '--tool-use-id', 't1')
        self.assertEqual((result.returncode, result.stdout), (0, 'hi\n'))
        failing = self.run_launcher('run', '--command=echo out; echo err >&2; exit 7')
        self.assertEqual((failing.returncode, failing.stdout, failing.stderr), (7, 'out\n', 'err\n'))
        self.assertTrue(self.wait_for(self.ready))

    def test_mcp_proxy_stdio_runs_the_wrapped_server_directly(self):
        result = self.run_launcher('mcp-proxy', '--server', 'x', '--stdio', 'echo', 'hi')
        self.assertEqual((result.returncode, result.stdout), (0, 'hi\n'))
        self.assertTrue(self.wait_for(self.ready))

    def test_mcp_proxy_http_without_the_venv_exits_3_and_sets_up_the_sdk(self):
        http = self.run_launcher('mcp-proxy', '--server', 'docs', '--http', 'https://example.test/mcp')
        self.assertEqual(http.returncode, 3)
        self.assertIn(f'run: {LAUNCHER} setup --mcp', http.stderr)
        self.assertTrue(self.wait_for(lambda: self.ready() and self.mcp_stamp.is_file()))
        self.assertTrue((self.data / 'mcp-wanted').is_file())

    def test_other_actions_need_the_venv(self):
        for action in (('stats',), ('delivered', '--delivery-token', 'cs-1'), ('web-adapter',), ('run',)):
            with self.subTest(action=action):
                result = self.run_launcher(*action)
                self.assertEqual(result.returncode, 3)
                self.assertIn(f'run: {LAUNCHER} setup', result.stderr)
        time.sleep(.3)
        self.assertFalse((self.data / 'setup.log').exists())

    # actions with a ready venv

    def test_ready_actions_go_to_the_engine(self):
        self.build()
        cases = [(('delivered', '--delivery-token', 'cs-1'), f'{ENGINE}/hook.py delivered --delivery-token cs-1'),
                 (('run', '--command', 'echo hi'), f'{ENGINE}/hook.py run --command echo hi'),
                 (('mcp-proxy', '--server', 'x', '--stdio', 'echo', 'hi'),
                  f'{ENGINE}/mcp_proxy.py --server x --stdio echo hi'),
                 (('web-adapter', '--decision-only'), f'{ENGINE}/web_adapter.py --decision-only')]
        for args, expected in cases:
            with self.subTest(args=args):
                result = self.run_launcher(*args)
                self.assertEqual((result.returncode, result.stdout), (0, f'ARGV {expected}\n'), result.stderr)

    def test_session_start_marks_the_venv_used(self):
        self.build()
        age(self.venv / '.ready', 10 * 86400)
        result = self.run_launcher('session-start')
        self.assertEqual(result.stdout, f'ARGV {ENGINE}/hook.py session-start\n')
        self.assertLess(time.time() - (self.venv / '.ready').stat().st_mtime, 60)

    def test_mcp_proxy_http_needs_the_mcp_extra(self):
        self.build()
        args = ('mcp-proxy', '--server', 'docs', '--http', 'https://example.test/mcp')
        missing = self.run_launcher(*args)
        self.assertEqual(missing.returncode, 3)
        self.assertIn(f'run: {LAUNCHER} setup --mcp', missing.stderr)
        # The missing SDK is set up in the background; the next start proxies.
        self.assertTrue(self.wait_for(lambda: self.ready() and self.mcp_stamp.is_file()))
        result = self.run_launcher(*args)
        self.assertEqual(result.stdout, f'ARGV {ENGINE}/mcp_proxy.py --server docs --http https://example.test/mcp\n')

    def test_a_rebuilt_venv_keeps_the_mcp_sdk_once_asked_for(self):
        self.build('--mcp')
        shutil.rmtree(self.venv)
        self.build()
        self.assertTrue(self.mcp_stamp.is_file())

    def test_python_override_is_ready_without_a_venv(self):
        env = {'CONTEXT_SAVER_PYTHON': str(self.stub)}
        self.assertEqual(self.run_launcher('ensure', '--background', env=env).returncode, 0)
        result = self.run_launcher('mcp-proxy', '--server', 'docs', '--http', 'https://example.test/mcp', env=env)
        self.assertEqual(result.stdout, f'ARGV {ENGINE}/mcp_proxy.py --server docs --http https://example.test/mcp\n')
        self.assertFalse(self.data.exists())

    def test_a_linked_launcher_finds_the_plugin_folder(self):
        self.build()
        link = self.tmp / 'bin' / 'context-saver'
        link.parent.mkdir()
        link.symlink_to(LAUNCHER)
        result = subprocess.run([str(link), 'delivered', '--delivery-token', 'cs-1'], env=self.env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.stdout, f'ARGV {ENGINE}/hook.py delivered --delivery-token cs-1\n')


class RequirementsTests(unittest.TestCase):
    def test_the_mcp_sdk_is_an_extra_with_the_engine_pins(self):
        core = (ENGINE / 'requirements.txt').read_text()
        extra = (ENGINE / 'requirements-mcp.txt').read_text()
        pins = lambda text: dict(re.findall(r'^([a-z0-9][a-z0-9._-]*)==(\S+)', text, re.M))
        self.assertIn('tiktoken', pins(core))
        self.assertNotIn('mcp', pins(core))
        self.assertEqual(pins(extra)['mcp'], '2.2.0')
        self.assertNotIn('tiktoken', pins(extra))
        # Compiled with -c requirements.txt: a shared package has one version.
        for name in set(pins(core)) & set(pins(extra)):
            self.assertEqual(pins(core)[name], pins(extra)[name], name)
        for text in (core, extra):
            self.assertNotIn('/Users/', text)
            self.assertTrue(all('--hash=sha256:' in block for block in re.split(r'\n(?=[a-z])', text)[1:]))


@unittest.skipUnless(os.environ.get('CONTEXT_SAVER_TEST_REAL_SETUP') == '1',
                     'downloads packages; set CONTEXT_SAVER_TEST_REAL_SETUP=1 to run')
class RealSetupTests(unittest.TestCase):
    def test_real_venv_runs_the_engine_after_the_move(self):
        base = shutil.which('python3')
        with tempfile.TemporaryDirectory() as folder:
            env = {key: value for key, value in os.environ.items() if not key.startswith('CONTEXT_SAVER_')}
            env.update(HOME=folder, CONTEXT_SAVER_DATA=str(Path(folder) / 'data'),
                       CONTEXT_SAVER_BASE_PYTHON=base, CONTEXT_SAVER_CONFIG=str(Path(folder) / 'config.json'),
                       CONTEXT_SAVER_STATE=str(Path(folder) / 'state'))
            for args in (('setup',), ('setup', '--mcp')):
                result = subprocess.run([str(LAUNCHER), *args], env=env, capture_output=True, text=True, timeout=600)
                self.assertEqual(result.returncode, 0, result.stderr)
            limits = subprocess.run([str(LAUNCHER), 'limits'], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(json.loads(limits.stdout)['minimum_tokens'], 3500)
            venv = next((Path(folder) / 'data').glob('venv-*'))
            probe = subprocess.run([str(venv / 'bin' / 'python'), '-c',
                                    'import tiktoken, mcp, mcp_types; print(len(tiktoken.get_encoding("o200k_base").encode("hello world")))'],
                                   env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(probe.stdout.strip(), '2', probe.stderr)


if __name__ == '__main__':
    unittest.main()
