import json
import os
import re
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import common
import hook
import summary_engine


def isolated_env(folder, **extra):
    """The environment of a subprocess test: its own HOME, state, config and
    data folders, so it never reads the developer's config.json or token file."""
    folder = Path(folder)
    env = {**os.environ, 'HOME': str(folder / 'home'), 'XDG_CONFIG_HOME': str(folder / 'home' / '.config'),
           'CONTEXT_SAVER_STATE': str(folder / 'state'), 'CONTEXT_SAVER_CONFIG': str(folder / 'config.json'),
           'CONTEXT_SAVER_DATA': str(folder / 'data')}
    for name in ('CONTEXT_SAVER_EFFORT', 'CONTEXT_SAVER_SUMMARIZE_SHELL', 'CONTEXT_SAVER_ACTIVE', 'CONTEXT_SAVER_RAW'):
        env.pop(name, None)
    env.update(extra)
    return env


def fake_claude(folder):
    """A `claude` on PATH that writes its pid and sleeps: a summary helper that
    is still running. Returns the environment that finds it and the pid file."""
    pid_file = Path(folder) / 'helper.pid'
    fake = Path(folder) / 'bin' / 'claude'
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\necho $$ > {shlex.quote(str(pid_file))}\nexec sleep 60\n')
    fake.chmod(0o755)
    return isolated_env(folder, PATH=f'{fake.parent}{os.pathsep}{os.environ["PATH"]}'), pid_file


def ledger(state):
    path = Path(state) / 'events.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def wait_for_pid(pid_file, timeout=10):
    deadline = time.monotonic() + timeout
    while not (pid_file.exists() and pid_file.read_text().strip()) and time.monotonic() < deadline:
        time.sleep(.05)
    return int(pid_file.read_text())


def is_gone(pid, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(.05)
    return False


class HookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)
        self.state_patch = patch.object(hook, 'STATE', self.state)
        self.state_patch.start()
        # Never read the developer's own config.json.
        self.config_patch = patch.object(hook, 'CONFIG_PATH', self.state / 'config.json')
        self.config_patch.start()
        if not hasattr(summary_engine, 'prune_cache'):
            # Stream C adds it; until then session_start needs a stand-in.
            prune = patch.object(summary_engine, 'prune_cache', lambda *args: 0, create=True)
            prune.start()
            self.addCleanup(prune.stop)
        self.cfg = dict(hook.DEFAULTS, minimum_tokens=1000)

    def tearDown(self):
        self.config_patch.stop()
        self.state_patch.stop()
        self.tmp.cleanup()

    def env(self, **extra):
        return isolated_env(self.state, **extra)

    def event(self, command='rg -n handler src', text=None):
        return {'hook_event_name': 'PostToolUse', 'tool_name': 'Bash', 'session_id': 'test',
                'tool_use_id': 'call1', 'tool_input': {'command': command},
                'tool_response': {'output': text or ('src/a.js:12: function handler() { return 1; }\n' * 300), 'exit_code': 0}}

    def engine(self, **kwargs):
        return {'status': 'summary', 'summary': 'src/a.js:12 defines handler returning 1.', 'cache_hit': False,
                'usage': {'input_tokens': 2000, 'output_tokens': 100, 'total_tokens': 2100}, 'elapsed_ms': 1}

    def test_tokens_match_exact_counts_for_short_text(self):
        import tiktoken
        encoding = tiktoken.get_encoding('o200k_base')
        for text in ['', 'src/a.py:12: ordinary output\nTests passed.\n', 'café 東京 🧬',
                     '<|endoftext|>', ('ordinary output\n' * 200)[:hook.TOKEN_SLICE_CHARS - 1]]:
            with self.subTest(text=text[:40]):
                self.assertEqual(hook.tokens(text), len(encoding.encode(text, disallowed_special=())))
                self.assertLessEqual(hook.tokens(text), len(text.encode('utf-8')))

    def test_tokens_bound_large_repetitive_input(self):
        text = 'x' * 4_000_000
        started = time.monotonic()
        count = hook.tokens(text)
        self.assertLess(time.monotonic() - started, 5)
        self.assertGreater(count, 0)
        self.assertLessEqual(count, len(text.encode('utf-8')))

    def test_tokens_never_exceed_utf8_bytes_across_slices(self):
        for text in ['café 東京 🧬\n' * 1000, 'x' * (hook.TOKEN_SLICE_CHARS - 1) + '🧬é東京']:
            with self.subTest(text=text[:40]):
                self.assertLessEqual(hook.tokens(text), len(text.encode('utf-8')))

    def test_compresses_and_accounts_raw_retrieval(self):
        out = hook.process(self.event(), self.cfg, self.engine)
        self.assertFalse(out['continue'])
        self.assertIn('Original output:', out['stopReason'])
        self.assertEqual(len(list((self.state / 'raw').glob('*.txt'))), 1)
        report = hook.stats()
        self.assertEqual(report['compressed_outputs'], 1)
        self.assertEqual(report['estimated_context_tokens_saved'], report['estimated_before_tokens'] - report['estimated_after_tokens'])
        self.assertEqual(report['helper_usage']['total_tokens'], 2100)

    def test_provider_and_cache_labels_match_complete_payload_ledger(self):
        # Claude is the only summary helper, whatever provider name a result carries.
        cases = [('claude_cli', 'sonnet', False, 'Claude (sonnet)'),
                 ('claude_cli', 'sonnet', True, 'Claude (sonnet)'),
                 ('other', 'custom-model', False, 'Claude (custom-model)'),
                 ('other', None, False, 'Claude')]
        for provider, model, cached, producer in cases:
            with self.subTest(provider=provider, cached=cached, model=model):
                result = dict(self.engine(), provider=provider, model=model, cache_hit=cached,
                              status='cache_hit' if cached else 'summary')
                event = self.event('npm test')
                event['tool_response']['exit_code'] = 1
                suffix = 'Exact sources: https://example.com [wordlim 200]'
                output = hook.process(event, self.cfg, lambda **kw: result, output_suffix=suffix)['stopReason']
                row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
                self.assertTrue(output.startswith(f"[{row['saved_tokens']} tokens saved by {producer}]"))
                self.assertIn('estimated context tokens', output.splitlines()[0])
                self.assertEqual('cache hit' in output.splitlines()[0], cached)
                self.assertEqual(row['cache_hit'], cached)
                self.assertEqual(row['after_tokens'], hook.tokens(output))
                self.assertEqual(row['saved_tokens'], hook.tokens(event['tool_response']['output']) - hook.tokens(output))
                self.assertIn('Exact failure excerpts (original):', output)
                self.assertTrue(output.endswith(suffix))

    def test_label_cost_is_included_at_acceptance_boundaries(self):
        output = hook.process(self.event(), self.cfg, self.engine)['stopReason']
        after = hook.tokens(output)
        without_label = hook.tokens(output.split('\n', 1)[1])
        self.assertLess(without_label, after)
        before = hook.tokens(self.event()['tool_response']['output'])
        limits = [dict(maximum_replacement_tokens=after - 1),
                  dict(minimum_saving_fraction=(before - (after + without_label) / 2) / before)]
        for limit in limits:
            with self.subTest(limit=limit):
                self.assertIsNone(hook.process(self.event(), dict(self.cfg, **limit), self.engine))
                row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
                self.assertEqual(row['status'], 'insufficient_saving')
                self.assertEqual(row['candidate_after_tokens'], after)
                self.assertEqual(row['saved_tokens'], 0)
        self.assertIsNotNone(hook.process(self.event(), dict(self.cfg, maximum_replacement_tokens=after), self.engine))

    def test_label_digit_boundary_and_nonconvergent_accounting(self):
        # Force iteration into a cycle (1000 -> 999 -> 1000), with a nearby
        # valid fixed point at 998 that the bounded search must find.
        def count(text):
            match = re.match(r'\[(\d+) tokens saved', text)
            if not match:
                return 1000
            saved = int(match[1])
            return {1000: 1001, 999: 1000, 998: 1002}.get(saved, 1000)
        with patch.object(hook, 'tokens', side_effect=count):
            labelled, after, saved = hook.label_savings('payload', 2000, self.engine())
            self.assertEqual(saved, 998)
            self.assertEqual(after, 1002)
            self.assertTrue(labelled.startswith('[998 tokens saved'))

        def no_fixed_point(text):
            match = re.match(r'\[(\d+) tokens saved', text)
            return 1001 if match and int(match[1]) == 1000 else 1000
        with patch.object(hook, 'tokens', side_effect=no_fixed_point):
            self.assertIsNone(hook.label_savings('payload', 2000, self.engine()))
        with patch.object(hook, 'label_savings', return_value=None):
            self.assertIsNone(hook.process(self.event(), self.cfg, self.engine))
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
        self.assertEqual(row['reason'], 'label_accounting_unstable')
        self.assertEqual(row['saved_tokens'], 0)
        self.assertEqual(hook.stats()['compressed_outputs'], 0)

    def test_small_exact_reads_and_bypass_pass_through(self):
        for command in ['cat src/a.js', 'sed -n 1,200p src/a.js', 'CONTEXT_SAVER_RAW=1 rg handler src']:
            self.assertIsNone(hook.process(self.event(command), self.cfg, self.engine))
        self.assertIsNone(hook.process(self.event(text='small'), self.cfg, self.engine))
        e = self.event();e['tool_name'] = 'mcp__codebase_memory_mcp__get_code_snippet'
        self.assertIsNone(hook.process(e, self.cfg, self.engine))

    def test_sensitive_and_oversize_not_sent(self):
        def never(**kwargs):
            self.fail('engine must not receive excluded output')
        self.assertIsNone(hook.process(self.event(text='api_key="' + 'secret12345' * 400 + '"'), self.cfg, never))
        self.assertIsNone(hook.process(self.event('rg thing .env'), self.cfg, never))
        out = hook.process(self.event(text='long line\n' * 20000), self.cfg, never)
        self.assertIn('tokens saved by local compaction', out['stopReason'])
        self.assertLessEqual(hook.tokens(out['stopReason']), self.cfg['maximum_replacement_tokens'])

    def test_oversized_repetitive_output_finishes_quickly(self):
        def never(**kwargs):
            self.fail('preview needs no helper')
        event = self.event(text='x' * 300_000)
        started = time.monotonic()
        out = hook.process(event, self.cfg, never)
        self.assertLess(time.monotonic() - started, 5)
        self.assertIn('tokens saved by local compaction', out['stopReason'])
        self.assertIn('ARCHIVE PREVIEW', out['stopReason'])
        self.assertLessEqual(hook.tokens(out['stopReason']), self.cfg['maximum_replacement_tokens'])
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
        self.assertEqual(Path(row['raw_file']).read_text(), event['tool_response']['output'])
        self.assertEqual(row['saved_tokens'], hook.tokens(event['tool_response']['output']) - hook.tokens(out['stopReason']))

    def test_bearer_schemes_are_never_sent(self):
        def never(**kwargs):
            self.fail('engine must not receive excluded output')
        for scheme in ('bearer', 'BEARER'):
            with self.subTest(scheme=scheme):
                text = self.event()['tool_response']['output'] + (
                    f'Authorization: {scheme} eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijkl\n')
                self.assertIsNone(hook.process(self.event(text=text), self.cfg, never))

    def test_failed_helper_and_structured_mcp_preserve_result(self):
        self.assertIsNone(hook.process(self.event(), self.cfg, lambda **kw: {'status': 'error', 'error': 'timeout'}))
        e = self.event();e['tool_name'] = 'mcp__codebase_memory_mcp__search_graph'
        e['tool_response'] = {'content': [{'type': 'text', 'text': 'x' * 20000}], 'structuredContent': {'value': 'exact'}}
        self.assertIsNone(hook.process(e, self.cfg, self.engine))

    def oversized_graph_event(self):
        rows = [{'name': f'handler_{i}', 'label': 'Function', 'file_path': f'src/worker_{i}.js',
                 'qualified_name': 'project.' + 'shared_package.' * 45 + f'handler_{i}',
                 'signature': '(input)', 'lines': 12, 'complexity': 1,
                 'docstring': 'Recorded declaration only.'} for i in range(150)]
        text = json.dumps({'total': 150, 'results': rows, 'has_more': False})
        event = self.event(text=text)
        event.update(tool_name='mcp__codebase_memory_mcp__search_graph',
                     tool_input={'name_pattern': 'handler.*'})
        return event

    def test_oversized_graph_compacts_every_identity_before_helper(self):
        event = self.oversized_graph_event()
        def helper(**kwargs):
            self.assertLessEqual(len(kwargs['text']), self.cfg['maximum_source_chars'])
            self.assertIn('handler_149', kwargs['text'])
            self.assertIn('line_count', kwargs['text'])
            return dict(self.engine(), provider='claude_cli', model='sonnet')
        output = hook.process(event, self.cfg, helper)['stopReason']
        self.assertIn('tokens saved by Claude (sonnet) + local compaction', output)
        self.assertIn('Local compaction coverage:', output)
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
        self.assertEqual(row['coverage']['shown'], 150)
        self.assertEqual(Path(row['raw_file']).read_text(), event['tool_response']['output'])
        self.assertEqual(row['saved_tokens'], hook.tokens(event['tool_response']['output']) - hook.tokens(output))

    def test_oversized_helper_failure_and_overbudget_use_bounded_archive_preview(self):
        event = self.oversized_graph_event()
        for failure in [dict(self.engine(), status='error', error='model_timeout'),
                        dict(self.engine(), summary='excessive output ' * 3000)]:
            with self.subTest(status=failure['status']):
                output = hook.process(event, self.cfg, lambda **kw: failure)['stopReason']
                self.assertIn('tokens saved by local compaction', output)
                self.assertNotIn('tokens saved by Claude', output)
                self.assertIn('ARCHIVE PREVIEW', output)
                self.assertLessEqual(hook.tokens(output), self.cfg['maximum_replacement_tokens'])
                row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
                self.assertLess(row['coverage']['shown'], 150)
                self.assertEqual(row['helper_usage']['total_tokens'], 2100)
                self.assertEqual(Path(row['raw_file']).read_text(), event['tool_response']['output'])

    def test_oversized_unknown_structured_contract_stays_raw(self):
        event = self.event(text=json.dumps({'machine_contract': 'value' * 15000}))
        self.assertIsNone(hook.process(event, self.cfg, lambda **kw: self.fail('must not summarize contract')))

    def test_oversized_token_dense_failure_stays_bounded_and_reports_omission(self):
        event = self.event('npm test', text='ERROR ' + '\U0001f9ec' * 60001)
        event['tool_response']['exit_code'] = 1
        out = hook.process(event, self.cfg, lambda **kw: self.fail('preview needs no helper'))['stopReason']
        self.assertIn('exit_code=1', out)
        self.assertIn('preview omitted', out)
        self.assertLessEqual(hook.tokens(out), 1800)
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
        self.assertEqual(row['coverage']['shown_chars'], 0)
        self.assertEqual(row['coverage']['omitted_chars'], len(event['tool_response']['output']))

    def test_graph_notice_only_records_zero_visible_rows(self):
        event = self.oversized_graph_event()
        out = hook.process(event, dict(self.cfg, maximum_replacement_tokens=300),
                           lambda **kw: {'status': 'error', 'error': 'model_timeout'})['stopReason']
        self.assertIn('preview omitted', out)
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
        self.assertEqual(row['coverage']['shown'], 0)
        self.assertEqual(row['coverage']['remaining'], 150)

    def test_preserves_exit_status_and_exact_failure(self):
        e = self.event('npm test');e['tool_response']['exit_code'] = 1
        e['tool_response']['output'] += '\nERROR exact failure marker 123\n'
        out = hook.process(e, self.cfg, self.engine)
        self.assertIn('exit_code=1', out['stopReason'])
        self.assertIn('ERROR exact failure marker 123', out['stopReason'])

    def claude_bash(self, command='npm test', output=None, stderr='', **extra):
        """A Claude Code Bash event: no exit_code, stderr given apart (C1)."""
        output = output or ('src/a.js:12: function handler() { return 1; }\n' * 300)
        return {'hook_event_name': 'PostToolUse', 'tool_name': 'Bash', 'session_id': 'cc', 'tool_use_id': 'toolu_1',
                'tool_input': {'command': command},
                'tool_response': {'output': output + ('\n' + stderr if stderr else ''), 'stderr': stderr, **extra}}

    def test_unknown_exit_code_is_not_reported_as_zero(self):
        out = hook.process(self.claude_bash('rg -n handler src'), self.cfg, self.engine)['stopReason'].splitlines()
        self.assertTrue(out[1].startswith('[Context summary via helper: search; isError=False; estimated original tokens='))
        self.assertNotIn('exit_code', out[1])
        event = self.claude_bash('rg -n handler src')
        event['tool_response']['exit_code'] = 0
        self.assertIn('; exit_code=0; ', hook.process(event, self.cfg, self.engine)['stopReason'].splitlines()[1])

    def test_original_output_is_line_three(self):
        engine = lambda **kw: dict(self.engine(), summary='Original output: /not/the/archive\n- handler at src/a.js:12')
        out = hook.process(self.event(), self.cfg, engine)['stopReason']
        lines = out.splitlines()
        row = ledger(self.state)[-1]
        self.assertEqual(lines[2], f"Original output: {row['raw_file']}")
        self.assertEqual(re.search(r'Original output: (\S+)', out)[1], row['raw_file'])
        self.assertEqual(lines[3], 'Original output: /not/the/archive')

    def test_error_lines_in_a_log_get_excerpts_without_an_exit_code(self):
        log = ('PASS src/ok.test.js\n' * 400) + 'Error: exact database failure 42\n    at db.js:9\n' + ('done\n' * 5)
        out = hook.process(self.claude_bash('pytest -q', log), self.cfg, self.engine)['stopReason']
        self.assertIn('Exact failure excerpts (original):', out)
        self.assertIn('Error: exact database failure 42\n    at db.js:9', out)
        # A search result is not a log: no excerpts without a known failure.
        text = 'src/error_handler.js:3: function onError() {}\n' * 300
        out = hook.process(self.claude_bash('rg -n onError src', text), self.cfg, self.engine)['stopReason']
        self.assertNotIn('Exact failure excerpts', out)

    def test_stderr_tail_is_kept_exactly_and_bounded(self):
        stderr = 'x' * 1000 + '\n' + ''.join(f'warning {i}: deprecated option\n' for i in range(20))
        out = hook.process(self.claude_bash('rg -n handler src', stderr=stderr), self.cfg, self.engine)['stopReason']
        tail = out.split('Exact stderr tail (original):\n', 1)[1]
        self.assertEqual(tail, '\n'.join(f'warning {i}: deprecated option' for i in range(12, 20)))
        out = hook.process(self.claude_bash('rg -n handler src', stderr='y' * 1000), self.cfg, self.engine)['stopReason']
        self.assertEqual(out.split('Exact stderr tail (original):\n', 1)[1], 'y' * 800)
        # Not repeated when the failure excerpts already hold it.
        short = 'fatal: exact stderr line'
        event = self.claude_bash('rg -n handler src', stderr=short)
        event['tool_response']['exit_code'] = 2
        out = hook.process(event, self.cfg, self.engine)['stopReason']
        self.assertIn('Exact failure excerpts (original):', out)
        self.assertEqual(out.count(short), 1)
        self.assertNotIn('Exact stderr tail', out)

    def test_extract_keeps_only_reported_metadata(self):
        text, metadata = hook.extract({'output': 'x', 'stderr': '', 'return_code_interpretation': 'No matches found'})
        self.assertEqual((text, metadata), ('x', {'return_code_interpretation': 'No matches found'}))
        _, metadata = hook.extract({'output': 'x', 'stderr': 'e', 'exit_code': 1, 'wall_time_seconds': 2.5,
                                    'return_code_interpretation': None})
        self.assertEqual(metadata, {'stderr': 'e', 'exit_code': 1})
        _, metadata = hook.extract({'output': 'x', 'isError': True})
        self.assertEqual(metadata, {'isError': True})
        self.assertEqual(hook.extract({'output': 'x', 'isError': 'yes'})[1], {})
        for content in ([{'type': 'text', 'text': 'a'}, 42], [], [{'type': 'text', 'text': 'a', '_meta': {}}]):
            self.assertEqual(hook.extract({'content': content}), ('', {'unsupported_content': True}))
        self.assertEqual(hook.extract({'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}],
                                       'isError': True}), ('a\nb', {'isError': True}))

    def test_shell_summary_context_is_the_command_only(self):
        contexts = []
        engine = lambda **kw: contexts.append(kw['context']) or self.engine()
        for extra in ({}, {'description': 'Find handlers', 'timeout': 60000}):
            event = self.event()
            event['tool_input'].update(extra)
            hook.process(event, self.cfg, engine)
        codex = dict(self.event(), tool_name='exec_command', tool_input={'cmd': 'rg -n handler src', 'yield_time_ms': 1000})
        hook.process(codex, self.cfg, engine)
        self.assertEqual(contexts[:2], ['Tool: Bash; command: rg -n handler src'] * 2)
        self.assertEqual(contexts[2], 'Tool: exec_command; command: rg -n handler src')
        mcp = dict(self.event(), tool_name='mcp__codebase_memory_mcp__search_code', tool_input={'pattern': 'handler', 'limit': 5},
                   tool_response={'content': [{'type': 'text', 'text': self.event()['tool_response']['output']}]})
        hook.process(mcp, dict(self.cfg, deterministic_first=False), engine)
        self.assertEqual(contexts[3], 'Tool: mcp__codebase_memory_mcp__search_code; request: {"pattern": "handler", "limit": 5}')

    def test_small_output_returns_before_the_credential_scan(self):
        for text in ('api_key="secret123456789"\n', 'x' * 999):
            self.assertIsNone(hook.process(self.event(text=text), self.cfg, self.engine))
        self.assertEqual(ledger(self.state), [])
        # The byte length decides: 1000 bytes of emoji are far fewer characters.
        self.assertIsNone(hook.process(self.event(text='api_key="secret123456789" ' + '\U0001f9ec' * 250), self.cfg, self.engine))
        self.assertEqual(ledger(self.state)[-1]['status'], 'skipped_sensitive')

    def test_persisted_output_rows_and_stats(self):
        event = dict(self.event(), persisted_output=True)
        hook.process(event, self.cfg, self.engine)
        hook.process(dict(event, tool_use_id='call2'), self.cfg, lambda **kw: {'status': 'error', 'error': 'model_timeout'})
        hook.process(self.event('rg -n other src'), self.cfg, self.engine)
        rows = ledger(self.state)
        self.assertEqual([row.get('persisted') for row in rows], [True, True, None])
        report = hook.stats()
        compressed = [row for row in rows if row['status'] == 'compressed']
        self.assertEqual(report['compressed_outputs'], 2)
        self.assertEqual(report['persisted_outputs'], 1)
        self.assertEqual(report['persisted_tokens_saved'], compressed[0]['before_tokens'] - compressed[0]['after_tokens'])
        self.assertIn('persisted_tokens_saved', report['limits'])
        self.assertIn('full output', report['limits'])

    def test_archive_is_written_only_for_an_accepted_replacement(self):
        raw = self.state / 'raw'
        self.assertIsNone(hook.process(self.event(), dict(self.cfg, maximum_replacement_tokens=200),
                                       lambda **kw: dict(self.engine(), summary='long summary ' * 400)))
        self.assertEqual(ledger(self.state)[-1]['status'], 'insufficient_saving')
        self.assertEqual(list(raw.glob('*')) if raw.exists() else [], [])
        self.assertIsNone(hook.process(self.event(), self.cfg, lambda **kw: {'status': 'error', 'error': 'model_timeout'}))
        self.assertEqual(list(raw.glob('*')) if raw.exists() else [], [])
        out = hook.process(self.event(), self.cfg, self.engine)['stopReason']
        path = Path(ledger(self.state)[-1]['raw_file'])
        self.assertEqual(list(raw.iterdir()), [path])
        self.assertEqual(path.read_text(), self.event()['tool_response']['output'])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertIn(str(path), out)
        # The same text again (another writer, or an earlier call) is success, unchanged.
        stamp = time.time() - 3 * 86400
        os.utime(path, (stamp, stamp))
        self.assertIsNotNone(hook.process(self.event(), self.cfg, self.engine))
        self.assertEqual(list(raw.iterdir()), [path])
        self.assertGreater(path.stat().st_mtime, stamp + 86400)
        # A concurrent writer linked the file first: FileExistsError is success.
        with patch.object(hook.Path, 'exists', return_value=False):
            self.assertEqual(hook.archive(self.event()['tool_response']['output']), path)
        self.assertEqual(list(raw.iterdir()), [path])

    def test_bracketed_logs_over_the_size_limit_get_a_preview(self):
        never = lambda **kw: self.fail('a preview needs no helper')
        for line in ('[INFO] worker finished a batch\n', '[2026-10-17 12:00:00] worker finished a batch\n'):
            with self.subTest(line=line):
                out = hook.process(self.event('tail -n 5000 app.log', line * 3000), self.cfg, never)
                self.assertIsNotNone(out)
                self.assertIn('ARCHIVE PREVIEW', out['stopReason'])
        cut_off = json.dumps([{'id': i, 'name': f'row {i}'} for i in range(5000)])[:-40]
        self.assertIsNone(hook.process(self.event('tail -n 5000 app.log', cut_off), self.cfg, never))
        self.assertEqual(ledger(self.state)[-1]['status'], 'skipped_oversize')

    def test_secret_patterns_and_references(self):
        # Built from pieces, so no complete token-shaped literal sits in this file.
        fake = 'Fake0' * 8
        secrets = ['glpat' + '-' + fake[:20], 'xox' + 'b-1234567890-' + fake[:12], 'xox' + 'p-1234567890-' + fake[:12],
                   'eyJ' + fake[:20] + '.eyJ' + fake[:20] + '.' + fake[:24],
                   'mongodb+srv://admin:' + fake[:10] + '@cluster0.example.net/db', 'postgres://app:' + fake[:6] + '@db:5432/app',
                   'aws_secret_access_key = ' + fake, 'AWS_SECRET_ACCESS_KEY=' + fake,
                   'sk' + '_live_' + fake[:24], 'rk' + '_test_' + fake[:24],
                   'npm' + '_' + fake[:36], '-----BEGIN PGP PRIVATE KEY BLOCK-----',
                   'GITHUB_TOKEN=abcdef123456', 'API_SECRET: "s3cr3t-value-x"', 'TOKEN = abcdefgh12345',
                   'export SLACK_BOT_TOKEN="xyz12345abc"']
        references = ['GITHUB_TOKEN=$GITHUB_TOKEN', 'GITHUB_TOKEN="${{ secrets.GITHUB_TOKEN }}"',
                      'API_TOKEN = process.env.API_TOKEN', 'SECRET=os.environ["SECRET"]', "TOKEN: ENV['TOKEN']",
                      'TOKEN: {{ .Values.token }}', 'aws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}',
                      'postgres://app:$DB_PASSWORD@db:5432/app', 'MAX_TOKEN = 128000', 'TOKEN_SLICE_CHARS = 2048',
                      'https://example.com:8080/path@v2', 'git@github.com:org/repo.git', 'ssh://git@github.com/org/repo']
        for value in secrets:
            with self.subTest(secret=value):
                self.assertTrue(hook.sensitive(value))
                self.assertTrue(hook.sensitive(json.dumps({'text': value})))
        for value in references:
            with self.subTest(reference=value):
                self.assertFalse(hook.sensitive(value))

    def test_secret_scan_stays_linear_on_long_runs(self):
        for text in ('A' * 300_000, 'a.' * 150_000, 'a1+' * 100_000):
            started = time.monotonic()
            self.assertFalse(hook.sensitive(text))
            self.assertLess(time.monotonic() - started, 5, text[:10])

    def test_classifies_tool_categories(self):
        for cmd, category in [('git diff --stat', 'git'), ('pytest -q', 'logs'), ('kubectl logs pod', 'logs'), ('rg -n needle src', 'search')]:
            self.assertEqual(hook.classify(self.event(cmd)), category)
        self.assertIsNone(hook.classify(self.event('python3 -c "print(\'pytest\')"')))

    def test_quoted_credentials_in_arguments_are_never_sent(self):
        self.assertIsNone(hook.classify(self.event('rg -l \'password="supersecret123"\' src')))
        self.assertEqual(hook.classify(self.event('rg -l \'password\' src')), 'search')

    def test_reads_through_command_runners_and_mixed_operands_stay_exact(self):
        for cmd in ['rg -l handler src | xargs cat', 'find src -type f -exec cat {} \\;',
                    "rg -n x src; sh -c 'head -n 50 src/a.py'", 'cat src/app.py build.log']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd, category in [('cat build.log test.log', 'logs'), ('rg -l handler src | wc -l', 'search')]:
            self.assertEqual(hook.classify(self.event(cmd)), category, cmd)

    def test_only_known_commands_and_their_filters_are_compressed(self):
        for cmd in ['rg -l handler src | xargs wc -l', 'echo start; rg -n x src',
                    "python3 -c 'print(open(\"src/a.py\").read())'; rg -n x src",
                    'fd -e py -x cat', 'rg -n x src | sort -o out.txt src/a.py', 'git grep -n handler']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd, category in [('cd src && rg -n x .', 'search'), ('pytest -q 2>&1 | tail -n 50', 'logs'),
                              ('rg -n x src | sort | uniq -c | tr a-z A-Z', 'search')]:
            self.assertEqual(hook.classify(self.event(cmd)), category, cmd)

    def test_named_file_reads_after_a_pipe_stay_exact(self):
        for cmd in ['rg handler src | head -n 100 src/file.py', 'rg handler src | cat src/a.py',
                    'rg handler src | sed -n 1,50p src/a.py', 'rg handler src | cat -- -notes.py',
                    'rg handler src | uniq -c src/a.py']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd in ['rg handler src | head -n 100', 'rg handler src | cat -']:
            self.assertEqual(hook.classify(self.event(cmd)), 'search', cmd)

    def test_redirected_input_after_a_pipe_stays_exact(self):
        for cmd in ["rg handler src | tr -d '\\r' < src/app.py", 'rg x src | sort < src/a.py',
                    'rg x src | head -n 5 <<< "$(true)"', 'rg x src | head -n 5 <<< "-source"',
                    'rg x src | cat <<EOF', 'rg x src | cat <<EOF\nsource\nEOF',
                    'rg x src | tr a-z A-Z <> src/app.py', 'rg x src | tr a-z A-Z 0<src/app.py',
                    'rg x src | tr a-z A-Z <&3', 'rg x src | head -n 5 0<&3',
                    'rg x src | tr a-z A-Z <<EOF', 'rg x src | cat < app.log',
                    'cat < src/app.py', 'head -n 5 < src/app.py', 'cat app.log < src/app.py']:
            with self.subTest(command=cmd):
                self.assertIsNone(hook.classify(self.event(cmd)))

    def test_plain_filters_and_output_redirections_keep_their_categories(self):
        for cmd, category in [('rg x src | tr a-z A-Z', 'search'),
                              ('pytest -q 2>&1 | tail -n 50', 'logs'),
                              ('rg x src 2>/dev/null | head -50', 'search'),
                              ('rg x src > matches.txt', 'search'), ('rg x src >> matches.txt', 'search'),
                              ('rg x src &> matches.txt', 'search'), ('cat < app.log', 'logs'),
                              ('rg x < src/app.py | head -50', 'search')]:
            with self.subTest(command=cmd):
                self.assertEqual(hook.classify(self.event(cmd)), category)

    def test_build_and_container_tools_count_only_for_their_real_subcommand(self):
        for cmd in ['docker exec logs cat /app/src/main.py', 'cargo run -- test', 'go run . test', 'npm run dev',
                    'kubectl exec logs -- cat src/a.py', 'git -c core.pager=cat log',
                    'git diff --no-index /dev/null src/a.py', 'rg x src | sort --files0-from=list']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd, category in [('npm run test:unit', 'logs'), ('pnpm -C web build', 'logs'), ('make -C src test', 'logs'),
                              ('docker compose -f dev.yml logs api', 'logs'), ('kubectl -n prod logs pod/a', 'logs'),
                              ('git --no-pager log -5', 'git'), ('rg x src | sort -rn | head --lines=20', 'search')]:
            self.assertEqual(hook.classify(self.event(cmd)), category, cmd)

    def test_fd_running_a_command_on_matches_stays_exact(self):
        for cmd in ['fd -xcat', 'fd -e py -Xcat', 'fd -HIxcat', 'fd --exec=cat', 'fd -e py -x cat']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        self.assertEqual(hook.classify(self.event('fd -e py handler src')), 'search')

    def test_command_substitutions_keep_the_output_exact(self):
        for cmd in ['git status `cat src/main.py >&2`', 'git status "$(cat src/main.py >&2)"', 'rg "`cat q.txt`" src',
                    'git status "$\\\n(cat src/main.py >&2)"']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd in ["rg '`code`' docs", 'rg "\\`x" src', "rg '$(x)' src"]:
            self.assertEqual(hook.classify(self.event(cmd)), 'search', cmd)

    def test_a_comment_keeps_the_next_line_a_separate_command(self):
        for cmd in ['rg handler src # find callers\ncat src/config.py', 'grep foo src # "\ncat source.py\n# "',
                    "rg x src # '\ncat source.py\n# '", 'rg x src # \\\ncat source.py',
                    'rg x src;# "\ncat source.py\n# "', '# "\ncat source.py\nrg x src',
                    'rg x src \\\n# "\ncat source.py\n# "']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd in ['rg handler src # find callers', "rg 'C#' src", 'rg C# src', 'rg a#b src',
                    "rg x src # it's fine", 'rg x src # "', 'rg x src # $(cat source.py) `cat source.py`',
                    'rg x src # \\', 'rg "#" src', 'rg \\# src', 'rg x src;# "\nrg y src',
                    'rg x src\\\n# "\ncat source.py\n# "']:
            self.assertEqual(hook.classify(self.event(cmd)), 'search', cmd)
        e = self.event('rg handler src # find callers\nrm -rf target');e['hook_event_name'] = 'PreToolUse'
        self.assertIsNone(hook.pre_tool(e))

    def test_quoted_or_escaped_word_boundaries_do_not_start_comments(self):
        for prefix in ['"x "', 'x\\ ', 'x\\|', '\\', '""']:
            cmd = 'rg ' + prefix + '# "\ncat source.py\n# "'
            self.assertEqual(hook.classify(self.event(cmd)), 'search', cmd)

    def test_unfinished_shell_quotes_and_escapes_stay_exact(self):
        for cmd in ["rg 'x # comment", 'rg "x # comment', 'rg x src \\']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)

    def test_credentials_inside_json_output_are_never_sent(self):
        def never(**kwargs):
            self.fail('engine must not receive output that holds a credential')
        padding = [{'path': f'src/m{i}.py', 'line': i, 'text': f'def handler_{i}(): return {i}'} for i in range(400)]
        for hidden in ['password = "supersecret123"', 'token in config/.env"', '"client_secret": "abcdefgh12345"']:
            snippet = json.dumps({'results': padding + [{'path': 'src/settings.py', 'text': hidden}]})
            for text in [snippet, json.dumps({'body': snippet})]:
                e = self.event();e['tool_name'] = 'mcp__codebase_memory_mcp__search_graph'
                e['tool_response'] = {'content': [{'type': 'text', 'text': text}]}
                self.assertIsNone(hook.process(e, self.cfg, never), hidden)
        events = [json.loads(line) for line in (self.state / 'events.jsonl').read_text().splitlines()]
        self.assertEqual({event['status'] for event in events}, {'skipped_sensitive'})
        self.assertEqual(len(events), 6)

    def test_git_blob_reads_stay_exact(self):
        for cmd in ['git show HEAD:src/main.py', 'git show main:README.md | head -50']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd in ['git show HEAD --stat', 'git show --format=%H:%s HEAD']:
            self.assertEqual(hook.classify(self.event(cmd)), 'git', cmd)

    def test_file_reads_next_to_a_search_stay_exact(self):
        for cmd in ['git status --short; cat src/main.py', 'rg -n x src && head -n 80 src/a.js',
                    'cd src && sed -n 1,200p main.py; git diff', 'rg -n handler src\ncat src/app.py']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd, category in [('rg -n x src | head -50', 'search'), ('git log --oneline | tail -n 20', 'git'),
                              ('tail -n 400 app.log', 'logs'), ("rg -n 'a\nb' src", 'search')]:
            self.assertEqual(hook.classify(self.event(cmd)), category, cmd)

    def test_annotated_mcp_text_blocks_stay_unchanged(self):
        e = self.event();e['tool_name'] = 'mcp__codebase_memory_mcp__search_graph'
        text = 'src/a.js:12: function handler() { return 1; }\n' * 300
        e['tool_response'] = {'content': [{'type': 'text', 'text': text}]}
        self.assertIsNotNone(hook.process(e, self.cfg, self.engine))
        for extra in [{'annotations': {'audience': ['user']}}, {'_meta': {'source': 'index'}}]:
            e['tool_response'] = {'content': [{'type': 'text', 'text': text, **extra}]}
            self.assertIsNone(hook.process(e, self.cfg, self.engine), extra)

    def test_timeout_config_accepts_five_minutes_and_rejects_more(self):
        path = self.state / 'config.json'
        with patch.object(hook, 'CONFIG_PATH', path):
            self.assertEqual(hook.config()['timeout_seconds'], 300)
            path.write_text(json.dumps({'timeout_seconds': 300}))
            self.assertEqual(hook.config()['timeout_seconds'], 300)
            path.write_text(json.dumps({'timeout_seconds': 301}))
            with self.assertRaisesRegex(ValueError, 'invalid timeout_seconds'):
                hook.config()

    def test_web_classification_does_not_claim_native_transport(self):
        event = {'tool_name': 'web.run', 'tool_input': {'search_query': [{'q': 'topic'}]}}
        self.assertEqual(hook.classify(event), 'search')
        event['hook_event_name'] = 'PreToolUse'
        self.assertIsNone(hook.pre_tool(event))
        for key in ['finance', 'weather', 'sports', 'screenshot', 'image_query', 'time', 'unknown']:
            event['tool_input'] = {'search_query': [{'q': 'topic'}], key: [{}]}
            self.assertIsNone(hook.classify(event))

    def test_transport_provenance_counts_towards_savings_and_ledger(self):
        suffix = 'Exact sources: https://example.com [wordlim 200]'
        out = hook.process(self.event(), self.cfg, self.engine, output_suffix=suffix)
        self.assertTrue(out['stopReason'].endswith(suffix))
        self.assertEqual(hook.stats()['estimated_after_tokens'], hook.tokens(out['stopReason']))
        out = hook.process(self.event(), self.cfg, self.engine, output_suffix='source ' * 3000)
        self.assertIsNone(out)
        self.assertEqual(hook.stats()['compressed_outputs'], 1)

    def test_web_savings_require_delivery_ack_without_double_counting_usage(self):
        event = self.event()
        event.update(tool_name='web.run', tool_input={'search_query': [{'q': 'topic'}]}, delivery_token='web_test_1')
        self.assertIsNotNone(hook.process(event, self.cfg, self.engine))
        self.assertEqual(hook.stats()['compressed_outputs'], 0)
        self.assertEqual(hook.stats()['helper_usage']['total_tokens'], 2100)
        # Ledgers written before 0.3.0 hold web_delivered rows.
        for _ in range(2):
            hook.append_event({'status': 'web_delivered', 'delivery_token': 'web_test_1', 'saved_tokens': 0})
        self.assertEqual(hook.stats()['compressed_outputs'], 1)
        self.assertEqual(hook.stats()['helper_usage']['total_tokens'], 2100)

    def test_any_event_with_a_delivery_token_counts_only_once_delivered(self):
        event = dict(self.event(), delivery_token='cs-toolu_01')
        replacement = hook.process(event, self.cfg, self.engine)['stopReason']
        row = ledger(self.state)[-1]
        self.assertEqual((row['status'], row['delivery_token'], row['tool']), ('compression_candidate', 'cs-toolu_01', 'Bash'))
        self.assertEqual(hook.stats()['compressed_outputs'], 0)
        result = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'delivered', '--delivery-token', 'cs-toolu_01'],
                                capture_output=True, text=True, env=self.env(CONTEXT_SAVER_STATE=str(self.state)))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'delivered': True})
        self.assertEqual({k: ledger(self.state)[-1][k] for k in ('status', 'delivery_token', 'saved_tokens')},
                         {'status': 'delivered', 'delivery_token': 'cs-toolu_01', 'saved_tokens': 0})
        report = hook.stats()
        self.assertEqual(report['compressed_outputs'], 1)
        self.assertEqual(report['estimated_after_tokens'], hook.tokens(replacement))
        self.assertEqual(report['statuses']['delivered'], 1)
        # No token (Codex run path, MCP proxy): counted at once.
        hook.process(self.event('rg -n other src'), self.cfg, self.engine)
        self.assertEqual(ledger(self.state)[-1]['status'], 'compressed')
        self.assertEqual(hook.stats()['compressed_outputs'], 2)

    def test_web_delivered_alias_writes_the_delivered_row_and_bad_tokens_fail(self):
        for action in ('web-delivered', 'delivered'):
            for token in ('', 'bad token', 'x' * 101):
                with self.subTest(action=action, token=token):
                    result = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), action, '--delivery-token', token],
                                            capture_output=True, text=True, env=self.env(CONTEXT_SAVER_STATE=str(self.state)))
                    self.assertEqual(result.returncode, 2)
        result = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'web-delivered', '--delivery-token', 'web_1'],
                                capture_output=True, text=True, env=self.env(CONTEXT_SAVER_STATE=str(self.state)))
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = ledger(self.state)
        self.assertEqual(len(rows), 1)
        self.assertEqual({k: v for k, v in rows[0].items() if k != 'time'},
                         {'status': 'delivered', 'delivery_token': 'web_1', 'saved_tokens': 0})

    def test_pre_tool_rewrite_is_a_plain_launcher_command(self):
        e = self.event("rg -n 'a b' src");e['hook_event_name'] = 'PreToolUse'
        e.update(session_id='s 1', tool_use_id="call'1")
        out = hook.pre_tool(e)['hookSpecificOutput']
        self.assertEqual(out['permissionDecision'], 'allow')
        launcher = str(hook.ROOT.parent / 'bin' / 'context-saver')
        wrapped = out['updatedInput']['command']
        # Exactly the launcher and quoted words: no VAR= prefix, $ expansion or substitution.
        self.assertEqual(wrapped, ' '.join(shlex.quote(word) for word in
                                           [launcher, 'run', '--command', "rg -n 'a b' src",
                                            '--session-id', 's 1', '--tool-use-id', "call'1"]))
        self.assertEqual(shlex.split(wrapped), [launcher, 'run', '--command', "rg -n 'a b' src",
                                                '--session-id', 's 1', '--tool-use-id', "call'1"])
        self.assertTrue(wrapped.startswith(shlex.quote(launcher) + ' run '))

    def test_pre_tool_preserves_original_command_and_skips_compounds(self):
        e = self.event("rg -n 'a b' src");e['hook_event_name'] = 'PreToolUse'
        out = hook.pre_tool(e)['hookSpecificOutput']
        self.assertEqual(out['permissionDecision'], 'allow')
        self.assertIn('--command', out['updatedInput']['command'])
        self.assertIn(str(hook.ROOT.parent / 'bin' / 'context-saver') + ' run', out['updatedInput']['command'])
        for cmd in ['rg hello src; rm -rf target', 'rg hello $(pwd)', 'rg hello src > result.txt', 'git diff; git push', 'git push origin log', 'rg hello src; kubectl delete pod x', 'kubectl logs -f pod', 'npm test --watch', 'rg hello src\nrm -rf target']:
            e['tool_input']['command'] = cmd
            self.assertIsNone(hook.pre_tool(e))

    def test_pre_tool_leaves_following_logs_unwrapped(self):
        commands = ['tail -F app.log', 'journalctl -fu example.service', 'kubectl logs -f pod',
                    'docker logs --follow=true api', 'tail -nF 50 app.log',
                    "docker logs --fo''llow=true api", "journalctl '-fu\nexample.service'"]
        for command in ['tail app.log', 'journalctl', 'docker logs api', 'podman logs api', 'kubectl logs pod']:
            commands.extend(command + ' ' + option for option in ['-f', '-F', '--follow', '--follow=name',
                                                                  '-fu', '-nF', '-fn100', '-nf100'])
        for cmd in commands:
            with self.subTest(command=cmd):
                e = self.event(cmd);e['hook_event_name'] = 'PreToolUse'
                self.assertIsNone(hook.pre_tool(e))
        for cmd in ['tail -n 50 app.log', 'journalctl -u example.service', 'docker logs --tail=50 api',
                    'podman logs api', 'kubectl logs pod', 'rg -f patterns src']:
            e = self.event(cmd);e['hook_event_name'] = 'PreToolUse'
            self.assertIn(' run --command ', hook.pre_tool(e)['hookSpecificOutput']['updatedInput']['command'], cmd)

    def test_pre_tool_leaves_watch_modes_unwrapped(self):
        for cmd in ['tsc -w', 'tsc --watch', 'jest --watchAll', 'vitest -w',
                    'npm test -- --watch', 'pytest -f', 'pytest --looponfail', 'py.test -f',
                    'pnpm test -- --watch=true', "jest --wat''chAll", "tsc '-w'",
                    'tsc --noEmit && pytest -f', 'tsc --noEmit | jest --watchAll', 'rg --watch src']:
            with self.subTest(command=cmd):
                e = self.event(cmd);e['hook_event_name'] = 'PreToolUse'
                self.assertIsNone(hook.pre_tool(e))

    def test_pre_tool_wraps_commands_with_finite_options(self):
        for cmd in ['tsc --noEmit', 'jest -w 4', 'npm test -w pkg', 'pnpm test -w', 'pytest -q',
                    "rg 'contains --watch text' src", 'tsc --noEmit # --watch']:
            with self.subTest(command=cmd):
                e = self.event(cmd);e['hook_event_name'] = 'PreToolUse'
                self.assertIn(' run --command ', hook.pre_tool(e)['hookSpecificOutput']['updatedInput']['command'])

    def test_recursion_guard(self):
        with patch.dict(os.environ, {'CONTEXT_SAVER_ACTIVE': '1'}):
            self.assertIsNone(hook.process(self.event(), self.cfg, self.engine))

    def test_run_preserves_small_output_and_failure(self):
        r = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'run', '--command', 'printf exact-output; exit 7'],
                           capture_output=True, text=True, cwd=self.state, env=self.env())
        self.assertEqual(r.returncode, 7)
        self.assertEqual(r.stdout, 'exact-output')

    def test_run_uses_the_users_shell_else_bin_sh(self):
        shells = [(shell, shell) for shell in ('/bin/zsh', '/bin/bash') if Path(shell).is_file()]
        shells += [('/nonexistent/shell', '/bin/sh'), (None, '/bin/sh')]
        for shell, expected in shells:
            with self.subTest(shell=shell):
                env = self.env(**({'SHELL': shell} if shell else {}))
                if shell is None:
                    env.pop('SHELL', None)
                r = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'run', '--command', 'printf %s "$0"'],
                                   capture_output=True, text=True, cwd=self.state, timeout=30, env=env)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(r.stdout, expected)

    def test_run_replays_output_when_the_wrapper_is_stopped(self):
        # The command stops its own wrapper ($PPID), as a tool timeout does.
        r = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'run', '--command',
                            'printf partial-output; kill -TERM $PPID; sleep 5'],
                           capture_output=True, text=True, cwd=self.state, timeout=30,
                           env=self.env(SHELL='/bin/sh'))
        self.assertEqual(r.returncode, 143)
        self.assertEqual(r.stdout, 'partial-output')

    def test_run_stops_descendants_when_the_wrapper_is_stopped(self):
        self.assert_run_stops_descendants()

    def test_run_stops_descendants_without_a_writable_temp_folder(self):
        self.assert_run_stops_descendants(direct=True)

    def test_run_kills_descendants_that_ignore_sigterm(self):
        for direct in (False, True):
            with self.subTest(direct=direct):
                self.assert_run_stops_descendants(direct=direct, ignore_term=True)

    def test_run_stops_descendants_on_other_stop_signals(self):
        for signum in (signal.SIGINT, signal.SIGHUP):
            with self.subTest(signum=signum):
                self.assert_run_stops_descendants(signum=signum)

    def test_run_stops_descendants_on_keyboard_interrupt(self):
        for direct in (False, True):
            with self.subTest(direct=direct):
                self.assert_run_stops_descendants(direct=direct, signum=signal.SIGINT, keyboard_interrupt=True)

    def assert_run_stops_descendants(self, direct=False, ignore_term=False, signum=signal.SIGTERM,
                                    keyboard_interrupt=False):
        pid_file = self.state / 'sleep.pid'
        pid_file.unlink(missing_ok=True)
        command = ('sleep 30 >/dev/null 2>&1 & ' + f'echo $! > {shlex.quote(str(pid_file))}; ' +
                   f'printf partial-output; kill -{signum.name[3:]} $PPID; wait')
        if ignore_term:
            command = "trap '' TERM; " + command.replace('printf partial-output', 'trap - TERM; printf partial-output')
        argv = [sys.executable, str(hook.ROOT / 'hook.py')]
        if direct or keyboard_interrupt:
            code = "import hook, signal; from unittest.mock import patch; "
            if direct:
                code += "patch.object(hook.tempfile, 'TemporaryFile', side_effect=OSError('read-only')).start(); "
            if keyboard_interrupt:
                code += "patch.object(hook, 'stop_wrapper', signal.default_int_handler).start()\n" + \
                        "try:\n    hook.main()\nexcept KeyboardInterrupt:\n    raise SystemExit(130)"
            else:
                code += 'hook.main()'
            argv = [sys.executable, '-c', code]
        try:
            result = subprocess.run([*argv, 'run', '--command', command],
                                    capture_output=True, text=True, cwd=hook.ROOT, timeout=10,
                                    env=self.env(SHELL='/bin/sh'))
            self.assertEqual(result.returncode, 128 + signum, result.stderr)
            self.assertEqual(result.stdout, 'partial-output')
            self.assertTrue(is_gone(wait_for_pid(pid_file)), 'The background sleep survived cancellation')
        finally:
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_wrapped_command_runs_the_original_and_keeps_its_status(self):
        e = self.event('grep -n handler missing.txt | cat');e['hook_event_name'] = 'PreToolUse'
        wrapped = hook.pre_tool(e)['hookSpecificOutput']['updatedInput']['command']
        argv = shlex.split(wrapped)
        self.assertEqual(argv[:3], [str(hook.ROOT.parent / 'bin' / 'context-saver'), 'run', '--command'])
        # The engine behind the launcher, with the same arguments.
        r = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), *argv[1:]], cwd=self.state,
                           env=self.env(SHELL='/bin/sh'), capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('missing.txt', r.stdout)

    def test_stopping_process_ends_the_running_summary_helper(self):
        env, pid_file = fake_claude(self.state)
        (self.state / 'config.json').write_text(json.dumps({'minimum_tokens': 1000, 'deterministic_first': False}))
        with subprocess.Popen([sys.executable, str(hook.ROOT / 'hook.py'), 'process'], env=env, cwd=self.state,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            process.stdin.write(json.dumps(self.event()).encode())
            process.stdin.close()
            helper = wait_for_pid(pid_file)
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=10)
        self.assertTrue(is_gone(helper))

    def test_run_executes_the_command_without_a_writable_temp_folder(self):
        code = "import hook; from unittest.mock import patch; " + \
               "p=patch.object(hook.tempfile, 'TemporaryFile', side_effect=OSError('read-only')); p.start(); " + \
               "raise SystemExit(hook.run_command('printf direct-output; exit 3'))"
        result = subprocess.run([sys.executable, '-c', code], cwd=hook.ROOT, capture_output=True, text=True, timeout=30,
                                env=self.env())
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stdout, 'direct-output')

    def test_run_replays_output_when_stopped_while_compressing(self):
        code = "import hook, os, signal; from unittest.mock import patch; " + \
               "p=patch.object(hook, 'process', side_effect=lambda *a, **k: os.kill(os.getpid(), signal.SIGTERM)); p.start(); " + \
               "raise SystemExit(hook.run_command('printf finished-output'))"
        result = subprocess.run([sys.executable, '-c', code], cwd=hook.ROOT, capture_output=True, text=True, timeout=30,
                                env=self.env())
        self.assertEqual(result.returncode, 143)
        self.assertEqual(result.stdout, 'finished-output')

    def test_weekly_reminder_is_throttled(self):
        first = hook.session_start()
        context = first['hookSpecificOutput']['additionalContext']
        self.assertIn('weekly check due', context)
        # Codex has no /context-saver command: the CLI form is given for both hosts.
        self.assertIn('`context-saver stats --days 7`', context)
        self.assertIn('`context-saver review --note=<finding>`', context)
        self.assertIn('Claude Code: /context-saver stats 7', context)
        self.assertIn('Claude Code: /context-saver review <finding>', context)
        self.assertIsNone(hook.session_start())

    def aged(self, path, days, text='content'):
        common.private_dir(path.parent)
        path.write_text(text)
        stamp = time.time() - days * 86400
        os.utime(path, (stamp, stamp))
        return path

    def test_weekly_check_prunes_old_debug_responses_with_raw_archive(self):
        debug = self.state / 'engine' / 'debug'
        old = self.aged(debug / 'invalid-response-old.txt', 8)
        recent = self.aged(debug / 'invalid-response-new.txt', 0)
        hook.session_start()
        self.assertFalse(old.exists())
        self.assertTrue(recent.exists())

    def test_retention_runs_when_the_reminder_is_not_due(self):
        common.write_json(self.state / 'review.json', {'last_reminder_at': time.time()})
        old = self.aged(self.state / 'raw' / 'old.txt', 8)
        self.assertIsNone(hook.session_start())
        self.assertFalse(old.exists())

    def test_session_start_sweeps_web_inputs_and_summary_cache(self):
        old_input, new_input = self.aged(self.state / 'web-input.AbC123', 2), self.aged(self.state / 'web-input.XyZ789', 0)
        kept = self.aged(self.state / 'raw' / 'recent.txt', 6)
        self.assertIsNotNone(hook.session_start(dict(hook.DEFAULTS)))
        self.assertFalse(old_input.exists())
        self.assertTrue(new_input.exists())
        self.assertTrue(kept.exists())

    @unittest.skipUnless(hasattr(summary_engine, 'prune_cache'), 'summary_engine.prune_cache (stream C) is required')
    def test_session_start_prunes_the_summary_cache_with_the_retention_days(self):
        cache = self.state / 'engine' / 'cache'
        old, recent = self.aged(cache / ('a' * 64 + '.json'), 3), self.aged(cache / ('b' * 64 + '.json'), 1)
        hook.session_start(dict(hook.DEFAULTS, retention_days=2))
        self.assertFalse(old.exists())
        self.assertTrue(recent.exists())

    def test_corrupt_review_file_does_not_stop_cleanup_or_the_reminder(self):
        for content in ('{broken', '[]', '{"last_reminder_at": "yesterday"}'):
            with self.subTest(content=content):
                (self.state / 'review.json').write_text(content)
                old = self.aged(self.state / 'raw' / 'old.txt', 8)
                output = hook.session_start()
                self.assertFalse(old.exists())
                self.assertIn('weekly check due', output['hookSpecificOutput']['additionalContext'])
                self.assertIsInstance(json.loads((self.state / 'review.json').read_text())['last_reminder_at'], float)

    def test_session_start_rewrites_a_large_ledger_without_year_old_rows(self):
        path = self.state / 'events.jsonl'
        now = time.time()
        old = json.dumps({'time': now - 400 * 86400, 'status': 'compressed', 'pad': 'x' * 2000}) + '\n'
        recent = json.dumps({'time': now - 10 * 86400, 'status': 'skipped_busy'}) + '\n'
        common.private_dir(self.state)
        with path.open('w') as handle:
            handle.write(old * 2700 + 'not json\n' + recent)
        path.chmod(0o600)
        self.assertGreater(path.stat().st_size, hook.LEDGER_LIMIT_BYTES)
        hook.session_start()
        self.assertEqual(path.read_text(), recent)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        # A ledger under the limit is left alone, old rows and all.
        with path.open('a') as handle:
            handle.write(old)
        before = path.read_text()
        common.write_json(self.state / 'review.json', {'last_reminder_at': time.time()})
        hook.session_start()
        self.assertEqual(path.read_text(), before)

    def test_ledger_rewrite_waits_for_appends_under_the_shared_lock(self):
        hook.append_event({'status': 'first'})
        with hook.ledger_lock(hook.fcntl.LOCK_SH):
            # An exclusive lock cannot be taken while an append holds the shared one.
            fd = os.open(self.state / 'events.lock', os.O_RDWR)
            try:
                with self.assertRaises(BlockingIOError):
                    hook.fcntl.flock(fd, hook.fcntl.LOCK_EX | hook.fcntl.LOCK_NB)
            finally:
                os.close(fd)
        self.assertEqual([row['status'] for row in ledger(self.state)], ['first'])

    def test_invalid_config_values_are_rejected_by_name(self):
        path = self.state / 'config.json'
        self.assertIs(hook.config()['debug_invalid_responses'], False)
        for key, value in [('debug_invalid_responses', 'yes'), ('enabled', 1), ('minimum_tokens', 999),
                           ('minimum_tokens', 'many'), ('minimum_saving_fraction', 1), ('timeout_seconds', None),
                           ('effort', 'turbo'), ('effort', None), ('minimum_tokens', True), ('minimum_tokens', 5000.5),
                           ('timeout_seconds', 'NaN'), ('minimum_saving_fraction', True),
                           ('minimum_saving_fraction', '0.05'), ('retention_days', 0), ('retention_days', '7 days'),
                           ('retention_days', 366), ('review_every_days', 0), ('review_every_days', False),
                           ('categories', 'search'), ('categories', ['search', 'source_code']), ('categories', [1]),
                           ('rtk_binary', None), ('rtk_binary', 7)]:
            with self.subTest(key=key, value=value):
                path.write_text(json.dumps({key: value}))
                with self.assertRaisesRegex(ValueError, f'invalid {key}'):
                    hook.config()

    def test_every_default_is_validated(self):
        self.assertLessEqual(set(hook.DEFAULTS), set(hook.VALID))
        with patch.dict(os.environ):
            for name in (hook.EFFORT_ENV, hook.SHELL_ENV):
                os.environ.pop(name, None)
            self.assertEqual(hook.config(), hook.DEFAULTS)

    def test_config_stores_converted_numbers(self):
        path = self.state / 'config.json'
        path.write_text(json.dumps({'minimum_tokens': '3500', 'maximum_source_chars': 60000.0,
                                    'timeout_seconds': ' 60 ', 'minimum_saving_fraction': '0.3',
                                    'maximum_replacement_tokens': 1500, 'retention_days': '14',
                                    'review_every_days': 30.0, 'categories': [], 'rtk_binary': '/opt/rtk'}))
        cfg = hook.config()
        expected = {'minimum_tokens': 3500, 'maximum_source_chars': 60000, 'timeout_seconds': 60,
                    'minimum_saving_fraction': 0.3, 'maximum_replacement_tokens': 1500, 'retention_days': 14,
                    'review_every_days': 30, 'categories': [], 'rtk_binary': '/opt/rtk'}
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(cfg[key], value)
                self.assertIs(type(cfg[key]), type(value))
        # Converted values work where the raw strings failed (timeout reaches the engine as an int).
        seen = {}
        hook.process(self.event(), dict(cfg, minimum_tokens=1000, categories=list(common.CATEGORIES)),
                     lambda **kw: seen.update(kw) or self.engine())
        self.assertEqual(seen['timeout_seconds'], 60)
        # With no category enabled nothing is compressed.
        self.assertIsNone(hook.process(self.event(), dict(cfg, minimum_tokens=1000), self.engine))

    def test_unreadable_config_files_are_named_errors(self):
        path = self.state / 'config.json'
        for content, message in (('{broken', 'not valid JSON'), ('[1, 2]', 'must hold a JSON object')):
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaisesRegex(ValueError, message):
                    hook.config()

    def test_config_errors_reach_the_host_from_limits_and_session_start(self):
        path = self.state / 'config.json'
        path.write_text(json.dumps({'timeout_seconds': 5000}))
        # -O: the checks are not asserts, so they still run under optimization.
        for action, extra in (('limits', {'minimum_tokens': None}), ('session-start', {})):
            with self.subTest(action=action):
                result = subprocess.run([sys.executable, '-O', str(hook.ROOT / 'hook.py'), action], capture_output=True,
                                        text=True, env=self.env())
                self.assertEqual(result.returncode, 0, result.stderr)
                output = json.loads(result.stdout)
                self.assertIn('invalid timeout_seconds', output.pop('configError'))
                self.assertEqual(output, extra)

    def test_stats_days_must_be_positive_and_all_time_is_null(self):
        hook.append_event({'status': 'compressed', 'before_tokens': 10, 'after_tokens': 4, 'saved_tokens': 6})
        for days in ('0', '-3', 'week'):
            with self.subTest(days=days):
                result = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'stats', '--days', days],
                                        capture_output=True, text=True, env=self.env(CONTEXT_SAVER_STATE=str(self.state)))
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, '')
        for argv, period in ((['stats'], None), (['stats', '--days', '1'], 1)):
            result = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), *argv],
                                    capture_output=True, text=True, env=self.env(CONTEXT_SAVER_STATE=str(self.state)))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertIs(report['period_days'], period)
            self.assertEqual(report['estimated_context_tokens_saved'], 6)
            self.assertNotIn('helper_tokens', report['rtk'])

    def test_configured_effort_reaches_engine_and_claude_cost_is_reported(self):
        seen = {}
        def engine(**kwargs):
            seen.update(kwargs)
            return dict(self.engine(), provider='claude_cli', model='sonnet', list_cost_usd=0.0684)
        hook.process(self.event(), dict(self.cfg, effort='low'), engine)
        self.assertEqual(seen['effort'], 'low')
        hook.process(self.event('rg -n other src'), self.cfg, lambda **kw: seen.update(default=kw['effort']) or self.engine())
        self.assertEqual(seen['default'], 'medium')
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[0])
        self.assertEqual(row['list_cost_usd'], 0.0684)
        self.assertEqual((row['provider'], row['model']), ('claude_cli', 'sonnet'))
        self.assertEqual(hook.stats()['claude_list_price_usd'], 0.0684)

    def test_helper_cache_counts_and_net_savings(self):
        usage = {'input_tokens': 24000, 'output_tokens': 400, 'cache_read_tokens': 18000,
                 'cache_write_tokens': 1000, 'total_tokens': 24400}
        result = dict(self.engine(), provider='claude_cli', model='sonnet', usage=usage)
        hook.process(self.event(), self.cfg, lambda **kwargs: result)
        report = hook.stats()
        self.assertEqual(report['helper_usage']['cache_write_tokens'], 1000)
        self.assertEqual(report['helper_usage']['cache_read_tokens'], 18000)
        self.assertEqual(report['estimated_net_tokens_saved_including_helper'],
                         report['estimated_context_tokens_saved'] - 24400)

    def test_run_replays_output_if_config_fails_after_command(self):
        code = "import hook; from unittest.mock import patch; " + \
               "p=patch.object(hook, 'config', side_effect=ValueError('bad config')); p.start(); " + \
               "raise SystemExit(hook.run_command('printf still-exact; exit 7'))"
        result = subprocess.run([sys.executable, '-c', code], cwd=hook.ROOT, capture_output=True, text=True,
                                env=self.env())
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, 'still-exact')



class PluginConfigTests(unittest.TestCase):
    """The Claude Code plugin passes its /config options as environment variables."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp.name) / 'config.json'
        self.config_patch = patch.object(hook, 'CONFIG_PATH', self.config_path)
        self.config_patch.start()

    def tearDown(self):
        self.config_patch.stop()
        self.tmp.cleanup()

    def test_without_plugin_options_config_json_decides(self):
        self.config_path.write_text(json.dumps({'shell_helpers_enabled': False, 'effort': 'low'}))
        with patch.dict(os.environ, {}, clear=False):
            for name in ('CONTEXT_SAVER_EFFORT', 'CONTEXT_SAVER_SUMMARIZE_SHELL'):
                os.environ.pop(name, None)
            cfg = hook.config()
        self.assertFalse(cfg['shell_helpers_enabled'])
        self.assertEqual(cfg['effort'], 'low')
        self.config_path.unlink()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('CONTEXT_SAVER_EFFORT', None)
            self.assertEqual(hook.config()['effort'], 'medium')

    def test_plugin_effort_overrides_config_json(self):
        self.config_path.write_text(json.dumps({'effort': 'low'}))
        for effort in ('high', 'none'):
            with self.subTest(effort=effort), patch.dict(os.environ, {'CONTEXT_SAVER_EFFORT': effort}):
                self.assertEqual(hook.config()['effort'], effort)
        with patch.dict(os.environ, {'CONTEXT_SAVER_EFFORT': 'turbo'}), \
                self.assertRaisesRegex(ValueError, 'invalid effort'):
            hook.config()

    def test_summarize_shell_option_overrides_config_json(self):
        self.config_path.write_text(json.dumps({'shell_helpers_enabled': True}))
        with patch.dict(os.environ, {'CONTEXT_SAVER_SUMMARIZE_SHELL': '0'}):
            self.assertFalse(hook.config()['shell_helpers_enabled'])
        self.config_path.write_text(json.dumps({'shell_helpers_enabled': False}))
        with patch.dict(os.environ, {'CONTEXT_SAVER_SUMMARIZE_SHELL': '1'}):
            self.assertTrue(hook.config()['shell_helpers_enabled'])

    def test_web_bridge_prints_this_machines_paths(self):
        state = Path(self.tmp.name) / 'state'
        result = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'web-bridge'], capture_output=True, text=True,
                                env=isolated_env(self.tmp.name, CONTEXT_SAVER_STATE=str(state)))
        self.assertEqual(result.returncode, 0)
        self.assertNotIn('__CONTEXT_SAVER_', result.stdout)
        self.assertIn(json.dumps(str(hook.ROOT.parent / 'bin' / 'context-saver')), result.stdout)
        self.assertIn(json.dumps(str(state)), result.stdout)
        self.assertTrue(state.is_dir())

    def test_disable_keeps_the_advanced_settings_in_config_json(self):
        stored = {'effort': 'low', 'shell_helpers_enabled': False}
        self.config_path.write_text(json.dumps(stored))
        result = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'disable'], capture_output=True, text=True,
                                env=isolated_env(self.tmp.name, CONTEXT_SAVER_CONFIG=str(self.config_path),
                                                 CONTEXT_SAVER_EFFORT='high', CONTEXT_SAVER_SUMMARIZE_SHELL='1'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.config_path.read_text()), {**stored, 'enabled': False})

    def test_limits_report_the_configured_minimum(self):
        self.config_path.write_text(json.dumps({'minimum_tokens': 2000}))
        result = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'limits'], capture_output=True, text=True,
                                env=isolated_env(self.tmp.name, CONTEXT_SAVER_CONFIG=str(self.config_path)))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'minimum_tokens': 2000})

    def test_launcher_runs_the_web_adapter(self):
        envelope = {'tool_name': 'web.run', 'tool_input': {'search_query': [{'q': 'topic'}]}, 'tool_response': 'short'}
        result = subprocess.run(['sh', str(hook.ROOT.parent / 'bin' / 'context-saver'), 'web-adapter', '--decision-only'],
                                input=json.dumps(envelope), capture_output=True, text=True,
                                env=isolated_env(self.tmp.name, CONTEXT_SAVER_PYTHON=sys.executable,
                                                 CONTEXT_SAVER_DATA=self.tmp.name, CONTEXT_SAVER_STATE=self.tmp.name))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'compressed': False})


if __name__ == '__main__':
    unittest.main()
