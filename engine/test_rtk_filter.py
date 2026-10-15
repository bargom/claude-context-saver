import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import hook
import rtk_filter


def event(text, command='vitest run', exit_code=0):
    return {'hook_event_name': 'PostToolUse', 'tool_name': 'Bash',
            'tool_input': {'command': command},
            'tool_response': {'output': text, 'exit_code': exit_code}}


def vitest():
    return (' RUN  v3.2.0 /tmp/fixture\n' +
            ''.join(f' ✓ src/file_{i}.test.ts (5 tests) 2ms\n' for i in range(400)) +
            '\n Test Files  400 passed (400)\n      Tests  2000 passed (2000)\n   Duration  1.0s\n')


class RtkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)
        self.state_patch = patch.object(hook, 'STATE', self.state)
        self.state_patch.start()
        self.cfg = dict(hook.DEFAULTS, rtk_enabled=True)
        self.engine = Mock(side_effect=AssertionError('RTK acceptance must not call a helper'))

    def tearDown(self):
        self.state_patch.stop()
        self.tmp.cleanup()

    def test_selection_excludes_exact_structured_compound_failed_and_go(self):
        cases = [
            ('git diff', 'diff --git a/x b/x\n', 0),
            ('git log -p', 'commit ' + 'a' * 40 + '\n', 0),
            ('git log --pretty=full', 'commit ' + 'a' * 40 + '\n', 0),
            ('git status --porcelain', ' M file.py\n', 0),
            ('cat file.py', vitest(), 0),
            ('vitest run | head', vitest(), 0),
            ('vitest run; git status', vitest(), 0),
            ('vitest run --reporter=json', vitest(), 0),
            ('vitest run', vitest(), 1),
            ('vitest run', vitest() + 'Warning: diagnostic\n', 0),
            ('vitest run', '{"tests":4}', 0),
            ('go test ./...', 'ok example.com/project 0.12s\n', 0),
        ]
        for command, source, code in cases:
            with self.subTest(command=command, code=code):
                self.assertIsNone(rtk_filter.select_filter(source, 'logs', event(source, command, code)))

    def test_retains_unknown_lines_for_all_test_filters(self):
        for selected, passing in [('vitest', ' ✓ x.test.ts (5 tests) 2ms'),
                                  ('pytest', 'test_x.py .... [100%]'),
                                  ('cargo-test', 'test x::works ... ok')]:
            source = passing + '\nCUSTOM runtime diagnostic\n  exact context λ\n'
            retained = rtk_filter.retained_lines(source, selected)
            self.assertNotIn(passing, retained)
            self.assertEqual(retained, 'CUSTOM runtime diagnostic\n  exact context λ')

    @unittest.skipUnless(Path('/opt/homebrew/bin/rtk').exists(), 'installed RTK required')
    def test_real_rtk_acceptance_archive_accounting_and_no_helper(self):
        source = vitest() + 'CUSTOM runtime diagnostic λ\n'
        output = hook.process(event(source), self.cfg, self.engine)['stopReason']
        self.engine.assert_not_called()
        self.assertIn('tokens saved by RTK (local)', output)
        self.assertIn('CUSTOM runtime diagnostic λ', output)
        self.assertIn('2000', output)
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
        self.assertEqual(row['model'], 'rtk-stdin-filter')
        self.assertEqual(row['helper_usage']['total_tokens'], 0)
        self.assertEqual(row['saved_tokens'], hook.tokens(source) - hook.tokens(output))
        self.assertEqual(row['after_tokens'], hook.tokens(output))
        self.assertLessEqual(row['after_tokens'], self.cfg['maximum_replacement_tokens'])
        self.assertEqual(Path(row['raw_file']).read_text(), source)
        self.assertEqual(Path(row['raw_file']).stat().st_mode & 0o777, 0o600)
        self.assertEqual(hook.stats()['rtk'], {'compressed_outputs': 1,
                         'estimated_context_tokens_saved': row['saved_tokens'], 'helper_tokens': 0})

    def test_missing_disabled_timeout_and_invalid_filter_result_decline(self):
        source = vitest()
        self.assertIsNone(rtk_filter.candidate(source, 'logs', event(source), dict(self.cfg, rtk_enabled=False)))
        self.assertIsNone(rtk_filter.candidate(source, 'logs', event(source), dict(self.cfg, rtk_binary='/missing/rtk')))
        with patch('rtk_filter.os.access', return_value=True):
            for result in [subprocess.CompletedProcess([], 1, '', 'bad'),
                           subprocess.CompletedProcess([], 0, '', ''),
                           subprocess.CompletedProcess([], 0, source, ''),
                           subprocess.CompletedProcess([], 0, 'ok', 'warning')]:
                with patch('rtk_filter.subprocess.run', return_value=result):
                    self.assertIsNone(rtk_filter.candidate(source, 'logs', event(source), self.cfg))
            with patch('rtk_filter.subprocess.run', side_effect=subprocess.TimeoutExpired('rtk', 2)):
                self.assertIsNone(rtk_filter.candidate(source, 'logs', event(source), self.cfg))

    @unittest.skipUnless(Path('/opt/homebrew/bin/rtk').exists(), 'installed RTK required')
    def test_real_pytest_cargo_and_git_filters(self):
        pytest = ''.join(f'tests/test_{i}.py .... [100%]\n' for i in range(400)) + '================ 1600 passed in 1.00s ================\n'
        cargo = ''.join(f'test suite::test_{i} ... ok\n' for i in range(400)) + 'test result: ok. 400 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n'
        git = ''.join(f'commit {i:040x}\nAuthor: Test <test@example.com>\nDate: Mon Sep 14 12:00:00 2026 +0200\n\n    Message {i}\n\n' for i in range(100))
        for command, source, count in [('pytest', pytest, '1600'), ('cargo test', cargo, '400'), ('git log', git, '0000000')]:
            with self.subTest(command=command):
                output = hook.process(event(source, command), self.cfg, self.engine)['stopReason']
                self.assertIn('RTK (local)', output)
                self.assertIn(count, output)
                self.engine.assert_not_called()
        source = git.replace('Message 1', 'Fix error handling 1')
        self.assertEqual(rtk_filter.select_filter(source, 'git', event(source, 'git log')), 'git-log')

    def test_raw_small_and_sensitive_results_do_not_call_rtk(self):
        cases = [event(vitest(), 'CONTEXT_SAVER_RAW=1 vitest run'),
                 event('small'), event(vitest() + 'password=example-secret-value')]
        with patch('rtk_filter.subprocess.run') as runner:
            for e in cases:
                self.assertIsNone(hook.process(e, self.cfg, self.engine))
            runner.assert_not_called()
            self.engine.assert_not_called()


if __name__ == '__main__':
    unittest.main()
