"""Shell policy keeps exact evidence and never spends helper tokens."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import hook
from test_deterministic import jest_text


class ShellLocalOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        self.patch = patch.object(hook, 'STATE', self.state)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.cfg = dict(hook.DEFAULTS, minimum_tokens=1000, shell_helpers_enabled=False,
                        rtk_enabled=True)
        self.engine = Mock(side_effect=AssertionError('Shell helper must not run'))

    def event(self, command, text, tool='Bash', exit_code=0):
        return {'hook_event_name': 'PostToolUse', 'tool_name': tool,
                'tool_input': {'command': command},
                'tool_response': {'output': text, 'exit_code': exit_code}}

    def test_shell_variants_and_oversized_outputs_stay_raw(self):
        for tool in ['Bash', 'exec_command', 'shell', 'shell_command']:
            for command, text in [
                ('rg -n handler src', 'src/app.js:12: handler()\n' * 600),
                ('git diff', 'diff --git a/app.py b/app.py\n-old\n+new\n' * 600),
                ('git show', 'diff --git a/app.py b/app.py\n-old\n+new\n' * 600),
                ('npm run build', 'unrecognized build output\n' * 6000),
            ]:
                with self.subTest(tool=tool, command=command):
                    event = self.event(command, text, tool)
                    self.assertIsNone(hook.process(event, self.cfg, self.engine))
        self.engine.assert_not_called()
        self.assertEqual(hook.stats()['helper_usage']['total_tokens'], 0)

    def test_successful_jest_still_uses_local_filter(self):
        text = jest_text()
        result = hook.process(self.event('npm test', text), self.cfg, self.engine)
        self.assertIsNotNone(result)
        self.assertIn('300 passed, 300 total', result['stopReason'])
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
        self.assertTrue(row['deterministic'])
        self.assertEqual(row['helper_usage']['total_tokens'], 0)
        self.assertEqual(Path(row['raw_file']).read_text(), text)
        self.engine.assert_not_called()

    def test_failed_shell_keeps_the_entire_log(self):
        with patch('deterministic.candidates') as candidates:
            self.assertIsNone(hook.process(self.event('npm test', jest_text(True), exit_code=1),
                                           self.cfg, self.engine))
        candidates.assert_not_called()
        self.engine.assert_not_called()

    def test_discovery_helper_remains_available(self):
        self.cfg['deterministic_first'] = False
        event = {'hook_event_name': 'PostToolUse', 'tool_name': 'mcp__codebase__search_graph',
                 'tool_input': {'name_pattern': 'handler'},
                 'tool_response': 'src/app.py:12 handler function\n' * 600}
        engine = Mock(return_value={'status': 'summary', 'summary': 'src/app.py:12 handler.',
                                    'usage': {'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110}})
        self.assertIsNotNone(hook.process(event, self.cfg, engine))
        engine.assert_called_once()

    def test_disabling_deterministic_stage_does_not_enable_shell_helpers(self):
        self.cfg['deterministic_first'] = False
        self.assertIsNone(hook.process(self.event('npm test', jest_text()), self.cfg, self.engine))
        self.engine.assert_not_called()


if __name__ == '__main__':
    unittest.main()
