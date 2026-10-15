import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import deterministic
import hook
from test_oversize import graph_rows, read_index


def code_source():
    row = {'node': 'worker', 'qualified_name': 'project.worker', 'label': 'Function',
           'file': 'src/worker.py', 'start_line': 100, 'end_line': 399,
           'in_degree': 7, 'out_degree': 8, 'match_lines': [101, 397],
           'context_start': 100,
           'context': '\n'.join(f'    value_{n} = "exact λ value {n}"' for n in range(100, 400))}
    return {'results': [row], 'raw_matches': [{'file': 'config.yml', 'line': 4, 'text': 'exact raw'}],
            'directories': {'src': 1, '.': 1}, 'total_grep_matches': 3, 'total_results': 1,
            'raw_match_count': 1, 'elapsed_ms': 19, 'dedup_ratio': '1.5x'}


def event(text, tool='search_code', **arguments):
    return {'hook_event_name': 'PostToolUse', 'tool_name': 'mcp__codebase_memory_mcp__' + tool,
            'tool_input': arguments, 'tool_response': {'content': [{'type': 'text', 'text': text}]}}


def jest_text(failed=False):
    lines = ''.join(f'PASS src/suite_{i}.test.js\n  ✓ works {i} (2 ms)\n' for i in range(300))
    failure = ('FAIL src/broken.test.js\n  ● exact failure title\n'
               '    Expected: 7\n    Received: 9\n    at src/broken.test.js:13:2\n'
               'PASS quoted/assertion.test.js\n  ✓ quoted success (2 ms)\n') if failed else ''
    return (lines + failure + f'Test Suites: {"1 failed, " if failed else ""}300 passed, {301 if failed else 300} total\n'
            f'Tests: {"1 failed, " if failed else ""}300 passed, {301 if failed else 300} total\n'
            'Snapshots: 0 total\nTime: 2.431 s\nRan all test suites.\n')


class DeterministicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)
        self.state_patch = patch.object(hook, 'STATE', self.state)
        self.state_patch.start()
        self.cfg = dict(hook.DEFAULTS, minimum_tokens=1000)
        self.engine = Mock(side_effect=AssertionError('local acceptance must not call a helper'))

    def tearDown(self):
        self.state_patch.stop()
        self.tmp.cleanup()

    def ledger(self):
        return json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])

    def check_accounting(self, source, output):
        row = self.ledger()
        self.engine.assert_not_called()
        self.assertTrue(row['deterministic'])
        self.assertEqual(row['provider'], 'local_compaction')
        self.assertFalse(row['cache_hit'])
        self.assertFalse(row['usage_incomplete'])
        self.assertEqual(row['helper_usage']['total_tokens'], 0)
        self.assertEqual(row['after_tokens'], hook.tokens(output))
        self.assertEqual(row['saved_tokens'], hook.tokens(source) - hook.tokens(output))
        self.assertLessEqual(row['after_tokens'], self.cfg['maximum_replacement_tokens'])
        self.assertEqual(Path(row['raw_file']).read_text(), source)
        self.assertEqual(Path(row['raw_file']).stat().st_mode & 0o777, 0o600)
        self.assertEqual(hook.stats()['helper_usage']['total_tokens'], 0)

    def test_graph_full_identities_pagination_omissions_no_helper(self):
        rows = graph_rows(12, docstring='verbose documentation ' * 400)
        source = json.dumps({'results': rows, 'total': 45, 'has_more': True})
        e = event(source, 'search_graph', offset=20, limit=12)
        output = hook.process(e, self.cfg, self.engine)['stopReason']
        compact = output[output.index('COMPACT GRAPH INDEX'):output.index('\n\nOriginal output:')]
        restored = read_index(compact)
        for before, after in zip(rows, restored):
            for key in ('name', 'label', 'qualified_name', 'file_path', 'signature'):
                self.assertEqual(before[key], after[key])
            self.assertEqual(before['lines'], after['line_count'])
        self.assertEqual(len(restored), 12)
        coverage = self.ledger()['coverage']
        self.assertEqual((coverage['total'], coverage['offset'], coverage['next_offset']), (45, 20, 32))
        self.assertTrue(coverage['has_more'])
        self.assertEqual(coverage['remaining'], 0)
        self.assertEqual(coverage['docstrings_omitted'], 12)
        self.assertEqual(coverage['metadata_fields_omitted'], ['complexity', 'extra_future_metadata'])
        self.check_accounting(source, output)

    def test_search_code_exact_sparse_excerpts_and_original_counters(self):
        source = code_source()
        body = json.dumps(source)
        e = event(body, mode='compact', context=3, limit=20)
        output = hook.process(e, self.cfg, self.engine)['stopReason']
        compact = json.loads(next(line for line in output.splitlines() if line.startswith('{"results":')))
        for key in source.keys() - {'results'}:
            self.assertEqual(compact[key], source[key])
        row = compact['results'][0]
        self.assertEqual(row['match_lines'], [101, 397])
        self.assertEqual([n for n, _ in row['excerpts']], [100, 101, 102, 396, 397, 398])
        for number, text in row['excerpts']:
            self.assertEqual(text, source['results'][0]['context'].splitlines()[number - 100])
        self.assertEqual(self.ledger()['coverage']['context_lines_omitted'], 294)
        self.assertNotIn('value_250', output)
        self.check_accounting(body, output)

    def test_missing_context_is_preserved_without_invented_excerpts(self):
        source = code_source()
        missing = {k: v for k, v in source['results'][0].items() if k not in ('context', 'context_start')}
        source['results'].append(missing)
        source['total_results'] = 2
        candidate = next(deterministic.candidates(json.dumps(source), 'search', event('', mode='compact')))
        compact = json.loads(candidate['text'].splitlines()[-1])
        self.assertEqual(compact['results'][1], missing)
        self.assertEqual(candidate['coverage']['rows_without_source_context'], 1)

    def test_search_code_preserves_source_lines_with_unicode_separators(self):
        lines = ['    text = "before\u2028after"', '    match = worker()',
                 '    next_value = 1', '    other = 2', '    last = 3']
        for newline in ('\n', '\r\n'):
            for trailing in ('', newline):
                with self.subTest(newline=newline, trailing=trailing):
                    source = code_source()
                    source['results'][0].update(context=newline.join(lines) + trailing, match_lines=[101])
                    candidates = list(deterministic.candidates(json.dumps(source), 'search', event('', mode='compact')))
                    self.assertEqual([c['coverage']['context_radius'] for c in candidates], [1, 0])
                    for candidate, selected in zip(candidates, ([0, 1, 2], [1])):
                        row = json.loads(candidate['text'].split('\n')[-1])['results'][0]
                        self.assertEqual(row['match_lines'], [101])
                        self.assertEqual(row['excerpts'], [[100 + i, lines[i]] for i in selected])
                        self.assertTrue(candidate['coverage']['all_returned_match_lines_retained'])
                        self.assertEqual(candidate['coverage']['context_lines_omitted'], len(lines) - len(selected))

    def test_ambiguous_or_unknown_search_contract_declines_local_filter(self):
        for modify in (lambda s: s.update(unknown='contract'),
                       lambda s: s['results'][0].update(context_start=102),
                       lambda s: s['results'][0].pop('context_start'),
                       lambda s: s['results'][0].update(match_lines=[999]),
                       lambda s: s['results'][0].update(context_start=True)):
            source = code_source()
            modify(source)
            self.assertEqual(list(deterministic.candidates(json.dumps(source), 'search', event('', mode='compact'))), [])
        self.assertEqual(list(deterministic.candidates(json.dumps(code_source()), 'search', event('', mode='full'))), [])

    def test_duplicate_keys_and_semantic_graph_decline_local_filter(self):
        source = json.dumps(code_source()).replace('"elapsed_ms": 19', '"elapsed_ms": 19, "elapsed_ms": 20')
        self.assertEqual(list(deterministic.candidates(source, 'search', event('', mode='compact'))), [])
        source = json.dumps({'results': graph_rows(1), 'total': 1, 'has_more': False})
        self.assertEqual(list(deterministic.candidates(source, 'search', event('', 'search_graph', semantic_query='worker'))), [])

    def test_jest_failure_evidence_fake_success_lines_and_counts_retained(self):
        source = jest_text(failed=True)
        e = {'hook_event_name': 'PostToolUse', 'tool_name': 'Bash', 'tool_input': {'command': 'npm test'},
             'tool_response': {'output': source, 'exit_code': 1}}
        output = hook.process(e, self.cfg, self.engine)['stopReason']
        self.assertIn(source[source.index('FAIL src/broken'):], output)
        self.assertIn('exit_code=1', output)
        self.assertEqual(self.ledger()['coverage']['passing_suite_lines_omitted'], 300)
        self.assertEqual(self.ledger()['coverage']['passing_test_lines_omitted'], 300)
        self.assertEqual(self.ledger()['coverage']['other_lines_omitted'], 0)
        self.check_accounting(source, output)

    def test_jest_ansi_and_console_diagnostics_preserved(self):
        source = jest_text().replace('PASS src/suite_0.test.js', '\x1b[32mPASS\x1b[0m src/suite_0.test.js')
        source = source.replace('PASS src/suite_2.test.js', '  console.warn\n    Warning: detail\nPASS src/suite_2.test.js')
        candidate = next(deterministic.candidates(source, 'logs', {}))
        self.assertEqual(candidate['coverage']['passing_suite_lines_omitted'], 2)
        self.assertIn(source[source.index('  console.warn'):], candidate['text'])

    def test_console_logged_pass_text_is_never_removed(self):
        source = jest_text().replace('PASS src/suite_2.test.js', '  console.log\nPASS source/string.test.js\nPASS src/suite_2.test.js')
        candidate = next(deterministic.candidates(source, 'logs', {}))
        self.assertEqual(candidate['coverage']['passing_suite_lines_omitted'], 2)
        self.assertIn(source[source.index('  console.log'):], candidate['text'])

    def test_jest_preserves_unicode_separators_and_line_endings(self):
        for newline in ('\n', '\r\n'):
            with self.subTest(newline=newline):
                message = 'source text\u2028PASS embedded.test.js' + newline
                source = message + jest_text().replace('\n', newline)
                candidate = next(deterministic.candidates(source, 'logs', {}))
                self.assertIn(message, candidate['text'])
                self.assertTrue(candidate['text'].endswith('Ran all test suites.' + newline))
                self.assertEqual(candidate['coverage']['original_lines'], 606)
                self.assertEqual(candidate['coverage']['retained_lines'], 6)

    def test_unknown_build_and_incomplete_jest_keep_existing_handling(self):
        for source in ('PASS custom task\n' * 1000, jest_text().replace('Snapshots: 0 total\n', ''),
                       'webpack compiled successfully\n' * 1000):
            self.assertEqual(list(deterministic.candidates(source, 'logs', {})), [])

    def test_budget_rejection_and_disable_use_existing_helper(self):
        source = json.dumps(code_source())
        e = event(source, mode='compact')
        for cfg in (dict(self.cfg, deterministic_first=False), dict(self.cfg, maximum_replacement_tokens=200)):
            helper = Mock(return_value={'status': 'summary', 'summary': 'Navigation only.',
                                        'usage': {'total_tokens': 7}, 'provider': 'test_helper'})
            hook.process(e, cfg, helper)
            helper.assert_called_once()
            self.assertEqual(helper.call_args.kwargs['text'], source)
            self.assertEqual(self.ledger()['helper_usage']['total_tokens'], 7)

    def test_failed_log_diagnostics_too_large_do_not_accept_local_filter(self):
        source = jest_text(True).replace('    Received: 9', '    Received: ' + 'exact failure evidence ' * 900)
        e = {'hook_event_name': 'PostToolUse', 'tool_name': 'Bash', 'tool_input': {'command': 'npm test'},
             'tool_response': {'output': source, 'exit_code': 1}}
        helper = Mock(return_value={'status': 'error', 'error': 'fixture'})
        self.assertIsNone(hook.process(e, self.cfg, helper))
        helper.assert_called_once()
        self.assertEqual(helper.call_args.kwargs['text'], source)

    def test_raw_sensitive_rich_exact_and_small_stay_untouched(self):
        source = json.dumps(code_source())
        events = [event(source, mode='compact', note='context-saver:raw'),
                  event(source, 'get_code_snippet'), event('small', mode='compact'),
                  event(source.replace('exact raw', 'api_key=123456789secret'), mode='compact')]
        rich = event(source, mode='compact')
        rich['tool_response']['content'].append({'type': 'image', 'data': 'fixture'})
        events.append(rich)
        structured = event(source, mode='compact')
        structured['tool_response']['structuredContent'] = {'exact': 1}
        events.append(structured)
        for e in events:
            self.assertIsNone(hook.process(e, self.cfg, self.engine))
        self.engine.assert_not_called()


if __name__ == '__main__':
    unittest.main()
