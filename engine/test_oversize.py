import json
import unittest

import oversize


EVENT = {'tool_name': 'mcp__codebase_memory_mcp__search_graph', 'tool_input': {}}


def incident_rows(count=164):
    """A redacted stand-in for the 164-match search_graph capture of the
    incident: short docstrings, bulky metadata the index leaves out."""
    return [{'name': f'match_{i}', 'label': ('Function', 'Method', 'Variable')[i % 3],
             'file_path': f'src/lib/session/module_{i % 17}.js',
             'qualified_name': f'project.src.lib.session.module_{i % 17}.match_{i}',
             'signature': f'match_{i}(session, options = {{}})', 'lines': 20 + i,
             'docstring': f'Redacted description of match {i}; keeps the session state for the call.',
             'complexity': i % 9, 'properties': {'body_hash': f'{i:064x}', 'callers': [f'caller_{j}' for j in range(40)],
                                                'source_excerpt': 'redacted source line\n' * 20}}
            for i in range(count)]


def graph_rows(count=164, docstring='Graph search documentation. ' * 70):
    labels = ('Function', 'Variable', 'Route', 'Class', 'Odd|Label\nλ')
    return [{'name': f'match_{i}', 'label': labels[i % len(labels)],
             'file_path': f'src/lib/session/file_{i}.js',
             'qualified_name': f'Project.src.lib.session.file_{i}.match_{i}',
             'signature': f'match_{i}(value, options)', 'lines': i,
             'docstring': docstring, 'complexity': 3, 'extra_future_metadata': {'a': 1}}
            for i in range(count)]


def source(rows, **extra):
    return json.dumps({'total': len(rows), 'results': rows, 'has_more': False, **extra})


def read_index(text):
    """Read the documented table representation as a downstream caller would."""
    lines = text.splitlines()
    columns = json.loads(next(line for line in lines if line.startswith('Columns: '))
                         .removeprefix('Columns: ').split('; null means')[0])
    prefixes = {}
    prefix_notice = 'Exact reconstruction: prepend each prefix to its corresponding row value: '
    for line in lines:
        if line.startswith(prefix_notice):
            prefixes = json.loads(line.removeprefix(prefix_notice))
    rows = []
    for line in lines:
        if line.startswith('['):
            row = dict(zip(columns, json.loads(line)))
            for key, prefix in prefixes.items():
                row[key] = prefix + row[key]
            rows.append(row)
    return rows


class OversizedResultsTests(unittest.TestCase):
    def test_graph_retains_every_identity_signature_and_line_count(self):
        rows = graph_rows()
        result = oversize.prepare(source(rows), 'search', EVENT)
        self.assertEqual(result['kind'], 'graph_index')
        self.assertLessEqual(len(result['text']), 60000)
        restored = read_index(result['text'])
        self.assertEqual(len(restored), 164)
        for expected, actual in zip(rows, restored):
            for key in ('name', 'label', 'file_path', 'qualified_name', 'signature'):
                self.assertEqual(actual[key], expected[key])
            self.assertEqual(actual['line_count'], expected['lines'])
        self.assertIn('never a source line position', result['text'])
        self.assertEqual(result['coverage']['remaining'], 0)
        self.assertEqual(result['coverage']['docstrings_omitted'], 164)
        self.assertIn('docstring omissions are counted', result['text'])
        self.assertIn('extra_future_metadata', result['text'])
        self.assertNotIn('next_offset', result['coverage'])

    def test_164_match_incident_preserves_all_docstrings(self):
        raw = source(incident_rows())
        expected = json.loads(raw)['results']
        result = oversize.prepare(raw, 'search', EVENT)
        self.assertGreater(len(raw), 180000)
        self.assertEqual(len(expected), 164)
        self.assertEqual(result['coverage']['shown'], 164)
        self.assertLessEqual(len(result['text']), 60000)
        actual = read_index(result['text'])
        for before, after in zip(expected, actual):
            for key in ('name', 'label', 'file_path', 'qualified_name', 'signature', 'docstring'):
                self.assertEqual(before.get(key), after.get(key))
            self.assertEqual(before.get('lines'), after.get('line_count'))
        self.assertEqual(result['coverage']['docstrings_omitted'], 0)

    def test_pagination_comes_from_request_and_returned_count(self):
        rows = graph_rows(3)
        event = {**EVENT, 'tool_input': {'offset': 7, 'limit': 3}}
        result = oversize.prepare(source(rows, total=15, has_more=True), 'search', event, max_chars=4000)
        self.assertEqual(result['coverage']['offset'], 7)
        self.assertEqual(result['coverage']['total'], 15)
        self.assertEqual(result['coverage']['returned'], 3)
        self.assertTrue(result['coverage']['has_more'])
        self.assertEqual(result['coverage']['next_offset'], 10)

    def test_unknown_root_semantic_and_malformed_contracts_are_not_previews(self):
        rows = graph_rows(4)
        for extra in ({'semantic_results': []}, {'warnings': ['partial']}, {'offset': 0}):
            with self.subTest(extra=extra):
                self.assertIsNone(oversize.prepare(source(rows, **extra), 'search', EVENT, max_chars=512))
        for text in ('{"data":' + ' ' * 1000, '[1,' + ' ' * 1000,
                     json.dumps({'machine_contract': 'x' * 2000}), '"' + 'x' * 2000 + '"'):
            self.assertIsNone(oversize.prepare(text, 'logs', EVENT, max_chars=512))
        event = {**EVENT, 'tool_input': {'semantic_query': ['x']}}
        self.assertIsNone(oversize.prepare(source(rows), 'search', event, max_chars=512))

    def test_duplicate_keys_and_invalid_pagination_are_rejected(self):
        text = source(graph_rows(3))
        duplicate = text.replace('"total": 3', '"total": 9, "total": 3', 1)
        for raw in (duplicate, source(graph_rows(3), total=2),
                    source(graph_rows(3), total=3, has_more=True)):
            self.assertIsNone(oversize.prepare(raw, 'search', EVENT, max_chars=512))

    def test_scalar_first_json_streams_preserve_unknown_contracts(self):
        payload = {'data': 'x' * 70000}
        for scalar in ('null', 'true', '42', '"header"'):
            for body in (json.dumps(payload), json.dumps(payload, indent=2)):
                with self.subTest(scalar=scalar, multiline='\n' in body):
                    self.assertIsNone(oversize.prepare('\n' + scalar + '\n' + body, 'logs'))
        self.assertIsNone(oversize.prepare('null\n' * 15000, 'logs'))
        self.assertIsNone(oversize.prepare('null\n{"truncated":' + ' ' * 70000, 'logs'))

    def test_bracketed_log_prefixes_are_prose_not_json(self):
        for line in ('[INFO] worker finished a batch\n', '[2026-10-17 12:00:00] worker finished a batch\n',
                     '{level=info} worker finished a batch\n', '[1] first step done\n'):
            with self.subTest(line=line):
                result = oversize.prepare(line * 4000, 'logs')
                self.assertEqual(result['kind'], 'archive_preview')
                self.assertIn(line.strip(), result['text'])

    def test_cut_off_json_is_preserved(self):
        rows = [{'id': i, 'name': f'row {i}', 'tags': ['a', 'b']} for i in range(3000)]
        array, lines = json.dumps(rows), '\n'.join(json.dumps(row) for row in rows)
        for text in (array[:-30], array[:int(len(array) * .95)], lines[:-10], json.dumps({'rows': rows})[:-1]):
            with self.subTest(end=text[-20:]):
                self.assertIsNone(oversize.prepare(text, 'logs'))
        # Valid JSON documents and streams stay exact too.
        for text in (array, lines, json.dumps({'rows': rows}, indent=2)):
            self.assertIsNone(oversize.prepare(text, 'logs'))

    def test_json_that_breaks_early_is_prose(self):
        # Valid for a while, then not JSON at all well before the end.
        text = '[1, 2, 3] ' + 'worker log line without structure\n' * 3000
        self.assertEqual(oversize.prepare(text, 'logs')['kind'], 'archive_preview')
        broken = '{"a": 1, oops ' + 'x' * 70000 + '}'
        self.assertEqual(oversize.prepare(broken, 'logs')['kind'], 'archive_preview')
        deep = '[' * 100000 + ']' * 100000
        self.assertIsNone(oversize.prepare(deep, 'logs'))

    def test_numeric_log_line_followed_by_prose_still_gets_preview(self):
        raw = '42\n' + ('Build worker completed a batch.\n' * 3000)
        result = oversize.prepare(raw, 'logs')
        self.assertEqual(result['kind'], 'archive_preview')
        self.assertIn('42\nBuild worker', result['text'])

    def test_graph_fallback_accounts_for_unshown_rows_without_behavior_claims(self):
        rows = graph_rows(164, docstring='')
        result = oversize.prepare(source(rows), 'search', EVENT, max_chars=4000)
        self.assertEqual(result['kind'], 'archive_preview')
        self.assertFalse(result['helper_eligible'])
        self.assertLessEqual(len(result['text']), 4000)
        self.assertIn('NOT A SUMMARY', result['text'])
        coverage = result['coverage']
        self.assertGreater(coverage['remaining'], 0)
        self.assertEqual(coverage['shown'] + coverage['remaining'], 164)
        shown = read_index(result['text'])
        self.assertEqual(len(shown), coverage['shown'])
        self.assertEqual([r['name'] for r in shown], [r['name'] for r in rows[:len(shown)]])

    def test_single_enormous_identity_gives_zero_row_archive_navigation(self):
        row = graph_rows(1)[0]
        row['qualified_name'] = 'q' * 100000
        result = oversize.prepare(source([row]), 'search', EVENT)
        self.assertLessEqual(len(result['text']), 4000)
        self.assertEqual(result['coverage']['shown'], 0)
        self.assertEqual(result['coverage']['remaining'], 1)
        self.assertFalse(result['helper_eligible'])

    def test_generic_preview_preserves_exact_unicode_edges_and_character_counts(self):
        raw = 'BEGIN λ\n' + ('middle not summarized\n' * 10000) + '\nEND Ω'
        for category in ('logs', 'search', 'documentation', 'git'):
            result = oversize.prepare(raw, category)
            self.assertFalse(result['helper_eligible'])
            self.assertLessEqual(len(result['text']), 4000)
            self.assertIn('BEGIN λ\n', result['text'])
            self.assertIn('\nEND Ω', result['text'])
            counts = result['coverage']
            self.assertEqual(counts['original_chars'], len(raw))
            self.assertEqual(counts['shown_chars'] + counts['omitted_chars'], len(raw))
            self.assertIn(f"{counts['omitted_chars']} CHARACTERS OMITTED", result['text'])
            self.assertIn('NOT A SUMMARY', result['text'])

    def test_small_results_unsupported_categories_and_bad_limits_stay_raw(self):
        self.assertIsNone(oversize.prepare('small', 'logs'))
        self.assertIsNone(oversize.prepare('a' * 100000, 'source_code'))
        self.assertIsNone(oversize.prepare('a' * 100000, 'logs', max_chars=1))


if __name__ == '__main__':
    unittest.main()
