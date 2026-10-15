"""Conservative discovery/log filtering without model calls or behavior claims.

Callers screen eligibility and secrets, archive originals and enforce the complete
replacement token budget. Graph/code candidates retain every match; docs search
can explicitly become a partial archive index when all identities cannot fit.
"""
import json
import os.path
import re

from oversize import _graph, _integer, _json, _render_graph, _source_lines, _unique_object


_SEARCH_KEYS = frozenset(('results', 'raw_matches', 'directories', 'total_grep_matches',
                          'total_results', 'raw_match_count', 'elapsed_ms', 'dedup_ratio'))
_ROW_KEYS = frozenset(('node', 'qualified_name', 'label', 'file', 'start_line', 'end_line',
                       'in_degree', 'out_degree', 'match_lines', 'context', 'context_start'))
_ANSI = re.compile(r'\x1b\[[0-9;]*m')
_JEST_SUMMARY = re.compile(r'^(Test Suites|Tests|Snapshots):\s+(?:\d+ (?:failed|passed|skipped|pending|todo|obsolete|written),?\s*)*\d+ total\s*$')
_PASS_SUITE = re.compile(r'^ ?PASS\s+\S+\.(?:[cm]?[jt]sx?)(?:\s+\([\d.]+\s*s\))?\s*$')
_PASS_TEST = re.compile(r'^\s+[✓✔]\s+.+\(\d+(?:\.\d+)?\s*(?:m?s)\)\s*$')
_DIAGNOSTIC = re.compile(r'\b(?:FAIL|Error|error|failed|Exception|Warning|warning)\b|^\s*●|^\s*console\.(?:log|info|debug|warn|error)\b')
_DOC_KEYS = frozenset(('hits', 'nbHits', 'page', 'nextCursor'))
_DOC_HIT_KEYS = frozenset(('url', 'url_without_anchor', 'anchor', 'content', 'type',
                           'hierarchy', 'objectID', '_snippetResult', '_highlightResult'))


def docs_search_source(text):
    """Recognize the docs search navigation envelope, never an unknown contract."""
    try:
        source = json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, TypeError, RecursionError):
        return None
    if (not isinstance(source, dict) or not {'hits', 'nbHits', 'page'} <= set(source) <= _DOC_KEYS or
            not isinstance(source['hits'], list) or not _integer(source['nbHits']) or
            not _integer(source['page']) or source['nbHits'] < len(source['hits']) or
            ('nextCursor' in source and source['nextCursor'] is not None and not isinstance(source['nextCursor'], str))):
        return None
    for hit in source['hits']:
        if (not isinstance(hit, dict) or not {'url', 'objectID', 'hierarchy', 'content'} <= set(hit) <= _DOC_HIT_KEYS or
                any(not isinstance(hit.get(k), str) for k in ('url', 'objectID')) or
                not hit['url'].startswith(('https://', 'http://')) or
                any(k in hit and hit[k] is not None and not isinstance(hit[k], str)
                    for k in ('url_without_anchor', 'anchor', 'content', 'type')) or
                not isinstance(hit['hierarchy'], dict) or
                not set(hit['hierarchy']) <= {f'lvl{i}' for i in range(7)} or
                any(v is not None and not isinstance(v, str) for v in hit['hierarchy'].values()) or
                any(k in hit and not isinstance(hit[k], dict) for k in ('_snippetResult', '_highlightResult'))):
            return None
    return source


def _docs_index(source, shown, snippet_chars):
    rows, snippet_omitted, title_omitted = [], 0, 0
    for hit in source['hits'][:shown]:
        title = ' > '.join(hit['hierarchy'].get(f'lvl{i}') for i in range(7) if hit['hierarchy'].get(f'lvl{i}'))
        snippet = hit.get('_snippetResult', {}).get('content', {})
        snippet = snippet.get('value') if isinstance(snippet, dict) else None
        if not isinstance(snippet, str):
            snippet = hit.get('content') or ''
        snippet_omitted += max(0, len(snippet) - snippet_chars)
        title_omitted += max(0, len(title) - 240)
        row = {'objectID': hit['objectID'], 'url': hit['url'], 'title': title[:240]}
        if 'type' in hit:
            row['type'] = hit['type']
        if snippet_chars:
            row['snippet'] = snippet[:snippet_chars]
        rows.append(row)
    coverage = {'returned': len(source['hits']), 'shown': shown, 'remaining': len(source['hits']) - shown,
                'snippet_characters_omitted': snippet_omitted, 'title_characters_omitted': title_omitted,
                'fields_omitted': sorted(set().union(*(set(h) for h in source['hits'])) - {'objectID', 'url', 'type'})}
    body = ('DETERMINISTIC DOCS SEARCH INDEX — navigation only; shown URLs and objectIDs are exact.\n'
            'Titles join hierarchy levels (prefix up to 240 characters); snippets are search-snippet/content prefixes, not exact page source.\n'
            'Full content, duplicate highlight/snippet wrappers and listed metadata are archived; remaining hits are archive-only.\n'
            'Server counts/page/nextCursor are unchanged; follow nextCursor for further pages and read the archive for omitted hits.\n'
            'Coverage: ' + _json(coverage) + '\n' + _json({**source, 'hits': rows}))
    return {'text': body, 'kind': 'docs_search_index', 'coverage': coverage}


def _docs_search(text, category, event):
    if category != 'documentation' or event.get('tool_name', '').split('__')[-1] != 'search_openai_docs':
        return
    source = docs_search_source(text)
    if source is None:
        return
    # The caller measures the *whole* replacement, including archive and label.
    # Keep all identities first, then explicitly reduce archive previews; no helper
    # is needed for a normal docs page even when an explicit limit returns many hits.
    for chars in (320, 160, 0):
        yield _docs_index(source, len(source['hits']), chars)
    shown = len(source['hits']) // 2
    while shown:
        yield _docs_index(source, shown, 160)
        shown //= 2
    yield _docs_index(source, 0, 0)


def _search_code(text, category, event, radius):
    if category != 'search' or event.get('tool_name', '').split('__')[-1] != 'search_code':
        return None
    if event.get('tool_input', {}).get('mode') != 'compact':
        return None
    try:
        source = json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(source, dict) or set(source) != _SEARCH_KEYS:
        return None
    rows, raw = source['results'], source['raw_matches']
    if (not isinstance(rows, list) or not isinstance(raw, list) or
            not isinstance(source['directories'], dict) or
            any(not _integer(source[k]) for k in ('total_grep_matches', 'total_results', 'raw_match_count', 'elapsed_ms')) or
            not isinstance(source['dedup_ratio'], str) or source['total_results'] < len(rows) or
            source['raw_match_count'] < len(raw)):
        return None
    # Raw matches are preserved verbatim as JSON values; never guess their schema.
    filtered, omitted, unavailable = [], 0, 0
    for row in rows:
        if (not isinstance(row, dict) or set(row) not in (_ROW_KEYS, _ROW_KEYS - {'context', 'context_start'}) or
                any(not isinstance(row[k], str) for k in ('node', 'qualified_name', 'label', 'file')) or
                any(not _integer(row[k]) for k in ('start_line', 'end_line', 'in_degree', 'out_degree')) or
                row['start_line'] < 1 or row['end_line'] < row['start_line']):
            return None
        matches = row['match_lines']
        if not isinstance(matches, list) or not matches or any(not _integer(n) or n < 1 for n in matches):
            return None
        if 'context' not in row:
            filtered.append(row)
            unavailable += 1
            continue
        if (not isinstance(row['context'], str) or not _integer(row['context_start']) or
                row['context_start'] < 1):
            return None
        lines = _source_lines(row['context'])
        start = row['context_start']
        if any(n < start or n >= start + len(lines) for n in matches):
            return None
        selected = set()
        for number in matches:
            selected.update(range(max(0, number - start - radius), min(len(lines), number - start + radius + 1)))
        omitted += len(lines) - len(selected)
        filtered.append({**{k: v for k, v in row.items() if k not in ('context', 'context_start')},
                         'excerpts': [[start + index, lines[index]] for index in sorted(selected)]})
    if not omitted:
        return None
    coverage = {'returned': len(rows), 'shown': len(rows), 'remaining': 0,
                'returned_raw_matches': len(raw), 'context_lines_omitted': omitted,
                'context_radius': radius, 'all_returned_match_lines_retained': True,
                'rows_without_source_context': unavailable}
    body = ('DETERMINISTIC SEARCH INDEX — all returned identities, match-line numbers and available matching excerpts retained; no behavior inferred.\n'
            'Excerpts are [exact source line number, exact line text]; gaps omit non-matching context.\n'
            'Rows without source context retain only original metadata; no source excerpt is invented.\n'
            'Counts are original server counts; returned results may be limited, and completeness is not inferred.\n'
            'Coverage: ' + _json(coverage) + '\n' + _json({**source, 'results': filtered}))
    return {'text': body, 'kind': 'search_code_index', 'coverage': coverage}


def _jest(text, category):
    if category != 'logs':
        return None
    lines = _source_lines(text, keepends=True)
    plain = [_ANSI.sub('', line).rstrip('\r\n') for line in lines]
    # Require Jest's complete count footer, not a lone PASS-looking application log.
    footers = {match.group(1) for line in plain if (match := _JEST_SUMMARY.fullmatch(line))}
    if not {'Test Suites', 'Tests', 'Snapshots'} <= footers:
        return None
    selected, suites, tests = [], 0, 0
    diagnostics_started, passing_suite = False, False
    for original, line in zip(lines, plain):
        # Once diagnostics begin, keep the entire remainder: received source or
        # application logs can themselves contain PASS/checkmark-looking lines.
        diagnostics_started = diagnostics_started or bool(_DIAGNOSTIC.search(line))
        if not diagnostics_started and _PASS_SUITE.fullmatch(line):
            suites += 1
            passing_suite = True
        elif not diagnostics_started and passing_suite and _PASS_TEST.fullmatch(line):
            tests += 1
        else:
            selected.append(original)
            if line.strip():
                passing_suite = False
    if not suites and not tests:
        return None
    coverage = {'original_lines': len(lines), 'retained_lines': len(selected),
                'passing_suite_lines_omitted': suites, 'passing_test_lines_omitted': tests,
                'other_lines_omitted': 0}
    body = ('DETERMINISTIC JEST LOG FILTER — only passing suite/test lines omitted.\n'
            'All other lines, including failure diagnostics and original count footers, follow unchanged.\n'
            'Coverage: ' + _json(coverage) + '\n' + ''.join(selected))
    return {'text': body, 'kind': 'jest_success_filter', 'coverage': coverage}


def candidates(text, category, event, cfg=None):
    """Yield richest candidates first, with explicit docs preview omissions."""
    yield from _docs_search(text, category, event)
    graph = _graph(text, category, event)
    if graph is not None:
        source, offset = graph
        prefixes = {}
        for field in ('file_path', 'qualified_name'):
            prefix = os.path.commonprefix([row[field] for row in source['results']])
            if len(prefix) >= 8:
                prefixes[field] = prefix
        for docs in (True, False):
            body, coverage = _render_graph(source, offset, len(source['results']), docs, prefixes)
            yield {'text': body, 'kind': 'graph_index', 'coverage': coverage}
        return
    for radius in (1, 0):
        candidate = _search_code(text, category, event, radius)
        if candidate:
            yield candidate
    candidate = _jest(text, category)
    if candidate:
        yield candidate
    if cfg and cfg.get('rtk_enabled', False):
        from rtk_filter import candidate as rtk_candidate
        candidate = rtk_candidate(text, category, event, cfg)
        if candidate:
            yield candidate
