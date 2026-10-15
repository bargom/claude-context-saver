"""Pure, bounded discovery-source reduction; callers own eligibility and archives.

This module never summarizes behavior. Graph indexes retain all returned identities
or explicitly become partial archive previews; unknown JSON contracts stay intact.
"""
import json
import os.path


_IDENTITY = ('name', 'label', 'file_path', 'qualified_name')
_RETAINED = frozenset((*_IDENTITY, 'signature', 'lines', 'docstring'))
_CATEGORIES = frozenset(('search', 'logs', 'git', 'documentation'))


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def _integer(value):
    return type(value) is int and value >= 0


def _source_lines(text, keepends=False):
    """Split LF/CRLF lines without splitting Unicode separators."""
    lines = text.split('\n')
    result = [line + '\n' if keepends else line.removesuffix('\r') for line in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _graph(text, category, context):
    if category != 'search' or not isinstance(context, dict):
        return None
    tool = context.get('tool_name', '')
    if not isinstance(tool, str) or tool.split('__')[-1] != 'search_graph':
        return None
    arguments = context.get('tool_input', {})
    if not isinstance(arguments, dict) or 'semantic_query' in arguments:
        return None
    offset = arguments.get('offset', 0)
    if not _integer(offset):
        return None
    try:
        source = json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, TypeError, RecursionError):
        return None
    # Additional top-level fields can represent separate result sets or contracts.
    if not isinstance(source, dict) or set(source) != {'total', 'results', 'has_more'}:
        return None
    total, rows, more = source['total'], source['results'], source['has_more']
    if not _integer(total) or not isinstance(rows, list) or type(more) is not bool:
        return None
    if total < offset + len(rows) or more != (total > offset + len(rows)):
        return None
    if more and not rows:
        return None
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in _IDENTITY):
            return None
        if 'lines' in row and not _integer(row['lines']):
            return None
        if any(k in row and not isinstance(row[k], str) for k in ('signature', 'docstring')):
            return None
    return source, offset


def _coverage(source, offset, shown, docs):
    returned = len(source['results'])
    result = {'total': source['total'], 'returned': returned,
              'has_more': source['has_more'], 'offset': offset,
              'shown': shown, 'remaining': returned - shown,
              'docstrings_included': docs,
              'docstrings_omitted': sum('docstring' in r for r in source['results']) - docs,
              'metadata_fields_omitted': sorted(set().union(
                  *(set(r) - _RETAINED for r in source['results'])))}
    if source['has_more']:
        result['next_offset'] = offset + returned
    return result


def _render_graph(source, offset, shown, include_docs, prefixes, preview=False):
    rows = source['results'][:shown]
    coverage = _coverage(source, offset, shown,
                         sum('docstring' in r for r in rows) if include_docs else 0)
    # Coverage remains exact in the return value; only metadata key names in the
    # human-readable source are bounded, since arbitrary key names can be huge.
    visible = {k: v for k, v in coverage.items() if k != 'metadata_fields_omitted'}
    omitted = coverage['metadata_fields_omitted']
    visible['metadata_field_names_omitted_count'] = len(omitted)
    title = ('ARCHIVE PREVIEW — NOT A SUMMARY; only the shown graph rows are included.'
             if preview else 'COMPACT GRAPH INDEX — all returned match identities are retained.')
    notices = [title, 'Coverage: ' + _json(visible),
               'lines is rendered as line_count; it is a count, never a source line position.',
               'Other row metadata is omitted; docstring omissions are counted above.']
    if omitted:
        names = _json(omitted)
        notices.append('Omitted metadata field names: ' + (names if len(names) <= 800 else
                       '[names too large; inspect archived original]'))
    if preview:
        notices.append('Read the archived original for remaining rows and omitted details; no behavior is inferred.')
    columns = list(_IDENTITY) + ['signature', 'line_count']
    if include_docs:
        columns.append('docstring')
    notices.append('Columns: ' + _json(columns) + '; null means field absent.')
    if prefixes:
        notices.append('Exact reconstruction: prepend each prefix to its corresponding row value: ' + _json(prefixes))
    for row in rows:
        values = [row[k][len(prefixes.get(k, '')):] for k in _IDENTITY]
        values.extend((row.get('signature'), row.get('lines')))
        if include_docs:
            values.append(row.get('docstring'))
        notices.append(_json(values))
    return '\n'.join(notices), coverage


def _graph_preview(source, offset, budget):
    # Deliberately omit prefixes here: a single huge common prefix must not make
    # the archive navigation fallback itself unbounded.
    best, coverage = _render_graph(source, offset, 0, False, {}, preview=True)
    if len(best) > budget:
        coverage = _coverage(source, offset, 0, 0)
        visible = {k: v for k, v in coverage.items() if k != 'metadata_fields_omitted'}
        best = ('ARCHIVE PREVIEW — NOT A SUMMARY.\nCoverage: ' + _json(visible) +
                '\nNo rows fit this preview; read the archived original for all identities and details.')
    for shown in range(1, len(source['results']) + 1):
        candidate, candidate_coverage = _render_graph(source, offset, shown, False, {}, preview=True)
        if len(candidate) > budget:
            break
        best, coverage = candidate, candidate_coverage
    return best, coverage


def _text_preview(text, budget):
    # Recompute after inserting exact counters, so Unicode and large digit counts
    # cannot push the replacement past the configured character budget.
    count = max(0, budget - 650)
    while True:
        head = (count + 1) // 2
        tail = count // 2
        omitted = len(text) - count
        preview = ('ARCHIVE PREVIEW — NOT A SUMMARY; excerpts do not cover the full output.\n'
                   f'Original characters: {len(text)}; shown: {count}; omitted: {omitted}.\n'
                   'Read the archived original for complete output and exact details.\n'
                   f'--- FIRST {head} CHARACTERS ---\n' + text[:head] +
                   f'\n--- {omitted} CHARACTERS OMITTED ---\n' +
                   (text[-tail:] if tail else '') + f'\n--- LAST {tail} CHARACTERS END ---')
        if len(preview) <= budget:
            return preview, {'original_chars': len(text), 'shown_chars': count,
                             'omitted_chars': omitted, 'complete': False}
        count = max(0, count - (len(preview) - budget))


def _json_contract(text):
    """Recognize a JSON value or whitespace-separated JSON stream, including scalars."""
    decoder = json.JSONDecoder()
    index = 0
    parsed = False
    while True:
        start = index
        while index < len(text) and text[index].isspace():
            index += 1
        if index == len(text):
            return parsed
        # Preserve truncated objects/arrays too, including after a scalar row.
        if text[index] in '{[':
            return True
        if parsed and index == start:
            return False
        try:
            _, index = decoder.raw_decode(text, index)
        except (ValueError, RecursionError):
            return False
        parsed = True


def prepare(text, category, context=None, max_chars=60000):
    """Prepare only oversized eligible text; return None for unknown contracts.

    ``context`` is the original hook event (tool_name/tool_input) for graph data.
    Every result supplies text, kind, coverage, helper_eligible, and fallback_text.
    A helper may summarize graph_index; archive_preview is a local-only excerpt.
    The caller must screen secrets/contracts first and archive the full original.
    """
    if (not isinstance(text, str) or category not in _CATEGORIES or
            type(max_chars) is not int or max_chars < 512 or len(text) <= max_chars):
        return None
    budget = min(4000, max_chars)
    graph = _graph(text, category, context)
    if graph is not None:
        source, offset = graph
        fallback, fallback_coverage = _graph_preview(source, offset, budget)
        if len(fallback) > budget:
            # Pathological pagination counters can themselves exceed the budget;
            # preserving the original is safer than clipping their exact values.
            return None
        prefixes = {}
        for field in ('file_path', 'qualified_name'):
            prefix = os.path.commonprefix([r[field] for r in source['results']])
            if len(prefix) >= 8:
                prefixes[field] = prefix
        for docs in (True, False):
            compact, coverage = _render_graph(source, offset, len(source['results']), docs, prefixes)
            if len(compact) <= max_chars:
                return {'text': compact, 'kind': 'graph_index', 'coverage': coverage,
                        'helper_eligible': True, 'fallback_text': fallback,
                        'fallback_coverage': fallback_coverage}
        return {'text': fallback, 'kind': 'archive_preview', 'coverage': fallback_coverage,
                'helper_eligible': False, 'fallback_text': fallback,
                'fallback_coverage': fallback_coverage}
    # Preserve JSON, malformed/truncated JSON, and JSON-lines contracts rather
    # than treating unrecognized structured output as a prose log.
    if _json_contract(text):
        return None
    preview, coverage = _text_preview(text, budget)
    return {'text': preview, 'kind': 'archive_preview', 'coverage': coverage,
            'helper_eligible': False, 'fallback_text': preview,
            'fallback_coverage': coverage}
