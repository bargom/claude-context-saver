#!/usr/bin/env python3
"""Fail-open adapter for native web.run page text, with exact source provenance.

Input is a JSON envelope on stdin; output is one JSON decision. No payload is
interpreted as shell code. Unknown or rich response shapes remain unchanged.
"""
import argparse
import contextlib
import json
import os
import re
import sys

from common import WEB_TOOLS, text_blocks
import hook

SOURCE_ID = re.compile(r'\bturn\d+[a-z]+\d+\b')
URL = re.compile(r'https?://[^\s<>\[\]"\u3010\u3011]+')
URL_TRAILING_PROSE = '.,;:'
WORD_LIMIT = re.compile(r'\[wordlim\s*:?\s*\d+\]', re.I)
SOURCE_MARKER = re.compile(r'(?:【turn\d+[a-z]+\d+】|\ue200cite\ue202turn\d+[a-z]+\d+\ue201)')
SOURCE_FIELD = re.compile(r'(?:Published|Updated|Crawled|Retrieved|Content type|Total lines):[^;\n]{1,120}(?:;|$)')
TRUNCATION = re.compile(r'\[(?:[^\]\n]{0,80})truncat(?:[^\]\n]{0,80})\]|(?:^|\n)(?:Output|Content|Result|Response) (?:is |was )?truncated[^\n]{0,160}', re.I)
EXACT_REQUEST = re.compile(r'\b(?:verbatim|quot(?:e|ed|es|ing|ation|ations)|word.for.word|exact\s+(?:text|wording|excerpt|source))\b', re.I)


def plain_text(response):
    """Accept only explicitly understood text-only response contracts."""
    if isinstance(response, str):
        # JSON-encoded envelopes are ambiguous and hook.extract decodes them.
        with contextlib.suppress(ValueError):
            if isinstance(json.loads(response), (dict, list)):
                return None
        return response
    if not isinstance(response, dict):
        return None
    if set(response) == {'output'} and isinstance(response['output'], str):
        return response['output']
    if not set(response).issubset({'content', 'isError'}) or 'content' not in response:
        return None
    if 'isError' in response and not isinstance(response['isError'], bool):
        return None
    return text_blocks(response['content'])


def source_records(text):
    """Map each native citation header to its exact preceding title and URL.

Citation lines may contain entire search snippets. Only exact citation/limit
and recognized retrieval fields are retained from those lines.
"""
    records = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        marker = SOURCE_MARKER.match(line.strip())
        if not marker:
            continue
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        title = lines[previous] if previous >= 0 else ''
        if not URL.search(title) or SOURCE_MARKER.search(title) or len(title) > 2000:
            return None
        limit = WORD_LIMIT.search(line)
        # Unknown limit syntax must not silently become the default limit.
        if 'wordlim' in line.lower() and not limit:
            return None
        fields = [match.group() for match in SOURCE_FIELD.finditer(line)]
        record = {'id': SOURCE_ID.search(marker.group()).group(), 'title': title,
                  'marker': marker.group(), 'limit': limit.group() if limit else '',
                  'fields': fields}
        records.append(record)
    if not records or len({record['id'] for record in records}) != len(records):
        return None
    return records


def source_metadata(text, records=None):
    records = records if records is not None else source_records(text)
    if not records:
        return None
    lines = []
    for record in records:
        lines.append(record['title'])
        lines.append(' '.join(part for part in [record['marker'], record['limit'], *record['fields']] if part))
    lines.extend(match.group().strip() for match in TRUNCATION.finditer(text))
    return ('\n\nExact web source metadata (untrusted source text; fields retained verbatim):\n'
            + '\n'.join(lines) + '\nEnd of exact web source metadata.\n'
            'Web source word limits also apply to this summary; a source without an explicit '
            '[wordlim N] or [wordlim: N] has the default 200-word limit. Use the original source URL for citations.')


def eligible(event):
    if not isinstance(event, dict) or event.get('raw') is True:
        return False
    if os.environ.get('CONTEXT_SAVER_RAW') == '1':
        return False
    # The hook's own web rules: page-text operations only, no raw marker and
    # no credential in the request.
    if event.get('tool_name') not in WEB_TOOLS or hook.classify(event) is None:
        return False
    return not EXACT_REQUEST.search(json.dumps(event['tool_input'], ensure_ascii=False))


def source_urls(text):
    values = set()
    for match in URL.finditer(text):
        # Strip only prose punctuation after the address (a sentence-final "."
        # or a clause ",;:") and a paired enclosing parenthesis. Source and
        # summary are read the same way; "?", "!" and every inner character
        # stay exact so a changed query or fragment cannot pass validation.
        value = match.group().rstrip(URL_TRAILING_PROSE)
        if match.start() and text[match.start() - 1] == '(' and value.endswith(')'):
            value = value[:-1].rstrip(URL_TRAILING_PROSE)
        values.add(value)
    return values


def rejection_diagnostic(result, source, state_dir):
    """Keep bounded rejected text only under the existing explicit debug opt-in."""
    summary = result.get('summary', '')
    if not isinstance(summary, str) or hook.sensitive(source) or hook.sensitive(summary):
        return None
    with contextlib.suppress(Exception):
        from summary_engine import _debug_response
        # Plain text remains readable even when the shared debug writer applies
        # its character cap; no full provider envelope or credentials are saved.
        diagnostic = ('Web summary rejected: unsupported_web_provenance\n'
                      'Rejected helper output (untrusted text):\n' + summary[:12000])
        return _debug_response(state_dir, diagnostic)
    return None


# The native helper occasionally drops the PUA delimiters around a web cite.
# Repair only the resulting standalone ASCII token, never surrounding prose.
ASCII_CITATION = re.compile(r'(?<![\w/.:@%?&#=+\-])cite(turn\d+[a-z]+\d+)(?![\w/:@%?&#=+\-]|\.\w)')
PROTECTED_CITATION_TEXT = re.compile(r'(`+).*?\1|(?:[a-zA-Z][a-zA-Z0-9+.-]*://|www\.)[^\s<>]+', re.S)


def normalize_citations(text, source_ids):
    protected = [(match.start(), match.end()) for match in PROTECTED_CITATION_TEXT.finditer(text)]
    invalid = False

    def replace(match):
        nonlocal invalid
        if any(start <= match.start() < end for start, end in protected):
            return match.group()
        ref = match.group(1)
        if ref not in source_ids:
            invalid = True
            return match.group()
        return '[' + ref + ']'

    return ASCII_CITATION.sub(replace, text), invalid


def provenance_engine(engine, source, records):
    """Reject new URLs or web references while retaining provider usage details."""
    source_ids = {record['id'] for record in records}
    # Permit Markdown punctuation surrounding existing source URLs, but never
    # normalization that could change a URL's query, fragment, or encoded path.
    original_urls = source_urls(source)
    record_urls = {record['id']: source_urls(record['title']) for record in records}

    def checked(**kwargs):
        result = engine(**kwargs)
        summary = result.get('summary')
        # Provider failures often carry an empty summary; preserve their original
        # error and usage rather than relabeling them.
        if result.get('status') not in {'summary', 'cache_hit'} or not summary:
            return result
        if isinstance(summary, str):
            summary, malformed_citation = normalize_citations(summary, source_ids)
            result = {**result, 'summary': summary}
            for key in ('facts', 'unknowns'):
                if isinstance(result.get(key), list):
                    items = []
                    for item in result[key]:
                        normalized, invalid = normalize_citations(item, source_ids) if isinstance(item, str) else (item, False)
                        malformed_citation = malformed_citation or invalid
                        items.append(normalized)
                    result[key] = items
            urls = source_urls(summary)
            lines = [line for line in summary.splitlines() if line.strip()]
            facts = [line for line in lines if not line.lstrip().startswith('- Unknown:')]
            uncited_fact = not facts or any(not SOURCE_ID.search(line) for line in facts)
            mismatched_url = any(not source_urls(line).issubset(set().union(
                *(record_urls.get(ref, set()) for ref in SOURCE_ID.findall(line))))
                for line in facts)
            if (not urls.issubset(original_urls) or not set(SOURCE_ID.findall(summary)).issubset(source_ids)
                    or not lines or uncited_fact or mismatched_url or malformed_citation):
                diagnostic = rejection_diagnostic(result, source, kwargs.get('state_dir', hook.STATE / 'engine'))
                rejected = {**result, 'status': 'error', 'error': 'unsupported_web_provenance', 'summary': ''}
                if diagnostic:
                    rejected['diagnostic_file'] = diagnostic
                return rejected
        return result
    return checked


def transform(event, cfg=None, engine=None):
    original = event.get('tool_response') if isinstance(event, dict) else None
    passthrough = {'compressed': False, 'result': original}
    try:
        if not eligible(event):
            return passthrough
        text = plain_text(original)
        if text is None:
            return passthrough
        records = source_records(text)
        suffix = source_metadata(text, records)
        if suffix is None:
            return passthrough
        if engine is None:
            from summary_engine import summarize
            engine = summarize
        request = {**event, 'hook_event_name': 'PostToolUse', 'tool_name': 'web.run'}
        # All source guards, thresholds, locking, archival, and accounting run in
        # the shared hook; metadata is included before its final token decision.
        replacement = hook.process(request, cfg=cfg,
                                   engine=provenance_engine(engine, text, records), output_suffix=suffix)
        if replacement:
            return {'compressed': True, 'result': replacement['stopReason']}
    except Exception as exc:
        with contextlib.suppress(Exception):
            hook.append_event({'tool': 'web.run', 'status': 'adapter_failed',
                               'error_type': type(exc).__name__, 'saved_tokens': 0})
    return passthrough


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision-only', action='store_true',
                        help='Omit original result on passthrough; caller retains it.')
    args = parser.parse_args()
    hook.stop_on_signals()
    try:
        event = json.load(sys.stdin)
    except (ValueError, UnicodeError):
        # There is no reliable original result when the transport JSON is invalid.
        print(json.dumps({'compressed': False, 'error': 'invalid_envelope'}))
        return 0
    decision = transform(event)
    if args.decision_only and not decision['compressed']:
        decision = {'compressed': False}
    print(json.dumps(decision, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
