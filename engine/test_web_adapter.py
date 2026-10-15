import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import hook
import web_adapter


class WebAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)
        self.state_patch = patch.object(hook, 'STATE', self.state)
        self.state_patch.start()
        self.cfg = dict(hook.DEFAULTS, minimum_tokens=1000)
        self.source = ('Official guide (https://example.org/docs?q=a%20b&lang=en#intro)\n'
                       '【turn0search0】 [wordlim 150] Published: Sep 13, 2026\n'
                       + 'The documented mechanism handles repeated input safely.\n' * 280)
        self.engine = Mock(return_value={'status': 'summary', 'summary': 'The documented mechanism handles repeated input safely. 【turn0search0】',
                                        'usage': {'input_tokens': 2300, 'output_tokens': 20, 'total_tokens': 2320},
                                        'model': 'test-helper', 'elapsed_ms': 1})

    def tearDown(self):
        self.state_patch.stop()
        self.tmp.cleanup()

    def event(self, response=None, request=None):
        return {'tool_name': 'web.run', 'tool_input': request or {'search_query': [{'q': 'mechanism docs'}]},
                'tool_response': self.source if response is None else response,
                'session_id': 'web-test', 'tool_use_id': 'web-call'}

    def transform(self, event):
        return web_adapter.transform(event, cfg=self.cfg, engine=self.engine)

    def ledger(self):
        path = self.state / 'events.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_metadata_exact_and_final_tokens_accounted(self):
        result = self.transform(self.event())
        self.assertTrue(result['compressed'])
        for line in self.source.splitlines()[:2]:
            self.assertIn(line, result['result'])
        row = self.ledger()[-1]
        self.assertEqual(row['tool'], 'web.run')
        self.assertEqual(row['after_tokens'], hook.tokens(result['result']))
        self.assertEqual(row['saved_tokens'], hook.tokens(self.source) - hook.tokens(result['result']))
        self.assertEqual(Path(row['raw_file']).read_text(), self.source)
        self.assertEqual(self.engine.call_args.kwargs['text'], self.source)

    def test_supported_text_envelopes(self):
        for response in [{'output': self.source}, {'content': [{'type': 'text', 'text': self.source}]},
                         {'content': [{'type': 'text', 'text': self.source}], 'isError': False}]:
            with self.subTest(shape=response.keys()):
                self.assertTrue(self.transform(self.event(response))['compressed'])

    def test_rich_structured_unknown_shapes_unchanged(self):
        text_block = {'type': 'text', 'text': self.source}
        responses = [
            {'content': [text_block, {'type': 'image', 'data': 'abc', 'mimeType': 'image/png'}]},
            {'content': [text_block], 'structuredContent': {'results': ['keep']}},
            {'content': [text_block], 'structuredContent': None},
            {'content': [text_block], '_meta': {'source': 'keep'}},
            {'content': [{**text_block, 'annotations': {'priority': 1}}]},
            {'content': [text_block, 42]}, {'content': 'text'}, {'results': self.source},
            {'output': self.source, 'unknown': 7}, [text_block], json.dumps({'output': self.source}),
        ]
        for response in responses:
            with self.subTest(response_type=type(response).__name__):
                result = self.transform(self.event(response))
                self.assertFalse(result['compressed'])
                self.assertIs(result['result'], response)
        self.engine.assert_not_called()

    def test_bypasses_sensitive_operations_and_exact_requests(self):
        requests = [{'screenshot': [{'ref_id': 'turn0view0', 'pageno': 0}]},
                    {'finance': [{'ticker': 'AMD', 'type': 'equity'}]},
                    {'weather': [{'location': 'Brussels'}]}, {'sports': [{'fn': 'standings', 'league': 'nba'}]},
                    {'time': [{'utc_offset': '+02:00'}]}, {'image_query': [{'q': 'mountains'}]},
                    {'open': [{'ref_id': 'https://example.org'}], 'weather': []},
                    {'search_query': [{'q': 'verbatim statement'}]},
                    {'search_query': [{'q': 'exact wording in docs'}]},
                    {'search_query': [{'q': 'context-saver:raw docs'}]},
                    {'open': [{'ref_id': 'https://example.org'}], 'unknown': 1}]
        for request in requests:
            self.assertFalse(self.transform(self.event(request=request))['compressed'])
        self.assertFalse(self.transform({**self.event(), 'raw': True})['compressed'])
        with patch.dict(os.environ, {'CONTEXT_SAVER_RAW': '1'}):
            self.assertFalse(self.transform(self.event())['compressed'])
        self.engine.assert_not_called()

    def test_small_missing_provenance_and_sensitive_bypass(self):
        for source in ['small', 'text without sources\n' * 1500,
                       self.source + '\napi_key="secret123456789"']:
            self.assertFalse(self.transform(self.event(source))['compressed'])
        self.engine.assert_not_called()

    def test_oversized_web_source_uses_local_preview_with_exact_source_metadata(self):
        source = self.source + 'oversize\n' * 10000
        out = self.transform(self.event(source))
        self.assertTrue(out['compressed'])
        self.assertIn('tokens saved by local compaction', out['result'])
        self.assertIn('ARCHIVE PREVIEW', out['result'])
        self.assertTrue(out['result'].endswith(web_adapter.source_metadata(source)))
        self.assertLessEqual(hook.tokens(out['result']), self.cfg['maximum_replacement_tokens'])
        self.engine.assert_not_called()

    def test_disabled_and_recursion_guard(self):
        with patch.dict(os.environ, {'CONTEXT_SAVER_ACTIVE': '1'}):
            self.assertFalse(self.transform(self.event())['compressed'])
        self.assertFalse(web_adapter.transform(self.event(), cfg={**self.cfg, 'enabled': False},
                                              engine=self.engine)['compressed'])
        self.engine.assert_not_called()

    def test_failure_preserves_original_and_records_no_savings(self):
        self.engine.return_value = {'status': 'error', 'error': 'timeout', 'usage': {'total_tokens': 50}}
        result = self.transform(self.event())
        self.assertEqual(result, {'compressed': False, 'result': self.source})
        self.assertEqual(self.ledger()[-1]['saved_tokens'], 0)
        self.assertEqual(self.ledger()[-1]['status'], 'helper_failed')
        self.engine.side_effect = TimeoutError('engine timeout')
        self.assertFalse(self.transform(self.event())['compressed'])
        self.assertEqual(self.ledger()[-1]['status'], 'adapter_failed')

    def test_empty_provider_failure_preserves_original_error_and_usage(self):
        original = {'status': 'error', 'summary': '', 'error': 'model_timeout', 'elapsed_ms': 60342,
                    'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
                    'usage_incomplete': True}
        self.engine.return_value = original
        checked = web_adapter.provenance_engine(self.engine, self.source, web_adapter.source_records(self.source))
        self.assertIs(checked(state_dir=self.state / 'engine'), original)
        self.assertFalse(self.transform(self.event())['compressed'])
        row = self.ledger()[-1]
        self.assertEqual(row['error'], 'model_timeout')
        self.assertEqual(row['latency_ms'], 60342)
        self.assertEqual(row['helper_usage'], original['usage'])
        self.assertEqual(row['saved_tokens'], 0)
        for result in [{**original, 'summary': 'Provider failure text'},
                       {'status': 'summary', 'summary': '', 'error': 'empty_response'}]:
            self.engine.return_value = result
            self.assertIs(checked(state_dir=self.state / 'engine'), result)

    def test_rejected_provenance_debug_is_private_bounded_and_opt_in(self):
        rejected_text = '- Unsupported claim without citation. ' * 1000
        self.engine.return_value = {'status': 'summary', 'summary': rejected_text,
                                    'usage': {'total_tokens': 37}}
        checked = web_adapter.provenance_engine(self.engine, self.source, web_adapter.source_records(self.source))
        with patch.dict(os.environ, {'CONTEXT_SAVER_DEBUG_INVALID_RESPONSE': '0'}):
            self.assertNotIn('diagnostic_file', checked(state_dir=self.state / 'engine'))
            self.assertFalse((self.state / 'engine' / 'debug').exists())
        with patch.dict(os.environ, {'CONTEXT_SAVER_DEBUG_INVALID_RESPONSE': '1'}):
            result = checked(state_dir=self.state / 'engine')
        self.assertEqual(result['error'], 'unsupported_web_provenance')
        self.assertEqual(result['usage']['total_tokens'], 37)
        diagnostic = Path(result['diagnostic_file'])
        self.assertEqual(diagnostic.stat().st_mode & 0o777, 0o600)
        self.assertEqual(diagnostic.parent.stat().st_mode & 0o777, 0o700)
        saved = diagnostic.read_text()
        self.assertLessEqual(len(saved), 16000)
        self.assertIn(rejected_text[:12000], saved)
        self.assertEqual(result['summary'], '')

    def test_provenance_debug_never_saves_sensitive_source_or_output(self):
        summary = 'Uncited claim api_key="secret123456789"'
        with patch.dict(os.environ, {'CONTEXT_SAVER_DEBUG_INVALID_RESPONSE': '1'}):
            self.assertIsNone(web_adapter.rejection_diagnostic({'summary': summary}, self.source, self.state))
            self.assertIsNone(web_adapter.rejection_diagnostic({'summary': 'Uncited claim'}, self.source + summary, self.state))
        self.assertFalse((self.state / 'debug').exists())

    def test_unknown_urls_and_source_ids_rejected_with_usage(self):
        for summary in ['See https://invented.example.org/page', 'A fact 【turn999search999】']:
            self.engine.return_value = {'status': 'summary', 'summary': summary, 'usage': {'total_tokens': 42}}
            self.assertFalse(self.transform(self.event())['compressed'])
            row = self.ledger()[-1]
            self.assertEqual(row['status'], 'helper_failed')
            self.assertEqual(row['error'], 'unsupported_web_provenance')
            self.assertEqual(row['helper_usage']['total_tokens'], 42)
            self.assertEqual(row['saved_tokens'], 0)

    def test_exact_existing_urls_allowed_but_modified_queries_rejected(self):
        url = 'https://example.org/docs?q=a%20b&lang=en#intro'
        for summary in [f'See {url} 【turn0search0】', f'See [guide]({url}) 【turn0search0】']:
            self.engine.return_value = {'status': 'summary', 'summary': summary}
            self.assertTrue(self.transform(self.event())['compressed'])
        for summary in [f'See {url}? 【turn0search0】', f'See {url}! 【turn0search0】',
                        'See https://example.org/docs?q=a+b&lang=en#intro 【turn0search0】',
                        'See https://example.org/docs?q=a+b&lang=en#intro. 【turn0search0】']:
            self.engine.return_value = {'status': 'summary', 'summary': summary}
            self.assertFalse(self.transform(self.event())['compressed'])
            self.assertEqual(self.ledger()[-1]['error'], 'unsupported_web_provenance')

    def test_sentence_punctuation_after_existing_url_is_not_a_new_url(self):
        url = 'https://example.org/docs?q=a%20b&lang=en#intro'
        for summary in [f'- Spec at {url}. Details follow 【turn0search0】', f'- See {url}, then 【turn0search0】',
                        f'- Source: {url}: guide 【turn0search0】', f'- Guide ({url}). 【turn0search0】',
                        f'- Guide ({url}.) 【turn0search0】']:
            self.engine.return_value = {'status': 'summary', 'summary': summary}
            self.assertTrue(self.transform(self.event())['compressed'], summary)
        dotted = self.source.replace('#intro)', '#intro.)')
        self.engine.return_value = {'status': 'summary', 'summary': f'- See {url} 【turn0search0】'}
        self.assertTrue(self.transform(self.event(dotted))['compressed'])
        self.assertEqual(web_adapter.source_urls(f'({url}).'), {url})

    def test_large_metadata_declines_compression_before_ledger_claim(self):
        source = ''.join(f'Page {i} (https://example.org/page/{i})\n【turn0search{i}】 [wordlim {100+i}]\n'
                         for i in range(1, 121)) + self.source
        result = self.transform(self.event(source))
        self.assertEqual(result, {'compressed': False, 'result': source})
        self.assertEqual(self.ledger()[-1]['status'], 'insufficient_saving')
        self.assertEqual(self.ledger()[-1]['saved_tokens'], 0)
        self.assertFalse(any(row['status'] == 'compressed' for row in self.ledger()))

    def test_multiple_sources_word_limits_and_default(self):
        source = (self.source + '\nSecond source (https://example.org/second)\n'
                  '【turn1view0】 [wordlim 25] Content type: text/html\nSecond page.\n'
                  'Third source (https://example.org/third)\n【turn2fetch0】 Content type: text/html\n')
        result = self.transform(self.event(source))
        self.assertTrue(result['compressed'])
        for fragment in ['[wordlim 150]', '[wordlim 25]', '【turn1view0】', '【turn2fetch0】',
                         'https://example.org/second', 'https://example.org/third', 'default 200-word limit']:
            self.assertIn(fragment, result['result'])

    def test_native_search_snippet_not_copied_into_metadata(self):
        title = 'Session management (https://ai.google.dev/gemini-api/docs/live-api/session-management)'
        marker = '\ue200cite\ue202turn4search0\ue201'
        body = 'This guide explains context limits and connection handling. ' * 400
        source = f'{title}\n{marker} [wordlim: 200] Crawled: last week; {body}'
        self.engine.return_value = {'status': 'summary', 'summary': '- Search snippet describes context limits. turn4search0'}
        result = self.transform(self.event(source))
        self.assertTrue(result['compressed'])
        suffix = web_adapter.source_metadata(source)
        self.assertIn(title, suffix)
        self.assertIn(marker + ' [wordlim: 200] Crawled: last week;', suffix)
        self.assertNotIn('This guide explains', suffix)
        self.assertLess(hook.tokens(suffix), 200)
        self.assertEqual(self.ledger()[-1]['after_tokens'], hook.tokens(result['result']))

    def test_all_facts_require_mapped_source_ids(self):
        for summary in ['- Uncited fact.', '- Cited fact. turn0search0\n- Uncited second fact.',
                        '- Wrong format source. [turn0Search0]', '- Unknown: No facts.']:
            self.engine.return_value = {'status': 'summary', 'summary': summary}
            self.assertFalse(self.transform(self.event())['compressed'])
            self.assertEqual(self.ledger()[-1]['error'], 'unsupported_web_provenance')

    def test_source_url_must_match_cited_identity(self):
        source = self.source + '\nSecond source (https://example.org/second)\n【turn1view0】 [wordlim: 25]\nSecond page.'
        self.engine.return_value = {'status': 'summary', 'summary': '- See https://example.org/second turn0search0'}
        self.assertFalse(self.transform(self.event(source))['compressed'])
        self.engine.return_value = {'status': 'summary', 'summary': '- See https://example.org/second turn1view0'}
        self.assertTrue(self.transform(self.event(source))['compressed'])

    def test_unmapped_source_and_malformed_word_limit_remain_raw(self):
        for source in [self.source.replace('[wordlim 150]', '[wordlim: unknown]'),
                       self.source.replace('Official guide (https://example.org/docs?q=a%20b&lang=en#intro)', 'No URL')]:
            self.assertEqual(self.transform(self.event(source)), {'compressed': False, 'result': source})
        self.engine.assert_not_called()

    def test_native_ascii_citations_normalize_known_ids_and_payload_consistently(self):
        source = self.source + '\nSecond source (https://example.org/second)\n\ue200cite\ue202turn5view1\ue201 [wordlim: 200]\nSecond page.'
        fact = 'Under citeturn0search0 (L90-L96), the API expects `FunctionResponse` objects.'
        unknown = 'The full content of citeturn5view1 is truncated (citeturn5view1).'
        original = {'status': 'cache_hit', 'summary': '- ' + fact + '\n- Unknown: ' + unknown,
                    'facts': [fact], 'unknowns': [unknown], 'usage': {'input_tokens': 8000, 'output_tokens': 200},
                    'elapsed_ms': 4679, 'cache_hit': True, 'provider': 'claude_cli', 'model': 'sonnet'}
        self.engine.return_value = original
        checked = web_adapter.provenance_engine(self.engine, source, web_adapter.source_records(source))
        result = checked(state_dir=self.state / 'engine')
        self.assertEqual(result['status'], 'cache_hit')
        self.assertEqual(result['facts'], [fact.replace('citeturn0search0', '[turn0search0]')])
        self.assertEqual(result['unknowns'], [unknown.replace('citeturn5view1', '[turn5view1]')])
        self.assertEqual(result['summary'], '- ' + result['facts'][0] + '\n- Unknown: ' + result['unknowns'][0])
        self.assertEqual(result['usage'], original['usage'])
        self.assertEqual(result['elapsed_ms'], 4679)
        self.assertIn('citeturn0search0', original['summary'])
        self.assertTrue(self.transform(self.event(source))['compressed'])

    def test_ascii_citation_unknown_and_arbitrary_prefixes_rejected(self):
        for summary in ['- Under citeturn999view0, an unsupported claim.',
                        '- Under turn0search0 and citeturn999view0, an unsupported claim.',
                        '- Under turn0search0, see citeturn999view0.',
                        '- Under foociteturn0search0, an unsupported claim.',
                        '- Under preciteturn0search0suffix, an unsupported claim.']:
            self.engine.return_value = {'status': 'summary', 'summary': summary}
            self.assertFalse(self.transform(self.event())['compressed'])
            self.assertEqual(self.ledger()[-1]['error'], 'unsupported_web_provenance')

    def test_ascii_citation_normalization_preserves_urls_code_and_word_boundaries(self):
        protected = ['https://example.org/citeturn0search0', 'https://example.org/?ref=citeturn0search0',
                     'wss://example.org/#citeturn0search0', 'www.example.org/citeturn0search0',
                     '`citeturn0search0`', '```text\nciteturn0search0\n```',
                     'prefixciteturn0search0', 'citeturn0search0suffix', '_citeturn0search0',
                     'path/citeturn0search0', 'key=citeturn0search0', 'citeturn0search0.json']
        for text in protected:
            self.assertEqual(web_adapter.normalize_citations(text, {'turn0search0'}), (text, False))
        mixed = 'Under citeturn0search0, `citeturn0search0` remains literal; see https://example.org/citeturn0search0'
        normalized, invalid = web_adapter.normalize_citations(mixed, {'turn0search0'})
        self.assertFalse(invalid)
        self.assertEqual(normalized, mixed.replace('Under citeturn0search0,', 'Under [turn0search0],'))

    def test_cli_payload_metacharacters_are_exact_data(self):
        payload = 'literal `touch /tmp/never-web-hook` $(printf nope) $HOME \\n\n"quote" λ'
        event = self.event(payload)
        completed = subprocess.run([sys.executable, str(hook.ROOT / 'web_adapter.py')],
                                   input=json.dumps(event), capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(completed.stdout), {'compressed': False, 'result': payload})
        completed = subprocess.run([sys.executable, str(hook.ROOT / 'web_adapter.py'), '--decision-only'],
                                   input=json.dumps(event), capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(completed.stdout), {'compressed': False})

    def test_cli_invalid_transport_has_no_claim(self):
        completed = subprocess.run([sys.executable, str(hook.ROOT / 'web_adapter.py'), '--decision-only'],
                                   input='{broken', capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(completed.stdout), {'compressed': False, 'error': 'invalid_envelope'})


if __name__ == '__main__':
    unittest.main()
