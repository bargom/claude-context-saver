"""Synthetic docs navigation fixtures; no provider calls or production ledger writes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import deterministic
import hook
import mcp_proxy
from test_mcp_proxy import request, response


def docs_source(count=3):
    hits = []
    for i in range(count):
        content = f'Guide {i} full text. ' * 900
        hits.append({'url': f'https://developers.openai.com/codex/guide-{i}#section',
                     'url_without_anchor': f'https://developers.openai.com/codex/guide-{i}',
                     'anchor': 'section', 'content': content, 'type': 'content',
                     'hierarchy': {'lvl0': 'Docs', 'lvl1': f'Guide {i}', 'lvl2': 'Section',
                                   'lvl3': None, 'lvl4': None, 'lvl5': None, 'lvl6': None},
                     'objectID': f'guide-{i}-section',
                     '_snippetResult': {'content': {'value': f'Relevant match {i}. ' * 40,
                                                    'matchLevel': 'full'}},
                     '_highlightResult': {'content': {'value': content, 'matchLevel': 'full'},
                                          'hierarchy': {'lvl1': {'value': f'Guide {i}'}}}})
    return {'hits': hits, 'nbHits': count + 40, 'page': 2, 'nextCursor': 'opaque-page-3-cursor'}


class DocsDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        state_patch = patch.object(hook, 'STATE', self.state)
        state_patch.start()
        self.addCleanup(state_patch.stop)
        self.cfg = dict(hook.DEFAULTS)
        self.engine = Mock(side_effect=AssertionError('docs projection must not call a provider'))
        self.proxy = mcp_proxy.Compressor('openaiDeveloperDocs',
                                        lambda event: hook.process(event, self.cfg, self.engine))

    async def transform(self, source, arguments=None, **extra):
        req = request(name='search_openai_docs', arguments=arguments or {'query': 'Codex setup'})
        original = response(content=[{'type': 'text', 'text': json.dumps(source, ensure_ascii=False)}], **extra)
        self.proxy.observe(req)
        result = await self.proxy.transform(original)
        self.engine.assert_not_called()
        return original, result

    def ledger(self):
        return json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])

    async def test_large_duplicate_content_reduced_with_exact_identity_paging_and_zero_usage(self):
        source = docs_source()
        original, result = await self.transform(source)
        self.assertNotEqual(result, original)
        text = result['result']['content'][0]['text']
        index = json.loads(next(line for line in text.splitlines() if line.startswith('{"hits":')))
        self.assertEqual({k: v for k, v in index.items() if k != 'hits'},
                         {k: v for k, v in source.items() if k != 'hits'})
        self.assertEqual(len(index['hits']), 3)
        for src, compact in zip(source['hits'], index['hits']):
            self.assertEqual(compact['objectID'], src['objectID'])
            self.assertEqual(compact['url'], src['url'])
            self.assertEqual(compact['title'], 'Docs > ' + src['hierarchy']['lvl1'] + ' > Section')
            self.assertEqual(compact['snippet'], src['_snippetResult']['content']['value'][:320])
        row = self.ledger()
        self.assertEqual(row['compaction_kind'], 'docs_search_index')
        self.assertEqual(row['coverage']['remaining'], 0)
        self.assertIn('_highlightResult', row['coverage']['fields_omitted'])
        self.assertGreater(row['coverage']['snippet_characters_omitted'], 0)
        self.assertTrue(row['deterministic'])
        self.assertEqual(row['helper_usage']['total_tokens'], 0)
        self.assertEqual(row['after_tokens'], hook.tokens(text))
        self.assertEqual(row['saved_tokens'], hook.tokens(original['result']['content'][0]['text']) - hook.tokens(text))
        self.assertLessEqual(hook.tokens(text), 1800)
        # The policy bounds delivered text; measure the complete transport too.
        self.assertLessEqual(hook.tokens(mcp_proxy.encode(result).decode()), 1800)
        archive = Path(row['raw_file'])
        self.assertEqual(json.loads(archive.read_text()), source)
        self.assertEqual(archive.stat().st_mode & 0o777, 0o600)
        self.assertIn(str(archive), text)

    async def test_explicit_large_page_is_bounded_with_honest_omissions_and_cursor(self):
        source = docs_source(120)
        for hit in source['hits']:
            hit['content'] = 'Source content. ' * 10
            hit['_highlightResult']['content']['value'] = hit['content']
        original, result = await self.transform(source, {'query': 'Codex', 'limit': 120, 'cursor': 'previous'})
        self.assertNotEqual(result, original)
        text = result['result']['content'][0]['text']
        coverage = self.ledger()['coverage']
        self.assertGreater(coverage['remaining'], 0)
        self.assertEqual(coverage['shown'] + coverage['remaining'], 120)
        self.assertIn('remaining hits are archive-only', text)
        self.assertIn('opaque-page-3-cursor', text)
        self.assertLessEqual(hook.tokens(text), 1800)

    async def test_known_shape_without_snippet_uses_content_and_nullable_cursor(self):
        source = docs_source(1)
        hit = source['hits'][0]
        del hit['_snippetResult']
        source['nextCursor'] = None
        _, result = await self.transform(source)
        text = result['result']['content'][0]['text']
        index = json.loads(next(line for line in text.splitlines() if line.startswith('{"hits":')))
        self.assertEqual(index['hits'][0]['snippet'], hit['content'][:320])
        self.assertIsNone(index['nextCursor'])

    async def test_unknown_shapes_errors_sensitive_raw_and_structured_are_untouched(self):
        mutations = [lambda s: s.update(unknown_contract={'must': 'remain exact'}),
                     lambda s: s['hits'][0].update(unknown_contract='exact'),
                     lambda s: s.update(nbHits=True),
                     lambda s: s.update(nextCursor={'opaque': 'object'}),
                     lambda s: s['hits'][0].update(hierarchy={'other': 'schema'})]
        for mutate in mutations:
            source = docs_source()
            mutate(source)
            original, result = await self.transform(source)
            self.assertIs(result, original)
        for arguments in ({'query': 'Codex context-saver:raw'}, {'query': 'api_key=secret-value-12345'}):
            original, result = await self.transform(docs_source(), arguments)
            self.assertIs(result, original)
        source = docs_source()
        source['hits'][0]['content'] += ' api_key=secret-value-12345'
        original, result = await self.transform(source)
        self.assertIs(result, original)
        for extra in ({'structuredContent': {}}, {'structuredContent': None}, {'isError': True}):
            original, result = await self.transform(docs_source(), **extra)
            self.assertIs(result, original)

    async def test_small_rich_annotated_and_duplicate_json_remain_exact(self):
        for blocks in ([{'type': 'text', 'text': 'small'}],
                       [{'type': 'text', 'text': json.dumps(docs_source()), 'annotations': {'priority': 1}}],
                       [{'type': 'text', 'text': json.dumps(docs_source())}, {'type': 'image', 'data': 'opaque'}],
                       [{'type': 'text', 'text': json.dumps(docs_source()).replace('"page": 2', '"page": 2, "page": 3')} ]):
            original = response(content=blocks)
            self.proxy.observe(request(name='search_openai_docs'))
            self.assertIs(await self.proxy.transform(original), original)
        self.engine.assert_not_called()

    async def test_small_known_docs_stays_exact(self):
        source = docs_source(1)
        source['hits'][0]['content'] = 'Short content'
        source['hits'][0].pop('_highlightResult')
        source['hits'][0].pop('_snippetResult')
        original, result = await self.transform(source)
        self.assertIs(result, original)

    async def test_busy_helper_slots_still_filter_locally_without_real_provider(self):
        self.proxy.active = mcp_proxy.MAX_SUMMARIES
        with patch('hook.config', return_value=self.cfg), patch('summary_engine.summarize') as provider:
            original, result = await self.transform(docs_source())
        self.assertNotEqual(result, original)
        self.assertEqual(self.proxy.active, mcp_proxy.MAX_SUMMARIES)
        self.assertEqual(self.ledger()['helper_usage']['total_tokens'], 0)
        provider.assert_not_called()

    async def test_busy_local_path_preserves_disable_raw_sensitive_and_budget_rejections(self):
        self.proxy.active = mcp_proxy.MAX_SUMMARIES
        for cfg in (dict(self.cfg, enabled=False), dict(self.cfg, deterministic_first=False),
                    dict(self.cfg, maximum_replacement_tokens=200)):
            with patch('hook.config', return_value=cfg), patch('summary_engine.summarize') as provider:
                original, result = await self.transform(docs_source())
                self.assertIs(result, original)
                provider.assert_not_called()
        with patch('hook.config', return_value=self.cfg), patch('summary_engine.summarize') as provider:
            original, result = await self.transform(docs_source(), {'query': 'context-saver:raw'})
            self.assertIs(result, original)
            source = docs_source()
            source['hits'][0]['content'] += ' api_key=secret-value-12345'
            original, result = await self.transform(source)
            self.assertIs(result, original)
            provider.assert_not_called()

    async def test_omitted_limit_default_preserves_explicit_limits_cursor_and_bypasses(self):
        original = request(name='search_openai_docs', arguments={'query': 'Codex', 'cursor': 'exact-cursor'})
        snapshot = copy.deepcopy(original)
        prepared = self.proxy.prepare_request(original)
        self.assertEqual(prepared['params']['arguments'], {'query': 'Codex', 'cursor': 'exact-cursor', 'limit': 3})
        self.assertEqual(original, snapshot)
        for limit in (0, 1, 80, None):
            explicit = request(name='search_openai_docs', arguments={'query': 'Codex', 'limit': limit, 'cursor': 'c'})
            self.assertIs(self.proxy.prepare_request(explicit), explicit)
        for name, arguments in (('fetch_openai_doc', {'url': 'https://developers.openai.com/codex/'}),
                                ('search_openai_docs', {'query': 'context-saver:raw'})):
            req = request(name=name, arguments=arguments)
            self.assertIs(self.proxy.prepare_request(req), req)
        with patch('hook.config', return_value={'enabled': False}):
            self.assertIs(self.proxy.prepare_request(original), original)
        with patch.dict('os.environ', {'CONTEXT_SAVER_ACTIVE': '1'}):
            self.assertIs(self.proxy.prepare_request(original), original)


if __name__ == '__main__':
    unittest.main()
