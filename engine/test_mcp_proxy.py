"""Protocol tests use a stub summarizer and never call a paid model."""
import asyncio
import contextlib
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import hook
import mcp_proxy
from test_hook import fake_claude, is_gone, wait_for_pid

ROOT = Path(__file__).resolve().parent


def private_env(folder, base=None):
    """The environment of a proxy subprocess that never reads the developer's
    own config.json, state or login files."""
    folder = Path(folder)
    return {**(base if base is not None else os.environ), 'HOME': str(folder),
            'XDG_CONFIG_HOME': str(folder / '.config'), 'CONTEXT_SAVER_CONFIG': str(folder / 'config.json'),
            'CONTEXT_SAVER_STATE': str(folder / 'state')}


def request(identifier=1, name='search_graph', arguments=None):
    return {'jsonrpc': '2.0', 'id': identifier, 'method': 'tools/call',
            'params': {'name': name, 'arguments': arguments or {'pattern': 'example'}}}


def response(identifier=1, content=None, **extra):
    return {'jsonrpc': '2.0', 'id': identifier, 'result': {
        'content': content if content is not None else [{'type': 'text', 'text': 'source ' * 1000}], **extra}}


class CompressorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events = []
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        # Never read the developer's own config.json or write to the real state.
        self.enterContext(patch.object(hook, 'CONFIG_PATH', folder / 'config.json'))
        self.enterContext(patch.object(hook, 'STATE', folder / 'state'))

        def process(event):
            self.events.append(event)
            text = event['tool_response']['content'][0]['text']
            return {'continue': False, 'stopReason': 'Summary with exact path'} if len(text) > 100 else None

        self.proxy = mcp_proxy.Compressor('codebase-memory-mcp', process, '/tmp/test', 'session-test')

    async def test_implicit_graph_page_limit_preserves_explicit_request_contracts(self):
        original = request(arguments={'name_pattern': 'transcri', 'offset': 40})
        prepared = self.proxy.prepare_request(original)
        self.assertEqual(prepared['params']['arguments'], {'name_pattern': 'transcri', 'offset': 40, 'limit': 20})
        self.assertNotIn('limit', original['params']['arguments'])
        for limit in [0, 5, 200]:
            explicit = request(arguments={'limit': limit, 'offset': 10})
            self.assertIs(self.proxy.prepare_request(explicit), explicit)
        unrelated = request(name='get_code_snippet')
        self.assertIs(self.proxy.prepare_request(unrelated), unrelated)
        other = mcp_proxy.Compressor('other-server', lambda event: None)
        self.assertIs(other.prepare_request(original), original)
        with patch('hook.config', return_value={'enabled': False}):
            self.assertIs(self.proxy.prepare_request(original), original)
        with patch('hook.config', side_effect=ValueError('invalid configuration')):
            self.assertIs(self.proxy.prepare_request(original), original)

    async def test_page_limit_is_left_alone_when_the_category_is_off(self):
        import hook
        original = request(arguments={'name_pattern': 'transcri'})
        with patch('hook.config', return_value={**hook.DEFAULTS, 'categories': ['logs']}):
            self.assertIs(self.proxy.prepare_request(original), original)

    async def test_large_text_replacement_and_envelope(self):
        req = request('request-7')
        original = response('request-7', isError=True, _meta={'progress': 'preserved'})
        before = copy.deepcopy(original)
        self.proxy.observe(req)
        result = await self.proxy.transform(original)
        self.assertEqual(result['id'], 'request-7')
        self.assertEqual(result['result']['content'], [{'type': 'text', 'text': 'Summary with exact path'}])
        self.assertTrue(result['result']['isError'])
        self.assertEqual(result['result']['_meta'], {'progress': 'preserved'})
        self.assertEqual(original, before)
        event = self.events[0]
        self.assertEqual(event['tool_input'], req['params']['arguments'])
        self.assertEqual(event['tool_response'], original['result'])
        self.assertEqual(event['tool_name'], 'mcp__codebase-memory-mcp__search_graph')
        self.assertEqual(event['tool_use_id'], 'request-7')
        self.assertEqual(event['session_id'], 'session-test')
        self.assertEqual(event['cwd'], '/tmp/test')

    async def test_small_result_identity(self):
        self.proxy.observe(request())
        original = response(content=[{'type': 'text', 'text': 'small'}])
        self.assertIs(await self.proxy.transform(original), original)

    async def test_preserve_structured_and_non_text_content(self):
        variants = [response(structuredContent={'answer': 1}), response(structuredContent=None),
                    response(content=[]), response(content=[None]),
                    response(content=[{'type': 'text', 'text': 123}]),
                    response(content=[{'type': 'image', 'mimeType': 'image/png', 'data': 'opaque'}]),
                    response(content=[{'type': 'resource_link', 'uri': 'file:///artifact', 'name': 'artifact'}]),
                    response(content=[{'type': 'resource', 'resource': {'uri': 'file:///a', 'text': 'exact'}}]),
                    response(content=[{'type': 'text', 'text': 'large ' * 1000, 'annotations': {'audience': ['user']}}])]
        for original in variants:
            self.proxy.observe(request())
            self.assertIs(await self.proxy.transform(original), original)
        self.assertEqual(self.events, [])

    async def test_tools_list_error_notifications_and_ids(self):
        original = {'jsonrpc': '2.0', 'id': 1, 'result': {'tools': [{'name': 'x', 'inputSchema': {'type': 'object'}}]}}
        self.proxy.observe({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})
        self.assertIs(await self.proxy.transform(original), original)
        self.proxy.observe(request())
        error = {'jsonrpc': '2.0', 'id': 1, 'error': {'code': -32603, 'message': 'exact'}}
        self.assertIs(await self.proxy.transform(error), error)
        self.assertEqual(self.proxy.pending, {})
        notification = {'jsonrpc': '2.0', 'method': 'notifications/progress', 'params': {'progress': 10}}
        self.assertIs(await self.proxy.transform(notification), notification)
        self.proxy.observe(request(1))
        unknown = response('1')
        self.assertIs(await self.proxy.transform(unknown), unknown)
        self.assertNotEqual(await self.proxy.transform(response(1)), response(1))
        self.assertEqual(len(self.events), 1)

    async def test_same_id_server_request_does_not_consume_pending(self):
        self.proxy.observe(request(1))
        server_request = {'jsonrpc': '2.0', 'id': 1, 'method': 'ping'}
        self.assertIs(await self.proxy.transform(server_request), server_request)
        self.assertNotEqual(await self.proxy.transform(response(1)), response(1))

    async def test_hook_failure_or_invalid_decision_fails_open(self):
        def failure(event):
            raise RuntimeError('do not print private content')
        for process in [failure, lambda event: {}, lambda event: {'continue': True, 'stopReason': 'x'},
                        lambda event: {'continue': False, 'stopReason': ''}]:
            proxy = mcp_proxy.Compressor('test', process)
            proxy.observe(request())
            original = response()
            self.assertIs(await proxy.transform(original), original)
            self.assertEqual(proxy.active, 0)

    async def test_exact_source_classifier_contract(self):
        from hook import classify
        self.assertIsNone(classify({'tool_name': 'mcp__codebase-memory-mcp__get_code_snippet',
                                    'tool_input': {'qualified_name': 'some.source'}}))

    async def test_stdio_raw_bytes_preserved(self):
        class Sink:
            data = b''
            async def write(self, data):
                self.data += data
        sink = Sink()
        self.proxy.observe(request(1))
        raw = b'{ "jsonrpc" : "2.0", "id": 1, "result": {"content":[{"type":"text","text":"exact"}]}}\n'
        await mcp_proxy.relay_result(raw, self.proxy, sink)
        self.assertEqual(sink.data, raw)

    async def test_concurrency_limit_fails_open(self):
        self.proxy.observe(request())
        self.proxy.active = mcp_proxy.MAX_SUMMARIES
        original = response()
        self.assertIs(await self.proxy.transform(original), original)
        self.assertEqual(self.events, [])

    async def test_text_is_taken_only_from_plain_text_blocks(self):
        self.proxy.observe(request())
        joined = await self.proxy.transform(response(content=[{'type': 'text', 'text': 'first ' * 20},
                                                              {'type': 'text', 'text': 'second ' * 20}]))
        self.assertEqual(joined['result']['content'], [{'type': 'text', 'text': 'Summary with exact path'}])
        self.assertEqual(self.events[0]['tool_response']['content'][1]['text'], 'second ' * 20)
        self.proxy.observe(request())
        mixed = response(content=[{'type': 'text', 'text': 'large ' * 100}, {'type': 'text', 'text': None}])
        self.assertIs(await self.proxy.transform(mixed), mixed)

    async def test_pending_is_bounded_and_reused_ids_clear(self):
        for index in range(mcp_proxy.MAX_PENDING + 10):
            self.proxy.observe(request(index))
        self.assertEqual(len(self.proxy.pending), mcp_proxy.MAX_PENDING)
        self.proxy.observe({'jsonrpc': '2.0', 'id': 20, 'method': 'ping'})
        original = response(20)
        self.assertIs(await self.proxy.transform(original), original)


BOOTSTRAP = '''import asyncio, os, sys, time
import mcp_proxy
mcp_proxy.MAX_LINE_BYTES = int(os.environ.get('TEST_MAX_LINE_BYTES') or mcp_proxy.MAX_LINE_BYTES)
def stub(event):
    text = event['tool_response']['content'][0]['text']
    if text.startswith('slow'):
        time.sleep(.3)
    return {'continue': False, 'stopReason': 'compressed'} if len(text) > 100 else None
proxy = mcp_proxy.Compressor('codebase-memory-mcp', stub)
asyncio.run(mcp_proxy.http_bridge(sys.argv[2], proxy) if sys.argv[1] == 'http' else mcp_proxy.stdio_bridge(sys.argv[2:], proxy))
'''
# The summary runs a real helper process (a fake `claude` on PATH that sleeps).
HELPER_BOOTSTRAP = '''import asyncio, sys
import mcp_proxy, summary_engine
def summarize(event):
    summary_engine.summarize(event['tool_response']['content'][0]['text'], 'search', '', sys.argv[1], 60, 'low')
try:
    asyncio.run(mcp_proxy.serve(mcp_proxy.Compressor('codebase-memory-mcp', summarize), stdio=sys.argv[2:]))
except asyncio.CancelledError:
    pass
'''
CHILD = '''import json, os, sys
print('child diagnostic', file=sys.stderr)
for line in sys.stdin:
    message = json.loads(line)
    if os.environ.get('CHILD_LOG_IDS'):
        print('received', message.get('id'), file=sys.stderr, flush=True)
    if 'id' not in message:
        print(line, end='', flush=True)
        continue
    if message.get('method') == 'tools/call':
        arguments = message['params']['arguments']
        text = arguments.get('text', 'raw') * arguments.get('repeat', 1)
        result = {'content': [{'type': 'text', 'text': text}], 'isError': False}
    else:
        result = {'capabilities': {'tools': {'listChanged': True}}, 'opaque': None}
    print(json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'result': result}), flush=True)
'''


def read_message(process, timeout=10):
    readable, _, _ = select.select([process.stdout], [], [], timeout)
    if not readable:
        raise AssertionError('Proxy response timed out')
    line = process.stdout.readline()
    if not line:
        raise AssertionError('Proxy exited before response')
    return json.loads(line)


class StdioTests(unittest.TestCase):
    def start(self, *args, bootstrap=BOOTSTRAP, env=None):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        env = private_env(folder, env)
        process = subprocess.Popen([sys.executable, '-u', '-c', bootstrap, *args], cwd=ROOT, env=env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.addCleanup(self.stop, process)
        return process

    @staticmethod
    def stop(process):
        if process.poll() is None:
            process.stdin.close()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        for handle in [process.stdin, process.stdout, process.stderr]:
            handle.close()

    def send(self, process, message):
        process.stdin.write(mcp_proxy.encode(message))
        process.stdin.flush()

    def test_stdio_protocol_and_concurrent_responses(self):
        process = self.start('stdio', sys.executable, '-u', '-c', CHILD)
        self.send(process, {'jsonrpc': '2.0', 'id': 'init', 'method': 'initialize'})
        initialized = read_message(process)
        self.assertEqual(initialized['result'], {'capabilities': {'tools': {'listChanged': True}}, 'opaque': None})
        notification = {'jsonrpc': '2.0', 'method': 'notifications/progress', 'params': {'progress': 1}}
        self.send(process, notification)
        self.assertEqual(read_message(process), notification)
        self.send(process, request('slow', arguments={'text': 'slow ' * 1000}))
        self.send(process, request('fast', arguments={'text': 'raw exact'}))
        self.assertEqual(read_message(process), response('fast', content=[{'type': 'text', 'text': 'raw exact'}], isError=False))
        self.assertEqual(read_message(process)['result']['content'][0]['text'], 'compressed')
        process.stdin.close()
        self.assertEqual(process.wait(timeout=8), 0)
        self.assertEqual(process.stderr.read(), b'child diagnostic\n')


    def test_lines_over_the_limit_are_dropped_and_the_bridge_keeps_running(self):
        env = {**os.environ, 'TEST_MAX_LINE_BYTES': '1024', 'CHILD_LOG_IDS': '1'}
        process = self.start('stdio', sys.executable, '-u', '-c', CHILD, env=env)
        self.send(process, request('oversized', arguments={'text': 'y' * 3000}))
        self.send(process, request('after', arguments={'text': 'raw exact'}))
        self.assertEqual(read_message(process)['id'], 'after')
        self.send(process, request('big', arguments={'text': 'x', 'repeat': 3000}))
        self.send(process, request('after2', arguments={'text': 'raw exact'}))
        answered = read_message(process)
        self.assertEqual((answered['id'], answered['result']['content'][0]['text']), ('after2', 'raw exact'))
        process.stdin.close()
        self.assertEqual(process.wait(timeout=8), 0)
        stderr = process.stderr.read().decode()
        self.assertIn('dropped a client line over 1024 bytes', stderr)
        self.assertIn('dropped a server line over 1024 bytes', stderr)
        self.assertNotIn('yyyy', stderr)
        self.assertNotIn('xxxx', stderr)
        # The oversized request never reached the server.
        self.assertEqual([line.split()[1] for line in stderr.splitlines() if line.startswith('received')],
                         ['after', 'big', 'after2'])

    def test_http_mode_without_the_sdk_names_the_setup_command(self):
        probe = ("import sys\n"
                 "sys.modules['mcp'] = None\n"
                 "sys.modules['mcp_types'] = None\n"
                 "import mcp_proxy\n"
                 "sys.argv = ['mcp_proxy', '--server', 'docs', '--http', 'http://127.0.0.1:9/mcp']\n"
                 "sys.exit(mcp_proxy.main())\n")
        folder = self.enterContext(tempfile.TemporaryDirectory())
        result = subprocess.run([sys.executable, '-c', probe], cwd=ROOT, env=private_env(folder),
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 3)
        launcher = ROOT.parent / 'bin' / 'context-saver'
        self.assertIn(f'--http needs the MCP SDK; run: {launcher} setup --mcp', result.stderr)
        self.assertEqual(result.stdout, '')

    def test_stopping_the_proxy_ends_a_running_summary_helper(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        env, pid_file = fake_claude(folder)
        process = self.start(str(folder / 'state'), sys.executable, '-u', '-c', CHILD, bootstrap=HELPER_BOOTSTRAP, env=env)
        self.send(process, request('slow', arguments={'text': 'slow ' * 1000}))
        helper = wait_for_pid(pid_file)
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=8), 0)
        self.assertTrue(is_gone(helper))


class LineReaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_oversized_line_is_dropped_whole(self):
        for chunks in ([b'{"a":1}\n' + b'x' * 100 + b'\n{"b":2}\n{"c":3}'],
                       # The long line arrives in pieces, past the buffer limit.
                       [b'{"a":1}\n', b'x' * 40, b'x' * 40, b'x' * 40 + b'\n{"b":2}\n', b'{"c":3}']):
            reader = asyncio.StreamReader(limit=16)
            stderr = io.StringIO()
            with self.subTest(chunks=len(chunks)), contextlib.redirect_stderr(stderr):
                async def feed():
                    for chunk in chunks:
                        reader.feed_data(chunk)
                        await asyncio.sleep(0)
                    reader.feed_eof()
                feeder = asyncio.create_task(feed())
                lines = []
                while line := await mcp_proxy.read_line(reader, 'server'):
                    lines.append(line)
                await feeder
                self.assertEqual(lines, [b'{"a":1}\n', b'{"b":2}\n', b'{"c":3}'])
                self.assertIn('dropped a server line over', stderr.getvalue())
                self.assertNotIn('xxx', stderr.getvalue())

    def test_protocol_version_needs_a_result_object(self):
        self.assertEqual(mcp_proxy.protocol_version_of({'id': 1, 'result': {'protocolVersion': '2025-03-26'}}),
                         '2025-03-26')
        for message in ({'id': 1, 'result': ['protocolVersion']}, {'id': 1, 'result': 'text'},
                        {'id': 1, 'result': None}, {'id': 1}, {'id': 1, 'result': {'protocolVersion': 7}}):
            self.assertIsNone(mcp_proxy.protocol_version_of(message))


@unittest.skipUnless(importlib.util.find_spec('mcp'), 'official MCP SDK not installed')
class HttpTests(StdioTests):
    # Inherit the stdio regression so both transports run with the installed SDK.
    def test_http_session_version_json_sse_and_notifications(self):
        observed = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def respond(self, code, body=b'', content_type='application/json', session=False):
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                if session:
                    self.send_header('Mcp-Session-Id', 'test-session')
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.respond(405)

            def do_DELETE(self):
                self.respond(200)

            def do_POST(self):
                message = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                observed.append((message, dict(self.headers)))
                method = message.get('method')
                if method == 'initialize':
                    result = {'protocolVersion': '2025-03-26', 'capabilities': {'tools': {}},
                              'serverInfo': {'name': 'fake-docs', 'version': '1'}}
                    self.respond(200, mcp_proxy.encode({'jsonrpc': '2.0', 'id': message['id'], 'result': result}), session=True)
                elif 'id' not in message:
                    self.respond(202)
                elif method == 'tools/list':
                    result = {'tools': [{'name': 'search_graph', 'inputSchema': {'type': 'object',
                                            'properties': {'nullable': {'default': None}}}}]}
                    self.respond(200, mcp_proxy.encode({'jsonrpc': '2.0', 'id': message['id'], 'result': result}))
                else:
                    body = b'event: message\ndata: ' + mcp_proxy.encode(response(message['id'])) + b'\n'
                    self.respond(200, body, 'text/event-stream')

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        process = self.start('http', f'http://127.0.0.1:{server.server_port}/mcp')
        self.send(process, {'jsonrpc': '2.0', 'id': 'init', 'method': 'initialize', 'params': {
            'protocolVersion': '2025-03-26', 'capabilities': {}, 'clientInfo': {'name': 'test', 'version': '1'}}})
        self.assertEqual(read_message(process)['result']['serverInfo']['name'], 'fake-docs')
        self.send(process, {'jsonrpc': '2.0', 'method': 'notifications/initialized'})
        self.send(process, {'jsonrpc': '2.0', 'id': 'list', 'method': 'tools/list'})
        tools = read_message(process)
        self.assertEqual(tools['result']['tools'][0]['inputSchema']['properties']['nullable'], {'default': None})
        self.send(process, request('sse'))
        self.assertEqual(read_message(process)['result']['content'], [{'type': 'text', 'text': 'compressed'}])
        for message, headers in observed[1:]:
            lowered = {key.lower(): value for key, value in headers.items()}
            self.assertEqual(lowered.get('mcp-session-id'), 'test-session')
            self.assertEqual(lowered.get('mcp-protocol-version'), '2025-03-26')
        self.assertTrue(any(message.get('method') == 'notifications/initialized' for message, _ in observed))
        process.stdin.close()
        self.assertEqual(process.wait(timeout=8), 0)


if __name__ == '__main__':
    unittest.main()
