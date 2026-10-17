#!/usr/bin/env python3
"""Transparent MCP transport with selective, fail-open text result compression.

HTTP mode uses the official MCP SDK's Streamable HTTP transport, including
session/version headers, SSE, notifications, and transport shutdown.
"""
import argparse
import asyncio
from collections import OrderedDict
import json
import logging
import os
from pathlib import Path
import signal
import sys
import uuid

from common import text_blocks

MAX_PENDING = 4096
MAX_SUMMARIES = 8
MAX_LINE_BYTES = 64 * 1024 * 1024


def request_key(value):
    # JSON-RPC string and numeric IDs must not collide.
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return type(value).__name__, value
    return None


class Compressor:
    def __init__(self, server, process=None, cwd=None, session_id=None):
        if process is None:
            from hook import process
        self.process = process
        self.server = server
        self.cwd = cwd or os.getcwd()
        self.session_id = session_id or os.environ.get('CODEX_THREAD_ID') or str(uuid.uuid4())
        self.pending = OrderedDict()
        self.active = 0
        self.local_lock = asyncio.Lock()

    @staticmethod
    def local_docs(event):
        """Use a serialized local-only path when helper slots are occupied."""
        from hook import config, process
        cfg = config()
        if not cfg['enabled'] or not cfg.get('deterministic_first', True):
            return None
        # A candidate that cannot fit must never start a provider in this path.
        return process(event, cfg, engine=lambda **kwargs: {
            'status': 'error', 'error': 'helper_slots_busy', 'usage': {'total_tokens': 0}})

    @staticmethod
    def stop():
        """Ends the summaries still running: the client is gone or the proxy is stopping.
        Cancelling their tasks leaves the helper threads (and asyncio.run) waiting."""
        from summary_engine import stop_helpers
        stop_helpers()

    def prepare_request(self, message):
        """Bound implicit discovery pages; preserve explicit limits and pagination."""
        defaults = {'codebase_memory_mcp': ('search_graph', 20),
                    'openaiDeveloperDocs': ('search_openai_docs', 3)}
        default = defaults.get(self.server.replace('-', '_'))
        if default is None or not isinstance(message, dict):
            return message
        params = message.get('params')
        name, limit = default
        if message.get('method') != 'tools/call' or not isinstance(params, dict) or params.get('name') != name:
            return message
        arguments = params.get('arguments', {})
        if not isinstance(arguments, dict) or 'limit' in arguments:
            return message
        try:
            from hook import config, classify
            cfg = config()
            if (not cfg['enabled'] or os.environ.get('CONTEXT_SAVER_ACTIVE') == '1' or
                    classify({'tool_name': f'mcp__{self.server}__{name}', 'tool_input': arguments})
                    not in cfg['categories']):
                return message
        except Exception:
            return message
        return {**message, 'params': {**params, 'arguments': {**arguments, 'limit': limit}}}

    def observe(self, message):
        if not isinstance(message, dict):
            return
        key = request_key(message.get('id'))
        if key is None or 'method' not in message:
            return
        # Reused IDs must never inherit earlier tool parameters.
        self.pending.pop(key, None)
        params = message.get('params')
        if message['method'] != 'tools/call' or not isinstance(params, dict):
            return
        name, arguments = params.get('name'), params.get('arguments', {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return
        self.pending[key] = (name, arguments)
        if len(self.pending) > MAX_PENDING:
            self.pending.popitem(last=False)

    async def transform(self, message):
        if not isinstance(message, dict) or 'method' in message:
            return message
        key = request_key(message.get('id'))
        tracked = self.pending.pop(key, None)
        if tracked is None or 'error' in message:
            return message
        result = message.get('result')
        # Structured content (even an explicit null), resource links, images,
        # audio, opaque blocks, per-block annotations or metadata and malformed
        # content are contracts: they stay exact.
        if not isinstance(result, dict) or 'structuredContent' in result:
            return message
        text = text_blocks(result.get('content'), result.get('structuredContent'))
        if text is None:
            return message
        name, arguments = tracked
        docs_search = self.server == 'openaiDeveloperDocs' and name == 'search_openai_docs'
        if docs_search:
            from deterministic import docs_search_source
            # This projection is specific to the known navigation JSON envelope;
            # unknown/error result contracts are not handed to a helper either.
            if result.get('isError') or docs_search_source(text) is None:
                return message
        local_only = self.active >= MAX_SUMMARIES
        if local_only and not docs_search:
            return message
        event = {
            'hook_event_name': 'PostToolUse',
            'tool_name': f'mcp__{self.server}__{name}',
            'tool_input': arguments, 'tool_response': result, 'cwd': self.cwd,
            'session_id': self.session_id, 'tool_use_id': str(message['id']),
        }
        if not local_only:
            self.active += 1
        try:
            if local_only:
                async with self.local_lock:
                    decision = await asyncio.to_thread(self.local_docs, event)
            else:
                decision = await asyncio.to_thread(self.process, event)
            if not isinstance(decision, dict) or decision.get('continue') is not False:
                return message
            summary = decision.get('stopReason')
            if not isinstance(summary, str) or not summary.strip():
                return message
            return {**message, 'result': {**result, 'content': [{'type': 'text', 'text': summary}]}}
        except Exception:
            # A helper failure must neither lose a response nor disclose its contents.
            return message
        finally:
            if not local_only:
                self.active -= 1


async def read_line(reader, source):
    """The next line, or b'' at the end. A line over the reader limit is
    dropped whole, with a note on stderr, and reading goes on after it."""
    while True:
        try:
            return await reader.readuntil(b'\n')
        except asyncio.IncompleteReadError as error:
            return error.partial
        except asyncio.LimitOverrunError:
            # Never the content: only that a line was dropped and from where.
            print(f'MCP compression proxy: dropped a {source} line over {MAX_LINE_BYTES} bytes',
                  file=sys.stderr, flush=True)
            await discard_line(reader)


async def discard_line(reader):
    """Drops buffered and incoming bytes up to and including the next newline."""
    while True:
        try:
            await reader.readuntil(b'\n')
            return
        except asyncio.LimitOverrunError as error:
            # error.consumed bytes are buffered and hold no newline.
            await reader.readexactly(error.consumed)
        except asyncio.IncompleteReadError:
            return


def protocol_version_of(message):
    """The protocolVersion of an initialize result, or None."""
    result = message.get('result')
    version = result.get('protocolVersion') if isinstance(result, dict) else None
    return version if isinstance(version, str) else None


class SdkMissing(RuntimeError):
    """The optional MCP SDK (engine/requirements-mcp.txt) is not installed."""


async def stdin_reader():
    reader = asyncio.StreamReader(limit=MAX_LINE_BYTES)
    protocol = asyncio.StreamReaderProtocol(reader)
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)
    return reader, transport


class Output:
    def __init__(self):
        self.lock = asyncio.Lock()

    async def write(self, data):
        async with self.lock:
            # A single write avoids interleaving concurrent responses.
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()


def encode(message):
    return (json.dumps(message, ensure_ascii=False, separators=(',', ':')) + '\n').encode()


async def relay_result(raw, compressor, output):
    try:
        message = json.loads(raw)
    except (ValueError, UnicodeError):
        await output.write(raw)
        return
    replacement = await compressor.transform(message)
    await output.write(raw if replacement is message else encode(replacement))


async def shutdown_child(child):
    if child.stdin:
        child.stdin.close()
    try:
        await asyncio.wait_for(child.wait(), 2)
    except asyncio.TimeoutError:
        child.terminate()
        try:
            await asyncio.wait_for(child.wait(), 2)
        except asyncio.TimeoutError:
            child.kill()
            await child.wait()


async def stdio_bridge(command, compressor):
    child = await asyncio.create_subprocess_exec(
        *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=None, limit=MAX_LINE_BYTES,
    )
    reader, transport = await stdin_reader()
    output = Output()
    results = set()

    async def requests():
        while raw := await read_line(reader, 'client'):
            try:
                original = json.loads(raw)
                message = compressor.prepare_request(original)
                compressor.observe(message)
                if message is not original:
                    raw = encode(message)
            except (ValueError, UnicodeError):
                pass
            child.stdin.write(raw)
            await child.stdin.drain()
        child.stdin.close()

    async def responses():
        while raw := await read_line(child.stdout, 'server'):
            # Backpressure bounds queued data even when summaries are slow.
            if len(results) >= MAX_SUMMARIES:
                await asyncio.wait(results, return_when=asyncio.FIRST_COMPLETED)
            task = asyncio.create_task(relay_result(raw, compressor, output))
            results.add(task)
            task.add_done_callback(results.discard)
        if results:
            await asyncio.gather(*results)

    tasks = [asyncio.create_task(requests()), asyncio.create_task(responses())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        if tasks[0] in done:
            # Let buffered child responses finish before closing the transport.
            try:
                await asyncio.wait_for(asyncio.shield(tasks[1]), 3)
            except asyncio.TimeoutError:
                pass
    finally:
        compressor.stop()
        transport.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for task in list(results):
            task.cancel()
        await asyncio.gather(*results, return_exceptions=True)
        await shutdown_child(child)
    return child.returncode or 0


async def http_bridge(url, compressor):
    # Lazy: only this mode needs the optional SDK (setup --mcp).
    try:
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared.message import ClientMessageMetadata, SessionMessage
        from mcp_types import jsonrpc_message_adapter
    except ImportError:
        raise SdkMissing() from None

    # The SDK can log response bodies at DEBUG; suppress third-party diagnostics.
    logging.getLogger('mcp').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.CRITICAL)
    logging.getLogger('httpx2').setLevel(logging.CRITICAL)
    logging.getLogger('httpcore').setLevel(logging.CRITICAL)
    reader, transport = await stdin_reader()
    output = Output()
    results = set()
    initialization_ids = set()
    protocol_version = None
    try:
        async with streamable_http_client(url) as (incoming, outgoing):
            async def requests():
                while raw := await read_line(reader, 'client'):
                    try:
                        message = json.loads(raw)
                        message = compressor.prepare_request(message)
                        parsed = jsonrpc_message_adapter.validate_python(message)
                    except (ValueError, UnicodeError):
                        await output.write(encode({'jsonrpc': '2.0', 'id': None, 'error': {
                            'code': -32700, 'message': 'Invalid JSON-RPC message'}}))
                        continue
                    compressor.observe(message)
                    if message.get('method') == 'initialize':
                        initialization_ids.add(request_key(message.get('id')))
                    headers = {'MCP-Protocol-Version': protocol_version} if protocol_version else None
                    await outgoing.send(SessionMessage(parsed, ClientMessageMetadata(headers=headers)))

            async def deliver(message):
                replacement = await compressor.transform(message)
                await output.write(encode(replacement))

            async def responses():
                nonlocal protocol_version
                async for item in incoming:
                    if isinstance(item, Exception):
                        raise RuntimeError('MCP upstream transport failure') from None
                    message = item.message.model_dump(mode='json', by_alias=True, exclude_unset=True)
                    key = request_key(message.get('id'))
                    if key in initialization_ids and 'method' not in message:
                        initialization_ids.discard(key)
                        version = protocol_version_of(message)
                        if version:
                            protocol_version = version
                    if len(results) >= MAX_SUMMARIES:
                        await asyncio.wait(results, return_when=asyncio.FIRST_COMPLETED)
                    task = asyncio.create_task(deliver(message))
                    results.add(task)
                    task.add_done_callback(results.discard)

            tasks = [asyncio.create_task(requests()), asyncio.create_task(responses())]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                compressor.stop()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for task in list(results):
                    task.cancel()
                await asyncio.gather(*results, return_exceptions=True)
    finally:
        transport.close()
    return 0


async def serve(compressor, stdio=None, http=None):
    """Runs one bridge; SIGTERM and SIGINT stop it through its shutdown path."""
    current = asyncio.current_task()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, current.cancel)
    return await (stdio_bridge(stdio, compressor) if stdio else http_bridge(http, compressor))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, help='Original MCP server name used for hook classification')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--stdio', nargs=argparse.REMAINDER, metavar='COMMAND')
    modes.add_argument('--http', metavar='URL')
    args = parser.parse_args()
    if args.stdio == []:
        parser.error('--stdio requires a command')
    try:
        return asyncio.run(serve(Compressor(args.server), stdio=args.stdio, http=args.http))
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 0
    except SdkMissing:
        launcher = Path(__file__).resolve().parent.parent / 'bin' / 'context-saver'
        print(f'MCP compression proxy: --http needs the MCP SDK; run: {launcher} setup --mcp', file=sys.stderr)
        return 3
    except Exception as exc:
        # Never emit exception strings, request arguments, URLs or response bodies.
        print(f'MCP compression proxy stopped ({type(exc).__name__})', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
