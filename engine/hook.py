#!/usr/bin/env python3
"""Selective shell wrapping and MCP result compression with local accounting."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from common import (CATEGORIES, DELIVERY_TOKEN, ENV_ASSIGNMENT, GIT_VALUE_OPTIONS, SHELL_TOOLS,
                    USAGE_KEYS, WEB_TOOLS, exit_code, failed, first_operand, private_dir,
                    text_blocks, write_json)
from summary_engine import DEFAULT_EFFORT, EFFORTS, MODEL

ROOT = Path(__file__).resolve().parent
STATE = Path(os.environ.get('CONTEXT_SAVER_STATE') or Path.home() / '.cache/context-saver')
CONFIG_PATH = Path(os.environ.get('CONTEXT_SAVER_CONFIG') or
                   Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config') / 'context-saver/config.json')
# Set by the Claude Code plugin from its /config options; they override config.json.
EFFORT_ENV, SHELL_ENV = 'CONTEXT_SAVER_EFFORT', 'CONTEXT_SAVER_SUMMARIZE_SHELL'
DEFAULTS = {
    'effort': DEFAULT_EFFORT, 'shell_helpers_enabled': True, 'debug_invalid_responses': False,
    # An empty rtk_binary resolves `rtk` on PATH; a missing binary skips the RTK stage.
    'rtk_enabled': True, 'rtk_binary': '',
    'enabled': True, 'deterministic_first': True, 'minimum_tokens': 3500, 'maximum_source_chars': 60000,
    'maximum_replacement_tokens': 1800, 'minimum_saving_fraction': 0.30,
    'timeout_seconds': 300, 'retention_days': 7, 'review_every_days': 7,
    'categories': list(CATEGORIES),
}
# A value that starts like this names a secret held elsewhere; it holds none.
REFERENCE = r'(?!process\.env|os\.environ|ENV\[|\$|\{\{)'
# The lookbehinds start a match only at the start of a run, so long runs of
# letters or dotted words cost linear time.
SECRET = re.compile(
    r'-----BEGIN (?:[A-Z ]*PRIVATE KEY|OPENSSH PRIVATE KEY|PGP PRIVATE KEY BLOCK)-----|'
    r'\b(?:AIza[\w-]{30,}|AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|'
    r'glpat-[A-Za-z0-9_-]{20,}|xox[abposr]-[A-Za-z0-9-]{10,}|[sr]k_(?:live|test)_[A-Za-z0-9]{10,}|'
    r'npm_[A-Za-z0-9]{36,})\b|'
    r'\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*|'
    # scheme://user:password@
    r'(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]{0,30}://[^\s/:@]+:' + REFERENCE + r'[^\s/@]+@|'
    r'\b(?i:Bearer)\s+[A-Za-z0-9._~+/-]{12,}|'
    r'(?i:access_token|refresh_token|client_secret|api[_-]?key|password|passwd)\s*["\']?\s*[:=]\s*["\']?[^\s"\',}]{8,}|'
    r'(?:(?<![A-Z0-9_])[A-Z0-9_]*(?:TOKEN|SECRET)|(?i:aws_secret_access_key))\s*["\']?\s*[:=]\s*["\']?' +
    REFERENCE + r'[^\s"\',}]{8,}'
)
SENSITIVE_PATH = re.compile(r'(?:^|[/\s"\'])(?:\.env(?:[.\s"\'/]|$)|oauth_creds\.json|credentials\.json|id_rsa|id_ed25519|\.ssh/|\.aws/credentials)')
RAW_MARKER = re.compile(r'CONTEXT_SAVER_RAW\s*=\s*1|context-saver:raw')
JSON_ESCAPE = re.compile(r'\\(?:u([0-9a-fA-F]{4})|(["\\/bfnrt]))')
JSON_CONTROLS = {'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}
# A 4 MiB capture, JSON-escaped inside the event envelope, stays well inside this.
STDIN_LIMIT = 32 * 1024 * 1024
TOKEN_SLICE_CHARS = 2048
ERROR_LINE = re.compile(r'error|fail(?:ed|ure)?|exception|traceback|panic|fatal|assert', re.I)
# The ledger is rewritten without year-old rows once it grows past this.
LEDGER_LIMIT_BYTES = 5 * 1024 * 1024
LEDGER_KEEP_DAYS = 365


def as_flag(value):
    if not isinstance(value, bool):
        raise ValueError
    return value


def as_integer(low, high):
    """An int setting: an int (not a bool), an integral float or a numeric string."""
    def convert(value):
        if isinstance(value, str):
            value = float(value)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if type(value) is not int or not low <= value <= high:
            raise ValueError
        return value
    return convert


def as_fraction(value):
    """A float setting in [0.1, 1): an int, a float or a numeric string."""
    if isinstance(value, str):
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not .1 <= value < 1:
        raise ValueError
    return float(value)


def as_choice(options):
    def convert(value):
        if not isinstance(value, str) or value not in options:
            raise ValueError
        return value
    return convert


def as_categories(value):
    if not isinstance(value, list) or any(not isinstance(x, str) or x not in CATEGORIES for x in value):
        raise ValueError
    return list(value)


def as_string(value):
    if not isinstance(value, str):
        raise ValueError
    return value


# What config() accepts for each key, from config.json or the plugin options;
# each returns the value config() hands out.
VALID = {
    'enabled': as_flag, 'deterministic_first': as_flag, 'rtk_enabled': as_flag,
    'shell_helpers_enabled': as_flag, 'debug_invalid_responses': as_flag,
    'minimum_tokens': as_integer(1000, 50000),
    'maximum_source_chars': as_integer(1000, 60000),
    'maximum_replacement_tokens': as_integer(200, 2000),
    'minimum_saving_fraction': as_fraction,
    'timeout_seconds': as_integer(5, 300),
    'retention_days': as_integer(1, 365),
    'review_every_days': as_integer(1, 365),
    'effort': as_choice(EFFORTS),
    'categories': as_categories,
    'rtk_binary': as_string,
}


def config():
    value = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            stored = json.loads(CONFIG_PATH.read_text())
        except ValueError as exc:
            raise ValueError(f'context-saver config: {CONFIG_PATH} is not valid JSON ({exc})') from None
        if not isinstance(stored, dict):
            raise ValueError(f'context-saver config: {CONFIG_PATH} must hold a JSON object')
        value.update(stored)
    if os.environ.get(EFFORT_ENV, '').strip():
        value['effort'] = os.environ[EFFORT_ENV].strip()
    if os.environ.get(SHELL_ENV) in ('0', '1'):
        value['shell_helpers_enabled'] = os.environ[SHELL_ENV] == '1'
    for key, convert in VALID.items():
        try:
            value[key] = convert(value[key])
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'context-saver config: invalid {key} {value[key]!r:.80}') from None
    return value


def tokens(text):
    """Bound repetitive BPE work; slice boundaries may change the estimate."""
    import tiktoken
    encoding = tiktoken.get_encoding('o200k_base')
    return sum(len(encoding.encode(text[i:i + TOKEN_SLICE_CHARS], disallowed_special=()))
               for i in range(0, len(text), TOKEN_SLICE_CHARS))


def unescaped(text, levels=4):
    """The text, then the text with JSON string escapes undone, one nesting level at a time."""
    yield text
    for _ in range(levels):
        decoded = JSON_ESCAPE.sub(lambda m: chr(int(m[1], 16)) if m[1] else JSON_CONTROLS.get(m[2], m[2]), text)
        if decoded == text:
            return
        text = decoded
        yield text


def sensitive(text):
    """True when the text names a credential or a credential file, also inside
    JSON-encoded strings, where escaping hides the quotes the patterns match."""
    return any(SECRET.search(view) or SENSITIVE_PATH.search(view) for view in unescaped(text))


SHELL_SEPARATORS = ';&|()\n'
REDIRECTIONS = '<>'


def shell_source(command):
    """Strip comments, rejecting command substitutions and unfinished quoting."""
    quoted, escaped = None, False
    source, comment, word_start = [], False, True
    for index, char in enumerate(command):
        if comment:
            if char != '\n':
                continue
            comment = False
        if escaped:
            escaped = False
            if char == '\n':
                source.pop()
                continue
            word_start = False
        elif quoted == "'":
            quoted = None if char == "'" else quoted
        elif char == '\\':
            escaped = True
        elif quoted is None and char == '#' and word_start:
            comment = True
            continue
        elif char == '`' or (char == '$' and re.match(r'(?:\\\n)*\(', command[index + 1:])):
            raise ValueError('Command substitution')
        elif char == '"':
            quoted = None if quoted == '"' else '"'
            word_start = False
        elif char == "'" and quoted is None:
            quoted = "'"
            word_start = False
        else:
            word_start = quoted is None and char in ' \t\r' + SHELL_SEPARATORS + REDIRECTIONS
        source.append(char)
    if quoted or escaped:
        raise ValueError('Unfinished shell quote or escape')
    return ''.join(source)


def shell_tokens(command):
    """Shell words and operator runs; an unquoted newline separates commands too."""
    lexer = shlex.shlex(shell_source(command), posix=True, punctuation_chars=SHELL_SEPARATORS + REDIRECTIONS)
    lexer.whitespace = ' \t\r'
    lexer.whitespace_split = True
    lexer.commenters = ''
    return list(lexer)


def is_operator(token):
    return bool(token) and all(c in SHELL_SEPARATORS + REDIRECTIONS for c in token)


def is_separator(token):
    return bool(token) and all(c in SHELL_SEPARATORS for c in token)


def is_redirection(token):
    """>, >>, 2>&1's >&, &>, <, << and <<<."""
    return any(c in REDIRECTIONS for c in token) and all(c in REDIRECTIONS + '&' for c in token)


def command_heads(command):
    """(name, arguments, reads a pipe, redirects input); None when unparsable."""
    try:
        parts = shell_tokens(command)
    except ValueError:
        return None
    heads, chunk, piped, redirection, input_redirected = [], [], False, None, False
    for part in parts + [';']:
        if redirection is not None:
            if is_operator(part):
                return None
            # A redirected file stays an operand (< file feeds a filter);
            # descriptors (2>&1) and /dev/null are not files a command prints.
            if not (part == '/dev/null' or ('&' in redirection and re.fullmatch(r'\d+|-', part))):
                chunk.append(part)
            redirection = None
        elif is_redirection(part):
            if len(chunk) > 1 and chunk[-1].isdigit():
                chunk.pop()  # the descriptor of 2>file
            redirection = part
            input_redirected = input_redirected or '<' in part
        elif is_separator(part):
            while chunk and ENV_ASSIGNMENT.fullmatch(chunk[0]):
                chunk.pop(0)
            if chunk:
                heads.append((Path(chunk[0]).name, chunk[1:], piped, input_redirected))
            chunk = []
            piped = part.strip('\n') in ['|', '|&']
            input_redirected = False
        elif is_operator(part):
            return None  # mixed runs such as <( or ;> are not parsed
        else:
            chunk.append(part)
    return heads if redirection is None else None


SEARCH_COMMANDS = {'rg', 'grep', 'egrep', 'fgrep', 'find', 'fd'}
LOG_COMMANDS = {'pytest', 'py.test', 'jest', 'vitest', 'tsc', 'eslint', 'ruff', 'journalctl'}
# Build tools count only for a test/build/lint subcommand or package script,
# found past the options that take a value.
BUILD_TOOLS = {'npm': {'--prefix', '-w', '--workspace'}, 'pnpm': {'-C', '--dir', '-F', '--filter'},
               'yarn': {'--cwd'}, 'bun': {'--cwd'}, 'cargo': {'--manifest-path', '--color', '--config', '-Z'},
               'go': set(), 'make': {'-C', '-f', '-I', '-o', '-W', '--directory', '--file'}}
CHECK_VERB = re.compile(r'(?:^|:)(?:test|build|lint|typecheck|check|verify)(?:$|:)')
# Container tools count only for their logs subcommand.
CONTAINER_TOOLS = {'docker', 'podman', 'kubectl'}
CONTAINER_OPTIONS = {'-H', '--host', '--context', '-c', '--config', '-l', '--log-level', '-n',
                     '--namespace', '--cluster', '--kubeconfig', '--user', '-s', '--server'}
COMPOSE_OPTIONS = {'-f', '--file', '-p', '--project-name', '--env-file', '--profile'}
# They only reshape the output piped into them, unless they name a file.
PIPE_FILTERS = {'head', 'tail', 'sort', 'uniq', 'wc', 'cut', 'tr', 'cat'}
FIND_EXEC_OPTIONS = {'-exec', '-execdir', '-ok', '-okdir'}


def file_operands(head, args):
    """The words a command reads as files; head/tail -n/-c values are not files."""
    values = {'-n', '-c', '--lines', '--bytes'} if head in ('head', 'tail') else set()
    operands, skip, ended = [], False, False
    for arg in args:
        if skip:
            skip = False
        elif ended or not arg.startswith('-'):
            operands.append(arg)
        elif arg == '--':
            ended = True
        elif arg in values:
            skip = True
    return operands


def runs_commands(head, args):
    """True when find or fd runs another command on each match."""
    if head == 'find':
        return any(a in FIND_EXEC_OPTIONS for a in args)
    # fd also takes -x/-X attached or in a cluster (-xcat, -HIxcat); a short
    # cluster with an x in a value (-tx, -exml) is declined with them.
    return head == 'fd' and any(a.split('=')[0] in ('--exec', '--exec-batch') or
                                (re.fullmatch(r'-[^-].*', a) and re.search('[xX]', a)) for a in args)


def command_category(head, args, piped, input_redirected):
    """The output category of one simple command: '' adds none; None means the
    command is not a known search, log or git command, so its output stays exact."""
    if head == 'cd':
        return ''
    operands = file_operands(head, args)
    if head in PIPE_FILTERS and piped:
        # Plain short options only: a long option such as --files0-from reads files.
        plain = all(not a.startswith('--') or a.split('=')[0] in ('--lines', '--bytes') for a in args)
        return '' if not input_redirected and plain and (head == 'tr' or not operands) else None
    if head in ('cat', 'head', 'tail'):
        return 'logs' if operands and all(a.endswith('.log') for a in operands) else None
    if head in SEARCH_COMMANDS:
        return None if runs_commands(head, args) else 'search'
    if head == 'git':
        index, verb = first_operand(args, GIT_VALUE_OPTIONS)
        rest = args[index + 1:]
        # -c can set a pager, external diff or alias that runs any command;
        # git show <rev>:<path> and git diff --no-index print files.
        if '-c' in args[:index] or (verb == 'show' and any(':' in a and not a.startswith('-') for a in rest)):
            return None
        if verb == 'diff' and '--no-index' in rest:
            return None
        return 'git' if verb in ['diff', 'show', 'log', 'status'] else None
    if head == 'gh':
        if args[:2] in [['pr', 'diff'], ['pr', 'view']]:
            return 'git'
        return 'logs' if args[:2] == ['run', 'view'] and '--log' in args else None
    if head in LOG_COMMANDS:
        return 'logs'
    if head in BUILD_TOOLS:
        index, verb = first_operand(args, BUILD_TOOLS[head])
        if head in ('npm', 'pnpm', 'yarn', 'bun') and verb in ('run', 'run-script'):
            verb = first_operand(args[index + 1:])[1]
        return 'logs' if verb and CHECK_VERB.search(verb) else None
    if head in CONTAINER_TOOLS:
        index, verb = first_operand(args, CONTAINER_OPTIONS)
        if verb == 'compose' and head != 'kubectl':
            verb = first_operand(args[index + 1:], COMPOSE_OPTIONS)[1]
        return 'logs' if verb == 'logs' else None
    return None


def classify(event):
    name = event.get('tool_name', '')
    arguments = event.get('tool_input') or {}
    if not isinstance(arguments, dict):
        return None
    command = arguments.get('command', arguments.get('cmd', ''))
    request = json.dumps(arguments, ensure_ascii=False)
    if RAW_MARKER.search(request) or sensitive(request):
        return None
    if 'context-saver' in request:
        return None
    # Explicit source reads stay exact; a search can be followed by a precise read.
    if name in ['Read', 'read_file', 'get_code_snippet'] or name.endswith('__get_code_snippet'):
        return None
    if name in ['Grep', 'Glob']:
        return 'search'
    if name in WEB_TOOLS:
        # Native web has no hook transport; only the explicit code-mode adapter
        # calls process for these results, before emitting them to the model.
        operations = {'search_query', 'open', 'find', 'click'}
        if not set(arguments) <= operations | {'response_length'}:
            return None
        if not any(arguments.get(key) for key in operations):
            return None
        return 'search' if arguments.get('search_query') else 'documentation'
    if 'codebase' in name and re.search(r'(?:search_graph|query_graph|trace_path|get_architecture|search_code)$', name):
        return 'search'
    if re.search(r'(?:fetch_openai_doc|fetch_documentation|read_documentation|search_openai_docs)$', name):
        return 'documentation'
    if name not in SHELL_TOOLS or not isinstance(command, str):
        return None
    # Every command must be a known search, log or git command, or a filter of
    # their piped output. Anything else (a file read, a script) may print source,
    # so the whole output stays exact, like a standalone Read.
    categories = [command_category(*head) for head in command_heads(command) or []]
    if not categories or None in categories:
        return None
    found = [category for category in categories if category]
    return 'logs' if 'logs' in found else (found[0] if found else None)


def extract(response):
    if isinstance(response, str):
        # Some adapters JSON-encode the output envelope into a text value.
        with contextlib.suppress(ValueError):
            decoded = json.loads(response)
            if isinstance(decoded, dict) and ('output' in decoded or 'content' in decoded):
                return extract(decoded)
        return response, {}
    if isinstance(response, dict):
        if isinstance(response.get('output'), str):
            # Claude Code reports no exit code: exit_code stays absent, not 0.
            metadata = {k: response[k] for k in ['exit_code', 'session_id'] if k in response}
            if isinstance(response.get('stderr'), str) and response['stderr']:
                metadata['stderr'] = response['stderr']
            if isinstance(response.get('return_code_interpretation'), str):
                metadata['return_code_interpretation'] = response['return_code_interpretation']
            if isinstance(response.get('isError'), bool):
                metadata['isError'] = response['isError']
            return response['output'], metadata
        if isinstance(response.get('content'), list):
            # Images, resources, annotated blocks and structured output are
            # contracts the caller must keep unchanged.
            text = text_blocks(response['content'], response.get('structuredContent'))
            if text is None:
                return '', {'unsupported_content': True}
            return text, {'isError': response.get('isError', False)}
        return json.dumps(response, ensure_ascii=False), {}
    return '', {}


def pre_tool(event):
    """Wrap eligible shell commands before execution; works in nested code mode."""
    cfg = config()
    if not cfg['enabled'] or os.environ.get('CONTEXT_SAVER_ACTIVE') == '1':
        return None
    if event.get('hook_event_name') != 'PreToolUse' or event.get('tool_name') != 'Bash':
        return None
    category = classify(event)
    if category not in cfg['categories']:
        return None
    command = event['tool_input']['command']
    # Avoid rewriting compound programs, substitutions and redirected commands.
    if re.search(r'`|\$\(|[<>]|\b(?:xargs|eval|exec|sudo)\b|--pre(?:=|\s)|-exec(?:dir)?\b', command):
        return None
    try:
        segments = shell_tokens(command)
    except ValueError:
        return None
    # Background jobs and subshells are never wrapped; && and |& are fine.
    if any(is_separator(item) and re.search(r'[()]|(?<![&|])&(?!&)', item) for item in segments):
        return None
    # Follow and watch modes never end, so their output must stream directly.
    for head, argv, _, _ in command_heads(command):
        if any(a.startswith('--watch') or
               (head in {'tsc', 'vitest'} and a == '-w') or
               (head in {'pytest', 'py.test'} and a in {'-f', '--looponfail'}) or
               (head in {'tail', 'docker', 'podman', 'kubectl', 'journalctl'} and
                (a.split('=')[0] == '--follow' or (re.match(r'-[^-]', a) and re.search('[fF]', a)))) for a in argv):
            return None
    # Preserve native workdir, max_output_tokens, timeout and exit-code handling.
    # Through the launcher, so the wrapped run gets the same venv and caches.
    # Plain quoted words only: no VAR= prefix, expansion or substitution, so
    # exec-policy prefix rules still see a simple command. Codex starts a fresh
    # shell per command, so there are no caller shell options to carry.
    wrapped = (shlex.quote(str(ROOT.parent / 'bin' / 'context-saver')) +
               ' run --command ' + shlex.quote(command) +
               ' --session-id ' + shlex.quote(str(event.get('session_id', ''))) +
               ' --tool-use-id ' + shlex.quote(str(event.get('tool_use_id', ''))))
    # Codex requires this PreToolUse value for rewrites; this is not a
    # PermissionRequest approval decision, and sandbox policy still applies.
    return {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'allow',
                                  'updatedInput': {'command': wrapped}}}


def run_command(command, session_id='', tool_use_id=''):
    child_env = os.environ.copy()
    # The user's shell ($SHELL), else /bin/sh.
    shell = shutil.which(child_env.get('SHELL', ''), path=child_env.get('PATH')) or '/bin/sh'
    child_env['CONTEXT_SAVER_ACTIVE'] = '1'
    # A stop signal unwinds through the replay below; the default handlers would
    # end this wrapper while it holds the only copy of the command's output.
    stop_on_signals()
    try:
        captured = tempfile.TemporaryFile()
    except OSError:
        # No writable temp folder (read-only sandbox): run the command as is.
        return exit_status(run_group([shell, '-c', command], env=child_env))
    with captured:
        try:
            returncode = run_group([shell, '-c', command], stdout=captured, stderr=subprocess.STDOUT, env=child_env)
            size = captured.tell()
            captured.seek(0)
            replacement = None
            with contextlib.suppress(Exception):
                # Bound memory independently of the much smaller helper input limit.
                if size <= 4 * 1024 * 1024:
                    raw = captured.read().decode('utf-8', errors='strict')
                    replacement = process({'hook_event_name': 'PostToolUse', 'tool_name': 'Bash',
                                           'tool_input': {'command': command}, 'cwd': os.getcwd(),
                                           'session_id': session_id, 'tool_use_id': tool_use_id,
                                           'tool_response': {'output': raw, 'exit_code': returncode}})
        except BaseException:
            # Stopped while running or compressing: the original still goes out.
            replay(captured)
            raise
        if replacement:
            sys.stdout.write(replacement['stopReason'] + '\n')
        else:
            replay(captured)
        return exit_status(returncode)


def run_group(argv, **options):
    """Run a command in its own group, keeping the caller's session."""
    process = subprocess.Popen(argv, preexec_fn=os.setpgrp, **options)
    try:
        return process.wait()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + 2
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            # Descendants may still be running after the shell exits; signal 0
            # raises ProcessLookupError once the whole group is gone.
            while time.monotonic() < deadline:
                os.killpg(process.pid, 0)
                time.sleep(.05)
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        finally:
            process.wait()
        raise


def stop_on_signals():
    """A stop signal unwinds through the cleanup that replays output and ends
    helper processes; the default handlers would exit at once and skip it."""
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop_wrapper)


def exit_status(returncode):
    """The shell's exit status for a child's return code (signals as 128+N)."""
    return returncode if returncode >= 0 else 128 - returncode


def stop_wrapper(signum, _frame):
    raise SystemExit(128 + signum)


def replay(captured):
    """Writes the command's captured output unchanged."""
    sys.stdout.flush()
    captured.seek(0)
    while chunk := captured.read(65536):
        sys.stdout.buffer.write(chunk)
    sys.stdout.buffer.flush()


@contextlib.contextmanager
def ledger_lock(mode):
    """Appends share the ledger (LOCK_SH); the size rewrite holds it alone (LOCK_EX)."""
    private_dir(STATE)
    fd = os.open(STATE / 'events.lock', os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, mode)
        yield
    finally:
        os.close(fd)


def append_event(value):
    value = {'time': time.time(), **value}
    encoded = (json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    with ledger_lock(fcntl.LOCK_SH):
        fd = os.open(STATE / 'events.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, encoded)
        finally:
            os.close(fd)


def trim_ledger(now):
    """Rewrites a ledger over LEDGER_LIMIT_BYTES without rows older than LEDGER_KEEP_DAYS."""
    path = STATE / 'events.jsonl'
    with contextlib.suppress(FileNotFoundError):
        if path.stat().st_size <= LEDGER_LIMIT_BYTES:
            return
        cutoff = now - LEDGER_KEEP_DAYS * 86400
        with ledger_lock(fcntl.LOCK_EX):
            kept = []
            with path.open('rb') as handle:
                for line in handle:
                    with contextlib.suppress(ValueError, TypeError, AttributeError):
                        if json.loads(line).get('time', 0) >= cutoff:
                            kept.append(line if line.endswith(b'\n') else line + b'\n')
            fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=STATE)
            try:
                with os.fdopen(fd, 'wb') as handle:
                    handle.writelines(kept)
                os.replace(temporary, path)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporary)


def archive_path(text):
    """Where archive() keeps this text; the replacement names it before it is written."""
    return STATE / 'raw' / (hashlib.sha256(text.encode()).hexdigest() + '.txt')


def archive(text):
    """Writes the original once, only for an accepted replacement. The file
    appears complete (link of a finished private file); a file that is
    already there holds the same text."""
    path = archive_path(text)
    private_dir(path.parent)
    if not path.exists():
        fd, temporary = tempfile.mkstemp(prefix='.pending-', suffix='.txt', dir=path.parent)
        try:
            with os.fdopen(fd, 'w') as handle:
                handle.write(text)
            with contextlib.suppress(FileExistsError):
                os.link(temporary, path)
        finally:
            os.unlink(temporary)
    path.touch()
    return path


def exact_diagnostics(text, metadata, category):
    """Original lines around failures: for a known non-zero exit code, isError,
    or a log that reports an error (Claude Code gives no exit code)."""
    if not failed(metadata) and category != 'logs':
        return ''
    lines = text.splitlines()
    errors = [i for i, line in enumerate(lines) if ERROR_LINE.search(line)]
    if not failed(metadata) and not errors:
        return ''
    selected = set(range(max(0, len(lines) - 8), len(lines)))
    for i in errors:
        selected.update(range(max(0, i - 1), min(len(lines), i + 2)))
        if len(selected) >= 30:
            break
    return '\n'.join(lines[i] for i in sorted(selected))[:2600]


def stderr_tail(metadata):
    """The last 8 stderr lines, at most 800 characters."""
    return '\n'.join(metadata.get('stderr', '').splitlines()[-8:])[-800:]


def label_savings(replacement, before, result):
    """Budget the visible label itself; return only a consistent token count."""
    model = result.get('model') or ''
    producer = f'Claude ({model})' if model else 'Claude'
    if result.get('local_compaction'):
        producer = ('RTK (local)' if model == 'rtk-stdin-filter' else 'local compaction') if result.get('local_only') else producer + ' + local compaction'
    cached = '; cache hit' if result.get('cache_hit') or result.get('status') == 'cache_hit' else ''

    def measure(saved):
        labelled = f'[{saved} tokens saved by {producer}] (estimated context tokens{cached})\n' + replacement
        return labelled, tokens(labelled)

    # Numeric text can change token length at digit boundaries, including a
    # two-cycle. Bound convergence and a nearby search, then preserve raw output
    # if no exact fixed point exists instead of reporting inconsistent savings.
    candidate = before - tokens(replacement)
    seen = set()
    for _ in range(16):
        if candidate in seen:
            break
        seen.add(candidate)
        labelled, after = measure(candidate)
        if candidate == before - after:
            return labelled, after, candidate
        candidate = before - after
    for saved in range(max(0, candidate - 64), min(before, candidate + 64) + 1):
        labelled, after = measure(saved)
        if saved == before - after:
            return labelled, after, saved
    return None


def render_replacement(text, before, category, metadata, raw, current, output_suffix='', coverage=None):
    """Shared final-payload accounting for local candidates and helper summaries."""
    code = exit_code(metadata)
    status = f"exit_code={code}" if code is not None else f"isError={metadata.get('isError', False)}"
    # The archive path is line 3, so the first `Original output:` is always it.
    replacement = (
        f"[Context summary via {current.get('model', 'helper')}: {category}; {status}; estimated original tokens={before}]\n"
        f"Original output: {raw}\n"
        f"{current['summary']}\n\n"
        "This is a navigation summary, not exact source; read the original output before edits or checking omitted details.\n"
        "To bypass this hook for a shell command, prefix it with CONTEXT_SAVER_RAW=1."
    )
    if coverage is not None and not current.get('local_only'):
        replacement += '\nLocal compaction coverage: ' + json.dumps(coverage, ensure_ascii=False)
    minimal = current.get('minimal_archive_notice')
    diagnostics = '' if minimal or current.get('complete_diagnostics') else exact_diagnostics(text, metadata, category)
    if diagnostics:
        replacement += '\nExact failure excerpts (original):\n' + diagnostics
    tail = '' if minimal else stderr_tail(metadata)
    if tail.strip() and tail not in diagnostics and tail not in current['summary']:
        replacement += '\nExact stderr tail (original):\n' + tail
    if output_suffix:
        replacement += '\n' + output_suffix
    return label_savings(replacement, before, current)


def helper_context(event):
    """The request part of the summary context (and so of its cache key): the
    command alone for a shell tool, so its description or timeout never miss
    the cache; the arguments JSON for MCP and web."""
    arguments = event.get('tool_input') or {}
    command = arguments.get('command', arguments.get('cmd')) if isinstance(arguments, dict) else None
    if event.get('tool_name') in SHELL_TOOLS and isinstance(command, str):
        return f"Tool: {event.get('tool_name')}; command: {command[:1600]}"
    return f"Tool: {event.get('tool_name')}; request: {json.dumps(arguments, ensure_ascii=False)[:1600]}"


def process(event, cfg=None, engine=None, *, output_suffix=''):
    cfg = cfg or config()
    if not cfg['enabled'] or os.environ.get('CONTEXT_SAVER_ACTIVE') == '1':
        return None
    if event.get('hook_event_name') != 'PostToolUse':
        return None
    category = classify(event)
    if category not in cfg['categories']:
        return None
    text, metadata = extract(event.get('tool_response'))
    if not text or metadata.get('session_id'):
        return None
    # A token is never shorter than a byte: small output skips the costly
    # credential scan and tokenizer.
    if len(text.encode('utf-8', 'surrogatepass')) < cfg['minimum_tokens']:
        return None
    shell_local_only = event.get('tool_name') in SHELL_TOOLS and not cfg.get('shell_helpers_enabled', True)
    # Failed shell commands are evidence, not filter candidates; an unknown
    # exit code (Claude Code Bash) is not a failure.
    if shell_local_only and failed(metadata):
        return None
    base = {'category': category, 'tool': event.get('tool_name'), 'session_id': event.get('session_id'),
            'tool_use_id': event.get('tool_use_id'), 'tokenizer': 'o200k_base', 'measurement': 'estimated_tool_payload_only',
            # The host saved this output to a file and showed the model a preview
            # only, so its savings count against the full output.
            **({'persisted': True} if event.get('persisted_output') is True else {})}
    if sensitive(text):
        append_event({**base, 'status': 'skipped_sensitive', 'saved_tokens': 0})
        return None
    before = tokens(text)
    if before < cfg['minimum_tokens']:
        # Avoid an event per trivial command; only record eligible large outputs.
        return None
    base['before_tokens'] = before
    prepared = None
    raw = None
    local_result = None
    if cfg.get('deterministic_first', True):
        local_started = time.monotonic()
        from deterministic import candidates
        for candidate in candidates(text, category, event, cfg):
            raw = raw or archive_path(text)
            current = {'status': 'summary', 'summary': candidate['text'],
                       'provider': 'local_compaction', 'model': candidate.get('model', 'deterministic-filter'),
                       'local_compaction': True, 'local_only': True, 'cache_hit': False,
                       'complete_diagnostics': candidate['kind'] == 'jest_success_filter',
                       'usage': dict.fromkeys(USAGE_KEYS, 0), 'elapsed_ms': 0, 'usage_incomplete': False}
            labelled = render_replacement(text, before, category, metadata, raw, current, output_suffix)
            if (labelled is not None and labelled[1] <= cfg['maximum_replacement_tokens'] and
                    labelled[2] / before >= cfg['minimum_saving_fraction']):
                local_result = current
                local_result['elapsed_ms'] = round((time.monotonic() - local_started) * 1000)
                base.update(compaction_kind=candidate['kind'], coverage=candidate['coverage'],
                            deterministic=True)
                break
    if shell_local_only and local_result is None:
        append_event({**base, 'status': 'skipped_shell_helpers_disabled', 'saved_tokens': 0})
        return None
    if local_result is None and len(text) > cfg['maximum_source_chars']:
        from oversize import prepare
        prepared = prepare(text, category, event, max_chars=cfg['maximum_source_chars'])
        if prepared is None:
            append_event({**base, 'status': 'skipped_oversize', 'saved_tokens': 0})
            return None
        # A bounded preview is useful only if the exact original is retrievable.
        raw = archive_path(text)
        base.update(compaction_kind=prepared['kind'], coverage=prepared['coverage'])

    def local_fallback(previous=None, reason=None):
        previous = previous or {}
        base['coverage'] = prepared['fallback_coverage']
        return {**previous, 'status': 'summary', 'summary': prepared['fallback_text'],
                'provider': 'local_compaction', 'model': 'local-compaction',
                'local_compaction': True, 'local_only': True, 'cache_hit': False,
                'fallback_reason': reason, 'error': None}

    private_dir(STATE)
    if local_result is not None:
        result = local_result
    elif prepared and not prepared['helper_eligible']:
        result = local_fallback(reason='archive_preview')
    else:
        with (STATE / 'active.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if not prepared:
                    append_event({**base, 'status': 'skipped_busy', 'saved_tokens': 0})
                    return None
                result = local_fallback(reason='helper_busy')
            else:
                if engine is None:
                    from summary_engine import summarize
                    engine = summarize
                context = helper_context(event)
                helper_text = prepared['text'] if prepared else text
                try:
                    result = engine(text=helper_text, category=category, context=context,
                                    state_dir=STATE / 'engine', timeout_seconds=cfg['timeout_seconds'],
                                    effort=cfg['effort'])
                except Exception as exc:
                    if not prepared:
                        raise
                    result = {'status': 'error', 'error': type(exc).__name__, 'usage_incomplete': True}
                if prepared:
                    if result.get('status') not in ['summary', 'cache_hit'] or not result.get('summary'):
                        result = local_fallback(result, result.get('error', 'helper_failed'))
                    else:
                        result = {**result, 'local_compaction': True}
    usage = result.get('usage') or {}
    base.update({'helper_usage': usage, 'cache_hit': bool(result.get('cache_hit') or result.get('status') == 'cache_hit'), 'latency_ms': result.get('elapsed_ms', 0),
                 'provider': result.get('provider') or 'unknown', 'model': result.get('model'),
                 'observed_model': result.get('observed_model'),
                 'usage_incomplete': bool(result.get('usage_incomplete')),
                 **({'list_cost_usd': result['list_cost_usd']} if 'list_cost_usd' in result else {}),
                 **({'fallback_reason': result.get('fallback_reason')} if prepared else {})})
    if result.get('status') not in ['summary', 'cache_hit'] or not result.get('summary'):
        append_event({**base, 'status': 'helper_failed', 'saved_tokens': 0, 'error': result.get('error', 'unknown')[:160]})
        return None
    raw = raw or archive_path(text)
    def render(current):
        return render_replacement(text, before, category, metadata, raw, current, output_suffix,
                                  prepared['coverage'] if prepared else None)

    labelled = render(result)
    if prepared and not result.get('local_only') and (labelled is None or
            labelled[1] > cfg['maximum_replacement_tokens'] or labelled[2] / before < cfg['minimum_saving_fraction']):
        result = local_fallback(result, 'helper_output_over_budget')
        base.update(provider=result['provider'], model=result['model'], cache_hit=False,
                    fallback_reason=result['fallback_reason'])
        labelled = render(result)
    if prepared and result.get('local_only') and (labelled is None or
            labelled[1] > cfg['maximum_replacement_tokens']):
        # An unusual token-dense preview must not trigger another giant raw dump.
        result = {**result, 'summary': 'Oversized result archived in full; preview omitted to fit the output budget. '
                  'Read the original file above for exact results and failure details.', 'minimal_archive_notice': True}
        coverage = dict(base['coverage'])
        if 'returned' in coverage:
            coverage.update(shown=0, remaining=coverage['returned'], docstrings_included=0,
                            docstrings_omitted=coverage.get('docstrings_included', 0) + coverage.get('docstrings_omitted', 0))
        if 'original_chars' in coverage:
            coverage.update(shown_chars=0, omitted_chars=coverage['original_chars'])
        base['coverage'] = coverage
        labelled = render(result)
    if labelled is None:
        append_event({**base, 'status': 'insufficient_saving', 'reason': 'label_accounting_unstable', 'saved_tokens': 0})
        return None
    replacement, after, saved = labelled
    if after > cfg['maximum_replacement_tokens'] or saved / before < cfg['minimum_saving_fraction']:
        append_event({**base, 'status': 'insufficient_saving', 'candidate_after_tokens': after, 'saved_tokens': 0})
        return None
    # Written only now: a declined replacement leaves no archive behind.
    archive(text)
    # A host that sends a delivery token reports later (`delivered`) whether it
    # used the replacement; only then does the saving count.
    delivery_token = event.get('delivery_token')
    pending_delivery = isinstance(delivery_token, str) and DELIVERY_TOKEN.fullmatch(delivery_token)
    append_event({**base, 'status': 'compression_candidate' if pending_delivery else 'compressed',
                  **({'delivery_token': delivery_token} if pending_delivery else {}),
                  'after_tokens': after, 'saved_tokens': saved,
                  'source_sha256': raw.stem, 'raw_file': str(raw)})
    return {'continue': False, 'stopReason': replacement}


def stats(days=None):
    rows = []
    path = STATE / 'events.jsonl'
    cutoff = time.time() - days * 86400 if days else 0
    if path.exists():
        for line in path.read_text().splitlines():
            with contextlib.suppress(ValueError, TypeError):
                row = json.loads(line)
                if row.get('time', 0) >= cutoff:
                    rows.append(row)
    # Delivery rows: `delivered`, and `web_delivered` in ledgers before 0.3.0.
    delivered = {x.get('delivery_token') for x in rows if x.get('status') in ('delivered', 'web_delivered')}
    compressed = [x for x in rows if x.get('status') == 'compressed' or
                  (x.get('status') == 'compression_candidate' and x.get('delivery_token') in delivered)]
    before = sum(x['before_tokens'] for x in compressed)
    after = sum(x['after_tokens'] for x in compressed)
    persisted = [x for x in compressed if x.get('persisted')]
    usage = {k: sum(int((x.get('helper_usage') or {}).get(k, 0)) for x in rows) for k in USAGE_KEYS}
    statuses = {}
    for row in rows:
        statuses[row.get('status')] = statuses.get(row.get('status'), 0) + 1
    rtk_rows = [x for x in compressed if x.get('model') == 'rtk-stdin-filter']
    return {'period_days': days, 'events': len(rows), 'compressed_outputs': len(compressed),
            'cache_hits': sum(bool(x.get('cache_hit')) for x in compressed), 'statuses': statuses,
            'estimated_before_tokens': before, 'estimated_after_tokens': after,
            'estimated_context_tokens_saved': before - after,
            'estimated_context_reduction_percent': round(100 * (before - after) / before, 4) if before else 0,
            'persisted_outputs': len(persisted),
            'persisted_tokens_saved': sum(x['before_tokens'] - x['after_tokens'] for x in persisted),
            'helper_usage': usage,
            'rtk': {'compressed_outputs': len(rtk_rows),
                    'estimated_context_tokens_saved': sum(x['saved_tokens'] for x in rtk_rows)},
            'usage_incomplete_events': sum(bool(x.get('usage_incomplete')) for x in rows),
            'claude_list_price_usd': round(sum(float(x.get('list_cost_usd', 0)) for x in rows), 6),
            'estimated_net_tokens_saved_including_helper': before - after - usage['total_tokens'],
            'tokenizer': 'o200k_base', 'billing': 'Claude uses the Claude Code subscription login (claude_list_price_usd is the CLI list-price equivalent, not a charge)',
            'limits': 'Payload estimate only; excludes main-model reasoning, repeated context, follow-up reads and account credit deductions. '
                      'persisted_outputs were saved to a file by the host, which showed the model only a preview, '
                      'so their savings (persisted_tokens_saved, included in the totals) count against the full output.'}


def sweep(cfg, now):
    """Deletes archives, debug responses and summary cache entries older than
    retention_days, and web adapter inputs older than a day."""
    import summary_engine
    retention = cfg['retention_days'] * 86400
    for path, limit in [*((p, retention) for p in (STATE / 'raw').glob('*.txt')),
                        *((p, retention) for p in (STATE / 'engine' / 'debug').glob('*.txt')),
                        *((p, 86400) for p in STATE.glob('web-input.*'))]:
        with contextlib.suppress(OSError):
            if now - path.lstat().st_mtime > limit:
                path.unlink()
    summary_engine.prune_cache(STATE / 'engine', retention)


def session_start(cfg=None):
    cfg = cfg or config()
    if not cfg['enabled'] or os.environ.get('CONTEXT_SAVER_ACTIVE') == '1':
        return None
    private_dir(STATE)
    now = time.time()
    # Cleanup comes first: it never waits for the review reminder, and a
    # damaged review.json must not stop it.
    sweep(cfg, now)
    trim_ledger(now)
    review_path = STATE / 'review.json'
    try:
        review = json.loads(review_path.read_text())
    except (OSError, ValueError):
        review = {}
    if not isinstance(review, dict):
        review = {}
    last = review.get('last_reminder_at')
    if not isinstance(last, (int, float)) or isinstance(last, bool):
        last = 0
    days = cfg['review_every_days']
    if now - last < days * 86400:
        return None
    report = stats(days)
    write_json(review_path, {**review, 'last_reminder_at': now})
    # Codex has no /context-saver command, so the CLI form comes first.
    context = (f"Context compression hook weekly check due: {report['compressed_outputs']} outputs compressed; "
               f"estimated context tokens saved {report['estimated_context_tokens_saved']}; "
               f"helper total tokens {report['helper_usage']['total_tokens']}. "
               f"Check `context-saver stats --days {days}` (launcher: {ROOT.parent / 'bin' / 'context-saver'}; "
               f"Claude Code: /context-saver stats {days}) and sample original-versus-summary quality "
               f"before claiming savings; record the review with `context-saver review --note=<finding>` "
               f"(Claude Code: /context-saver review <finding>). "
               f"Docs: {ROOT.parent / 'README.md'}.")
    return {'hookSpecificOutput': {'hookEventName': 'SessionStart', 'additionalContext': context}}


def days_argument(value):
    days = int(value)
    if days < 1:
        raise argparse.ArgumentTypeError('must be 1 or more')
    return days


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', nargs='?', default='process', choices=['process', 'pre-tool', 'run', 'session-start', 'stats', 'status', 'enable', 'disable', 'review', 'delivered', 'web-delivered', 'web-bridge', 'limits'])
    # Omitted: all time (period_days null).
    parser.add_argument('--days', type=days_argument)
    parser.add_argument('--note')
    parser.add_argument('--command')
    parser.add_argument('--session-id', default='')
    parser.add_argument('--tool-use-id', default='')
    parser.add_argument('--delivery-token', default='')
    args = parser.parse_args()
    if args.action in ['delivered', 'web-delivered']:
        # The host used the replacement of this token; web-delivered is the
        # older name the Codex web bridge still calls.
        if not DELIVERY_TOKEN.fullmatch(args.delivery_token):
            parser.error('invalid delivery token')
        append_event({'status': 'delivered', 'delivery_token': args.delivery_token, 'saved_tokens': 0})
        print('{"delivered":true}')
    elif args.action == 'limits':
        # The plugin skips the engine for output that can never reach this.
        try:
            print(json.dumps({'minimum_tokens': config()['minimum_tokens']}))
        except ValueError as exc:
            print(json.dumps({'configError': str(exc), 'minimum_tokens': None}))
    elif args.action == 'web-bridge':
        private_dir(STATE)
        source = (ROOT / 'web_bridge.js').read_text()
        for name, value in [('LAUNCHER', str(ROOT.parent / 'bin' / 'context-saver')), ('STATE', str(STATE))]:
            source = source.replace(f"'__CONTEXT_SAVER_{name}__'", json.dumps(value))
        print(source, end='')
    elif args.action == 'run':
        if args.command is None:
            parser.error('--command required for run')
        raise SystemExit(run_command(args.command, args.session_id, args.tool_use_id))
    elif args.action == 'pre-tool':
        output = pre_tool(json.loads(sys.stdin.read(STDIN_LIMIT)))
        if output:
            print(json.dumps(output))
    elif args.action == 'process':
        stop_on_signals()
        event = json.loads(sys.stdin.read(STDIN_LIMIT))
        output = process(event)
        if output:
            print(json.dumps(output, ensure_ascii=False))
    elif args.action == 'session-start':
        # A broken config is reported to the host, never a failed hook.
        try:
            cfg = config()
        except ValueError as exc:
            print(json.dumps({'configError': str(exc)}))
            return
        output = session_start(cfg)
        if output:
            print(json.dumps(output))
    elif args.action == 'stats':
        print(json.dumps(stats(args.days), indent=2))
    elif args.action == 'status':
        cfg = config()
        print(json.dumps({'config': cfg, 'summary_model': f"Claude ({MODEL}), effort {cfg['effort']}",
                          'state': str(STATE), 'config_path': str(CONFIG_PATH), 'python': sys.executable}, indent=2))
    elif args.action in ['enable', 'disable']:
        # Only the switch changes: plugin options in the environment and
        # defaults stay out of the file.
        stored = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        stored['enabled'] = args.action == 'enable'
        write_json(CONFIG_PATH, stored)
        print('enabled' if stored['enabled'] else 'disabled')
    elif args.action == 'review':
        if not args.note:
            parser.error('--note is required')
        report = stats(args.days or 7)
        write_json(STATE / 'review.json', {'last_review_at': time.time(), 'last_reminder_at': time.time(), 'note': args.note[:2000], 'stats': report})
        print('Review recorded.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Runtime hook failures never block or corrupt the original tool result.
        if len(sys.argv) == 1 or sys.argv[1] in ['process', 'pre-tool', 'session-start']:
            with contextlib.suppress(Exception):
                append_event({'status': 'adapter_failed', 'error_type': type(exc).__name__, 'saved_tokens': 0})
        else:
            raise
