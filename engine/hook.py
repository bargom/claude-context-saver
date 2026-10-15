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

from summary_engine import DEFAULT_EFFORT, EFFORTS, MODEL, PROVIDER

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
    'categories': ['search', 'logs', 'git', 'documentation'],
}
SECRET = re.compile(
    r'-----BEGIN (?:[A-Z ]*PRIVATE KEY|OPENSSH PRIVATE KEY)-----|'
    r'\b(?:AIza[\w-]{30,}|AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})\b|'
    r'\b(?i:Bearer)\s+[A-Za-z0-9._~+/-]{12,}|'
    r'(?i:access_token|refresh_token|client_secret|api[_-]?key|password|passwd)\s*["\']?\s*[:=]\s*["\']?[^\s"\',}]{8,}'
)
SENSITIVE_PATH = re.compile(r'(?:^|[/\s"\'])(?:\.env(?:[.\s"\'/]|$)|oauth_creds\.json|credentials\.json|id_rsa|id_ed25519|\.ssh/|\.aws/credentials)')
RAW_MARKER = re.compile(r'CONTEXT_SAVER_RAW\s*=\s*1|context-saver:raw')
JSON_ESCAPE = re.compile(r'\\(?:u([0-9a-fA-F]{4})|(["\\/bfnrt]))')
JSON_CONTROLS = {'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}
# A 4 MiB capture, JSON-escaped inside the event envelope, stays well inside this.
STDIN_LIMIT = 32 * 1024 * 1024
TOKEN_SLICE_CHARS = 2048
ERROR_LINE = re.compile(r'error|fail(?:ed|ure)?|exception|traceback|panic|fatal|assert', re.I)


def secure_dir(path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def save_json(path, value):
    secure_dir(path.parent)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
    os.replace(temporary, path)


def is_flag(value):
    return isinstance(value, bool)


# What config() accepts for each key, from config.json or the plugin options.
VALID = {
    'enabled': is_flag, 'deterministic_first': is_flag, 'rtk_enabled': is_flag,
    'shell_helpers_enabled': is_flag, 'debug_invalid_responses': is_flag,
    'minimum_tokens': lambda x: 1000 <= int(x) <= 50000,
    'maximum_source_chars': lambda x: 1000 <= int(x) <= 60000,
    'maximum_replacement_tokens': lambda x: 200 <= int(x) <= 2000,
    'minimum_saving_fraction': lambda x: .1 <= float(x) < 1,
    'timeout_seconds': lambda x: 5 <= int(x) <= 300,
    'effort': lambda x: x in EFFORTS,
}


def config():
    value = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        value.update(json.loads(CONFIG_PATH.read_text()))
    if os.environ.get(EFFORT_ENV, '').strip():
        value['effort'] = os.environ[EFFORT_ENV].strip()
    if os.environ.get(SHELL_ENV) in ('0', '1'):
        value['shell_helpers_enabled'] = os.environ[SHELL_ENV] == '1'
    for key, valid in VALID.items():
        try:
            ok = valid(value[key])
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f'context-saver config: invalid {key} {value[key]!r:.80}')
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


def first_operand(args, value_options=()):
    """(index, word) of the first non-option word, past the values of value_options."""
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in value_options:
            index += 2
        elif arg.startswith('-'):
            index += 1
        else:
            return index, arg
    return index, None


def git_verb(args):
    return first_operand(args, ['-C', '-c', '--git-dir', '--work-tree', '--namespace'])[1]


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
            while chunk and re.fullmatch(r'[A-Za-z_]\w*=.*', chunk[0]):
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
        index, verb = first_operand(args, ['-C', '-c', '--git-dir', '--work-tree', '--namespace'])
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
    if name in ['web.run', 'web__run']:
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
    if name not in ['Bash', 'exec_command', 'shell', 'shell_command'] or not isinstance(command, str):
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
            return response['output'], {k: response[k] for k in ['exit_code', 'session_id', 'wall_time_seconds'] if k in response}
        if isinstance(response.get('content'), list):
            # Keep images/resources and structured output contracts untouched.
            # Per-block annotations/metadata are contracts too (as in mcp_proxy).
            if response.get('structuredContent') is not None or any(
                    x.get('type') != 'text' or set(x) - {'type', 'text'} for x in response['content'] if isinstance(x, dict)):
                return '', {'unsupported_content': True}
            return '\n'.join(x.get('text', '') for x in response['content'] if isinstance(x, dict)), {'isError': response.get('isError', False)}
        return json.dumps(response, ensure_ascii=False), {}
    return '', {}


def pre_tool(event):
    """Wrap eligible shell commands before execution; works in nested code mode."""
    if not config()['enabled'] or os.environ.get('CONTEXT_SAVER_ACTIVE') == '1':
        return None
    if event.get('hook_event_name') != 'PreToolUse' or event.get('tool_name') != 'Bash':
        return None
    category = classify(event)
    if category not in config()['categories']:
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
    # Through the launcher, so the wrapped run gets the same venv and caches; the
    # caller's errexit/nounset ($-) and pipefail (set +o) go with the command.
    wrapped = ('CONTEXT_SAVER_SHELL="$0" CONTEXT_SAVER_SHELL_FLAGS="$-" CONTEXT_SAVER_SHELL_OPTIONS="$(set +o)" ' +
               shlex.quote(str(ROOT.parent / 'bin' / 'context-saver')) +
               ' run --command ' + shlex.quote(command) +
               ' --session-id ' + shlex.quote(str(event.get('session_id', ''))) +
               ' --tool-use-id ' + shlex.quote(str(event.get('tool_use_id', ''))))
    # Codex requires this PreToolUse value for rewrites; this is not a
    # PermissionRequest approval decision, and sandbox policy still applies.
    return {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'allow',
                                  'updatedInput': {'command': wrapped}}}


def run_command(command, session_id='', tool_use_id=''):
    child_env = os.environ.copy()
    shell = child_env.get('CONTEXT_SAVER_SHELL', '').lstrip('-')
    shell = (shutil.which(shell, path=child_env.get('PATH')) or
             shutil.which(child_env.get('SHELL', ''), path=child_env.get('PATH')) or '/bin/sh')
    child_env['CONTEXT_SAVER_ACTIVE'] = '1'
    command_line = caller_options(child_env.pop('CONTEXT_SAVER_SHELL_FLAGS', ''),
                                  child_env.pop('CONTEXT_SAVER_SHELL_OPTIONS', '')) + command
    # A stop signal unwinds through the replay below; the default handlers would
    # end this wrapper while it holds the only copy of the command's output.
    stop_on_signals()
    try:
        captured = tempfile.TemporaryFile()
    except OSError:
        # No writable temp folder (read-only sandbox): run the command as is.
        return exit_status(run_group([shell, '-c', command_line], env=child_env))
    with captured:
        try:
            returncode = run_group([shell, '-c', command_line], stdout=captured, stderr=subprocess.STDOUT, env=child_env)
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


def caller_options(flags, options):
    """`set` lines that give the fresh shell the caller's errexit, nounset and
    pipefail ($- and `set +o` of the shell that ran the rewritten command)."""
    lines = [f'set -{flag}' for flag in 'eu' if flag in flags]
    if 'set -o pipefail' in options.splitlines():
        lines.append('set -o pipefail')
    return ''.join(line + '\n' for line in lines)


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


def append_event(value):
    secure_dir(STATE)
    value = {'time': time.time(), **value}
    encoded = (json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    fd = os.open(STATE / 'events.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, encoded)
    finally:
        os.close(fd)


def archive(text):
    digest = hashlib.sha256(text.encode()).hexdigest()
    folder = STATE / 'raw'
    secure_dir(folder)
    path = folder / (digest + '.txt')
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as handle:
            handle.write(text)
    path.touch()
    return path


def exact_diagnostics(text, metadata):
    if metadata.get('exit_code', 0) in [0, None] and not metadata.get('isError'):
        return ''
    lines = text.splitlines()
    selected = set(range(max(0, len(lines) - 8), len(lines)))
    for i, line in enumerate(lines):
        if ERROR_LINE.search(line):
            selected.update(range(max(0, i - 1), min(len(lines), i + 2)))
            if len(selected) >= 30:
                break
    return '\n'.join(lines[i] for i in sorted(selected))[:2600]


def label_savings(replacement, before, result):
    """Budget the visible label itself; return only a consistent token count."""
    model = result.get('model') or ''
    if result.get('provider') == PROVIDER:
        producer = f'Claude ({model})' if model else 'Claude'
    else:
        producer = model or result.get('provider') or 'helper'
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
    status = f"exit_code={metadata['exit_code']}" if 'exit_code' in metadata else f"isError={metadata.get('isError', False)}"
    replacement = (
        f"[Context summary via {current.get('model', 'helper')}: {category}; {status}; estimated original tokens={before}]\n"
        f"{current['summary']}\n\n"
        f"Original output: {raw}\n"
        "This is a navigation summary, not exact source; read the original before edits or checking omitted details.\n"
        "To bypass this hook for a shell command, prefix it with CONTEXT_SAVER_RAW=1."
    )
    if coverage is not None and not current.get('local_only'):
        replacement += '\nLocal compaction coverage: ' + json.dumps(coverage, ensure_ascii=False)
    diagnostics = '' if current.get('minimal_archive_notice') or current.get('complete_diagnostics') else exact_diagnostics(text, metadata)
    if diagnostics:
        replacement += '\nExact failure excerpts (original):\n' + diagnostics
    if output_suffix:
        replacement += '\n' + output_suffix
    return label_savings(replacement, before, current)


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
    shell_local_only = (event.get('tool_name') in
                        {'Bash', 'exec_command', 'shell', 'shell_command'} and
                        not cfg.get('shell_helpers_enabled', True))
    # Failed or unfinished shell commands are evidence, not filter candidates.
    if shell_local_only and metadata.get('exit_code') != 0:
        return None
    base = {'category': category, 'tool': event.get('tool_name'), 'session_id': event.get('session_id'),
            'tool_use_id': event.get('tool_use_id'), 'tokenizer': 'o200k_base', 'measurement': 'estimated_tool_payload_only'}
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
            raw = raw or archive(text)
            current = {'status': 'summary', 'summary': candidate['text'],
                       'provider': 'local_compaction', 'model': candidate.get('model', 'deterministic-filter'),
                       'local_compaction': True, 'local_only': True, 'cache_hit': False,
                       'complete_diagnostics': candidate['kind'] == 'jest_success_filter',
                       'usage': dict.fromkeys(('input_tokens', 'output_tokens', 'thinking_tokens',
                                              'cache_read_tokens', 'cache_write_tokens', 'total_tokens'), 0),
                       'elapsed_ms': 0, 'usage_incomplete': False}
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
        raw = archive(text)
        base.update(compaction_kind=prepared['kind'], coverage=prepared['coverage'])

    def local_fallback(previous=None, reason=None):
        previous = previous or {}
        base['coverage'] = prepared['fallback_coverage']
        return {**previous, 'status': 'summary', 'summary': prepared['fallback_text'],
                'provider': 'local_compaction', 'model': 'local-compaction',
                'local_compaction': True, 'local_only': True, 'cache_hit': False,
                'fallback_reason': reason, 'error': None}

    secure_dir(STATE)
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
                arguments = event.get('tool_input') or {}
                context = f"Tool: {event.get('tool_name')}; request: {json.dumps(arguments, ensure_ascii=False)[:1600]}"
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
    raw = raw or archive(text)
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
                  'Read the original file below for exact results and failure details.', 'minimal_archive_notice': True}
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
    delivery_token = event.get('delivery_token', '')
    pending_delivery = (event.get('tool_name') in ['web.run', 'web__run'] and
                        isinstance(delivery_token, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', delivery_token))
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
    delivered = {x.get('delivery_token') for x in rows if x.get('status') == 'web_delivered'}
    compressed = [x for x in rows if x.get('status') == 'compressed' or
                  (x.get('status') == 'compression_candidate' and x.get('delivery_token') in delivered)]
    before = sum(x['before_tokens'] for x in compressed)
    after = sum(x['after_tokens'] for x in compressed)
    usage = {k: sum(int((x.get('helper_usage') or {}).get(k, 0)) for x in rows) for k in
             ['input_tokens', 'output_tokens', 'thinking_tokens', 'cache_read_tokens', 'cache_write_tokens', 'total_tokens']}
    statuses = {}
    for row in rows:
        statuses[row['status']] = statuses.get(row['status'], 0) + 1
    rtk_rows = [x for x in compressed if x.get('model') == 'rtk-stdin-filter']
    return {'period_days': days, 'events': len(rows), 'compressed_outputs': len(compressed),
            'cache_hits': sum(bool(x.get('cache_hit')) for x in compressed), 'statuses': statuses,
            'estimated_before_tokens': before, 'estimated_after_tokens': after,
            'estimated_context_tokens_saved': before - after,
            'estimated_context_reduction_percent': round(100 * (before - after) / before, 4) if before else 0,
            'helper_usage': usage,
            'rtk': {'compressed_outputs': len(rtk_rows),
                    'estimated_context_tokens_saved': sum(x['saved_tokens'] for x in rtk_rows),
                    'helper_tokens': 0},
            'usage_incomplete_events': sum(bool(x.get('usage_incomplete')) for x in rows),
            'claude_list_price_usd': round(sum(float(x.get('list_cost_usd', 0)) for x in rows), 6),
            'estimated_net_tokens_saved_including_helper': before - after - usage['total_tokens'],
            'tokenizer': 'o200k_base', 'billing': 'Claude uses the Claude Code subscription login (claude_list_price_usd is the CLI list-price equivalent, not a charge)',
            'limits': 'Payload estimate only; excludes main-model reasoning, repeated context, follow-up reads and account credit deductions.'}


def session_start():
    cfg = config()
    if not cfg['enabled'] or os.environ.get('CONTEXT_SAVER_ACTIVE') == '1':
        return None
    secure_dir(STATE)
    review_path = STATE / 'review.json'
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    now = time.time()
    # Retention does not wait for the review reminder.
    for path in [*(STATE / 'raw').glob('*.txt'), *(STATE / 'engine' / 'debug').glob('*.txt')]:
        with contextlib.suppress(OSError):
            if now - path.stat().st_mtime > cfg['retention_days'] * 86400:
                path.unlink()
    if now - review.get('last_reminder_at', 0) < cfg['review_every_days'] * 86400:
        return None
    report = stats(cfg['review_every_days'])
    save_json(review_path, {**review, 'last_reminder_at': now})
    context = (f"Context compression hook weekly check due: {report['compressed_outputs']} outputs compressed; "
               f"estimated context tokens saved {report['estimated_context_tokens_saved']}; "
               f"helper total tokens {report['helper_usage']['total_tokens']}. "
               f"Check /context-saver stats (CLI: context-saver stats --days 7) and sample original-versus-summary quality "
               f"before claiming savings; record the review with /context-saver review <finding>. "
               f"Docs: {ROOT.parent / 'README.md'}.")
    return {'hookSpecificOutput': {'hookEventName': 'SessionStart', 'additionalContext': context}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', nargs='?', default='process', choices=['process', 'pre-tool', 'run', 'session-start', 'stats', 'status', 'enable', 'disable', 'review', 'web-delivered', 'web-bridge', 'limits'])
    parser.add_argument('--days', type=int)
    parser.add_argument('--note')
    parser.add_argument('--command')
    parser.add_argument('--session-id', default='')
    parser.add_argument('--tool-use-id', default='')
    parser.add_argument('--delivery-token', default='')
    args = parser.parse_args()
    if args.action == 'web-delivered':
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', args.delivery_token):
            parser.error('invalid delivery token')
        append_event({'tool': 'web.run', 'status': 'web_delivered',
                      'delivery_token': args.delivery_token, 'saved_tokens': 0})
        print('{"delivered":true}')
    elif args.action == 'limits':
        # The plugin skips the engine for output that can never reach this.
        print(json.dumps({'minimum_tokens': config()['minimum_tokens']}))
    elif args.action == 'web-bridge':
        secure_dir(STATE)
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
        output = session_start()
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
        save_json(CONFIG_PATH, stored)
        print('enabled' if stored['enabled'] else 'disabled')
    elif args.action == 'review':
        if not args.note:
            parser.error('--note is required')
        report = stats(args.days or 7)
        save_json(STATE / 'review.json', {'last_review_at': time.time(), 'last_reminder_at': time.time(), 'note': args.note[:2000], 'stats': report})
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
