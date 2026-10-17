"""Small pieces the engine modules share, so the hook, the MCP proxy, the web
adapter and the filters agree on one definition of each."""
import json
import os
from pathlib import Path
import re
import stat
import tempfile

# Shell tools of Claude Code (Bash) and Codex (exec_command, shell, shell_command).
SHELL_TOOLS = frozenset({'Bash', 'exec_command', 'shell', 'shell_command'})
# A leading VAR=value word of a simple command.
ENV_ASSIGNMENT = re.compile(r'[A-Za-z_]\w*=.*')
ANSI = re.compile(r'\x1b\[[0-9;]*m')
# git options that take the next word as their value.
GIT_VALUE_OPTIONS = ('-C', '-c', '--git-dir', '--work-tree', '--namespace')
# Helper usage fields, in ledger and stats order.
USAGE_KEYS = ('input_tokens', 'output_tokens', 'thinking_tokens',
              'cache_read_tokens', 'cache_write_tokens', 'total_tokens')
# A delivery token the host hands back once it used a replacement.
DELIVERY_TOKEN = re.compile(r'[A-Za-z0-9_-]{1,100}')
# Output categories, in config default order.
CATEGORIES = ('search', 'logs', 'git', 'documentation')
# Native web tool of Codex, reached only through the web adapter.
WEB_TOOLS = frozenset({'web.run', 'web__run'})


def exit_code(response):
    """The reported exit code, or None when the host reports none (Claude Code Bash)."""
    code = response.get('exit_code')
    return code if type(code) is int else None


def failed(response):
    """True for a known non-zero exit code or an isError result; an unknown
    exit code is not a failure."""
    code = exit_code(response)
    return (code is not None and code != 0) or bool(response.get('isError'))


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


def text_blocks(content, structured=None):
    """The joined text of an MCP content list made only of plain text blocks,
    or None when anything else is in it: an image, a resource, a block with
    annotations or metadata, structured content, or no blocks at all. Those
    are contracts the caller must pass on unchanged."""
    if structured is not None or not isinstance(content, list) or not content:
        return None
    if any(not isinstance(block, dict) or set(block) != {'type', 'text'} or
           block['type'] != 'text' or not isinstance(block['text'], str) for block in content):
        return None
    return '\n'.join(block['text'] for block in content)


def private_dir(path):
    """Creates the folder if needed and makes it private (0700); refuses a
    link or a folder another user owns."""
    path = Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ValueError(f'unsafe_state_directory: {path}')
    path.chmod(0o700)


def write_json(path, value, *, indent=2):
    """Writes JSON through a private temporary file in the same folder, so a
    reader sees the old or the new file, never half of one."""
    path = Path(path)
    private_dir(path.parent)
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=indent)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
