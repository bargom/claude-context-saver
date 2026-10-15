"""Apply installed RTK stdin filters to selected completed shell results only."""
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

_ANSI = re.compile(r'\x1b\[[0-9;]*m')
_DIAGNOSTIC = re.compile(r'\b(?:error|failed|failure|exception|traceback|panic|fatal|warning|warn)\b', re.I)


def select_filter(text, category, event):
    if category not in ('git', 'logs') or event.get('tool_name') not in ('Bash', 'exec_command', 'shell', 'shell_command'):
        return None
    response = event.get('tool_response')
    if not isinstance(response, dict) or response.get('exit_code') != 0 or response.get('isError'):
        return None
    arguments = event.get('tool_input') or {}
    command = arguments.get('command', arguments.get('cmd', ''))
    # Never infer a parser for pipelines, scripts, substitutions or exact reads.
    if not isinstance(command, str) or re.search(r'[\n\r;&|<>`$()]', command):
        return None
    try:
        args = shlex.split(command)
    except ValueError:
        return None
    while args and re.fullmatch(r'[A-Za-z_]\w*=.*', args[0]):
        args.pop(0)
    if not args or '\x00' in text:
        return None
    head = Path(args.pop(0)).name
    clean = _ANSI.sub('', text)
    diagnostic_text = re.sub(r'^test result: ok\. \d+ passed; 0 failed; \d+ ignored; \d+ measured; \d+ filtered out; finished in [\d.]+s\s*$', '', clean, flags=re.M) if head == 'cargo' else clean
    if clean.lstrip().startswith(('{', '[')) or (category == 'logs' and _DIAGNOSTIC.search(diagnostic_text)):
        return None
    # Machine-readable/custom outputs must not go through a display filter.
    if any(re.match(r'(?:--(?:json|porcelain|format|pretty|reporter|output|junitxml|xml)|-z)(?:=|$)', a) for a in args):
        return None
    if head == 'git':
        while len(args) >= 2 and args[0] == '-C':
            args = args[2:]
        if args == ['status'] and clean.startswith(('On branch ', 'HEAD detached ')):
            return 'git-status'
        if args and args[0] == 'log' and re.match(r'^commit [0-9a-f]{40,64}\n', clean):
            # Default log format only; patch/stat/name output is exact evidence.
            if all(a.isdigit() or re.fullmatch(r'-\d+|--max-count=\d+', a) or a in ('-n', '--max-count') for a in args[1:]):
                return 'git-log'
        return None
    if head in ('vitest', 'npm', 'pnpm', 'yarn', 'bun', 'make') and re.search(r'^\s*Test Files\s+\d+ passed', clean, re.M):
        return 'vitest'
    if head in ('pytest', 'py.test') and re.search(r'^=+ .*\d+ passed.* =+$', clean, re.M):
        return 'pytest'
    if head == 'cargo' and args[:1] == ['test'] and re.search(r'^test result: ok\.', clean, re.M):
        return 'cargo-test'
    return None


def retained_lines(text, selected):
    """Keep every log line except explicitly recognized passing-test records."""
    patterns = {
        'vitest': r'^\s*[✓✔]\s+\S+\s+\(\d+ tests?\)\s+[\d.]+m?s\s*$',
        'pytest': r'^\S+\.py\s+[.s]+\s+\[\s*\d+%\]\s*$',
        'cargo-test': r'^test\s+\S+\s+\.\.\.\s+ok\s*$',
    }
    pattern = patterns.get(selected)
    if pattern is None:
        return ''
    return '\n'.join(line for line in text.splitlines()
                     if line.strip() and not re.fullmatch(pattern, _ANSI.sub('', line)))


def candidate(text, category, event, cfg):
    if not cfg.get('rtk_enabled', False):
        return None
    selected = select_filter(text, category, event)
    # An empty rtk_binary means `rtk` on PATH.
    binary = cfg.get('rtk_binary') or shutil.which('rtk') or ''
    if not selected or not isinstance(binary, str) or not Path(binary).is_absolute() or not os.access(binary, os.X_OK):
        return None
    # pipe is a fixed built-in filter: it never executes the original command.
    # Avoid project-local filters; pipe does not execute project commands.
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(Path.home()), 'LANG': 'en_US.UTF-8',
           'RTK_TELEMETRY_DISABLED': '1', 'DO_NOT_TRACK': '1'}
    try:
        result = subprocess.run([binary, 'pipe', '--filter', selected], input=text,
                                text=True, capture_output=True, timeout=2,
                                cwd='/', env=env)
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None
    if result.returncode or result.stderr.strip() or not result.stdout.strip() or len(result.stdout) >= len(text):
        return None
    body = (f'RTK stdin filter: {selected}; display details may be omitted; '
            'the complete original is archived below.\n' + result.stdout.strip())
    retained = retained_lines(text, selected)
    if retained:
        body += '\nOther original lines (exact):\n' + retained
    return {'text': body, 'kind': 'rtk_' + selected,
            'model': 'rtk-stdin-filter', 'coverage': {'filter': selected,
            'original_chars': len(text), 'filtered_chars': len(result.stdout),
            'details_may_be_omitted': True}}
