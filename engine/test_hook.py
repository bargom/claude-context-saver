import json
import os
import re
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import hook


def fake_claude(folder):
    """A `claude` on PATH that writes its pid and sleeps: a summary helper that
    is still running. Returns the environment that finds it and the pid file."""
    pid_file = Path(folder) / 'helper.pid'
    fake = Path(folder) / 'bin' / 'claude'
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\necho $$ > {shlex.quote(str(pid_file))}\nexec sleep 60\n')
    fake.chmod(0o755)
    return {**os.environ, 'PATH': f'{fake.parent}{os.pathsep}{os.environ["PATH"]}'}, pid_file


def wait_for_pid(pid_file, timeout=10):
    deadline = time.monotonic() + timeout
    while not (pid_file.exists() and pid_file.read_text().strip()) and time.monotonic() < deadline:
        time.sleep(.05)
    return int(pid_file.read_text())


def is_gone(pid, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(.05)
    return False


class HookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)
        self.state_patch = patch.object(hook, 'STATE', self.state)
        self.state_patch.start()
        # Never read the developer's own config.json.
        self.config_patch = patch.object(hook, 'CONFIG_PATH', self.state / 'config.json')
        self.config_patch.start()
        self.cfg = dict(hook.DEFAULTS, minimum_tokens=1000)

    def tearDown(self):
        self.config_patch.stop()
        self.state_patch.stop()
        self.tmp.cleanup()

    def event(self, command='rg -n handler src', text=None):
        return {'hook_event_name': 'PostToolUse', 'tool_name': 'Bash', 'session_id': 'test',
                'tool_use_id': 'call1', 'tool_input': {'command': command},
                'tool_response': {'output': text or ('src/a.js:12: function handler() { return 1; }\n' * 300), 'exit_code': 0}}

    def engine(self, **kwargs):
        return {'status': 'summary', 'summary': 'src/a.js:12 defines handler returning 1.', 'cache_hit': False,
                'usage': {'input_tokens': 2000, 'output_tokens': 100, 'total_tokens': 2100}, 'elapsed_ms': 1}

    def test_tokens_match_exact_counts_for_short_text(self):
        import tiktoken
        encoding = tiktoken.get_encoding('o200k_base')
        for text in ['', 'src/a.py:12: ordinary output\nTests passed.\n', 'café 東京 🧬',
                     '<|endoftext|>', ('ordinary output\n' * 200)[:hook.TOKEN_SLICE_CHARS - 1]]:
            with self.subTest(text=text[:40]):
                self.assertEqual(hook.tokens(text), len(encoding.encode(text, disallowed_special=())))
                self.assertLessEqual(hook.tokens(text), len(text.encode('utf-8')))

    def test_tokens_bound_large_repetitive_input(self):
        text = 'x' * 4_000_000
        started = time.monotonic()
        count = hook.tokens(text)
        self.assertLess(time.monotonic() - started, 5)
        self.assertGreater(count, 0)
        self.assertLessEqual(count, len(text.encode('utf-8')))

    def test_tokens_never_exceed_utf8_bytes_across_slices(self):
        for text in ['café 東京 🧬\n' * 1000, 'x' * (hook.TOKEN_SLICE_CHARS - 1) + '🧬é東京']:
            with self.subTest(text=text[:40]):
                self.assertLessEqual(hook.tokens(text), len(text.encode('utf-8')))

    def test_compresses_and_accounts_raw_retrieval(self):
        out = hook.process(self.event(), self.cfg, self.engine)
        self.assertFalse(out['continue'])
        self.assertIn('Original output:', out['stopReason'])
        self.assertEqual(len(list((self.state / 'raw').glob('*.txt'))), 1)
        report = hook.stats()
        self.assertEqual(report['compressed_outputs'], 1)
        self.assertEqual(report['estimated_context_tokens_saved'], report['estimated_before_tokens'] - report['estimated_after_tokens'])
        self.assertEqual(report['helper_usage']['total_tokens'], 2100)

    def test_provider_and_cache_labels_match_complete_payload_ledger(self):
        cases = [('cursor_cli', 'composer-2.5', False, 'Cursor (composer-2.5)'),
                 ('cursor_cli', 'composer-2.5', True, 'Cursor (composer-2.5)'),
                 ('openai_codex', 'gpt-5.6-sol', False, 'Sol'),
                 ('openai_codex', 'gpt-5.6-sol', True, 'Sol'),
                 ('claude_cli', 'claude-sonnet-5', False, 'Claude (claude-sonnet-5)'),
                 ('claude_cli', 'claude-sonnet-5', True, 'Claude (claude-sonnet-5)'),
                 ('other', 'custom-model', False, 'custom-model'),
                 ('other', None, False, 'other')]
        for provider, model, cached, producer in cases:
            with self.subTest(provider=provider, cached=cached, model=model):
                result = dict(self.engine(), provider=provider, model=model, cache_hit=cached,
                              status='cache_hit' if cached else 'summary')
                event = self.event('npm test')
                event['tool_response']['exit_code'] = 1
                suffix = 'Exact sources: https://example.com [wordlim 200]'
                output = hook.process(event, self.cfg, lambda **kw: result, output_suffix=suffix)['stopReason']
                row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
                self.assertTrue(output.startswith(f"[{row['saved_tokens']} tokens saved by {producer}]"))
                self.assertIn('estimated context tokens', output.splitlines()[0])
                self.assertEqual('cache hit' in output.splitlines()[0], cached)
                self.assertEqual(row['cache_hit'], cached)
                self.assertEqual(row['after_tokens'], hook.tokens(output))
                self.assertEqual(row['saved_tokens'], hook.tokens(event['tool_response']['output']) - hook.tokens(output))
                self.assertIn('Exact failure excerpts (original):', output)
                self.assertTrue(output.endswith(suffix))

    def test_label_cost_is_included_at_acceptance_boundaries(self):
        output = hook.process(self.event(), self.cfg, self.engine)['stopReason']
        after = hook.tokens(output)
        without_label = hook.tokens(output.split('\n', 1)[1])
        self.assertLess(without_label, after)
        before = hook.tokens(self.event()['tool_response']['output'])
        limits = [dict(maximum_replacement_tokens=after - 1),
                  dict(minimum_saving_fraction=(before - (after + without_label) / 2) / before)]
        for limit in limits:
            with self.subTest(limit=limit):
                self.assertIsNone(hook.process(self.event(), dict(self.cfg, **limit), self.engine))
                row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
                self.assertEqual(row['status'], 'insufficient_saving')
                self.assertEqual(row['candidate_after_tokens'], after)
                self.assertEqual(row['saved_tokens'], 0)
        self.assertIsNotNone(hook.process(self.event(), dict(self.cfg, maximum_replacement_tokens=after), self.engine))

    def test_label_digit_boundary_and_nonconvergent_accounting(self):
        # Force iteration into a cycle (1000 -> 999 -> 1000), with a nearby
        # valid fixed point at 998 that the bounded search must find.
        def count(text):
            match = re.match(r'\[(\d+) tokens saved', text)
            if not match:
                return 1000
            saved = int(match[1])
            return {1000: 1001, 999: 1000, 998: 1002}.get(saved, 1000)
        with patch.object(hook, 'tokens', side_effect=count):
            labelled, after, saved = hook.label_savings('payload', 2000, self.engine())
            self.assertEqual(saved, 998)
            self.assertEqual(after, 1002)
            self.assertTrue(labelled.startswith('[998 tokens saved'))

        def no_fixed_point(text):
            match = re.match(r'\[(\d+) tokens saved', text)
            return 1001 if match and int(match[1]) == 1000 else 1000
        with patch.object(hook, 'tokens', side_effect=no_fixed_point):
            self.assertIsNone(hook.label_savings('payload', 2000, self.engine()))
        with patch.object(hook, 'label_savings', return_value=None):
            self.assertIsNone(hook.process(self.event(), self.cfg, self.engine))
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
        self.assertEqual(row['reason'], 'label_accounting_unstable')
        self.assertEqual(row['saved_tokens'], 0)
        self.assertEqual(hook.stats()['compressed_outputs'], 0)

    def test_small_exact_reads_and_bypass_pass_through(self):
        for command in ['cat src/a.js', 'sed -n 1,200p src/a.js', 'CONTEXT_SAVER_RAW=1 rg handler src']:
            self.assertIsNone(hook.process(self.event(command), self.cfg, self.engine))
        self.assertIsNone(hook.process(self.event(text='small'), self.cfg, self.engine))
        e = self.event();e['tool_name'] = 'mcp__codebase_memory_mcp__get_code_snippet'
        self.assertIsNone(hook.process(e, self.cfg, self.engine))

    def test_sensitive_and_oversize_not_sent(self):
        def never(**kwargs):
            self.fail('engine must not receive excluded output')
        self.assertIsNone(hook.process(self.event(text='api_key="' + 'secret12345' * 400 + '"'), self.cfg, never))
        self.assertIsNone(hook.process(self.event('rg thing .env'), self.cfg, never))
        out = hook.process(self.event(text='long line\n' * 20000), self.cfg, never)
        self.assertIn('tokens saved by local compaction', out['stopReason'])
        self.assertLessEqual(hook.tokens(out['stopReason']), self.cfg['maximum_replacement_tokens'])

    def test_oversized_repetitive_output_finishes_quickly(self):
        def never(**kwargs):
            self.fail('preview needs no helper')
        event = self.event(text='x' * 300_000)
        started = time.monotonic()
        out = hook.process(event, self.cfg, never)
        self.assertLess(time.monotonic() - started, 5)
        self.assertIn('tokens saved by local compaction', out['stopReason'])
        self.assertIn('ARCHIVE PREVIEW', out['stopReason'])
        self.assertLessEqual(hook.tokens(out['stopReason']), self.cfg['maximum_replacement_tokens'])
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
        self.assertEqual(Path(row['raw_file']).read_text(), event['tool_response']['output'])
        self.assertEqual(row['saved_tokens'], hook.tokens(event['tool_response']['output']) - hook.tokens(out['stopReason']))

    def test_bearer_schemes_are_never_sent(self):
        def never(**kwargs):
            self.fail('engine must not receive excluded output')
        for scheme in ('bearer', 'BEARER'):
            with self.subTest(scheme=scheme):
                text = self.event()['tool_response']['output'] + (
                    f'Authorization: {scheme} eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijkl\n')
                self.assertIsNone(hook.process(self.event(text=text), self.cfg, never))

    def test_failed_helper_and_structured_mcp_preserve_result(self):
        self.assertIsNone(hook.process(self.event(), self.cfg, lambda **kw: {'status': 'error', 'error': 'timeout'}))
        e = self.event();e['tool_name'] = 'mcp__codebase_memory_mcp__search_graph'
        e['tool_response'] = {'content': [{'type': 'text', 'text': 'x' * 20000}], 'structuredContent': {'value': 'exact'}}
        self.assertIsNone(hook.process(e, self.cfg, self.engine))

    def oversized_graph_event(self):
        rows = [{'name': f'handler_{i}', 'label': 'Function', 'file_path': f'src/worker_{i}.js',
                 'qualified_name': 'project.' + 'shared_package.' * 45 + f'handler_{i}',
                 'signature': '(input)', 'lines': 12, 'complexity': 1,
                 'docstring': 'Recorded declaration only.'} for i in range(150)]
        text = json.dumps({'total': 150, 'results': rows, 'has_more': False})
        event = self.event(text=text)
        event.update(tool_name='mcp__codebase_memory_mcp__search_graph',
                     tool_input={'name_pattern': 'handler.*'})
        return event

    def test_oversized_graph_compacts_every_identity_before_helper(self):
        event = self.oversized_graph_event()
        def helper(**kwargs):
            self.assertLessEqual(len(kwargs['text']), self.cfg['maximum_source_chars'])
            self.assertIn('handler_149', kwargs['text'])
            self.assertIn('line_count', kwargs['text'])
            return dict(self.engine(), provider='claude_cli', model='sonnet')
        output = hook.process(event, self.cfg, helper)['stopReason']
        self.assertIn('tokens saved by Claude (sonnet) + local compaction', output)
        self.assertIn('Local compaction coverage:', output)
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
        self.assertEqual(row['coverage']['shown'], 150)
        self.assertEqual(Path(row['raw_file']).read_text(), event['tool_response']['output'])
        self.assertEqual(row['saved_tokens'], hook.tokens(event['tool_response']['output']) - hook.tokens(output))

    def test_oversized_helper_failure_and_overbudget_use_bounded_archive_preview(self):
        event = self.oversized_graph_event()
        for failure in [dict(self.engine(), status='error', error='model_timeout'),
                        dict(self.engine(), summary='excessive output ' * 3000)]:
            with self.subTest(status=failure['status']):
                output = hook.process(event, self.cfg, lambda **kw: failure)['stopReason']
                self.assertIn('tokens saved by local compaction', output)
                self.assertNotIn('tokens saved by Claude', output)
                self.assertIn('ARCHIVE PREVIEW', output)
                self.assertLessEqual(hook.tokens(output), self.cfg['maximum_replacement_tokens'])
                row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
                self.assertLess(row['coverage']['shown'], 150)
                self.assertEqual(row['helper_usage']['total_tokens'], 2100)
                self.assertEqual(Path(row['raw_file']).read_text(), event['tool_response']['output'])

    def test_oversized_unknown_structured_contract_stays_raw(self):
        event = self.event(text=json.dumps({'machine_contract': 'value' * 15000}))
        self.assertIsNone(hook.process(event, self.cfg, lambda **kw: self.fail('must not summarize contract')))

    def test_oversized_token_dense_failure_stays_bounded_and_reports_omission(self):
        event = self.event('npm test', text='ERROR ' + '\U0001f9ec' * 60001)
        event['tool_response']['exit_code'] = 1
        out = hook.process(event, self.cfg, lambda **kw: self.fail('preview needs no helper'))['stopReason']
        self.assertIn('exit_code=1', out)
        self.assertIn('preview omitted', out)
        self.assertLessEqual(hook.tokens(out), 1800)
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
        self.assertEqual(row['coverage']['shown_chars'], 0)
        self.assertEqual(row['coverage']['omitted_chars'], len(event['tool_response']['output']))

    def test_graph_notice_only_records_zero_visible_rows(self):
        event = self.oversized_graph_event()
        out = hook.process(event, dict(self.cfg, maximum_replacement_tokens=300),
                           lambda **kw: {'status': 'error', 'error': 'model_timeout'})['stopReason']
        self.assertIn('preview omitted', out)
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[-1])
        self.assertEqual(row['coverage']['shown'], 0)
        self.assertEqual(row['coverage']['remaining'], 150)

    def test_preserves_exit_status_and_exact_failure(self):
        e = self.event('npm test');e['tool_response']['exit_code'] = 1
        e['tool_response']['output'] += '\nERROR exact failure marker 123\n'
        out = hook.process(e, self.cfg, self.engine)
        self.assertIn('exit_code=1', out['stopReason'])
        self.assertIn('ERROR exact failure marker 123', out['stopReason'])

    def test_classifies_tool_categories(self):
        for cmd, category in [('git diff --stat', 'git'), ('pytest -q', 'logs'), ('kubectl logs pod', 'logs'), ('rg -n needle src', 'search')]:
            self.assertEqual(hook.classify(self.event(cmd)), category)
        self.assertIsNone(hook.classify(self.event('python3 -c "print(\'pytest\')"')))

    def test_quoted_credentials_in_arguments_are_never_sent(self):
        self.assertIsNone(hook.classify(self.event('rg -l \'password="supersecret123"\' src')))
        self.assertEqual(hook.classify(self.event('rg -l \'password\' src')), 'search')

    def test_reads_through_command_runners_and_mixed_operands_stay_exact(self):
        for cmd in ['rg -l handler src | xargs cat', 'find src -type f -exec cat {} \\;',
                    "rg -n x src; sh -c 'head -n 50 src/a.py'", 'cat src/app.py build.log']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd, category in [('cat build.log test.log', 'logs'), ('rg -l handler src | wc -l', 'search')]:
            self.assertEqual(hook.classify(self.event(cmd)), category, cmd)

    def test_only_known_commands_and_their_filters_are_compressed(self):
        for cmd in ['rg -l handler src | xargs wc -l', 'echo start; rg -n x src',
                    "python3 -c 'print(open(\"src/a.py\").read())'; rg -n x src",
                    'fd -e py -x cat', 'rg -n x src | sort -o out.txt src/a.py', 'git grep -n handler']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd, category in [('cd src && rg -n x .', 'search'), ('pytest -q 2>&1 | tail -n 50', 'logs'),
                              ('rg -n x src | sort | uniq -c | tr a-z A-Z', 'search')]:
            self.assertEqual(hook.classify(self.event(cmd)), category, cmd)

    def test_named_file_reads_after_a_pipe_stay_exact(self):
        for cmd in ['rg handler src | head -n 100 src/file.py', 'rg handler src | cat src/a.py',
                    'rg handler src | sed -n 1,50p src/a.py', 'rg handler src | cat -- -notes.py',
                    'rg handler src | uniq -c src/a.py']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd in ['rg handler src | head -n 100', 'rg handler src | cat -']:
            self.assertEqual(hook.classify(self.event(cmd)), 'search', cmd)

    def test_redirected_input_after_a_pipe_stays_exact(self):
        for cmd in ["rg handler src | tr -d '\\r' < src/app.py", 'rg x src | sort < src/a.py',
                    'rg x src | head -n 5 <<< "$(true)"', 'rg x src | head -n 5 <<< "-source"',
                    'rg x src | cat <<EOF', 'rg x src | cat <<EOF\nsource\nEOF',
                    'rg x src | tr a-z A-Z <> src/app.py', 'rg x src | tr a-z A-Z 0<src/app.py',
                    'rg x src | tr a-z A-Z <&3', 'rg x src | head -n 5 0<&3',
                    'rg x src | tr a-z A-Z <<EOF', 'rg x src | cat < app.log',
                    'cat < src/app.py', 'head -n 5 < src/app.py', 'cat app.log < src/app.py']:
            with self.subTest(command=cmd):
                self.assertIsNone(hook.classify(self.event(cmd)))

    def test_plain_filters_and_output_redirections_keep_their_categories(self):
        for cmd, category in [('rg x src | tr a-z A-Z', 'search'),
                              ('pytest -q 2>&1 | tail -n 50', 'logs'),
                              ('rg x src 2>/dev/null | head -50', 'search'),
                              ('rg x src > matches.txt', 'search'), ('rg x src >> matches.txt', 'search'),
                              ('rg x src &> matches.txt', 'search'), ('cat < app.log', 'logs'),
                              ('rg x < src/app.py | head -50', 'search')]:
            with self.subTest(command=cmd):
                self.assertEqual(hook.classify(self.event(cmd)), category)

    def test_build_and_container_tools_count_only_for_their_real_subcommand(self):
        for cmd in ['docker exec logs cat /app/src/main.py', 'cargo run -- test', 'go run . test', 'npm run dev',
                    'kubectl exec logs -- cat src/a.py', 'git -c core.pager=cat log',
                    'git diff --no-index /dev/null src/a.py', 'rg x src | sort --files0-from=list']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd, category in [('npm run test:unit', 'logs'), ('pnpm -C web build', 'logs'), ('make -C src test', 'logs'),
                              ('docker compose -f dev.yml logs api', 'logs'), ('kubectl -n prod logs pod/a', 'logs'),
                              ('git --no-pager log -5', 'git'), ('rg x src | sort -rn | head --lines=20', 'search')]:
            self.assertEqual(hook.classify(self.event(cmd)), category, cmd)

    def test_fd_running_a_command_on_matches_stays_exact(self):
        for cmd in ['fd -xcat', 'fd -e py -Xcat', 'fd -HIxcat', 'fd --exec=cat', 'fd -e py -x cat']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        self.assertEqual(hook.classify(self.event('fd -e py handler src')), 'search')

    def test_command_substitutions_keep_the_output_exact(self):
        for cmd in ['git status `cat src/main.py >&2`', 'git status "$(cat src/main.py >&2)"', 'rg "`cat q.txt`" src',
                    'git status "$\\\n(cat src/main.py >&2)"']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd in ["rg '`code`' docs", 'rg "\\`x" src', "rg '$(x)' src"]:
            self.assertEqual(hook.classify(self.event(cmd)), 'search', cmd)

    def test_a_comment_keeps_the_next_line_a_separate_command(self):
        for cmd in ['rg handler src # find callers\ncat src/config.py', 'grep foo src # "\ncat source.py\n# "',
                    "rg x src # '\ncat source.py\n# '", 'rg x src # \\\ncat source.py',
                    'rg x src;# "\ncat source.py\n# "', '# "\ncat source.py\nrg x src',
                    'rg x src \\\n# "\ncat source.py\n# "']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd in ['rg handler src # find callers', "rg 'C#' src", 'rg C# src', 'rg a#b src',
                    "rg x src # it's fine", 'rg x src # "', 'rg x src # $(cat source.py) `cat source.py`',
                    'rg x src # \\', 'rg "#" src', 'rg \\# src', 'rg x src;# "\nrg y src',
                    'rg x src\\\n# "\ncat source.py\n# "']:
            self.assertEqual(hook.classify(self.event(cmd)), 'search', cmd)
        e = self.event('rg handler src # find callers\nrm -rf target');e['hook_event_name'] = 'PreToolUse'
        self.assertIsNone(hook.pre_tool(e))

    def test_quoted_or_escaped_word_boundaries_do_not_start_comments(self):
        for prefix in ['"x "', 'x\\ ', 'x\\|', '\\', '""']:
            cmd = 'rg ' + prefix + '# "\ncat source.py\n# "'
            self.assertEqual(hook.classify(self.event(cmd)), 'search', cmd)

    def test_unfinished_shell_quotes_and_escapes_stay_exact(self):
        for cmd in ["rg 'x # comment", 'rg "x # comment', 'rg x src \\']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)

    def test_credentials_inside_json_output_are_never_sent(self):
        def never(**kwargs):
            self.fail('engine must not receive output that holds a credential')
        padding = [{'path': f'src/m{i}.py', 'line': i, 'text': f'def handler_{i}(): return {i}'} for i in range(400)]
        for hidden in ['password = "supersecret123"', 'token in config/.env"', '"client_secret": "abcdefgh12345"']:
            snippet = json.dumps({'results': padding + [{'path': 'src/settings.py', 'text': hidden}]})
            for text in [snippet, json.dumps({'body': snippet})]:
                e = self.event();e['tool_name'] = 'mcp__codebase_memory_mcp__search_graph'
                e['tool_response'] = {'content': [{'type': 'text', 'text': text}]}
                self.assertIsNone(hook.process(e, self.cfg, never), hidden)
        events = [json.loads(line) for line in (self.state / 'events.jsonl').read_text().splitlines()]
        self.assertEqual({event['status'] for event in events}, {'skipped_sensitive'})
        self.assertEqual(len(events), 6)

    def test_git_blob_reads_stay_exact(self):
        for cmd in ['git show HEAD:src/main.py', 'git show main:README.md | head -50']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd in ['git show HEAD --stat', 'git show --format=%H:%s HEAD']:
            self.assertEqual(hook.classify(self.event(cmd)), 'git', cmd)

    def test_file_reads_next_to_a_search_stay_exact(self):
        for cmd in ['git status --short; cat src/main.py', 'rg -n x src && head -n 80 src/a.js',
                    'cd src && sed -n 1,200p main.py; git diff', 'rg -n handler src\ncat src/app.py']:
            self.assertIsNone(hook.classify(self.event(cmd)), cmd)
        for cmd, category in [('rg -n x src | head -50', 'search'), ('git log --oneline | tail -n 20', 'git'),
                              ('tail -n 400 app.log', 'logs'), ("rg -n 'a\nb' src", 'search')]:
            self.assertEqual(hook.classify(self.event(cmd)), category, cmd)

    def test_annotated_mcp_text_blocks_stay_unchanged(self):
        e = self.event();e['tool_name'] = 'mcp__codebase_memory_mcp__search_graph'
        text = 'src/a.js:12: function handler() { return 1; }\n' * 300
        e['tool_response'] = {'content': [{'type': 'text', 'text': text}]}
        self.assertIsNotNone(hook.process(e, self.cfg, self.engine))
        for extra in [{'annotations': {'audience': ['user']}}, {'_meta': {'source': 'index'}}]:
            e['tool_response'] = {'content': [{'type': 'text', 'text': text, **extra}]}
            self.assertIsNone(hook.process(e, self.cfg, self.engine), extra)

    def test_timeout_config_accepts_five_minutes_and_rejects_more(self):
        path = self.state / 'config.json'
        with patch.object(hook, 'CONFIG_PATH', path):
            self.assertEqual(hook.config()['timeout_seconds'], 300)
            path.write_text(json.dumps({'timeout_seconds': 300}))
            self.assertEqual(hook.config()['timeout_seconds'], 300)
            path.write_text(json.dumps({'timeout_seconds': 301}))
            with self.assertRaises(AssertionError):
                hook.config()

    def test_web_classification_does_not_claim_native_transport(self):
        event = {'tool_name': 'web.run', 'tool_input': {'search_query': [{'q': 'topic'}]}}
        self.assertEqual(hook.classify(event), 'search')
        event['hook_event_name'] = 'PreToolUse'
        self.assertIsNone(hook.pre_tool(event))
        for key in ['finance', 'weather', 'sports', 'screenshot', 'image_query', 'time', 'unknown']:
            event['tool_input'] = {'search_query': [{'q': 'topic'}], key: [{}]}
            self.assertIsNone(hook.classify(event))

    def test_transport_provenance_counts_towards_savings_and_ledger(self):
        suffix = 'Exact sources: https://example.com [wordlim 200]'
        out = hook.process(self.event(), self.cfg, self.engine, output_suffix=suffix)
        self.assertTrue(out['stopReason'].endswith(suffix))
        self.assertEqual(hook.stats()['estimated_after_tokens'], hook.tokens(out['stopReason']))
        out = hook.process(self.event(), self.cfg, self.engine, output_suffix='source ' * 3000)
        self.assertIsNone(out)
        self.assertEqual(hook.stats()['compressed_outputs'], 1)

    def test_web_savings_require_delivery_ack_without_double_counting_usage(self):
        event = self.event()
        event.update(tool_name='web.run', tool_input={'search_query': [{'q': 'topic'}]}, delivery_token='web_test_1')
        self.assertIsNotNone(hook.process(event, self.cfg, self.engine))
        self.assertEqual(hook.stats()['compressed_outputs'], 0)
        self.assertEqual(hook.stats()['helper_usage']['total_tokens'], 2100)
        for _ in range(2):
            hook.append_event({'status': 'web_delivered', 'delivery_token': 'web_test_1', 'saved_tokens': 0})
        self.assertEqual(hook.stats()['compressed_outputs'], 1)
        self.assertEqual(hook.stats()['helper_usage']['total_tokens'], 2100)

    def test_pre_tool_preserves_original_command_and_skips_compounds(self):
        e = self.event("rg -n 'a b' src");e['hook_event_name'] = 'PreToolUse'
        out = hook.pre_tool(e)['hookSpecificOutput']
        self.assertEqual(out['permissionDecision'], 'allow')
        self.assertIn('--command', out['updatedInput']['command'])
        self.assertIn(str(hook.ROOT.parent / 'bin' / 'context-saver') + ' run', out['updatedInput']['command'])
        for cmd in ['rg hello src; rm -rf target', 'rg hello $(pwd)', 'rg hello src > result.txt', 'git diff; git push', 'git push origin log', 'rg hello src; kubectl delete pod x', 'kubectl logs -f pod', 'npm test --watch', 'rg hello src\nrm -rf target']:
            e['tool_input']['command'] = cmd
            self.assertIsNone(hook.pre_tool(e))

    def test_pre_tool_leaves_following_logs_unwrapped(self):
        commands = ['tail -F app.log', 'journalctl -fu example.service', 'kubectl logs -f pod',
                    'docker logs --follow=true api', 'tail -nF 50 app.log',
                    "docker logs --fo''llow=true api", "journalctl '-fu\nexample.service'"]
        for command in ['tail app.log', 'journalctl', 'docker logs api', 'podman logs api', 'kubectl logs pod']:
            commands.extend(command + ' ' + option for option in ['-f', '-F', '--follow', '--follow=name',
                                                                  '-fu', '-nF', '-fn100', '-nf100'])
        for cmd in commands:
            with self.subTest(command=cmd):
                e = self.event(cmd);e['hook_event_name'] = 'PreToolUse'
                self.assertIsNone(hook.pre_tool(e))
        for cmd in ['tail -n 50 app.log', 'journalctl -u example.service', 'docker logs --tail=50 api',
                    'podman logs api', 'kubectl logs pod', 'rg -f patterns src']:
            e = self.event(cmd);e['hook_event_name'] = 'PreToolUse'
            self.assertIn(' run --command ', hook.pre_tool(e)['hookSpecificOutput']['updatedInput']['command'], cmd)

    def test_pre_tool_leaves_watch_modes_unwrapped(self):
        for cmd in ['tsc -w', 'tsc --watch', 'jest --watchAll', 'vitest -w',
                    'npm test -- --watch', 'pytest -f', 'pytest --looponfail', 'py.test -f',
                    'pnpm test -- --watch=true', "jest --wat''chAll", "tsc '-w'",
                    'tsc --noEmit && pytest -f', 'tsc --noEmit | jest --watchAll', 'rg --watch src']:
            with self.subTest(command=cmd):
                e = self.event(cmd);e['hook_event_name'] = 'PreToolUse'
                self.assertIsNone(hook.pre_tool(e))

    def test_pre_tool_wraps_commands_with_finite_options(self):
        for cmd in ['tsc --noEmit', 'jest -w 4', 'npm test -w pkg', 'pnpm test -w', 'pytest -q',
                    "rg 'contains --watch text' src", 'tsc --noEmit # --watch']:
            with self.subTest(command=cmd):
                e = self.event(cmd);e['hook_event_name'] = 'PreToolUse'
                self.assertIn(' run --command ', hook.pre_tool(e)['hookSpecificOutput']['updatedInput']['command'])

    def test_recursion_guard(self):
        with patch.dict(os.environ, {'CONTEXT_SAVER_ACTIVE': '1'}):
            self.assertIsNone(hook.process(self.event(), self.cfg, self.engine))

    def test_run_preserves_small_output_and_failure(self):
        r = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'run', '--command', 'printf exact-output; exit 7'],
                           capture_output=True, text=True, cwd=self.state)
        self.assertEqual(r.returncode, 7)
        self.assertEqual(r.stdout, 'exact-output')

    @unittest.skipUnless(shutil.which('bash') and Path('/bin/zsh').is_file(), 'bash and /bin/zsh are required')
    def test_run_resolves_the_callers_shell_on_path(self):
        r = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'run', '--command',
                            'printf %s "${BASH_VERSION:+bash}"'],
                           capture_output=True, text=True, cwd=self.state, timeout=30,
                           env={**os.environ, 'CONTEXT_SAVER_SHELL': 'bash', 'SHELL': '/bin/zsh'})
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout, 'bash')

    def test_run_replays_output_when_the_wrapper_is_stopped(self):
        # The command stops its own wrapper ($PPID), as a tool timeout does.
        r = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'run', '--command',
                            'printf partial-output; kill -TERM $PPID; sleep 5'],
                           capture_output=True, text=True, cwd=self.state, timeout=30,
                           env={**os.environ, 'CONTEXT_SAVER_SHELL': '/bin/sh'})
        self.assertEqual(r.returncode, 143)
        self.assertEqual(r.stdout, 'partial-output')

    def test_run_stops_descendants_when_the_wrapper_is_stopped(self):
        self.assert_run_stops_descendants()

    def test_run_stops_descendants_without_a_writable_temp_folder(self):
        self.assert_run_stops_descendants(direct=True)

    def test_run_kills_descendants_that_ignore_sigterm(self):
        for direct in (False, True):
            with self.subTest(direct=direct):
                self.assert_run_stops_descendants(direct=direct, ignore_term=True)

    def test_run_stops_descendants_on_other_stop_signals(self):
        for signum in (signal.SIGINT, signal.SIGHUP):
            with self.subTest(signum=signum):
                self.assert_run_stops_descendants(signum=signum)

    def test_run_stops_descendants_on_keyboard_interrupt(self):
        for direct in (False, True):
            with self.subTest(direct=direct):
                self.assert_run_stops_descendants(direct=direct, signum=signal.SIGINT, keyboard_interrupt=True)

    def assert_run_stops_descendants(self, direct=False, ignore_term=False, signum=signal.SIGTERM,
                                    keyboard_interrupt=False):
        pid_file = self.state / 'sleep.pid'
        pid_file.unlink(missing_ok=True)
        command = ('sleep 30 >/dev/null 2>&1 & ' + f'echo $! > {shlex.quote(str(pid_file))}; ' +
                   f'printf partial-output; kill -{signum.name[3:]} $PPID; wait')
        if ignore_term:
            command = "trap '' TERM; " + command.replace('printf partial-output', 'trap - TERM; printf partial-output')
        argv = [sys.executable, str(hook.ROOT / 'hook.py')]
        if direct or keyboard_interrupt:
            code = "import hook, signal; from unittest.mock import patch; "
            if direct:
                code += "patch.object(hook.tempfile, 'TemporaryFile', side_effect=OSError('read-only')).start(); "
            if keyboard_interrupt:
                code += "patch.object(hook, 'stop_wrapper', signal.default_int_handler).start()\n" + \
                        "try:\n    hook.main()\nexcept KeyboardInterrupt:\n    raise SystemExit(130)"
            else:
                code += 'hook.main()'
            argv = [sys.executable, '-c', code]
        try:
            result = subprocess.run([*argv, 'run', '--command', command],
                                    capture_output=True, text=True, cwd=hook.ROOT, timeout=10,
                                    env={**os.environ, 'CONTEXT_SAVER_SHELL': '/bin/sh'})
            self.assertEqual(result.returncode, 128 + signum, result.stderr)
            self.assertEqual(result.stdout, 'partial-output')
            self.assertTrue(is_gone(wait_for_pid(pid_file)), 'The background sleep survived cancellation')
        finally:
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_wrapped_pipeline_keeps_the_callers_pipefail(self):
        e = self.event('grep -n handler missing.txt | cat');e['hook_event_name'] = 'PreToolUse'
        wrapped = hook.pre_tool(e)['hookSpecificOutput']['updatedInput']['command']
        env = {**os.environ, 'CONTEXT_SAVER_STATE': str(self.state), 'CONTEXT_SAVER_CONFIG': str(self.state / 'config.json')}
        for options, status in [('', 0), ('set -o pipefail; ', 2)]:
            r = subprocess.run(['bash', '-c', options + wrapped], cwd=self.state, env=env,
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, status, options)

    def test_stopping_process_ends_the_running_summary_helper(self):
        env, pid_file = fake_claude(self.state)
        (self.state / 'config.json').write_text(json.dumps({'minimum_tokens': 1000, 'deterministic_first': False}))
        env.update(CONTEXT_SAVER_STATE=str(self.state / 'state'), CONTEXT_SAVER_CONFIG=str(self.state / 'config.json'),
                   CONTEXT_SAVER_MODEL='sonnet')
        with subprocess.Popen([sys.executable, str(hook.ROOT / 'hook.py'), 'process'], env=env, cwd=self.state,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            process.stdin.write(json.dumps(self.event()).encode())
            process.stdin.close()
            helper = wait_for_pid(pid_file)
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=10)
        self.assertTrue(is_gone(helper))

    def test_run_executes_the_command_without_a_writable_temp_folder(self):
        code = "import hook; from unittest.mock import patch; " + \
               "p=patch.object(hook.tempfile, 'TemporaryFile', side_effect=OSError('read-only')); p.start(); " + \
               "raise SystemExit(hook.run_command('printf direct-output; exit 3'))"
        result = subprocess.run([sys.executable, '-c', code], cwd=hook.ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stdout, 'direct-output')

    def test_run_replays_output_when_stopped_while_compressing(self):
        code = "import hook, os, signal; from unittest.mock import patch; " + \
               "p=patch.object(hook, 'process', side_effect=lambda *a, **k: os.kill(os.getpid(), signal.SIGTERM)); p.start(); " + \
               "raise SystemExit(hook.run_command('printf finished-output'))"
        result = subprocess.run([sys.executable, '-c', code], cwd=hook.ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 143)
        self.assertEqual(result.stdout, 'finished-output')

    def test_weekly_reminder_is_throttled(self):
        first = hook.session_start()
        self.assertIn('weekly check due', first['hookSpecificOutput']['additionalContext'])
        self.assertIsNone(hook.session_start())

    def test_weekly_check_prunes_old_debug_responses_with_raw_archive(self):
        debug = self.state / 'engine' / 'debug'
        hook.secure_dir(debug)
        old, recent = debug / 'invalid-response-old.txt', debug / 'invalid-response-new.txt'
        for path in (old, recent):
            path.write_text('rejected helper output')
        stale = time.time() - 8 * 86400
        os.utime(old, (stale, stale))
        hook.session_start()
        self.assertFalse(old.exists())
        self.assertTrue(recent.exists())

    def test_retention_runs_when_the_reminder_is_not_due(self):
        hook.save_json(self.state / 'review.json', {'last_reminder_at': time.time()})
        raw = self.state / 'raw'
        hook.secure_dir(raw)
        old = raw / 'old.txt'
        old.write_text('original')
        stale = time.time() - 8 * 86400
        os.utime(old, (stale, stale))
        self.assertIsNone(hook.session_start())
        self.assertFalse(old.exists())

    def test_debug_flag_must_be_boolean(self):
        path = self.state / 'config.json'
        with patch.object(hook, 'CONFIG_PATH', path):
            self.assertIs(hook.config()['debug_invalid_responses'], False)
            path.write_text(json.dumps({'debug_invalid_responses': 'yes'}))
            with self.assertRaises(AssertionError):
                hook.config()

    def test_provider_accounting_and_cached_credit_rate(self):
        def backup(**kwargs):
            return dict(self.engine(), provider='openai_codex', model='gpt-5.6-sol',
                        per_provider_usage={'openai_codex': {'input_tokens': 2000, 'cache_read_tokens': 1000,
                                                            'output_tokens': 100, 'total_tokens': 2100}})
        hook.process(self.event(), self.cfg, backup)
        report = hook.stats()
        self.assertEqual(report['per_provider_usage']['openai_codex']['total_tokens'], 2100)
        self.assertEqual(report['estimated_sol_subscription_credits'], 0.16)

    def test_configured_helpers_reach_engine_and_claude_cost_is_reported(self):
        seen = {}
        chain = [{'provider': 'claude_cli', 'model': 'claude-sonnet-5', 'effort': 'low'}]
        def engine(**kwargs):
            seen.update(kwargs)
            sol = {'input_tokens': 2000, 'cache_read_tokens': 1000, 'output_tokens': 100, 'total_tokens': 2100}
            return dict(self.engine(), provider='claude_cli', model='claude-sonnet-5', list_cost_usd=0.0684,
                        per_provider_usage={'claude_cli:claude-sonnet-5': dict(sol), 'openai_codex:gpt-5.6-sol': sol})
        hook.process(self.event(), dict(self.cfg, helpers=chain), engine)
        self.assertEqual(seen['helpers'], chain)
        hook.process(self.event('rg -n other src'), self.cfg, lambda **kw: seen.update(default=kw['helpers']) or self.engine())
        self.assertIsNone(seen['default'])
        row = json.loads((self.state / 'events.jsonl').read_text().splitlines()[0])
        self.assertEqual(row['list_cost_usd'], 0.0684)
        report = hook.stats()
        self.assertEqual(report['claude_list_price_usd'], 0.0684)
        self.assertEqual(report['estimated_sol_subscription_credits'], 0.16)
        self.assertEqual(report['per_provider_usage']['claude_cli:claude-sonnet-5']['total_tokens'], 2100)

    def test_cursor_usage_and_cache_counts_are_separate_from_sol_credits(self):
        usage = {'input_tokens': 24000, 'output_tokens': 400, 'cache_read_tokens': 18000,
                 'cache_write_tokens': 1000, 'total_tokens': 24400}
        result = dict(self.engine(), provider='cursor_cli', model='composer-2.5', usage=usage,
                      per_provider_usage={'cursor_cli': usage})
        hook.process(self.event(), self.cfg, lambda **kwargs: result)
        report = hook.stats()
        self.assertEqual(report['per_provider_usage']['cursor_cli']['total_tokens'], 24400)
        self.assertEqual(report['helper_usage']['cache_write_tokens'], 1000)
        self.assertEqual(report['helper_usage']['cache_read_tokens'], 18000)
        self.assertEqual(report['estimated_sol_subscription_credits'], 0)
        self.assertEqual(report['estimated_net_tokens_saved_including_helper'],
                         report['estimated_context_tokens_saved'] - 24400)

    def test_run_replays_output_if_config_fails_after_command(self):
        code = "import hook; from unittest.mock import patch; " + \
               "p=patch.object(hook, 'config', side_effect=ValueError('bad config')); p.start(); " + \
               "raise SystemExit(hook.run_command('printf still-exact; exit 7'))"
        result = subprocess.run([sys.executable, '-c', code], cwd=hook.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, 'still-exact')



class PluginConfigTests(unittest.TestCase):
    """The Claude Code plugin passes its /config options as environment variables."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp.name) / 'config.json'
        self.config_patch = patch.object(hook, 'CONFIG_PATH', self.config_path)
        self.config_patch.start()

    def tearDown(self):
        self.config_patch.stop()
        self.tmp.cleanup()

    def chain(self, cfg):
        from summary_engine import normalize_helpers
        return [(h['provider'], h['model'], h['effort']) for h in normalize_helpers(cfg.get('helpers'))]

    def test_without_plugin_options_config_json_decides(self):
        self.config_path.write_text(json.dumps({'shell_helpers_enabled': False, 'helpers': [
            {'provider': 'openai_codex', 'model': 'gpt-5.6-sol', 'effort': 'low'}]}))
        with patch.dict(os.environ, {}, clear=False):
            for name in ('CONTEXT_SAVER_MODEL', 'CONTEXT_SAVER_EFFORT', 'CONTEXT_SAVER_SUMMARIZE_SHELL'):
                os.environ.pop(name, None)
            cfg = hook.config()
        self.assertFalse(cfg['shell_helpers_enabled'])
        self.assertEqual(self.chain(cfg), [('openai_codex', 'gpt-5.6-sol', 'low')])

    def test_plugin_model_and_effort_replace_the_first_helper_and_keep_fallbacks(self):
        self.config_path.write_text(json.dumps({'helpers': [
            {'provider': 'claude_cli', 'model': 'claude-sonnet-5-5', 'effort': 'medium'},
            {'provider': 'openai_codex', 'model': 'gpt-5.6-sol', 'effort': 'low'}]}))
        with patch.dict(os.environ, {'CONTEXT_SAVER_MODEL': 'opus', 'CONTEXT_SAVER_EFFORT': 'high'}):
            cfg = hook.config()
        self.assertEqual(self.chain(cfg), [('claude_cli', 'opus', 'high'), ('openai_codex', 'gpt-5.6-sol', 'low')])

    def test_plugin_defaults_alone_give_one_claude_helper(self):
        with patch.dict(os.environ, {'CONTEXT_SAVER_MODEL': 'sonnet', 'CONTEXT_SAVER_EFFORT': 'medium'}):
            cfg = hook.config()
        self.assertEqual(self.chain(cfg), [('claude_cli', 'sonnet', 'medium')])

    def test_effort_none_turns_thinking_off(self):
        with patch.dict(os.environ, {'CONTEXT_SAVER_MODEL': 'sonnet', 'CONTEXT_SAVER_EFFORT': 'none'}):
            cfg = hook.config()
        self.assertEqual(self.chain(cfg), [('claude_cli', 'sonnet', None)])

    def test_summarize_shell_option_overrides_config_json(self):
        self.config_path.write_text(json.dumps({'shell_helpers_enabled': True}))
        with patch.dict(os.environ, {'CONTEXT_SAVER_SUMMARIZE_SHELL': '0'}):
            self.assertFalse(hook.config()['shell_helpers_enabled'])
        self.config_path.write_text(json.dumps({'shell_helpers_enabled': False}))
        with patch.dict(os.environ, {'CONTEXT_SAVER_SUMMARIZE_SHELL': '1'}):
            self.assertTrue(hook.config()['shell_helpers_enabled'])

    def test_web_bridge_prints_this_machines_paths(self):
        state = Path(self.tmp.name) / 'state'
        result = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'web-bridge'], capture_output=True, text=True,
                                env={**os.environ, 'CONTEXT_SAVER_STATE': str(state)})
        self.assertEqual(result.returncode, 0)
        self.assertNotIn('__CONTEXT_SAVER_', result.stdout)
        self.assertIn(json.dumps(str(hook.ROOT.parent / 'bin' / 'context-saver')), result.stdout)
        self.assertIn(json.dumps(str(state)), result.stdout)
        self.assertTrue(state.is_dir())

    def test_disable_keeps_the_advanced_settings_in_config_json(self):
        stored = {'helpers': [{'provider': 'openai_codex', 'model': 'gpt-5.6-sol', 'effort': 'low'}],
                  'shell_helpers_enabled': False}
        self.config_path.write_text(json.dumps(stored))
        result = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'disable'], capture_output=True, text=True,
                                env={**os.environ, 'CONTEXT_SAVER_CONFIG': str(self.config_path),
                                     'CONTEXT_SAVER_MODEL': 'sonnet', 'CONTEXT_SAVER_SUMMARIZE_SHELL': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.config_path.read_text()), {**stored, 'enabled': False})

    def test_limits_report_the_configured_minimum(self):
        self.config_path.write_text(json.dumps({'minimum_tokens': 2000}))
        result = subprocess.run([sys.executable, str(hook.ROOT / 'hook.py'), 'limits'], capture_output=True, text=True,
                                env={**os.environ, 'CONTEXT_SAVER_CONFIG': str(self.config_path)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'minimum_tokens': 2000})

    def test_launcher_runs_the_web_adapter(self):
        envelope = {'tool_name': 'web.run', 'tool_input': {'search_query': [{'q': 'topic'}]}, 'tool_response': 'short'}
        result = subprocess.run(['sh', str(hook.ROOT.parent / 'bin' / 'context-saver'), 'web-adapter', '--decision-only'],
                                input=json.dumps(envelope), capture_output=True, text=True,
                                env={**os.environ, 'CONTEXT_SAVER_PYTHON': sys.executable,
                                     'CONTEXT_SAVER_DATA': self.tmp.name, 'CONTEXT_SAVER_STATE': self.tmp.name})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'compressed': False})


if __name__ == '__main__':
    unittest.main()
