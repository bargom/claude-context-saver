"""Deterministic tests; no native model calls are made here."""
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import summary_engine as engine

PAYLOAD = {"facts": ["src/app.py:12: ValueError: missing name"], "unknowns": []}


def claude_output(payload=None, *, is_error=False, subtype="success", result="", denials=None,
                  usage=None, model="claude-sonnet-5-5"):
    wrapper = {"type": "result", "subtype": subtype, "is_error": is_error, "result": result,
               "num_turns": 2, "permission_denials": denials or [], "total_cost_usd": 0.0123,
               "usage": usage if usage is not None else {
                   "input_tokens": 900, "output_tokens": 60, "cache_read_input_tokens": 100,
                   "cache_creation_input_tokens": 0, "output_tokens_details": {"thinking_tokens": 0}},
               "modelUsage": {model: {"inputTokens": 900}}}
    if payload is not None:
        wrapper["structured_output"] = payload
    return json.dumps(wrapper)


def answer(facts=None, unknowns=None):
    return claude_output({"facts": facts or PAYLOAD["facts"], "unknowns": unknowns or []})


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.state = Path(self.temporary.name) / "state"
        self.source = "src/app.py:12: ValueError: missing name\n"
        self.env = patch.dict(os.environ, {"CONTEXT_SAVER_CACHE_TTL_SECONDS": "604800"})
        self.env.start()
        # Never read the real login file or start the real CLI in tests.
        self.token_file = Path(self.temporary.name) / "claude-token.txt"
        self.token_patch = patch.object(engine, "CLAUDE_TOKEN_FILE", self.token_file)
        self.token_patch.start()
        self.which = patch.object(engine.shutil, "which", return_value=sys.executable)
        self.which.start()

    def tearDown(self):
        self.which.stop()
        self.token_patch.stop()
        self.env.stop()
        self.temporary.cleanup()

    def summarize(self, **kwargs):
        args = dict(text=self.source, category="search", context="find error", state_dir=self.state)
        args.update(kwargs)
        return engine.summarize(**args)

    def popen(self, *outputs, returncode=0, stderr=""):
        """Patches the CLI start: one prepared stdout per call; records each call."""
        calls, queue = [], list(outputs)
        def start(argv, **kwargs):
            process = Mock(returncode=returncode, pid=1)
            def communicate(input=None, timeout=None):
                calls.append({**kwargs, "argv": argv, "input": input, "timeout": timeout})
                return queue.pop(0), stderr
            process.communicate.side_effect = communicate
            return process
        return patch.object(engine.subprocess, "Popen", side_effect=start), calls

    def summarize_answer(self, stdout, **kwargs):
        popen, _ = self.popen(stdout)
        with popen:
            return self.summarize(**kwargs)

    def test_summary_cache_permissions_and_actual_cost(self):
        popen, calls = self.popen(answer())
        with popen:
            first = self.summarize()
            second = self.summarize()
        self.assertEqual(first["status"], "summary")
        self.assertEqual(second["status"], "cache_hit")
        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["usage"]["total_tokens"], 1060)
        self.assertEqual(first["list_cost_usd"], 0.0123)
        self.assertEqual(second["usage"]["total_tokens"], 0)
        self.assertEqual(second["cached_usage"]["total_tokens"], 1060)
        self.assertEqual(second["observed_model"], "claude-sonnet-5-5")
        for directory in (self.state, self.state / "cache"):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        cache_file = self.state / "cache" / (first["cache_key"] + ".json")
        self.assertEqual(stat.S_IMODE(cache_file.stat().st_mode), 0o600)

    def test_default_run_is_sonnet_at_medium_effort(self):
        popen, calls = self.popen(answer())
        with popen:
            result = self.summarize()
        argv = calls[0]["argv"]
        self.assertEqual(argv[argv.index("--model") + 1], "sonnet")
        self.assertEqual(argv[argv.index("--effort") + 1], "medium")
        self.assertEqual(calls[0]["timeout"], 300)
        self.assertEqual((result["provider"], result["model"]), ("claude_cli", "sonnet"))

    def test_context_category_source_effort_and_prompt_partition_cache(self):
        popen, calls = self.popen(*[answer()] * 6)
        with popen:
            results = [self.summarize(), self.summarize(category="logs"),
                       self.summarize(context="other task"), self.summarize(text=self.source + "other"),
                       self.summarize(effort="low")]
            with patch.object(engine, "PROMPT_HASH", "new-prompt"):
                results.append(self.summarize())
        self.assertEqual(len({item["cache_key"] for item in results}), 6)
        self.assertEqual(len(calls), 6)

    def test_missing_or_invalid_usage_is_incomplete_and_cached_truthfully(self):
        for usage in ({}, {"input_tokens": 9},
                      {"input_tokens": 9, "output_tokens": 2, "cache_read_input_tokens": False,
                       "cache_creation_input_tokens": 0}):
            with self.subTest(usage=usage):
                popen, _ = self.popen(claude_output(PAYLOAD, usage=usage))
                with popen:
                    result = self.summarize(context=str(usage))
                    cached = self.summarize(context=str(usage))
                self.assertEqual(result["status"], "summary")
                self.assertTrue(result["usage_incomplete"])
                self.assertTrue(cached["cached_usage_incomplete"])
                self.assertEqual(cached["usage"]["total_tokens"], 0)

    def test_expired_corrupt_and_disabled_cache_regenerate(self):
        popen, calls = self.popen(*[answer()] * 4)
        with popen:
            first = self.summarize()
            cache_path = self.state / "cache" / (first["cache_key"] + ".json")
            entry = json.loads(cache_path.read_text())
            entry["created_at"] -= engine.DEFAULT_CACHE_TTL_SECONDS + 5
            cache_path.write_text(json.dumps(entry))
            self.assertEqual(self.summarize()["status"], "summary")
            cache_path.write_text("not-json")
            self.assertEqual(self.summarize()["status"], "summary")
            with patch.dict(os.environ, {"CONTEXT_SAVER_CACHE_TTL_SECONDS": "0"}):
                self.assertEqual(self.summarize()["status"], "summary")
        self.assertEqual(len(calls), 4)

    def test_symlinked_state_is_rejected(self):
        self.state.symlink_to(Path(self.temporary.name))
        with patch.object(engine.subprocess, "Popen") as popen:
            result = self.summarize()
        self.assertEqual(result["error"], "unsafe_state_directory")
        popen.assert_not_called()

    def test_inputs_and_effort_are_checked_before_call(self):
        with patch.object(engine.subprocess, "Popen") as popen:
            cases = [(dict(text=""), "empty_source"),
                     (dict(text="a" * 60001), "source_too_long"),
                     (dict(context="a" * 4001), "context_too_long"),
                     (dict(timeout_seconds=0), "invalid_timeout"),
                     (dict(effort="turbo"), "invalid_effort"),
                     (dict(effort=None), "invalid_effort")]
            for args, error in cases:
                with self.subTest(error=error):
                    self.assertEqual(self.summarize(**args)["error"], error)
        popen.assert_not_called()

    def test_failures_never_expose_model_output_or_secrets(self):
        cases = [("secret non-json", "claude_cli_failed"),
                 ("[]", "invalid_claude_json"),
                 (claude_output(is_error=True, subtype="error_during_execution", result="secret failure"),
                  "claude_cli_failed"),
                 (claude_output(None, result="secret prose"), "invalid_summary_json"),
                 (claude_output({"facts": [], "unknowns": []}), "invalid_fact_count"),
                 (claude_output({"facts": ["x"], "unknowns": [], "extra": "secret"}), "invalid_summary_schema"),
                 (answer(facts=["x"] * 7), "invalid_fact_count"),
                 (answer(unknowns=["x"] * 4), "invalid_unknown_count"),
                 (claude_output({"facts": [42], "unknowns": []}), "invalid_summary_item"),
                 (answer(facts=["word " * 351]), "summary_too_long"),
                 (answer(facts=["other.py:12 failure"]), "unsupported_file_reference"),
                 (answer(facts=["src/app.py:13 failure"]), "unsupported_line_reference"),
                 (claude_output(PAYLOAD, denials=[{"tool_name": "Bash", "secret": "value"}]),
                  "unexpected_agent_activity")]
        for stdout, error in cases:
            with self.subTest(error=error):
                result = self.summarize_answer(stdout, context=error)
                self.assertEqual(result["error"], error)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["summary"], "")
                self.assertNotIn("secret", json.dumps(result))

    def test_citations_require_a_matching_source_path_and_line(self):
        source = "src/app.py:12: def handler(): ...\n"
        cases = [(source, "app.py:9999"), (source, "src/app.py:9999"),
                 (source, "pp.py:12"), (source, "rc/app.py:12"),
                 (source + "src/other.py:9999: def other(): ...\n", "app.py:9999"),
                 ("src/app.py defines handler.\n", "app.py:12"),
                 ("src/app.py defines handler.\nsrc/other.py:12: def other(): ...\n", "app.py:12")]
        for source, citation in cases:
            for field in ("facts", "unknowns"):
                payload = {"facts": ["A handler is shown."], "unknowns": []}
                payload[field] = [f"See {citation}."]
                with self.subTest(source=source, citation=citation, field=field):
                    result = self.summarize_answer(claude_output(payload), text=source,
                                                   context=f"{field}:{citation}")
                    self.assertEqual(result["status"], "error")
                    self.assertEqual(result["error"], "unsupported_line_reference")
                    self.assertEqual(result["summary"], "")

    def test_supported_citations_and_filenames_without_lines_are_accepted(self):
        source = "src/app.py:12: def handler(): ...\n"
        cases = [(source, "src/app.py:12"), (source, "app.py:12"), (source, "app.py:12:3"),
                 ("project/" + source, "src/app.py:12"),
                 (source + "tests/app.py:20: def test_handler(): ...\n", "app.py:20"),
                 ("src/app.py defines handler.\n", "src/app.py"),
                 ("src/app.py defines handler.\n", "app.py")]
        for source, citation in cases:
            for field in ("facts", "unknowns"):
                payload = {"facts": ["A handler is shown."], "unknowns": []}
                payload[field] = [f"See {citation}."]
                with self.subTest(source=source, citation=citation, field=field):
                    result = self.summarize_answer(claude_output(payload), text=source,
                                                   context=f"{field}:{citation}")
                    self.assertEqual(result["status"], "summary")
                    self.assertIn(citation, result["summary"])

    def test_structured_search_citations_are_accepted(self):
        source = json.dumps({"file": "src/worker.py", "match_lines": [101]})
        ranged = json.dumps({"results": [{"file_path": "src/worker.py", "start_line": 100, "end_line": 102}]})
        cases = [(source, "src/worker.py:101"), (source, "worker.py:101"),
                 (json.dumps({"results": [{"path": "src/worker.py", "line": 101}]}), "worker.py:101"),
                 (json.dumps([{"file_path": "src/worker.py", "line_number": 101}]), "src/worker.py:101"),
                 (ranged, "src/worker.py:100"), (ranged, "src/worker.py:101"), (ranged, "worker.py:102"),
                 ("Search results:\n" + json.dumps({"file": "src/worker.py", "match_lines": [101],
                                                   "context": "before after"}, ensure_ascii=False) + "\n{}",
                  "src/worker.py:101"),
                 (json.dumps({"results": [{"file": "src/worker.py", "match_lines": [101]}]}, indent=2),
                  "src/worker.py:101"),
                 (source.replace("/", "\\/"), "src/worker.py:101")]
        for source, citation in cases:
            for field in ("facts", "unknowns"):
                payload = {"facts": ["A worker is shown."], "unknowns": []}
                payload[field] = [f"See {citation}."]
                with self.subTest(source=source, citation=citation, field=field):
                    result = self.summarize_answer(claude_output(payload), text=source,
                                                   context=f"{field}:{citation}")
                    self.assertEqual(result["status"], "summary")
                    self.assertIn(citation, result["summary"])

    def test_structured_search_citations_still_require_matching_path_and_line(self):
        source = {"file": "src/worker.py", "match_lines": [101]}
        ranged = {"file_path": "src/worker.py", "start_line": 100, "end_line": 102}
        cases = [(source, "src/worker.py:999"), (source, "orker.py:101"), (source, "rc/worker.py:101"),
                 (ranged, "src/worker.py:99"), (ranged, "src/worker.py:103"),
                 ([{"file": "src/worker.py"}, {"file": "src/other.py", "line": 101}], "worker.py:101"),
                 ({"file": "src/worker.py", "result": {"line": 101}}, "worker.py:101"),
                 ({"file": "src/worker.py", "lines": 101}, "worker.py:101"),
                 ({"file": "src/worker.py", "start_line": 102, "end_line": 100}, "worker.py:101"),
                 ({"file": "src/worker.py", "start_line": 101}, "worker.py:101"),
                 ({"file": "src/worker.py", "line": "101", "line_number": 101.0,
                   "match_lines": [True, "101", 101.0]}, "worker.py:101")]
        for source, citation in cases:
            for field in ("facts", "unknowns"):
                payload = {"facts": ["A worker is shown."], "unknowns": []}
                payload[field] = [f"See {citation}."]
                with self.subTest(source=source, citation=citation, field=field):
                    result = self.summarize_answer(claude_output(payload), text=json.dumps(source),
                                                   context=f"{field}:{citation}")
                    self.assertEqual(result["status"], "error")
                    self.assertEqual(result["error"], "unsupported_line_reference")
                    self.assertEqual(result["summary"], "")

    def test_unknowns_are_in_self_contained_summary_and_logs_need_no_citations(self):
        result = self.summarize_answer(answer(facts=["Connection failed."], unknowns=["The output is truncated."]),
                                       text="Connection failed. [output truncated]")
        self.assertEqual(result["status"], "summary")
        self.assertIn("Unknown: The output is truncated.", result["summary"])

    def test_timeout_kills_process_group(self):
        process = Mock(pid=7654321)
        process.communicate.side_effect = [subprocess.TimeoutExpired("claude", 1), ("", "")]
        with patch.object(engine.subprocess, "Popen", return_value=process), \
                patch.object(engine.os, "killpg") as killpg:
            result = self.summarize(timeout_seconds=1)
        self.assertEqual(result["error"], "model_timeout")
        self.assertTrue(result["usage_incomplete"])
        self.assertEqual(killpg.call_args_list[0].args, (7654321, signal.SIGKILL))

    def test_missing_cli_and_bad_usage(self):
        with patch.object(engine.shutil, "which", return_value=None), \
                patch.object(engine.Path, "home", return_value=Path(self.temporary.name)), \
                patch.object(engine.subprocess, "Popen") as popen:
            self.assertEqual(self.summarize()["error"], "claude_cli_missing")
        popen.assert_not_called()
        with patch.object(engine.subprocess, "Popen", side_effect=FileNotFoundError()):
            self.assertEqual(self.summarize()["error"], "cli_or_state_missing")
        result = self.summarize_answer(claude_output(PAYLOAD, usage={
            "input_tokens": -1, "output_tokens": True, "cache_read_input_tokens": "secret"}))
        self.assertEqual(result["usage"], dict.fromkeys(engine.USAGE_KEYS, 0))
        self.assertTrue(result["usage_incomplete"])

    def test_failed_calls_keep_reported_usage(self):
        result = self.summarize_answer(claude_output(is_error=True, subtype="error_during_execution",
                                                     result="Unexpected internal failure",
                                                     usage={"input_tokens": 503, "output_tokens": 429,
                                                            "cache_read_input_tokens": 0,
                                                            "cache_creation_input_tokens": 0}))
        # Usage numbers are never read as HTTP status markers.
        self.assertEqual(result["error"], "claude_cli_failed")
        self.assertEqual(result["usage"]["total_tokens"], 932)

    def test_invalid_response_diagnostics_are_opt_in_private_and_bounded(self):
        with patch.dict(os.environ), patch("hook.config", return_value={"debug_invalid_responses": False}):
            os.environ.pop("CONTEXT_SAVER_DEBUG_INVALID_RESPONSE", None)
            self.assertIsNone(engine._debug_response(self.state, "private text"))
        with patch.dict(os.environ, {"CONTEXT_SAVER_DEBUG_INVALID_RESPONSE": "1"}):
            path = Path(engine._debug_response(self.state, "synthetic invalid json"))
            bounded = Path(engine._debug_response(self.state, "x" * 20000))
        self.assertEqual(path.read_text(), "synthetic invalid json")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
        self.assertEqual(len(bounded.read_text()), engine.MAX_RESPONSE_CHARS)

    def test_config_flag_enables_diagnostics_and_environment_overrides_it(self):
        with patch.dict(os.environ), patch("hook.config", return_value={"debug_invalid_responses": True}):
            os.environ.pop("CONTEXT_SAVER_DEBUG_INVALID_RESPONSE", None)
            self.assertEqual(Path(engine._debug_response(self.state, "kept")).read_text(), "kept")
            os.environ["CONTEXT_SAVER_DEBUG_INVALID_RESPONSE"] = "0"
            self.assertIsNone(engine._debug_response(self.state, "dropped"))
        with patch.dict(os.environ), patch("hook.config", side_effect=ValueError("invalid configuration")):
            os.environ.pop("CONTEXT_SAVER_DEBUG_INVALID_RESPONSE", None)
            self.assertIsNone(engine._debug_response(self.state, "dropped"))

    def test_ttl_is_bounded(self):
        for value, expected in [("invalid", 604800), ("-5", 0), ("999999999", 2592000)]:
            with patch.dict(os.environ, {"CONTEXT_SAVER_CACHE_TTL_SECONDS": value}):
                self.assertEqual(engine._ttl(), expected)

    def test_interrupted_wait_stops_the_helper(self):
        with subprocess.Popen(["sleep", "30"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
            with patch.object(process, "communicate", side_effect=KeyboardInterrupt()), \
                    self.assertRaises(KeyboardInterrupt):
                engine._communicate(process, 15, "input")
            self.assertIsNotNone(process.poll())

    def test_claude_runner_is_isolated_and_thinking_follows_effort(self):
        blocked = {"ANTHROPIC_API_KEY": "secret", "ANTHROPIC_BASE_URL": "secret", "CLAUDECODE": "1",
                   "CLAUDE_CODE_ENTRYPOINT": "cli", "OPENAI_API_KEY": "secret",
                   "SSH_AUTH_SOCK": "/synthetic/agent.sock", "GOOGLE_CLOUD_PROJECT": "secret"}
        popen, calls = self.popen(answer(), answer())
        with patch.dict(os.environ, {**blocked, "CLAUDE_CODE_PROMPT_CACHE_TTL": "1h"}), popen:
            payload, usage, incomplete, extra = engine._run_claude("SENTINEL-7f3", 30, "none")
            engine._run_claude("SENTINEL-7f3", 30, "low")
        off, low = calls
        self.assertEqual(payload, PAYLOAD)
        self.assertEqual(usage["input_tokens"], 1000)
        self.assertEqual(usage["cache_read_tokens"], 100)
        self.assertEqual(usage["total_tokens"], 1060)
        self.assertFalse(incomplete)
        self.assertEqual(extra, {"observed_model": "claude-sonnet-5-5", "list_cost_usd": 0.0123})
        self.assertEqual(off["input"], "CAPTURED_DATA_JSON:\nSENTINEL-7f3")
        self.assertEqual(off["env"]["MAX_THINKING_TOKENS"], "0")
        self.assertNotIn("--effort", off["argv"])
        self.assertNotIn("MAX_THINKING_TOKENS", low["env"])
        # One-shot sources are never re-read: short cache TTL, caller's 1h ignored.
        self.assertEqual([off["env"]["CLAUDE_CODE_PROMPT_CACHE_TTL"], low["env"]["CLAUDE_CODE_PROMPT_CACHE_TTL"]],
                         ["5m", "5m"])
        self.assertNotIn("CLAUDE_CODE_PROMPT_CACHE_TTL", engine._environment())
        self.assertEqual(low["argv"][low["argv"].index("--effort") + 1], "low")
        argv = off["argv"]
        self.assertEqual(argv[argv.index("--model") + 1], "sonnet")
        for flag in ("--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "")
        self.assertEqual(json.loads(argv[argv.index("--settings") + 1]), {"disableAllHooks": True})
        self.assertEqual(argv[argv.index("--system-prompt") + 1], engine.INSTRUCTIONS)
        self.assertEqual(json.loads(argv[argv.index("--json-schema") + 1]), engine.SCHEMA)
        self.assertNotIn("SENTINEL-7f3", " ".join(argv))
        self.assertIs(off["stdin"], subprocess.PIPE)
        self.assertTrue(off["start_new_session"])
        self.assertNotIn("shell", off)
        self.assertEqual(off["env"]["CONTEXT_SAVER_ACTIVE"], "1")
        for key in blocked:
            self.assertNotIn(key, off["env"])
        self.assertFalse(Path(off["cwd"]).exists())

    def test_token_file_reaches_child_env_only(self):
        token = "sk-ant-oat01-" + "A" * 40 + "_b-" + "c" * 40
        self.token_file.write_text(token[:60] + "\n   " + token[60:] + "\n")  # pasted with a wrap
        self.token_file.chmod(0o600)
        popen, calls = self.popen(answer())
        with popen:
            engine._run_claude("SENTINEL-7f3", 30, "low")
        self.assertEqual(calls[0]["env"]["CLAUDE_CODE_OAUTH_TOKEN"], token)
        self.assertNotIn(token, " ".join(calls[0]["argv"]))
        self.assertNotIn(token, calls[0]["input"])
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", os.environ)

    def test_missing_token_file_uses_normal_login(self):
        popen, calls = self.popen(answer())
        with popen:
            engine._run_claude("SENTINEL-7f3", 30, "low")
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", calls[0]["env"])

    def test_unsafe_or_invalid_token_file_fails_closed(self):
        valid = "sk-ant-oat01-" + "A" * 60
        target = Path(self.temporary.name) / "real-token.txt"
        target.write_text(valid)
        target.chmod(0o600)
        cases = [("claude_token_file_insecure", valid, 0o644, False),
                 ("claude_token_file_invalid", "not a token", 0o600, False),
                 ("claude_token_file_invalid", 'export TOKEN="' + valid + '"', 0o600, False),
                 ("claude_token_file_unreadable", None, None, True)]
        for code, content, mode, link in cases:
            self.token_file.unlink(missing_ok=True)
            if link:
                self.token_file.symlink_to(target)
            else:
                self.token_file.write_text(content)
                self.token_file.chmod(mode)
            with self.subTest(code=code), patch.object(engine.subprocess, "Popen") as popen, \
                    self.assertRaises(engine.ProviderFailure) as failure:
                engine._run_claude("SENTINEL-7f3", 30, "low")
            self.assertEqual(failure.exception.code, code)
            popen.assert_not_called()

    def test_claude_failures_are_classified(self):
        cases = [(claude_output(is_error=True, subtype="error_during_execution", result="API Error: 529 Overloaded"), 0, "", "provider_capacity"),
                 (claude_output(is_error=True, result="Claude AI usage limit reached|1790600000"), 0, "", "provider_capacity"),
                 (claude_output(is_error=True, result="Request timed out"), 0, "", "model_timeout"),
                 (claude_output(is_error=True, result="Invalid API key · Please run /login"), 0, "", "claude_cli_failed"),
                 (claude_output(is_error=True, result="Not logged in; timeout"), 0, "", "claude_cli_failed"),
                 ("not json", 1, "HTTP 429 quota", "provider_capacity"),
                 ("not json", 1, "permission denied", "claude_cli_failed")]
        for stdout, returncode, stderr, code in cases:
            popen, _ = self.popen(stdout, returncode=returncode, stderr=stderr)
            with self.subTest(code=code, stdout=stdout[:60], stderr=stderr), popen, \
                    self.assertRaises(engine.ProviderFailure) as failure:
                engine._run_claude("captured", 30, "none")
            self.assertEqual(failure.exception.code, code)


if __name__ == "__main__":
    unittest.main()
