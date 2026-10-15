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

# Legacy chain pinned for the Cursor/Codex runner tests; Claude tests below.
LEGACY_HELPERS = ({"provider": "cursor_cli", "model": "composer-2.5", "effort": None},
                  {"provider": "openai_codex", "model": "gpt-5.6-sol", "effort": "low"})
CURSOR_KEY = "cursor_cli:composer-2.5"
CODEX_KEY = "openai_codex:gpt-5.6-sol"
CLAUDE_CHAIN = [{"provider": "claude_cli", "model": "claude-haiku-4-5"},
                {"provider": "claude_cli", "model": "claude-sonnet-5", "effort": "low"}]


def success(facts=None, unknowns=None, **overrides):
    wrapper = {
        "type": "result", "subtype": "success", "is_error": False, "num_turns": 1,
        "result": json.dumps({"facts": facts or ["src/app.py:12: ValueError: missing name"],
                                "unknowns": unknowns or []}),
        "usage": {"inputTokens": 90, "outputTokens": 20, "reasoningTokens": 2,
                  "cacheReadTokens": 0, "cacheWriteTokens": 0},
    }
    wrapper.update(overrides)
    return json.dumps(wrapper), 0


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.state = Path(self.temporary.name) / "state"
        self.source = "src/app.py:12: ValueError: missing name\n"
        self.env = patch.dict(os.environ, {"CONTEXT_SAVER_CACHE_TTL_SECONDS": "604800"})
        self.env.start()
        self.helpers = patch.object(engine, "DEFAULT_HELPERS", LEGACY_HELPERS)
        self.helpers.start()

    def tearDown(self):
        self.helpers.stop()
        self.env.stop()
        self.temporary.cleanup()

    def summarize(self, **kwargs):
        args = dict(text=self.source, category="search", context="find error", state_dir=self.state)
        args.update(kwargs)
        return engine.summarize(**args)

    def test_summary_cache_permissions_and_actual_cost(self):
        with patch.object(engine, "_run", return_value=success()) as run:
            first = self.summarize()
            second = self.summarize()
        self.assertEqual(first["status"], "summary")
        self.assertEqual(second["status"], "cache_hit")
        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(first["usage"]["total_tokens"], 110)
        self.assertEqual(second["usage"]["total_tokens"], 0)
        self.assertEqual(second["cached_usage"]["total_tokens"], 110)
        for directory in (self.state, self.state / "cache"):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        cache_file = self.state / "cache" / (first["cache_key"] + ".json")
        self.assertEqual(stat.S_IMODE(cache_file.stat().st_mode), 0o600)

    def test_context_category_source_model_and_prompt_partition_cache(self):
        with patch.object(engine, "_run", return_value=success()) as run:
            results = [self.summarize(), self.summarize(category="logs"),
                       self.summarize(context="other task"), self.summarize(text=self.source + "other")]
            results.append(self.summarize(helpers=[{"provider": "cursor_cli", "model": "other-model"},
                                                   LEGACY_HELPERS[1]]))
            with patch.object(engine, "PROMPT_HASH", "new-prompt"):
                results.append(self.summarize())
        self.assertEqual(len({item["cache_key"] for item in results}), 6)
        self.assertEqual(run.call_count, 6)

    def test_cursor_cache_categories_are_disjoint_and_count_once(self):
        cursor_usage = {"inputTokens": 7, "outputTokens": 41,
                        "cacheReadTokens": 147695, "cacheWriteTokens": 39331}
        with patch.object(engine, "_run", return_value=success(usage=cursor_usage)):
            result = self.summarize()
            cached = self.summarize()
        self.assertEqual(result["usage"]["input_tokens"], 187033)
        self.assertEqual(result["usage"]["total_tokens"], 187074)
        self.assertEqual(result["usage"]["cache_read_tokens"], 147695)
        self.assertEqual(result["usage"]["cache_write_tokens"], 39331)
        self.assertEqual(cached["cached_usage"], result["usage"])
        self.assertEqual(cached["usage"], dict.fromkeys(engine.USAGE_KEYS, 0))
        self.assertFalse(cached["cached_usage_incomplete"])
        self.assertNotIn("usage_incomplete", result)

    def test_missing_or_invalid_cursor_usage_is_incomplete_and_cached_truthfully(self):
        for usage in (None, {}, {"inputTokens": 9},
                      {"inputTokens": 9, "outputTokens": 2, "cacheReadTokens": False, "cacheWriteTokens": 0}):
            with self.subTest(usage=usage), patch.object(engine, "_run", return_value=success(usage=usage)):
                result = self.summarize(context=str(usage))
                cached = self.summarize(context=str(usage))
                self.assertEqual(result["status"], "summary")
                self.assertTrue(result["usage_incomplete"])
                self.assertTrue(cached["cached_usage_incomplete"])
                self.assertEqual(cached["usage"]["total_tokens"], 0)

    def test_provider_and_cli_version_partition_cache(self):
        with patch.object(engine, "_run", return_value=success()) as run:
            first = self.summarize()
            provider = self.summarize(helpers=[LEGACY_HELPERS[0], {**LEGACY_HELPERS[1], "effort": "medium"}])
            with patch.object(engine, "CLI_VERSION", "future-version"):
                version = self.summarize()
        self.assertEqual(len({x["cache_key"] for x in (first, provider, version)}), 3)
        self.assertEqual(run.call_count, 3)

    def test_inner_schema_rejections_never_trigger_fallback(self):
        malformed = [{"facts": ["x"], "unknowns": [], "extra": "x"},
                     {"facts": [], "unknowns": []}, {"facts": ["x"] * 7, "unknowns": []},
                     {"facts": ["x"], "unknowns": ["x"] * 4},
                     {"facts": [42], "unknowns": []}]
        for payload in malformed:
            with self.subTest(payload=payload), \
                    patch.object(engine, "_run", return_value=success(result=json.dumps(payload))), \
                    patch.object(engine, "_run_codex") as backup:
                result = self.summarize()
                self.assertEqual(result["status"], "error")
                backup.assert_not_called()
        self.assertNotIn("minItems", json.dumps(engine.SCHEMA))
        self.assertNotIn("maxItems", json.dumps(engine.SCHEMA))

    def test_capacity_without_usage_marks_fallback_usage_incomplete(self):
        payload = {"facts": ["Failure"], "unknowns": []}
        with patch.object(engine, "_run", return_value=success(
                subtype="error", is_error=True, result="HTTP 429 quota", usage=None)), \
                patch.object(engine, "_run_codex", return_value=(payload, engine._usage({
                    "input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 80}), False)) as backup:
            result = self.summarize()
        self.assertEqual(result["status"], "summary")
        self.assertTrue(result["usage_incomplete"])
        self.assertEqual(result["usage"]["total_tokens"], 110)
        self.assertEqual(result["provider"], engine.CODEX_PROVIDER)
        backup.assert_called_once()

    def test_expired_corrupt_and_disabled_cache_regenerate(self):
        with patch.object(engine, "_run", return_value=success()) as run:
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
        self.assertEqual(run.call_count, 4)

    def test_symlinked_state_is_rejected(self):
        self.state.symlink_to(Path(self.temporary.name))
        with patch.object(engine, "_run") as run:
            result = self.summarize()
        self.assertEqual(result["error"], "unsafe_state_directory")
        run.assert_not_called()

    def test_sources_and_context_are_bounded_before_call(self):
        with patch.object(engine, "_run") as run:
            cases = [(dict(text=""), "empty_source"),
                     (dict(text="a" * 60001), "source_too_long"),
                     (dict(context="a" * 4001), "context_too_long"),
                     (dict(timeout_seconds=0), "invalid_timeout")]
            for args, error in cases:
                with self.subTest(error=error):
                    self.assertEqual(self.summarize(**args)["error"], error)
        run.assert_not_called()

    def test_failures_never_expose_model_output_or_secrets(self):
        cases = [(("secret cli diagnostic", 1), "cli_failed"),
                 (("secret non-json", 0), "invalid_cli_json"),
                 (("[]", 0), "cli_unsuccessful"),
                 (success(subtype="error", is_error=True), "cli_unsuccessful"),
                 (success(result="secret non-json"), "invalid_summary_json"),
                 (success(result='{"facts": [], "unknowns": []}'), "invalid_fact_count"),
                 (success(facts=["x"] * 7), "invalid_fact_count"),
                 (success(unknowns=["x"] * 4), "invalid_unknown_count"),
                 (success(facts=["word " * 351]), "summary_too_long"),
                 (success(facts=["other.py:12 failure"]), "unsupported_file_reference"),
                 (success(facts=["src/app.py:13 failure"]), "unsupported_line_reference"),
                 (success(tool_calls=[{"secret": "value"}]), "unexpected_agent_activity"),
                 (success(num_turns=2), "unexpected_agent_activity")]
        for response, error in cases:
            with self.subTest(error=error), patch.object(engine, "_run", return_value=response):
                result = self.summarize()
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
                with self.subTest(source=source, citation=citation, field=field), \
                        patch.object(engine, "_run", return_value=success(result=json.dumps(payload))), \
                        patch.object(engine, "_run_codex") as backup:
                    result = self.summarize(text=source, context=f"{field}:{citation}")
                    self.assertEqual(result["status"], "error")
                    self.assertEqual(result["error"], "unsupported_line_reference")
                    self.assertEqual(result["summary"], "")
                    backup.assert_not_called()

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
                with self.subTest(source=source, citation=citation, field=field), \
                        patch.object(engine, "_run", return_value=success(result=json.dumps(payload))):
                    result = self.summarize(text=source, context=f"{field}:{citation}")
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
                                                   "context": "before\u2028after"}, ensure_ascii=False) + "\n{}",
                  "src/worker.py:101"),
                 (json.dumps({"results": [{"file": "src/worker.py", "match_lines": [101]}]}, indent=2),
                  "src/worker.py:101"),
                 (source.replace("/", "\\/"), "src/worker.py:101")]
        for source, citation in cases:
            for field in ("facts", "unknowns"):
                payload = {"facts": ["A worker is shown."], "unknowns": []}
                payload[field] = [f"See {citation}."]
                with self.subTest(source=source, citation=citation, field=field), \
                        patch.object(engine, "_run", return_value=success(result=json.dumps(payload))):
                    result = self.summarize(text=source, context=f"{field}:{citation}")
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
                with self.subTest(source=source, citation=citation, field=field), \
                        patch.object(engine, "_run", return_value=success(result=json.dumps(payload))):
                    result = self.summarize(text=json.dumps(source), context=f"{field}:{citation}")
                    self.assertEqual(result["status"], "error")
                    self.assertEqual(result["error"], "unsupported_line_reference")
                    self.assertEqual(result["summary"], "")

    def test_primary_stream_requires_one_final_result_and_records_observed_model(self):
        events = [{"type": "system", "subtype": "init", "model": "Composer 2.5"},
                  {"type": "assistant", "message": {"content": [{"type": "text", "text": "answer"}]}},
                  json.loads(success()[0])]
        with patch.object(engine, "_run", return_value=("\n".join(map(json.dumps, events)), 0)):
            result = self.summarize()
            cached = self.summarize()
        self.assertEqual(result["status"], "summary")
        self.assertEqual(cached["observed_model"], "Composer 2.5")
        self.assertEqual(result["observed_model"], "Composer 2.5")
        self.assertEqual(result["model"], "composer-2.5")
        self.assertEqual(result["usage"]["total_tokens"], 110)

    def test_primary_stream_rejects_tools_missing_duplicate_or_nonfinal_results(self):
        final = json.loads(success()[0])
        cases = [([{"type": "tool_call", "subtype": "started"}, final], "unexpected_agent_activity"),
                 ([{"type": "tool_result"}, final], "unexpected_agent_activity"),
                 ([{"type": "assistant", "message": {"content": [{"type": "tool_use"}]}}, final], "unexpected_agent_activity"),
                 ([final, final], "invalid_result_count"),
                 ([{"type": "system", "subtype": "init"}], "missing_result"),
                 ([final, {"type": "assistant"}], "invalid_event_sequence")]
        for events, error in cases:
            with self.subTest(error=error), \
                    patch.object(engine, "_run", return_value=("\n".join(map(json.dumps, events)), 0)), \
                    patch.object(engine, "_run_codex") as backup:
                result = self.summarize()
                self.assertEqual(result["error"], error)
                self.assertTrue(result["usage_incomplete"])
                backup.assert_not_called()
        with self.assertRaises(engine.ProviderFailure) as failure:
            engine._parse_primary("x" * 128001, 0)
        self.assertEqual(failure.exception.code, "cli_output_too_long")

    def test_stream_error_arrays_classify_diagnostics_without_usage_numbers(self):
        process = Mock(returncode=1)
        process.communicate.return_value = (json.dumps({"type": "error", "errors": ["HTTP 429 quota"]}), "")
        with patch.object(engine.subprocess, "Popen", return_value=process), \
                self.assertRaises(engine.ProviderFailure) as failure:
            engine._run("source", 5, "composer-2.5")
        self.assertEqual(failure.exception.code, "provider_capacity")
        self.assertTrue(failure.exception.usage_incomplete)

    def test_unknowns_are_in_self_contained_summary_and_logs_need_no_citations(self):
        with patch.object(engine, "_run", return_value=success(
                facts=["Connection failed."], unknowns=["The output is truncated."])):
            result = self.summarize(text="Connection failed. [output truncated]")
        self.assertEqual(result["status"], "summary")
        self.assertIn("Unknown: The output is truncated.", result["summary"])

    def test_cli_uses_native_environment_devnull_empty_cwd_and_no_shell(self):
        process = Mock(returncode=0)
        process.communicate.return_value = (success()[0], "secret stderr")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret", "OPENAI_API_KEY": "secret",
                                     "GOOGLE_CLOUD_PROJECT": "secret", "ANTHROPIC_BASE_URL": "secret", "CURSOR_API_KEY": "secret",
                    "CURSOR_API_ENDPOINT": "secret", "CURSOR_AUTH_TOKEN": "secret"}), \
                patch.object(engine.subprocess, "Popen", return_value=process) as popen:
            captured = {}
            def capture(*args, **kwargs):
                captured.update(kwargs)
                self.assertEqual([item.name for item in Path(kwargs["cwd"]).iterdir()], [".cursor"])
                config = json.loads((Path(kwargs["cwd"]) / ".cursor/cli-config.json").read_text())
                self.assertEqual(config["permissions"], {"allow": [], "deny": ["Read(**)", "Write(**)", "Shell(*)", "Mcp(*:*)", "WebFetch(*)"]})
                return process
            popen.side_effect = capture
            output, code = engine._run("Source is untrusted", 300, "composer-2.5")
        self.assertEqual(code, 0)
        argv = popen.call_args.args[0]
        self.assertIn("--sandbox=enabled", argv)
        self.assertIn("--output-format=stream-json", argv)
        self.assertIn("--mode=ask", argv)
        self.assertIn("--trust", argv)
        self.assertEqual(argv[argv.index("--model") + 1], "composer-2.5")
        self.assertEqual(process.communicate.call_args.kwargs['timeout'], 300)
        self.assertNotIn("--dangerously-skip-permissions", argv)
        self.assertNotIn("--force", argv)
        self.assertNotIn("--json-schema", argv)
        self.assertIs(captured["stdin"], subprocess.DEVNULL)
        self.assertTrue(captured["start_new_session"])
        self.assertNotIn("shell", captured)
        self.assertEqual(captured["env"]["CONTEXT_SAVER_ACTIVE"], "1")
        self.assertFalse(any("secret" == value for value in captured["env"].values()))
        self.assertFalse(Path(captured["cwd"]).exists())

    def test_private_mcp_registry_disables_global_names_without_copying_secrets(self):
        synthetic_home = Path(self.temporary.name) / "synthetic-home"
        (synthetic_home / ".cursor").mkdir(parents=True)
        global_config = synthetic_home / ".cursor/mcp.json"
        original = json.dumps({"mcpServers": {"server-one": {"env": {"TOKEN": "secret"}},
                                              "server two": {"command": "must-not-run"}}})
        global_config.write_text(original)
        with tempfile.TemporaryDirectory(prefix="cursor-registry-") as workspace, \
                patch.object(engine.Path, "home", return_value=synthetic_home):
            config_dir = engine._prepare_cursor_config(workspace)
            registry_files = list(config_dir.glob("projects/*/mcp-disabled.json"))
            self.assertEqual(len(registry_files), 1)
            self.assertEqual(json.loads(registry_files[0].read_text()), ["server-one", "server two"])
            self.assertNotIn("secret", registry_files[0].read_text())
            self.assertEqual(stat.S_IMODE(registry_files[0].stat().st_mode), 0o600)
        self.assertEqual(global_config.read_text(), original)
        global_config.write_text("malformed")
        with tempfile.TemporaryDirectory() as workspace, \
                patch.object(engine.Path, "home", return_value=synthetic_home), \
                self.assertRaisesRegex(ValueError, "invalid_cursor_mcp_configuration"):
            engine._prepare_cursor_config(workspace)

    def test_primary_uses_native_file_auth_without_ssh_or_secret_overrides(self):
        ssh_context = {"SSH_CONNECTION": "192.0.2.1 1234 192.0.2.2 22",
                       "SSH_CLIENT": "192.0.2.1 1234 22", "SSH_TTY": "/dev/ttys999"}
        excluded = {"SSH_AUTH_SOCK": "/synthetic/agent.sock", "GEMINI_API_KEY": "secret",
                    "OPENAI_API_KEY": "secret", "GOOGLE_CLOUD_PROJECT": "secret",
                    "ANTHROPIC_BASE_URL": "secret", "CURSOR_API_KEY": "secret",
                    "CURSOR_API_ENDPOINT": "secret", "CURSOR_AUTH_TOKEN": "secret"}
        process = Mock(returncode=0)
        process.communicate.return_value = (success()[0], "")
        for context in (ssh_context, {}):
            with self.subTest(has_ssh=bool(context)), \
                    patch.dict(os.environ, {"HOME": self.temporary.name, **context, **excluded}, clear=True), \
                    patch.object(engine.subprocess, "Popen", return_value=process) as popen:
                engine._run("synthetic source", 5, "composer-2.5")
                primary_env = popen.call_args.kwargs["env"]
                self.assertEqual(primary_env["HOME"], self.temporary.name)
                for key in ssh_context:
                    self.assertNotIn(key, primary_env)
                self.assertEqual(primary_env["AGENT_CLI_CREDENTIAL_STORE"], "file")
                self.assertEqual(primary_env["CURSOR_CONFIG_DIR"], primary_env["CURSOR_DATA_DIR"])
                for key in excluded:
                    self.assertNotIn(key, primary_env)

    def test_timeout_kills_process_group_and_exhausts_budget(self):
        process = Mock(pid=7654321)
        process.communicate.side_effect = [subprocess.TimeoutExpired("agy", 1), ("", "")]
        with patch.object(engine.subprocess, "Popen", return_value=process), \
                patch.object(engine.os, "killpg") as killpg, \
                patch.object(engine, "_run_codex", side_effect=TimeoutError()):
            result = self.summarize(timeout_seconds=1)
        self.assertEqual(result["error"], "model_timeout")
        self.assertEqual(killpg.call_args_list[0].args, (7654321, signal.SIGKILL))

    def test_cli_missing_and_bad_usage(self):
        with patch.object(engine, "_run", side_effect=FileNotFoundError()):
            self.assertEqual(self.summarize()["error"], "cli_or_state_missing")
        with patch.object(engine, "_run", return_value=success(
                usage={"inputTokens": -1, "outputTokens": True, "cacheReadTokens": "secret"})):
            result = self.summarize()
        self.assertEqual(result["usage"], dict.fromkeys(engine.USAGE_KEYS, 0))

    def test_accepts_whole_json_fence_but_rejects_prose_with_embedded_json(self):
        payload = json.dumps({"facts": ["Failure"], "unknowns": []})
        for response in (payload, "```json\n" + payload + "\n```", "```\n" + payload + "\n```"):
            with self.subTest(result=response):
                parsed, usage = engine._parse_primary(*success(result=response))
                self.assertEqual(parsed["facts"], ["Failure"])
        with self.assertRaises(engine.ProviderFailure):
            engine._parse_primary(*success(result="Here is the answer: " + payload))

    def test_cli_timeout_markers_and_stderr_capacity_keep_reported_usage(self):
        for message in ("HTTP 503 service unavailable", "RESOURCE_EXHAUSTED", "Request timed out"):
            with self.subTest(message=message), self.assertRaises(engine.ProviderFailure) as failure:
                engine._parse_primary(*success(subtype="error", is_error=True, result=message))
            self.assertIn(failure.exception.code, ("provider_capacity", "model_timeout"))
            self.assertEqual(failure.exception.usage["total_tokens"], 110)
        process = Mock(returncode=1)
        process.communicate.return_value = (success()[0], "HTTP 429 quota")
        with patch.object(engine.subprocess, "Popen", return_value=process), \
                self.assertRaises(engine.ProviderFailure) as failure:
            engine._run("synthetic source", 5, "composer-2.5")
        self.assertEqual(failure.exception.code, "provider_capacity")
        self.assertEqual(failure.exception.usage["total_tokens"], 110)

    def test_invalid_response_diagnostics_are_opt_in_private_and_bounded(self):
        with patch.dict(os.environ), patch("hook.config", return_value={"debug_invalid_responses": False}):
            os.environ.pop("CONTEXT_SAVER_DEBUG_INVALID_RESPONSE", None)
            self.assertIsNone(engine._debug_response(self.state, "private text"))
        with patch.dict(os.environ, {"CONTEXT_SAVER_DEBUG_INVALID_RESPONSE": "1"}), \
                patch.object(engine, "_run", return_value=success(result="synthetic invalid json")):
            result = self.summarize()
        path = Path(result["diagnostic_file"])
        self.assertEqual(path.read_text(), "synthetic invalid json")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
        with patch.dict(os.environ, {"CONTEXT_SAVER_DEBUG_INVALID_RESPONSE": "1"}):
            bounded = Path(engine._debug_response(self.state, "x" * 20000))
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

    def test_capacity_falls_back_preserves_both_provider_usage_and_cached_provider(self):
        primary = success(subtype="error", is_error=True, result="RESOURCE_EXHAUSTED: quota exceeded")
        backup_usage = engine._usage({"input_tokens": 100, "output_tokens": 10, "thinking_tokens": 5})
        payload = {"facts": ["src/app.py:12: ValueError: missing name"], "unknowns": []}
        with patch.object(engine, "_run", return_value=primary), \
                patch.object(engine, "_run_codex", return_value=(payload, backup_usage, False)) as backup:
            result = self.summarize()
            cached = self.summarize()
        self.assertEqual(result["status"], "summary")
        self.assertEqual(result["provider"], engine.CODEX_PROVIDER)
        self.assertEqual(result["model"], "gpt-5.6-sol")
        self.assertEqual(result["usage"]["total_tokens"], 220)
        self.assertEqual(result["per_provider_usage"][CURSOR_KEY]["total_tokens"], 110)
        self.assertEqual(result["per_provider_usage"][CODEX_KEY]["total_tokens"], 110)
        self.assertEqual(cached["provider"], engine.CODEX_PROVIDER)
        self.assertEqual(cached["model"], "gpt-5.6-sol")
        self.assertEqual(cached["usage"]["total_tokens"], 0)
        self.assertEqual(cached["cached_usage"]["total_tokens"], 220)
        self.assertLessEqual(backup.call_args.args[1], 300)
        self.assertEqual(backup.call_count, 1)

    def test_authentication_schema_and_unrelated_errors_never_fall_back(self):
        responses = [success(subtype="error", is_error=True, result="unauthenticated: quota endpoint"),
                     success(result="not-json"), ("permission denied", 1),
                     success(subtype="error", is_error=True, result="Invalid request schema"),
                     success(subtype="error", is_error=True, result="Invalid request schema; HTTP 503"),
                     success(subtype="error", is_error=True, result="HTTP 401 quota endpoint"),
                     success(subtype="error", is_error=True, result="Not logged in; timeout") ]
        for primary in responses:
            with self.subTest(primary=primary), patch.object(engine, "_run", return_value=primary), \
                    patch.object(engine, "_run_codex") as backup:
                self.assertEqual(self.summarize()["status"], "error")
                backup.assert_not_called()

    def test_usage_counts_are_never_capacity_error_markers(self):
        primary = success(subtype="error", is_error=True, result="Unexpected internal failure",
                          usage={"inputTokens": 503, "outputTokens": 429,
                                 "cacheReadTokens": 0, "cacheWriteTokens": 0})
        with patch.object(engine, "_run", return_value=primary), \
                patch.object(engine, "_run_codex") as backup:
            result = self.summarize()
        self.assertEqual(result["error"], "cli_unsuccessful")
        self.assertEqual(result["usage"]["total_tokens"], 932)
        backup.assert_not_called()

    def test_timeout_fallback_and_remaining_deadline(self):
        payload = {"facts": ["Failure"], "unknowns": []}
        with patch.object(engine, "_run", side_effect=TimeoutError()), \
                patch.object(engine, "_run_codex", return_value=(payload, engine._usage(None), True)) as backup, \
                patch.object(engine.time, "monotonic", side_effect=[100, 128, 130]):
            result = self.summarize(timeout_seconds=45)
        self.assertEqual(result["status"], "summary")
        self.assertEqual(result["primary_error"], "model_timeout")
        self.assertEqual(backup.call_args.args[1], 17)
        with patch.object(engine, "_run", side_effect=TimeoutError()), \
                patch.object(engine, "_run_codex") as backup, \
                patch.object(engine.time, "monotonic", side_effect=[100, 145, 146]):
            result = self.summarize(context="no time", timeout_seconds=45)
        self.assertEqual(result["error"], "model_timeout")
        backup.assert_not_called()

    def test_five_minute_default_and_shared_fallback_deadline(self):
        with patch.dict(os.environ, {'CONTEXT_SAVER_PRIMARY_TIMEOUT_SECONDS': '300',
                                     'CONTEXT_SAVER_BACKUP_TIMEOUT_SECONDS': '300'}), \
                patch.object(engine, '_run', return_value=success()) as primary:
            self.assertEqual(self.summarize()['status'], 'summary')
        self.assertEqual(primary.call_args.args[1], 300)
        payload = {'facts': ['Failure'], 'unknowns': []}
        with patch.object(engine, '_run', side_effect=TimeoutError()), \
                patch.object(engine, '_run_codex', return_value=(payload, engine._usage(None), True)) as backup, \
                patch.object(engine.time, 'monotonic', side_effect=[100, 360, 365]):
            result = self.summarize(context='five-minute early timeout')
        self.assertEqual(result['status'], 'summary')
        self.assertEqual(backup.call_args.args[1], 40)
        with patch.object(engine, '_run', side_effect=TimeoutError()), \
                patch.object(engine, '_run_codex') as backup, \
                patch.object(engine.time, 'monotonic', side_effect=[100, 400, 401]):
            result = self.summarize(context='five-minute exhausted timeout')
        self.assertEqual(result['error'], 'model_timeout')
        backup.assert_not_called()

    def test_native_backup_flags_schema_stdin_and_usage(self):
        process = Mock(returncode=0)
        event = {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 15,
                  "cached_input_tokens": 80, "thinking_tokens": 4, "total_tokens": 119}}
        process.communicate.return_value = (json.dumps(event), "")
        captured = {}
        def launch(argv, **kwargs):
            captured.update(argv=argv, **kwargs)
            output = Path(argv[argv.index("--output-last-message") + 1])
            output.write_text(json.dumps({"facts": ["Failure"], "unknowns": []}))
            return process
        blocked_env = {"SSH_CONNECTION": "192.0.2.1 1234 192.0.2.2 22",
                       "SSH_CLIENT": "192.0.2.1 1234 22", "SSH_TTY": "/dev/ttys999",
                       "SSH_AUTH_SOCK": "/synthetic/agent.sock", "GEMINI_API_KEY": "secret",
                       "OPENAI_API_KEY": "secret", "GOOGLE_CLOUD_PROJECT": "secret",
                       "ANTHROPIC_BASE_URL": "secret", "CURSOR_API_KEY": "secret",
                    "CURSOR_API_ENDPOINT": "secret", "CURSOR_AUTH_TOKEN": "secret"}
        with patch.dict(os.environ, blocked_env), \
                patch.object(engine.subprocess, "Popen", side_effect=launch), \
                patch.object(engine.shutil, "which", return_value="/mock/codex"):
            payload, usage, incomplete = engine._run_codex("untrusted captured source", 15, "gpt-5.6-sol", "low")
        self.assertEqual(payload["facts"], ["Failure"])
        self.assertEqual(usage["total_tokens"], 115)
        self.assertFalse(incomplete)
        self.assertEqual(usage["cache_read_tokens"], 80)
        self.assertEqual(usage["thinking_tokens"], 4)
        self.assertEqual(process.communicate.call_args.kwargs["input"], "untrusted captured source")
        for flag in ("--ignore-user-config", "--ephemeral", "features.hooks=false",
                     "features.shell_tool=false", "features.multi_agent=false", 'web_search="disabled"'):
            self.assertIn(flag, captured["argv"])
        self.assertIn("read-only", captured["argv"])
        self.assertIs(captured["stdin"], subprocess.PIPE)
        for key in blocked_env:
            self.assertNotIn(key, captured["env"])
        self.assertFalse(Path(captured["cwd"]).exists())

    def test_backup_missing_invalid_usage_is_incomplete_through_summary_and_cache(self):
        for raw in (None, {}, {"input_tokens": 100},
                    {"input_tokens": 100, "output_tokens": False, "cached_input_tokens": 80}):
            process = Mock(returncode=0)
            process.communicate.return_value = (json.dumps({"type": "turn.completed", "usage": raw}), "")
            def launch(argv, **kwargs):
                Path(argv[argv.index("--output-last-message") + 1]).write_text(
                    json.dumps({"facts": ["Failure"], "unknowns": []}))
                return process
            with self.subTest(raw=raw), patch.object(engine.subprocess, "Popen", side_effect=launch), \
                    patch.object(engine.shutil, "which", return_value="/mock/codex"), \
                    patch.object(engine, "_run", side_effect=engine.ProviderFailure("provider_capacity")):
                result = self.summarize(context=str(raw))
                cached = self.summarize(context=str(raw))
            self.assertEqual(result["status"], "summary")
            self.assertTrue(result["usage_incomplete"])
            self.assertTrue(cached["cached_usage_incomplete"])

    def test_backup_oversized_output_marks_unparsed_usage_incomplete(self):
        process = Mock(returncode=0)
        process.communicate.return_value = ("x" * 256001, "")
        with patch.object(engine.subprocess, "Popen", return_value=process), \
                patch.object(engine.shutil, "which", return_value="/mock/codex"), \
                patch.object(engine, "_run", return_value=success(
                    subtype="error", is_error=True, result="HTTP 429 quota")):
            result = self.summarize()
        self.assertEqual(result["error"], "backup_output_too_long")
        self.assertTrue(result["usage_incomplete"])
        self.assertEqual(result["usage"]["total_tokens"], 110)

    def test_backup_failure_preserves_reported_usage(self):
        usage = engine._usage({"input_tokens": 5, "output_tokens": 3})
        with patch.object(engine, "_run", side_effect=engine.ProviderFailure("provider_capacity", usage)), \
                patch.object(engine, "_run_codex", side_effect=engine.ProviderFailure("backup_cli_failed", usage)):
            result = self.summarize()
        self.assertEqual(result["error"], "backup_cli_failed")
        self.assertEqual(result["usage"]["total_tokens"], 16)

    def test_interrupted_wait_stops_the_helper(self):
        with subprocess.Popen(["sleep", "30"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
            with patch.object(process, "communicate", side_effect=KeyboardInterrupt()), \
                    self.assertRaises(KeyboardInterrupt):
                engine._communicate(process, 15, "input")
            self.assertIsNotNone(process.poll())

    def test_backup_capacity_and_timeout_failures_let_the_chain_continue(self):
        for event, code in [({"type": "turn.failed", "error": {"message": "429 Too Many Requests: usage limit"}},
                             "provider_capacity"),
                            ({"type": "error", "message": "stream timed out"}, "model_timeout"),
                            ({"type": "turn.failed", "error": {"message": "401 Unauthorized"}}, "backup_cli_failed")]:
            process = Mock(returncode=1)
            process.communicate.return_value = (json.dumps(event), "")
            with self.subTest(event=event), patch.object(engine.subprocess, "Popen", return_value=process), \
                    patch.object(engine.shutil, "which", return_value="/mock/codex"), \
                    self.assertRaises(engine.ProviderFailure) as failure:
                engine._run_codex("synthetic text", 5, "gpt-5.6-sol", "low")
            self.assertEqual(failure.exception.code, code)

    def test_backup_tool_activity_is_rejected_with_final_usage_preserved(self):
        process = Mock(returncode=0)
        events = [{"type": "item.completed", "item": {"type": "command_execution"}},
                  {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 5}}]
        process.communicate.return_value = ("\n".join(json.dumps(event) for event in events), "")
        with patch.object(engine.subprocess, "Popen", return_value=process), \
                patch.object(engine.shutil, "which", return_value="/mock/codex"), \
                self.assertRaises(engine.ProviderFailure) as failure:
            engine._run_codex("synthetic text", 5, "gpt-5.6-sol", "low")
        self.assertEqual(failure.exception.code, "unexpected_backup_activity")
        self.assertEqual(failure.exception.usage["total_tokens"], 105)



def claude_output(payload=None, *, is_error=False, subtype="success", result="", denials=None,
                  usage=None, model="claude-haiku-4-5"):
    wrapper = {"type": "result", "subtype": subtype, "is_error": is_error, "result": result,
               "num_turns": 2, "permission_denials": denials or [], "total_cost_usd": 0.0123,
               "usage": usage if usage is not None else {
                   "input_tokens": 900, "output_tokens": 60, "cache_read_input_tokens": 100,
                   "cache_creation_input_tokens": 0, "output_tokens_details": {"thinking_tokens": 0}},
               "modelUsage": {model: {"inputTokens": 900}}}
    if payload is not None:
        wrapper["structured_output"] = payload
    return json.dumps(wrapper)


class ClaudeHelperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.state = Path(self.temporary.name) / "state"
        self.source = "src/app.py:12: ValueError: missing name\n"
        self.payload = {"facts": ["src/app.py:12: ValueError: missing name"], "unknowns": []}
        # Never read the real login file in tests.
        self.token_file = Path(self.temporary.name) / "claude-token.txt"
        self.token_patch = patch.object(engine, "CLAUDE_TOKEN_FILE", self.token_file)
        self.token_patch.start()

    def tearDown(self):
        self.token_patch.stop()
        self.temporary.cleanup()

    def summarize(self, **kwargs):
        args = dict(text=self.source, category="search", context="find error", state_dir=self.state)
        args.update(kwargs)
        return engine.summarize(**args)

    def launch(self, *outputs, returncode=0):
        """Popen stand-in returning one prepared stdout per call; records argv/env/stdin."""
        calls, queue = [], list(outputs)
        def popen(argv, **kwargs):
            process = Mock(returncode=returncode, pid=1)
            def communicate(input=None, timeout=None):
                calls.append({"argv": argv, "env": kwargs["env"], "cwd": kwargs["cwd"], "input": input,
                              "timeout": timeout})
                return queue.pop(0), ""
            process.communicate.side_effect = communicate
            return process
        return popen, calls

    def test_default_chain_is_sonnet_medium_only(self):
        chain = engine.normalize_helpers(None)
        self.assertEqual([(h["provider"], h["model"], h["effort"]) for h in chain],
                         [("claude_cli", "sonnet", "medium")])

    def test_helper_config_validation(self):
        valid = [{"provider": "claude_cli", "model": "claude-sonnet-5", "effort": "low"},
                 {"provider": "openai_codex", "model": "gpt-5.6-sol", "effort": "low"},
                 {"provider": "cursor_cli", "model": "composer-2.5"}]
        self.assertEqual(len(engine.normalize_helpers(valid)), 3)
        self.assertIsNone(engine.normalize_helpers(valid)[2]["effort"])
        invalid = [[], valid + valid[:2], "claude", [{"provider": "claude_cli"}],
                   [{"provider": "bedrock", "model": "x"}],
                   [{"provider": "cursor_cli", "model": "composer-2.5", "effort": "low"}],
                   [{"provider": "claude_cli", "model": "claude-sonnet-5", "effort": "extreme"}],
                   [{"provider": "claude_cli", "model": "bad model"}],
                   [{"provider": "claude_cli", "model": "--settings"}],
                   [{"provider": "claude_cli", "model": "claude-sonnet-5", "temperature": 0}],
                   [valid[0], dict(valid[0], effort="high")]]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                engine.normalize_helpers(value)

    def test_invalid_config_returns_error_without_helper_calls(self):
        with patch.object(engine.subprocess, "Popen") as popen:
            result = self.summarize(helpers=[{"provider": "claude_cli", "model": "x", "effort": "turbo"}])
        self.assertEqual(result["error"], "invalid_helpers_config")
        popen.assert_not_called()

    def test_claude_runner_is_isolated_and_thinking_follows_effort(self):
        blocked = {"ANTHROPIC_API_KEY": "secret", "ANTHROPIC_BASE_URL": "secret", "CLAUDECODE": "1",
                   "CLAUDE_CODE_ENTRYPOINT": "cli", "OPENAI_API_KEY": "secret"}
        popen, calls = self.launch(claude_output(self.payload), claude_output(self.payload))
        with patch.dict(os.environ, {**blocked, "CLAUDE_CODE_PROMPT_CACHE_TTL": "1h"}), \
                patch.object(engine.subprocess, "Popen", side_effect=popen), \
                patch.object(engine.shutil, "which", return_value=sys.executable):
            payload, usage, incomplete, extra = engine._run_claude("SENTINEL-7f3", 30, "claude-haiku-4-5", None)
            engine._run_claude("SENTINEL-7f3", 30, "claude-sonnet-5", "low")
        off, low = calls
        self.assertEqual(payload, self.payload)
        self.assertEqual(usage["input_tokens"], 1000)
        self.assertEqual(usage["cache_read_tokens"], 100)
        self.assertEqual(usage["total_tokens"], 1060)
        self.assertFalse(incomplete)
        self.assertEqual(extra, {"observed_model": "claude-haiku-4-5", "list_cost_usd": 0.0123})
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
        for flag in ("--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "")
        self.assertEqual(json.loads(argv[argv.index("--settings") + 1]), {"disableAllHooks": True})
        self.assertEqual(argv[argv.index("--system-prompt") + 1], engine.INSTRUCTIONS)
        self.assertEqual(json.loads(argv[argv.index("--json-schema") + 1]), engine.CLAUDE_SCHEMA)
        self.assertNotIn("SENTINEL-7f3", " ".join(argv))
        for key in blocked:
            self.assertNotIn(key, off["env"])
        self.assertFalse(Path(off["cwd"]).exists())

    def test_token_file_reaches_child_env_only(self):
        token = "sk-ant-oat01-" + "A" * 40 + "_b-" + "c" * 40
        self.token_file.write_text(token[:60] + "\n   " + token[60:] + "\n")  # pasted with a wrap
        self.token_file.chmod(0o600)
        popen, calls = self.launch(claude_output(self.payload))
        with patch.object(engine.subprocess, "Popen", side_effect=popen), \
                patch.object(engine.shutil, "which", return_value=sys.executable):
            engine._run_claude("SENTINEL-7f3", 30, "claude-sonnet-5", "low")
        self.assertEqual(calls[0]["env"]["CLAUDE_CODE_OAUTH_TOKEN"], token)
        self.assertNotIn(token, " ".join(calls[0]["argv"]))
        self.assertNotIn(token, calls[0]["input"])
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", os.environ)

    def test_missing_token_file_uses_normal_login(self):
        popen, calls = self.launch(claude_output(self.payload))
        with patch.object(engine.subprocess, "Popen", side_effect=popen), \
                patch.object(engine.shutil, "which", return_value=sys.executable):
            engine._run_claude("SENTINEL-7f3", 30, "claude-sonnet-5", "low")
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
                engine._run_claude("SENTINEL-7f3", 30, "claude-sonnet-5", "low")
            self.assertEqual(failure.exception.code, code)
            popen.assert_not_called()

    def test_claude_failures_are_classified(self):
        cases = [(claude_output(is_error=True, subtype="error_during_execution", result="API Error: 529 Overloaded"), "provider_capacity"),
                 (claude_output(is_error=True, result="Claude AI usage limit reached|1790600000"), "provider_capacity"),
                 (claude_output(is_error=True, result="Invalid API key · Please run /login"), "claude_cli_failed"),
                 (claude_output(None), "invalid_summary_json"),
                 (claude_output(self.payload, denials=[{"tool_name": "Bash"}]), "unexpected_agent_activity"),
                 ("not json", "claude_cli_failed"),
                 ("[]", "invalid_claude_json")]
        for stdout, code in cases:
            popen, _ = self.launch(stdout)
            with self.subTest(code=code, stdout=stdout[:60]), patch.object(engine.subprocess, "Popen", side_effect=popen), \
                    patch.object(engine.shutil, "which", return_value=sys.executable), \
                    self.assertRaises(engine.ProviderFailure) as failure:
                engine._run_claude("captured", 30, "claude-haiku-4-5", None)
            self.assertEqual(failure.exception.code, code)

    def test_claude_capacity_falls_back_to_next_helper_and_caches_producer(self):
        popen, calls = self.launch(
            claude_output(is_error=True, subtype="error_during_execution", result="API Error: 529 Overloaded"),
            claude_output(self.payload, model="claude-sonnet-5"))
        with patch.object(engine.subprocess, "Popen", side_effect=popen), \
                patch.object(engine.shutil, "which", return_value=sys.executable):
            result = self.summarize(helpers=CLAUDE_CHAIN)
            cached = self.summarize(helpers=CLAUDE_CHAIN)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "summary")
        self.assertEqual((result["provider"], result["model"]), ("claude_cli", "claude-sonnet-5"))
        self.assertEqual(result["primary_error"], "provider_capacity")
        self.assertEqual(set(result["per_provider_usage"]),
                         {"claude_cli:claude-haiku-4-5", "claude_cli:claude-sonnet-5"})
        self.assertEqual(result["usage"]["total_tokens"], 2120)
        self.assertEqual(result["list_cost_usd"], 0.0123)
        self.assertEqual(cached["status"], "cache_hit")
        self.assertEqual(cached["model"], "claude-sonnet-5")

    def test_claude_auth_failure_never_falls_back(self):
        popen, calls = self.launch(claude_output(is_error=True, result="Not logged in · Please run /login"))
        with patch.object(engine.subprocess, "Popen", side_effect=popen), \
                patch.object(engine.shutil, "which", return_value=sys.executable):
            result = self.summarize(helpers=CLAUDE_CHAIN)
        self.assertEqual(result["error"], "claude_cli_failed")
        self.assertEqual(len(calls), 1)

    def test_chain_change_partitions_cache(self):
        popen, calls = self.launch(*[claude_output(self.payload)] * 2)
        with patch.object(engine.subprocess, "Popen", side_effect=popen), \
                patch.object(engine.shutil, "which", return_value=sys.executable):
            first = self.summarize(helpers=CLAUDE_CHAIN)
            other = self.summarize(helpers=CLAUDE_CHAIN[1:])
        self.assertNotEqual(first["cache_key"], other["cache_key"])
        self.assertEqual(len(calls), 2)

if __name__ == "__main__":
    unittest.main()
