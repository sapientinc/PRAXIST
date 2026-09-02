"""Run-scoped Codex relay lifecycle and secret-boundary tests."""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from praxist.plugins.agent_runtimes.codex_sdk._relay import (
    RelayHandle,
    _available_port,
    _relay_binary,
    _wait_for_listener,
    needs_relay,
    provider_key_var,
    provider_name,
    start_relay,
)


class _Process:
    def __init__(self, *, running: bool = True, wait_times_out: bool = False) -> None:
        self.running = running
        self.wait_times_out = wait_times_out
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.running = False

    def kill(self) -> None:
        self.kill_calls += 1
        self.running = False

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.wait_times_out:
            self.wait_times_out = False
            raise subprocess.TimeoutExpired("codex-relay", timeout)
        self.running = False
        return 0


class ProviderRoutingTest(unittest.TestCase):
    def test_provider_refs_normalize_to_sdk_routing_names(self) -> None:
        self.assertEqual(provider_name("model_provider:deepseek_alias"), "deepseek")
        self.assertEqual(provider_name("model_provider:openai_compatible"), "openai")
        self.assertEqual(provider_name("model_provider:openai"), "openai")
        self.assertEqual(provider_name("model_provider:groq_alias"), "groq")
        self.assertEqual(provider_name("model_provider:mistral_alias"), "mistral")
        self.assertEqual(provider_name("model_provider:xai_alias"), "xai")

    def test_only_openai_uses_direct_responses_transport(self) -> None:
        self.assertFalse(needs_relay("openai"))
        self.assertFalse(needs_relay(""))
        self.assertTrue(needs_relay("deepseek"))
        self.assertTrue(needs_relay("groq"))
        self.assertTrue(needs_relay("mistral"))
        self.assertTrue(needs_relay("xai"))

    def test_provider_key_variables_are_explicit(self) -> None:
        self.assertEqual(provider_key_var("deepseek"), "DEEPSEEK_API_KEY")
        self.assertEqual(provider_key_var("openrouter"), "OPENROUTER_API_KEY")
        self.assertEqual(provider_key_var("groq"), "GROQ_API_KEY")
        self.assertEqual(provider_key_var("mistral"), "MISTRAL_API_KEY")
        self.assertEqual(provider_key_var("xai"), "XAI_API_KEY")
        self.assertEqual(provider_key_var("unknown"), "OPENAI_API_KEY")


class RelayLifecycleTest(unittest.TestCase):
    def test_available_port_is_ephemeral_and_released(self) -> None:
        port = _available_port()
        self.assertGreater(port, 0)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))

    def test_relay_binary_prefers_interpreter_sibling_then_path(self) -> None:
        with (
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay.Path.is_file",
                return_value=True,
            ),
            patch("praxist.plugins.agent_runtimes.codex_sdk._relay.os.access", return_value=True),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay.sys.executable", "/opt/bin/python"
            ),
        ):
            self.assertEqual(_relay_binary(), "/opt/bin/codex-relay")

        with (
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay.Path.is_file",
                return_value=False,
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay.shutil.which",
                return_value="/usr/bin/codex-relay",
            ),
        ):
            self.assertEqual(_relay_binary(), "/usr/bin/codex-relay")

    def test_listener_probe_observes_ready_and_exited_processes(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = int(listener.getsockname()[1])
            self.assertTrue(_wait_for_listener(port, _Process(), timeout=0.2))  # type: ignore[arg-type]

        self.assertFalse(
            _wait_for_listener(port, _Process(running=False), timeout=0.2)  # type: ignore[arg-type]
        )

    def test_start_relay_scopes_subprocess_environment_and_uses_ephemeral_port(self) -> None:
        process = _Process()
        captured: dict[str, Any] = {}

        def popen(args: list[str], **kwargs: Any) -> _Process:
            captured["args"] = args
            captured.update(kwargs)
            return process

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {
                    "HOME": "/home/test",
                    "PATH": "/usr/bin",
                    "LANG": "C.UTF-8",
                    "UNRELATED_API_KEY": "must-not-leak",
                    "SESSION_TOKEN": "must-not-leak",
                },
                clear=True,
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._relay_binary",
                return_value="/usr/bin/codex-relay",
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._available_port",
                return_value=41237,
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._wait_for_listener",
                return_value=True,
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay.subprocess.Popen",
                side_effect=popen,
            ),
        ):
            handle = start_relay(
                provider="deepseek",
                api_key="local-test-key",
                state_dir=Path(tmp),
            )

        self.assertEqual(handle.port, 41237)
        self.assertEqual(handle.base_url, "http://127.0.0.1:41237/v1")
        self.assertEqual(
            captured["args"],
            [
                "/usr/bin/codex-relay",
                "--port",
                "41237",
                "--upstream",
                "https://api.deepseek.com/v1",
            ],
        )
        child_env = captured["env"]
        self.assertEqual(child_env["DEEPSEEK_API_KEY"], "local-test-key")
        self.assertEqual(child_env["OPENAI_API_KEY"], "local-test-key")
        self.assertEqual(child_env["CODEX_RELAY_API_KEY"], "local-test-key")
        self.assertNotIn("UNRELATED_API_KEY", child_env)
        self.assertNotIn("SESSION_TOKEN", child_env)
        self.assertTrue(captured["start_new_session"])

    def test_openrouter_relay_adds_sticky_session_without_response_cache(self) -> None:
        process = _Process()
        captured: dict[str, Any] = {}

        def popen(args: list[str], **kwargs: Any) -> _Process:
            captured["args"] = args
            captured.update(kwargs)
            return process

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._relay_binary",
                return_value="/usr/bin/codex-relay",
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._available_port",
                return_value=41240,
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._wait_for_listener",
                return_value=True,
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay.subprocess.Popen",
                side_effect=popen,
            ),
        ):
            start_relay(
                provider="openrouter",
                api_key="openrouter-test-key",
                state_dir=Path(tmp),
                upstream_session_id="praxist-stable-session",
            )

        self.assertEqual(
            captured["args"],
            [
                "/usr/bin/codex-relay",
                "--port",
                "41240",
                "--upstream",
                "https://openrouter.ai/api/v1",
                "--upstream-extra-params",
                '{"session_id":"praxist-stable-session"}',
            ],
        )
        self.assertNotIn("cache", " ".join(captured["args"]).lower())

    def test_deepseek_relay_forwards_thinking_policy(self) -> None:
        process = _Process()
        captured: dict[str, Any] = {}

        def popen(args: list[str], **kwargs: Any) -> _Process:
            captured["args"] = args
            return process

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._relay_binary",
                return_value="/usr/bin/codex-relay",
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._available_port",
                return_value=41241,
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._wait_for_listener",
                return_value=True,
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay.subprocess.Popen",
                side_effect=popen,
            ),
        ):
            start_relay(
                provider="deepseek",
                api_key="deepseek-test-key",
                state_dir=Path(tmp),
                upstream_extra_params={
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "max",
                },
                drop_upstream_params=("unused", "reasoning_effort", "unused"),
            )

        self.assertEqual(
            captured["args"][-4:],
            [
                "--upstream-extra-params",
                '{"reasoning_effort":"max","thinking":{"type":"enabled"}}',
                "--drop-upstream-params",
                '["reasoning_effort","unused"]',
            ],
        )

    def test_listener_failure_terminates_process_and_reports_log_path(self) -> None:
        process = _Process()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._relay_binary",
                return_value="/usr/bin/codex-relay",
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._available_port",
                return_value=41238,
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._wait_for_listener",
                return_value=False,
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay.subprocess.Popen",
                return_value=process,
            ),
            self.assertRaisesRegex(RuntimeError, r"relay-deepseek-41238\.log"),
        ):
            start_relay(
                provider="deepseek",
                api_key="local-test-key",
                state_dir=Path(tmp),
            )

        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.wait_calls, [2])

    def test_missing_binary_key_and_unsupported_provider_fail_before_spawn(self) -> None:
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        with (
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._relay_binary",
                return_value=None,
            ),
            self.assertRaisesRegex(RuntimeError, "codex-relay is required"),
        ):
            start_relay(
                provider="deepseek",
                api_key="key",
                state_dir=Path(tmp),
            )

        with (
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._relay_binary",
                return_value="/usr/bin/codex-relay",
            ),
            self.assertRaisesRegex(RuntimeError, "does not support"),
        ):
            start_relay(
                provider="unsupported",
                api_key="key",
                state_dir=Path(tmp),
            )
        with (
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._relay_binary",
                return_value="/usr/bin/codex-relay",
            ),
            self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"),
        ):
            start_relay(
                provider="deepseek",
                api_key="",
                state_dir=Path(tmp),
            )

    def test_close_is_idempotent_and_escalates_after_terminate_timeout(self) -> None:
        process = _Process(wait_times_out=True)
        handle = RelayHandle(provider="deepseek", port=41239, process=process)  # type: ignore[arg-type]

        handle.close()
        handle.close()

        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertEqual(process.wait_calls, [5, 5])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
