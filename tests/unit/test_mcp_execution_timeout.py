"""Runtime MCP execution deadlines stay separate from server startup deadlines."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from praxist.core.runtimes import AgentRuntimeExecutionContext
from praxist.plugins.agent_runtimes.claude_sdk import adapter as claude_adapter
from praxist.plugins.agent_runtimes.codex_sdk import adapter as codex_adapter
from praxist.plugins.agent_runtimes.codex_sdk._mcp import mcp_configuration
from tests.unit.test_codex_sdk_adapter import _request, _SdkHarness


class McpExecutionTimeoutConfigurationTest(unittest.TestCase):
    def test_omitted_execution_timeout_preserves_sdk_default(self) -> None:
        config = mcp_configuration([{"server_name": "memory-tools"}]).config
        server = config["mcp_servers"]["memory-tools"]  # type: ignore[index]
        self.assertNotIn("tool_timeout_sec", server)
        self.assertEqual(server["startup_timeout_sec"], 30)

    def test_execution_timeout_applies_to_every_selected_server(self) -> None:
        config = mcp_configuration(
            [
                {"server_name": "memory-tools"},
                {"server_name": "task-tools", "factory": "task_tools:create_server"},
            ],
            tool_execution_timeout_seconds=28800.5,
        ).config
        for name in ("memory-tools", "task-tools"):
            server = config["mcp_servers"][name]  # type: ignore[index]
            self.assertEqual(server["tool_timeout_sec"], 28800.5)
            self.assertEqual(server["startup_timeout_sec"], 30)

    def test_invalid_execution_timeout_fails_instead_of_using_sdk_default(self) -> None:
        for value in (True, 0, -1, float("nan"), float("inf"), "28800"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "tool_execution_timeout_seconds"),
            ):
                mcp_configuration(
                    [{"server_name": "memory-tools"}],
                    tool_execution_timeout_seconds=value,  # type: ignore[arg-type]
                )


class McpExecutionTimeoutRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_codex_request_reaches_native_tool_timeout_config(self) -> None:
        runtime = codex_adapter.CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(codex_adapter, "_load_sdk", side_effect=harness.sdk),
        ):
            request = _request(
                tmp,
                runtime_options={"tool_execution_timeout_seconds": 28800},
                tool_servers=[{"server_name": "memory-tools"}],
            )
            result = await runtime.execute(
                request,
                AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "test-key"}),
            )
        self.assertTrue(result.success, result.error)
        server = harness.clients[0].thread_calls[0]["config"]["mcp_servers"]["memory-tools"]
        self.assertEqual(server["tool_timeout_sec"], 28800)
        self.assertEqual(server["startup_timeout_sec"], 30)

    async def _execute_claude(
        self,
        runtime_options: dict[str, Any],
        *,
        env: dict[str, str] | None = None,
        servers: dict[str, Any] | None = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []

        class ResultMessage:
            result = {"status": "done"}
            is_error = False
            errors: list[str] = []
            usage: dict[str, int] = {}

        def options(**kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(**kwargs)

        async def query(prompt: str, options: Any) -> Any:
            yield ResultMessage()

        sdk = {
            "ClaudeAgentOptions": options,
            "HookMatcher": SimpleNamespace,
            "query": query,
            "AssistantMessage": type("AssistantMessage", (), {}),
            "ResultMessage": ResultMessage,
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(claude_adapter, "_load_claude_sdk", return_value=sdk),
        ):
            request = _request(
                tmp,
                runtime_ref="agent_runtime:claude_sdk",
                runtime_options=runtime_options,
            )
            result = await claude_adapter.ClaudeSdkAgentRuntime().execute(
                request,
                AgentRuntimeExecutionContext(
                    env=dict(env or {}),
                    tool_servers=servers
                    if servers is not None
                    else {"task-tools": {"type": "sdk", "name": "task-tools"}},
                ),
            )
        return result, calls

    async def test_claude_execution_timeout_uses_milliseconds_without_startup_change(self) -> None:
        original_env = {"MCP_TOOL_TIMEOUT": "5000", "MCP_TIMEOUT": "30000"}
        result, calls = await self._execute_claude(
            {"tool_execution_timeout_seconds": 28800.5}, env=original_env
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(calls[0]["env"]["MCP_TOOL_TIMEOUT"], "28800500")
        self.assertEqual(calls[0]["env"]["MCP_TIMEOUT"], "30000")
        self.assertEqual(original_env["MCP_TOOL_TIMEOUT"], "5000")

    async def test_claude_external_server_timeout_overrides_do_not_shorten_the_policy(self) -> None:
        sdk_instance = object()
        servers = {
            "sdk": {"type": "sdk", "name": "sdk", "instance": sdk_instance},
            "stdio": {"command": "python", "args": ["tool.py"], "timeout": 1000},
            "http": {"type": "http", "url": "http://localhost/tools", "timeout": 2000},
            "sse": {"type": "sse", "url": "http://localhost/sse"},
        }
        result, calls = await self._execute_claude(
            {"tool_execution_timeout_seconds": 28800}, servers=servers
        )
        self.assertTrue(result.success, result.error)
        selected = calls[0]["mcp_servers"]
        self.assertNotIn("timeout", selected["sdk"])
        self.assertIs(selected["sdk"]["instance"], sdk_instance)
        for name in ("stdio", "http", "sse"):
            self.assertEqual(selected[name]["timeout"], 28800000)
        self.assertEqual(servers["stdio"]["timeout"], 1000)
        self.assertEqual(servers["http"]["timeout"], 2000)
        self.assertNotIn("timeout", servers["sse"])

    async def test_claude_unknown_transport_rejects_an_unenforceable_timeout(self) -> None:
        for descriptor in ({"type": "custom"}, {"type": []}, object()):
            with self.subTest(descriptor=descriptor):
                result, calls = await self._execute_claude(
                    {"tool_execution_timeout_seconds": 28800},
                    servers={"custom": descriptor},
                )
                self.assertFalse(result.success)
                self.assertIn("tool_execution_timeout_seconds", result.error or "")
                self.assertEqual(calls, [])

    async def test_claude_supported_boundaries_and_rounding_keep_the_requested_allowance(
        self,
    ) -> None:
        for seconds, milliseconds in ((1, "1000"), (1.0001, "1001"), (2147483.647, "2147483647")):
            with self.subTest(seconds=seconds):
                result, calls = await self._execute_claude(
                    {"tool_execution_timeout_seconds": seconds}
                )
                self.assertTrue(result.success, result.error)
                self.assertEqual(calls[0]["env"]["MCP_TOOL_TIMEOUT"], milliseconds)

    async def test_claude_omitted_timeout_preserves_environment_and_sdk_default(self) -> None:
        for env in ({}, {"MCP_TOOL_TIMEOUT": "120000"}):
            with self.subTest(env=env):
                result, calls = await self._execute_claude({}, env=env)
                self.assertTrue(result.success, result.error)
                self.assertEqual(
                    calls[0]["env"].get("MCP_TOOL_TIMEOUT"), env.get("MCP_TOOL_TIMEOUT")
                )

    async def test_claude_rejects_values_outside_native_execution_timeout_range(self) -> None:
        for value in (True, 0, -1, 0.5, float("nan"), float("inf"), 2147484, "28800"):
            with self.subTest(value=value):
                result, calls = await self._execute_claude(
                    {"tool_execution_timeout_seconds": value}
                )
                self.assertFalse(result.success)
                self.assertIn("tool_execution_timeout_seconds", result.error or "")
                self.assertEqual(calls, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
