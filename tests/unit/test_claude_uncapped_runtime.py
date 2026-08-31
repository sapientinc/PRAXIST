"""Uncapped Claude sessions preserve results without imposing a fallback deadline."""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from praxist.core.runtimes import AgentRunResult, AgentRuntimeExecutionContext
from praxist.plugins.agent_runtimes.claude_sdk import adapter
from praxist.plugins.agent_runtimes.claude_sdk._mcp import mcp_execution_options
from tests.unit.test_codex_sdk_adapter import _request


class ClaudeUncappedRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def _execute_past_deadline(
        self, timeout_seconds: int | None
    ) -> tuple[AgentRunResult, list[Any], list[dict[str, Any]]]:
        options_seen: list[dict[str, Any]] = []
        observed: list[Any] = []
        elapsed = 0.0

        class TextBlock:
            text = "Retained evidence."

        class AssistantMessage:
            content = [TextBlock()]

        class ResultMessage:
            result = {"status": "done"}
            is_error = False
            errors: list[str] = []
            usage = {"input_tokens": 7, "output_tokens": 3}

        def options(**kwargs: Any) -> SimpleNamespace:
            options_seen.append(kwargs)
            return SimpleNamespace(**kwargs)

        async def query(prompt: str, options: Any) -> Any:
            nonlocal elapsed
            yield AssistantMessage()
            await asyncio.sleep(0.03)
            # Advance only the adapter clock; keep asyncio and thread scheduling real.
            elapsed = 7200.0
            await asyncio.sleep(0.03)
            yield ResultMessage()

        sdk = {
            "ClaudeAgentOptions": options,
            "HookMatcher": SimpleNamespace,
            "query": query,
            "AssistantMessage": AssistantMessage,
            "ResultMessage": ResultMessage,
            "ToolUseBlock": type("ToolUseBlock", (), {}),
        }
        clock = SimpleNamespace(monotonic=lambda: time.monotonic() + elapsed, time=time.time)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_claude_sdk", return_value=sdk),
            patch.object(adapter, "time", clock),
            patch.object(
                adapter, "_SDK_LIVENESS_POLL_SECONDS", 0.005 if timeout_seconds is None else 0.2
            ),
            patch.object(adapter, "_SDK_STREAM_POLL_SECONDS", 0.005),
        ):
            request = replace(
                _request(
                    tmp,
                    runtime_ref="agent_runtime:claude_sdk",
                    runtime_options={"tool_execution_timeout_seconds": None},
                ),
                timeout_seconds=timeout_seconds,
            )
            result = await asyncio.wait_for(
                adapter.ClaudeSdkAgentRuntime().execute(
                    request,
                    AgentRuntimeExecutionContext(message_callback=observed.append),
                ),
                timeout=2,
            )
        return result, observed, options_seen

    async def test_explicit_none_keeps_both_deadlines_uncapped_and_preserves_events(self) -> None:
        result, observed, options_seen = await self._execute_past_deadline(None)
        self.assertTrue(result.success, result.error)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.terminal_status, "completed")
        self.assertEqual(
            [type(message).__name__ for message in observed], ["AssistantMessage", "ResultMessage"]
        )
        self.assertIn(
            "Retained evidence.",
            [
                event.payload.get("text")
                for event in result.events
                if event.type == "assistant_text"
            ],
        )
        self.assertEqual(result.usage["input_tokens"], 7)
        self.assertNotIn("max_turns", options_seen[0])
        self.assertNotIn("MCP_TOOL_TIMEOUT", options_seen[0]["env"])

    async def test_explicit_numeric_deadline_still_times_out_and_preserves_partial_evidence(
        self,
    ) -> None:
        result, observed, _ = await self._execute_past_deadline(1)
        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.terminal_status, "timeout")
        self.assertEqual([type(message).__name__ for message in observed], ["AssistantMessage"])
        self.assertIn(
            "Retained evidence.",
            [
                event.payload.get("text")
                for event in result.events
                if event.type == "assistant_text"
            ],
        )

    def test_none_tool_deadline_preserves_native_configuration_and_startup_limits(self) -> None:
        for env in ({}, {"MCP_TOOL_TIMEOUT": "120000", "MCP_TIMEOUT": "30000"}):
            with self.subTest(env=env):
                servers = {
                    "sdk": {"type": "sdk", "name": "sdk"},
                    "stdio": {"command": "python", "timeout": 15000},
                }
                selected, selected_env = mcp_execution_options(servers, env, None)
                self.assertEqual(selected, servers)
                self.assertEqual(selected_env, env)
                self.assertIsNot(selected, servers)
                self.assertIsNot(selected_env, env)


if __name__ == "__main__":
    unittest.main()
