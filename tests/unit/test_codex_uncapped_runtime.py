"""Nullable request deadlines leave Codex execution uncapped and cancellable."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from praxist.core.runtimes import AgentRuntimeExecutionContext
from praxist.plugins.agent_runtimes.codex_sdk import adapter
from tests.unit.test_codex_sdk_adapter import _request, _SdkHarness


class CodexUncappedRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_none_request_timeout_preserves_optional_native_tool_limit(self) -> None:
        for tool_timeout in (None, 300):
            with self.subTest(tool_timeout=tool_timeout):
                runtime = adapter.CodexSdkRuntime()
                self.addAsyncCleanup(runtime.aclose)
                harness = _SdkHarness()
                with (
                    tempfile.TemporaryDirectory() as tmp,
                    patch.object(adapter, "_load_sdk", side_effect=harness.sdk),
                ):
                    request = replace(
                        _request(
                            tmp,
                            tool_servers=[{"server_name": "memory-tools"}],
                            runtime_options={"tool_execution_timeout_seconds": tool_timeout},
                        ),
                        timeout_seconds=None,
                    )
                    result = await runtime.execute(
                        request,
                        AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "test-key"}),
                    )
                self.assertTrue(result.success, result.error)
                self.assertFalse(result.timed_out)
                server = harness.clients[0].thread_calls[0]["config"]["mcp_servers"]["memory-tools"]
                if tool_timeout is None:
                    self.assertNotIn("tool_timeout_sec", server)
                else:
                    self.assertEqual(server["tool_timeout_sec"], tool_timeout)
                self.assertEqual(server["startup_timeout_sec"], 30)

    async def test_none_request_timeout_still_honors_operator_stop(self) -> None:
        runtime = adapter.CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(turn_mode="interruptible")

        def stop_after_start() -> bool:
            return bool(harness.clients and harness.clients[0].turns)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=harness.sdk),
        ):
            result = await runtime.execute(
                replace(_request(tmp), timeout_seconds=None),
                AgentRuntimeExecutionContext(
                    env={"OPENAI_API_KEY": "test-key"},
                    stop_requested=stop_after_start,
                ),
            )
        self.assertTrue(result.cancelled)
        self.assertFalse(result.timed_out)
        self.assertEqual(harness.clients[0].turns[0].interrupt_calls, 1)


if __name__ == "__main__":
    unittest.main()
