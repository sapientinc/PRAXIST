"""Absent scoreless budgets stay absent through peer and runtime dispatch."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from praxist.core.execution_policy import task_execution_deadline_scope
from praxist.core.protocol import AgentRunRequest
from praxist.core.run_config import DEFAULT_FULL_AUTO_MAX_RUNTIME_SECONDS
from praxist.plugins.workflow_stages.research_loop.backend import agent


def _make_loop(root: Path, *, scoreless: bool = True, **limits: Any) -> agent.AutonomousAgentLoop:
    findings = root / "run" / "shared_findings"
    findings.mkdir(parents=True)
    return agent.AutonomousAgentLoop(
        peer_id="gen0_peer0",
        generation_id=0,
        task_prompt="Inspect the available evidence.",
        workspace=root,
        logs_dir=root / "run" / "gen_0" / "gen0_peer0",
        findings_dir=findings,
        stop_signal_path=root / "run" / "STOP_SIGNAL",
        local_mode=True,
        task_spec=SimpleNamespace(research_loop={"mode": "scoreless" if scoreless else "metric"}),
        **limits,
    )


def _make_agent(root: Path, **limits: Any) -> agent.BaseAgent:
    return agent.BaseAgent(
        name="offline-research",
        workspace=root,
        allowed_tools=["Read"],
        mcp_servers={},
        execution_role="research",
        **limits,
    )


class AgentOptionalLimitTests(unittest.IsolatedAsyncioTestCase):
    def assert_no_invented_budgets(self, request: AgentRunRequest) -> None:
        self.assertIsNone(request.budget_grant_id)
        budget_keys = {
            "max_turns",
            "max_sessions",
            "max_tokens",
            "max_output_tokens",
            "max_tool_calls",
            "token_budget",
            "session_budget",
            "tool_budget",
            "tool_execution_timeout_seconds",
        }
        for surface in (request.runtime_options, request.model_call.parameters):
            self.assertFalse(budget_keys.intersection(surface), surface)

    def test_scoreless_omitted_and_null_runtime_limits_remain_unbounded(self) -> None:
        cases: tuple[dict[str, Any], ...] = ({}, {"max_runtime_seconds": None})
        for limits in cases:
            with self.subTest(limits=limits), tempfile.TemporaryDirectory() as tmp:
                loop = _make_loop(Path(tmp), **limits)
                self.assertIsNone(loop.max_runtime_seconds)
                self.assertIsNone(loop.stop_checker.max_runtime)
                with patch.object(agent.time, "time", return_value=10**12):
                    self.assertIsNone(loop.stop_checker.check())
                    self.assertIsNone(loop._remaining_runtime_seconds())
                    assert loop.stop_signal_path is not None
                    loop.stop_signal_path.touch()
                    self.assertEqual(loop.stop_checker.check(), agent.StopReason.SYNTHESIS_TRIGGER)

    def test_metric_default_and_explicit_scoreless_runtime_limit_are_preserved(self) -> None:
        for scoreless, limits, expected in (
            (False, {}, DEFAULT_FULL_AUTO_MAX_RUNTIME_SECONDS),
            (False, {"max_runtime_seconds": None}, DEFAULT_FULL_AUTO_MAX_RUNTIME_SECONDS),
            (True, {"max_runtime_seconds": 17}, 17),
        ):
            with (
                self.subTest(scoreless=scoreless, limits=limits),
                tempfile.TemporaryDirectory() as tmp,
            ):
                loop = _make_loop(Path(tmp), scoreless=scoreless, **limits)
                self.assertEqual(loop.max_runtime_seconds, expected)
                with patch.object(
                    agent.time, "time", return_value=loop.stop_checker.start_time + 5
                ):
                    self.assertIsNone(loop.stop_checker.check())
                    self.assertEqual(loop._remaining_runtime_seconds(), expected - 5)
                    request = loop._create_agent("session")._build_agent_run_request("Inspect.", {})
                    self.assertEqual(request.timeout_seconds, expected - 5)
                with patch.object(
                    agent.time, "time", return_value=loop.stop_checker.start_time + expected
                ):
                    self.assertEqual(loop.stop_checker.check(), agent.StopReason.TIMEOUT)

    def test_request_timeout_preserves_absence_and_explicit_positive_values(self) -> None:
        for limits, env, expected in (
            ({}, {}, None),
            ({"runtime_timeout_seconds": None}, {}, None),
            ({}, {"PRAXIST_AGENT_TIMEOUT_SECONDS": "0"}, None),
            ({"runtime_timeout_seconds": 0}, {}, None),
            ({}, {"PRAXIST_AGENT_TIMEOUT_SECONDS": "17"}, 17),
            ({"runtime_timeout_seconds": 9}, {"PRAXIST_AGENT_TIMEOUT_SECONDS": "17"}, 9),
        ):
            with (
                self.subTest(limits=limits, env=env),
                tempfile.TemporaryDirectory() as tmp,
                patch.dict(os.environ, env, clear=True),
            ):
                request = _make_agent(Path(tmp), **limits)._build_agent_run_request("Inspect.", {})
                self.assertEqual(request.timeout_seconds, expected)
                self.assertEqual(request.to_dict()["timeout_seconds"], expected)
                self.assert_no_invented_budgets(request)

    def test_explicit_deadline_limits_an_otherwise_unbounded_request(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {}, clear=True),
            patch("praxist.core.execution_policy.time.time", return_value=1000.0),
            task_execution_deadline_scope(1007.25),
        ):
            request = _make_agent(Path(tmp))._build_agent_run_request("Inspect.", {})
            self.assertEqual(request.timeout_seconds, 8)
            self.assert_no_invented_budgets(request)

    async def test_scoreless_bootstrap_retry_keeps_both_requests_unbounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            loop = _make_loop(Path(tmp))
            requests: list[AgentRunRequest] = []

            async def execute(base: agent.BaseAgent, task: str) -> agent.AgentResult:
                requests.append(base._build_agent_run_request(task, {}))
                retry = len(requests) == 2
                return agent.AgentResult(
                    success=True,
                    output={
                        "text_outputs": [
                            "Evidence inspected." if retry else "Waiting for your instruction."
                        ]
                    },
                    duration=1.0,
                    iteration_count=1 if retry else 0,
                    usage={"input_tokens": 3.0 if retry else 7.0},
                )

            with patch.object(agent.BaseAgent, "execute", new=execute):
                result = await loop._run_session()

            self.assertTrue(result.success)
            self.assertEqual(result.usage, {"input_tokens": 10.0})
            self.assertEqual(len(requests), 2)
            for request in requests:
                self.assertIsNone(request.timeout_seconds)
                self.assertEqual(request.runtime_options["execution_role"], "research")
                self.assert_no_invented_budgets(request)

    async def test_unbounded_peer_still_preserves_cancellation_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            loop = _make_loop(Path(tmp))
            started = asyncio.Event()
            never = asyncio.Event()
            requests: list[AgentRunRequest] = []

            async def execute(base: agent.BaseAgent, task: str) -> agent.AgentResult:
                requests.append(base._build_agent_run_request(task, {}))
                started.set()
                await never.wait()
                raise AssertionError("The active session should be cancelled")

            with patch.object(agent.BaseAgent, "execute", new=execute):
                running = asyncio.create_task(loop.run())
                try:
                    await asyncio.wait_for(started.wait(), timeout=2)
                    running.cancel("parent deadline")
                    with self.assertRaises(asyncio.CancelledError) as caught:
                        await asyncio.wait_for(running, timeout=2)
                    self.assertEqual(str(caught.exception), "parent deadline")
                finally:
                    if not running.done():
                        running.cancel()
                    await asyncio.gather(running, return_exceptions=True)

            self.assertEqual(len(requests), 1)
            self.assertIsNone(requests[0].timeout_seconds)
            assert loop.last_result is not None
            self.assertEqual(loop.last_result["stop_reason"], "interrupted")
            self.assertEqual(loop.last_result["sessions"], 0)
            persisted = json.loads((loop.logs_dir / "peer_result.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted, loop.last_result)
