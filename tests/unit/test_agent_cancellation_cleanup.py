from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from praxist.plugins.workflow_stages.research_loop.backend import agent


class _FindingsSyncProbe:
    def __init__(self) -> None:
        self.stopped = False

    def sync_once(self) -> int:
        return 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True


def _make_loop(root: Path) -> agent.AutonomousAgentLoop:
    findings = root / "run" / "shared_findings"
    findings.mkdir(parents=True)
    (findings / "finding.json").write_text(
        json.dumps({"finding_id": "finding", "title": "An unconsumed finding"}),
        encoding="utf-8",
    )
    return agent.AutonomousAgentLoop(
        peer_id="gen0_peer0",
        generation_id=0,
        task_prompt="Inspect the shared evidence.",
        workspace=root,
        logs_dir=root / "run" / "gen_0" / "gen0_peer0",
        findings_dir=findings,
        local_mode=True,
        max_runtime_seconds=60,
    )


class AgentCancellationCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_run_preserves_result_before_waiting_for_cleanup(self) -> None:
        async def exercise_phase(phase: str) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                loop = _make_loop(Path(tmp))
                sync = _FindingsSyncProbe()
                loop.findings_sync = sync
                loop._active_resource_supply_lease_id = "lease-test"
                phase_reached = asyncio.Event()
                cleanup_started = asyncio.Event()
                cleanup_allowed = asyncio.Event()
                never = asyncio.Event()
                calls = 0
                cleanup_completed = False
                cancellation_requested = False

                async def execute(_base: agent.BaseAgent, task: str) -> agent.AgentResult:
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return agent.AgentResult(
                            success=True,
                            output={"text_outputs": ["Evidence inspected."]},
                            duration=1.0,
                            iteration_count=1,
                            usage={"input_tokens": 7.0, "output_tokens": 3.0, "total_tokens": 10.0},
                        )
                    if phase == "billing_pause":
                        raise RuntimeError("invalid api key")
                    phase_reached.set()
                    await never.wait()
                    raise AssertionError("The blocked session should be cancelled")

                async def wait_for_next_session(*, productive: bool) -> None:
                    if phase == "idle_wait":
                        phase_reached.set()
                        await never.wait()

                async def billing_sleep(_seconds: float) -> None:
                    phase_reached.set()
                    await never.wait()

                async def release_lease(*, declined: bool = False) -> None:
                    nonlocal cleanup_completed
                    if cancellation_requested:
                        cleanup_started.set()
                        await cleanup_allowed.wait()
                        cleanup_completed = True
                    loop._active_resource_supply_lease_id = ""

                with (
                    patch.object(agent.BaseAgent, "execute", new=execute),
                    patch.object(loop, "_wait_for_next_session_event", new=wait_for_next_session),
                    patch.object(loop, "_release_active_supply_lease", new=release_lease),
                    patch.object(agent.asyncio, "sleep", new=billing_sleep),
                ):
                    running = asyncio.create_task(loop.run())
                    cleanup_waiter: asyncio.Task[bool] | None = None
                    try:
                        await asyncio.wait_for(phase_reached.wait(), timeout=2)
                        cancellation_requested = True
                        running.cancel("cohort deadline")
                        cleanup_waiter = asyncio.create_task(cleanup_started.wait())
                        await asyncio.wait(
                            {running, cleanup_waiter},
                            timeout=2,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        snapshot = getattr(loop, "last_result", None)
                        self.assertIsInstance(snapshot, dict)
                        assert isinstance(snapshot, dict)
                        self.assertEqual(snapshot["sessions"], 1)
                        self.assertEqual(snapshot["stop_reason"], "interrupted")
                        self.assertEqual(
                            snapshot["runtime_usage"],
                            {"input_tokens": 7.0, "output_tokens": 3.0, "total_tokens": 10.0},
                        )
                        self.assertEqual(snapshot["total_tokens"], 10.0)
                        persisted = json.loads(
                            (loop.logs_dir / "peer_result.json").read_text(encoding="utf-8")
                        )
                        self.assertEqual(persisted, snapshot)
                        self.assertTrue(cleanup_started.is_set())
                        self.assertFalse(running.done())
                        cleanup_allowed.set()
                        with self.assertRaises(asyncio.CancelledError):
                            await asyncio.wait_for(running, timeout=2)
                        self.assertTrue(cleanup_completed)
                        self.assertTrue(sync.stopped)
                        self.assertEqual(loop._active_resource_supply_lease_id, "")
                    finally:
                        cleanup_allowed.set()
                        if not running.done():
                            running.cancel()
                        await asyncio.gather(running, return_exceptions=True)
                        if cleanup_waiter is not None:
                            cleanup_waiter.cancel()
                            await asyncio.gather(cleanup_waiter, return_exceptions=True)

        for phase in ("active_session", "idle_wait", "billing_pause"):
            with self.subTest(phase=phase):
                await exercise_phase(phase)

    async def test_cancelled_session_records_nonempty_error_without_consuming_findings(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _make_loop(Path(tmp))
            started = asyncio.Event()
            never = asyncio.Event()

            async def execute(_base: agent.BaseAgent, task: str) -> agent.AgentResult:
                started.set()
                await never.wait()
                raise AssertionError("The blocked session should be cancelled")

            with patch.object(agent.BaseAgent, "execute", new=execute):
                running = asyncio.create_task(loop._run_session())
                await asyncio.wait_for(started.wait(), timeout=2)
                running.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(running, timeout=2)

            row = self._session_row(loop)
            self.assertFalse(row["success"])
            self.assertTrue(row["error"].strip())
            self._assert_no_findings_consumed(loop)
            self._assert_session_log_ended(loop)

    async def test_cancelled_bootstrap_retry_preserves_usage_without_recording_success(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _make_loop(Path(tmp))
            retry_started = asyncio.Event()
            never = asyncio.Event()
            calls = 0

            async def execute(_base: agent.BaseAgent, task: str) -> agent.AgentResult:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return agent.AgentResult(
                        success=True,
                        output={"text_outputs": ["Waiting for your instruction."]},
                        duration=1.0,
                        iteration_count=0,
                        usage={"input_tokens": 7.0},
                    )
                retry_started.set()
                await never.wait()
                raise AssertionError("The bootstrap retry should be cancelled")

            with patch.object(agent.BaseAgent, "execute", new=execute):
                running = asyncio.create_task(loop._run_session())
                await asyncio.wait_for(retry_started.wait(), timeout=2)
                running.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(running, timeout=2)

            row = self._session_row(loop)
            self.assertFalse(row["success"])
            self.assertTrue(row["error"].strip())
            self.assertEqual(loop.runtime_usage, {"input_tokens": 7.0})
            self._assert_no_findings_consumed(loop)
            self._assert_session_log_ended(loop)

    async def test_memory_failure_does_not_replace_session_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _make_loop(Path(tmp))
            started = asyncio.Event()
            never = asyncio.Event()

            async def execute(_base: agent.BaseAgent, task: str) -> agent.AgentResult:
                started.set()
                await never.wait()
                raise AssertionError("The blocked session should be cancelled")

            with (
                patch.object(agent.BaseAgent, "execute", new=execute),
                patch.object(
                    loop.peer_memory, "record_session_result", side_effect=OSError("disk full")
                ),
            ):
                running = asyncio.create_task(loop._run_session())
                await asyncio.wait_for(started.wait(), timeout=2)
                running.cancel("parent cancellation")
                with self.assertRaises(asyncio.CancelledError) as caught:
                    await asyncio.wait_for(running, timeout=2)
                self.assertEqual(str(caught.exception), "parent cancellation")

            self._assert_session_log_ended(loop)

    def _session_row(self, loop: agent.AutonomousAgentLoop) -> dict[str, Any]:
        memory = loop.peer_memory
        assert isinstance(memory, agent.PeerSessionMemory)
        rows = memory.ledger_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 1)
        return json.loads(rows[0])

    def _assert_no_findings_consumed(self, loop: agent.AutonomousAgentLoop) -> None:
        memory = loop.peer_memory
        assert isinstance(memory, agent.PeerSessionMemory)
        seen = (
            json.loads(memory.seen_findings_path.read_text(encoding="utf-8"))
            if memory.seen_findings_path.exists()
            else []
        )
        self.assertEqual(seen, [])

    def _assert_session_log_ended(self, loop: agent.AutonomousAgentLoop) -> None:
        logs = list(loop.logs_dir.glob("session_*.log"))
        self.assertEqual(len(logs), 1)
        self.assertIn("# Ended:", logs[0].read_text(encoding="utf-8"))
