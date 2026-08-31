"""Optional cohort deadlines preserve event-driven closure and operator cancellation."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner, synthesis_trigger


class UnboundedCohortTests(unittest.IsolatedAsyncioTestCase):
    def _loop(self, root: Path, *, trigger_enabled: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            task_spec=SimpleNamespace(
                research_loop={"mode": "scoreless"},
                generation_policy=SimpleNamespace(cohort_size=1, per_generation_hours=None),
                synthesis_trigger=SimpleNamespace(
                    enabled=trigger_enabled,
                    min_findings=1,
                    min_interval_minutes=1,
                    max_interval_minutes=60,
                    min_contributing_peers=1,
                    poll_interval_seconds=1,
                ),
                agent=SimpleNamespace(premium_mode=False),
            ),
            run_dir=root / "run",
            workspace=root,
            base_template=root / "base.jinja2",
            task_prompt_path=root / "task.jinja2",
            gen_template=root / "gen.jinja2",
            findings_dir=root / "findings",
            model="offline",
            local_mode=True,
            mcp_servers={},
            _peer_allowed_tools=["Read"],
            plugin_registry=None,
            _findings_sync=None,
            _task_lifecycle=SimpleNamespace(remaining_seconds=lambda: None),
            _build_prompt_context=lambda generation, peer, cohort: {"peer": peer},
            _persist_prompt_layout_artifacts=lambda **kwargs: kwargs["manifest"],
        )

    def _patches(self, peer_type, trigger_type) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(cohort_runner, "AutonomousAgentLoop", peer_type))
        stack.enter_context(
            patch.object(cohort_runner, "resolve_prompt_with_layout", return_value=("prompt", {}))
        )
        stack.enter_context(
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                trigger_type,
            )
        )
        stack.enter_context(
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.protected_pids.list_active_jobs",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler_client.freeze_generation"
            )
        )
        return stack

    def test_unbounded_peers_keep_only_the_configured_trigger_close_horizon(self) -> None:
        configured = SimpleNamespace(enabled=True, max_interval_minutes=60, adaptive={})
        self.assertEqual(
            cohort_runner._effective_generation_cap_seconds(
                configured, per_peer_safety_seconds=None
            ),
            3600,
        )
        configured.enabled = False
        self.assertIsNone(
            cohort_runner._effective_generation_cap_seconds(
                configured, per_peer_safety_seconds=None
            )
        )

    def test_enabled_untimed_trigger_adds_no_default_or_adaptive_horizon(self) -> None:
        configured = SimpleNamespace(
            enabled=True,
            max_interval_minutes=None,
            adaptive={"enabled": True, "max_interval_ceiling_minutes": 120},
        )
        self.assertIsNone(
            cohort_runner._effective_generation_cap_seconds(
                configured, per_peer_safety_seconds=None
            )
        )
        self.assertEqual(
            cohort_runner._effective_generation_cap_seconds(configured, per_peer_safety_seconds=10),
            10,
        )

    async def _run_waiting_peer(
        self,
        *,
        trigger_enabled: bool = False,
        trigger_interval: float | None = 60,
        cancel: bool = False,
        per_generation_hours: float | None = None,
        lifecycle_seconds: float | None = None,
    ) -> tuple[list[dict], list[float | None], list[float | None], list[str]]:
        started = asyncio.Event()
        release = asyncio.Event()
        trigger_release = asyncio.Event()
        peer_limits: list[float | None] = []
        scheduler_deadlines: list[float | None] = []
        stops: list[str] = []

        class Peer:
            def __init__(self, *, peer_id, max_runtime_seconds, **kwargs):
                self.peer_id = peer_id
                peer_limits.append(max_runtime_seconds)

            async def run(self):
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    return {"peer_id": self.peer_id, "status": "cancelled", "partial": "kept"}
                return {"peer_id": self.peer_id, "status": "completed", "evidence": "kept"}

        class Trigger:
            def __init__(self, *, gen_dir, **kwargs):
                self.gen_dir = gen_dir
                self.fired = False
                self.closing = False
                self.required_mature_result_peers = 0
                self.mature_result_count = lambda: 0

            async def wait_until_fire(self, abort_event):
                await trigger_release.wait()
                self.fired = True
                (self.gen_dir / "STOP_SIGNAL").write_text("findings_ready", encoding="utf-8")

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="peers_finished")

            def write_postgen_marker(self, snapshot):
                (self.gen_dir / "STOP_SIGNAL_POSTGEN").write_text(snapshot.reason, encoding="utf-8")

            def fire_deadline(self, reason="generation_wall_timeout"):
                self.fired = True
                stops.append(reason)
                (self.gen_dir / "STOP_SIGNAL").write_text(reason, encoding="utf-8")

        with tempfile.TemporaryDirectory() as temporary:
            loop = self._loop(Path(temporary), trigger_enabled=trigger_enabled)
            loop.task_spec.synthesis_trigger.max_interval_minutes = trigger_interval
            loop.task_spec.generation_policy.per_generation_hours = per_generation_hours
            loop._experiment_scheduler = SimpleNamespace(
                open_generation=lambda generation, *, deadline, **kwargs: (
                    scheduler_deadlines.append(deadline)
                ),
                configure_generation_maturity=lambda *args, **kwargs: None,
            )
            if lifecycle_seconds is not None:
                deadline = time.monotonic() + lifecycle_seconds
                loop._task_lifecycle.remaining_seconds = lambda: max(0, deadline - time.monotonic())
            with (
                self._patches(Peer, Trigger),
                patch.object(cohort_runner, "_PEER_DRAIN_GRACE_SECONDS", 0),
            ):
                watchdog_patch = (
                    patch.object(
                        cohort_runner,
                        "_start_generation_deadline_watchdog",
                        side_effect=AssertionError("unbounded generation started a watchdog"),
                    )
                    if (not trigger_enabled or trigger_interval is None)
                    and per_generation_hours is None
                    and lifecycle_seconds is None
                    else ExitStack()
                )
                with watchdog_patch:
                    cohort = asyncio.create_task(cohort_runner.run_generation_cohort(loop, 0))
                    started_task = asyncio.create_task(started.wait())
                    try:
                        done, _ = await asyncio.wait(
                            {cohort, started_task}, timeout=1, return_when=asyncio.FIRST_COMPLETED
                        )
                        if cohort in done:
                            await cohort
                        self.assertTrue(started.is_set(), "cohort did not start its peer")
                        if trigger_enabled:
                            trigger_release.set()
                        elif cancel:
                            cohort.cancel()
                        elif per_generation_hours is None and lifecycle_seconds is None:
                            done, _ = await asyncio.wait({cohort}, timeout=0.03)
                            self.assertFalse(done, "unbounded peer was stopped without an event")
                            release.set()
                        if cancel:
                            with self.assertRaises(asyncio.CancelledError):
                                await asyncio.wait_for(cohort, timeout=0.5)
                        else:
                            await asyncio.wait_for(cohort, timeout=1.5)
                        results = json.loads(
                            (loop.run_dir / "gen_0/generation_results.json").read_text(
                                encoding="utf-8"
                            )
                        )
                    finally:
                        release.set()
                        trigger_release.set()
                        started_task.cancel()
                        if not cohort.done():
                            cohort.cancel()
                        await asyncio.gather(cohort, started_task, return_exceptions=True)
        return results, peer_limits, scheduler_deadlines, stops

    async def test_disabled_trigger_waits_for_unbounded_peer_without_a_watchdog(self) -> None:
        results, limits, deadlines, stops = await self._run_waiting_peer()
        self.assertEqual(results[0]["status"], "completed")
        self.assertEqual(limits, [None])
        self.assertEqual(deadlines, [None])
        self.assertEqual(stops, [])

    async def test_unbounded_cohort_cancellation_preserves_peer_output(self) -> None:
        results, limits, deadlines, stops = await self._run_waiting_peer(cancel=True)
        self.assertEqual(results[0]["partial"], "kept")
        self.assertEqual(limits, [None])
        self.assertEqual(deadlines, [None])
        self.assertEqual(stops, ["cohort_cancelled"])

    async def test_enabled_trigger_still_closes_unbounded_peers(self) -> None:
        before = time.time()
        results, limits, deadlines, _ = await self._run_waiting_peer(trigger_enabled=True)
        self.assertEqual(results[0]["status"], "cancelled")
        self.assertEqual(limits, [None])
        self.assertGreaterEqual(deadlines[0], before + 3600)
        self.assertLessEqual(deadlines[0], time.time() + 3600)

    async def test_untimed_enabled_trigger_still_closes_peers_without_a_watchdog(self) -> None:
        results, limits, deadlines, _ = await self._run_waiting_peer(
            trigger_enabled=True, trigger_interval=None
        )
        self.assertEqual(results[0]["status"], "cancelled")
        self.assertEqual(limits, [None])
        self.assertEqual(deadlines, [None])

    async def test_explicit_peer_cap_still_closes_disabled_trigger_cohort(self) -> None:
        results, limits, deadlines, _ = await self._run_waiting_peer(per_generation_hours=0.00001)
        self.assertEqual(results[0]["status"], "cancelled")
        self.assertEqual(limits, [1])
        self.assertIsNotNone(deadlines[0])

    async def test_lifecycle_deadline_bounds_otherwise_unbounded_peers(self) -> None:
        results, limits, deadlines, _ = await self._run_waiting_peer(lifecycle_seconds=0.03)
        self.assertEqual(results[0]["status"], "cancelled")
        self.assertEqual(limits, [1])
        self.assertIsNotNone(deadlines[0])

    async def test_cancelling_unbounded_postgeneration_evaluation_preserves_peer_results(
        self,
    ) -> None:
        evaluation_started = asyncio.Event()
        evaluation_cancelled = asyncio.Event()
        release = asyncio.Event()

        class Peer:
            def __init__(self, *, peer_id, **kwargs):
                self.peer_id = peer_id

            async def run(self):
                return {"peer_id": self.peer_id, "success": True, "evidence": "completed research"}

        class Trigger:
            def __init__(self, *, gen_dir, **kwargs):
                self.gen_dir = gen_dir
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                await abort_event.wait()

            async def evaluate_async(self):
                evaluation_started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    evaluation_cancelled.set()
                    raise
                return SimpleNamespace(fired=False, reason="peers_finished")

            def fire_deadline(self, reason="generation_wall_timeout"):
                self.fired = True
                (self.gen_dir / "STOP_SIGNAL").write_text(reason, encoding="utf-8")

        with tempfile.TemporaryDirectory() as temporary:
            loop = self._loop(Path(temporary), trigger_enabled=True)
            loop.task_spec.synthesis_trigger.max_interval_minutes = None
            with self._patches(Peer, Trigger):
                cohort = asyncio.create_task(cohort_runner.run_generation_cohort(loop, 0))
                try:
                    await asyncio.wait_for(evaluation_started.wait(), timeout=0.5)
                    cohort.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(cohort, timeout=0.5)
                    self.assertTrue(evaluation_cancelled.is_set())
                    results_path = loop.run_dir / "gen_0/generation_results.json"
                    self.assertTrue(
                        results_path.exists(),
                        "post-generation cancellation discarded completed peer research",
                    )
                    self.assertEqual(
                        json.loads(results_path.read_text(encoding="utf-8"))[0]["evidence"],
                        "completed research",
                    )
                    self.assertEqual(
                        (loop.run_dir / "gen_0/STOP_SIGNAL").read_text(encoding="utf-8"),
                        "cohort_cancelled",
                    )
                finally:
                    release.set()
                    if not cohort.done():
                        cohort.cancel()
                    await asyncio.gather(cohort, return_exceptions=True)

    async def _completion_event_closes_after_protected_work(self, *, protected_work: bool) -> None:
        waiting_for_event = asyncio.Event()
        completion_rechecked = asyncio.Event()
        release = asyncio.Event()
        active_protected_work = protected_work
        waits = 0

        class Peer:
            def __init__(self, *, peer_id, **kwargs):
                self.peer_id = peer_id

            async def run(self):
                await waiting_for_event.wait()
                return {"peer_id": self.peer_id, "success": True}

        async def quiet_filesystem(paths, **kwargs):
            nonlocal waits
            waits += 1
            waiting_for_event.set()
            if waits > 1:
                completion_rechecked.set()
            await release.wait()
            return SimpleNamespace(reason="event")

        with tempfile.TemporaryDirectory() as temporary:
            loop = self._loop(Path(temporary), trigger_enabled=True)
            loop.task_spec.synthesis_trigger.max_interval_minutes = None
            loop.task_spec.synthesis_trigger.min_interval_minutes = 0
            loop.task_spec.synthesis_trigger.poll_interval_seconds = 3600
            loop.run_dir.mkdir()
            with sqlite3.connect(loop.run_dir / "shared_store.db") as connection:
                connection.execute(
                    "CREATE TABLE findings (id TEXT, peer_id TEXT, generation_id INTEGER)"
                )
                connection.execute("INSERT INTO findings VALUES ('one', 'gen0_peer0', 0)")
            with (
                self._patches(Peer, synthesis_trigger.SynthesisTrigger),
                patch.object(synthesis_trigger, "wait_for_filesystem_event", quiet_filesystem),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "protected_pids.list_active_jobs",
                    side_effect=lambda **kwargs: (
                        [SimpleNamespace(peer_id="gen0_peer0", pid=12345, eta_seconds=1)]
                        if active_protected_work
                        else []
                    ),
                ),
            ):
                cohort = asyncio.create_task(cohort_runner.run_generation_cohort(loop, 0))
                try:
                    if protected_work:
                        await asyncio.wait_for(completion_rechecked.wait(), timeout=0.5)
                        self.assertFalse(cohort.done())
                        self.assertFalse((loop.run_dir / "gen_0/STOP_SIGNAL").exists())
                        active_protected_work = False
                        release.set()
                    done, _ = await asyncio.wait({cohort}, timeout=0.5)
                    self.assertIn(
                        cohort,
                        done,
                        "completed peers waited for the trigger heartbeat instead of closing",
                    )
                    self.assertTrue((await cohort)[0]["success"])
                    self.assertIn(
                        "trigger_reason=info_density",
                        (loop.run_dir / "gen_0/STOP_SIGNAL").read_text(encoding="utf-8"),
                    )
                finally:
                    release.set()
                    if not cohort.done():
                        cohort.cancel()
                    await asyncio.gather(cohort, return_exceptions=True)

    async def test_completed_cohort_rechecks_a_closing_trigger_without_waiting_for_heartbeat(
        self,
    ) -> None:
        await self._completion_event_closes_after_protected_work(protected_work=False)

    async def test_completion_event_does_not_bypass_protected_background_work(self) -> None:
        await self._completion_event_closes_after_protected_work(protected_work=True)


if __name__ == "__main__":
    unittest.main()
