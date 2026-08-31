"""Cohort cancellation preserves evidence without borrowing finalization time."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner


class _Deadline:
    def __init__(self, seconds: float = 30.0) -> None:
        self.deadline = time.monotonic() + seconds

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())


class CohortDeadlineCleanupTests(unittest.IsolatedAsyncioTestCase):
    def _loop(self, root: Path, *, peers: int = 1) -> SimpleNamespace:
        return SimpleNamespace(
            task_spec=SimpleNamespace(
                research_loop={"mode": "scoreless"},
                generation_policy=SimpleNamespace(cohort_size=peers, per_generation_hours=1),
                synthesis_trigger=SimpleNamespace(
                    enabled=True,
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
            _task_lifecycle=_Deadline(),
            _build_prompt_context=lambda generation, peer, cohort: {"peer": peer},
            _persist_prompt_layout_artifacts=lambda **kwargs: kwargs["manifest"],
        )

    def _patches(self, peer_type, trigger_type, *, jobs=()) -> ExitStack:
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
                return_value=list(jobs),
            )
        )
        return stack

    async def test_cancellation_drains_peers_and_persists_outputs_before_propagation(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        cancelled = asyncio.Event()

        class Peer:
            def __init__(self, *, peer_id, **kwargs):
                self.peer_id = peer_id

            async def run(self):
                if self.peer_id.endswith("0"):
                    return {"peer_id": self.peer_id, "success": True, "evidence": "retained"}
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    return {"peer_id": self.peer_id, "success": False, "partial": "captured"}

        class Trigger:
            def __init__(self, *, gen_dir, **kwargs):
                self.gen_dir = gen_dir
                self.fired = False
                self.closing = True
                self.adaptive_policy = SimpleNamespace(drain_grace_minutes=60)

            async def wait_until_fire(self, abort_event):
                await release.wait()

            async def evaluate_async(self):
                raise AssertionError("expired research must not start post-generation evaluation")

            def fire_deadline(self, reason="generation_wall_timeout"):
                self.fired = True
                (self.gen_dir / "STOP_SIGNAL").write_text(reason, encoding="utf-8")

        job = SimpleNamespace(peer_id="gen0_peer9", pid=12345, eta_seconds=3600)
        with tempfile.TemporaryDirectory() as temporary:
            loop = self._loop(Path(temporary), peers=2)
            with self._patches(Peer, Trigger, jobs=[job]):
                task = asyncio.create_task(cohort_runner.run_generation_cohort(loop, 0))
                await started.wait()
                loop._task_lifecycle.deadline = time.monotonic() - 1
                task.cancel()
                try:
                    done, _ = await asyncio.wait({task}, timeout=0.2)
                    self.assertIn(task, done, "cohort cleanup exceeded its expired deadline")
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertTrue(cancelled.is_set(), "active peer never received cancellation")
                    results = json.loads(
                        (loop.run_dir / "gen_0/generation_results.json").read_text()
                    )
                    self.assertEqual(results[0]["evidence"], "retained")
                    self.assertEqual(results[1]["partial"], "captured")
                    self.assertEqual(results[2]["status"], "late_quarantined_protected_job")
                    self.assertFalse(results[2]["promotion_eligible"])
                    self.assertTrue((loop.run_dir / "gen_0/STOP_SIGNAL").exists())
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_closing_assessment_cannot_use_finalization_reserve(self) -> None:
        release = asyncio.Event()

        class Peer:
            def __init__(self, *, peer_id, **kwargs):
                self.peer_id = peer_id

            async def run(self):
                return {"peer_id": self.peer_id, "success": True}

        class Trigger:
            def __init__(self, *, gen_dir, **kwargs):
                self.gen_dir = gen_dir
                self.fired = False
                self.closing = True
                self.adaptive_policy = SimpleNamespace(drain_grace_minutes=60)

            async def wait_until_fire(self, abort_event):
                await release.wait()
                self.fired = True

            async def evaluate_async(self):
                raise AssertionError("expired assessment must not start another evaluation")

            def fire_deadline(self, reason="generation_wall_timeout"):
                self.fired = True
                (self.gen_dir / "STOP_SIGNAL").write_text(reason, encoding="utf-8")

        with tempfile.TemporaryDirectory() as temporary:
            loop = self._loop(Path(temporary))
            loop._task_lifecycle = _Deadline(0.03)
            with self._patches(Peer, Trigger):
                task = asyncio.create_task(cohort_runner.run_generation_cohort(loop, 0))
                try:
                    done, _ = await asyncio.wait({task}, timeout=0.2)
                    self.assertIn(task, done, "closing trigger ignored remaining research time")
                    self.assertEqual((await task)[0]["peer_id"], "gen0_peer0")
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)

    async def _postgeneration_evaluation_deadline(self, *, outer_timeout: bool) -> None:
        release = asyncio.Event()
        evaluation_started = asyncio.Event()
        evaluation_cancelled = asyncio.Event()

        class Peer:
            def __init__(self, *, peer_id, **kwargs):
                self.peer_id = peer_id

            async def run(self):
                return {"peer_id": self.peer_id, "success": True}

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
                return SimpleNamespace(fired=False, reason="complete")

            def fire_deadline(self, reason="generation_wall_timeout"):
                self.fired = True
                (self.gen_dir / "STOP_SIGNAL").write_text(reason, encoding="utf-8")

            def write_postgen_marker(self, snapshot):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            loop = self._loop(Path(temporary))
            loop._task_lifecycle = _Deadline(30 if outer_timeout else 0.03)
            with self._patches(Peer, Trigger):
                task = asyncio.create_task(cohort_runner.run_generation_cohort(loop, 0))
                try:
                    if outer_timeout:
                        await evaluation_started.wait()
                        loop._task_lifecycle.deadline = time.monotonic() - 1
                        task = asyncio.create_task(asyncio.wait_for(task, timeout=0.001))
                    done, _ = await asyncio.wait({task}, timeout=0.2)
                    self.assertIn(task, done, "post-generation evaluation ignored deadline")
                    if outer_timeout:
                        with self.assertRaises(TimeoutError):
                            await task
                    else:
                        self.assertEqual((await task)[0]["peer_id"], "gen0_peer0")
                    self.assertTrue(evaluation_cancelled.is_set())
                    self.assertTrue((loop.run_dir / "gen_0/generation_results.json").exists())
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_postgeneration_evaluation_is_bounded_by_remaining_research_time(self) -> None:
        await self._postgeneration_evaluation_deadline(outer_timeout=False)

    async def test_outer_timeout_during_postgeneration_evaluation_preserves_results(self) -> None:
        await self._postgeneration_evaluation_deadline(outer_timeout=True)

    async def test_outer_timeout_during_trigger_cleanup_still_commits_peer_results(self) -> None:
        release = asyncio.Event()
        closing_started = asyncio.Event()

        class Peer:
            def __init__(self, *, peer_id, **kwargs):
                self.peer_id = peer_id

            async def run(self):
                return {"peer_id": self.peer_id, "success": True, "evidence": "finished"}

        class Trigger:
            def __init__(self, *, gen_dir, **kwargs):
                self.gen_dir = gen_dir
                self.fired = False
                self.closing = True
                self.adaptive_policy = SimpleNamespace(drain_grace_minutes=60)

            async def wait_until_fire(self, abort_event):
                closing_started.set()
                await release.wait()

            def fire_deadline(self, reason="generation_wall_timeout"):
                self.fired = True
                (self.gen_dir / "STOP_SIGNAL").write_text(reason, encoding="utf-8")

        with tempfile.TemporaryDirectory() as temporary:
            loop = self._loop(Path(temporary))
            with self._patches(Peer, Trigger):
                cohort = asyncio.create_task(cohort_runner.run_generation_cohort(loop, 0))
                await closing_started.wait()
                # Let the completed peer reach the asynchronous closing wait.
                await asyncio.sleep(0.01)
                loop._task_lifecycle.deadline = time.monotonic() - 1
                task = asyncio.create_task(asyncio.wait_for(cohort, timeout=0.001))
                try:
                    done, _ = await asyncio.wait({task}, timeout=0.2)
                    self.assertIn(task, done)
                    with self.assertRaises(TimeoutError):
                        await task
                    self.assertTrue(
                        (loop.run_dir / "gen_0/generation_results.json").exists(),
                        "cancellation during trigger cleanup discarded completed peer output",
                    )
                    results = json.loads(
                        (loop.run_dir / "gen_0/generation_results.json").read_text()
                    )
                    self.assertEqual(results[0]["evidence"], "finished")
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_scoreless_bypasses_evaluator_dependent_dig_and_cohort_allocation(self) -> None:
        class Peer:
            def __init__(self, *, peer_id, **kwargs):
                self.peer_id = peer_id

            async def run(self):
                return {"peer_id": self.peer_id, "success": True}

        class Trigger:
            def __init__(self, **kwargs):
                self.fired = True
                self.closing = False

            async def wait_until_fire(self, abort_event):
                return

        async def reject_dig(**kwargs):
            raise AssertionError("scoreless research cannot invoke evaluator-dependent DIG")

        with tempfile.TemporaryDirectory() as temporary:
            loop = self._loop(Path(temporary))
            loop.dig_lite_config = SimpleNamespace(enabled=True, max_attempts=1)
            loop.quality_diversity_config = SimpleNamespace(
                enabled=True, cohort=SimpleNamespace(enabled=True)
            )
            with (
                self._patches(Peer, Trigger),
                patch.object(cohort_runner, "run_dig_lite", reject_dig),
                patch.object(
                    cohort_runner,
                    "allocate_cohort_qd_contracts",
                    side_effect=AssertionError("scoreless cohort cannot rank DIG candidates"),
                ),
            ):
                result = await cohort_runner.run_generation_cohort(loop, 0)
            self.assertEqual(result, [{"peer_id": "gen0_peer0", "success": True}])
            self.assertFalse((loop.run_dir / "gen_0/peers/gen0_peer0/dig").exists())

    async def test_slow_peer_cancellation_is_quarantined_without_extending_deadline(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        peer_task: list[asyncio.Task] = []

        class Peer:
            def __init__(self, *, peer_id, **kwargs):
                self.peer_id = peer_id
                self.last_result = None

            async def run(self):
                peer_task.append(asyncio.current_task())
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    self.last_result = {"sessions_completed": 3, "stop_reason": "interrupted"}
                    await release.wait()
                    raise

        class Trigger:
            def __init__(self, **kwargs):
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                await release.wait()

            def fire_deadline(self, reason="generation_wall_timeout"):
                self.fired = True

        with tempfile.TemporaryDirectory() as temporary:
            loop = self._loop(Path(temporary))
            with self._patches(Peer, Trigger):
                task = asyncio.create_task(cohort_runner.run_generation_cohort(loop, 0))
                await started.wait()
                loop._task_lifecycle.deadline = time.monotonic() - 1
                task.cancel()
                try:
                    done, _ = await asyncio.wait({task}, timeout=0.2)
                    self.assertIn(task, done)
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    result = json.loads(
                        (loop.run_dir / "gen_0/generation_results.json").read_text()
                    )[0]
                    self.assertEqual(result["status"], "late_quarantined_peer_task")
                    self.assertEqual(result["sessions_completed"], 3)
                    self.assertFalse(result["promotion_eligible"])
                finally:
                    release.set()
                    await asyncio.gather(task, *peer_task, return_exceptions=True)

    async def test_repeated_cancellation_during_peer_drain_preserves_completed_output(self) -> None:
        started = asyncio.Event()
        interrupted = asyncio.Event()
        release = asyncio.Event()

        class Peer:
            def __init__(self, *, peer_id, **kwargs):
                self.peer_id = peer_id

            async def run(self):
                if self.peer_id.endswith("0"):
                    return {"peer_id": self.peer_id, "evidence": "retained"}
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    interrupted.set()
                    await release.wait()
                    raise

        class Trigger:
            def __init__(self, **kwargs):
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                await release.wait()

            def fire_deadline(self, reason="generation_wall_timeout"):
                self.fired = True

        with tempfile.TemporaryDirectory() as temporary:
            loop = self._loop(Path(temporary), peers=2)
            with self._patches(Peer, Trigger):
                task = asyncio.create_task(cohort_runner.run_generation_cohort(loop, 0))
                await started.wait()
                task.cancel()
                await interrupted.wait()
                task.cancel()
                try:
                    done, _ = await asyncio.wait({task}, timeout=0.2)
                    self.assertIn(task, done)
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertTrue(
                        (loop.run_dir / "gen_0/generation_results.json").exists(),
                        "repeat cancellation discarded already completed peer evidence",
                    )
                    results = json.loads(
                        (loop.run_dir / "gen_0/generation_results.json").read_text()
                    )
                    self.assertEqual(results[0]["evidence"], "retained")
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
