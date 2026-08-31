"""Untimed synthesis still reacts to evidence, cohort completion, and cancellation."""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger


class UnboundedSynthesisTriggerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gen_dir = self.root / "gen_0"
        self.gen_dir.mkdir()

    def _trigger(self, **overrides) -> synthesis_trigger.SynthesisTrigger:
        return synthesis_trigger.SynthesisTrigger(
            **{
                "run_dir": self.root,
                "gen_dir": self.gen_dir,
                "gen_id": 0,
                "gen_start_time": time.time() - 20 * 365 * 24 * 3600,
                "min_findings": 2,
                "min_contributing_peers": 2,
                "min_interval_minutes": 0,
                "max_interval_minutes": None,
                **overrides,
            }
        )

    def _publish_findings(self) -> None:
        with sqlite3.connect(self.root / "shared_store.db") as connection:
            connection.execute(
                "CREATE TABLE findings (id TEXT, peer_id TEXT, generation_id INTEGER)"
            )
            connection.executemany(
                "INSERT INTO findings VALUES (?, ?, 0)",
                [("one", "gen0_peer0"), ("two", "gen0_peer1")],
            )

    def test_elapsed_time_never_closes_a_trigger_without_a_maximum(self) -> None:
        for adaptive in ({}, {"enabled": True, "max_interval_ceiling_minutes": 15}):
            with self.subTest(adaptive=adaptive):
                trigger = self._trigger(adaptive_policy=adaptive)
                self.assertIsNone(trigger.max_interval_minutes)
                snapshot = trigger.evaluate()
                self.assertFalse(snapshot.fired)
                self.assertEqual(snapshot.reason, "not_yet")
                self.assertEqual(
                    trigger._seconds_until_next_timer_check(snapshot),
                    trigger.poll_interval_seconds,
                )

    def test_evidence_closes_an_untimed_generation_after_active_work_drains(self) -> None:
        active_work = [1]
        trigger = self._trigger(cohort_active_peers_callback=lambda: active_work[0])
        self._publish_findings()
        with patch(
            "praxist.plugins.workflow_stages.research_loop.backend."
            "experiment_scheduler_client.freeze_generation"
        ):
            snapshot = trigger.evaluate()
        self.assertFalse(snapshot.fired)
        self.assertEqual(snapshot.reason, "assessment_draining")
        self.assertTrue((self.gen_dir / "CLOSING_SIGNAL").exists())
        active_work[0] = 0
        snapshot = trigger.evaluate()
        self.assertTrue(snapshot.fired)
        self.assertEqual(snapshot.reason, "info_density")

    def test_drained_cohort_closes_without_a_timer_or_enough_findings(self) -> None:
        trigger = self._trigger(cohort_active_peers_callback=lambda: 0)
        snapshot = trigger.evaluate()
        self.assertTrue(snapshot.fired)
        self.assertEqual(snapshot.reason, "cohort_drained")

    def test_explicit_finite_maximum_keeps_its_safety_cap(self) -> None:
        trigger = self._trigger(max_interval_minutes=30)
        snapshot = trigger.evaluate()
        self.assertTrue(snapshot.fired)
        self.assertEqual(snapshot.reason, "safety_cap")

    async def test_wait_loop_publishes_stop_on_evidence_event_without_a_timer(self) -> None:
        trigger = self._trigger()
        waits: list[float] = []

        async def publish_on_event(paths, *, timeout_seconds, **kwargs):
            waits.append(timeout_seconds)
            self._publish_findings()
            return SimpleNamespace(reason="event")

        with (
            patch.object(synthesis_trigger, "wait_for_filesystem_event", publish_on_event),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client.freeze_generation"
            ),
            self.assertLogs(synthesis_trigger.logger, level="INFO") as logs,
        ):
            snapshot = await asyncio.wait_for(trigger.wait_until_fire(), timeout=1)
        self.assertEqual(waits, [trigger.poll_interval_seconds])
        self.assertTrue(snapshot.fired)
        self.assertEqual(snapshot.reason, "info_density")
        self.assertIn("unbounded", "\n".join(logs.output))
        self.assertIn(
            "trigger_reason=info_density",
            (self.gen_dir / "STOP_SIGNAL").read_text(encoding="utf-8"),
        )

    async def test_cancellation_releases_filesystem_wait_when_cohort_wakeup_is_enabled(
        self,
    ) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()
        trigger = self._trigger(cohort_completed_event=asyncio.Event())

        async def waiting_filesystem(paths, **kwargs):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with patch.object(synthesis_trigger, "wait_for_filesystem_event", waiting_filesystem):
            task = asyncio.create_task(trigger.wait_until_fire())
            try:
                await asyncio.wait_for(started.wait(), timeout=0.5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=0.5)
                self.assertTrue(cancelled.is_set())
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
