"""Nullable scoreless limits remain explicit in status and operator views."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from praxist.cli import monitor
from praxist.cli.status import SOURCE_REGISTRY, StatusRow
from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
    collect_peer_memory_health,
)
from praxist.plugins.workflow_stages.research_loop.backend.status_snapshot import (
    build_orchestrator_status_snapshot,
)


def _task(max_generations=None, per_generation_hours=None):
    return SimpleNamespace(
        task_id="research",
        task_name="Research",
        research_loop={"mode": "scoreless"},
        generation_policy=SimpleNamespace(
            max_generations=max_generations,
            per_generation_hours=per_generation_hours,
            cohort_size=1,
        ),
    )


class ScorelessUnboundedStatusTest(unittest.TestCase):
    def test_actual_snapshot_and_both_monitor_views_preserve_nullable_limits(self) -> None:
        for count, hours, expected_count, expected_hours in (
            (None, None, "unbounded", "unbounded"),
            (12, 2.5, "12", "2.5"),
        ):
            with self.subTest(count=count, hours=hours), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                payload = build_orchestrator_status_snapshot(
                    run_started_at="2026-08-31T00:00:00+00:00",
                    run_dir=root,
                    task_spec=_task(count, hours),
                    frontier=SimpleNamespace(),
                    current_gen=9,
                    gens_completed=9,
                    frontier_strategy="auto",
                    strategy_for_gen=lambda _: "explore",
                    findings=[{"finding_type": "hypothesis", "content": "Still investigating"}],
                ).to_dict()

                self.assertEqual(payload["max_generations"], count)
                self.assertEqual(payload["per_generation_hours"], hours)
                self.assertEqual(payload["current_generation"], 9)
                row = StatusRow(
                    pid=123,
                    ppid=1,
                    etime="01:00",
                    command="praxist start",
                    source=SOURCE_REGISTRY,
                    state="running",
                    run_dir=str(root),
                    run_id="research",
                )
                snapshot = monitor.MonitorSnapshot(
                    rows=[row],
                    selected=row,
                    target=monitor.MonitorTarget(run_id="research"),
                    generated_at="2026-08-31T01:00:00+00:00",
                    orchestrator_status=payload,
                )
                views = (
                    monitor.TextMonitorRenderer().render(snapshot),
                    "\n".join(monitor.TuiMonitorRenderer(color=False)._selected_lines(snapshot)),
                )
                for view in views:
                    self.assertIn(f"generation_limit: {expected_count}", view)
                    self.assertIn(f"per_generation_hours: {expected_hours}", view)

    def test_peer_health_past_generation_eight_depends_on_session_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, success in enumerate((True, False)):
                memory = root / "gen_9" / "peers" / f"gen9_peer{index}" / "memory"
                memory.mkdir(parents=True)
                (memory / "peer_state.yaml").write_text(
                    yaml.safe_dump({"last_session_success": success, "research_state": "active"}),
                    encoding="utf-8",
                )
            health = collect_peer_memory_health(run_dir=root, generation_id=None, task_spec=_task())

        self.assertEqual(health.generation_id, 9)
        self.assertEqual(health.summary, {"red": 1, "yellow": 0, "green": 1})
        self.assertTrue(all(peer.baseline_status == "not_applicable" for peer in health.peers))
