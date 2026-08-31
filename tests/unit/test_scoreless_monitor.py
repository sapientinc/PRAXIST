"""Scoreless stop audits remain visibly unscored in both monitor views."""

from __future__ import annotations

import unittest
from typing import Any

from praxist.cli import monitor
from praxist.cli.status import SOURCE_REGISTRY, StatusRow


def _snapshot(stop_audit: dict[str, Any]) -> monitor.MonitorSnapshot:
    row = StatusRow(
        pid=1234,
        ppid=1,
        etime="00:10",
        command="praxist start",
        run_dir="/run",
        source=SOURCE_REGISTRY,
        state="running",
        run_id="example",
    )
    return monitor.MonitorSnapshot(
        rows=[row],
        selected=row,
        target=monitor.MonitorTarget(run_id="example"),
        generated_at="2026-08-31T00:00:00+00:00",
        orchestrator_status={"last_stop_audit": stop_audit},
    )


class ScorelessMonitorTests(unittest.TestCase):
    def _assert_stop_lines(self, audit: dict[str, Any], expected: str) -> None:
        snapshot = _snapshot(audit)
        views = {
            "text": monitor.TextMonitorRenderer().render(snapshot).splitlines(),
            "tui": monitor.TuiMonitorRenderer(color=False)._selected_lines(snapshot),
        }
        for view, lines in views.items():
            with self.subTest(view=view):
                self.assertEqual(
                    [line.removeprefix("- ") for line in lines if "last_stop:" in line],
                    [expected],
                )

    def test_scoreless_audits_show_unscored_maturity_and_preserve_stop_reason(self) -> None:
        for reason_fields, reason in (
            ({"trigger_reason": "deadline", "signal_file": "STOP_SIGNAL"}, "deadline"),
            ({"signal_file": "STOP_SIGNAL"}, "STOP_SIGNAL"),
            ({}, "-"),
        ):
            with self.subTest(reason=reason):
                self._assert_stop_lines(
                    {"mode": "scoreless", "evidence_status": "not_scored", **reason_fields},
                    f"last_stop: {reason} maturity=not_scored",
                )

    def test_scoreless_mode_does_not_display_stale_metric_counts(self) -> None:
        self._assert_stop_lines(
            {
                "mode": "scoreless",
                "trigger_reason": "deadline",
                "mature_result_peers": 3,
                "required_mature_result_peers": 4,
            },
            "last_stop: deadline maturity=not_scored",
        )

    def test_metric_audit_output_is_unchanged(self) -> None:
        self._assert_stop_lines(
            {
                "trigger_reason": "mature_quorum",
                "mature_result_peers": 3,
                "required_mature_result_peers": 4,
            },
            "last_stop: mature_quorum mature=3/4",
        )
        self._assert_stop_lines({"signal_file": "STOP_SIGNAL"}, "last_stop: STOP_SIGNAL mature=0/0")


if __name__ == "__main__":
    unittest.main()
