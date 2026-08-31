"""Scoreless orchestrator status does not imply missing evaluation evidence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from praxist.plugins.workflow_stages.research_loop.backend.status_snapshot import (
    build_orchestrator_status_snapshot,
)


def _snapshot(root: Path, findings: list[dict], *, completed: int = 0):
    return build_orchestrator_status_snapshot(
        run_started_at="2026-08-31T00:00:00+00:00",
        run_dir=root,
        task_spec=SimpleNamespace(
            task_id="evidence",
            task_name="Evidence research",
            research_loop={"mode": "scoreless"},
            generation_policy=SimpleNamespace(
                max_generations=3,
                cohort_size=2,
                promote_top_k=2,
                promote_criterion="primary_metric",
            ),
            evaluation=SimpleNamespace(primary_metric="", direction="maximize"),
            baselines=[],
        ),
        frontier=SimpleNamespace(get_summary=lambda: [{"metric_value": 0.99}]),
        current_gen=completed,
        gens_completed=completed,
        frontier_strategy="auto",
        strategy_for_gen=lambda gen: "top_k",
        findings=findings,
        gems_context={"gems_count": 9, "gems": [{"metric_value": 0.99}]},
    )


class ScorelessStatusSnapshotTests(unittest.TestCase):
    def test_scoreless_findings_are_counted_without_ranking_or_baseline_warnings(self):
        findings = [
            {"finding_type": kind, "content": "Unresolved evidence", "metrics": {}}
            for kind in ("result", "hypothesis", "insight", "challenge", "error")
        ]
        # Numeric evidence is still evidence, never an implicit optimization score.
        findings.append({"finding_type": "result", "metrics": {"": 0.9, "samples": 1000}})
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertNoLogs(
                "praxist.plugins.workflow_stages.research_loop.backend.status_snapshot",
                level="WARNING",
            ),
        ):
            snapshot = _snapshot(Path(tmp), findings)
        payload = snapshot.to_dict()
        self.assertEqual(payload["findings_total"], 6)
        self.assertEqual(payload["research_mode"], "scoreless")
        self.assertEqual(payload["selection_status"], "disabled")
        self.assertEqual(payload["evaluation_status"], "not_configured")
        self.assertEqual(payload["retention_policy"], "all_findings")
        self.assertEqual(payload["strategy"], "scoreless")
        self.assertEqual(payload["gen_promotion_blocker"], "")
        self.assertIn("retain all findings", payload["gen_promotion_criteria"].lower())
        self.assertEqual(payload["variants_total"], 0)
        self.assertEqual(payload["variants_above_baseline"], 0)
        self.assertEqual(payload["frontier_candidates"], 0)
        self.assertEqual(payload["best_mature_result"], {})
        self.assertEqual(payload["best_validation_signal"], {})
        self.assertEqual(payload["gems_count"], 0)
        self.assertEqual(payload["gems_refs"], [])

    def test_empty_scoreless_generation_has_no_missing_metric_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = _snapshot(Path(tmp), [])
        self.assertEqual(snapshot.findings_total, 0)
        self.assertEqual(snapshot.gen_promotion_blocker, "")
        self.assertEqual(snapshot.mature_quorum_required, 0)
        self.assertEqual(snapshot.exit_condition, "in_progress")

    def test_scoreless_snapshot_preserves_real_run_and_resource_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gen_0").mkdir()
            (root / "gen_0" / "generation_boundary.json").write_text(
                json.dumps(
                    {
                        "stop_audit": {
                            "close_reason": "cohort_drained",
                            "maturity_status": "not_applicable",
                        },
                        "peer_mix": {"primary_metric": "stale"},
                    }
                )
            )
            (root / "resource_scheduler").mkdir()
            (root / "resource_scheduler" / "status.json").write_text(
                json.dumps(
                    {
                        "mode": "adaptive",
                        "running": 2,
                        "failed": 0,
                        "queued": 1,
                    }
                )
            )
            snapshot = _snapshot(root, [{"finding_type": "hypothesis"}], completed=1)
        self.assertEqual(snapshot.generations_completed, 1)
        self.assertEqual(snapshot.current_generation, 1)
        self.assertEqual(snapshot.cohort_size, 2)
        self.assertEqual(snapshot.max_generations, 3)
        self.assertEqual(snapshot.last_stop_audit["close_reason"], "cohort_drained")
        self.assertEqual(snapshot.last_peer_mix, {})
        self.assertEqual(snapshot.resource_scheduler["running"], 2)
        self.assertEqual(snapshot.resource_scheduler["queued"], 1)
        self.assertTrue(snapshot.operator_manifest_paths)


if __name__ == "__main__":
    unittest.main()
