"""Scoreless output-integrity failures remain failures at finalization."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from praxist.core.storage import read_jsonl
from praxist.plugins.workflow_stages.research_loop import startup
from praxist.plugins.workflow_stages.research_loop.backend.resume_state import write_boundary_marker
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import (
    write_scoreless_evidence_manifest,
)
from tests.helpers.paths import REPO_ROOT


class ScorelessFinalizerFailureTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.task = self.root / "task"
        shutil.copytree(REPO_ROOT / "templates/tasks/toy_math", self.task)
        self.run_dir = self.root / "run"
        self.enterContext(
            patch.dict(
                os.environ,
                {
                    "PRAXIST_STATE_DIR": str(self.root / "registry"),
                    "PRAXIST_BUNDLED_PLUGIN_ROOTS": str(REPO_ROOT / "tests/fixtures/plugins"),
                },
                clear=True,
            )
        )

    def _prepare(self, mode: str) -> startup.ResearchLoopPluginRun:
        path = self.task / "task.yaml"
        descriptor = yaml.safe_load(path.read_text(encoding="utf-8"))
        descriptor["research_loop"] = {"mode": mode}
        if mode == "scoreless":
            descriptor.pop("evaluation", None)
            descriptor.pop("task_entrypoints", None)
            descriptor["synthesis_trigger"] = {"enabled": False}
            descriptor["dig_lite"] = {"enabled": False}
            descriptor["quality_diversity"] = {"enabled": False}
            descriptor["gems"] = {"enabled": False}
            descriptor["praxist_plugins"]["evaluations"] = []
        path.write_text(yaml.safe_dump(descriptor), encoding="utf-8")
        return startup.prepare_research_loop_plugin_run(
            task_project_path=self.task,
            workspace=self.root,
            run_dir=self.run_dir,
            runtime_ref="agent_runtime:fake_runtime",
            model_provider_ref="model_provider:fake_provider",
            budget_policy_ref="budget_policy:fake_tiered",
            model="fake-deterministic",
            local_mode=True,
            frontier_strategy="auto",
            credential_profile="fake_multi_key",
        )

    def _changed_evidence(self) -> tuple[Path, bytes, dict]:
        manifest = write_scoreless_evidence_manifest(
            self.run_dir,
            gen_id=0,
            findings=[{"id": "finding-1", "title": "Frozen claim", "metrics": {}}],
            evidence_cutoff_at="2026-08-31T00:00:00Z",
            evidence_source_snapshot={},
        )
        write_boundary_marker(self.run_dir, gen_id=0, promoted_count=0, pi_status="disabled")
        changed = manifest.read_bytes() + b"\n"
        manifest.write_bytes(changed)
        delivery = self.run_dir / "delivery.json"
        delivery.write_bytes(b'{"answer": "preserved research"}\n')
        result = {
            "status": "succeeded",
            "generations_completed": 1,
            "exit_condition": "task_complete",
            "task_delivery": {
                "status": "completed",
                "artifacts": ["delivery.json"],
                "artifact_hashes": {
                    "delivery.json": hashlib.sha256(delivery.read_bytes()).hexdigest()
                },
                "summary": {},
            },
        }
        return manifest, changed, result

    def _summary(self) -> dict:
        return json.loads((self.run_dir / "run_summary.json").read_text(encoding="utf-8"))

    def _assert_failed_surfaces(self, summary: dict) -> None:
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["exit_code"], 1)
        self.assertEqual(summary["stage_summary"], {"research_loop": "failed"})
        self.assertTrue(summary["error"])
        self.assertIn("scoreless evidence hash", summary["materialization_error"])
        metadata = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "failed")
        self.assertTrue(metadata["finalized_at"])
        events, errors = read_jsonl(self.run_dir / "trajectory.jsonl")
        self.assertEqual(errors, [])
        terminal = [
            event
            for event in events
            if event["kind"] in {"workflow.stage_succeeded", "workflow.stage_failed"}
        ]
        self.assertEqual([event["kind"] for event in terminal], ["workflow.stage_failed"])
        self.assertEqual(terminal[0]["payload"]["error"], summary["error"])
        finalized = [event for event in events if event["kind"] == "run.finalized"]
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0]["payload"]["status"], "failed")
        self.assertEqual(finalized[0]["payload"]["exit_code"], 1)

    def test_successful_scoreless_run_records_failure_before_raising(self) -> None:
        prepared = self._prepare("scoreless")
        manifest, changed, result = self._changed_evidence()

        with self.assertRaises(ValueError):
            startup.finalize_research_loop_plugin_run(prepared, success=True, result=result)

        self._assert_failed_surfaces(self._summary())
        self.assertEqual(manifest.read_bytes(), changed)
        self.assertEqual(
            (self.run_dir / "delivery.json").read_bytes(),
            b'{"answer": "preserved research"}\n',
        )

    def test_failed_scoreless_finalization_preserves_error_without_raising_again(self) -> None:
        prepared = self._prepare("scoreless")
        manifest, changed, result = self._changed_evidence()

        startup.finalize_research_loop_plugin_run(
            prepared,
            success=False,
            result=result,
            error="scoreless output materialization failed",
            exit_code=1,
        )

        summary = self._summary()
        self._assert_failed_surfaces(summary)
        self.assertEqual(summary["error"], "scoreless output materialization failed")
        self.assertEqual(manifest.read_bytes(), changed)
        self.assertTrue((self.run_dir / "delivery.json").is_file())

    def test_metric_materialization_failure_retains_legacy_success_semantics(self) -> None:
        prepared = self._prepare("metric")
        retained = self.run_dir / "partial.txt"
        retained.write_text("preserved", encoding="utf-8")

        with patch.object(
            startup,
            "_materialize_legacy_outputs",
            side_effect=RuntimeError("materializer unavailable"),
        ):
            startup.finalize_research_loop_plugin_run(prepared, success=True, result={})

        summary = self._summary()
        self.assertEqual(summary["status"], "succeeded")
        self.assertEqual(summary["exit_code"], 0)
        self.assertIsNone(summary["error"])
        self.assertEqual(summary["materialization_error"], "materializer unavailable")
        self.assertEqual(retained.read_text(encoding="utf-8"), "preserved")


if __name__ == "__main__":
    unittest.main()
