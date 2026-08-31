"""Offline contracts for retaining research without a measured objective."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from praxist.plugins.workflow_stages.research_loop.backend import (
    findings_collection,
    generation_boundary,
)
from praxist.plugins.workflow_stages.research_loop.backend.frontier import FrontierStore


def _task(mode: str = "scoreless") -> SimpleNamespace:
    return SimpleNamespace(
        research_loop={"mode": mode},
        evaluation=SimpleNamespace(diversity_dimensions=[], maturity_policy={}),
        research_memory=SimpleNamespace(enabled=False),
        generation_policy=SimpleNamespace(max_generations=2, cohort_size=2),
    )


def _write_finding(run_dir: Path, finding: dict) -> None:
    directory = run_dir / "shared_findings"
    directory.mkdir(exist_ok=True)
    (directory / f"{finding['id']}.json").write_text(json.dumps(finding), encoding="utf-8")


class ScorelessResearchTest(unittest.TestCase):
    def test_filesystem_collection_rejects_linked_sources_without_losing_valid_findings(
        self,
    ) -> None:
        for link_kind in ("leaf_symlink", "parent_symlink", "hardlink"):
            with self.subTest(link_kind=link_kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_dir = root / "run"
                run_dir.mkdir()
                private_dir = root / "private"
                private_dir.mkdir()
                private_file = private_dir / "private.json"
                private_file.write_text(
                    json.dumps(
                        {"id": "private", "generation_id": 0, "content": "Synthetic private data"}
                    ),
                    encoding="utf-8",
                )
                findings_dir = run_dir / "shared_findings"
                if link_kind == "parent_symlink":
                    findings_dir.symlink_to(private_dir, target_is_directory=True)
                else:
                    findings_dir.mkdir()
                    linked_file = findings_dir / "linked.json"
                    if link_kind == "leaf_symlink":
                        linked_file.symlink_to(private_file)
                    else:
                        os.link(private_file, linked_file)
                generation_findings = run_dir / "gen_0" / "shared_findings"
                generation_findings.mkdir(parents=True)
                (generation_findings / "valid.json").write_text(
                    json.dumps(
                        {"id": "valid", "generation_id": 0, "content": "Useful public finding"}
                    ),
                    encoding="utf-8",
                )
                with patch.dict(os.environ, {"PRAXIST_CONTROLLER_STATE_DIR": str(private_dir)}):
                    rows = findings_collection.collect_findings_for_generation(
                        findings_dir=findings_dir,
                        gen_id=0,
                        local_mode=False,
                        task_spec=_task(),
                    )

                self.assertEqual([row["id"] for row in rows], ["valid"])
                self.assertEqual(rows[0]["content"], "Useful public finding")

    def test_source_enrichment_rejects_replacement_during_cutoff_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = {"id": "frozen", "generation_id": 0, "content": "Before cutoff"}
            _write_finding(root, original)
            source = root / "shared_findings" / "frozen.json"
            row = {**original, "source_filepath": str(source)}
            cutoff = datetime.now(UTC)
            identity = findings_collection._logical_result_identity
            replaced = False

            def replace_before_hash(path, *, run_dir):
                nonlocal replaced
                if path == source and not replaced:
                    replaced = True
                    source.write_text(
                        json.dumps({**original, "content": "After cutoff"}), encoding="utf-8"
                    )
                return identity(path, run_dir=run_dir)

            # Filesystem timestamp resolution may put both writes in the same tick.
            # Content admission must remain safe even when that clock cannot order them.
            with patch.object(
                findings_collection,
                "_result_publication_mtime",
                return_value=cutoff.timestamp(),
            ):
                with patch.object(
                    findings_collection, "_logical_result_identity", replace_before_hash
                ):
                    snapshot = findings_collection.include_finding_sources_in_snapshot(
                        {},
                        [row],
                        run_dir=root,
                        findings_dir=root / "shared_findings",
                        gen_id=0,
                        cutoff=cutoff,
                    )
                snapshot = findings_collection.preserve_scoreless_finding_sources(
                    snapshot,
                    run_dir=root,
                    cutoff=cutoff,
                )
            restored = findings_collection.scoreless_findings_with_frozen_sources(snapshot, [row])
            self.assertTrue(replaced)
            self.assertEqual(restored[0]["content"], "Before cutoff")
            self.assertNotIn("source_payload", restored[0])
            self.assertEqual(restored[0]["source_payload_status"], "source_unavailable_or_changed")

    def test_new_raw_source_snapshot_redacts_secret_fields_before_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = {
                "id": "source",
                "generation_id": 0,
                "content": "A finding",
                "api_key": "synthetic-redaction-test-value",
            }
            _write_finding(root, original)
            row = {
                "id": "source",
                "generation_id": 0,
                "content": "A finding",
                "source_filepath": str(root / "shared_findings" / "source.json"),
            }
            cutoff = datetime.now(UTC)
            snapshot = findings_collection.include_finding_sources_in_snapshot(
                {},
                [row],
                run_dir=root,
                findings_dir=root / "shared_findings",
                gen_id=0,
                cutoff=cutoff,
            )
            snapshot = findings_collection.preserve_scoreless_finding_sources(
                snapshot, run_dir=root, cutoff=cutoff
            )
            self.assertNotIn("synthetic-redaction-test-value", json.dumps(snapshot))

    def test_own_manifest_writer_never_exceeds_its_reader(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.scoreless import (
            read_scoreless_evidence_manifest,
            write_scoreless_evidence_manifest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = "a" * (16 * 1024 * 1024)
            write_scoreless_evidence_manifest(
                root,
                gen_id=0,
                findings=[{"id": "large", "content": content}],
                evidence_cutoff_at="2026-08-31T00:00:00+00:00",
                evidence_source_snapshot={},
            )
            manifest = read_scoreless_evidence_manifest(root, 0)
            self.assertIsNotNone(manifest)
            self.assertEqual(len(manifest["findings"][0]["content"]), len(content))

    def test_local_boundary_freezes_full_source_content_and_structured_fields(self) -> None:
        class InterruptedPI:
            async def run(self, *, completed_gen_id: int):
                raise RuntimeError("interrupted PI")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_content = "Evidence details. " * 600
            finding = {
                "id": "structured",
                "finding_type": "hypothesis",
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "content": full_content,
                "scenario_probabilities": [{"case": "A", "probability": 0.4}],
            }
            _write_finding(root, finding)
            (root / "gen_0").mkdir()
            (root / "gen_0" / "generation_results.json").write_text("[]", encoding="utf-8")
            loop = SimpleNamespace(
                run_dir=root,
                findings_dir=root / "shared_findings",
                local_mode=True,
                task_spec=_task(),
                frontier=FrontierStore(root / "frontier"),
                _strategy_for_gen=lambda gen_id: "explore",
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )
            loop._collect_findings_for_generation = lambda gen_id: (
                findings_collection.collect_loop_findings(loop, gen_id)
            )
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root)}):
                with self.assertRaisesRegex(RuntimeError, "interrupted PI"):
                    asyncio.run(
                        generation_boundary.complete_generation_boundary(
                            loop,
                            gen_id=0,
                            pi_agent=InterruptedPI(),
                            pi_cfg=SimpleNamespace(strict=True),
                        )
                    )
                frozen_path = root / "gen_0" / "scoreless_evidence.json"
                from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
                    get_findings,
                )

                self.assertFalse(
                    any(
                        row["metrics"].get("late_after_generation_boundary")
                        for row in get_findings(generation_id=0)
                    )
                )
                original = frozen_path.read_bytes()
                frozen = json.loads(original)["findings"][0]
                self.assertEqual(frozen["content"], full_content)
                self.assertEqual(
                    frozen["scenario_probabilities"], finding["scenario_probabilities"]
                )
                self.assertEqual(frozen["source_payload"], finding)
                self.assertEqual(frozen["source_payload_status"], "frozen")
                _write_finding(root, {**finding, "content": "Replacement"})
                del loop._boundary_evidence_cutoff
                asyncio.run(
                    generation_boundary.complete_generation_boundary(
                        loop,
                        gen_id=0,
                        pi_agent=None,
                        pi_cfg=SimpleNamespace(strict=False),
                    )
                )
                self.assertEqual(frozen_path.read_bytes(), original)

    def test_local_store_retains_hypotheses_and_errors_without_primary_metric_hoisting(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ("metric", "scoreless"):
                with self.subTest(mode=mode):
                    run_dir = root / mode
                    run_dir.mkdir()
                    for finding_type in ("hypothesis", "error"):
                        _write_finding(
                            run_dir,
                            {
                                "id": finding_type,
                                "finding_type": finding_type,
                                "generation_id": 0,
                                "peer_id": "gen0_peer0",
                                "content": f"Preserve {finding_type} for the next generation.",
                                "details": {"task_score": 0.6},
                            },
                        )
                    task = _task(mode)
                    task.evaluation.primary_metric = "task_score"
                    loop = SimpleNamespace(
                        task_spec=task, findings_dir=run_dir / "shared_findings", local_mode=True
                    )
                    with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}):
                        rows = findings_collection.collect_loop_findings(loop, 0)
                    self.assertEqual({row["finding_type"] for row in rows}, {"hypothesis", "error"})
                    self.assertTrue(all(row["content"].startswith("Preserve") for row in rows))
                    for row in rows:
                        self.assertEqual("task_score" in row["metrics"], mode == "metric")

    def test_loop_and_background_options_cannot_reenable_score_materialization(self) -> None:
        task = _task()
        task.gems = SimpleNamespace(result_artifact_materialization=True)
        options = findings_collection.result_artifact_options_from_task_spec(task)
        self.assertFalse(options["materialize_result_artifacts"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "results" / "gen0_peer0_candidate"
            result.mkdir(parents=True)
            (result / "summary.json").write_text(
                json.dumps(
                    {
                        "metrics": {"metric_value": 0.8, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )
            loop = SimpleNamespace(
                task_spec=task, findings_dir=root / "shared_findings", local_mode=False
            )
            self.assertEqual(findings_collection.collect_loop_findings(loop, 0), [])

    def test_frozen_evidence_context_is_bounded_and_generation_scoped(self) -> None:
        scoreless = importlib.import_module(
            "praxist.plugins.workflow_stages.research_loop.backend.scoreless"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for gen_id in range(3):
                scoreless.write_scoreless_evidence_manifest(
                    root,
                    gen_id=gen_id,
                    findings=[
                        {
                            "id": f"f{gen_id}",
                            "finding_type": "question",
                            "generation_id": gen_id,
                            "content": "x" * 2000,
                            "details": {"large_payload": "y" * 5000},
                        }
                    ],
                    evidence_cutoff_at="2026-08-31T00:00:00+00:00",
                    evidence_source_snapshot={},
                )
            rows = scoreless.load_scoreless_evidence(
                root,
                1,
                max_findings=1,
                max_content_chars=80,
            )
            self.assertEqual([row["id"] for row in rows], ["f1"])
            self.assertLessEqual(len(rows[0]["content"]), 80)
            self.assertTrue(rows[0]["content_truncated"])
            self.assertNotIn("details", rows[0])
            full = json.loads((root / "gen_1" / "scoreless_evidence.json").read_text())
            self.assertEqual(len(full["findings"][0]["content"]), 2000)
            self.assertEqual(scoreless.load_scoreless_evidence(root, -1), [])
            self.assertTrue(scoreless.is_scoreless({"research_loop": {"mode": "scoreless"}}))
            self.assertTrue(
                scoreless.is_scoreless(
                    SimpleNamespace(_raw={"research_loop": {"mode": "scoreless"}})
                )
            )
            self.assertFalse(scoreless.is_scoreless(SimpleNamespace()))

    def test_mode_disables_result_materialization_but_keeps_declared_findings(self) -> None:
        """A result-like artifact must not silently manufacture a scored finding."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_finding(
                root,
                {
                    "id": "source-observation",
                    "generation_id": 0,
                    "finding_type": "observation",
                    "peer_id": "gen0_peer0",
                    "title": "Documented constraint",
                    "content": "The source is incomplete.",
                },
            )
            result = root / "results" / "gen0_peer0_candidate"
            result.mkdir(parents=True)
            (result / "summary.json").write_text(
                json.dumps(
                    {
                        "primary_metric": "task_score",
                        "status": "ok",
                        "metrics": {"task_score": 0.8, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )
            rows = findings_collection.collect_findings_for_generation(
                findings_dir=root / "shared_findings",
                gen_id=0,
                local_mode=False,
                primary_metric="task_score",
                result_scoring_metric_keys=["task_score"],
                task_spec=_task(),
            )
            self.assertEqual([row["id"] for row in rows], ["source-observation"])
            self.assertTrue((result / "summary.json").exists())
            self.assertEqual(len(list((root / "shared_findings").glob("*.json"))), 1)

    def test_scoreless_boundary_retains_all_types_without_numeric_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings = [
                {
                    "id": "constraint",
                    "finding_type": "observation",
                    "generation_id": 0,
                    "peer_id": "gen0_peer0",
                    "content": "Source coverage remains unresolved.",
                },
                {
                    "id": "hypothesis",
                    "finding_type": "hypothesis",
                    "generation_id": 0,
                    "peer_id": "gen0_peer1",
                    "content": "A competing mechanism is plausible.",
                },
                {
                    "id": "numeric-note",
                    "finding_type": "result",
                    "generation_id": 0,
                    "peer_id": "gen0_peer0",
                    "variant_name": "calculation",
                    "metrics": {
                        "metric_value": 0.7,
                        "scored_complete": True,
                        "training_budget_ratio": 1.0,
                        "eval_budget_ratio": 1.0,
                    },
                },
            ]
            for finding in findings:
                _write_finding(root, finding)
            gen_dir = root / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "STOP_SIGNAL").write_text(
                json.dumps(
                    {
                        "trigger_reason": "cohort_drained",
                        "evidence_sufficient": False,
                    }
                ),
                encoding="utf-8",
            )
            graph_syncs: list[int] = []

            class Graph:
                def sync_once_blocking(self, *, timeout: float) -> dict:
                    graph_syncs.append(0)
                    return {"status": "ok"}

            loop = SimpleNamespace(
                run_dir=root,
                findings_dir=root / "shared_findings",
                local_mode=False,
                task_spec=_task(),
                frontier=FrontierStore(root / "frontier"),
                _collect_findings_for_generation=lambda gen_id: findings,
                _strategy_for_gen=lambda gen_id: "explore",
                _graph_maintainer=Graph(),
                _findings_sync=None,
                gems=None,
            )
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=0,
                    pi_agent=None,
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )
            self.assertEqual(loop.frontier.get_summary(), [])
            manifest_path = gen_dir / "scoreless_evidence.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text())
            for actual, original in zip(manifest["findings"], findings, strict=True):
                for key, value in original.items():
                    self.assertEqual(actual[key], value)
            self.assertEqual(manifest["mode"], "scoreless")
            self.assertEqual(manifest["retained_count"], 3)
            self.assertTrue(manifest["evidence_cutoff_at"])
            marker = json.loads((gen_dir / "generation_boundary.json").read_text())
            self.assertEqual(marker["promoted_count"], 0)
            self.assertEqual(marker["stop_audit"]["evidence_status"], "not_scored")
            self.assertIsNone(marker["stop_audit"]["evidence_sufficient"])
            self.assertEqual(marker["peer_mix"]["mode"], "scoreless")
            self.assertNotIn("constructive_deficit", marker["peer_mix"])
            self.assertEqual(graph_syncs, [0])

    def test_scoreless_mode_never_resets_evidence_through_gems(self) -> None:
        class Gems:
            enabled = True

            def maybe_trigger_after_boundary(self, *, completed_gen_id: int):
                raise AssertionError("Scoreless evidence must never enter metric-driven resets")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = SimpleNamespace(
                run_dir=root,
                local_mode=False,
                task_spec=_task(),
                frontier=FrontierStore(root / "frontier"),
                _collect_findings_for_generation=lambda gen_id: [],
                _strategy_for_gen=lambda gen_id: "explore",
                _graph_maintainer=None,
                _findings_sync=None,
                gems=Gems(),
            )
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=0,
                    pi_agent=None,
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )
            self.assertTrue((root / "gen_0" / "generation_boundary.json").exists())

    def test_metric_mode_still_promotes_measured_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings = [
                {
                    "id": "measured",
                    "finding_type": "result",
                    "generation_id": 1,
                    "peer_id": "gen1_peer0",
                    "variant_name": "measured",
                    "metrics": {
                        "metric_value": 0.7,
                        "scored_complete": True,
                        "training_budget_ratio": 1.0,
                        "eval_budget_ratio": 1.0,
                    },
                }
            ]
            loop = SimpleNamespace(
                run_dir=root,
                local_mode=False,
                task_spec=_task("metric"),
                frontier=FrontierStore(root / "frontier"),
                _collect_findings_for_generation=lambda gen_id: findings,
                _strategy_for_gen=lambda gen_id: "explore",
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=1,
                    pi_agent=None,
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )
            self.assertEqual(len(loop.frontier.get_summary()), 1)
            self.assertFalse((root / "gen_1" / "scoreless_evidence.json").exists())

    def test_boundary_retry_keeps_frozen_evidence_when_sources_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings = [
                {
                    "id": "observation",
                    "finding_type": "observation",
                    "generation_id": 0,
                    "content": "Before boundary",
                }
            ]
            _write_finding(root, findings[0])
            loop = SimpleNamespace(
                run_dir=root,
                findings_dir=root / "shared_findings",
                local_mode=False,
                task_spec=_task(),
                frontier=FrontierStore(root / "frontier"),
                _collect_findings_for_generation=lambda gen_id: findings,
                _strategy_for_gen=lambda gen_id: "explore",
                _graph_maintainer=None,
                _findings_sync=None,
                gems=None,
            )
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=0,
                    pi_agent=None,
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )
            original = (root / "gen_0" / "scoreless_evidence.json").read_bytes()
            findings[0] = {**findings[0], "content": "After boundary"}
            _write_finding(root, findings[0])
            findings.append(
                {
                    "id": "late",
                    "finding_type": "insight",
                    "generation_id": 0,
                    "content": "Late publication",
                }
            )
            _write_finding(root, findings[1])
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=0,
                    pi_agent=None,
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )
            self.assertEqual((root / "gen_0" / "scoreless_evidence.json").read_bytes(), original)
            self.assertTrue((root / "shared_findings" / "late.json").exists())
            (root / "gen_0" / "scoreless_evidence.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "frozen scoreless evidence"):
                asyncio.run(
                    generation_boundary.complete_generation_boundary(
                        loop,
                        gen_id=0,
                        pi_agent=None,
                        pi_cfg=SimpleNamespace(strict=False),
                    )
                )
            self.assertFalse((root / "gen_0" / "scoreless_evidence.json").exists())


if __name__ == "__main__":
    unittest.main()
