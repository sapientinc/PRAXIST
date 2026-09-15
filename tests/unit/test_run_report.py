from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.tool_servers import (
    DEFAULT_RESEARCH_TOOL_SERVER_REFS,
    PANEL_TOOL_SERVER_REFS,
    tool_server_for_ref,
)
from praxist.plugins.tools.run_report.adapter import generate_run_report as tool_report
from praxist.plugins.workflow_stages.research_loop.backend import (
    run_report as run_report_module,
)
from praxist.plugins.workflow_stages.research_loop.backend.run_report import (
    REPORT_STATE_REL,
    generate_boundary_report_safely,
    generate_run_report,
    maybe_generate_boundary_report,
)


class RunReportTest(unittest.TestCase):
    def _write_fixture(self, root: Path, *, frontier_score: float = 1.2) -> tuple[Path, Path]:
        root = root.resolve()
        task_dir = root / "task"
        run_dir = task_dir / "experiments" / "run_2026-01-01_report"
        (task_dir / "assets" / "baselines").mkdir(parents=True)
        (task_dir / "assets" / "baselines" / "results.jsonl").write_text(
            '{"baseline":"base","task_score":1.0}\n',
            encoding="utf-8",
        )
        (task_dir / "task_spec.yaml").write_text(
            json.dumps(
                {
                    "evaluation": {
                        "primary_metric": "task_score",
                        "direction": "maximize",
                    }
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "frontier").mkdir(parents=True)
        (run_dir / "frontier" / "frontier_manifest.json").write_text(
            json.dumps(
                {
                    "lane_frontiers": {
                        "confirmed": [
                            {
                                "variant_name": "variant_strong",
                                "generation_id": 0,
                                "metric_name": "task_score",
                                "metric_value": frontier_score,
                                "metrics": {
                                    "task_score": frontier_score,
                                    "scored_complete": True,
                                },
                                "extra": {
                                    "evidence_stage": "full_eval",
                                    "parent_candidate": "baseline",
                                    "parent_usage": "complete_validation",
                                    "source_result_path": "results/variant_strong/summary.json",
                                },
                                "summary": "Improves the mature task score.",
                            }
                        ]
                    },
                    "validation_candidates": {
                        "cumulative": [
                            {
                                "variant_name": "variant_signal",
                                "metric_name": "task_score",
                                "metric_value": 0.9,
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "gen_0").mkdir()
        (run_dir / "gen_0" / "generation_boundary.json").write_text(
            json.dumps({"generation_id": 0, "status": "complete"}),
            encoding="utf-8",
        )
        (run_dir / "run_summary.json").write_text(
            json.dumps({"status": "running", "generations_completed": 1, "findings_total": 3}),
            encoding="utf-8",
        )
        (run_dir / "orchestrator_status.json").write_text(
            json.dumps({"current_generation": 1, "max_generations": 5}),
            encoding="utf-8",
        )
        return task_dir, run_dir

    def test_manual_report_contains_human_facing_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir, run_dir = self._write_fixture(Path(tmp))

            result = generate_run_report(
                run_dir=run_dir,
                task_dir=task_dir,
                trigger="manual_test",
                generation_id=0,
            )

            self.assertTrue(result.path.exists())
            self.assertIsNotNone(result.pdf_path)
            assert result.pdf_path is not None
            self.assertTrue(result.pdf_path.exists())
            self.assertEqual(result.pdf_path.suffix, ".pdf")
            self.assertEqual(result.path.parent, task_dir / "docs" / "praxist_reports")
            text = result.path.read_text(encoding="utf-8")
            self.assertIn("## A. Strongest Variants And Pareto Front", text)
            self.assertIn("## B. Strong-Variant Evolution And Lineage", text)
            self.assertIn("## C. Run Health And Evidence State", text)
            self.assertIn("Mature Dimension Winners", text)
            self.assertIn("PDF report with charts", text)
            self.assertIn("variant_strong", text)
            self.assertIn("baseline", text)
            self.assertIn("Validation candidates retained for follow-up", text)

    def test_report_survives_pdf_failure_and_health_infers_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir, run_dir = self._write_fixture(Path(tmp))
            with patch.object(
                run_report_module,
                "_write_pdf_report",
                side_effect=RuntimeError("renderer unavailable"),
            ):
                result = generate_run_report(run_dir=run_dir, task_dir=task_dir)

            self.assertTrue(result.path.exists())
            self.assertIsNone(result.pdf_path)
            completed = run_report_module._health_summary(
                run_summary={"exit_condition": "max_generations"},
                orchestrator={},
            )
            failed = run_report_module._health_summary(
                run_summary={"exit_condition": "error"},
                orchestrator={},
            )
            self.assertIn("Status: `succeeded`", completed[0])
            self.assertIn("Status: `failed`", failed[0])

    def test_report_prefers_final_status_and_reads_nested_boundary_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir, run_dir = self._write_fixture(Path(tmp))
            (run_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "exit_code": 0,
                        "exit_condition": "completed",
                        "generations_completed": 1,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "orchestrator_status.final.json").write_text(
                json.dumps(
                    {
                        "generations_completed": 1,
                        "max_generations": 1,
                        "exit_condition": "completed",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "gen_0" / "generation_boundary.json").write_text(
                json.dumps(
                    {
                        "generation_id": 0,
                        "artifact_semantics": {"status": "complete"},
                    }
                ),
                encoding="utf-8",
            )

            result = generate_run_report(
                run_dir=run_dir,
                task_dir=task_dir,
                trigger="final_status_test",
                generation_id=0,
            )
            text = result.path.read_text(encoding="utf-8")
            self.assertIn("Status: `succeeded`, exit condition: `completed`", text)
            self.assertIn("`gen_0` status `complete`", text)

    def test_appledouble_baseline_is_ignored_without_hiding_valid_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task"
            baseline_dir = task_dir / "assets" / "baselines"
            baseline_dir.mkdir(parents=True)
            (baseline_dir / "results.jsonl").write_text(
                '{"task_score": 1.25}\n',
                encoding="utf-8",
            )
            (baseline_dir / "._results.jsonl").write_bytes(b"\xff\xfeappledouble")

            values = run_report_module._baseline_metric_values(task_dir)

        self.assertEqual(values["task_score"], [1.25])

    def test_ranked_frontier_displays_the_task_primary_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir, run_dir = self._write_fixture(Path(tmp))
            manifest_path = run_dir / "frontier" / "frontier_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = manifest["lane_frontiers"]["confirmed"][0]
            entry.pop("metric_name")
            entry.pop("metric_value")
            entry["metrics"]["score"] = 999.0
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = generate_run_report(
                run_dir=run_dir,
                task_dir=task_dir,
                trigger="manual_primary_display",
                generation_id=0,
            )

            text = result.path.read_text(encoding="utf-8")
            self.assertIn("| `task_score` | 1.2 |", text)
            self.assertNotIn("| `score` | 999 |", text)

    def test_report_uses_run_docs_when_run_dir_is_outside_task_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            task_dir, source_run_dir = self._write_fixture(root)
            run_dir = root / "external_run"
            run_dir.mkdir()
            for child in source_run_dir.iterdir():
                if child.is_dir():
                    shutil.copytree(child, run_dir / child.name)
                else:
                    (run_dir / child.name).write_text(child.read_text(encoding="utf-8"))

            result = generate_run_report(
                run_dir=run_dir,
                task_dir=task_dir,
                trigger="manual_external_run_dir_test",
                generation_id=0,
            )

            self.assertEqual(result.path.parent, run_dir / "docs" / "praxist_reports")
            self.assertFalse((task_dir / "docs" / "praxist_reports").exists())

    def test_boundary_triggers_baseline_periodic_and_final_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir, run_dir = self._write_fixture(Path(tmp))

            first = maybe_generate_boundary_report(
                run_dir=run_dir,
                task_dir=task_dir,
                generation_id=0,
            )
            self.assertIsNotNone(first)
            assert first is not None
            self.assertEqual(first.trigger, "first_credible_baseline_beat")
            self.assertIsNone(
                maybe_generate_boundary_report(
                    run_dir=run_dir,
                    task_dir=task_dir,
                    generation_id=0,
                )
            )

            periodic = maybe_generate_boundary_report(
                run_dir=run_dir,
                task_dir=task_dir,
                generation_id=2,
            )
            self.assertIsNotNone(periodic)
            assert periodic is not None
            self.assertEqual(periodic.trigger, "periodic_3_generation")

            final = maybe_generate_boundary_report(
                run_dir=run_dir,
                task_dir=task_dir,
                generation_id=4,
                final=True,
            )
            self.assertIsNotNone(final)
            assert final is not None
            self.assertEqual(final.trigger, "final_run_completion")

            state = json.loads((run_dir / REPORT_STATE_REL).read_text(encoding="utf-8"))
            self.assertTrue(state["first_credible_baseline_beat_reported"])
            self.assertEqual(len(state["generated_reports"]), 3)

    def test_no_first_report_without_baseline_beat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir, run_dir = self._write_fixture(Path(tmp), frontier_score=0.8)

            self.assertIsNone(
                maybe_generate_boundary_report(
                    run_dir=run_dir,
                    task_dir=task_dir,
                    generation_id=0,
                )
            )

    def test_baseline_beat_requires_task_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()

            self.assertFalse(run_report_module._has_baseline_beat(run_dir=run_dir, task_dir=None))

    def test_boundary_trigger_uses_manifest_metric_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir, run_dir = self._write_fixture(Path(tmp), frontier_score=1.2)
            manifest_path = run_dir / "frontier" / "frontier_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["primary_metric"] = "task_score"
            manifest["metric_direction"] = "maximize"
            entry = manifest["lane_frontiers"]["confirmed"][0]
            entry.pop("metric_name")
            entry["metric_value"] = 1.2
            entry["metrics"] = {"scored_complete": True}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            first = maybe_generate_boundary_report(
                run_dir=run_dir,
                task_dir=task_dir,
                generation_id=0,
            )

            self.assertIsNotNone(first)
            assert first is not None
            self.assertEqual(first.trigger, "first_credible_baseline_beat")

    def test_minimize_metric_complex_manifest_and_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            run_dir = task_dir / "experiments" / "run_complex_report"
            (task_dir / "assets" / "baselines").mkdir(parents=True)
            (task_dir / "assets" / "baselines" / "results.jsonl").write_text(
                "\n"
                '{"_protocol":"skip","nested":{"loss":{"mean":1.0}}}\n'
                "{bad json}\n"
                '{"loss": 1.2, "_ignored": 99}\n',
                encoding="utf-8",
            )
            (run_dir / "frontier").mkdir(parents=True)
            (run_dir / "shared_findings").mkdir()
            (run_dir / "shared_findings" / "finding.json").write_text(
                json.dumps({"id": "finding-1"}),
                encoding="utf-8",
            )
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "primary_metric": "loss",
                        "metric_direction": "minimize",
                        "cumulative_top": [
                            {
                                "variant_id": "variant_low_loss",
                                "generation_id": 2,
                                "metric_value": 0.7,
                                "metrics": {
                                    "headline": "Nested baseline beat",
                                    "parent_usage": "validate",
                                },
                            }
                        ],
                        "lane_frontiers": {
                            "task_candidate": {
                                "cumulative": [{"variant_name": "candidate_a", "loss": 0.95}],
                                "top": [{"frontier_entity_key": "candidate_b", "loss": 0.9}],
                                "entries": [{"candidate_entity_key": "candidate_c", "loss": 0.8}],
                                "generations": {
                                    "2": [
                                        {
                                            "id": "candidate_d",
                                            "metrics": {"loss": 0.85},
                                            "extra": {"source_result_path": "x.json"},
                                        }
                                    ]
                                },
                            }
                        },
                        "validation_candidates": [
                            {"variant_name": "validation_signal", "metric_value": 1.1}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "gen_2").mkdir()
            (run_dir / "gen_2" / "generation_boundary.json").write_text(
                json.dumps({"generation_id": 2, "status": "complete"}),
                encoding="utf-8",
            )
            (run_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "finding_summary": {"accepted": 4},
                        "warnings": ["summary warning"],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "orchestrator_status.json").write_text(
                json.dumps({"max_generations": 6, "warnings": ["orchestrator warning"]}),
                encoding="utf-8",
            )

            first = maybe_generate_boundary_report(
                run_dir=run_dir,
                task_dir=task_dir,
                generation_id=2,
            )
            self.assertIsNotNone(first)
            assert first is not None
            text = first.path.read_text(encoding="utf-8")
            self.assertIn("variant_low_loss", text)
            self.assertIn("summary warning", text)
            self.assertIn("Latest generation boundary", text)

    def test_empty_report_and_safe_hook_failure_are_non_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_empty"
            run_dir.mkdir()

            result = generate_run_report(run_dir=run_dir, trigger="manual_empty")
            text = result.path.read_text(encoding="utf-8")
            self.assertIn("No clean frontier or Pareto-front entries were found yet.", text)
            self.assertIn("No numeric validation or result-summary signals were found yet.", text)
            self.assertIn("No lineage can be inferred from the available result signals yet.", text)

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.run_report.maybe_generate_boundary_report",
                side_effect=RuntimeError("boom"),
            ):
                self.assertIsNone(
                    generate_boundary_report_safely(
                        run_dir=run_dir,
                        task_dir=None,
                        generation_id=0,
                    )
                )

    def test_tool_adapter_and_plugin_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir, run_dir = self._write_fixture(Path(tmp))

            payload = tool_report(
                run_dir=str(run_dir),
                task_dir=str(task_dir),
                trigger="manual_tool_test",
                generation_id=1,
            )

            self.assertTrue(payload["ok"])
            self.assertTrue(Path(str(payload["report_path"])).exists())
            self.assertTrue(Path(str(payload["pdf_path"])).exists())
            self.assertIn("tool_server:run_report", DEFAULT_RESEARCH_TOOL_SERVER_REFS)
            self.assertIn("tool_server:run_report", PANEL_TOOL_SERVER_REFS)
            spec = tool_server_for_ref("tool_server:run_report")
            self.assertEqual(spec.server_name, "run-report")
            self.assertEqual(spec.tool_names, ("generate_run_report",))
        with patch(
            "praxist.plugins.tools.run_report.adapter.active_run_dir",
            return_value=None,
        ):
            self.assertFalse(tool_report()["ok"])

    def test_report_falls_back_to_dimension_winners_without_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            run_dir = task_dir / "experiments" / "run_signal_only"
            (task_dir / "assets" / "baselines").mkdir(parents=True)
            (task_dir / "assets" / "baselines" / "results.jsonl").write_text(
                '{"return_pct": 5.0, "loss": 1.0}\n',
                encoding="utf-8",
            )
            (task_dir / "task_spec.yaml").write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "primary_metric": "return_pct",
                            "direction": "maximize",
                            "anchor_metrics": [{"name": "loss", "direction": "minimize"}],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "frontier").mkdir(parents=True)
            (run_dir / "frontier" / "frontier_manifest.json").write_text("{}", encoding="utf-8")
            result_dir = run_dir / "results" / "gen2_peer7_signal"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "gen2_peer7_signal",
                        "source_generation_id": 2,
                        "tier_reached": "T1",
                        "result_status": "scored_complete",
                        "return_pct": 7.5,
                        "loss": 0.8,
                        "summary": "Single-sided signal that did not pass clean promotion.",
                    }
                ),
                encoding="utf-8",
            )

            result = generate_run_report(
                run_dir=run_dir,
                task_dir=task_dir,
                trigger="manual_signal_only",
                generation_id=2,
            )

            text = result.path.read_text(encoding="utf-8")
            self.assertIn("No clean frontier or Pareto-front entries were found yet.", text)
            self.assertIn("Mature Dimension Winners", text)
            self.assertIn("gen2_peer7_signal", text)
            self.assertIn("return_pct", text)
            self.assertIn("beats baseline 5", text)
            self.assertIn("loss", text)
            self.assertIn("beats baseline 1", text)
            self.assertIn("result summary", text)
            self.assertIsNotNone(result.pdf_path)
            assert result.pdf_path is not None
            self.assertTrue(result.pdf_path.exists())
            self.assertTrue(result.pdf_path.read_bytes().startswith(b"%PDF-"))

    def test_report_keeps_strong_preliminary_signal_without_ranking_it_as_mature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            run_dir = task_dir / "experiments" / "run_evidence_split"
            task_dir.mkdir()
            (task_dir / "task_spec.yaml").write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "primary_metric": "score",
                            "direction": "maximize",
                            "maturity_policy": {
                                "complete_stage_labels": ["complete"],
                                "preliminary_stage_labels": ["preview"],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "frontier").mkdir(parents=True)
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                "{}",
                encoding="utf-8",
            )
            for name, score, stage, partial in (
                ("mature_candidate", 0.8, "complete", False),
                ("strong_preview", 0.99, "preview", True),
            ):
                result_dir = run_dir / "results" / name
                result_dir.mkdir(parents=True)
                (result_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "variant_name": name,
                            "metrics": {
                                "score": score,
                                "evidence_stage": stage,
                                "scored_complete": not partial,
                                "partial": partial,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            payload = run_report_module.collect_report_payload(
                run_dir=run_dir,
                task_dir=task_dir,
            )

            mature = {winner["metric"]: winner for winner in payload["dimension_winners"]}
            signals = {winner["metric"]: winner for winner in payload["signal_dimension_winners"]}
            self.assertEqual(mature["score"]["entry"]["variant_name"], "mature_candidate")
            self.assertEqual(signals["score"]["entry"]["variant_name"], "strong_preview")
            self.assertEqual(signals["score"]["block_reason"], "not_scored_complete")

            result = generate_run_report(run_dir=run_dir, task_dir=task_dir)
            text = result.path.read_text(encoding="utf-8")
            self.assertIn("Mature Dimension Winners", text)
            self.assertIn("Strong Signals Requiring Validation", text)
            self.assertIn("mature_candidate", text)
            self.assertIn("strong_preview", text)

    def test_report_separates_authorized_reduced_maturity_from_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            run_dir = task_dir / "experiments" / "run_reduced_protocol"
            task_dir.mkdir()
            (task_dir / "task_spec.yaml").write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "primary_metric": "score",
                            "direction": "maximize",
                            "maturity_policy": {
                                "complete_stage_labels": ["reduced"],
                                "preliminary_stage_labels": ["diagnostic"],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "frontier").mkdir(parents=True)
            (run_dir / "frontier" / "frontier_manifest.json").write_text("{}", encoding="utf-8")
            result_dir = run_dir / "results" / "approved_reduced"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "approved_reduced",
                        "metrics": {
                            "score": 0.91,
                            "evidence_stage": "reduced",
                            "tier_status": "partial",
                            "scored_complete": True,
                            "partial": True,
                            "promotion_eligible": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            validation_dir = run_dir / "results" / "validation_only_reduced"
            validation_dir.mkdir(parents=True)
            (validation_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "validation_only_reduced",
                        "metrics": {
                            "score": 0.99,
                            "evidence_stage": "reduced",
                            "tier_status": "partial",
                            "scored_complete": True,
                            "partial": True,
                            "validation_only": True,
                            "promotion_eligible": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = run_report_module.collect_report_payload(
                run_dir=run_dir,
                task_dir=task_dir,
            )

            mature = {winner["metric"]: winner for winner in payload["dimension_winners"]}
            self.assertEqual(mature["score"]["entry"]["variant_name"], "approved_reduced")
            signals = {winner["metric"]: winner for winner in payload["signal_dimension_winners"]}
            self.assertEqual(
                signals["score"]["entry"]["variant_name"],
                "validation_only_reduced",
            )

            text = generate_run_report(run_dir=run_dir, task_dir=task_dir).path.read_text(
                encoding="utf-8"
            )
            self.assertIn("approved_reduced", text)
            self.assertIn("validation_only_reduced", text)
            self.assertIn("promotion_eligible=false", text)

    def test_nonpromotable_frontier_entry_is_rendered_only_as_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            run_dir = task_dir / "experiments" / "run_frontier_signal"
            task_dir.mkdir()
            (task_dir / "task_spec.yaml").write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "primary_metric": "score",
                            "direction": "maximize",
                            "maturity_policy": {"complete_stage_labels": ["reduced"]},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "frontier").mkdir(parents=True)
            entry = {
                "finding_id": "validation-signal",
                "variant_name": "validation_signal",
                "metric_name": "score",
                "metric_value": 1.0,
                "metrics": {
                    "score": 1.0,
                    "evidence_stage": "reduced",
                    "scored_complete": True,
                    "promotion_eligible": False,
                    "parent_eligible": False,
                },
            }
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                json.dumps({"cumulative_top": [entry]}),
                encoding="utf-8",
            )

            payload = run_report_module.collect_report_payload(
                run_dir=run_dir,
                task_dir=task_dir,
            )

            self.assertEqual(payload["top_entries"], [])
            self.assertEqual(payload["dimension_winners"], [])
            self.assertEqual(
                payload["signal_dimension_winners"][0]["entry"]["variant_name"],
                "validation_signal",
            )
            self.assertEqual(payload["strong_entries"], [])

    def test_result_summary_wins_dedupe_over_matching_validation_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            run_dir = task_dir / "experiments" / "run_dedupe_priority"
            task_dir.mkdir()
            (task_dir / "task_spec.yaml").write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "primary_metric": "score",
                            "direction": "maximize",
                            "maturity_policy": {
                                "complete_stage_labels": ["complete"],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "frontier").mkdir(parents=True)
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "validation_candidates": {
                            "cumulative": [
                                {
                                    "variant_name": "same_candidate",
                                    "generation_id": 0,
                                    "metric_name": "score",
                                    "metric_value": 1.0,
                                    "evidence_stage": "preview",
                                    "validation_only": True,
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            result_dir = run_dir / "results" / "same_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "result_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "same_candidate",
                        "generation_id": 0,
                        "metrics": {
                            "score": 1.0,
                            "evidence_stage": "complete",
                            "scored_complete": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = run_report_module.collect_report_payload(
                run_dir=run_dir,
                task_dir=task_dir,
            )

        winner = next(item for item in payload["dimension_winners"] if item["metric"] == "score")
        self.assertEqual(winner["entry"]["report_signal_source"], "result_summary")
        self.assertEqual(
            winner["entry"]["source_result_path"],
            "results/same_candidate/result_summary.json",
        )

    def test_matching_late_routing_fact_overrides_raw_summary_in_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            run_dir = task_dir / "experiments" / "run_late_result"
            task_dir.mkdir()
            (task_dir / "task_spec.yaml").write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "primary_metric": "score",
                            "direction": "maximize",
                            "maturity_policy": {"complete_stage_labels": ["complete"]},
                        }
                    }
                ),
                encoding="utf-8",
            )
            result_path = "results/late_candidate/result_summary.json"
            summary_path = run_dir / result_path
            summary_payload = {
                "variant_name": "late_candidate",
                "generation_id": 0,
                "metrics": {
                    "score": 1.0,
                    "evidence_stage": "complete",
                    "scored_complete": True,
                },
            }
            normalized = run_report_module.normalized_result_summary(
                summary_payload,
                summary_path=summary_path,
                maturity_policy={"complete_stage_labels": ["complete"]},
            )
            result_sha = run_report_module._json_digest(normalized)
            (run_dir / "frontier").mkdir(parents=True)
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "validation_candidates": {
                            "cumulative": [
                                {
                                    "variant_name": "late_candidate",
                                    "generation_id": 0,
                                    "metric_name": "score",
                                    "metric_value": 1.0,
                                    "evidence_stage": "complete",
                                    "source_result_path": result_path,
                                    "source_result_sha256": result_sha,
                                    "late_after_generation_boundary": True,
                                    "validation_only": True,
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(summary_payload),
                encoding="utf-8",
            )

            payload = run_report_module.collect_report_payload(
                run_dir=run_dir,
                task_dir=task_dir,
            )

        self.assertEqual(payload["dimension_winners"], [])
        signal = next(
            item for item in payload["signal_dimension_winners"] if item["metric"] == "score"
        )
        self.assertEqual(signal["entry"]["report_signal_source"], "validation_candidate")
        self.assertEqual(payload["strong_entries"], [])
        self.assertTrue(payload["charts"])
        self.assertTrue(
            all(
                str(chart.get("title") or "").startswith("Signal-only")
                for chart in payload["charts"]
            )
        )

    def test_first_report_trigger_can_use_credible_signal_without_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            run_dir = task_dir / "experiments" / "run_signal_trigger"
            (task_dir / "assets" / "baselines").mkdir(parents=True)
            (task_dir / "assets" / "baselines" / "results.jsonl").write_text(
                '{"return_pct": 5.0}\n',
                encoding="utf-8",
            )
            (task_dir / "task_spec.yaml").write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "primary_metric": "return_pct",
                            "direction": "maximize",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "frontier").mkdir(parents=True)
            (run_dir / "frontier" / "frontier_manifest.json").write_text("{}", encoding="utf-8")
            result_dir = run_dir / "results" / "gen1_peer2_signal"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "gen1_peer2_signal",
                        "source_generation_id": 1,
                        "tier_reached": "T1",
                        "result_status": "scored_complete",
                        "return_pct": 6.0,
                    }
                ),
                encoding="utf-8",
            )

            first = maybe_generate_boundary_report(
                run_dir=run_dir,
                task_dir=task_dir,
                generation_id=1,
            )

            self.assertIsNotNone(first)
            assert first is not None
            self.assertEqual(first.trigger, "first_credible_baseline_beat")

    def test_report_covers_top_level_frontier_lineage_and_charts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            run_dir = task_dir / "experiments" / "run_top_level_manifest"
            (task_dir / "assets" / "baselines").mkdir(parents=True)
            (task_dir / "assets" / "baselines" / "results.jsonl").write_text(
                '{"return_pct": 3.0, "risk_pct": 12.0}\n',
                encoding="utf-8",
            )
            (task_dir / "task_spec.yaml").write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "primary_metric": "return_pct",
                            "direction": "maximize",
                            "anchor_metrics": [{"name": "risk_pct", "direction": "minimize"}],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "frontier").mkdir(parents=True)
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "primary_metric": "return_pct",
                        "metric_direction": "maximize",
                        "cumulative_top": [
                            {
                                "variant_name": "top_parented",
                                "generation_id": 1,
                                "metric_value": 4.0,
                                "metrics": {
                                    "return_pct": 4.0,
                                    "risk_pct": 10.0,
                                    "parent_candidate": "baseline",
                                    "parent_usage": "direct improvement",
                                },
                                "source_result_path": "results/top_parented/eval_summary.json",
                                "summary": "Top-level frontier entry with explicit lineage.",
                            },
                            {
                                "variant_name": "top_next_gen",
                                "generation_id": 2,
                                "metric_value": 5.0,
                                "metrics": {"return_pct": 5.0, "risk_pct": 9.0},
                            },
                        ],
                        "frontier": [
                            {
                                "variant_id": "variant::frontier_alias",
                                "source_generation_id": 1,
                                "metrics": {"return_pct": 4.5, "risk_pct": 8.0},
                            }
                        ],
                        "frontier_summary": [{"id": "summary_entry", "metric_value": 4.1}],
                        "lane_frontiers": {
                            "exploration": [
                                {
                                    "candidate_entity_key": "lane_list_entry",
                                    "generation_id": 2,
                                    "metrics": {"return_pct": 4.2},
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = generate_run_report(
                run_dir=run_dir,
                task_dir=task_dir,
                trigger="manual_top_level",
                generation_id=1,
            )

            text = result.path.read_text(encoding="utf-8")
            self.assertIn("top_parented", text)
            self.assertIn("top_next_gen", text)
            self.assertIn("baseline", text)
            self.assertIn("direct improvement", text)
            self.assertIn("Best return_pct by generation", text)
            self.assertIn("return_pct vs risk_pct", text)
            self.assertIsNotNone(result.pdf_path)
            assert result.pdf_path is not None
            self.assertTrue(result.pdf_path.exists())
            self.assertGreater(result.pdf_path.stat().st_size, 200)

    def test_report_boundaries_for_unreadable_baseline_final_state_and_smoke_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            run_dir = task_dir / "experiments" / "run_boundaries"
            baseline_dir = task_dir / "assets" / "baselines"
            baseline_dir.mkdir(parents=True)
            (task_dir / "task_spec.yaml").write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "primary_metric": "return_pct",
                            "direction": "maximize",
                        }
                    }
                ),
                encoding="utf-8",
            )
            blocked_baseline = baseline_dir / "blocked.jsonl"
            blocked_baseline.mkdir()
            (run_dir / "frontier").mkdir(parents=True)
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "validation_candidates": [
                            {
                                "variant_name": "smoke_only",
                                "metric_name": "return_pct",
                                "metric_value": 1.0,
                                "metrics": {
                                    "return_pct": 1.0,
                                    "evidence_stage": "smoke",
                                    "source_result_path": "results/smoke/eval_summary.json",
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result_dir = run_dir / "results" / "nested_parent"
            result_dir.mkdir(parents=True)
            (result_dir / "eval_summary.json").write_text(
                json.dumps(
                    {
                        "description": "Result summary with parent metadata in nested metrics.",
                        "metrics": {
                            "return_pct": 2.0,
                            "parent_candidate": "prior_variant",
                            "parent_usage": "repair follow-up",
                            "source_generation_id": 3,
                            "result_status": "scored_complete",
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertIsNone(
                maybe_generate_boundary_report(
                    run_dir=run_dir,
                    task_dir=task_dir,
                    generation_id=3,
                )
            )

            final = maybe_generate_boundary_report(
                run_dir=run_dir,
                task_dir=task_dir,
                generation_id=3,
                final=True,
            )
            self.assertIsNotNone(final)
            self.assertIsNone(
                maybe_generate_boundary_report(
                    run_dir=run_dir,
                    task_dir=task_dir,
                    generation_id=3,
                    final=True,
                )
            )
            assert final is not None
            text = final.path.read_text(encoding="utf-8")
            self.assertIn("nested_parent", text)
            self.assertIn("prior_variant", text)
            self.assertIn("repair follow-up", text)
            self.assertIn("Validation candidates retained for follow-up", text)

    def test_task_metric_registry_drives_minimize_winners_charts_and_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            run_dir = task_dir / "experiments" / "run_minimize_registry"
            baseline_dir = task_dir / "assets" / "baselines"
            baseline_dir.mkdir(parents=True)
            (task_dir / "task_spec.yaml").write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "primary_metric": "rmse",
                            "direction": "minimize",
                            "anchor_metrics": [
                                {"name": "runtime_seconds", "direction": "minimize"}
                            ],
                            "frontier_lanes": [
                                {
                                    "name": "efficient",
                                    "axes": [{"name": "parameter_count", "direction": "minimize"}],
                                    "optional_axes": [
                                        {"name": "memory_mb", "direction": "minimize"}
                                    ],
                                }
                            ],
                        },
                        "baselines": [
                            {
                                "name": "base",
                                "metric_name": "energy_joules",
                                "metric_value": 12.0,
                                "direction": "minimize",
                            }
                        ],
                        "gems": {"result_metric_aliases": {"rmse": "validation_rmse"}},
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "results.jsonl").write_text(
                json.dumps(
                    {
                        "validation_rmse": 1.0,
                        "runtime_seconds": 20.0,
                        "parameter_count": 200.0,
                        "memory_mb": 40.0,
                        "energy_joules": 12.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "frontier").mkdir(parents=True)
            (run_dir / "frontier" / "frontier_manifest.json").write_text("{}", encoding="utf-8")
            for generation, variant, scale in (
                (0, "larger", 1.0),
                (1, "smaller", 0.5),
            ):
                result_dir = run_dir / "results" / variant
                result_dir.mkdir(parents=True)
                (result_dir / "result_summary.json").write_text(
                    json.dumps(
                        {
                            "variant_name": variant,
                            "source_generation_id": generation,
                            "result_status": "scored_complete",
                            "validation_rmse": 0.8 * scale,
                            "runtime_seconds": 16.0 * scale,
                            "parameter_count": 160.0 * scale,
                            "memory_mb": 32.0 * scale,
                            "energy_joules": 10.0 * scale,
                        }
                    ),
                    encoding="utf-8",
                )

            payload = run_report_module.collect_report_payload(
                run_dir=run_dir,
                task_dir=task_dir,
            )
            winners = {winner["metric"]: winner for winner in payload["dimension_winners"]}

            for metric in (
                "rmse",
                "validation_rmse",
                "runtime_seconds",
                "parameter_count",
                "memory_mb",
                "energy_joules",
            ):
                self.assertEqual(winners[metric]["direction"], "minimize")
                self.assertEqual(winners[metric]["entry"]["variant_name"], "smaller")
                self.assertIn("beats baseline", winners[metric]["baseline_relation"])
            self.assertTrue(
                any(
                    any(item.get("label") == "rmse" for item in chart.get("items", []))
                    for chart in payload["charts"]
                )
            )
            trend = next(
                chart
                for chart in payload["charts"]
                if chart.get("kind") == "line" and chart.get("metric") == "rmse"
            )
            self.assertEqual(
                [point["generation"] for point in trend["points"]],
                [0, 1],
            )
            registry = run_report_module._task_metric_registry(task_dir)
            self.assertEqual(
                run_report_module._metric_direction_for_name(
                    "rmse",
                    {"metric_name": "rmse", "metric_direction": "maximize"},
                    metric_registry=registry,
                ),
                "minimize",
            )

    def test_task_metric_registry_rejects_conflicted_task_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task"
            task_dir.mkdir()
            (task_dir / "task_spec.yaml").write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "primary_metric": "rmse",
                            "direction": "minimize",
                        },
                        "baselines": [
                            {
                                "name": "conflicting_baseline",
                                "metric_name": "validation_rmse",
                                "metric_value": 1.0,
                                "direction": "maximize",
                            }
                        ],
                        "gems": {"result_metric_aliases": {"rmse": "validation_rmse"}},
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(run_report_module.logger, "warning") as warning:
                registry = run_report_module._task_metric_registry(task_dir)

            self.assertEqual(registry.directions, {})
            self.assertEqual(registry.declared_metrics, {"rmse", "validation_rmse"})
            self.assertIsNone(
                run_report_module._metric_direction_for_name(
                    "rmse",
                    {"metric_name": "rmse", "metric_direction": "maximize"},
                    metric_registry=registry,
                )
            )
            warning.assert_called_once()

    def test_metric_registry_tolerates_invalid_task_and_missing_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task"
            task_dir.mkdir()
            (task_dir / "task.yaml").write_text("[invalid", encoding="utf-8")

            with patch.object(run_report_module.logger, "warning") as warning:
                registry = run_report_module._task_metric_registry(task_dir)

            self.assertEqual(registry.directions, {})
            self.assertEqual(registry.declared_metrics, frozenset())
            self.assertEqual(run_report_module._baseline_metric_values(task_dir), {})
            warning.assert_called_once()

    def test_unknown_direction_is_unranked_and_never_claims_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            run_dir = task_dir / "experiments" / "run_unknown_direction"
            baseline_dir = task_dir / "assets" / "baselines"
            baseline_dir.mkdir(parents=True)
            (task_dir / "task_spec.yaml").write_text(
                json.dumps(
                    {
                        "evaluation": {
                            "primary_metric": "known_score",
                            "direction": "maximize",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "results.jsonl").write_text(
                '{"mystery_metric": 10.0}\n', encoding="utf-8"
            )
            (run_dir / "frontier").mkdir(parents=True)
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "cumulative_top": [
                            {
                                "variant_name": "unknown_high",
                                "metric_name": "mystery_metric",
                                "metric_value": 20.0,
                                "metrics": {"mystery_metric": 20.0, "scored_complete": True},
                            },
                            {
                                "variant_name": "unknown_low",
                                "metric_name": "mystery_metric",
                                "metric_value": 5.0,
                                "metrics": {"mystery_metric": 5.0, "scored_complete": True},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            payload = run_report_module.collect_report_payload(
                run_dir=run_dir,
                task_dir=task_dir,
            )
            self.assertEqual(payload["top_entries"], [])
            self.assertEqual(len(payload["unranked_frontier_entries"]), 2)
            self.assertNotIn(
                "mystery_metric",
                {winner["metric"] for winner in payload["dimension_winners"]},
            )
            self.assertFalse(
                any(
                    chart.get("metric") == "mystery_metric"
                    or chart.get("x_metric") == "mystery_metric"
                    or chart.get("y_metric") == "mystery_metric"
                    for chart in payload["charts"]
                )
            )
            self.assertIsNone(
                maybe_generate_boundary_report(
                    run_dir=run_dir,
                    task_dir=task_dir,
                    generation_id=0,
                )
            )
            report = generate_run_report(run_dir=run_dir, task_dir=task_dir)
            text = report.path.read_text(encoding="utf-8")
            self.assertIn("Unranked Frontier Entries (Direction Unknown)", text)
            self.assertIn("mystery_metric", text)
            self.assertNotIn("beats baseline 10", text)

    def test_run_report_helper_edge_cases_are_tolerant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            results_dir = run_dir / "results"
            (results_dir / "a").mkdir(parents=True)
            (results_dir / "a" / "ignore_me.json").write_text(
                '{"return_pct": 99}', encoding="utf-8"
            )
            (results_dir / "b").mkdir()
            (results_dir / "b" / "eval_summary.json").write_text("[1, 2]", encoding="utf-8")
            (results_dir / "c").mkdir()
            (results_dir / "c" / "result_summary.json").write_text(
                '{"variant_name":"no_metrics","summary":"ignored"}',
                encoding="utf-8",
            )
            (results_dir / "d").mkdir()
            (results_dir / "d" / "final_summary.json").write_text(
                '{"metrics":{"return_pct":1.5,"variant_name":"variant::metric_named"}}',
                encoding="utf-8",
            )

            validation = run_report_module._collect_validation_candidates(
                {
                    "validation_candidates": {
                        "generations": {
                            "1": [{"variant_name": "from_generation"}],
                            "bad": "ignored",
                        }
                    }
                }
            )
            shared = run_report_module._collect_shared_finding_entries(
                [
                    {"title": "Finding for gen3_peer7_candidate", "metrics": {"score": 1.0}},
                    {
                        "metrics": {
                            "variant_name": "variant::metric_variant",
                            "score": 2.0,
                            "parent_candidate": "parent",
                        }
                    },
                    {"title": "no metrics"},
                ]
            )
            summaries = run_report_module._collect_result_summary_entries(run_dir, cap=2)
            winners = run_report_module._entries_from_dimension_winners(
                [
                    {"entry": "not a dict"},
                    {"entry": {"variant_name": "dup", "score": 1}},
                    {"entry": {"variant_name": "dup", "score": 2}},
                ]
            )
            pdf = run_report_module._SimplePdf()
            pdf.draw_chart({"kind": "bar", "items": []})
            pdf.draw_chart({"kind": "line", "points": [{"value": 1}]})
            pdf.draw_chart(
                {
                    "kind": "line",
                    "points": [
                        {"generation": 0, "value": 1},
                        {"generation": 1, "value": 1},
                    ],
                }
            )
            pdf.draw_chart({"kind": "scatter", "points": [{"x": 1, "y": 1}]})
            pdf._y = pdf.margin
            pdf._ensure_space(1)

            self.assertEqual(validation[0]["variant_name"], "from_generation")
            self.assertEqual(shared[0]["variant_name"], "gen3_peer7_candidate")
            self.assertEqual(shared[1]["variant_name"], "metric_variant")
            self.assertEqual(summaries[0]["variant_name"], "d")
            self.assertEqual(
                run_report_module._best_metric({"metrics": {"reward": 3.0}}), ("reward", 3.0)
            )
            self.assertEqual(
                run_report_module._variant_name({"metrics": {"variant_id": "variant::inner"}}),
                "inner",
            )
            self.assertEqual(len(winners), 1)
            self.assertTrue(run_report_module._metric_value_better(1.0, None, direction="maximize"))
            self.assertEqual(
                run_report_module._baseline_relation(
                    "score",
                    1.0,
                    direction="maximize",
                    baseline_values={"score": [1.0]},
                ),
                "ties baseline 1",
            )
            self.assertEqual(run_report_module._pad_range(2.0, 2.0), (1.0, 3.0))
            self.assertIsNone(run_report_module._number(float("nan")))
            self.assertTrue(run_report_module._clip("x" * 20, 5).endswith("…"))
            self.assertEqual(
                run_report_module._entry_signal_label(shared[0]), "shared finding / unknown"
            )
            self.assertEqual(
                run_report_module._entry_signal_label(
                    {"report_signal_source": "validation_candidate"}
                ),
                "validation candidate / unknown",
            )

    def test_result_report_prefers_current_aggregate_over_stale_legacy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result_dir = run_dir / "results" / "candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "result_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "candidate",
                        "result_status": "scored_complete",
                        "metrics": {"score": 1.0},
                        "current_aggregate": {
                            "score": 10.0,
                            "scored_complete": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            entries = run_report_module._collect_result_summary_entries(run_dir, cap=10)

            self.assertEqual(len(entries), 1)
            metrics = entries[0]["metrics"]
            self.assertEqual(metrics["score"], 10.0)
            self.assertFalse(metrics["scored_complete"])
            self.assertFalse(run_report_module._credible_for_report_trigger(entries[0]))

    def test_protocol_failed_result_is_never_reported_as_credible(self) -> None:
        entry = {
            "report_signal_source": "result_summary",
            "metrics": {
                "score": 10.0,
                "evidence_stage": "approved_reduced",
                "scored_complete": True,
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "protocol_integrity_passed": False,
            },
        }
        policy = {
            "complete_stage_labels": ["approved_reduced"],
            "require_ratio_gate": False,
        }

        self.assertFalse(
            run_report_module._credible_for_report_trigger(
                entry,
                maturity_policy=policy,
            )
        )
        self.assertEqual(
            run_report_module._report_evidence_class(
                entry,
                maturity_policy=policy,
            ),
            "signal",
        )

    def test_incomplete_or_suspect_results_do_not_trigger_credible_report(self) -> None:
        policy = {"min_effort_ratio": 0.5, "min_coverage_ratio": 0.5}
        complete_metrics = {
            "score": 10.0,
            "effort_ratio": 1.0,
            "coverage_ratio": 1.0,
            "scored_complete": True,
        }
        incomplete = {
            "report_signal_source": "result_summary",
            "metrics": {
                **complete_metrics,
                "scored_complete": False,
                "incomplete_eval": True,
            },
        }
        suspect = {
            "report_signal_source": "result_summary",
            "metrics": {**complete_metrics, "suspect_protocol": True},
        }

        for entry in (incomplete, suspect):
            self.assertFalse(
                run_report_module._credible_for_report_trigger(
                    entry,
                    maturity_policy=policy,
                )
            )
            self.assertEqual(
                run_report_module._report_evidence_class(
                    entry,
                    maturity_policy=policy,
                ),
                "signal",
            )

    def test_failed_frontier_entry_is_reported_as_signal(self) -> None:
        entry = {
            "report_signal_source": "frontier",
            "status": "failed",
            "metrics": {"score": 10.0},
        }

        self.assertFalse(run_report_module._credible_for_report_trigger(entry))
        self.assertEqual(
            run_report_module._report_evidence_class(entry, maturity_policy={}),
            "signal",
        )

    def test_tier_does_not_authorize_conflicting_reduced_mode_in_report(self) -> None:
        entry = {
            "report_signal_source": "result_summary",
            "metrics": {
                "score": 10.0,
                "tier_reached": "full",
                "scored_complete": True,
                "capped": True,
            },
        }
        policy = {"complete_stage_labels": ["full"]}

        self.assertFalse(
            run_report_module._credible_for_report_trigger(
                entry,
                maturity_policy=policy,
            )
        )
        self.assertEqual(
            run_report_module._report_evidence_class(
                entry,
                maturity_policy=policy,
            ),
            "signal",
        )

    def test_report_dedupe_keeps_distinct_immutable_results_in_either_order(self) -> None:
        clean = {
            "variant_name": "same_display",
            "generation_id": 0,
            "metric_name": "score",
            "metric_value": 1.0,
            "source_result_path": "results/clean.json",
            "source_result_sha256": "clean-sha",
        }
        stale_validation = {
            **clean,
            "source_result_path": "results/stale.json",
            "source_result_sha256": "stale-sha",
            "validation_only": True,
        }

        for entries in ([stale_validation, clean], [clean, stale_validation]):
            deduped = run_report_module._dedupe_entries(entries)
            self.assertEqual(len(deduped), 2)
            self.assertEqual(
                {entry["source_result_sha256"] for entry in deduped},
                {"clean-sha", "stale-sha"},
            )

    def test_report_dedupe_applies_routing_to_same_artifact_across_display_fields(self) -> None:
        raw = {
            "variant_name": "raw_name",
            "generation_id": 0,
            "metric_name": "score",
            "metric_value": 1.0,
            "source_result_path": "results/shared.json",
            "source_result_sha256": "shared-sha",
            "scored_complete": True,
        }
        restricted = {
            **raw,
            "variant_name": "canonical_name",
            "generation_id": 1,
            "metric_name": "alternate_score",
            "validation_only": True,
        }

        for entries in ([raw, restricted], [restricted, raw]):
            deduped = run_report_module._dedupe_entries(entries)
            self.assertEqual(len(deduped), 1)
            self.assertTrue(deduped[0]["validation_only"])
            self.assertFalse(run_report_module._credible_for_report_trigger(deduped[0]))

    def test_result_report_normalizes_tier_only_legacy_metrics_before_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result_dir = run_dir / "results" / "tier_candidate"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "tier_candidate",
                        "tiers": [
                            {
                                "tier": "complete",
                                "status": "scored_complete",
                                "metrics_summary": {"score": 3.0},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            entries = run_report_module._collect_result_summary_entries(run_dir, cap=10)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["metrics"]["score"], 3.0)
            self.assertTrue(entries[0]["metrics"]["scored_complete"])

    def test_codex_runtime_maps_run_report_server_directly(self) -> None:
        import sys

        from praxist.plugins.agent_runtimes.codex_sdk._mcp import (
            MCP_STDIO_MODULE,
            mcp_configuration,
        )

        result = mcp_configuration([{"server_name": "run-report"}])
        server = result.config["mcp_servers"]["run-report"]

        self.assertEqual(server["command"], sys.executable)
        self.assertEqual(server["args"][:2], ["-m", MCP_STDIO_MODULE])
        self.assertIn("run_report.adapter:create_run_report_server", server["args"][2])


if __name__ == "__main__":
    unittest.main()
