from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from praxist.core.task_project import build_task_project_manifest
from praxist.plugins.workflow_stages.research_loop import startup
from praxist.plugins.workflow_stages.research_loop.backend import resume_state
from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
    ResumePlan,
    append_resume_event,
    ensure_resumable_run_dir,
    inspect_resume_plan,
    load_generation_results,
    lock_pid,
    pid_is_alive,
    repair_inferred_gems_boundary_markers,
    validate_resume_startup_identity,
    write_boundary_marker,
)


class ResearchLoopResumeContractsTest(unittest.TestCase):
    def test_resume_plan_serialization_and_policy_guard(self) -> None:
        plan = ResumePlan(
            enabled=True,
            policy="completed_generation",
            start_generation=2,
            completed_generations=2,
            pending_boundary_generation=2,
            warnings=("needs boundary repair",),
        )

        self.assertTrue(plan.has_pending_boundary)
        self.assertEqual(
            plan.to_dict(),
            {
                "enabled": True,
                "policy": "completed_generation",
                "start_generation": 2,
                "completed_generations": 2,
                "pending_boundary_generation": 2,
                "warnings": ["needs boundary repair"],
            },
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "unsupported resume policy"),
        ):
            inspect_resume_plan(
                Path(tmp),
                max_generations=1,
                pi_enabled=False,
                policy="unsafe",
            )

    def test_resume_sidecar_preparation_clears_only_a_rerun_cohort_signals(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_resume

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            names = (
                "CLOSING_SIGNAL",
                "STOP_SIGNAL",
                "STOP_SIGNAL_POSTGEN",
                "CLOSING_SIGNAL.tmp",
                "STOP_SIGNAL.tmp",
            )
            for name in names:
                (gen_dir / name).write_text("stale\n", encoding="utf-8")

            plan = generation_resume.prepare_resume_for_sidecars(
                run_dir,
                max_generations=2,
                pi_enabled=True,
                policy="completed_generation",
            )

            self.assertEqual(plan.start_generation, 0)
            self.assertFalse(plan.has_pending_boundary)
            self.assertFalse(any((gen_dir / name).exists() for name in names))

    def test_resume_sidecar_preparation_preserves_pending_boundary_signals(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_resume

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)
            signal = run_dir / "gen_0" / "CLOSING_SIGNAL"
            signal.write_text("trigger_reason=boundary_finalization\n", encoding="utf-8")

            plan = generation_resume.prepare_resume_for_sidecars(
                run_dir,
                max_generations=2,
                pi_enabled=True,
                policy="completed_generation",
            )

            self.assertTrue(plan.has_pending_boundary)
            self.assertEqual(plan.pending_boundary_generation, 0)
            self.assertTrue(signal.exists())

    def test_resume_artifact_helpers_cover_invalid_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                ensure_resumable_run_dir(run_dir)

            run_dir.write_text("not a dir", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                ensure_resumable_run_dir(run_dir)
            run_dir.unlink()

            run_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "missing Praxist startup artifacts"):
                ensure_resumable_run_dir(run_dir)
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            (run_dir / "startup_config.json").write_text("{}", encoding="utf-8")
            ensure_resumable_run_dir(run_dir)

            bad_json = run_dir / "bad.json"
            bad_json.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                resume_state._read_json_object_for_resume(bad_json)
            list_json = run_dir / "list.json"
            list_json.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                resume_state._read_json_object_for_resume(list_json)

            self.assertEqual(resume_state._read_json_object_optional(bad_json), {})
            self.assertEqual(resume_state._read_json_object_optional(list_json), {})
            self.assertEqual(load_generation_results(run_dir, 99), [])
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "generation_results.json").write_text("{}", encoding="utf-8")
            self.assertEqual(load_generation_results(run_dir, 0), [])
            (gen_dir / "generation_results.json").write_text("{", encoding="utf-8")
            self.assertEqual(load_generation_results(run_dir, 0), [])

            append_resume_event(run_dir, {"event": "resume.test", "path": run_dir})
            event = json.loads((run_dir / "resume_events.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(event["event"], "resume.test")
            self.assertEqual(event["schema_version"], "praxist.resume_event.v1")

    def test_resume_plan_starts_after_completed_generation_with_agenda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)
            self._write_frontier_manifest(run_dir, {"0": [{"variant_name": "v0"}]})
            agenda = run_dir / "agendas" / "research_agenda_gen1.yaml"
            agenda.parent.mkdir(parents=True)
            agenda.write_text("generation: 1\npeer_contracts: {}\n", encoding="utf-8")

            plan = inspect_resume_plan(
                run_dir,
                max_generations=3,
                pi_enabled=True,
            )

            self.assertEqual(plan.start_generation, 1)
            self.assertEqual(plan.completed_generations, 1)
            self.assertIsNone(plan.pending_boundary_generation)

    def test_resume_plan_accepts_legacy_fenced_next_agenda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)
            self._write_frontier_manifest(run_dir, {"0": [{"variant_name": "v0"}]})
            agenda = run_dir / "agendas" / "research_agenda_gen1.yaml"
            agenda.parent.mkdir(parents=True)
            agenda.write_text(
                "```yaml\ngeneration: 1\npeer_contracts: {}\n```\n",
                encoding="utf-8",
            )

            plan = inspect_resume_plan(
                run_dir,
                max_generations=3,
                pi_enabled=True,
            )

            self.assertEqual(plan.start_generation, 1)
            self.assertEqual(plan.completed_generations, 1)
            self.assertIsNone(plan.pending_boundary_generation)

    def test_modern_resume_requires_marker_even_with_frontier_and_agenda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "startup_config.json").write_text(
                json.dumps({"schema_version": "praxist.startup.v1"}),
                encoding="utf-8",
            )
            self._write_generation_results(run_dir, 0)
            self._write_frontier_manifest(run_dir, {"0": [{"variant_name": "v0"}]})
            agenda = run_dir / "agendas" / "research_agenda_gen1.yaml"
            agenda.parent.mkdir(parents=True)
            agenda.write_text("generation: 1\npeer_contracts: {}\n", encoding="utf-8")

            plan = inspect_resume_plan(run_dir, max_generations=3, pi_enabled=True)

            self.assertEqual(plan.start_generation, 0)
            self.assertEqual(plan.completed_generations, 0)
            self.assertEqual(plan.pending_boundary_generation, 0)
            self.assertIn("required generation boundary marker is missing", plan.warnings[0])

    def test_resume_plan_repairs_finished_cohort_without_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)

            plan = inspect_resume_plan(
                run_dir,
                max_generations=3,
                pi_enabled=True,
            )

            self.assertEqual(plan.start_generation, 0)
            self.assertEqual(plan.completed_generations, 0)
            self.assertEqual(plan.pending_boundary_generation, 0)
            self.assertTrue(plan.warnings)

    def test_boundary_marker_counts_generation_complete_without_legacy_agenda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)
            write_boundary_marker(
                run_dir,
                gen_id=0,
                promoted_count=0,
                pi_status="failed_non_strict",
            )

            plan = inspect_resume_plan(
                run_dir,
                max_generations=3,
                pi_enabled=True,
            )

            self.assertEqual(plan.start_generation, 1)
            self.assertEqual(plan.completed_generations, 1)

    def test_boundary_marker_persists_final_evidence_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cutoff_at = "2027-01-15T08:01:00+00:00"
            write_boundary_marker(
                run_dir,
                gen_id=0,
                promoted_count=1,
                pi_status="succeeded",
                evidence_cutoff_at=cutoff_at,
                evidence_source_snapshot_at_cutoff={
                    "results/z/summary.json": "target:1:9",
                    "results/a/summary.json": "target:1:2",
                },
            )

            payload = json.loads(
                (run_dir / "gen_0" / "generation_boundary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["evidence_cutoff_at"], cutoff_at)
        self.assertEqual(
            payload["evidence_source_snapshot_at_cutoff"],
            {
                "results/a/summary.json": "target:1:2",
                "results/z/summary.json": "target:1:9",
            },
        )

    def test_boundary_checkpoint_retries_without_overwriting_concurrent_close_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "generation_results.json").write_text("[]", encoding="utf-8")
            signal = gen_dir / "CLOSING_SIGNAL"
            signal.write_text("trigger_reason=initial\ngen_id=0\n", encoding="utf-8")
            cutoff = datetime.now(UTC)
            source_snapshot = {"results/candidate.json": "target:1:2"}
            append = resume_state._append_text_durable
            calls = 0

            def append_with_one_signal_replacement(path: Path, payload: str) -> None:
                nonlocal calls
                calls += 1
                append(path, payload)
                if calls == 1:
                    path.write_text(
                        "trigger_reason=mature_quorum\ngen_id=0\nfindings_count=3\n",
                        encoding="utf-8",
                    )

            with patch.object(
                resume_state,
                "_append_text_durable",
                side_effect=append_with_one_signal_replacement,
            ):
                written = resume_state.write_boundary_evidence_checkpoint(
                    run_dir,
                    gen_id=0,
                    cutoff=cutoff,
                    evidence_source_snapshot=source_snapshot,
                )

            self.assertTrue(written)
            self.assertEqual(calls, 2)
            signal_text = signal.read_text(encoding="utf-8")
            self.assertIn("trigger_reason=mature_quorum", signal_text)
            self.assertIn("findings_count=3", signal_text)
            self.assertEqual(
                resume_state.read_boundary_evidence_checkpoint(run_dir, 0),
                (cutoff, source_snapshot),
            )

    def test_boundary_checkpoint_raises_after_repeated_signal_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "generation_results.json").write_text("[]", encoding="utf-8")
            signal = gen_dir / "CLOSING_SIGNAL"
            signal.write_text("trigger_reason=initial\ngen_id=0\n", encoding="utf-8")
            cutoff = datetime.now(UTC)
            source_snapshot = {"results/candidate.json": "target:1:2"}
            append = resume_state._append_text_durable
            calls = 0

            def append_then_replace_signal(path: Path, payload: str) -> None:
                nonlocal calls
                calls += 1
                append(path, payload)
                path.write_text(
                    "trigger_reason=concurrent_owner\ngen_id=0\n",
                    encoding="utf-8",
                )

            with (
                patch.object(
                    resume_state,
                    "_append_text_durable",
                    side_effect=append_then_replace_signal,
                ),
                self.assertRaisesRegex(OSError, "after 3 attempts"),
            ):
                resume_state.write_boundary_evidence_checkpoint(
                    run_dir,
                    gen_id=0,
                    cutoff=cutoff,
                    evidence_source_snapshot=source_snapshot,
                )

            self.assertEqual(calls, 3)
            self.assertIn("trigger_reason=concurrent_owner", signal.read_text(encoding="utf-8"))
            self.assertIsNone(resume_state.read_boundary_evidence_checkpoint(run_dir, 0))

    def test_gems_reset_event_counts_boundary_complete_without_marker_or_agenda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)
            self._write_generation_results(run_dir, 1)
            self._write_generation_results(run_dir, 2)
            write_boundary_marker(run_dir, gen_id=0, promoted_count=1, pi_status="succeeded")
            write_boundary_marker(run_dir, gen_id=1, promoted_count=1, pi_status="succeeded")
            gems_dir = run_dir / "gems"
            gems_dir.mkdir(parents=True)
            (gems_dir / "gems_state.json").write_text(
                json.dumps(
                    {
                        "cycle_start_generation": 3,
                        "reset_count": 1,
                        "reset_events": [
                            {
                                "reset_count": 1,
                                "completed_gen_id": 2,
                                "next_absolute_generation": 3,
                                "reason": "periodic_reset_every_6_generations:completed_in_cycle=6",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {},
                        "cumulative_top": [],
                        "lane_frontiers": {},
                        "gems": {"reset_count": 1, "cycle_start_generation": 3},
                    }
                ),
                encoding="utf-8",
            )

            plan = inspect_resume_plan(
                run_dir,
                max_generations=5,
                pi_enabled=True,
            )

            self.assertEqual(plan.start_generation, 3)
            self.assertEqual(plan.completed_generations, 3)
            self.assertIsNone(plan.pending_boundary_generation)

    def test_gems_reset_event_with_stale_manifest_does_not_complete_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)
            self._write_generation_results(run_dir, 1)
            self._write_generation_results(run_dir, 2)
            write_boundary_marker(run_dir, gen_id=0, promoted_count=1, pi_status="succeeded")
            write_boundary_marker(run_dir, gen_id=1, promoted_count=1, pi_status="succeeded")
            gems_dir = run_dir / "gems"
            gems_dir.mkdir(parents=True)
            (gems_dir / "gems_state.json").write_text(
                json.dumps(
                    {
                        "cycle_start_generation": 3,
                        "reset_count": 1,
                        "reset_events": [
                            {
                                "reset_count": 1,
                                "completed_gen_id": 2,
                                "next_absolute_generation": 3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self._write_frontier_manifest(run_dir, {"2": [{"variant_name": "stale"}]})

            plan = inspect_resume_plan(
                run_dir,
                max_generations=5,
                pi_enabled=True,
            )

            self.assertEqual(plan.pending_boundary_generation, 2)
            self.assertTrue(plan.warnings)

    def test_historical_committed_gems_reset_repairs_marker_even_after_later_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)
            self._write_generation_results(run_dir, 1)
            self._write_generation_results(run_dir, 2)
            write_boundary_marker(run_dir, gen_id=0, promoted_count=1, pi_status="succeeded")
            write_boundary_marker(run_dir, gen_id=1, promoted_count=1, pi_status="succeeded")
            cutoff = datetime.now(UTC)
            source_snapshot = {"results/gen2_peer0/summary.json": "target:1:2"}
            self.assertTrue(
                resume_state.write_boundary_evidence_checkpoint(
                    run_dir,
                    gen_id=2,
                    cutoff=cutoff,
                    evidence_source_snapshot=source_snapshot,
                )
            )
            gems_dir = run_dir / "gems"
            gems_dir.mkdir(parents=True)
            (gems_dir / "gems_state.json").write_text(
                json.dumps(
                    {
                        "cycle_start_generation": 6,
                        "reset_count": 2,
                        "reset_events": [
                            {
                                "reset_count": 1,
                                "completed_gen_id": 2,
                                "next_absolute_generation": 3,
                                "committed": True,
                            },
                            {
                                "reset_count": 2,
                                "completed_gen_id": 5,
                                "next_absolute_generation": 6,
                                "committed": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {},
                        "cumulative_top": [],
                        "lane_frontiers": {},
                        "gems": {"reset_count": 2, "cycle_start_generation": 6},
                    }
                ),
                encoding="utf-8",
            )

            plan = inspect_resume_plan(
                run_dir,
                max_generations=8,
                pi_enabled=True,
            )
            repairs = repair_inferred_gems_boundary_markers(
                run_dir,
                max_generations=8,
                pi_enabled=True,
            )

            self.assertEqual(plan.start_generation, 3)
            self.assertEqual(repairs[0]["generation_id"], 2)
            marker = json.loads((run_dir / "gen_2" / "generation_boundary.json").read_text())
            self.assertEqual(marker["pi_status"], "skipped_gems_reset_repaired")
            self.assertEqual(marker["evidence_cutoff_at"], cutoff.isoformat())
            self.assertEqual(
                marker["evidence_source_snapshot_at_cutoff"],
                source_snapshot,
            )
            self.assertIsNone(resume_state.read_boundary_evidence_checkpoint(run_dir, 2))

    def test_pending_gems_reset_does_not_count_as_completed_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)
            self._write_generation_results(run_dir, 1)
            self._write_generation_results(run_dir, 2)
            write_boundary_marker(run_dir, gen_id=0, promoted_count=1, pi_status="succeeded")
            write_boundary_marker(run_dir, gen_id=1, promoted_count=1, pi_status="succeeded")
            gems_dir = run_dir / "gems"
            gems_dir.mkdir(parents=True)
            (gems_dir / "gems_state.json").write_text(
                json.dumps(
                    {
                        "cycle_start_generation": 3,
                        "reset_count": 1,
                        "pending_reset": {"completed_gen_id": 2},
                        "reset_events": [
                            {
                                "reset_count": 1,
                                "completed_gen_id": 2,
                                "next_absolute_generation": 3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(parents=True)
            (frontier_dir / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {},
                        "cumulative_top": [],
                        "lane_frontiers": {},
                        "gems": {"reset_count": 1, "cycle_start_generation": 3},
                    }
                ),
                encoding="utf-8",
            )

            plan = inspect_resume_plan(
                run_dir,
                max_generations=5,
                pi_enabled=True,
            )

            self.assertEqual(plan.pending_boundary_generation, 2)

    def test_final_generation_pending_gems_reset_requires_boundary_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)
            self._write_generation_results(run_dir, 1)
            write_boundary_marker(run_dir, gen_id=0, promoted_count=1, pi_status="succeeded")
            gems_dir = run_dir / "gems"
            gems_dir.mkdir(parents=True)
            (gems_dir / "gems_state.json").write_text(
                json.dumps(
                    {
                        "cycle_start_generation": 0,
                        "reset_count": 0,
                        "pending_reset": {"completed_gen_id": 1},
                    }
                ),
                encoding="utf-8",
            )
            self._write_frontier_manifest(run_dir, {"1": [{"variant_name": "final_alpha"}]})

            plan = inspect_resume_plan(
                run_dir,
                max_generations=2,
                pi_enabled=True,
            )

            self.assertEqual(plan.start_generation, 1)
            self.assertEqual(plan.completed_generations, 1)
            self.assertEqual(plan.pending_boundary_generation, 1)
            self.assertTrue(plan.warnings)

    def test_startup_accepts_existing_run_dir_only_in_resume_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            (run_dir / "startup_config.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(ValueError) as cm:
                startup._ensure_fresh_run_dir(run_dir)
            self.assertIn("already contains Praxist run artifacts", str(cm.exception))
            self.assertIn("Use --resume or --resume-from", str(cm.exception))
            self.assertNotIn("Resume mode is not implemented", str(cm.exception))

            startup._ensure_fresh_run_dir(run_dir, resume=True)

    def test_startup_non_resume_distinguishes_run_artifacts_from_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dir_with_unrelated = Path(tmp) / "unrelated"
            dir_with_unrelated.mkdir()
            (dir_with_unrelated / "user_notes.txt").write_text("hello", encoding="utf-8")

            with self.assertRaises(ValueError) as cm:
                startup._ensure_fresh_run_dir(dir_with_unrelated)
            self.assertIn("already exists and is not empty", str(cm.exception))
            self.assertIn("Choose a fresh run directory.", str(cm.exception))
            self.assertNotIn("Resume mode is not implemented", str(cm.exception))
            self.assertNotIn("--resume-from", str(cm.exception))

    def test_startup_non_resume_detects_individual_praxist_artifacts(self) -> None:
        for rel in startup.PRAXIST_RUN_ARTIFACT_REL_PATHS:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "test_run"
                run_dir.mkdir()
                artifact_path = run_dir / rel
                if rel in {"findings", "frontier", "agendas", "memory", "gems", "gen_0"}:
                    artifact_path.mkdir(parents=True)
                else:
                    artifact_path.parent.mkdir(parents=True, exist_ok=True)
                    artifact_path.write_text("{}", encoding="utf-8")

                with self.assertRaises(ValueError) as cm:
                    startup._ensure_fresh_run_dir(run_dir)
                self.assertIn("already contains Praxist run artifacts", str(cm.exception))
                self.assertIn("Use --resume or --resume-from", str(cm.exception))
                self.assertNotIn("Resume mode is not implemented", str(cm.exception))

    def test_resume_mode_on_missing_or_damaged_artifacts_requires_manual_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty_dir = Path(tmp) / "empty"
            empty_dir.mkdir()
            with self.assertRaises(ValueError) as cm:
                startup._ensure_fresh_run_dir(empty_dir, resume=True)
            self.assertIn("missing Praxist startup artifacts", str(cm.exception))
            self.assertIn("Manual inspection is required", str(cm.exception))

        with tempfile.TemporaryDirectory() as tmp:
            bad_json_dir = Path(tmp) / "bad_json"
            bad_json_dir.mkdir()
            (bad_json_dir / "run.json").write_text("{not-valid-json", encoding="utf-8")
            (bad_json_dir / "startup_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError) as cm:
                resume_state._read_json_object_for_resume(bad_json_dir / "run.json")
            self.assertIn("not valid JSON", str(cm.exception))
            self.assertIn("Manual inspection is required", str(cm.exception))

        with tempfile.TemporaryDirectory() as tmp:
            non_obj_dir = Path(tmp) / "non_obj"
            non_obj_dir.mkdir()
            (non_obj_dir / "run.json").write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(ValueError) as cm:
                resume_state._read_json_object_for_resume(non_obj_dir / "run.json")
            self.assertIn("must be a JSON object", str(cm.exception))
            self.assertIn("Manual inspection is required", str(cm.exception))

    def test_resume_plan_safely_handles_already_completed_all_generations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "completed_run"
            run_dir.mkdir()
            (run_dir / "startup_config.json").write_text(
                json.dumps({"contract_version": 2, "marker_contract_start": 0}), encoding="utf-8"
            )
            # Generation 0 complete
            self._write_generation_results(run_dir, 0)
            write_boundary_marker(run_dir, gen_id=0, promoted_count=1, pi_status="succeeded")

            # Generation 1 complete
            self._write_generation_results(run_dir, 1)
            write_boundary_marker(run_dir, gen_id=1, promoted_count=1, pi_status="succeeded")

            plan = inspect_resume_plan(run_dir, max_generations=2, pi_enabled=True)
            self.assertEqual(plan.completed_generations, 2)
            self.assertEqual(plan.start_generation, 2)
            self.assertIsNone(plan.pending_boundary_generation)
            self.assertFalse(plan.has_pending_boundary)
            self.assertEqual(plan.warnings, ())

    def test_startup_resume_identity_accepts_matching_canonical_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_a"
            task_path = Path(tmp) / "task"
            run_dir.mkdir()
            task_path.mkdir()
            existing_config = self._startup_config(
                run_dir=run_dir,
                task_path=task_path,
                model_provider="model_provider:deepseek_alias",
                model="deepseek-v4-pro[1m]",
            )
            (run_dir / "startup_config.json").write_text(
                json.dumps(existing_config),
                encoding="utf-8",
            )
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": run_dir.name,
                        "task_ref": "task:demo",
                        "created_at": "2026-06-01T00:00:00Z",
                        "status": "interrupted",
                    }
                ),
                encoding="utf-8",
            )

            previous = validate_resume_startup_identity(run_dir, existing_config)

            self.assertEqual(previous["created_at"], "2026-06-01T00:00:00Z")
            events = [
                json.loads(line)
                for line in (run_dir / "resume_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(events[-1]["status"], "accepted")
            self.assertFalse(events[-1]["mismatches"])

    def test_startup_resume_identity_accepts_unchanged_relocated_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_relocated"
            old_task_path = Path(tmp) / "old_checkout" / "task"
            new_task_path = Path(tmp) / "new_checkout" / "task"
            run_dir.mkdir()
            new_task_path.mkdir(parents=True)
            existing_config = self._startup_config(
                run_dir=run_dir,
                task_path=old_task_path,
                model_provider="model_provider:deepseek_alias",
                model="deepseek-v4-pro[1m]",
            )
            candidate_config = self._startup_config(
                run_dir=run_dir,
                task_path=new_task_path,
                model_provider="model_provider:deepseek_alias",
                model="deepseek-v4-pro[1m]",
            )
            (run_dir / "startup_config.json").write_text(
                json.dumps(existing_config),
                encoding="utf-8",
            )
            (run_dir / "run.json").write_text(
                json.dumps({"run_id": run_dir.name, "task_ref": "task:demo"}),
                encoding="utf-8",
            )

            validate_resume_startup_identity(run_dir, candidate_config)

            event = json.loads(
                (run_dir / "resume_events.jsonl").read_text(encoding="utf-8").splitlines()[-1]
            )
            self.assertEqual(event["status"], "accepted")
            self.assertFalse(event["mismatches"])

    def test_startup_resume_identity_rejects_provider_or_model_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_b"
            task_path = Path(tmp) / "task"
            run_dir.mkdir()
            task_path.mkdir()
            existing_config = self._startup_config(
                run_dir=run_dir,
                task_path=task_path,
                model_provider="model_provider:openrouter",
                model="anthropic/claude-opus-4.1",
            )
            candidate_config = self._startup_config(
                run_dir=run_dir,
                task_path=task_path,
                model_provider="model_provider:deepseek_alias",
                model="deepseek-v4-pro[1m]",
            )
            (run_dir / "startup_config.json").write_text(
                json.dumps(existing_config),
                encoding="utf-8",
            )
            (run_dir / "run.json").write_text(
                json.dumps({"run_id": run_dir.name, "task_ref": "task:demo"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "model_provider"):
                validate_resume_startup_identity(run_dir, candidate_config)

            events = [
                json.loads(line)
                for line in (run_dir / "resume_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(events[-1]["status"], "rejected")
            self.assertIn(
                "model_provider",
                {item["field"] for item in events[-1]["mismatches"]},
            )

    def test_startup_resume_identity_rejects_task_manifest_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_c"
            task_path = Path(tmp) / "task"
            run_dir.mkdir()
            task_path.mkdir()
            existing_config = self._startup_config(
                run_dir=run_dir,
                task_path=task_path,
                model_provider="model_provider:deepseek_alias",
                model="deepseek-v4-pro[1m]",
            )
            candidate_config = self._startup_config(
                run_dir=run_dir,
                task_path=task_path,
                model_provider="model_provider:deepseek_alias",
                model="deepseek-v4-pro[1m]",
            )
            candidate_config["resume_identity"]["task_project_manifest_sha256"] = "newhash"
            (run_dir / "startup_config.json").write_text(
                json.dumps(existing_config),
                encoding="utf-8",
            )
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": run_dir.name,
                        "task_ref": "task:demo",
                        "task_project": {"manifest_sha256": "oldhash"},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "task_project_manifest_sha256"):
                validate_resume_startup_identity(run_dir, candidate_config)

            events = [
                json.loads(line)
                for line in (run_dir / "resume_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(events[-1]["status"], "rejected")
            self.assertIn(
                "task_project_manifest_sha256",
                {item["field"] for item in events[-1]["mismatches"]},
            )

    def test_startup_resume_identity_accepts_legacy_generated_report_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_report_manifest"
            task_path = Path(tmp) / "task"
            run_dir.mkdir()
            task_path.mkdir()
            (task_path / "task.yaml").write_text(
                "id: demo\npraxist_plugins:\n  task_ref: task:demo\n",
                encoding="utf-8",
            )
            report_path = task_path / "docs" / "praxist_reports" / "run_report.md"
            report_path.parent.mkdir(parents=True)
            report_path.write_text("generated report\n", encoding="utf-8")
            candidate_manifest = build_task_project_manifest(task_path)
            legacy_digest = hashlib.sha256()
            legacy_files = []
            for relative in ("task.yaml", "docs/praxist_reports/run_report.md"):
                content = (task_path / relative).read_bytes()
                legacy_files.append(
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "bytes": len(content),
                    }
                )
                legacy_digest.update(relative.encode("utf-8"))
                legacy_digest.update(b"\0")
                legacy_digest.update(content)
                legacy_digest.update(b"\0")
            existing_manifest = {
                **candidate_manifest,
                "sha256": legacy_digest.hexdigest(),
                "files": legacy_files,
            }
            existing_config = self._startup_config(
                run_dir=run_dir,
                task_path=task_path,
                model_provider="model_provider:deepseek_alias",
                model="deepseek-v4-pro[1m]",
            )
            candidate_config = json.loads(json.dumps(existing_config))
            existing_config["resume_identity"]["task_project_manifest_sha256"] = existing_manifest[
                "sha256"
            ]
            candidate_config["resume_identity"]["task_project_manifest_sha256"] = (
                candidate_manifest["sha256"]
            )
            (run_dir / "startup_config.json").write_text(
                json.dumps(existing_config), encoding="utf-8"
            )
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": run_dir.name,
                        "task_ref": "task:demo",
                        "task_project": {"manifest_sha256": existing_manifest["sha256"]},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "task_project_manifest.json").write_text(
                json.dumps(existing_manifest), encoding="utf-8"
            )

            validate_resume_startup_identity(
                run_dir,
                candidate_config,
                candidate_task_project_manifest=candidate_manifest,
            )

            relocated_task = Path(tmp) / "relocated_task"
            relocated_task.mkdir()
            (relocated_task / "task.yaml").write_bytes((task_path / "task.yaml").read_bytes())
            relocated_manifest = build_task_project_manifest(relocated_task)
            relocated_config = json.loads(json.dumps(candidate_config))
            relocated_config["canonical_args"]["task_path"] = str(relocated_task)
            validate_resume_startup_identity(
                run_dir,
                relocated_config,
                candidate_task_project_manifest=relocated_manifest,
            )

            (run_dir / "task_project_manifest.json").write_text(
                json.dumps({**existing_manifest, "sha256": "unbound-sidecar"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "task_project_manifest_sha256"):
                validate_resume_startup_identity(
                    run_dir,
                    candidate_config,
                    candidate_task_project_manifest=candidate_manifest,
                )
            (run_dir / "task_project_manifest.json").write_text(
                json.dumps(existing_manifest), encoding="utf-8"
            )

            incompatible_manifest = {
                **candidate_manifest,
                "files": [
                    {
                        **candidate_manifest["files"][0],
                        "sha256": "0" * 64,
                    }
                ],
            }
            with self.assertRaisesRegex(ValueError, "task_project_manifest_sha256"):
                validate_resume_startup_identity(
                    run_dir,
                    candidate_config,
                    candidate_task_project_manifest=incompatible_manifest,
                )

    def test_startup_resume_identity_rejects_legacy_run_without_task_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_d"
            task_path = Path(tmp) / "task"
            run_dir.mkdir()
            task_path.mkdir()
            legacy_config = self._startup_config(
                run_dir=run_dir,
                task_path=task_path,
                model_provider="model_provider:deepseek_alias",
                model="deepseek-v4-pro[1m]",
            )
            legacy_config.pop("resume_identity", None)
            candidate_config = self._startup_config(
                run_dir=run_dir,
                task_path=task_path,
                model_provider="model_provider:deepseek_alias",
                model="deepseek-v4-pro[1m]",
            )
            (run_dir / "startup_config.json").write_text(
                json.dumps(legacy_config),
                encoding="utf-8",
            )
            (run_dir / "run.json").write_text(
                json.dumps({"run_id": run_dir.name, "task_ref": "task:demo"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "task_project_manifest_sha256"):
                validate_resume_startup_identity(run_dir, candidate_config)

    def test_startup_resume_identity_reports_missing_and_secondary_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_e"
            task_path = Path(tmp) / "task"
            run_dir.mkdir()
            task_path.mkdir()
            candidate_config = self._startup_config(
                run_dir=run_dir,
                task_path=task_path,
                model_provider="model_provider:deepseek_alias",
                model="deepseek-v4-pro[1m]",
            )
            (run_dir / "startup_config.json").write_text(
                json.dumps({"resume_identity": {}}),
                encoding="utf-8",
            )
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            mismatches = resume_state._resume_canonical_mismatches(
                {"resume_identity": {}},
                candidate_config,
                {},
                run_dir,
            )
            self.assertEqual(mismatches[0]["field"], "canonical_args")

            existing_config = self._startup_config(
                run_dir=run_dir,
                task_path=task_path,
                model_provider="model_provider:deepseek_alias",
                model="deepseek-v4-pro[1m]",
            )
            candidate_without_args = dict(candidate_config)
            candidate_without_args.pop("canonical_args")
            mismatches = resume_state._resume_canonical_mismatches(
                existing_config,
                candidate_without_args,
                {},
                run_dir,
            )
            self.assertEqual(mismatches[0]["field"], "canonical_args")

            existing_config["resume_identity"] = {
                "task_project_manifest_sha256": "oldhash",
                "effective_task_descriptor_sha256": "olddesc",
                "local_mode": False,
            }
            candidate_config["resume_identity"] = {
                "task_project_manifest_sha256": "newhash",
                "effective_task_descriptor_sha256": "newdesc",
                "local_mode": True,
            }
            existing_run = {"run_id": "other_run", "task_ref": "task:other"}
            fields = {
                item["field"]
                for item in resume_state._resume_canonical_mismatches(
                    existing_config,
                    candidate_config,
                    existing_run,
                    run_dir,
                )
            }
            self.assertIn("run_id", fields)
            self.assertIn("run.task_ref", fields)
            self.assertIn("task_project_manifest_sha256", fields)
            self.assertIn("effective_task_descriptor_sha256", fields)
            self.assertIn("local_mode", fields)

            existing_config["resume_identity"] = {}
            candidate_config["resume_identity"] = {
                "task_project_manifest_sha256": "manifest-from-candidate",
                "effective_task_descriptor_sha256": "descriptor-from-candidate",
                "local_mode": True,
            }
            fields = {
                item["field"]
                for item in resume_state._resume_canonical_mismatches(
                    existing_config,
                    candidate_config,
                    {},
                    run_dir,
                )
            }
            self.assertIn("task_project_manifest_sha256", fields)
            self.assertIn("effective_task_descriptor_sha256", fields)
            self.assertIn("local_mode", fields)

    def test_resume_low_level_boundary_helpers_cover_invalid_and_edge_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.assertFalse(pid_is_alive(0))
            self.assertTrue(pid_is_alive(os.getpid()))
            self.assertIsNone(lock_pid("pid=abc\n"))

            self.assertEqual(resume_state._frontier_generations(run_dir), set())
            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir()
            manifest = frontier_dir / "frontier_manifest.json"
            manifest.write_text("{", encoding="utf-8")
            self.assertEqual(resume_state._frontier_generations(run_dir), set())
            manifest.write_text(json.dumps({"generations": []}), encoding="utf-8")
            self.assertEqual(resume_state._frontier_generations(run_dir), set())
            manifest.write_text(
                json.dumps({"generations": {"0": [], "skip": [], "2": []}}),
                encoding="utf-8",
            )
            self.assertEqual(resume_state._frontier_generations(run_dir), {0, 2})

            state_dir = run_dir / "gems"
            state_dir.mkdir()
            state_path = state_dir / "gems_state.json"
            state_path.write_text("[]", encoding="utf-8")
            self.assertFalse(resume_state._has_pending_gems_reset(run_dir, 0))
            state_path.write_text(
                json.dumps({"pending_reset": {"completed_gen_id": "bad"}}),
                encoding="utf-8",
            )
            self.assertFalse(resume_state._has_pending_gems_reset(run_dir, 0))
            state_path.write_text(
                json.dumps({"pending_reset": {"completed_gen_id": "0"}}),
                encoding="utf-8",
            )
            self.assertTrue(resume_state._has_pending_gems_reset(run_dir, 0))
            done, reason = resume_state._generation_boundary_done(
                run_dir,
                0,
                max_generations=2,
                pi_enabled=True,
                frontier_generations={0},
            )
            self.assertFalse(done)
            self.assertEqual(reason, "pending Gems reset transaction exists")

            state_path.write_text("[]", encoding="utf-8")
            self.assertFalse(resume_state._gems_reset_boundary_done(run_dir, 0))
            state_path.write_text(
                json.dumps({"pending_reset": {"completed_gen_id": 0}, "reset_events": []}),
                encoding="utf-8",
            )
            self.assertFalse(resume_state._gems_reset_boundary_done(run_dir, 0))
            state_path.write_text(json.dumps({"reset_events": "bad"}), encoding="utf-8")
            self.assertFalse(resume_state._gems_reset_boundary_done(run_dir, 0))
            state_path.write_text(
                json.dumps({"reset_events": ["bad", {"completed_gen_id": "bad"}]}),
                encoding="utf-8",
            )
            self.assertFalse(resume_state._gems_reset_boundary_done(run_dir, 0))
            state_path.write_text(
                json.dumps(
                    {
                        "reset_count": "bad",
                        "reset_events": [
                            {
                                "completed_gen_id": 0,
                                "next_absolute_generation": 1,
                                "committed": True,
                                "reset_count": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(resume_state._gems_reset_boundary_done(run_dir, 0))
            state_path.write_text(
                json.dumps(
                    {
                        "cycle_start_generation": 4,
                        "reset_events": [
                            {
                                "completed_gen_id": 0,
                                "next_absolute_generation": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(resume_state._gems_reset_boundary_done(run_dir, 0))

            state = {"reset_count": 1, "cycle_start_generation": 1}
            manifest.write_text("[]", encoding="utf-8")
            self.assertFalse(resume_state._gems_frontier_manifest_committed(run_dir, state))
            manifest.write_text(json.dumps({"gems": []}), encoding="utf-8")
            self.assertFalse(resume_state._gems_frontier_manifest_committed(run_dir, state))
            manifest.write_text(
                json.dumps(
                    {
                        "gems": {"reset_count": 2, "cycle_start_generation": 1},
                        "generations": {},
                        "lane_frontiers": {},
                        "cumulative_top": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(resume_state._gems_frontier_manifest_committed(run_dir, state))
            manifest.write_text(
                json.dumps(
                    {
                        "gems": {"reset_count": 1, "cycle_start_generation": 2},
                        "generations": {},
                        "lane_frontiers": {},
                        "cumulative_top": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(resume_state._gems_frontier_manifest_committed(run_dir, state))
            manifest.write_text(
                json.dumps(
                    {
                        "gems": {"reset_count": 1, "cycle_start_generation": 1},
                        "generations": {"0": []},
                        "lane_frontiers": {},
                        "cumulative_top": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(resume_state._gems_frontier_manifest_committed(run_dir, state))
            manifest.write_text(
                json.dumps(
                    {
                        "gems": {"reset_count": 1, "cycle_start_generation": 1},
                        "generations": {},
                        "lane_frontiers": {"lane": []},
                        "cumulative_top": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(resume_state._gems_frontier_manifest_committed(run_dir, state))

    def test_existing_marker_disables_legacy_boundary_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)
            write_boundary_marker(
                run_dir,
                gen_id=0,
                promoted_count=1,
                pi_status="succeeded",
            )
            self._write_generation_results(run_dir, 1)
            self._write_frontier_manifest(run_dir, {"1": [{"variant_name": "v1"}]})

            repairs = repair_inferred_gems_boundary_markers(
                run_dir,
                max_generations=2,
                pi_enabled=False,
            )

            self.assertEqual(repairs, [])
            done, reason = resume_state._generation_boundary_done(
                run_dir,
                1,
                max_generations=2,
                pi_enabled=False,
                frontier_generations={1},
            )
            self.assertFalse(done)
            self.assertEqual(reason, "required generation boundary marker is missing")

    def test_inferred_repair_reclassifies_post_cutoff_result_before_marker(self) -> None:
        from types import SimpleNamespace

        from praxist.plugins.workflow_stages.research_loop.backend import (
            generation_loop,
            generation_resume,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync import (
            FindingsSync,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)
            self._write_frontier_manifest(run_dir, {"0": [{"variant_name": "candidate"}]})
            cutoff_epoch = 1_800_000_000
            cutoff = datetime.fromtimestamp(cutoff_epoch, tz=UTC)
            resume_state.write_boundary_evidence_checkpoint(
                run_dir,
                gen_id=0,
                cutoff=cutoff,
                evidence_source_snapshot={},
            )
            result_dir = run_dir / "results" / "candidate"
            result_dir.mkdir(parents=True)
            summary = result_dir / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "variant_name": "candidate",
                        "generation_id": 0,
                        "current_aggregate": {"score": 0.9, "scored_complete": True},
                    }
                ),
                encoding="utf-8",
            )
            os.utime(summary, (cutoff_epoch + 60, cutoff_epoch + 60))
            sync = FindingsSync(
                findings_dir=run_dir / "shared_findings",
                run_dir=run_dir,
                local_mode=False,
                materialize_result_artifacts=True,
                result_scoring_metric_keys=["score"],
            )
            loop = object.__new__(generation_loop.GenerationLoop)
            loop.run_dir = run_dir
            loop.findings_dir = run_dir / "shared_findings"
            loop.local_mode = False
            loop.task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(primary_metric="score"),
                gems=None,
            )
            loop._findings_sync = sync
            loop._boundary_evidence_cutoff = None

            repairs = generation_resume.repair_inferred_boundaries_for_resume(
                loop,
                max_generations=1,
                pi_enabled=False,
            )

            self.assertEqual([repair["generation_id"] for repair in repairs], [0])
            finding_paths = list((run_dir / "shared_findings").glob("*.json"))
            self.assertEqual(len(finding_paths), 1)
            finding = json.loads(finding_paths[0].read_text(encoding="utf-8"))
            self.assertTrue(finding["metrics"]["late_after_generation_boundary"])
            self.assertFalse(finding["metrics"]["promotion_eligible"])
            self.assertIsNone(loop._boundary_evidence_cutoff)
            self.assertIsNone(sync._boundary_evidence_cutoff)
            self.assertIsNone(resume_state.read_boundary_evidence_checkpoint(run_dir, 0))

    def test_resume_discards_boundary_cutoff_for_full_generation_rerun(self) -> None:
        from types import SimpleNamespace

        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
            generation_resume,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "generation_results.json").write_text("[]", encoding="utf-8")
            resume_state.write_boundary_evidence_checkpoint(
                run_dir,
                gen_id=0,
                cutoff=datetime.now(UTC),
                evidence_source_snapshot={},
            )
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "abandoned-late",
                        "generation_id": 0,
                        "metrics": {
                            "score": 1.0,
                            "promotion_eligible": True,
                            "generation_boundary_pending_commit": True,
                            "late_after_generation_boundary": True,
                            "late_observed_generation_id": 0,
                        },
                    }
                )
            loop = SimpleNamespace(
                run_dir=run_dir,
                local_mode=True,
                _findings_sync=None,
                _boundary_evidence_cutoff=None,
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}):
                generation_resume.prime_resume_boundary_evidence_cutoff(
                    loop,
                    max_generations=1,
                )
                [finding] = local_store.get_all_findings()

            self.assertIsNone(loop._boundary_evidence_cutoff)
            self.assertIsNone(resume_state.read_boundary_evidence_checkpoint(run_dir, 0))
            self.assertEqual(
                finding["metrics"],
                {"score": 1.0, "promotion_eligible": True},
            )
            self.assertIsNone(
                findings_collection._preserved_late_boundary_info(
                    {
                        "id": "abandoned-late",
                        "generation_id": 0,
                        "metrics": {
                            "generation_boundary_pending_commit": True,
                            "late_after_generation_boundary": True,
                        },
                    },
                    run_dir=run_dir,
                    source_gen_id=0,
                )
            )

            resume_state.write_boundary_evidence_checkpoint(
                run_dir,
                gen_id=0,
                cutoff=datetime.now(UTC),
                evidence_source_snapshot={},
            )
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}):
                repairs = generation_resume.repair_inferred_boundaries_for_resume(
                    loop,
                    max_generations=1,
                    pi_enabled=False,
                )

            self.assertEqual(repairs, [])
            self.assertIsNone(loop._boundary_evidence_cutoff)
            self.assertIsNone(resume_state.read_boundary_evidence_checkpoint(run_dir, 0))

    def test_resume_primes_complete_boundary_checkpoint_before_sidecars(self) -> None:
        from types import SimpleNamespace

        from praxist.plugins.workflow_stages.research_loop.backend import generation_resume

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 0)
            cutoff = datetime.now(UTC)
            source_snapshot = {"results/candidate/summary.json": "content:stable"}
            resume_state.write_boundary_evidence_checkpoint(
                run_dir,
                gen_id=0,
                cutoff=cutoff,
                evidence_source_snapshot=source_snapshot,
            )
            loop = SimpleNamespace(
                run_dir=run_dir,
                local_mode=False,
                _findings_sync=None,
                _boundary_evidence_cutoff=None,
            )

            generation_resume.prime_resume_boundary_evidence_cutoff(
                loop,
                max_generations=1,
            )

        self.assertEqual(loop._boundary_evidence_cutoff, (0, cutoff, source_snapshot))

    def test_abandoned_checkpoint_cleanup_survives_local_store_failure(self) -> None:
        from types import SimpleNamespace

        from praxist.plugins.workflow_stages.research_loop.backend import generation_resume
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "generation_results.json").write_text("[]", encoding="utf-8")
            resume_state.write_boundary_evidence_checkpoint(
                run_dir,
                gen_id=0,
                cutoff=datetime.now(UTC),
                evidence_source_snapshot={},
            )
            loop = SimpleNamespace(
                run_dir=run_dir,
                local_mode=True,
                _findings_sync=None,
                _boundary_evidence_cutoff=None,
            )

            with patch.object(
                local_store,
                "clear_pending_boundary_validation",
                side_effect=OSError("store unavailable"),
            ):
                generation_resume.prime_resume_boundary_evidence_cutoff(
                    loop,
                    max_generations=1,
                )

            self.assertIsNone(resume_state.read_boundary_evidence_checkpoint(run_dir, 0))
            self.assertIsNone(loop._boundary_evidence_cutoff)

    def test_inferred_boundary_repair_falls_back_to_sidecar_sync(self) -> None:
        from types import SimpleNamespace

        from praxist.plugins.workflow_stages.research_loop.backend import generation_resume

        sync_calls: list[bool] = []
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_generation_results(run_dir, 1)
            cutoff = datetime.now(UTC)
            source_snapshot = {"results/candidate/summary.json": "content:stable"}
            resume_state.write_boundary_evidence_checkpoint(
                run_dir,
                gen_id=1,
                cutoff=cutoff,
                evidence_source_snapshot=source_snapshot,
            )
            sync = SimpleNamespace(sync_once=lambda: sync_calls.append(True))
            loop = SimpleNamespace(
                run_dir=run_dir,
                local_mode=False,
                _findings_sync=sync,
                _boundary_evidence_cutoff=None,
            )

            with patch.object(
                generation_resume,
                "repair_inferred_gems_boundary_markers",
                return_value=[{"generation_id": 1}],
            ):
                repairs = generation_resume.repair_inferred_boundaries_for_resume(
                    loop,
                    max_generations=2,
                    pi_enabled=False,
                )

        self.assertEqual(repairs, [{"generation_id": 1}])
        self.assertEqual(sync_calls, [True])
        self.assertIsNone(loop._boundary_evidence_cutoff)

    def test_legacy_prefix_remains_complete_after_first_resumed_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for gen_id in range(3):
                self._write_generation_results(run_dir, gen_id)
            self._write_frontier_manifest(
                run_dir,
                {
                    "0": [{"variant_name": "v0"}],
                    "1": [{"variant_name": "v1"}],
                    "2": [{"variant_name": "v2"}],
                },
            )
            agendas = run_dir / "agendas"
            agendas.mkdir()
            for next_gen in (1, 2):
                (agendas / f"research_agenda_gen{next_gen}.yaml").write_text(
                    f"generation: {next_gen}\npeer_contracts: {{}}\n",
                    encoding="utf-8",
                )
            write_boundary_marker(
                run_dir,
                gen_id=2,
                promoted_count=1,
                pi_status="succeeded",
            )

            plan = inspect_resume_plan(run_dir, max_generations=4, pi_enabled=True)

            self.assertEqual(plan.completed_generations, 3)
            self.assertIsNone(plan.pending_boundary_generation)
            self.assertEqual(
                resume_state.reported_completed_generations({"generations_completed": 0}, run_dir),
                3,
            )
            self.assertEqual(resume_state.canonical_completed_generation_count(run_dir), 0)

            repairs = repair_inferred_gems_boundary_markers(
                run_dir,
                max_generations=4,
                pi_enabled=True,
            )

            self.assertEqual([repair["generation_id"] for repair in repairs], [0, 1])
            self.assertEqual(resume_state.canonical_completed_generation_count(run_dir), 3)
            marker = json.loads(
                (run_dir / "gen_0" / "generation_boundary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["pi_status"], "legacy_inferred_boundary_repaired")

    def test_lock_pid_parser(self) -> None:
        self.assertEqual(lock_pid("pid=123\nstarted=1\n"), 123)
        self.assertIsNone(lock_pid("pid=not-an-int\n"))
        self.assertIsNone(lock_pid("started=1\n"))

    @staticmethod
    def _write_generation_results(run_dir: Path, gen_id: int) -> None:
        gen_dir = run_dir / f"gen_{gen_id}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "generation_results.json").write_text(
            json.dumps([{"peer_id": f"gen{gen_id}_peer0", "success": True}]),
            encoding="utf-8",
        )

    @staticmethod
    def _write_frontier_manifest(run_dir: Path, generations: dict[str, list[dict]]) -> None:
        frontier_dir = run_dir / "frontier"
        frontier_dir.mkdir(parents=True, exist_ok=True)
        (frontier_dir / "frontier_manifest.json").write_text(
            json.dumps({"generations": generations, "cumulative_top": []}),
            encoding="utf-8",
        )

    @staticmethod
    def _startup_config(
        *,
        run_dir: Path,
        task_path: Path,
        model_provider: str,
        model: str,
    ) -> dict:
        return {
            "schema_version": "praxist.startup.v1",
            "canonical_args": {
                "task": "task:demo",
                "task_path": str(task_path),
                "runtime": "agent_runtime:claude_sdk",
                "model_provider": model_provider,
                "budget_policy": "budget_policy:default_basic",
                "model": model,
                "frontier_strategy": "pareto",
                "run_dir": str(run_dir),
            },
            "resume": {"enabled": True, "policy": "completed_generation"},
            "resume_identity": {
                "task_project_manifest_sha256": "oldhash",
                "effective_task_descriptor_sha256": "descriptorhash",
                "local_mode": False,
            },
        }


if __name__ == "__main__":
    unittest.main()
