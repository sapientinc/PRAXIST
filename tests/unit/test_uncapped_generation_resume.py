"""Resume and checkpoint recovery without a finite generation ceiling."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from praxist.plugins.workflow_stages.research_loop.backend import (
    generation_resume,
    resume_state,
    scoreless,
)


class UncappedGenerationResumeTest(unittest.TestCase):
    def _generation(self, root: Path, generation: int, *, committed: bool = False) -> Path:
        directory = root / f"gen_{generation}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "generation_results.json").write_text(
            json.dumps([{"peer_id": f"gen{generation}_peer0", "status": "completed"}]),
            encoding="utf-8",
        )
        if committed:
            scoreless.write_scoreless_evidence_manifest(
                root,
                gen_id=generation,
                findings=[{"id": f"finding-{generation}", "content": "Retained evidence"}],
                evidence_cutoff_at="2026-08-31T00:00:00+00:00",
                evidence_source_snapshot={},
            )
            resume_state.write_boundary_marker(
                root, gen_id=generation, promoted_count=0, pi_status="disabled"
            )
        return directory

    def _checkpoint(self, root: Path, generation: int) -> tuple[datetime, dict[str, str]]:
        cutoff = datetime(2026, 8, 31, 0, generation % 60, tzinfo=UTC)
        snapshot = {f"shared_findings/finding-{generation}.json": "content:frozen"}
        resume_state.write_boundary_evidence_checkpoint(
            root, gen_id=generation, cutoff=cutoff, evidence_source_snapshot=snapshot
        )
        return cutoff, snapshot

    def _loop(self, root: Path) -> SimpleNamespace:
        return SimpleNamespace(
            run_dir=root,
            local_mode=False,
            _findings_sync=None,
            _boundary_evidence_cutoff=None,
        )

    def test_uncapped_resume_continues_after_nine_committed_generations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for generation in range(9):
                self._generation(root, generation, committed=True)
            marker = root / "gen_8" / resume_state.BOUNDARY_MARKER_FILENAME
            before = marker.read_bytes()

            plan = resume_state.inspect_resume_plan(root, max_generations=None, pi_enabled=True)
            bounded = resume_state.inspect_resume_plan(root, max_generations=8, pi_enabled=True)

            self.assertEqual(plan.completed_generations, 9)
            self.assertEqual(plan.start_generation, 9)
            self.assertFalse(plan.has_pending_boundary)
            self.assertEqual(bounded.completed_generations, 8)
            self.assertEqual(marker.read_bytes(), before)
            self.assertIn("scoreless_evidence_sha256", json.loads(before))

    def test_uncapped_resume_preserves_pending_generation_nine_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for generation in range(9):
                self._generation(root, generation, committed=True)
            self._generation(root, 9)
            checkpoint = self._checkpoint(root, 9)

            plan = generation_resume.prepare_resume_for_sidecars(
                root, max_generations=None, pi_enabled=True, policy="completed_generation"
            )
            loop = self._loop(root)
            generation_resume.prime_resume_boundary_evidence_cutoff(loop, max_generations=None)

            self.assertEqual(plan.completed_generations, 9)
            self.assertEqual(plan.pending_boundary_generation, 9)
            self.assertIn("boundary marker is missing", plan.warnings[0])
            self.assertEqual(resume_state.read_boundary_evidence_checkpoint(root, 9), checkpoint)
            self.assertEqual(loop._boundary_evidence_cutoff, (9, *checkpoint))

    def test_uncapped_resume_stops_at_gap_but_primes_sparse_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for generation in range(5):
                self._generation(root, generation, committed=True)
            self._generation(root, 9)
            self._checkpoint(root, 9)
            self._generation(root, 12)
            newest = self._checkpoint(root, 12)
            loop = self._loop(root)

            plan = resume_state.inspect_resume_plan(root, max_generations=None, pi_enabled=False)
            generation_resume.prime_resume_boundary_evidence_cutoff(loop, max_generations=None)

            self.assertEqual(plan.start_generation, 5)
            self.assertFalse(plan.has_pending_boundary)
            self.assertEqual(loop._boundary_evidence_cutoff, (12, *newest))

    def test_uncapped_preparation_clears_rerun_signals_beyond_bounded_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for generation in range(9):
                self._generation(root, generation, committed=True)
            rerun = root / "gen_9"
            rerun.mkdir()
            signal = rerun / "STOP_SIGNAL"
            signal.write_text("stale", encoding="utf-8")

            generation_resume.prepare_resume_for_sidecars(
                root, max_generations=8, pi_enabled=False, policy="completed_generation"
            )
            self.assertTrue(signal.exists())
            plan = generation_resume.prepare_resume_for_sidecars(
                root, max_generations=None, pi_enabled=False, policy="completed_generation"
            )

            self.assertEqual(plan.start_generation, 9)
            self.assertFalse(signal.exists())

    def test_uncapped_legacy_boundary_always_needs_successor_agenda_when_pi_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._generation(root, 0)
            frontier = root / "frontier" / "frontier_manifest.json"
            frontier.parent.mkdir()
            frontier.write_text(json.dumps({"generations": {"0": []}}), encoding="utf-8")

            bounded = resume_state.inspect_resume_plan(root, max_generations=1, pi_enabled=True)
            uncapped = resume_state.inspect_resume_plan(root, max_generations=None, pi_enabled=True)

            self.assertEqual(bounded.completed_generations, 1)
            self.assertEqual(uncapped.pending_boundary_generation, 0)
            self.assertIn("next-generation agenda is missing", uncapped.warnings[0])
            agenda = root / "agendas" / "research_agenda_gen1.yaml"
            agenda.parent.mkdir()
            agenda.write_text("generation: 1\npeer_contracts: {}\n", encoding="utf-8")
            completed = resume_state.inspect_resume_plan(
                root, max_generations=None, pi_enabled=True
            )
            self.assertEqual(completed.completed_generations, 1)

    def test_uncapped_inferred_repair_commits_generation_nine_and_retires_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for generation in range(10):
                self._generation(root, generation)
            frontier = root / "frontier" / "frontier_manifest.json"
            frontier.parent.mkdir()
            frontier.write_text(
                json.dumps({"generations": {str(generation): [] for generation in range(10)}}),
                encoding="utf-8",
            )
            cutoff, snapshot = self._checkpoint(root, 9)
            collected: list[tuple] = []
            loop = self._loop(root)
            loop._collect_findings_for_boundary = lambda generation, **kwargs: collected.append(
                (generation, kwargs)
            )

            repairs = generation_resume.repair_inferred_boundaries_for_resume(
                loop, max_generations=None, pi_enabled=False
            )

            self.assertEqual([item["generation_id"] for item in repairs], list(range(10)))
            self.assertEqual(
                collected,
                [(9, {"evidence_cutoff": cutoff, "evidence_source_snapshot": snapshot})],
            )
            self.assertIsNone(loop._boundary_evidence_cutoff)
            self.assertIsNone(resume_state.read_boundary_evidence_checkpoint(root, 9))
            marker = json.loads(
                (root / "gen_9" / resume_state.BOUNDARY_MARKER_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(marker["evidence_cutoff_at"], cutoff.isoformat())
            self.assertEqual(marker["evidence_source_snapshot_at_cutoff"], snapshot)
            plan = resume_state.inspect_resume_plan(root, max_generations=None, pi_enabled=False)
            self.assertEqual(plan.start_generation, 10)

    def test_uncapped_sparse_repair_collects_evidence_without_skipping_prefix_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._generation(root, 9)
            checkpoint = self._checkpoint(root, 9)
            synced: list[bool] = []
            loop = self._loop(root)
            loop._findings_sync = SimpleNamespace(sync_once=lambda: synced.append(True))

            generation_resume.prime_resume_boundary_evidence_cutoff(loop, max_generations=9)
            bounded_repairs = generation_resume.repair_inferred_boundaries_for_resume(
                loop, max_generations=9, pi_enabled=False
            )
            self.assertEqual(bounded_repairs, [])
            self.assertEqual(synced, [])
            self.assertIsNone(loop._boundary_evidence_cutoff)

            repairs = generation_resume.repair_inferred_boundaries_for_resume(
                loop, max_generations=None, pi_enabled=False
            )

            self.assertEqual(repairs, [])
            self.assertEqual(synced, [True])
            self.assertEqual(loop._boundary_evidence_cutoff, (9, *checkpoint))
            self.assertIsNotNone(resume_state.read_boundary_evidence_checkpoint(root, 9))

    def test_uncapped_scan_discards_abandoned_checkpoint_and_ignores_non_generations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = self._generation(root, 9)
            (directory / "generation_results.json").write_text("[]", encoding="utf-8")
            self._checkpoint(root, 9)
            for name in ("gen_bad", "gen_-1", "gen_01"):
                (root / name).mkdir()
            (root / "gen_14").write_text("not a directory", encoding="utf-8")
            loop = self._loop(root)

            generation_resume.prime_resume_boundary_evidence_cutoff(loop, max_generations=None)

            self.assertIsNone(resume_state.read_boundary_evidence_checkpoint(root, 9))
            self.assertIsNone(loop._boundary_evidence_cutoff)


if __name__ == "__main__":
    unittest.main()
