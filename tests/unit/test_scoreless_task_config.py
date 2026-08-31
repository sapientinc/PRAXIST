from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from praxist.task_spec import load_task_spec


class ScorelessTaskConfigurationTest(unittest.TestCase):
    def load(self, root: Path, **updates):
        raw = {"task_id": "evidence", "research_loop": {"mode": "scoreless"}}
        raw.update(updates)
        path = root / "task.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return load_task_spec(str(path))

    def load_lifecycle(self, root: Path, *, lifecycle=None, **updates):
        (root / "coordinator.py").write_text(
            "raise AssertionError('task code must not execute during validation')\n"
            "async def handle(context): return {}\n"
        )
        return self.load(
            root,
            research_loop={
                "mode": "scoreless",
                "lifecycle": {"entrypoint": "coordinator.py:handle", **(lifecycle or {})},
            },
            **updates,
        )

    def test_scoreless_does_not_invent_a_primary_metric_or_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self.load(Path(tmp))
            self.assertEqual(spec.research_loop["mode"], "scoreless")
            self.assertEqual(spec.evaluation.primary_metric, "")
            self.assertEqual(spec.baselines, [])
            self.assertEqual(spec.toolchain.eval_entrypoint, "")
            self.assertEqual(spec.generation_policy.promote_top_k, 0)

    def test_scoreless_rejects_scored_selection_instead_of_ignoring_it(self):
        cases = (
            {"evaluation": {"primary_metric": "quality"}},
            {"evaluation": {"anchor_metrics": [{"name": "quality", "direction": "maximize"}]}},
            {"baselines": [{"name": "dummy", "metric_value": 0}]},
            {"gems": {"enabled": True}},
            {"synthesis_trigger": {"mature_quorum_fraction": 0.5}},
        )
        with tempfile.TemporaryDirectory() as tmp:
            for updates in cases:
                with self.subTest(updates=updates), self.assertRaises(ValueError):
                    self.load(Path(tmp), **updates)

    def test_lifecycle_entrypoint_and_time_reserve_are_validated_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "coordinator.py").write_text("async def handle(context): return {}\n")
            spec = self.load(
                root,
                run_lifecycle={"max_wall_clock_hours": 8},
                research_loop={
                    "mode": "scoreless",
                    "lifecycle": {
                        "entrypoint": "coordinator.py:handle",
                        "initial_seconds": 1800,
                        "finalization_seconds": 1800,
                        "config": {"binding_id": "example"},
                    },
                },
            )
            self.assertEqual(spec.research_loop["lifecycle"]["initial_seconds"], 1800)
            for entrypoint in (
                "../outside.py:handle",
                "/tmp/outside.py:handle",
                "missing.py:handle",
            ):
                with self.subTest(entrypoint=entrypoint), self.assertRaises(ValueError):
                    self.load(
                        root,
                        run_lifecycle={"max_wall_clock_hours": 8},
                        research_loop={
                            "mode": "scoreless",
                            "lifecycle": {
                                "entrypoint": entrypoint,
                                "initial_seconds": 1800,
                                "finalization_seconds": 1800,
                            },
                        },
                    )

    def test_uncapped_lifecycle_defaults_to_no_phase_time_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            for updates in (
                {},
                {"run_lifecycle": {}},
                {"run_lifecycle": {"max_wall_clock_hours": None}},
            ):
                with self.subTest(updates=updates):
                    spec = self.load_lifecycle(
                        Path(tmp),
                        lifecycle={"after_generation": True, "config": {"label": "research"}},
                        **updates,
                    )
                    self.assertIsNone(spec.run_lifecycle.max_wall_clock_hours)
                    self.assertIsNone(spec.research_loop["lifecycle"]["initial_seconds"])
                    self.assertIsNone(spec.research_loop["lifecycle"]["finalization_seconds"])
                    self.assertTrue(spec.research_loop["lifecycle"]["after_generation"])
                    self.assertEqual(
                        spec.research_loop["lifecycle"]["config"], {"label": "research"}
                    )

    def test_uncapped_lifecycle_accepts_independent_explicit_phase_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            for caps, expected in (
                ({"initial_seconds": 12.5}, (12.5, None)),
                ({"finalization_seconds": 60}, (None, 60.0)),
                ({"initial_seconds": 30, "finalization_seconds": 45}, (30.0, 45.0)),
                ({"initial_seconds": None, "finalization_seconds": None}, (None, None)),
            ):
                with self.subTest(caps=caps):
                    spec = self.load_lifecycle(Path(tmp), lifecycle=caps)
                    normalized = spec.research_loop["lifecycle"]
                    self.assertEqual(
                        (normalized["initial_seconds"], normalized["finalization_seconds"]),
                        expected,
                    )

    def test_bounded_lifecycle_keeps_default_phase_reserves(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self.load_lifecycle(Path(tmp), run_lifecycle={"max_wall_clock_hours": 8})
            normalized = spec.research_loop["lifecycle"]
            self.assertEqual(normalized["initial_seconds"], 1800)
            self.assertEqual(normalized["finalization_seconds"], 1800)
            with self.assertRaisesRegex(ValueError, "leave time for research"):
                self.load_lifecycle(Path(tmp), run_lifecycle={"max_wall_clock_hours": 1})

    def test_bounded_lifecycle_accepts_null_phase_caps_without_reserving_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            for caps, expected in (
                ({"initial_seconds": None}, (None, 1800.0)),
                ({"finalization_seconds": None}, (1800.0, None)),
                ({"initial_seconds": None, "finalization_seconds": None}, (None, None)),
            ):
                with self.subTest(caps=caps):
                    spec = self.load_lifecycle(
                        Path(tmp), lifecycle=caps, run_lifecycle={"max_wall_clock_hours": 1}
                    )
                    normalized = spec.research_loop["lifecycle"]
                    self.assertEqual(
                        (normalized["initial_seconds"], normalized["finalization_seconds"]),
                        expected,
                    )
            with self.assertRaisesRegex(ValueError, "leave time for research"):
                self.load_lifecycle(
                    Path(tmp),
                    lifecycle={"initial_seconds": None, "finalization_seconds": 3600},
                    run_lifecycle={"max_wall_clock_hours": 1},
                )

    def test_phase_limits_remain_finite_positive_numbers_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            for total in (None, 8):
                for field in ("initial_seconds", "finalization_seconds"):
                    for value in (True, False, 0, -1, float("nan"), float("inf"), "invalid", []):
                        with (
                            self.subTest(total=total, field=field, value=value),
                            self.assertRaisesRegex(ValueError, f"lifecycle.{field}"),
                        ):
                            self.load_lifecycle(
                                Path(tmp),
                                lifecycle={field: value},
                                run_lifecycle={"max_wall_clock_hours": total},
                            )

    def test_invalid_explicit_total_budget_is_not_treated_as_uncapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in (True, False, 0, -1, float("nan"), float("inf"), "invalid", []):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    self.load_lifecycle(Path(tmp), run_lifecycle={"max_wall_clock_hours": value})

    def test_unknown_mode_fails_and_metric_default_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                self.load(root, research_loop={"mode": "typo"})
            spec = self.load(root, research_loop={})
            self.assertEqual(spec.evaluation.primary_metric, "metric_value")
            self.assertEqual(spec.generation_policy.promote_top_k, 2)

    def test_explicit_nonmapping_configuration_is_not_silently_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in (False, [], 0, "", None):
                with (
                    self.subTest(field="research_loop", value=value),
                    self.assertRaises(ValueError),
                ):
                    self.load(Path(tmp), research_loop=value)
                with self.subTest(field="lifecycle", value=value), self.assertRaises(ValueError):
                    self.load(Path(tmp), research_loop={"mode": "scoreless", "lifecycle": value})


if __name__ == "__main__":
    unittest.main()
