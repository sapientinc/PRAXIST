"""Scoreless task limits are explicit and may remain genuinely absent."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

from praxist.core.execution_policy import (
    apply_task_execution_policy,
    task_execution_deadline_scope,
    task_execution_policy_scope,
    validate_task_execution_policy,
)
from praxist.task_spec import load_task_spec
from tests.unit.test_codex_sdk_adapter import _request


class ScorelessOptionalLimitTests(unittest.TestCase):
    def load(self, mode: str, **updates):
        raw = {"task_id": "evidence", "research_loop": {"mode": mode}, **updates}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            return load_task_spec(path)

    def test_scoreless_missing_and_null_limits_are_absent(self):
        for updates in (
            {},
            {
                "generation_policy": {"max_generations": None, "per_generation_hours": None},
                "pi_agent": {"max_runtime_minutes": None},
                "multi_pi": {
                    "pi_max_runtime_minutes": None,
                    "chair_max_runtime_minutes": None,
                    "round2_max_runtime_minutes": None,
                },
                "synthesis_trigger": {"max_interval_minutes": None, "min_interval_minutes": None},
            },
        ):
            with self.subTest(updates=updates):
                spec = self.load("scoreless", **updates)
                self.assertIsNone(spec.generation_policy.max_generations)
                self.assertIsNone(spec.generation_policy.per_generation_hours)
                self.assertIsNone(spec.pi_agent.max_runtime_minutes)
                self.assertIsNone(spec.multi_pi.pi_max_runtime_minutes)
                self.assertIsNone(spec.multi_pi.chair_max_runtime_minutes)
                self.assertIsNone(spec.multi_pi.round2_max_runtime_minutes)
                self.assertIsNone(spec.synthesis_trigger.max_interval_minutes)
                self.assertEqual(spec.synthesis_trigger.min_interval_minutes, 0)
                self.assertEqual(spec.generation_policy.cohort_size, 5)

    def test_explicit_scoreless_limits_and_metric_defaults_are_retained(self):
        bounded = self.load(
            "scoreless",
            generation_policy={"max_generations": 3, "per_generation_hours": 2},
            pi_agent={"max_runtime_minutes": 4},
            multi_pi={
                "pi_max_runtime_minutes": 5,
                "chair_max_runtime_minutes": 6,
                "round2_max_runtime_minutes": 7,
            },
            synthesis_trigger={"max_interval_minutes": 30, "min_interval_minutes": 5},
        )
        self.assertEqual(bounded.generation_policy.max_generations, 3)
        self.assertEqual(bounded.generation_policy.per_generation_hours, 2)
        self.assertEqual(bounded.pi_agent.max_runtime_minutes, 4)
        self.assertEqual(bounded.multi_pi.pi_max_runtime_minutes, 5)
        self.assertEqual(bounded.multi_pi.chair_max_runtime_minutes, 6)
        self.assertEqual(bounded.multi_pi.round2_max_runtime_minutes, 7)
        self.assertEqual(bounded.synthesis_trigger.max_interval_minutes, 30)
        self.assertEqual(bounded.synthesis_trigger.min_interval_minutes, 5)
        metric = self.load("metric")
        self.assertEqual(metric.generation_policy.max_generations, 8)
        self.assertEqual(metric.generation_policy.per_generation_hours, 5)
        self.assertEqual(metric.pi_agent.max_runtime_minutes, 15)
        self.assertEqual(metric.multi_pi.pi_max_runtime_minutes, 12)
        self.assertEqual(metric.multi_pi.chair_max_runtime_minutes, 8)
        self.assertEqual(metric.multi_pi.round2_max_runtime_minutes, 6)
        self.assertEqual(metric.synthesis_trigger.max_interval_minutes, 240)
        self.assertEqual(metric.synthesis_trigger.min_interval_minutes, 120)

    def test_null_tool_timeout_does_not_invent_a_native_limit(self):
        policy = validate_task_execution_policy({"tool_execution_timeout_seconds": None})
        request = replace(_request("/tmp"), timeout_seconds=None)
        with task_execution_policy_scope(policy):
            bound = apply_task_execution_policy(request)
        self.assertIsNone(bound.timeout_seconds)
        self.assertIsNone(bound.to_dict()["timeout_seconds"])
        self.assertNotIn("tool_execution_timeout_seconds", bound.runtime_options)

    def test_explicit_limits_reject_zero_negative_nonfinite_and_fractional_counts(self):
        for field, values in (
            ("max_generations", [0, -1, True, float("inf"), 1.5]),
            ("per_generation_hours", [0, -1, True, float("nan")]),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.load("scoreless", generation_policy={field: value})

    def test_optional_request_timeout_preserves_explicit_tool_and_phase_limits(self):
        request = replace(_request("/tmp"), timeout_seconds=None)
        with task_execution_policy_scope({"tool_execution_timeout_seconds": 300}):
            bound = apply_task_execution_policy(request)
            with (
                patch("praxist.core.execution_policy.time.time", return_value=100),
                task_execution_deadline_scope(140),
            ):
                limited = apply_task_execution_policy(request)
        self.assertIsNone(bound.timeout_seconds)
        self.assertEqual(bound.runtime_options["tool_execution_timeout_seconds"], 300)
        self.assertEqual(limited.timeout_seconds, 40)
        self.assertEqual(limited.runtime_options["tool_execution_timeout_seconds"], 40)


if __name__ == "__main__":
    unittest.main()
