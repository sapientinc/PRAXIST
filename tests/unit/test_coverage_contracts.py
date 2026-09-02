from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from praxist.core.credentials import CredentialRef
from praxist.core.protocol import (
    AgentEvent,
    AgentRunRequest,
    AgentRunResult,
    BudgetDecision,
    BudgetGrant,
    BudgetRequest,
    CachePolicy,
    EnvPolicy,
    ModelCallSpec,
    ModelProfile,
    ModelResult,
    ToolCallRecord,
    ToolCallResult,
    ToolPermissionSet,
    ToolServerRef,
)
from praxist.plugins.graph_maintainers.finding_graph_mvp import engine, viz
from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
    FrontierStore,
    _walk_for_metric,
    annotate_findings_with_diversity_overlap,
    compute_dimension_overlap,
)
from praxist.plugins.workflow_stages.research_loop.backend.generation_loop import (
    GenerationLoop,
)
from praxist.plugins.workflow_stages.research_loop.backend.prompt_strategy import (
    _build_axis_assignment_block,
    _build_diversity_penalty_block,
    _generate_variant_hint,
)
from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store
from praxist.task_spec import (
    _normalize_anchor_metrics,
    _normalize_diversity_dimensions,
    _normalize_frontier_lanes,
    _normalize_must_explore_axes,
    load_task_spec,
)


class ProtocolCoverageContractsTest(unittest.TestCase):
    def test_runtime_protocol_serializes_nested_redacted_refs(self) -> None:
        credential = CredentialRef(
            scope="model_provider",
            provider="fake_provider",
            target_ref="model_provider:fake_provider",
            key_id="fake:key",
            source="runtime_profile",
        )
        profile = ModelProfile(
            profile_id="cheap",
            provider_ref="model_provider:fake_provider",
            model="fake-model",
            api_format="fake",
            capability_tags=["tool_use"],
            cost_tier="cheap",
            default_parameters={"temperature": 0},
        )
        call = ModelCallSpec(
            profile_id=profile.profile_id,
            provider_ref=profile.provider_ref,
            api_format=profile.api_format,
            model=profile.model,
            parameters=profile.default_parameters,
            credential_ref=credential,
        )
        tool_permissions = ToolPermissionSet(allowed_tools=["Read"], denied_tools=["Bash"])
        tool_server = ToolServerRef(
            ref="tool_server:evaluation_tools",
            server_name="evaluation-tools",
            tool_names=["share_finding"],
            credential_ref=credential,
            metadata={"mode": "offline"},
        )
        env_policy = EnvPolicy(
            exposed_env_keys=["PYTHONPATH"],
            scoped_credential_refs=[credential],
        )
        cache_policy = CachePolicy(
            mode="runtime_managed",
            frozen_prefix_hash="abc123",
            cache_breakpoints=["frozen"],
            runtime_cache_strategy="stable_prefix",
            provider_cache_strategy="automatic",
        )
        request = AgentRunRequest(
            request_id="req",
            run_id="run",
            stage_id="research_loop",
            role_ref="task_role:peer",
            agent_runtime_ref="agent_runtime:fake_runtime",
            prompt_ref={"path": "prompt.txt"},
            system_prompt_ref=None,
            cwd="/tmp",
            model_profile_ref=profile.profile_id,
            model_call=call,
            tool_permissions=tool_permissions,
            tool_servers=[tool_server.to_dict()],
            env_policy=env_policy,
            credential_ref=credential,
            credential_mode="single",
            budget_grant_id="grant",
            artifact_scope="peer",
            timeout_seconds=30,
            cache_policy=cache_policy,
            runtime_options={"stream": False},
        )
        event = AgentEvent(
            event_id="evt",
            run_id="run",
            agent_run_id="agent",
            stage_id="research_loop",
            type="message",
            payload={"text": "ok"},
            artifact_refs=[{"path": "out.txt"}],
            credential_refs=[credential],
            timestamp_ms=1,
        )
        tool_use = ToolCallRecord(
            tool_call_id="tool",
            server_name="evaluation-tools",
            tool_name="share_finding",
            started_at_ms=1,
            finished_at_ms=2,
            success=True,
            artifact_refs=[],
            failover_reason=None,
        )
        result = AgentRunResult(
            success=True,
            events=[event],
            text_output_refs=[{"path": "out.txt"}],
            tool_uses=[tool_use],
            error=None,
            failover_reason=None,
            credential_ref=credential,
        )

        self.assertEqual(profile.to_dict()["model"], "fake-model")
        self.assertEqual(
            ModelResult(True, "p", "m", "text", {"tokens": 1}, {}, None, None).to_dict()["usage"][
                "tokens"
            ],
            1,
        )
        self.assertEqual(
            ToolCallResult("s", "t", True, {"ok": True}).to_dict()["output"]["ok"], True
        )
        self.assertEqual(request.to_dict()["model_call"]["credential_ref"]["key_id"], "fake:key")
        self.assertEqual(
            request.to_dict()["env_policy"]["scoped_credential_refs"][0]["provider"],
            "fake_provider",
        )
        self.assertEqual(
            result.to_dict()["events"][0]["credential_refs"][0]["source"], "runtime_profile"
        )

    def test_budget_protocol_serializes_grant_and_review_decision(self) -> None:
        from praxist.core.budget import _invalid_budget_units, policy_for_ref
        from praxist.plugins.budget_policies.default_basic.policy import (
            DefaultBasicBudgetPolicy,
            create_policy,
        )

        request = BudgetRequest(
            request_id="req",
            requester_id="peer",
            experiment_id="exp",
            model_profile_ref=None,
            requested={"tokens": 10},
            expected_value={"confidence": "medium"},
            evidence_refs=["finding:a"],
            cheaper_alternatives=["downscope"],
            abort_conditions=["leak"],
        )
        grant = BudgetGrant(
            grant_id="grant",
            approved={"tokens": 5},
            conditions=["record_usage"],
            expires_at_generation=2,
        )
        decision = BudgetDecision(
            decision="downscope",
            reason_codes=["over_budget"],
            grant=grant,
            model_profile_override="cheap",
            review_target="chair",
        )

        self.assertEqual(request.requested["tokens"], 10)
        self.assertEqual(decision.to_dict()["grant"]["approved"]["tokens"], 5)
        self.assertIsNone(BudgetDecision("deny", ["risk"], None).to_dict()["grant"])
        self.assertEqual(
            _invalid_budget_units(
                {
                    "unknown": 1,
                    "tokens": "bad",
                    "wall_clock_seconds": float("inf"),
                    "gpu_hours": -1,
                }
            ),
            ["unknown", "tokens", "wall_clock_seconds", "gpu_hours"],
        )

        policy = create_policy()
        self.assertIsInstance(policy, DefaultBasicBudgetPolicy)
        impossible = policy.decide(replace(request, expected_value={"impossible": True}))
        self.assertEqual(impossible.decision, "deny")
        cheap = policy.decide(
            replace(
                request,
                request_id="cheap",
                requested={"tokens": 100, "wall_clock_seconds": 30, "gpu_hours": 0},
                expected_value={"confidence": "weak"},
                evidence_refs=[],
            )
        )
        self.assertEqual(cheap.decision, "grant")
        review = policy.decide(
            replace(
                request,
                request_id="review",
                requested={"tokens": 10_000, "wall_clock_seconds": 300, "gpu_hours": 1},
                expected_value={"confidence": "medium"},
                evidence_refs=[],
            )
        )
        self.assertEqual(review.decision, "require_review")
        with self.assertRaises(ValueError):
            policy_for_ref("agent_runtime:not_budget")


class TaskSpecCoverageContractsTest(unittest.TestCase):
    def test_task_spec_normalizers_are_tolerant_and_typed(self) -> None:
        from praxist import task_spec

        class BadString:
            def __str__(self) -> str:
                raise TypeError("bad string")

        self.assertEqual(_normalize_diversity_dimensions(None), [])
        self.assertEqual(_normalize_diversity_dimensions({"bad": True}), [])
        self.assertEqual(
            _normalize_diversity_dimensions(
                [
                    "mechanism",
                    {"name": "cost", "description": "runtime"},
                    {"name": BadString()},
                    {"bad": True},
                    3,
                ]
            ),
            [
                {"name": "mechanism", "description": "", "examples": ""},
                {"name": "cost", "description": "runtime", "examples": ""},
            ],
        )
        self.assertEqual(_normalize_must_explore_axes({"bad": True}), [])
        self.assertEqual(
            _normalize_must_explore_axes(
                ["axis A", {"name": "axis B", "description": None}, {"name": BadString()}, {}]
            ),
            [{"name": "axis A", "description": ""}, {"name": "axis B", "description": ""}],
        )
        self.assertEqual(_normalize_anchor_metrics({"bad": True}), [])
        self.assertEqual(
            _normalize_anchor_metrics(
                [
                    {"name": "acc", "direction": "maximize"},
                    ["gap", "minimize"],
                    {"name": "bad", "direction": "asc"},
                    {"name": ""},
                    [BadString(), "maximize"],
                    ["malformed"],
                ]
            ),
            [("acc", "maximize"), ("gap", "minimize")],
        )
        self.assertEqual(task_spec._optional_float("bad", field_name="x"), None)
        self.assertEqual(task_spec._normalize_str_list([None, BadString(), " ok "]), ["None", "ok"])
        self.assertEqual(
            task_spec._normalize_metric_bounds(
                {"": 1, "bad": "nan?", "ok": "2.5"},
                field_name="min_metrics",
            ),
            {"ok": 2.5},
        )
        self.assertEqual(_normalize_frontier_lanes({"bad": True}), [])
        self.assertEqual(
            _normalize_frontier_lanes(
                [
                    1,
                    {"name": ""},
                    {
                        "name": "candidate",
                        "k": "bad",
                        "cumulative_cap": "bad",
                        "include_lanes": "candidate",
                        "exclude_roles": ["theorist"],
                        "require_metrics": ["task_score"],
                        "require_truthy_metrics": ["promotion_eligible"],
                        "require_falsey_metrics": ["risk_repair_required"],
                        "min_metrics": {"task_score": "0.0"},
                        "max_metrics": {"resource_cost": 10},
                        "allow_risk_violating": "false",
                        "allow_lower_tier": "true",
                        "allow_non_promotable": "true",
                        "allow_missing_tier": "false",
                        "admit_new_high": "false",
                        "axes": [{"metric": "task_score", "direction": "maximize"}],
                        "optional_axes": [{"metric": "future_fitness", "direction": "maximize"}],
                    },
                    {"bad": True},
                ]
            ),
            [
                {
                    "name": "candidate",
                    "k": 1,
                    "axes": [("task_score", "maximize")],
                    "optional_axes": [("future_fitness", "maximize")],
                    "include_lanes": ["candidate"],
                    "exclude_lanes": [],
                    "include_families": [],
                    "exclude_families": [],
                    "include_tags": [],
                    "exclude_tags": [],
                    "include_roles": [],
                    "exclude_roles": ["theorist"],
                    "require_metrics": ["task_score"],
                    "require_truthy_metrics": ["promotion_eligible"],
                    "require_falsey_metrics": ["risk_repair_required"],
                    "min_metrics": {"task_score": 0.0},
                    "max_metrics": {"resource_cost": 10.0},
                    "allow_risk_violating": False,
                    "allow_lower_tier": True,
                    "parent_eligible": False,
                    "allow_non_promotable": True,
                    "allow_missing_tier": False,
                    "admit_new_high": False,
                    "description": "",
                }
            ],
        )
        string_axis_lane = _normalize_frontier_lanes(
            [
                {
                    "name": "novelty",
                    "direction": "minimize",
                    "axes": ["novelty_score"],
                    "optional_axes": ["runtime_cost"],
                }
            ]
        )[0]
        self.assertEqual(string_axis_lane["axes"], [("novelty_score", "minimize")])
        self.assertEqual(string_axis_lane["optional_axes"], [("runtime_cost", "minimize")])

    def test_task_spec_loads_defensive_config_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "description.md").write_text("desc", encoding="utf-8")
            prompt_base = root / "base.jinja2"
            prompt_base.write_text("base", encoding="utf-8")
            spec_path = root / "task.yaml"
            spec_path.write_text(
                """
task_id: t
task_name: Test Task
description_file: description.md
evaluation:
  requires_tier: "false"
  frontier_lanes:
    - name: capped
      cumulative_cap: 0
      max_metrics: bad
gems:
  result_cell_metric_derivations:
    - name: validation_metric
      source_keys: [score]
      validation_only: "false"
generation_policy:
  max_generations: 3
  cohort_size: 2
  per_generation_hours: 2
run_lifecycle: bad
multi_pi: bad
research_memory: bad
prompt_layout: bad
synthesis_trigger:
  max_interval_minutes: 110
  adaptive: bad
praxist_plugins: bad
""",
                encoding="utf-8",
            )

            loaded = load_task_spec(spec_path)

            self.assertEqual(loaded.run_lifecycle.max_wall_clock_hours, None)
            self.assertEqual(loaded.multi_pi.n_rounds, 2)
            self.assertFalse(loaded.research_memory.enabled)
            self.assertEqual(loaded.prompt_layout.base_template, "")
            self.assertEqual(loaded.gems.max_gems_total, 4)
            self.assertEqual(loaded.panel_topology_ref, "panel_topology:legacy_multi_pi_two_round")
            self.assertFalse(loaded.evaluation.requires_tier)
            self.assertEqual(loaded.evaluation.frontier_lanes[0]["cumulative_cap"], 1)
            self.assertFalse(loaded.gems.result_cell_metric_derivations[0]["validation_only"])
            self.assertEqual(loaded.get_prompt_base_path(prompt_base), prompt_base)

            dir_base = root / "base_dir"
            dir_base.mkdir()
            dir_spec_path = root / "task_with_dir_base.yaml"
            dir_spec_path.write_text(
                f"prompt_layout:\n  base_template: {dir_base.name}\n",
                encoding="utf-8",
            )
            dir_loaded = load_task_spec(dir_spec_path)
            with self.assertRaises(ValueError):
                dir_loaded.get_prompt_base_path(prompt_base)

            fatal_spec_path = root / "task_with_tiny_cap.yaml"
            fatal_spec_path.write_text(
                """
generation_policy:
  per_generation_hours: 0.1
synthesis_trigger:
  max_interval_minutes: 20
""",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_task_spec(fatal_spec_path)

            adaptive_spec_path = root / "task_with_adaptive_clamp.yaml"
            adaptive_spec_path.write_text(
                """
generation_policy:
  per_generation_hours: 2
synthesis_trigger:
  max_interval_minutes: 60
  adaptive:
    enabled: true
    max_interval_ceiling_minutes: 180
gems:
  reset_interval_generations: bad
  max_gems_total: 99
  prompt_max_gems: 99
""",
                encoding="utf-8",
            )
            adaptive_loaded = load_task_spec(adaptive_spec_path)
            self.assertEqual(adaptive_loaded.synthesis_trigger.max_interval_minutes, 60)
            self.assertEqual(
                adaptive_loaded.synthesis_trigger.adaptive["max_interval_ceiling_minutes"],
                90,
            )
            self.assertEqual(adaptive_loaded.gems.reset_interval_generations, 6)
            self.assertEqual(adaptive_loaded.gems.max_gems_total, 99)

    def test_load_task_spec_applies_cross_field_safety_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "description.md").write_text("desc", encoding="utf-8")
            (root / "prompt_task.jinja2").write_text("prompt", encoding="utf-8")
            (root / "eval.py").write_text("print('ok')\n", encoding="utf-8")
            spec_path = root / "task.yaml"
            spec_path.write_text(
                """
task_id: t
task_name: Test Task
description_file: description.md
research_direction: fallback
evaluation:
  primary_metric: loss
  direction: minimize
  aux_metrics: [gap]
  anchor_metrics:
    - {name: cost, direction: minimize}
  frontier_lanes:
    - name: alpha
      k: 2
      cumulative_cap: 10
      include_lanes: [alpha]
      allow_lower_tier: true
      allow_non_promotable: true
      require_truthy_metrics: [promotion_eligible]
      require_falsey_metrics: [risk_repair_required]
      min_metrics: {active_alpha: 0.0}
      axes:
        - {name: active_alpha, direction: maximize}
      optional_axes:
        - {name: future_fitness, direction: maximize}
  diversity_dimensions:
    - name: mechanism
  must_explore_axes:
    - name: axis-a
  requires_tier: true
compute_budget:
  per_experiment_gpu_hours: 1.5
  max_parallel_runs_per_peer: 3
  peer_gpu_memory_gb: 12
  peer_gpu_util_pct: 40
  peer_cpu_cores: 2
generation_policy:
  max_generations: 2
  cohort_size: 4
  per_generation_hours: 2
synthesis_trigger:
  enabled: false
  max_interval_minutes: 60
  adaptive:
    enabled: true
    min_evidence_units: 3
    min_formal_result_peers: 2
pi_agent:
  enabled: false
multi_pi:
  enabled: true
  panel_mode_default: invalid
  chair_peer_budget: 2
  n_rounds: 7
research_memory:
  enabled: false
  rollout_phase: 1
agent:
  premium_mode: true
tiered_eval:
  T1: {seeds: 1}
baselines:
  - {name: base, expected_acc: 0.7}
toolchain:
  framework: python
  entrypoint_template: train.py
  eval_entrypoint: eval.py
  benchmark_entrypoint: harness/benchmark.py
""",
                encoding="utf-8",
            )

            spec = load_task_spec(str(spec_path))

            self.assertEqual(spec.task_id, "t")
            self.assertEqual(spec.get_description(), "desc")
            self.assertEqual(spec.get_prompt_task_path(), root / "prompt_task.jinja2")
            self.assertEqual(spec.evaluation.anchor_metrics, [("cost", "minimize")])
            self.assertEqual(spec.evaluation.frontier_lanes[0]["name"], "alpha")
            self.assertEqual(spec.evaluation.frontier_lanes[0]["k"], 2)
            self.assertEqual(spec.evaluation.frontier_lanes[0]["cumulative_cap"], 10)
            self.assertEqual(
                spec.evaluation.frontier_lanes[0]["axes"], [("active_alpha", "maximize")]
            )
            self.assertEqual(
                spec.evaluation.frontier_lanes[0]["optional_axes"],
                [("future_fitness", "maximize")],
            )
            self.assertTrue(spec.evaluation.frontier_lanes[0]["allow_lower_tier"])
            self.assertTrue(spec.evaluation.frontier_lanes[0]["allow_non_promotable"])
            self.assertEqual(
                spec.evaluation.frontier_lanes[0]["require_truthy_metrics"],
                ["promotion_eligible"],
            )
            self.assertEqual(
                spec.evaluation.frontier_lanes[0]["require_falsey_metrics"],
                ["risk_repair_required"],
            )
            self.assertEqual(spec.evaluation.direction, "minimize")
            self.assertTrue(spec.evaluation.requires_tier)
            self.assertEqual(spec.compute_budget.peer_gpu_util_pct, 40)
            self.assertEqual(spec.multi_pi.panel_mode_default, "full")
            self.assertEqual(spec.multi_pi.n_rounds, 2)
            self.assertEqual(spec.multi_pi.chair_peer_budget, 4)
            self.assertEqual(spec.synthesis_trigger.adaptive["min_evidence_units"], 3)
            self.assertEqual(spec.synthesis_trigger.adaptive["min_formal_result_peers"], 2)
            self.assertTrue(spec.research_memory.enabled)
            self.assertEqual(spec.research_memory.rollout_phase, 2)
            self.assertTrue(spec.agent.premium_mode)
            self.assertEqual(spec.tiered_eval["T1"]["seeds"], 1)
            self.assertEqual(spec.toolchain.eval_entrypoint, "eval.py")
            self.assertEqual(spec.toolchain.benchmark_entrypoint, "harness/benchmark.py")
            self.assertEqual(spec.baselines[0].metric_value, 0.7)
            self.assertEqual(spec.baselines[0].expected_acc, 0.7)

    def test_load_task_spec_parses_research_loop_control_fields(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "description.md").write_text("desc", encoding="utf-8")
            (root / "prompt_task.jinja2").write_text("prompt", encoding="utf-8")
            spec_path = root / "task.yaml"
            spec_path.write_text(
                """
task_id: generic
task_name: Generic Task
description_file: description.md
evaluation:
  primary_metric: score
  direction: maximize
  maturity_policy:
    min_effort_ratio: 0.6
    min_coverage_ratio: 0.7
    require_ratio_gate: true
  constructive_peer_mix_enabled: false
  constructive_target_ratio: 0.8
  launch_guard:
    enabled: true
    estimated_heavy_eval_minutes: 12
    estimated_close_grade_eval_minutes: 10
    safety_factor: 1.5
  frontier_lanes:
    - name: incubator
      admit_new_high: true
      axes:
        - {metric: score, direction: maximize}
        - {metric: risk_score, direction: minimize}
synthesis_trigger:
  mature_quorum_fraction: 0.4
  max_interval_minutes: 60
generation_policy:
  per_generation_hours: 2
""",
                encoding="utf-8",
            )

            spec = load_task_spec(spec_path)

        self.assertEqual(spec.evaluation.maturity_policy["min_effort_ratio"], 0.6)
        self.assertEqual(spec.evaluation.maturity_policy["min_coverage_ratio"], 0.7)
        self.assertTrue(spec.evaluation.maturity_policy["require_ratio_gate"])
        self.assertFalse(spec.evaluation.constructive_peer_mix_enabled)
        self.assertEqual(spec.evaluation.constructive_target_ratio, 0.8)
        self.assertTrue(spec.evaluation.launch_guard["enabled"])
        self.assertEqual(spec.evaluation.launch_guard["estimated_heavy_eval_minutes"], 12.0)
        self.assertEqual(
            spec.evaluation.launch_guard["estimated_close_grade_eval_minutes"],
            10.0,
        )
        self.assertEqual(spec.evaluation.launch_guard["safety_factor"], 1.5)
        self.assertTrue(spec.evaluation.frontier_lanes[0]["admit_new_high"])
        self.assertEqual(
            spec.evaluation.frontier_lanes[0]["axes"],
            [("score", "maximize"), ("risk_score", "minimize")],
        )
        self.assertEqual(spec.synthesis_trigger.mature_quorum_fraction, 0.4)

    def test_task_entrypoint_evaluation_command_falls_back_to_toolchain(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "description.md").write_text("desc", encoding="utf-8")
            (root / "prompt_task.jinja2").write_text("prompt", encoding="utf-8")
            evaluator = root / "evaluations" / "public" / "run.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text("print('ok')\n", encoding="utf-8")
            spec_path = root / "task.yaml"
            spec_path.write_text(
                """
task_id: generic
task_name: Generic Task
description_file: description.md
task_entrypoints:
  evaluation:
    command: evaluations/public/run.py
""",
                encoding="utf-8",
            )
            spec = load_task_spec(spec_path)

        self.assertEqual(spec.toolchain.eval_entrypoint, "evaluations/public/run.py")

    def test_task_entrypoint_is_validated_relative_to_task_root(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_path = root / "task.yaml"
            spec_path.write_text(
                "task_entrypoints:\n  evaluation:\n    command: evaluations/missing/run.py\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "relative to task root"):
                load_task_spec(spec_path)

    def test_wrapped_task_entrypoints_are_validated_relative_to_task_root(self) -> None:
        from praxist.task_spec import load_task_spec

        commands = (
            "python3.8 evaluations/missing/run.py",
            "/usr/bin/python3 -u evaluations/missing/run.py",
            "python --check-hash-based-pycs always evaluations/missing/run.py",
            "env MODE=test python evaluations/missing/run.py",
            "bash -lc 'python -u evaluations/missing/run.py'",
            "bash -lc 'python evaluations/missing/run.py > result.log'",
            "bash -lc 'exec python evaluations/missing/run.py'",
            "bash -lc 'command -p python evaluations/missing/run.py'",
        )
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            for command in commands:
                with self.subTest(command=command):
                    spec_path.write_text(
                        "task_entrypoints:\n  evaluation:\n    command: "
                        + json.dumps(command)
                        + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(FileNotFoundError, "relative to task root"):
                        load_task_spec(spec_path)

    def test_direct_task_entrypoint_with_arguments_is_still_validated(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            spec_path.write_text(
                "task_entrypoints:\n"
                "  evaluation:\n"
                "    command: evaluations/missing/run.py --mode full\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "relative to task root"):
                load_task_spec(spec_path)

    def test_static_env_chdir_entrypoint_is_validated_from_task_root(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = root / "evaluations" / "v2" / "run.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text("print('ok')\n", encoding="utf-8")
            spec_path = root / "task.yaml"
            command = "env -C evaluations/v2 python run.py"
            spec_path.write_text(
                "task_entrypoints:\n  evaluation:\n    command: " + json.dumps(command) + "\n",
                encoding="utf-8",
            )

            loaded = load_task_spec(spec_path)

        self.assertEqual(loaded.toolchain.eval_entrypoint, command)

    def test_static_shell_cd_and_redirection_entrypoint_is_validated(self) -> None:
        from praxist.task_spec import (
            load_task_spec,
            resolve_declared_evaluation_entrypoint,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = root / "evaluations" / "run.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text("print('ok')\n", encoding="utf-8")
            command = "bash -lc 'cd evaluations && python run.py 2>&1'"
            spec_path = root / "task.yaml"
            spec_path.write_text(
                "task_entrypoints:\n  evaluation:\n    command: " + json.dumps(command) + "\n",
                encoding="utf-8",
            )

            loaded = load_task_spec(spec_path)
            resolved = resolve_declared_evaluation_entrypoint(command, task_dir=root)

        self.assertEqual(loaded.toolchain.eval_entrypoint, command)
        self.assertEqual(resolved, evaluator.resolve())

    def test_static_env_chdir_missing_entrypoint_is_rejected(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluations" / "v2").mkdir(parents=True)
            spec_path = root / "task.yaml"
            spec_path.write_text(
                "task_entrypoints:\n"
                "  evaluation:\n"
                "    command: env --chdir=evaluations/v2 python missing.py\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "relative to task root"):
                load_task_spec(spec_path)

    def test_static_env_chdir_is_validated_through_transparent_wrappers(self) -> None:
        from praxist.task_spec import load_task_spec

        commands = (
            "exec env --chdir evaluations/v2 python run.py",
            "bash -lc 'env -C evaluations/v2 python run.py'",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = root / "evaluations" / "v2" / "run.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text("print('ok')\n", encoding="utf-8")
            spec_path = root / "task.yaml"
            for command in commands:
                with self.subTest(command=command):
                    spec_path.write_text(
                        "task_entrypoints:\n"
                        "  evaluation:\n"
                        "    command: " + json.dumps(command) + "\n",
                        encoding="utf-8",
                    )

                    loaded = load_task_spec(spec_path)

                    self.assertEqual(loaded.toolchain.eval_entrypoint, command)

    def test_evaluator_chdir_argument_is_not_parsed_as_env_chdir(self) -> None:
        from praxist.task_spec import (
            declared_evaluation_entrypoint_chdir,
            load_task_spec,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = root / "evaluate.py"
            evaluator.write_text("print('ok')\n", encoding="utf-8")
            command = "env MODE=test python evaluate.py --chdir outputs"
            spec_path = root / "task.yaml"
            spec_path.write_text(
                "task_entrypoints:\n  evaluation:\n    command: " + json.dumps(command) + "\n",
                encoding="utf-8",
            )

            loaded = load_task_spec(spec_path)

        self.assertEqual(loaded.toolchain.eval_entrypoint, command)
        self.assertEqual(declared_evaluation_entrypoint_chdir(command), "")

    def test_failed_cd_alternative_does_not_rebase_static_evaluator(self) -> None:
        from praxist.task_spec import (
            load_task_spec,
            resolve_declared_evaluation_entrypoint,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = root / "evaluations" / "run.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text("print('ok')\n", encoding="utf-8")
            command = "bash -lc 'cd missing || python evaluations/run.py'"
            spec_path = root / "task.yaml"
            spec_path.write_text(
                "task_entrypoints:\n  evaluation:\n    command: " + json.dumps(command) + "\n",
                encoding="utf-8",
            )

            loaded = load_task_spec(spec_path)
            resolved = resolve_declared_evaluation_entrypoint(command, task_dir=root)

        self.assertEqual(loaded.toolchain.eval_entrypoint, command)
        self.assertEqual(resolved, evaluator.resolve())

    def test_compound_entrypoint_selects_last_path_launch(self) -> None:
        from praxist.task_spec import resolve_declared_evaluation_entrypoint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare = root / "prepare.py"
            evaluator = root / "evaluations" / "run.py"
            prepare.write_text("print('prepare')\n", encoding="utf-8")
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text("print('evaluate')\n", encoding="utf-8")
            command = "bash -lc 'python prepare.py && python evaluations/run.py'"

            resolved = resolve_declared_evaluation_entrypoint(command, task_dir=root)

        self.assertEqual(resolved, evaluator.resolve())

    def test_compound_entrypoint_may_generate_evaluator_at_runtime(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "build.py").write_text("print('build')\n", encoding="utf-8")
            command = "bash -lc 'python build.py && python generated/evaluate.py'"
            spec_path = root / "task.yaml"
            spec_path.write_text(
                "task_entrypoints:\n  evaluation:\n    command: " + json.dumps(command) + "\n",
                encoding="utf-8",
            )

            loaded = load_task_spec(spec_path)

        self.assertEqual(loaded.toolchain.eval_entrypoint, command)

    def test_dynamic_evaluator_commands_remain_runtime_resolved(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            spec_path.write_text(
                "task_entrypoints:\n"
                "  evaluation:\n"
                "    command: \"bash -lc 'python $TASK_EVALUATOR --mode full'\"\n",
                encoding="utf-8",
            )

            loaded = load_task_spec(spec_path)

        self.assertIn("$TASK_EVALUATOR", loaded.toolchain.eval_entrypoint)

    def test_shell_entrypoint_with_attached_operator_is_left_to_runtime(self) -> None:
        from praxist.task_spec import load_task_spec

        commands = (
            "bash -lc 'python evaluations/missing/run.py>result.log'",
            "bash -lc 'python evaluations/missing/run.py # operator note'",
        )
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            for command in commands:
                with self.subTest(command=command):
                    spec_path.write_text(
                        "task_entrypoints:\n  evaluation:\n    command: "
                        + json.dumps(command)
                        + "\n",
                        encoding="utf-8",
                    )
                    loaded = load_task_spec(spec_path)
                    self.assertEqual(loaded.toolchain.eval_entrypoint, command)

    def test_bare_external_evaluator_command_remains_supported(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            spec_path.write_text(
                "task_entrypoints:\n  evaluation:\n    command: external-evaluator\n",
                encoding="utf-8",
            )

            loaded = load_task_spec(spec_path)

        self.assertEqual(loaded.toolchain.eval_entrypoint, "external-evaluator")

    def test_evaluator_entrypoint_can_be_relative_to_configured_runtime_cwd(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluations = root / "evaluations"
            evaluations.mkdir()
            (evaluations / "run.py").write_text("print('ok')\n", encoding="utf-8")
            spec_path = root / "task.yaml"
            spec_path.write_text(
                "runtime_environment:\n"
                "  cwd: evaluations\n"
                "task_entrypoints:\n"
                "  evaluation:\n"
                "    command: python run.py\n",
                encoding="utf-8",
            )

            loaded = load_task_spec(spec_path)

        self.assertEqual(loaded.toolchain.eval_entrypoint, "python run.py")

    def test_mature_evaluator_must_fit_generation_close_horizon(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_path = root / "task.yaml"
            spec_path.write_text(
                """
evaluation:
  launch_guard:
    enabled: true
    estimated_heavy_eval_minutes: 330
    estimated_close_grade_eval_minutes: 330
    safety_factor: 1.1
generation_policy:
  per_generation_hours: 7
synthesis_trigger:
  max_interval_minutes: 350
  mature_quorum_fraction: 0.25
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "evaluation timing is unreachable"):
                load_task_spec(spec_path)

            spec_path.write_text(
                spec_path.read_text(encoding="utf-8").replace(
                    "max_interval_minutes: 350", "max_interval_minutes: 420"
                ),
                encoding="utf-8",
            )
            loaded = load_task_spec(spec_path)

        self.assertEqual(loaded.synthesis_trigger.max_interval_minutes, 420)

    def test_legacy_heavy_estimate_warns_without_rejecting_an_unknown_close_protocol(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            spec_path.write_text(
                """
evaluation:
  launch_guard:
    enabled: true
    estimated_heavy_eval_minutes: 330
    safety_factor: 1.1
generation_policy:
  per_generation_hours: 7
synthesis_trigger:
  max_interval_minutes: 350
  mature_quorum_fraction: 0.25
""",
                encoding="utf-8",
            )

            with self.assertLogs("praxist.task_spec", level="WARNING") as captured:
                legacy = load_task_spec(spec_path)
            self.assertTrue(
                any("legacy task timing may be unreachable" in line for line in captured.output)
            )
            spec_path.write_text(
                spec_path.read_text(encoding="utf-8").replace(
                    "estimated_heavy_eval_minutes: 330",
                    "estimated_heavy_eval_minutes: 330\n"
                    "    estimated_close_grade_eval_minutes: 180",
                ),
                encoding="utf-8",
            )
            loaded = load_task_spec(spec_path)

        self.assertEqual(legacy.evaluation.launch_guard["estimated_heavy_eval_minutes"], 330.0)
        self.assertEqual(loaded.evaluation.launch_guard["estimated_heavy_eval_minutes"], 330.0)
        self.assertEqual(
            loaded.evaluation.launch_guard["estimated_close_grade_eval_minutes"],
            180.0,
        )

    def test_nonpositive_close_grade_estimate_keeps_legacy_task_runnable(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            for invalid_estimate in ("0", "not-a-number"):
                with self.subTest(invalid_estimate=invalid_estimate):
                    spec_path.write_text(
                        "evaluation:\n"
                        "  launch_guard:\n"
                        "    estimated_heavy_eval_minutes: 330\n"
                        f"    estimated_close_grade_eval_minutes: {invalid_estimate}\n"
                        "    safety_factor: 1.1\n"
                        "generation_policy:\n"
                        "  per_generation_hours: 7\n"
                        "synthesis_trigger:\n"
                        "  max_interval_minutes: 350\n"
                        "  mature_quorum_fraction: 0.25\n",
                        encoding="utf-8",
                    )

                    with self.assertLogs("praxist.task_spec", level="WARNING") as captured:
                        loaded = load_task_spec(spec_path)
                    self.assertTrue(
                        any(
                            "legacy task timing may be unreachable" in line
                            for line in captured.output
                        )
                    )
                    self.assertEqual(
                        loaded.evaluation.launch_guard["estimated_heavy_eval_minutes"],
                        330.0,
                    )

    def test_explicit_late_signal_workflow_can_disable_timing_gate(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            spec_path.write_text(
                """
evaluation:
  launch_guard:
    enabled: true
    estimated_heavy_eval_minutes: 330
    safety_factor: 1.1
generation_policy:
  per_generation_hours: 7
synthesis_trigger:
  max_interval_minutes: 350
  mature_quorum_fraction: 0
""",
                encoding="utf-8",
            )

            loaded = load_task_spec(spec_path)

        self.assertTrue(loaded.evaluation.launch_guard["enabled"])

    def test_optional_adaptive_formal_target_does_not_override_information_density_close(
        self,
    ) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            spec_path.write_text(
                """
evaluation:
  launch_guard:
    enabled: true
    estimated_close_grade_eval_minutes: 330
    safety_factor: 1.1
generation_policy:
  per_generation_hours: 7
synthesis_trigger:
  max_interval_minutes: 350
  mature_quorum_fraction: 0
  adaptive:
    enabled: true
    min_formal_result_peers: 2
""",
                encoding="utf-8",
            )

            loaded = load_task_spec(spec_path)

        self.assertEqual(loaded.synthesis_trigger.mature_quorum_fraction, 0.0)

    def test_user_authorized_close_grade_protocol_can_fit_below_optional_heavy_eval(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            spec_path.write_text(
                """
evaluation:
  launch_guard:
    enabled: true
    estimated_heavy_eval_minutes: 330
    estimated_close_grade_eval_minutes: 180
    safety_factor: 1.1
generation_policy:
  per_generation_hours: 7
synthesis_trigger:
  max_interval_minutes: 350
  mature_quorum_fraction: 0.25
""",
                encoding="utf-8",
            )

            loaded = load_task_spec(spec_path)

        self.assertEqual(
            loaded.evaluation.launch_guard["estimated_close_grade_eval_minutes"],
            180.0,
        )

    def test_disabled_adaptive_ceiling_does_not_extend_close_horizon(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            spec_path.write_text(
                """
evaluation:
  launch_guard:
    enabled: true
    estimated_close_grade_eval_minutes: 330
    safety_factor: 1.1
generation_policy:
  per_generation_hours: 10
synthesis_trigger:
  max_interval_minutes: 350
  mature_quorum_fraction: 0.25
  adaptive:
    enabled: false
    max_interval_ceiling_minutes: 500
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "evaluation timing is unreachable"):
                load_task_spec(spec_path)

    def test_disabled_adaptive_ceiling_does_not_clamp_fixed_synthesis_horizon(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            spec_path.write_text(
                """
generation_policy:
  per_generation_hours: 7
synthesis_trigger:
  max_interval_minutes: 350
  adaptive:
    enabled: false
    max_interval_ceiling_minutes: 500
""",
                encoding="utf-8",
            )

            loaded = load_task_spec(spec_path)

        self.assertEqual(loaded.synthesis_trigger.max_interval_minutes, 350)
        self.assertEqual(
            loaded.synthesis_trigger.adaptive["max_interval_ceiling_minutes"],
            500,
        )

    def test_disabled_launch_guard_still_validates_required_close_evidence(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            spec_path.write_text(
                """
evaluation:
  launch_guard:
    enabled: false
    estimated_close_grade_eval_minutes: 330
    safety_factor: 1.1
generation_policy:
  per_generation_hours: 7
synthesis_trigger:
  max_interval_minutes: 350
  mature_quorum_fraction: 0.25
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "evaluation timing is unreachable"):
                load_task_spec(spec_path)

            spec_path.write_text(
                spec_path.read_text(encoding="utf-8").replace(
                    "mature_quorum_fraction: 0.25", "mature_quorum_fraction: 0"
                ),
                encoding="utf-8",
            )
            loaded = load_task_spec(spec_path)

        self.assertFalse(loaded.evaluation.launch_guard["enabled"])

    def test_load_task_spec_tolerates_malformed_synthesis_trigger_numbers(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "description.md").write_text("desc", encoding="utf-8")
            (root / "prompt_task.jinja2").write_text("prompt", encoding="utf-8")
            spec_path = root / "task.yaml"
            spec_path.write_text(
                """
task_id: generic
task_name: Generic Task
description_file: description.md
evaluation:
  primary_metric: score
  direction: maximize
  maturity_policy:
    min_effort_ratio: .nan
    min_coverage_ratio: .inf
    require_ratio_gate: "false"
  constructive_peer_mix_enabled: "not-a-bool"
  constructive_target_ratio: .inf
  launch_guard:
    enabled: "false"
    estimated_heavy_eval_minutes: -.inf
    safety_factor: .nan
synthesis_trigger:
  enabled: "false"
  min_findings: .nan
  min_contributing_peers: .inf
  poll_interval_seconds: -.inf
  mature_quorum_fraction: .nan
  min_interval_minutes: -.inf
  max_interval_minutes: .inf
  adaptive:
    enabled: "false"
pi_agent:
  enabled: "false"
  strict: "false"
generation_policy:
  max_generations: .nan
  cohort_size: .inf
  per_generation_hours: .nan
  promote_top_k: -.inf
""",
                encoding="utf-8",
            )

            spec = load_task_spec(spec_path)

        self.assertEqual(spec.evaluation.maturity_policy["min_effort_ratio"], 0.75)
        self.assertEqual(spec.evaluation.maturity_policy["min_coverage_ratio"], 0.80)
        self.assertFalse(spec.evaluation.maturity_policy["require_ratio_gate"])
        self.assertTrue(spec.evaluation.constructive_peer_mix_enabled)
        self.assertEqual(spec.evaluation.constructive_target_ratio, 0.75)
        self.assertFalse(spec.evaluation.launch_guard["enabled"])
        self.assertEqual(spec.evaluation.launch_guard["estimated_heavy_eval_minutes"], 0.0)
        self.assertEqual(
            spec.evaluation.launch_guard["estimated_close_grade_eval_minutes"],
            0.0,
        )
        self.assertEqual(spec.evaluation.launch_guard["safety_factor"], 1.25)
        self.assertFalse(spec.synthesis_trigger.enabled)
        self.assertEqual(spec.synthesis_trigger.min_findings, 30)
        self.assertEqual(spec.synthesis_trigger.min_contributing_peers, 3)
        self.assertEqual(spec.synthesis_trigger.poll_interval_seconds, 30)
        self.assertEqual(spec.synthesis_trigger.mature_quorum_fraction, 0.0)
        self.assertEqual(spec.synthesis_trigger.min_interval_minutes, 120.0)
        self.assertEqual(spec.synthesis_trigger.max_interval_minutes, 240.0)
        self.assertFalse(spec.synthesis_trigger.adaptive.get("enabled"))
        self.assertFalse(spec.pi_agent.enabled)
        self.assertFalse(spec.pi_agent.strict)
        self.assertEqual(spec.generation_policy.max_generations, 8)
        self.assertEqual(spec.generation_policy.cohort_size, 5)
        self.assertEqual(spec.generation_policy.per_generation_hours, 5.0)
        self.assertEqual(spec.generation_policy.promote_top_k, 2)

    def test_load_task_spec_clamps_mature_quorum_fraction_to_valid_ratio(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "description.md").write_text("desc", encoding="utf-8")
            (root / "prompt_task.jinja2").write_text("prompt", encoding="utf-8")
            spec_path = root / "task.yaml"
            spec_path.write_text(
                """
task_id: generic
task_name: Generic Task
description_file: description.md
evaluation:
  primary_metric: score
  direction: maximize
synthesis_trigger:
  mature_quorum_fraction: 1.4
generation_policy:
  cohort_size: 4
""",
                encoding="utf-8",
            )

            spec = load_task_spec(spec_path)

        self.assertEqual(spec.synthesis_trigger.mature_quorum_fraction, 1.0)

    def test_tiered_stage_names_are_not_implicitly_all_mature(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
            evidence_maturity_snapshot,
        )
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "description.md").write_text("desc", encoding="utf-8")
            (root / "prompt_task.jinja2").write_text("prompt", encoding="utf-8")
            spec_path = root / "task.yaml"
            spec_path.write_text(
                """
task_id: generic
task_name: Generic Task
description_file: description.md
evaluation:
  primary_metric: score
  direction: maximize
tiered_eval:
  triage: {effort: low}
  full: {effort: high}
""",
                encoding="utf-8",
            )

            spec = load_task_spec(spec_path)

        policy = spec.evaluation.maturity_policy
        self.assertEqual(policy["complete_stage_labels"], [])
        self.assertEqual(policy["preliminary_stage_labels"], [])
        self.assertIsNone(
            evidence_maturity_snapshot({"evidence_stage": "triage"}, policy)["mature_enough"]
        )
        self.assertIsNone(
            evidence_maturity_snapshot({"evidence_stage": "full"}, policy)["mature_enough"]
        )


class FrontierAndPromptCoverageContractsTest(unittest.TestCase):
    def test_frontier_promotes_primary_and_anchor_without_mutating_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot"
            snapshot.mkdir()
            (snapshot / "model.py").write_text("x = 1\n", encoding="utf-8")
            (snapshot / "__pycache__").mkdir()
            (snapshot / "__pycache__" / "skip.pyc").write_bytes(b"skip")
            findings = [
                {
                    "id": "f1",
                    "finding_id": "f1",
                    "finding_type": "result",
                    "variant_name": "A",
                    "metrics": {
                        "score": 0.5,
                        "cost": 5.0,
                        "tier": "T3",
                        "scored_complete": True,
                    },
                    "snapshot_local_path": str(snapshot),
                },
                {
                    "id": "f2",
                    "finding_id": "f2",
                    "finding_type": "result",
                    "variant_name": "B",
                    "metrics": {
                        "score": 0.9,
                        "cost": 10.0,
                        "tier": "T3",
                        "scored_complete": True,
                    },
                },
                {
                    "id": "f3",
                    "finding_id": "f3",
                    "finding_type": "insight",
                    "variant_name": "C",
                    "metrics": {
                        "score": 0.8,
                        "cost": 1.0,
                        "tier": "T3",
                        "scored_complete": True,
                    },
                },
                {
                    "id": "low",
                    "finding_id": "low",
                    "finding_type": "result",
                    "variant_name": "Low",
                    "metrics": {"score": 1.0, "tier": "T2", "scored_complete": True},
                },
                {
                    "id": "no",
                    "finding_id": "no",
                    "finding_type": "result",
                    "variant_name": "No",
                    "metrics": {
                        "score": 2.0,
                        "tier": "T3",
                        "promotion_eligible": "no",
                        "scored_complete": True,
                    },
                },
                {
                    "id": "theory",
                    "finding_id": "theory",
                    "finding_type": "result",
                    "variant_name": "Theory",
                    "metrics": {"score": 3.0, "tier": "T3", "scored_complete": True},
                },
            ]
            original_first_metrics = dict(findings[0]["metrics"])
            store = FrontierStore(
                Path(tmp) / "frontier",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="maximize",
                anchor_metrics=[("cost", "minimize")],
                require_tier=True,
            )

            promoted = store.promote(0, findings)

            self.assertEqual([p["finding_id"] for p in promoted], ["theory", "f3"])
            self.assertEqual(promoted[1]["promoted_for_anchor"], "cost")
            self.assertEqual(findings[0]["metrics"], original_first_metrics)
            self.assertTrue((Path(tmp) / "frontier" / "frontier_manifest.json").exists())
            self.assertIsNone(promoted[0]["snapshot_path"])
            self.assertIsNone(promoted[1]["snapshot_path"])

            promoted_again = FrontierStore(
                Path(tmp) / "frontier_min",
                promote_top_k=1,
                primary_metric="score",
                metric_direction="minimize",
            ).promote(1, findings[:2])
            self.assertEqual(promoted_again[0]["finding_id"], "f1")

    def test_metric_walk_and_diversity_overlap_are_conservative(self) -> None:
        shared = {"score": 0.7}
        cyclic: dict[str, object] = {"details": {"nested": shared}}
        cyclic["self"] = cyclic
        self.assertEqual(_walk_for_metric(cyclic, "score"), 0.7)
        self.assertIsNone(_walk_for_metric({"score": True}, "score"))
        self.assertIsNone(
            _walk_for_metric({"details": {"score": 2.0}}, "score", _strict_canonical=True)
        )

        finding = {"design_dimensions": {"mechanism": "A", "cost": "low", "info": "grad"}}
        anchor = {
            "variant_name": "anchor",
            "metrics": {"design_dimensions": {"mechanism": "A", "cost": "high", "info": "grad"}},
        }
        overlap = compute_dimension_overlap(finding, anchor)
        self.assertEqual(overlap["overlap_count"], 2)
        annotated = annotate_findings_with_diversity_overlap(
            [finding, {"metrics": {}}], [anchor], 3
        )
        self.assertEqual(annotated[0]["metrics"]["diversity_overlap_status"], "narrow")
        self.assertEqual(annotated[1]["metrics"]["diversity_overlap_status"], "no_data")
        self.assertEqual(
            annotate_findings_with_diversity_overlap([finding], [], 3)[0]["metrics"][
                "diversity_overlap_status"
            ],
            "no_anchors",
        )

    def test_prompt_strategy_covers_axis_diversity_and_strategy_modes(self) -> None:
        class Frontier:
            def __init__(self, entries):
                self._entries = entries

            def get_summary(self):
                return list(self._entries)

        axes = [{"name": "axis-a", "description": "try A"}, {"name": "axis-b"}]
        axis_block = _build_axis_assignment_block(0, 0, 3, axes)
        self.assertIn("axis-a", axis_block)
        self.assertIn("peers 2..2 are free-explore", axis_block.lower())
        self.assertEqual(_build_axis_assignment_block(0, 5, 3, axes), "")

        anchors = [
            {
                "finding_id": "f1",
                "variant_name": "A",
                "metric_value": 0.9,
                "design_dimensions": {"mechanism": "x"},
                "generation_id": 0,
            }
        ]
        diversity = _build_diversity_penalty_block(anchors, [{"name": "mechanism"}])
        self.assertIn("finding_id=f1", diversity)
        self.assertIn("mechanism", diversity)
        self.assertIn("task's", diversity)
        self.assertIn("frontier policy", diversity)
        self.assertNotIn("primary metric alone", diversity)
        self.assertEqual(_build_diversity_penalty_block([]), "")

        cold_hint = _generate_variant_hint(0, 0, 2, "explore", Frontier([]), must_explore_axes=axes)
        self.assertIn("axis-a", cold_hint)
        self.assertEqual(
            _generate_variant_hint(0, 0, 2, "mixed", Frontier([]), must_explore_axes=axes), ""
        )
        pi_hint = _generate_variant_hint(
            1,
            0,
            2,
            "pi_directed",
            Frontier(anchors),
            diversity_dimensions=[{"name": "mechanism"}],
        )
        self.assertIn("authoritative directive", pi_hint)
        self.assertIn("DIVERSITY GUIDANCE", pi_hint)
        self.assertIn("design_dimensions", pi_hint)
        self.assertIn("planned_dimensions", pi_hint)
        self.assertIn("actually implemented and evaluated", pi_hint)
        self.assertIn("finding_id=f1", pi_hint)
        self.assertIn(
            "DIFFERENT direction", _generate_variant_hint(1, 0, 2, "explore", Frontier(anchors))
        )
        self.assertIn(
            "Download its snapshot", _generate_variant_hint(1, 1, 2, "exploit", Frontier(anchors))
        )
        self.assertIn(
            "DIFFERENT direction", _generate_variant_hint(1, 0, 4, "mixed", Frontier(anchors))
        )
        self.assertIn(
            "Download its snapshot", _generate_variant_hint(1, 3, 4, "mixed", Frontier(anchors))
        )

    def test_generic_boundary_helpers_preserve_neutral_fallbacks(self) -> None:
        from praxist.core.cache import frozen_prefix_hash
        from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
            is_committed_runtime_fact_source,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.role_bindings import (
            instantiate_pi_roles,
        )
        from praxist.task_spec_compat import migrate_legacy_gems_config

        self.assertTrue(frozen_prefix_hash({"scalar": 1}).startswith("sha256:"))
        self.assertFalse(
            is_committed_runtime_fact_source(
                {
                    "artifact_semantics": {
                        "role": "canonical_state",
                        "status": "committed",
                        "runtime_fact_source": False,
                    }
                }
            )
        )
        self.assertEqual(
            instantiate_pi_roles(
                ["unknown_role"],
                run_dir=Path("/tmp/run"),
                workspace=Path("/tmp/workspace"),
                model="model",
                max_runtime_minutes=1,
                mcp_servers=None,
                stop_check_fn=None,
                premium_mode=False,
            ),
            [],
        )
        self.assertEqual(migrate_legacy_gems_config(None), ({}, ()))


class GraphAndLocalStoreCoverageContractsTest(unittest.TestCase):
    def test_graph_rule_engine_builds_and_resolves_edges(self) -> None:
        older_uuid = "84d85198-d1b1-4290-a8c0-0b667a4ef593"
        newer_uuid = "94d85198-d1b1-4290-a8c0-0b667a4ef594"
        findings = [
            {
                "id": older_uuid,
                "timestamp": "2026-01-01T00:00:00",
                "peer_id": "gen0_peer0",
                "finding_type": "result",
                "title": "VARIANT-X baseline",
                "variant_name": "VARIANT-X alpha=0.1",
                "content": "original",
            },
            {
                "id": "fs_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "timestamp": "2026-01-01T00:01:00",
                "peer_id": "gen0_peer0",
                "finding_type": "insight",
                "title": "VARIANT-X insight",
                "variant_name": "VARIANT-X",
                "content": "confirmed and consistent with baseline",
            },
            {
                "id": newer_uuid,
                "timestamp": "2026-01-01T00:02:00",
                "peer_id": "gen0_peer1",
                "finding_type": "result",
                "title": "VARIANT-X followup",
                "variant_name": "VARIANT-X",
                "content": f"failed to reproduce {older_uuid}",
                "links": json.dumps(
                    [
                        {
                            "target_finding_id": older_uuid,
                            "edge_type": "supports",
                            "rationale": "agent intent wins",
                        },
                        {"target_finding_id": "missing", "edge_type": "junk"},
                    ]
                ),
            },
        ]
        builder = engine.FindingGraphBuilder(findings)

        self.assertEqual(builder._norm_variant("VARIANT-X alpha=0.3"), "variant-x")
        self.assertEqual(builder.chronological()[0]["id"], older_uuid)
        self.assertEqual(
            engine._extract_referenced_ids(
                f"see {older_uuid} and fs_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            )[0],
            older_uuid,
        )
        self.assertTrue(engine._has_any("confirmed", ("confirmed",)))
        self.assertFalse(engine._has_any_non_negated("not consistent with x", ("consistent with",)))

        edges = builder.propose_edges_for(findings[-1])
        edge_types = {edge["edge_type"] for edge in edges}
        self.assertIn("supports", edge_types)
        self.assertIn("related_to", edge_types)
        self.assertTrue(all(edge["src_finding_id"] == newer_uuid for edge in edges))
        self.assertGreaterEqual(len(builder.build_all_edges()), len(edges))
        self.assertEqual(engine._previous_generation_peer_id("gen3_peer2"), "gen2_peer2")
        self.assertIsNone(engine._previous_generation_peer_id("gen0_peer2"))
        self.assertNotIn("\n", engine._snippet("a\n```b```", 20))
        self.assertGreater(
            engine._score_edge_pair({"edge_type": "supports", "confidence": 0.8}, "p1", "p2"), 0.8
        )

    def test_local_store_graph_health_viz_and_maintainer_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {"LOCAL_STORE_DIR": tmp}
            with patch.dict(os.environ, env, clear=False):
                local_store.init_db()
                older = local_store.insert_finding(
                    {
                        "id": "old",
                        "finding_type": "result",
                        "title": "VARIANT-Y old",
                        "content": "baseline",
                        "metrics": {
                            "score": 0.5,
                            "gap": 0.2,
                            "tier": "T3",
                            "promotion_eligible": True,
                            "clean_promotion_eligible": True,
                            "is_smoke_eval": False,
                        },
                        "variant_name": "VARIANT-Z",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "timestamp": "2026-01-01T00:00:00",
                    }
                )
                newer = local_store.insert_finding(
                    {
                        "id": "new",
                        "finding_type": "result",
                        "title": "VARIANT-Y new",
                        "content": "confirmed improvement",
                        "metrics": {
                            "score": 0.8,
                            "gap": 0.1,
                            "tier": "T3",
                            "promotion_eligible": True,
                            "clean_promotion_eligible": True,
                            "is_smoke_eval": False,
                        },
                        "variant_name": "VARIANT-Y",
                        "peer_id": "gen1_peer0",
                        "generation_id": 1,
                        "timestamp": "2026-01-01T00:01:00",
                    }
                )
                metric_id = local_store.insert_metric({"run_id": "r", "metrics": {"score": 1}})
                self.assertGreater(metric_id, 0)
                self.assertEqual(local_store.count_findings(), 2)
                self.assertEqual(local_store.get_findings(peer_id="gen0_peer0")[0]["id"], older)

                edge_id = local_store.insert_edge(
                    {
                        "src_finding_id": newer,
                        "dst_finding_id": older,
                        "edge_type": "supports",
                        "confidence": 0.9,
                        "created_by": "test",
                        "provenance": {"rule": "manual"},
                    }
                )
                self.assertIsNotNone(edge_id)
                self.assertIsNone(
                    local_store.insert_edge(
                        {
                            "src_finding_id": newer,
                            "dst_finding_id": older,
                            "edge_type": "supports",
                            "confidence": 0.9,
                            "created_by": "test",
                        }
                    )
                )
                self.assertEqual(local_store.count_edges(), 1)
                self.assertEqual(local_store.edge_count_by_type()["supports"], 1)
                self.assertEqual(
                    local_store.get_edges_for_finding(newer, direction="out")[0]["dst_finding_id"],
                    older,
                )
                self.assertEqual(
                    local_store.get_subgraph(newer, max_depth=1)["nodes"][0]["graph_depth"], 0
                )
                self.assertEqual(
                    local_store.get_leaderboard(primary_metric="score")[0]["id"], newer
                )
                pareto = local_store.get_pareto_leaderboard(
                    "score",
                    "maximize",
                    [{"name": "gap", "direction": "minimize"}],
                    requires_tier=True,
                )
                self.assertEqual(pareto["n_total"], 2)
                self.assertGreaterEqual(pareto["n_pareto"], 1)

                engine.reset_graph_observability_state()
                maintainer = engine.FindingGraphMaintainer(Path(tmp), poll_interval=1)
                cycle = maintainer.sync_once()
                self.assertIn(cycle["status"], {"ok", "empty"})
                health = engine.write_graph_health(Path(tmp) / "graph")
                self.assertEqual(health["num_findings"], 2)
                context = engine.build_session_start_graph_context("gen1_peer0")
                self.assertIn("Graph-surfaced context", context)
                orientation = engine._render_orientation_context(2)
                self.assertIn("Graph-surfaced context", orientation)

                (Path(tmp) / "task_spec.yaml").write_text(
                    """
evaluation:
  primary_metric: score
  direction: maximize
  aux_metrics: [gap]
baselines:
  - {name: base, expected_acc: 0.5}
""",
                    encoding="utf-8",
                )
                payload = viz.build_viz_payload()
                self.assertEqual(payload["meta"]["num_findings"], 2)
                self.assertEqual(payload["meta"]["leaderboard"][0]["id"], newer)
                html_path = Path(tmp) / "graph" / "graph.html"
                viz.render_graph_html(html_path)
                self.assertIn("Finding Graph", html_path.read_text(encoding="utf-8"))

    def test_local_store_rejects_invalid_edges_and_batch_resolves_conflicts(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_STORE_DIR": tmp}, clear=False),
        ):
            local_store.init_db()
            for fid in ("a", "b"):
                local_store.insert_finding(
                    {"id": fid, "title": fid, "timestamp": f"2026-01-01T00:00:0{fid == 'b'}"}
                )
            with self.assertRaises(ValueError):
                local_store.insert_edge(
                    {
                        "src_finding_id": "b",
                        "dst_finding_id": "a",
                        "edge_type": "bad",
                        "confidence": 0.5,
                        "created_by": "test",
                    }
                )
            with self.assertRaises(ValueError):
                local_store.insert_edge(
                    {
                        "src_finding_id": "b",
                        "dst_finding_id": "a",
                        "edge_type": "supports",
                        "confidence": 2.0,
                        "created_by": "test",
                    }
                )
            inserted = local_store.insert_edges_batch(
                [
                    {
                        "edge_id": "weak",
                        "src_finding_id": "b",
                        "dst_finding_id": "a",
                        "edge_type": "supports",
                        "confidence": 0.7,
                        "created_by": "rule_engine",
                    },
                    {
                        "edge_id": "strong",
                        "src_finding_id": "b",
                        "dst_finding_id": "a",
                        "edge_type": "derived_from",
                        "confidence": 0.6,
                        "created_by": "rule_engine",
                    },
                    {
                        "edge_id": "agent",
                        "src_finding_id": "b",
                        "dst_finding_id": "a",
                        "edge_type": "challenges",
                        "confidence": 0.55,
                        "created_by": "agent_declared",
                    },
                    {"edge_type": "bad"},
                ]
            )
            self.assertEqual(inserted, 3)
            strong = [
                edge
                for edge in local_store.get_edges_for_finding("b", direction="out")
                if edge["edge_type"] in {"derived_from", "supports", "challenges"}
            ]
            self.assertEqual(strong[0]["edge_type"], "challenges")


class EvaluationToolsCoverageContractsTest(unittest.TestCase):
    @staticmethod
    def _payload(response: dict[str, object]) -> dict[str, object]:
        content = response["content"]  # type: ignore[index]
        text = content[0]["text"]  # type: ignore[index]
        return json.loads(text)

    def test_evaluation_tool_handlers_persist_metrics_findings_and_leaderboards(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir = root / "logs"
            findings_dir = root / "findings"
            env = {
                "LOCAL_MODE": "true",
                "LOCAL_STORE_DIR": str(root),
                "LOCAL_FINDINGS_DIR": str(findings_dir),
                "LOGS_DIR": str(logs_dir),
                "GENERATION_ID": "9",
                "PRIMARY_METRIC": "score",
                "METRIC_DIRECTION": "maximize",
                "ANCHOR_METRICS": json.dumps([{"name": "gap", "direction": "minimize"}]),
                "REQUIRES_TIER": "true",
            }
            with patch.dict(os.environ, env, clear=False):
                metrics = asyncio.run(
                    adapter._handle_log_experiment_metrics(
                        {
                            "run_id": "run",
                            "variant_name": "A",
                            "metrics": '{"score": 0.4}',
                            "notes": "note",
                            "step": 3,
                            "peer_id": "Gen2_Peer3",
                        }
                    )
                )
                self.assertEqual(self._payload(metrics)["status"], "recorded")
                self.assertTrue((logs_dir / "metrics_log.jsonl").exists())

                invalid_metrics = asyncio.run(
                    adapter._handle_log_experiment_metrics(
                        {
                            "run_id": "run",
                            "variant_name": "A",
                            "metrics": "{bad",
                            "peer_id": "gen0_peer0",
                        }
                    )
                )
                self.assertTrue(invalid_metrics.get("is_error"))

                bad_type = asyncio.run(
                    adapter._handle_share_finding(
                        {
                            "finding_type": "bad",
                            "title": "bad",
                            "content": "bad",
                        }
                    )
                )
                self.assertTrue(bad_type.get("is_error"))

                first = asyncio.run(
                    adapter._handle_share_finding(
                        {
                            "finding_type": "result",
                            "title": "Variant A result",
                            "content": "scored result",
                            "metrics": json.dumps(
                                {
                                    "score": 0.5,
                                    "gap": 0.2,
                                    "tier": "T3",
                                    "promotion_eligible": True,
                                    "clean_promotion_eligible": True,
                                    "is_smoke_eval": False,
                                }
                            ),
                            "variant_name": "A",
                            "notes": "short",
                            "peer_id": "gen2_peer0",
                            "design_dimensions": json.dumps({"mechanism": "a"}),
                            "extra": json.dumps({"peer_role": "exploit"}),
                        }
                    )
                )
                first_id = self._payload(first)["finding_id"]
                second = asyncio.run(
                    adapter._handle_share_finding(
                        {
                            "finding_type": "insight",
                            "title": "Variant B insight",
                            "content": "confirmed and supports previous",
                            "metrics": json.dumps(
                                {
                                    "score": 0.7,
                                    "gap": 0.1,
                                    "tier": "T3",
                                    "promotion_eligible": "true",
                                    "clean_promotion_eligible": "true",
                                    "is_smoke_eval": False,
                                }
                            ),
                            "variant_name": "B",
                            "notes": "short",
                            "peer_id": "gen2_peer1",
                            "links": json.dumps(
                                [
                                    {
                                        "target_finding_id": first_id,
                                        "edge_type": "supports",
                                        "rationale": "replicated",
                                    }
                                ]
                            ),
                            "design_dimensions": "{bad",
                            "extra": "not-json",
                        }
                    )
                )
                self.assertEqual(self._payload(second)["type"], "insight")
                self.assertEqual(local_store.count_findings(), 2)
                self.assertGreaterEqual(local_store.count_edges(), 1)

                pareto = self._payload(
                    asyncio.run(adapter._handle_get_leaderboard({"generation": "bad", "top_k": -1}))
                )
                self.assertEqual(pareto["mode"], "pareto")
                self.assertEqual(pareto["n_total"], 2)

                with patch.dict(os.environ, {"ANCHOR_METRICS": ""}, clear=False):
                    single = self._payload(
                        asyncio.run(adapter._handle_get_leaderboard({"generation": 2, "top_k": 1}))
                    )
                self.assertEqual(single["mode"], "single_metric")
                self.assertEqual(len(single["entries"]), 1)

    def test_wait_for_file_and_filesystem_leaderboard_cover_degraded_paths(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = root / "ready.json"
            ready.write_text("prefix\nDONE\n", encoding="utf-8")
            findings = root / "findings"
            findings.mkdir()
            (findings / "f.json").write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "title": "fallback",
                        "metrics": {"score": 0.9, "tier": "T3"},
                        "variant_name": "fallback",
                        "generation_id": 0,
                    }
                ),
                encoding="utf-8",
            )
            env = {
                "LOCAL_STORE_DIR": str(root),
                "LOCAL_FINDINGS_DIR": str(findings),
                "PRIMARY_METRIC": "score",
                "METRIC_DIRECTION": "maximize",
            }
            with patch.dict(os.environ, env, clear=False):
                missing = asyncio.run(adapter._handle_wait_for_file_impl({}))
                self.assertTrue(missing.get("is_error"))

                unsafe = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {"path": "/etc/passwd", "timeout_seconds": 1}
                    )
                )
                self.assertTrue(unsafe.get("is_error"))

                ready_payload = self._payload(
                    asyncio.run(
                        adapter._handle_wait_for_file_impl(
                            {
                                "path": f"{ready},{ready}",
                                "timeout_seconds": 1,
                                "poll_interval_seconds": 2,
                                "min_bytes": 1,
                                "contains_text": "DONE",
                                "mode": "all",
                            }
                        )
                    )
                )
                self.assertEqual(ready_payload["status"], "ready")
                self.assertEqual(ready_payload["deduped_count"], 1)

                too_many = asyncio.run(
                    adapter._handle_wait_for_file_impl(
                        {
                            "path": ",".join(str(root / f"f{i}") for i in range(33)),
                            "timeout_seconds": 1,
                        }
                    )
                )
                self.assertTrue(too_many.get("is_error"))

                self.assertEqual(adapter._parse_anchor_metrics_env(), [])
                with patch.dict(
                    os.environ,
                    {"ANCHOR_METRICS": '[{"name":"gap","direction":"bad"}]'},
                    clear=False,
                ):
                    self.assertEqual(adapter._parse_anchor_metrics_env(), [])
                fallback = json.loads(adapter._filesystem_leaderboard(0, 5))
                self.assertEqual(fallback["mode"], "filesystem_fallback")
                self.assertEqual(fallback["entries"][0]["variant_name"], "fallback")
                plugin = adapter.create_tool_plugin()
                self.assertIn("wait_for_file", plugin["tool_names"])
                with (
                    patch.object(adapter, "create_sdk_mcp_server", None),
                    patch.object(adapter, "tool", None),
                    self.assertRaises(ImportError),
                ):
                    adapter.create_evaluation_tools_server()


class ResearchMemoryCoverageContractsTest(unittest.TestCase):
    def test_ledgers_cards_memory_tools_and_evidence_pack_are_queryable(self) -> None:
        from praxist.plugins.tools.memory_tools import adapter as memory_tools
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
            _detect_negative,
            _evidence_id,
            _safe_get_metric,
            build_card_from_finding,
            build_cards_from_db,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.context_auditor import (
            _has_overclaim_language,
            _has_source_id,
            audit_agenda,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _role_filter,
            _sanitize_value,
            build_evidence_pack,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
            LedgerEntry,
            LedgerStore,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.claim_ledger import (
            ClaimLedger,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.coverage_matrix import (
            CoverageMatrix,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.dissent_ledger import (
            DissentLedger,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.frontier_delta_ledger import (
            FrontierDeltaLedger,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.negative_evidence_ledger import (
            NegativeEvidenceLedger,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.retired_claim_ledger import (
            RetiredClaimLedger,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.role_roi_ledger import (
            RoleROILedger,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "finding_a",
                        "finding_type": "result",
                        "title": "Variant A champion",
                        "content": "positive",
                        "metrics": {
                            "mean_test_accuracy": 0.8,
                            "mean_train_test_gap": 0.1,
                            "promotion_eligible": True,
                            "tier": "T3",
                            "seed_count": 5,
                        },
                        "variant_name": "A",
                        "notes": "scaling synergy",
                        "peer_id": "gen1_peer0",
                        "generation_id": 1,
                        "timestamp": "2026-01-01T00:00:00",
                        "extra": {"peer_role": "exploit"},
                    }
                )
                local_store.insert_finding(
                    {
                        "id": "finding_b",
                        "finding_type": "challenge",
                        "title": "Variant B failed",
                        "content": "negative",
                        "metrics": {
                            "mean_test_accuracy": 0.4,
                            "promotion_eligible": False,
                            "tier": "T3",
                        },
                        "variant_name": "B",
                        "notes": "did not improve baseline",
                        "peer_id": "gen1_peer1",
                        "generation_id": 1,
                        "timestamp": "2026-01-01T00:01:00",
                        "extra": {"is_negative": True, "peer_role": "falsifier"},
                    }
                )

                raw_card = build_card_from_finding(local_store.get_findings()[0], run_dir)
                self.assertTrue(raw_card["evidence_id"].startswith("E_"))
                self.assertEqual(_safe_get_metric({"x": {"mean": "0.5"}}, "x"), 0.5)
                self.assertIsNone(_safe_get_metric({"x": float("nan")}, "x"))
                self.assertTrue(_detect_negative({"title": "kill result", "extra": "{}"}))
                self.assertNotEqual(
                    _evidence_id("", 0, "gen0_peer0", content_seed="x"),
                    _evidence_id("", 0, "gen0_peer0", content_seed="y"),
                )
                self.assertEqual(len(build_cards_from_db(run_dir, only_gen=1)), 2)

                store = LedgerStore(
                    run_dir / "research_memory" / "ledgers" / "custom.yaml", "custom"
                )
                created = store.upsert("entry", {"a": 1}, created_by="test")
                updated = store.upsert("entry", {"b": 2}, created_by="test", action="update")
                self.assertEqual(created.id, updated.id)
                self.assertEqual(store.get("entry").data["b"], 2)
                appended = store.append_only("append", {"x": True})
                self.assertEqual(LedgerEntry.from_dict(appended.to_dict()).id, "append")
                with self.assertRaises(ValueError):
                    store.append_only("append", {})
                self.assertEqual(len(store.filter(lambda entry: entry.id.startswith("e"))), 1)
                self.assertEqual(len(store), 2)

                claims = ClaimLedger(run_dir)
                claims.upsert_claim(
                    "C1",
                    "Universal champion claim",
                    "active",
                    0.8,
                    supports=[raw_card["evidence_id"]],
                    missing_tests=["control"],
                )
                claims.upsert_claim(
                    "C2",
                    "Retired claim",
                    "retired",
                    0.2,
                    boundary="only small tasks",
                    revive_if=["new seed"],
                )
                self.assertEqual(len(claims.list_active()), 1)
                self.assertEqual(len(claims.list_recently_killed()), 1)
                with self.assertRaises(ValueError):
                    claims.upsert_claim("bad", "bad", "unknown", 0.1)
                with self.assertRaises(ValueError):
                    claims.upsert_claim("bad", "bad", "active", 1.5)

                coverage = CoverageMatrix(run_dir)
                coverage.record_grid_point("A", "rho", 0.1, seed_count=2, source_evidence_id="E1")
                coverage.record_grid_point("A", "rho", 0.2, seed_count=3, source_evidence_id="E2")
                coverage.record_bridge_point("A", "B", "mix", "mid", source_evidence_id="E3")
                self.assertTrue(coverage.query_grid("A", "rho")["values_tested"])
                self.assertTrue(coverage.is_bridge_covered("B", "A", "mix"))
                self.assertEqual(coverage.query_bridge("A", "B", "mix")["variant_pair"], ["A", "B"])

                dissent = DissentLedger(run_dir)
                dissent.add(
                    "D1",
                    "C1",
                    {"skeptic": "needs control"},
                    resolving_experiment="run control",
                    decision_rule={"metric": "score"},
                )
                dissent.update_status("D1", "experiment_assigned")
                self.assertEqual(len(dissent.list_open()), 1)
                neg = NegativeEvidenceLedger(run_dir)
                neg.add("N1", "failed", "control", summary="control failed")
                retired = RetiredClaimLedger(run_dir)
                retired.retire("C2", "Retired claim", "obsolete", "small only", ["new evidence"])
                frontier_delta = FrontierDeltaLedger(run_dir)
                frontier_delta.record_promote(
                    generation_id=1,
                    axis="mean_test_accuracy",
                    current_anchor={"variant": "A", "value": 0.8},
                    previous_anchor={"variant": "B", "value": 0.6},
                )
                role_roi = RoleROILedger(run_dir)
                role_roi.record_gen_summary(1, {"exploit": {"accepted": 1}})

                self.assertEqual(memory_tools.list_active_claims(run_dir)[0]["id"], "C1")
                self.assertEqual(memory_tools.list_open_objections(run_dir)[0]["id"], "D1")
                self.assertTrue(
                    memory_tools.query_coverage_matrix(
                        run_dir, variant_family="A", parameter="rho"
                    )["covered"]
                )
                self.assertTrue(
                    memory_tools.query_coverage_matrix(
                        run_dir,
                        bridge_pair=["A", "B"],
                        bridge_dimension="mix",
                    )["covered"]
                )
                self.assertIn("error", memory_tools.query_coverage_matrix(run_dir))
                self.assertEqual(
                    memory_tools.get_ledger_entry(run_dir, "claim_ledger", "C1")["id"],
                    "C1",
                )
                self.assertIn("error", memory_tools.get_ledger_entry(run_dir, "bad", "x"))
                self.assertIn(
                    "error", memory_tools.get_ledger_entry(run_dir, "claim_ledger", "missing")
                )
                cards = memory_tools.query_evidence_cards(
                    run_dir,
                    mechanism="champion",
                    peer_id="gen1_peer0",
                    generation_id=1,
                    is_negative=False,
                )
                self.assertEqual(cards[0]["peer_id"], "gen1_peer0")
                self.assertIn(
                    "error",
                    memory_tools.get_evidence_card(run_dir, "missing"),
                )

                pack = build_evidence_pack(
                    run_dir,
                    panel_mode="high_stakes",
                    current_gen_id=1,
                    target_decisions=["next"],
                    pi_roles=["builder", "skeptic", "portfolio", "external_validity"],
                    max_cards_total=4,
                    max_cards_per_pack=3,
                    findings_summary={"count": 2},
                )
                self.assertTrue(pack.pack_id.startswith("EP::"))
                self.assertIn("builder", pack.private_packs)
                self.assertIsNone(_sanitize_value(float("inf")))
                self.assertIn("{\u200b{", _sanitize_value("{{ x }}"))
                self.assertGreater(_role_filter("skeptic", raw_card), 0)

                agenda = {
                    "consensus_actions": [{"id": "A1", "claim_or_hypothesis": "universal win"}],
                    "retired_claims": [{"id": "C2", "boundary": "", "revive_if": []}],
                    "peer_contracts": {"gen2_peer0": {"role": "Bridge"}},
                }
                report = audit_agenda(
                    agenda,
                    {"private_packs": {"skeptic": pack.all_cards}},
                    {"builder": {}},
                    completed_gen_id=1,
                )
                self.assertFalse(report.pass_)
                self.assertFalse(_has_source_id({"supports": [""]}))
                self.assertTrue(_has_source_id({"supports": ["E1"]}))
                self.assertIn("universal", _has_overclaim_language("Universal result"))
                plugin = memory_tools.create_tool_plugin()
                self.assertEqual(plugin["server_name"], "memory-tools")
                with (
                    patch.object(memory_tools, "create_sdk_mcp_server", None),
                    patch.object(memory_tools, "tool", None),
                    self.assertRaises(ImportError),
                ):
                    memory_tools.create_memory_tools_server(run_dir)


class PIAgentCoverageContractsTest(unittest.TestCase):
    @staticmethod
    def _agenda(next_gen_id: int, cohort_size: int = 5) -> dict[str, object]:
        roles = ["exploit", "falsifier", "bridge", "anti_mainline"]
        return {
            "generation": next_gen_id,
            "mainline_observation": {"main_risk": "control"},
            "cross_peer_hypotheses": [
                {
                    "id": f"H_g{next_gen_id}_01",
                    "claim": "testable claim",
                    "minimal_test": "run control",
                    "kill_condition": "control wins",
                    "promote_condition": "claim survives",
                    "source_findings": ["finding_a"],
                }
            ],
            "bridge_hypothesis": {"id": f"B_g{next_gen_id}_01"},
            "anti_mainline_contract": {"target_axes": ["non-mainline"]},
            "falsification_contract": {"target_hypothesis": f"H_g{next_gen_id}_01"},
            "peer_contracts": {
                f"gen{next_gen_id}_peer{i}": {
                    "role": roles[i % len(roles)],
                    "target_hypothesis": f"H_g{next_gen_id}_01",
                    "success_signal": "valid result",
                }
                for i in range(cohort_size)
            },
        }

    def test_pi_agent_loads_state_validates_agenda_and_runs_with_fake_agent(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import pi_agent

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            template = Path(tmp) / "pi_template.jinja2"
            template.write_text(
                "completed={{ completed_gen_id }} next={{ next_gen_id }} "
                "findings={{ n_findings }} edges={{ n_edges }} output={{ agenda_output_path }}",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "finding_a",
                        "finding_type": "result",
                        "title": "result",
                        "content": "content",
                        "metrics": {"score": 1.0},
                        "variant_name": "A",
                        "peer_id": "gen1_peer0",
                        "generation_id": 1,
                        "timestamp": "2026-01-01T00:00:00",
                    }
                )
                local_store.insert_finding(
                    {
                        "id": "finding_old",
                        "finding_type": "insight",
                        "title": "old",
                        "content": "content",
                        "metrics": {"score": 0.5},
                        "variant_name": "Old",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "timestamp": "2026-01-01T00:00:00",
                    }
                )
                local_store.insert_edge(
                    {
                        "src_finding_id": "finding_a",
                        "dst_finding_id": "finding_old",
                        "edge_type": "supports",
                        "confidence": 0.8,
                        "created_by": "test",
                        "rationale": "supports",
                    }
                )

                frontier_dir = run_dir / "frontier"
                frontier_dir.mkdir()
                (frontier_dir / "frontier_manifest.json").write_text(
                    json.dumps(
                        {
                            "generations": {"0": [{"finding_id": "finding_old"}]},
                            "cumulative_top": [
                                {
                                    "finding_id": "finding_a",
                                    "variant_name": "A",
                                    "metrics": {"bad": float("nan"), "ok": 1.0},
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                agendas = run_dir / pi_agent.AGENDAS_DIRNAME
                agendas.mkdir()
                (agendas / pi_agent.AGENDA_FILE_PATTERN.format(1)).write_text(
                    "```yaml\n" + yaml_dump(self._agenda(1)) + "```\n",
                    encoding="utf-8",
                )

                pi = pi_agent.PIAgent(
                    run_dir=run_dir,
                    workspace=Path(tmp),
                    cohort_size=5,
                    model="fake-model",
                    max_runtime_minutes=1,
                    prompt_template_path=template,
                    mcp_servers={"evaluation-tools": object()},
                )
                self.assertEqual(pi.expected_peer_ids(2)[0], "gen2_peer0")
                self.assertEqual(pi._normalize_role("Anti-Mainline"), "anti_mainline")
                self.assertEqual(len(pi._load_gen_findings(1)), 1)
                self.assertEqual(len(pi._load_gen_edges(1)), 1)
                self.assertIsNone(pi._sanitize_json_value(float("nan")))
                self.assertEqual(
                    pi._trim_prior_metrics({"score": 1, "tier": "T3"}),
                    {"score": 1, "tier": "T3"},
                )
                self.assertEqual(pi._load_prior_agenda(1)["generation"], 1)
                self.assertEqual(pi._load_prior_agendas_summary(2)[0]["generation"], 1)
                self.assertTrue(pi._load_prior_findings_summary(1))
                self.assertEqual(
                    pi._build_findings_summary_for_panel(1)["total_since_last_synthesis"], 1
                )

                valid = self._agenda(2)
                self.assertIsNone(pi.validate_agenda(valid, 2))
                self.assertEqual(valid["cross_peer_hypotheses"][0]["id"], "H_g2_01")
                self.assertIn("missing top-level", pi.validate_agenda({}, 2))
                self.assertIn(
                    "cannot be parsed",
                    pi.validate_agenda({**self._agenda(2), "generation": "bad"}, 2),
                )
                self.assertIn(
                    "exactly cohort_size",
                    pi.validate_agenda({**self._agenda(2), "peer_contracts": {}}, 2),
                )
                placeholder = self._agenda(2)
                placeholder["cross_peer_hypotheses"][0]["claim"] = "<one paragraph>"
                self.assertIn("placeholder", pi.validate_agenda(placeholder, 2))

                class FakeBaseAgent:
                    def __init__(self, **kwargs):
                        self.kwargs = kwargs

                    async def execute(self, task: str):
                        out_path = run_dir / "agendas" / pi_agent.AGENDA_FILE_PATTERN.format(2)
                        out_path = out_path.with_suffix(out_path.suffix + ".candidate")
                        out_path = (
                            run_dir / "peer_workspaces" / self.kwargs["request_id"] / out_path.name
                        )
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(
                            yaml_dump(PIAgentCoverageContractsTest._agenda(2)), encoding="utf-8"
                        )
                        return SimpleNamespace(
                            success=True,
                            error=None,
                            request_id=self.kwargs["request_id"],
                        )

                with patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                    FakeBaseAgent,
                ):
                    result = asyncio.run(pi.run(1))

                self.assertTrue(result.success)
                self.assertEqual(result.next_gen_id, 2)
                self.assertTrue(result.agenda_path and result.agenda_path.exists())
                self.assertTrue((agendas / "pi_prompt_for_gen2.md").exists())

    def test_pi_agent_handles_invalid_single_pi_and_multi_pi_fallback_paths(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import pi_agent
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import PanelResult

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            template = Path(tmp) / "pi_template.jinja2"
            template.write_text("output={{ agenda_output_path }}", encoding="utf-8")

            class BadAgendaAgent:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                async def execute(self, task: str):
                    out_path = run_dir / "agendas" / pi_agent.AGENDA_FILE_PATTERN.format(1)
                    out_path = out_path.with_suffix(out_path.suffix + ".candidate")
                    out_path = (
                        run_dir / "peer_workspaces" / self.kwargs["request_id"] / out_path.name
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(
                        "generation: 99\ncross_peer_hypotheses: []\npeer_contracts: {}\n",
                        encoding="utf-8",
                    )
                    return SimpleNamespace(
                        success=True,
                        error=None,
                        request_id=self.kwargs["request_id"],
                    )

            pi = pi_agent.PIAgent(
                run_dir=run_dir,
                workspace=Path(tmp),
                cohort_size=5,
                model="fake",
                max_runtime_minutes=1,
                prompt_template_path=template,
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                BadAgendaAgent,
            ):
                invalid = asyncio.run(pi.run(0))
            self.assertFalse(invalid.success)
            self.assertTrue((run_dir / "agendas" / "research_agenda_gen1.yaml.rejected").exists())

            class MultiPiConfig:
                panel_mode_default = "full"
                auto_escalate_to_high_stakes = False
                pi_max_runtime_minutes = 1
                chair_max_runtime_minutes = 1
                n_rounds = 1
                round2_max_runtime_minutes = 1
                fallback_to_single_pi_on_panel_failure = False

            async def fake_panel_success(**kwargs):
                return PanelResult(success=True, panel_mode="full", agenda=self._agenda(1))

            async def fake_panel_failure(**kwargs):
                return PanelResult(success=False, panel_mode="full", error="panel failed")

            pi_panel = pi_agent.PIAgent(
                run_dir=run_dir,
                workspace=Path(tmp),
                cohort_size=5,
                model="fake",
                use_multi_pi_panel=True,
                multi_pi_config=MultiPiConfig(),
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.multi_pi.run_panel",
                fake_panel_success,
            ):
                panel_result = asyncio.run(pi_panel.run(0))
            self.assertTrue(panel_result.success)

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.multi_pi.run_panel",
                fake_panel_failure,
            ):
                panel_failure = asyncio.run(pi_panel.run(0))
            self.assertFalse(panel_failure.success)
            self.assertIn("panel failed", panel_failure.error or "")
            self.assertFalse(
                (run_dir / "agendas" / pi_agent.AGENDA_FILE_PATTERN.format(1)).exists()
            )

            bad_agenda_path = run_dir / "agendas" / pi_agent.AGENDA_FILE_PATTERN.format(3)
            bad_agenda_path.write_text("peer_contracts: []\n", encoding="utf-8")
            self.assertIsNone(pi_agent.load_agenda_for_gen(run_dir, 3))
            good_agenda_path = run_dir / "agendas" / pi_agent.AGENDA_FILE_PATTERN.format(4)
            good_agenda_path.write_text(yaml_dump(self._agenda(4)), encoding="utf-8")
            self.assertEqual(pi_agent.load_agenda_for_gen(run_dir, 4)["generation"], 4)


def yaml_dump(data: object) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


class GenerationLoopCoverageContractsTest(unittest.TestCase):
    def test_generation_loop_run_success_and_error_paths_are_offline(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
            write_boundary_marker,
        )

        async def fake_complete(loop, *, gen_id, pi_agent, pi_cfg, generation_results):
            self.assertEqual(generation_results, [{"generation": gen_id}])
            loop.frontier.promote(
                gen_id,
                [
                    {
                        "id": f"f{gen_id}",
                        "finding_id": f"f{gen_id}",
                        "finding_type": "result",
                        "variant_name": f"v{gen_id}",
                        "metrics": {
                            "score": 0.5 + gen_id,
                            "tier": "T3",
                            "scored_complete": True,
                            "effort_ratio": 1.0,
                            "coverage_ratio": 1.0,
                        },
                    }
                ],
            )
            write_boundary_marker(
                loop.run_dir,
                gen_id=gen_id,
                promoted_count=1,
                pi_status="test_committed",
            )

        async def fake_generation(loop, gen_id):
            return [{"generation": gen_id}]

        with tempfile.TemporaryDirectory() as tmp:
            spec = load_task_spec("templates/tasks/toy_math/task.yaml")
            spec = replace(
                spec,
                evaluation=replace(
                    spec.evaluation,
                    primary_metric="score",
                    direction="maximize",
                    frontier_lanes=[],
                ),
                generation_policy=replace(spec.generation_policy, max_generations=2, cohort_size=1),
                pi_agent=replace(spec.pi_agent, enabled=False),
                synthesis_trigger=replace(spec.synthesis_trigger, max_interval_minutes=1),
            )
            run_dir = Path(tmp) / "run"
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.run_generation_cohort",
                    fake_generation,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.complete_generation_boundary",
                    fake_complete,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.configure_runtime_environment"
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.initialize_local_store_if_needed"
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.validate_baseline_cache_for_run"
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.start_sidecars"
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.sidecars.stop_sidecars"
                ),
            ):
                loop = GenerationLoop(
                    task_spec=spec,
                    workspace=Path(tmp),
                    run_dir=run_dir,
                    local_mode=True,
                    tool_server_refs=[],
                )
                self.assertEqual(loop._strategy_for_gen(0), "explore")
                self.assertEqual(loop._strategy_for_gen(1), "pi_directed")
                summary = asyncio.run(loop.run())

            self.assertEqual(summary["generations_completed"], 2)
            self.assertEqual(summary["exit_condition"], "max_generations")
            self.assertEqual(summary["status"], "succeeded")
            self.assertEqual(summary["exit_code"], 0)
            self.assertEqual(summary["stop_reason"], "max_generations")
            self.assertFalse((run_dir / "orchestrator.lock").exists())
            self.assertTrue((run_dir / "run_summary.json").exists())
            self.assertTrue((run_dir / "run_stop_report.json").exists())
            self.assertGreaterEqual(len(summary["frontier_summary"]), 1)

        async def boom_generation(loop, gen_id):
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            spec = load_task_spec("templates/tasks/toy_math/task.yaml")
            spec = replace(
                spec,
                evaluation=replace(spec.evaluation, primary_metric="score"),
                generation_policy=replace(spec.generation_policy, max_generations=1, cohort_size=1),
                pi_agent=replace(spec.pi_agent, enabled=False),
            )
            run_dir = Path(tmp) / "run_error"
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.run_generation_cohort",
                    boom_generation,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.configure_runtime_environment"
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.initialize_local_store_if_needed"
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.validate_baseline_cache_for_run"
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.start_sidecars"
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.sidecars.stop_sidecars"
                ),
            ):
                loop = GenerationLoop(
                    task_spec=spec,
                    workspace=Path(tmp),
                    run_dir=run_dir,
                    local_mode=True,
                    tool_server_refs=[],
                )
                with self.assertRaises(RuntimeError):
                    asyncio.run(loop.run())
            error_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(error_summary["exit_condition"], "error")
            self.assertEqual(error_summary["error_type"], "RuntimeError")


class MultiPIPanelCoverageContractsTest(unittest.TestCase):
    def test_base_pi_loads_kb_runs_round1_and_round2_with_tolerant_yaml(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"
            prompt_template_name = "base.jinja2"

            def fixed_questions(self) -> list[str]:
                return ["What is the strongest claim?"]

        class FakeAgentResult:
            success = True
            error = None

            def __init__(self, text: str):
                self.output = {"text_outputs": [text]}

        class FakeBaseAgent:
            calls: list[dict[str, object]] = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                FakeBaseAgent.calls.append(kwargs)

            async def execute(self, task: str) -> FakeAgentResult:
                self.task = task
                if self.kwargs["name"].endswith("round2"):
                    return FakeAgentResult(
                        "preamble\n```yaml\n"
                        "role: tester\n"
                        "round: 2\n"
                        "strongest_agreement:\n"
                        '  peer_label: "PI #A"\n'
                        "  claim_id: peer\n"
                        "  why: useful\n"
                        "strongest_objection:\n"
                        '  peer_label: "PI #A"\n'
                        "  claim_id: peer\n"
                        "  objection: weak evidence\n"
                        "  proposed_kill_test: rerun\n"
                        "missing_experiment:\n"
                        "  description: ablation\n"
                        "  why_critical: isolate mechanism\n"
                        "private_kb_revealed_blind_spot:\n"
                        "  triggered: false\n"
                        "  peer_label: null\n"
                        "  blind_spot: none\n"
                        "claim_that_should_be_downgraded:\n"
                        "  claim_id: c1\n"
                        "  current_language: useful\n"
                        "  recommended_language: bounded useful\n"
                        "  reason: needs validation\n"
                        "singleton_high_upside_idea_to_preserve:\n"
                        "  source: self\n"
                        "  peer_label: null\n"
                        "  idea_summary: preserve weak signal\n"
                        "  protected_budget_recommendation: 1 peer\n"
                        "own_revisions:\n"
                        "  - claim_id: c1\n    revision: keep\n```"
                    )
                return FakeAgentResult(
                    "Now the memo follows\nrole: tester\n"
                    "top_claims:\n  - id: c1\n    claim: useful\n"
                    "private_knowledge_used:\n  - kb1\n"
                )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            kb_dir = workspace / "kb"
            kb_dir.mkdir()
            (kb_dir / "kb1.md").write_text(
                "accuracy generalization optimizer evidence", encoding="utf-8"
            )
            (kb_dir / "ignored.txt").write_text("ignored", encoding="utf-8")
            pi = TestPI(
                run_dir=workspace / "run",
                workspace=workspace,
                model="fake-model",
                mcp_servers={"memory-tools": object()},
                premium_mode=True,
                reasoning_effort="high",
            )
            pi._private_kb_path = lambda: kb_dir  # type: ignore[method-assign]

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                FakeBaseAgent,
            ):
                memo = asyncio.run(
                    pi.run(
                        shared_core={"claim": "accuracy evidence"},
                        private_pack=[{"id": "p1"}],
                        target_decisions=["choose next experiment"],
                    )
                )
                round2 = asyncio.run(
                    pi.run_cross_review(
                        own_memo=memo.parsed,
                        anon_peers={"PI #A": {"top_claims": [{"id": "peer"}]}},
                        round2_max_runtime_minutes=1,
                    )
                )

        self.assertTrue(memo.success)
        self.assertEqual(memo.parsed["role"], "tester")
        self.assertEqual(memo.private_kb_used, ["kb1"])
        self.assertTrue(round2.success)
        self.assertEqual(round2.parsed["round"], 2)
        self.assertEqual(FakeBaseAgent.calls[0]["reasoning_effort"], "high")
        self.assertEqual(FakeBaseAgent.calls[1]["reasoning_effort"], "high")
        self.assertIn(
            "mcp__memory-tools__query_evidence_cards",
            FakeBaseAgent.calls[0]["allowed_tools"],
        )
        for raw_tool in ("Read", "Bash", "Glob", "Grep"):
            self.assertNotIn(raw_tool, FakeBaseAgent.calls[0]["allowed_tools"])
            self.assertNotIn(raw_tool, FakeBaseAgent.calls[1]["allowed_tools"])

    def test_base_pi_failure_paths_return_structured_unavailable_memos(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class TestPI(_base_pi.BasePI):
            role_name = "tester"

        class FailedAgentResult:
            success = False
            error = "model failed"
            output: dict[str, object] = {}

        class FailedBaseAgent:
            def __init__(self, **_kwargs):
                pass

            async def execute(self, task: str) -> FailedAgentResult:
                return FailedAgentResult()

        with tempfile.TemporaryDirectory() as tmp:
            pi = TestPI(run_dir=Path(tmp) / "run", workspace=Path(tmp), model="fake-model")
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.BaseAgent",
                FailedBaseAgent,
            ):
                memo = asyncio.run(pi.run({}, [], []))
                round2 = asyncio.run(pi.run_cross_review({}, {}, round2_max_runtime_minutes=1))

        self.assertFalse(memo.success)
        self.assertEqual(memo.parsed["error"], "model failed")
        self.assertFalse(round2.success)
        self.assertTrue(round2.parsed["_round2_failed"])

    def test_two_round_panel_persists_evidence_memos_agenda_and_audit(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor as panel,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
            _base_pi,
        )

        class FakePI:
            def __init__(self, role_name: str):
                self.role_name = role_name

            async def run(self, shared_core, private_pack, target_decisions):
                if self.role_name == "external_validity":
                    raise RuntimeError("offline PI unavailable")
                parsed = {
                    "role": self.role_name,
                    "top_claims": [{"id": f"{self.role_name}_claim", "claim": "works"}],
                    "objections_or_warnings": [],
                    "proposed_experiments": [],
                    "proposed_peer_contracts": [],
                    "private_knowledge_used": [],
                }
                if self.role_name == "skeptic":
                    parsed["objections_or_warnings"] = [
                        {
                            "severity": "blocking",
                            "target_claim": "builder_claim",
                            "objection": "needs replication",
                            "resolving_experiment": "replicate",
                        }
                    ]
                return _base_pi.PIMemo(
                    role=self.role_name,
                    raw_text="raw",
                    parsed=parsed,
                    private_kb_used=[],
                    success=True,
                )

            async def run_cross_review(self, own_memo, anon_peers, round2_max_runtime_minutes):
                return _base_pi.PIMemo(
                    role=self.role_name,
                    raw_text="round2",
                    parsed={
                        "own_revisions": [
                            {"claim_id": own_memo["top_claims"][0]["id"], "revision": "keep"},
                            {"claim_id": "hallucinated", "revision": "drop"},
                        ]
                    },
                    private_kb_used=[],
                    success=True,
                )

        class FakeChair:
            calls: list[dict[str, object]] = []
            init_calls: list[dict[str, object]] = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.__class__.init_calls.append(dict(kwargs))

            async def run(self, **kwargs):
                self.kwargs.update(kwargs)
                self.__class__.calls.append(dict(kwargs))
                attempt = len(self.__class__.calls)
                return SimpleNamespace(
                    success=True,
                    agenda={
                        "agenda_version": "2.0",
                        "attempt": attempt,
                        "peer_contracts": [],
                    },
                    raw_text=f"agenda yaml {attempt}",
                    error=None,
                )

        pack = SimpleNamespace(
            shared_core={"shared_core_id": "abcdef12", "claim": "dominates prior result"},
            all_cards=[{"id": "card1"}],
            audit={"negative_evidence_ratio_global": 0.25},
        )
        validation_missing_role = SimpleNamespace(
            valid=False,
            blocking_issues=["peer_contracts missing roles: ['peer_generalist'] in full panel"],
            warnings=[],
        )
        validation = SimpleNamespace(valid=True, blocking_issues=[], warnings=["soft"])
        audit = SimpleNamespace(
            pass_=True,
            blocking_issues=[],
            warnings=["audit warning"],
            metrics={"citation_coverage": 0.8},
        )
        specs = [
            SimpleNamespace(legacy_role_id=r) for r in ("builder", "skeptic", "external_validity")
        ]

        def fake_instantiate_pi_roles(
            roles,
            *,
            run_dir,
            workspace,
            model,
            max_runtime_minutes,
            mcp_servers,
            stop_check_fn,
            premium_mode,
            reasoning_effort,
            prompts_dir=None,
            task_project_path=None,
            plugin_registry=None,
        ):
            self.assertEqual(
                [role.legacy_role_id for role in roles], [role.legacy_role_id for role in specs]
            )
            self.assertEqual(reasoning_effort, "high")
            return [FakePI(role.legacy_role_id) for role in roles]

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            with (
                patch.object(panel, "_select_pi_roles", return_value=["builder", "skeptic"]),
                patch.object(panel, "_select_pi_role_specs", return_value=specs),
                patch.object(panel, "_has_high_stakes_signal", return_value=True),
                patch.object(panel, "build_evidence_pack", return_value=pack),
                patch.object(
                    panel,
                    "fit_pack_to_budget",
                    return_value={
                        "shared_core": pack.shared_core,
                        "private_packs": {
                            "builder": [{"id": "b"}],
                            "skeptic": [{"id": "s"}],
                        },
                    },
                ),
                patch.object(
                    panel,
                    "instantiate_pi_roles",
                    side_effect=fake_instantiate_pi_roles,
                ),
                patch.object(panel, "ChairArbiter", FakeChair),
                patch.object(
                    panel,
                    "validate_agenda_v2",
                    side_effect=[validation_missing_role, validation],
                ),
                patch.object(panel, "audit_agenda", return_value=audit),
                patch.object(panel, "log_synthesis_metrics") as log_metrics,
            ):
                result = asyncio.run(
                    panel.run_panel(
                        run_dir=run_dir,
                        workspace=workspace,
                        model="fake-model",
                        completed_gen_id=0,
                        panel_mode="mini",
                        cohort_size=3,
                        n_rounds=3,
                        round2_max_runtime_minutes=1,
                        reasoning_effort="high",
                    )
                )

            out_dir = run_dir / "research_memory" / "synth_gen0_to_1"
            self.assertTrue((out_dir / "evidence_pack.json").exists())
            self.assertTrue((out_dir / "memo_external_validity.yaml").exists())
            self.assertTrue((out_dir / "round2_label_maps.yaml").exists())
            self.assertTrue((out_dir / "final_agenda.yaml").exists())
            self.assertTrue((out_dir / "audit.json").exists())
            self.assertTrue(result.success)
            self.assertEqual(result.panel_mode, "high_stakes")
            self.assertEqual(
                result.cross_reviews["builder"]["own_revisions"][0]["claim_id"],
                "builder_claim",
            )
            self.assertEqual(len(result.cross_reviews["builder"]["own_revisions"]), 1)
            self.assertEqual(len(FakeChair.calls), 2)
            self.assertEqual(FakeChair.init_calls[0]["reasoning_effort"], "high")
            self.assertEqual(
                FakeChair.calls[1]["validation_feedback"],
                tuple(validation_missing_role.blocking_issues),
            )
            self.assertEqual(FakeChair.calls[1]["validation_candidate"]["attempt"], 1)
            self.assertEqual((out_dir / "chair_raw_failed.txt").read_text(), "agenda yaml 1")
            self.assertEqual((out_dir / "chair_raw.txt").read_text(), "agenda yaml 2")
            log_metrics.assert_called_once()

    def test_panel_chair_failure_preserves_raw_output(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor as panel,
        )

        class FakePI:
            role_name = "builder"

            async def run(self, shared_core, private_pack, target_decisions):
                return SimpleNamespace(
                    parsed={
                        "role": "builder",
                        "top_claims": [{"id": "c1"}],
                        "objections_or_warnings": [],
                        "claim_boundaries": [],
                        "private_knowledge_used": [],
                    }
                )

        class FailingChair:
            def __init__(self, **_kwargs):
                pass

            async def run(self, **_kwargs):
                return SimpleNamespace(
                    success=False,
                    agenda={},
                    raw_text="bad yaml",
                    error="parse failed",
                )

        pack = SimpleNamespace(shared_core={"shared_core_id": "not-hex"}, all_cards=[], audit={})
        specs = [SimpleNamespace(legacy_role_id="builder")]

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            with (
                patch.object(panel, "_select_pi_roles", return_value=["builder"]),
                patch.object(panel, "_select_pi_role_specs", return_value=specs),
                patch.object(panel, "build_evidence_pack", return_value=pack),
                patch.object(
                    panel,
                    "fit_pack_to_budget",
                    return_value={"shared_core": pack.shared_core, "private_packs": {}},
                ),
                patch.object(panel, "instantiate_pi_roles", return_value=[FakePI()]),
                patch.object(panel, "ChairArbiter", FailingChair),
            ):
                result = asyncio.run(
                    panel.run_panel(
                        run_dir=run_dir,
                        workspace=Path(tmp),
                        model="fake-model",
                        completed_gen_id=2,
                        panel_mode="mini",
                        cohort_size=1,
                        n_rounds=1,
                    )
                )

            out_dir = run_dir / "research_memory" / "synth_gen2_to_3"
            self.assertFalse(result.success)
            self.assertEqual(result.error, "chair: parse failed")
            self.assertEqual((out_dir / "chair_raw_failed.txt").read_text(), "bad yaml")
            self.assertEqual((out_dir / "chair_error.txt").read_text(), "parse failed")


if __name__ == "__main__":
    unittest.main()
