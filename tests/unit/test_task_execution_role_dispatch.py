"""Workflow call origins select task execution policies explicitly."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from praxist.core.execution_policy import task_execution_policy_scope
from praxist.core.protocol import AgentRunRequest
from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent
from praxist.plugins.workflow_stages.research_loop.backend.dig.config import DIGLiteConfig
from praxist.plugins.workflow_stages.research_loop.backend.dig.runner import run_dig_lite
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.chair_arbiter import (
    ChairArbiter,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles._base_pi import (
    BasePI,
)
from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent


@contextmanager
def _capture_requests(outputs: list[str] | None = None) -> Iterator[list[AgentRunRequest]]:
    requests: list[AgentRunRequest] = []

    async def execute(agent: BaseAgent, task: str) -> SimpleNamespace:
        request = agent._build_agent_run_request(task, {})
        requests.append(request)
        output = {} if outputs is None else {"text": outputs.pop(0)}
        return SimpleNamespace(
            success=outputs is not None,
            error="offline dispatch capture" if outputs is None else None,
            output=output,
            request_id=request.request_id,
            duration=0.0,
            iteration_count=1,
        )

    roles = {
        role: {"model": f"{role}-model", "reasoning_effort": "high"}
        for role in ("research", "pi", "review", "final")
    }
    with (
        patch.dict(os.environ, {"PRAXIST_ROLE_REF": "task_role:final"}, clear=True),
        task_execution_policy_scope({"model_by_role": roles}),
        patch.object(BaseAgent, "execute", execute),
    ):
        yield requests


def _dig_outputs() -> list[str]:
    cells = [
        ("alpha", "worker", "repair"),
        ("beta", "control", "explore"),
        ("gamma", "cache", "exploit"),
        ("diagnostic_falsifier", "logging", "falsify"),
    ]
    baseline = {
        "task_objective": {"primary_metric": "score"},
        "baseline_core_path": [{"file": "worker.py", "role": "worker"}],
        "intervention_surfaces": [{"name": "worker", "files": ["worker.py"], "allowed": True}],
    }
    pool = {
        "candidates": [
            {
                "candidate_id": f"C{i}",
                "name": f"candidate_{i}",
                "mechanism_family": family,
                "intervention_surface": surface,
                "intent": intent,
                "hypothesis": f"Test {family}.",
                "implementation_sketch": {
                    "files_to_modify": ["worker.py"],
                    "changes": [f"Implement {family}."],
                },
                "diversity_signature": {
                    "mechanism_family": family,
                    "intervention_surface": surface,
                    "intent": intent,
                },
            }
            for i, (family, surface, intent) in enumerate(cells, 1)
        ]
    }
    reviews = {
        "reviews": [
            {
                "candidate_id": f"C{i}",
                "scores": {
                    "mechanism_plausibility": 5 if i == 1 else 2,
                    "implementability": 5 if i == 1 else 2,
                    "diagnostic_clarity": 5 if i == 1 else 2,
                    "diversity_value": 5 if i == 1 else 2,
                    "shortcut_risk": 1,
                    "silent_bug_risk": 1,
                    "compute_risk": 1,
                },
                "fatal_flaws": [],
            }
            for i in range(1, 5)
        ]
    }
    contract = {
        "selected_candidate_id": "C1",
        "variant_name": "dispatch_probe",
        "diversity_cell": pool["candidates"][0]["diversity_signature"],
        "mechanism_hypothesis": "Test alpha.",
        "why_selected": "Highest review score.",
        "rejected_alternatives": [
            {"candidate_id": f"C{i}", "reason": "Lower review score."} for i in range(2, 5)
        ],
        "files_to_modify": ["worker.py"],
        "allowed_changes": ["Implement alpha."],
        "forbidden_changes": [
            "do not modify evaluator",
            "do not change data split",
            "do not change metric calculation",
        ],
        "implementation_plan": [{"step": 1, "action": "Modify worker.py."}],
        "expected_metric_signature": {"primary": "Improves.", "diagnostic": "Changes."},
        "ablation_hooks": ["disable_alpha"],
        "fail_fast_checks": ["Output remains valid."],
    }
    return [json.dumps(value) for value in (baseline, pool, reviews, contract)]


class TaskExecutionRoleDispatchTests(unittest.TestCase):
    def assert_role(self, requests: list[AgentRunRequest], role: str) -> None:
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.runtime_options["execution_role"], role)
        self.assertEqual(request.model_call.model, f"{role}-model")
        self.assertEqual(request.runtime_options["reasoning_effort"], "high")
        self.assertEqual(request.role_ref, "task_role:final")

    def test_panel_round_one_selects_pi_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _capture_requests() as requests:
            root = Path(tmp)
            pi = BasePI(root, root, "unbound-model")
            asyncio.run(pi.run({}, [], []))
        self.assert_role(requests, "pi")

    def test_panel_cross_review_selects_review_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _capture_requests() as requests:
            root = Path(tmp)
            pi = BasePI(root, root, "unbound-model")
            asyncio.run(pi.run_cross_review({}, {"peer-a": {}}))
        self.assert_role(requests, "review")

    def test_single_pi_selects_pi_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _capture_requests() as requests:
            root = Path(tmp)
            pi = PIAgent(root, root, cohort_size=1, model="unbound-model")
            asyncio.run(
                pi._invoke_synthesizer(
                    "plan next generation", root / "agenda.yaml", request_id="arbitrary-call"
                )
            )
        self.assert_role(requests, "pi")

    def test_panel_failure_and_exception_fallback_select_pi_policy(self) -> None:
        for raises in (False, True):
            with (
                self.subTest(panel_raises=raises),
                tempfile.TemporaryDirectory() as tmp,
                _capture_requests() as requests,
            ):
                root = Path(tmp)
                template = root / "synthesis.jinja2"
                template.write_text("plan next generation", encoding="utf-8")
                pi = PIAgent(
                    root,
                    root,
                    cohort_size=1,
                    model="unbound-model",
                    use_multi_pi_panel=True,
                    multi_pi_config=SimpleNamespace(fallback_to_single_pi_on_panel_failure=True),
                    prompt_template_path=template,
                )
                panel = AsyncMock(
                    side_effect=RuntimeError("offline panel failure") if raises else None,
                    return_value=SimpleNamespace(success=False, error="offline panel failure"),
                )
                with (
                    patch.object(pi, "_run_multi_pi_panel", panel),
                    self.assertLogs(
                        "praxist.plugins.workflow_stages.research_loop.backend.pi_agent",
                        level="WARNING",
                    ),
                ):
                    asyncio.run(pi.run(completed_gen_id=0))
                self.assert_role(requests, "pi")

    def test_chair_and_correction_select_pi_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chair = ChairArbiter(root, root, "unbound-model")
            for feedback in ((), ("required role is missing",)):
                with self.subTest(feedback=feedback), _capture_requests() as requests:
                    asyncio.run(
                        chair.run(
                            shared_core_digest={},
                            pi_memos={},
                            cross_reviews={},
                            confidence_revisions={},
                            next_gen_id=1,
                            completed_gen_id=0,
                            panel_mode="standard",
                            shared_core_id="core-0",
                            validation_feedback=feedback,
                            validation_candidate={} if feedback else None,
                        )
                    )
                    self.assert_role(requests, "pi")

    def test_dig_phases_repair_and_checkpoint_resume_select_research_policy(self) -> None:
        baseline, pool, reviews, contract = _dig_outputs()
        outputs = [baseline, pool, "not a mapping", reviews, contract, contract]
        with tempfile.TemporaryDirectory() as tmp, _capture_requests(outputs) as requests:
            root = Path(tmp)
            kwargs = {
                "ctx": {"peer_id": "unrelated-role-name", "gen_id": 0},
                "config": DIGLiteConfig(candidate_count=4, min_mechanism_families=4),
                "dig_dir": root / "dig",
                "workspace": root,
                "model": "unbound-model",
                "mcp_servers": {},
                "plugin_registry": None,
            }
            asyncio.run(run_dig_lite(**kwargs))
            self.assertEqual(len(requests), 5)
            asyncio.run(run_dig_lite(**kwargs))
            self.assertEqual(len(requests), 6)
            self.assertEqual(outputs, [])
        for request in requests:
            self.assert_role([request], "research")
