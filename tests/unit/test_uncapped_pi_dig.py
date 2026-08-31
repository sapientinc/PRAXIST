"""Optional research limits survive PI and DIG dispatch without fallback caps."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from praxist.core.protocol import AgentRunRequest
from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent
from praxist.plugins.workflow_stages.research_loop.backend.dig.config import DIGLiteConfig
from praxist.plugins.workflow_stages.research_loop.backend.dig.runner import run_dig_lite
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
    legacy_two_round_executor as panel,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.chair_arbiter import (
    ChairArbiter,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles._base_pi import BasePI
from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent
from tests.unit.test_task_execution_role_dispatch import _dig_outputs


@contextmanager
def _dispatches(
    outputs: list[str] | None = None,
) -> Iterator[tuple[list[AgentRunRequest], list[float | None]]]:
    requests: list[AgentRunRequest] = []
    timeouts: list[float | None] = []
    real_wait_for = asyncio.wait_for

    async def execute(agent: BaseAgent, task: str) -> SimpleNamespace:
        request = agent._build_agent_run_request(task, {})
        requests.append(request)
        return SimpleNamespace(
            success=outputs is not None,
            error=None if outputs is not None else "offline capture",
            output={"text": outputs.pop(0)} if outputs is not None else {},
            request_id=request.request_id,
            duration=0.0,
            iteration_count=1,
        )

    async def wait_for(awaitable, timeout):
        timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout)

    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(BaseAgent, "execute", execute),
        patch.object(asyncio, "wait_for", wait_for),
    ):
        yield requests, timeouts


class UncappedPIRequestsTest(unittest.TestCase):
    def test_single_pi_and_panel_fallback_dispatch_without_runtime_limit(self) -> None:
        for panel_failure in (None, False, True):
            with self.subTest(panel_failure=panel_failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                template = root / "prompt.jinja2"
                template.write_text("Plan the next inquiry.", encoding="utf-8")
                pi = PIAgent(
                    root,
                    root,
                    cohort_size=1,
                    model="offline",
                    max_runtime_minutes=None,
                    prompt_template_path=template,
                    task_spec=SimpleNamespace(research_loop={"mode": "scoreless"}),
                    use_multi_pi_panel=panel_failure is not None,
                    multi_pi_config=SimpleNamespace(fallback_to_single_pi_on_panel_failure=True),
                )
                panel = AsyncMock(
                    side_effect=RuntimeError("offline panel failure") if panel_failure else None,
                    return_value=SimpleNamespace(success=False, error="offline panel failure"),
                )
                with (
                    _dispatches() as (requests, timeouts),
                    patch.object(pi, "_run_multi_pi_panel", panel),
                ):
                    asyncio.run(pi.run(0))
                self.assertEqual(len(requests), 1)
                self.assertEqual(timeouts, [None])

    def test_panel_roles_accept_uncapped_and_explicit_runtime_limits(self) -> None:
        for minutes, seconds in ((None, None), (3, 180)):
            for role in ("pi", "review", "chair"):
                with self.subTest(minutes=minutes, role=role), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    with _dispatches() as (requests, timeouts):
                        if role == "chair":
                            agent = ChairArbiter(root, root, "offline", max_runtime_minutes=minutes)
                            asyncio.run(agent.run({}, {}, {}, {}, 1, 0, "full", "core"))
                        else:
                            agent = BasePI(root, root, "offline", max_runtime_minutes=minutes)
                            if role == "pi":
                                asyncio.run(agent.run({}, [], []))
                            else:
                                asyncio.run(
                                    agent.run_cross_review(
                                        {}, {"other": {}}, round2_max_runtime_minutes=minutes
                                    )
                                )
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(timeouts, [seconds])
                    prompt = requests[0].prompt_ref["text"]
                    if minutes is None:
                        self.assertIn("No role-specific runtime limit is configured.", prompt)
                    else:
                        self.assertIn("You have 3 minutes.", prompt)

    def test_omitted_metric_limits_keep_dispatch_deadlines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _dispatches() as (requests, timeouts):
            root = Path(tmp)
            pi = BasePI(root, root, "offline")
            asyncio.run(pi.run({}, [], []))
            asyncio.run(pi.run_cross_review({}, {"other": {}}))
            asyncio.run(
                ChairArbiter(root, root, "offline").run({}, {}, {}, {}, 1, 0, "full", "core")
            )
        self.assertEqual(len(requests), 3)
        self.assertEqual(timeouts, [720, 360, 480])


class UncappedDIGRequestsTest(unittest.TestCase):
    def test_scoreless_absent_and_null_limits_do_not_restore_metric_defaults(self) -> None:
        limits = (
            "planner_max_runtime_minutes",
            "attempt_max_runtime_minutes",
            "max_total_runtime_minutes",
            "max_attempts",
            "max_refinement_rounds",
        )
        for raw in ({"enabled": True}, {key: None for key in limits}):
            with self.subTest(raw=raw):
                config = DIGLiteConfig.from_raw(raw, scoreless=True)
                self.assertEqual([getattr(config, key) for key in limits], [None] * 5)
        explicit = DIGLiteConfig.from_raw({key: 3 for key in limits}, scoreless=True)
        self.assertEqual([getattr(explicit, key) for key in limits], [3] * 5)
        metric = DIGLiteConfig.from_raw({"enabled": True})
        self.assertEqual([getattr(metric, key) for key in limits], [10, 0, 40, 10, 1])

    def test_uncapped_dig_repairs_and_resumed_checkpoint_dispatch_without_deadlines(self) -> None:
        baseline, pool, reviews, contract = _dig_outputs()
        outputs = [baseline, pool, "not a mapping", reviews, contract, contract]
        with tempfile.TemporaryDirectory() as tmp, _dispatches(outputs) as (requests, timeouts):
            root = Path(tmp)
            kwargs = {
                "ctx": {"peer_id": "peer", "gen_id": 0, "research_loop_mode": "scoreless"},
                "config": DIGLiteConfig(
                    candidate_count=4,
                    min_mechanism_families=4,
                    planner_max_runtime_minutes=None,
                    attempt_max_runtime_minutes=None,
                    max_total_runtime_minutes=None,
                    max_attempts=None,
                    max_refinement_rounds=None,
                ),
                "dig_dir": root / "dig",
                "workspace": root,
                "model": "offline",
                "mcp_servers": {},
                "plugin_registry": None,
            }
            asyncio.run(run_dig_lite(**kwargs))
            asyncio.run(run_dig_lite(**kwargs))
        self.assertEqual(len(requests), 6)
        self.assertEqual(timeouts, [None] * 6)
        self.assertTrue(all(request.timeout_seconds in (None, 0) for request in requests))
        self.assertNotIn("max_refinement_rounds:", requests[0].prompt_ref["text"])
        self.assertEqual(outputs, [])


class UncappedPanelCoordinatorTest(unittest.TestCase):
    @contextmanager
    def _pack(self, *, mode: str = "scoreless"):
        shared_core = {"shared_core_id": "abcdef12", "research_loop_mode": mode}
        pack = SimpleNamespace(shared_core=shared_core, all_cards=[], audit={})
        with (
            patch.object(panel, "build_evidence_pack", return_value=pack),
            patch.object(
                panel,
                "fit_pack_to_budget",
                return_value={"shared_core": shared_core, "private_packs": {}},
            ),
        ):
            yield

    def test_pi_task_routes_omitted_and_null_panel_limits_as_uncapped(self) -> None:
        for config in (
            {},
            {
                "pi_max_runtime_minutes": None,
                "chair_max_runtime_minutes": None,
                "round2_max_runtime_minutes": None,
            },
        ):
            with (
                self.subTest(config=config),
                tempfile.TemporaryDirectory() as tmp,
                self._pack(),
                _dispatches() as (requests, timeouts),
            ):
                root = Path(tmp)
                pi = PIAgent(
                    root,
                    root,
                    cohort_size=1,
                    model="offline",
                    max_runtime_minutes=None,
                    task_spec=SimpleNamespace(research_loop={"mode": "scoreless"}),
                    use_multi_pi_panel=True,
                    multi_pi_config=SimpleNamespace(**config),
                )
                asyncio.run(pi.run(0))
                self.assertEqual(len(requests), 5)
                self.assertEqual(timeouts, [None] * 5)

    def test_uncapped_panel_reaches_chair_including_high_stakes_review(self) -> None:
        for rounds in (1, 2):
            with (
                self.subTest(rounds=rounds),
                tempfile.TemporaryDirectory() as tmp,
                self._pack(),
                _dispatches() as (requests, timeouts),
            ):
                root = Path(tmp)
                asyncio.run(
                    panel.run_panel(
                        root,
                        root,
                        "offline",
                        0,
                        panel_mode="high_stakes",
                        n_rounds=rounds,
                        pi_max_runtime_minutes=None,
                        chair_max_runtime_minutes=None,
                        round2_max_runtime_minutes=None,
                    )
                )
                self.assertEqual(len(requests), 5)
                self.assertEqual(timeouts, [None] * 5)

    def test_scoreless_explicit_review_limit_is_not_increased(self) -> None:
        for mode, expected in (("scoreless", 3), ("metric", 9)):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory() as tmp,
                self._pack(mode=mode),
            ):
                root = Path(tmp)
                limits = []

                async def cross_review(
                    pis, pi_memos, round2_max_runtime_minutes, rng_seed, limits=limits
                ):
                    limits.append(round2_max_runtime_minutes)
                    return {}, {}

                with _dispatches(), patch.object(panel, "_run_round2_parallel", cross_review):
                    asyncio.run(
                        panel.run_panel(
                            root,
                            root,
                            "offline",
                            0,
                            panel_mode="high_stakes",
                            round2_max_runtime_minutes=3,
                        )
                    )
                self.assertEqual(limits, [expected])

    def test_uncapped_chair_correction_runs_without_a_synthetic_deadline(self) -> None:
        attempts = []

        async def chair_run(chair, **kwargs):
            attempts.append(kwargs)
            return SimpleNamespace(
                success=True,
                agenda={"peer_contracts": [], "generation": "gen1"},
                raw_text="offline candidate",
                error=None,
            )

        invalid = SimpleNamespace(
            valid=False,
            blocking_issues=["peer_contracts missing roles: ['peer_generalist'] in full panel"],
            warnings=[],
        )
        valid = SimpleNamespace(valid=True, blocking_issues=[], warnings=[])
        audit = SimpleNamespace(pass_=True, blocking_issues=[], warnings=[], metrics={})
        with (
            tempfile.TemporaryDirectory() as tmp,
            self._pack(),
            _dispatches() as (_, timeouts),
            patch.object(ChairArbiter, "run", chair_run),
            patch.object(panel, "validate_agenda_v2", side_effect=[invalid, valid]),
            patch.object(panel, "audit_agenda", return_value=audit),
        ):
            root = Path(tmp)
            result = asyncio.run(
                panel.run_panel(
                    root,
                    root,
                    "offline",
                    0,
                    n_rounds=1,
                    pi_max_runtime_minutes=None,
                    chair_max_runtime_minutes=None,
                )
            )
            self.assertTrue(result.success, result.error)
            self.assertEqual(len(attempts), 2)
            self.assertTrue(attempts[1]["validation_feedback"])
            self.assertTrue(all(timeout is None for timeout in timeouts))


if __name__ == "__main__":
    unittest.main()
