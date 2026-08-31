"""Default multi-PI prompts preserve narrative evidence without scored policy."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.chair_arbiter import (
    ChairArbiter,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles.builder_pi import (
    BuilderPI,
)


def _shared_core() -> dict:
    return {
        "research_loop_mode": "scoreless",
        "scoreless_evidence": [
            {
                "id": "prior-hypothesis",
                "finding_type": "hypothesis",
                "content": "Prior mechanism remains unresolved.",
                "evidence_manifest": "gen_0/scoreless_evidence.json",
            },
            {
                "id": "current-insight",
                "finding_type": "insight",
                "content": "Current evidence suggests an alternative explanation.",
                "evidence_manifest": "gen_1/scoreless_evidence.json",
            },
        ],
    }


class ScorelessPanelRenderingTest(unittest.TestCase):
    def test_scoreless_round_one_preserves_task_questions_without_role_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = BuilderPI(root, root, "offline")
            skill = SimpleNamespace(to_prompt_context=lambda: {"role_ref": "task_role:custom"})
            with patch.object(pi, "skill", return_value=skill):
                prompt = pi.render_prompt(
                    _shared_core(), [], [], [], ["Which task-specific source needs reconciliation?"]
                )

        self.assertIn("Which task-specific source needs reconciliation?", prompt)

    def test_round_one_renders_narratives_and_qualitative_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = BuilderPI(root, root, "offline")
            pi._role_skill = False
            prompt = pi.render_prompt(_shared_core(), [], [], [], pi.fixed_questions())

        self.assertIn("Prior mechanism remains unresolved.", prompt)
        self.assertIn("Current evidence suggests an alternative explanation.", prompt)
        self.assertIn("gen_0/scoreless_evidence.json", prompt)
        self.assertIn("qualitative", prompt)
        self.assertIn("proposed_peer_contracts:", prompt)
        self.assertIn("C_builder_", prompt)
        for text in ("Pareto", "Gems", "full_eval", "promotion_attempt", "mature validation"):
            self.assertNotIn(text, prompt)

    def test_chair_retains_evidence_and_allocates_without_scored_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chair = ChairArbiter(root, root, "offline", peer_budget=3)
            chair._role_skill = False
            prompt = chair.render_prompt(_shared_core(), {}, {}, {}, 2, 1, "full", "frozen-core")

        self.assertIn("Prior mechanism remains unresolved.", prompt)
        self.assertIn("Current evidence suggests an alternative explanation.", prompt)
        self.assertIn("qualitative", prompt)
        self.assertIn("gen2_peer0:", prompt)
        self.assertIn("gen2_peer2:", prompt)
        self.assertNotIn("gen2_peer3:", prompt)
        self.assertIn("expected_pareto_movement:", prompt)
        self.assertIn("success_metrics:", prompt)
        for text in ("Gems", "Pareto axes", "full evaluation", "full_eval", "promotion_attempt"):
            self.assertNotIn(text, prompt)

    def test_round_two_uses_controller_mode_even_when_memos_do_not_name_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = BuilderPI(root, root, "offline")
            pi._role_skill = False
            pi.render_prompt(_shared_core(), [], [], [])
            execute = AsyncMock(
                return_value=SimpleNamespace(success=False, error="offline capture")
            )
            with patch.object(BaseAgent, "execute", execute):
                asyncio.run(
                    pi.run_cross_review({"role": "builder"}, {"PI #A": {"role": "skeptic"}})
                )
            prompt = execute.call_args.kwargs["task"]

        self.assertIn("scoreless", prompt)
        self.assertIn("qualitative", prompt)
        self.assertIn("claim_that_should_be_downgraded:", prompt)
        self.assertNotIn("missing baseline", prompt)

    def test_metric_prompts_keep_existing_promotion_and_diversity_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pi = BuilderPI(root, root, "offline")
            pi._role_skill = False
            prompt = pi.render_prompt({}, [], [], [], pi.fixed_questions())
            chair = ChairArbiter(root, root, "offline")
            chair._role_skill = False
            chair_prompt = chair.render_prompt({}, {}, {}, {}, 2, 1, "full", "metric-core")

        self.assertIn("Which Pareto arm has the most remaining headroom?", prompt)
        self.assertIn("## Gems Preservation Guard", prompt)
        self.assertIn("full_eval|replication|promotion_attempt", prompt)
        self.assertIn("A task-defined full evaluation is required", chair_prompt)
        self.assertIn("at_least_1_anti_mainline_full_evaluation_completed", chair_prompt)
