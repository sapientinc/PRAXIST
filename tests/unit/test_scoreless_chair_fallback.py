"""Failed Chair output must retain scoreless research contracts."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import chair_arbiter
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
    validate_agenda_v2,
)
from praxist.plugins.workflow_stages.research_loop.backend.prompt_context import (
    _compact_research_agenda_for_prompt,
)


def _memos() -> dict:
    return {
        "builder": {
            "top_claims": [
                {
                    "id": "C_builder_1",
                    "statement": "The source dates support two competing explanations.",
                    "confidence": 0.8,
                    "supports": ["finding-source-dates"],
                }
            ],
            "proposed_experiments": [
                {"description": "Compare the original publication dates with the archived notices."}
            ],
            "objections_or_warnings": [
                {
                    "target_claim": "C_builder_1",
                    "objection": "One notice may have been revised.",
                    "resolving_experiment": "Check the archived notice for later revisions.",
                }
            ],
        }
    }


class ScorelessChairFallbackTest(unittest.TestCase):
    def _run_chair(self, shared_core: dict, *, nonmapping: bool = False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chair = chair_arbiter.ChairArbiter(root, root, "offline", peer_budget=5)
            chair._role_skill = False
            execute = AsyncMock(
                return_value=SimpleNamespace(
                    success=True, error=None, output={"text_outputs": ["not valid YAML: ["]}
                )
            )
            reviews = {
                "builder": {
                    "singleton_high_upside_idea_to_preserve": {
                        "idea_summary": "Keep the competing chronology open.",
                        "source": "self",
                    },
                    "own_revisions": [
                        {
                            "claim_id": "C_builder_1",
                            "boundary_old": "All notices agree.",
                            "boundary_new": "One notice is unresolved.",
                            "confidence_old": 0.8,
                            "confidence_new": 0.4,
                            "triggered_by": "Check the archived notice for later revisions.",
                        }
                    ],
                }
            }
            with patch.object(BaseAgent, "execute", execute):
                if nonmapping:
                    with patch.object(
                        chair_arbiter,
                        "_parse_chair_agenda_text",
                        return_value=SimpleNamespace(agenda=[], cleaned_text="[]"),
                    ):
                        return asyncio.run(
                            chair.run(shared_core, _memos(), reviews, {}, 2, 1, "full", "core")
                        )
                return asyncio.run(
                    chair.run(shared_core, _memos(), reviews, {}, 2, 1, "full", "core")
                )

    def test_failed_chair_keeps_scoreless_in_both_fallback_branches(self) -> None:
        for nonmapping in (False, True):
            with self.subTest(nonmapping=nonmapping):
                result = self._run_chair({"research_loop_mode": "scoreless"}, nonmapping=nonmapping)
                self.assertTrue(result.success)
                agenda = result.agenda
                self.assertEqual(agenda.get("research_loop_mode"), "scoreless")
                validation = validate_agenda_v2(agenda, 2, cohort_size=5, pi_memos=_memos())
                self.assertTrue(validation.valid, validation.blocking_issues)
                first = agenda["cross_peer_hypotheses"][0]
                self.assertEqual(
                    first["claim"], "The source dates support two competing explanations."
                )
                self.assertEqual(first["source_findings"][0]["finding_id"], "finding-source-dates")
                self.assertEqual(
                    first["minimal_test"], "Check the archived notice for later revisions."
                )
                self.assertEqual(first["parent_candidate"], "")
                self.assertEqual(first["parent_usage"], "")
                self.assertEqual(first["confidence"], 0.4)
                serialized = json.dumps(agenda)
                self.assertIn("Keep the competing chronology open.", serialized)
                self.assertIn("One notice is unresolved.", serialized)
                for peer_id in ("gen2_peer0", "gen2_peer1", "gen2_peer3"):
                    peer_context = _compact_research_agenda_for_prompt(agenda, peer_id)
                    self.assertIn("One notice is unresolved.", json.dumps(peer_context))
                minority_context = _compact_research_agenda_for_prompt(agenda, "gen2_peer3")
                self.assertIn("Keep the competing chronology open.", json.dumps(minority_context))
                for gate in (
                    "task-valid metrics",
                    "evaluation gate",
                    "primary task metric",
                    "measurable signal",
                    "measured result",
                    "raw metrics",
                    "evaluation tier",
                ):
                    self.assertNotIn(gate, serialized)

    def test_omitted_mode_keeps_metric_fallback_behavior(self) -> None:
        agenda = self._run_chair({}).agenda
        self.assertIn("task-valid metrics", agenda["cross_peer_hypotheses"][0]["promote_condition"])
        self.assertIn("primary task metric", agenda["anti_mainline_contract"]["target_axes"])
        self.assertIn("measurable signal", agenda["minority_high_upside"][0]["stop_condition"])

    def test_no_claims_produces_open_inquiries_without_invented_sources(self) -> None:
        agenda = chair_arbiter._build_deterministic_fallback_agenda(
            pi_memos={},
            cross_reviews={},
            next_gen_id=2,
            completed_gen_id=1,
            panel_mode="full",
            shared_core_id="core",
            peer_budget=5,
            parse_error="malformed",
            research_loop_mode="scoreless",
        )
        self.assertTrue(validate_agenda_v2(agenda, 2).valid)
        self.assertTrue(all(not h["source_findings"] for h in agenda["cross_peer_hypotheses"]))
        self.assertEqual(len(agenda["peer_contracts"]), 5)


if __name__ == "__main__":
    unittest.main()
