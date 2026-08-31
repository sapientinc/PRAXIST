"""Stock prompt rendering must carry scoreless evidence into the next peer."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from praxist.core.prompt_layout import build_legacy_jinja_prompt_layout
from praxist.plugins.workflow_stages.research_loop.backend.prompt_context import (
    build_prompt_context,
)

BACKEND = (
    Path(__file__).resolve().parents[2] / "praxist/plugins/workflow_stages/research_loop/backend"
)


class ScorelessPromptRenderingTest(unittest.TestCase):
    def test_scoreless_pi_keeps_narrative_without_inventing_a_missing_finding_id(self) -> None:
        from jinja2 import Template

        prompt = Template((BACKEND / "synthesis_prompt.jinja2").read_text()).render(
            research_loop_mode="scoreless",
            scoreless_evidence=[{"content": "Partial evidence still matters."}],
            completed_gen_id=0,
            next_gen_id=1,
            cohort_size=1,
            required_peer_roles=["investigator"],
            agenda_output_path="/run/agenda.yaml",
        )

        self.assertIn("Partial evidence still matters.", prompt)
        schema_match = re.search(r"```yaml\n(.*?)\n```", prompt, re.DOTALL)
        self.assertIsNotNone(schema_match)
        agenda = yaml.safe_load(schema_match.group(1))
        self.assertEqual(agenda["cross_peer_hypotheses"][0]["source_findings"], [])

    def test_stock_scoreless_pi_prompt_renders_narratives_and_a_valid_agenda_schema(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for generation_id, content in (
                (0, "Prior hypothesis with an unresolved mechanism."),
                (1, "Current insight identifies an alternative cause."),
            ):
                generation_dir = root / f"gen_{generation_id}"
                generation_dir.mkdir()
                (generation_dir / "scoreless_evidence.json").write_text(
                    json.dumps(
                        {
                            "mode": "scoreless",
                            "generation_id": generation_id,
                            "findings": [
                                {
                                    "id": f"finding-{generation_id}",
                                    "finding_type": "hypothesis"
                                    if generation_id == 0
                                    else "insight",
                                    "title": "Research evidence",
                                    "peer_id": f"gen{generation_id}_peer0",
                                    "content": content,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            agent = PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=4,
                model="offline",
                task_spec=SimpleNamespace(research_loop={"mode": "scoreless"}),
            )
            prompt = agent._build_synthesis_prompt(
                completed_gen_id=1,
                findings=[],
                edges=[],
                frontier=[],
                prior_agenda=None,
                prior_agendas_summary=[],
                prior_findings_summary=[],
                agenda_output_path=root / "agenda.yaml",
            )

        self.assertIn("Prior hypothesis with an unresolved mechanism.", prompt)
        self.assertIn("Current insight identifies an alternative cause.", prompt)
        self.assertIn("finding-0", prompt)
        self.assertIn("finding-1", prompt)
        for forbidden in ("Pareto", "Gems", "frontier", "full evaluation", "at least 3"):
            self.assertNotIn(forbidden, prompt)
        schema_match = re.search(r"```yaml\n(.*?)\n```", prompt, re.DOTALL)
        self.assertIsNotNone(schema_match)
        agenda = yaml.safe_load(schema_match.group(1))
        self.assertIsNone(agent.validate_agenda(agenda, next_gen_id=2))
        self.assertEqual(set(agenda["peer_contracts"]), {f"gen2_peer{i}" for i in range(4)})

    def _render_templates(self, context: dict) -> str:
        return build_legacy_jinja_prompt_layout(
            base_template_path=BACKEND / "prompt_base.jinja2",
            task_prompt_path=None,
            generation_template_path=BACKEND / "prompt_generation.jinja2",
            context=context,
            run_id="offline",
            stage_id="research_loop",
            prompt_id="peer",
        ).prompt_text

    def _context(self) -> dict:
        return {
            "peer_id": "gen1_peer0",
            "gen_id": 1,
            "logical_gen_id": 1,
            "cohort_size": 1,
            "workspace_dir": "/task",
            "run_dir": "/run",
            "results_dir": "/run/results",
            "variants_dir": "/run/variants",
            "findings_dir": "/run/shared_findings",
            "notebook_path": "/run/work/notebooks/peer.json",
            "notebook_dir": "/run/work/notebooks",
            "logs_dir": "/run/gen_1",
            "local_mode": True,
            "graph_session_context": "",
            "frontier_summary": [],
            "variant_hint": "Investigate the open question.",
            "research_agenda": None,
            "peer_role_descriptions": {},
            "literature_lookup_enabled": False,
            "gems_context": {"enabled": False},
            "task_spec": SimpleNamespace(),
        }

    def test_scoreless_agenda_preserves_task_contract_without_metric_role_include(self) -> None:
        context = self._context()
        context.update(
            {
                "research_loop_mode": "scoreless",
                "scoreless_evidence": [
                    {
                        "id": "prior",
                        "content": "Existing contrary evidence.",
                        "evidence_manifest": "gen_0/scoreless_evidence.json",
                    }
                ],
                "peer_role_descriptions": {"bridge": "Compare independent source accounts."},
                "research_agenda": {
                    "peer_contracts": {
                        "gen1_peer0": {
                            "role": "bridge",
                            "research_plan": "Test an alternative explanation.",
                        }
                    }
                },
            }
        )

        text = self._render_templates(context)

        self.assertIn("Test an alternative explanation.", text)
        self.assertIn("Compare independent source accounts.", text)
        self.assertIn("Existing contrary evidence.", text)
        self.assertNotIn("Pareto", text)
        self.assertNotIn("Previous Generations' Top Results", text)

    def test_metric_mode_retains_default_stock_evaluation_guidance(self) -> None:
        context = self._context()
        default_text = self._render_templates(context)
        context["research_loop_mode"] = "metric"
        metric_text = self._render_templates(context)

        self.assertEqual(default_text, metric_text)
        self.assertIn("Stage-Gated Evidence Discipline", metric_text)
        self.assertIn("Gems Preservation Guard", metric_text)
        self.assertIn("log_experiment_metrics", metric_text)
        self.assertIn("get_frontier", metric_text)

    def test_stock_next_generation_prompt_renders_prior_narrative_without_fake_evaluation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gen_0").mkdir()
            (root / "gen_0" / "scoreless_evidence.json").write_text(
                json.dumps(
                    {
                        "mode": "scoreless",
                        "generation_id": 0,
                        "findings": [
                            {
                                "id": "gen0-hypothesis",
                                "finding_type": "hypothesis",
                                "peer_id": "gen0_peer0",
                                "title": "Retained hypothesis",
                                "content": "Prior narrative: two independent causes may explain the observation.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            context = build_prompt_context(
                task_spec=SimpleNamespace(
                    research_loop={"mode": "scoreless"}, evaluation=SimpleNamespace(), _raw={}
                ),
                workspace=root,
                run_dir=root,
                results_dir=root / "results",
                variants_dir=root / "variants",
                findings_dir=root / "shared_findings",
                frontier=SimpleNamespace(get_summary=lambda: []),
                local_mode=True,
                gen_id=1,
                peer_index=0,
                cohort_size=1,
                strategy="explore",
            )
            layout = build_legacy_jinja_prompt_layout(
                base_template_path=BACKEND / "prompt_base.jinja2",
                task_prompt_path=None,
                generation_template_path=BACKEND / "prompt_generation.jinja2",
                context=context,
                run_id="offline",
                stage_id="research_loop",
                prompt_id="scoreless-peer",
            )
            text = layout.prompt_text

        self.assertIn("Prior narrative: two independent causes may explain the observation.", text)
        self.assertIn("gen0-hypothesis", text)
        self.assertIn("scoreless", text.lower())
        self.assertIn("gen_0/scoreless_evidence.json", text)
        self.assertIn("work/notebooks", text)
        self.assertIn("Create", text)
        for forbidden in (
            "### First Generation",
            "there is no prior work to build on",
            "seed the Frontier",
            "Gems Preservation Guard",
            "Stage-Gated Evidence Discipline",
            "log_experiment_metrics",
            "get_leaderboard",
            "get_frontier",
            "Run the task-declared protocol",
            "full aggregated metrics across all seeds",
        ):
            self.assertNotIn(forbidden, text)
