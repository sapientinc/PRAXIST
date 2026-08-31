"""An unbounded generation policy never invents a terminal generation."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary
from praxist.plugins.workflow_stages.research_loop.backend.frontier import FrontierStore


class ScorelessUnboundedBoundaryTest(unittest.TestCase):
    def test_unbounded_generations_continue_pi_and_retain_evidence_after_generation_eight(
        self,
    ) -> None:
        for gen_id, max_generations, expected_pi_calls in ((0, None, 1), (9, None, 1), (0, 1, 0)):
            with (
                self.subTest(gen_id=gen_id, limit=max_generations),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                findings_dir = root / "shared_findings"
                findings_dir.mkdir()
                findings = [{"id": "evidence", "generation_id": gen_id, "content": "Open inquiry"}]
                (findings_dir / "evidence.json").write_text(json.dumps(findings[0]))
                task = SimpleNamespace(
                    research_loop={"mode": "scoreless"},
                    evaluation=SimpleNamespace(diversity_dimensions=[], maturity_policy={}),
                    research_memory=SimpleNamespace(enabled=False),
                    generation_policy=SimpleNamespace(
                        max_generations=max_generations, per_generation_hours=None, cohort_size=1
                    ),
                )
                loop = SimpleNamespace(
                    run_dir=root,
                    findings_dir=findings_dir,
                    local_mode=False,
                    task_spec=task,
                    frontier=FrontierStore(root / "frontier"),
                    _collect_findings_for_generation=lambda generation, rows=findings: rows,
                    _strategy_for_gen=lambda generation: "explore",
                    _graph_maintainer=None,
                    _findings_sync=None,
                    gems=None,
                )
                pi = SimpleNamespace(
                    run=AsyncMock(
                        return_value=SimpleNamespace(
                            success=True,
                            next_gen_id=gen_id + 1,
                            agenda_path=root / "agenda.yaml",
                            duration_seconds=0.0,
                        )
                    )
                )

                asyncio.run(
                    generation_boundary.complete_generation_boundary(
                        loop, gen_id=gen_id, pi_agent=pi, pi_cfg=SimpleNamespace(strict=True)
                    )
                )

                self.assertEqual(pi.run.await_count, expected_pi_calls)
                boundary = json.loads(
                    (root / f"gen_{gen_id}" / "generation_boundary.json").read_text()
                )
                self.assertEqual(
                    boundary["pi_status"],
                    "succeeded" if expected_pi_calls else "skipped_last_generation",
                )
                manifest = json.loads(
                    (root / f"gen_{gen_id}" / "scoreless_evidence.json").read_text()
                )
                self.assertEqual(manifest["findings"][0]["content"], "Open inquiry")
