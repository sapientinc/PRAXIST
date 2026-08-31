"""Scoreless research has no implicit generation limit and resumes its decision."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from praxist.core.controller_state import write_private_startup_config
from praxist.plugins.workflow_stages.research_loop.backend.generation_loop import GenerationLoop
from praxist.task_spec import load_task_spec


class UnboundedScorelessLoopTest(unittest.TestCase):
    def test_review_can_complete_research_after_nine_unbounded_generations(self):
        self._exercise(resume_at_ninth_boundary=False)

    def test_uncapped_resume_finishes_pending_ninth_review_without_repeating_cohorts(self):
        self._exercise(resume_at_ninth_boundary=True)

    def _exercise(self, *, resume_at_ninth_boundary):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            task = root / "task"
            task.mkdir()
            run = root / "run"
            (task / "coordinator.py").write_text(
                "import json\n"
                "async def handle(ctx):\n"
                "    assert ctx.deadline_at is None\n"
                "    assert ctx.phase_deadline_at is None\n"
                "    name = f'{ctx.phase}-{ctx.generation_id}.json'\n"
                "    path = ctx.run_dir / name\n"
                "    contents = [item['content'] for item in ctx.findings]\n"
                "    path.write_text(json.dumps(contents))\n"
                "    events = ctx.run_dir / 'callback-events.json'\n"
                "    prior = json.loads(events.read_text()) if events.exists() else []\n"
                "    events.write_text(json.dumps(prior + [name]))\n"
                "    summary = {'phase': ctx.phase}\n"
                "    if ctx.phase == 'review':\n"
                "        summary['research_complete'] = ctx.generation_id >= 8\n"
                "    return {'status':'completed', 'artifacts':[name], 'summary':summary}\n",
                encoding="utf-8",
            )
            (task / "task.yaml").write_text(
                yaml.safe_dump(
                    {
                        "task_id": "uncapped-evidence",
                        "research_loop": {
                            "mode": "scoreless",
                            "lifecycle": {
                                "entrypoint": "coordinator.py:handle",
                                "after_generation": True,
                            },
                        },
                        "generation_policy": {"cohort_size": 1},
                        "synthesis_trigger": {"enabled": False},
                        "pi_agent": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            spec = load_task_spec(task / "task.yaml")
            self.assertIsNone(spec.generation_policy.max_generations)
            self.assertIsNone(spec.generation_policy.per_generation_hours)
            cohorts = []

            async def cohort(loop, generation):
                cohorts.append(generation)
                finding = {
                    "id": f"evidence-{generation}",
                    "generation_id": generation,
                    "peer_id": f"gen{generation}_peer0",
                    "finding_type": "hypothesis",
                    "content": f"Unscored evidence from generation {generation}",
                    "metrics": {},
                }
                (loop.findings_dir / f"{generation}.json").write_text(json.dumps(finding))
                results = [{"peer_id": finding["peer_id"], "success": True}]
                gen_dir = run / f"gen_{generation}"
                gen_dir.mkdir(parents=True, exist_ok=True)
                (gen_dir / "generation_results.json").write_text(json.dumps(results))
                if resume_at_ninth_boundary and generation == 8:
                    (run / "ORCHESTRATOR_SHUTDOWN").write_text("operator interrupted review")
                return results

            prefix = "praxist.plugins.workflow_stages.research_loop.backend.generation_loop."
            with ExitStack() as stack:
                stack.enter_context(
                    patch.dict(
                        os.environ,
                        {
                            "PRAXIST_CONTROLLER_STATE_DIR": str(root / "private"),
                            "LOCAL_STORE_DIR": str(run),
                        },
                    )
                )
                for name in (
                    "configure_runtime_environment",
                    "initialize_local_store_if_needed",
                    "validate_baseline_cache_for_run",
                    "start_sidecars",
                    "generate_loop_boundary_report",
                ):
                    stack.enter_context(patch(prefix + name))
                stack.enter_context(
                    patch(
                        prefix + "build_legacy_mcp_servers",
                        return_value=SimpleNamespace(
                            servers={}, refs=(), unavailable=[], skipped=[]
                        ),
                    )
                )
                stack.enter_context(patch(prefix + "peer_mcp_context", return_value=({}, [])))
                stack.enter_context(patch(prefix + "run_generation_cohort", cohort))
                write_private_startup_config(
                    run,
                    {
                        "schema_version": "praxist.startup.v1",
                        "canonical_args": {
                            "task": spec.task_id,
                            "task_path": str(task),
                            "run_dir": str(run),
                            "runtime": "agent_runtime:fake",
                            "model_provider": "model_provider:fake",
                            "budget_policy": "budget_policy:fixed",
                            "model": "offline-test",
                        },
                    },
                )
                loop = GenerationLoop(
                    spec, workspace=run, run_dir=run, local_mode=True, task_project_path=task
                )
                result = asyncio.run(asyncio.wait_for(loop.run(), timeout=5))
                if resume_at_ninth_boundary:
                    self.assertEqual(result["status"], "incomplete")
                    self.assertEqual(result["generations_completed"], 9)
                    self.assertFalse((run / "finalize-None.json").exists())
                    (run / "ORCHESTRATOR_SHUTDOWN").unlink()
                    loop = GenerationLoop(
                        spec,
                        workspace=run,
                        run_dir=run,
                        local_mode=True,
                        task_project_path=task,
                        resume=True,
                    )
                    result = asyncio.run(asyncio.wait_for(loop.run(), timeout=5))

            self.assertEqual(cohorts, list(range(9)))
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["exit_condition"], "research_complete")
            self.assertEqual(result["generations_completed"], 9)
            self.assertIsNone(result["max_generations"])
            self.assertNotIn("deadline_at", result)
            self.assertEqual(
                json.loads((run / "finalize-None.json").read_text()),
                [f"Unscored evidence from generation {generation}" for generation in range(9)],
            )
            events = json.loads((run / "callback-events.json").read_text())
            self.assertEqual(events.count("initial-None.json"), 1)
            self.assertEqual(events.count("review-8.json"), 1)
            self.assertEqual(events.count("finalize-None.json"), 1)
            self.assertTrue(loop._task_lifecycle.research_completed)
