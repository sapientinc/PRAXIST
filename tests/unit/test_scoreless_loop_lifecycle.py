from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
import unittest
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import yaml

from praxist.plugins.workflow_stages.research_loop.backend import task_lifecycle_execution
from praxist.plugins.workflow_stages.research_loop.backend.generation_loop import GenerationLoop
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import (
    write_scoreless_evidence_manifest,
)
from praxist.plugins.workflow_stages.research_loop.backend.task_lifecycle import TaskLifecycle
from praxist.task_spec import load_task_spec


class ScorelessLoopLifecycleTest(unittest.TestCase):
    def test_two_generation_delivery_runs_before_teardown_with_all_findings(self):
        self._exercise_loop()

    def test_external_stop_before_initial_starts_no_task_or_peer_agents(self):
        self._exercise_loop(stop_before_initial=True)

    def test_external_stop_after_cohort_preserves_evidence_without_final_agent(self):
        self._exercise_loop(stop_after_cohort=True)

    def _exercise_loop(self, *, stop_before_initial=False, stop_after_cohort=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            task = root / "task"
            task.mkdir()
            run = root / "run"
            (task / "coordinator.py").write_text(
                "import json\n"
                "async def handle(ctx):\n"
                "    assert (ctx.run_dir / 'orchestrator.lock').exists()\n"
                "    path = ctx.run_dir / (ctx.phase + '.json')\n"
                "    path.write_text(json.dumps({'phase':ctx.phase, 'contents':[f['content'] for f in ctx.findings], 'types':[f['finding_type'] for f in ctx.findings]}))\n"
                "    return {'status':'completed','artifacts':[path.name],'summary':{'phase':ctx.phase}}\n",
                encoding="utf-8",
            )
            (task / "task.yaml").write_text(
                yaml.safe_dump(
                    {
                        "task_id": "evidence",
                        "research_loop": {
                            "mode": "scoreless",
                            "lifecycle": {
                                "entrypoint": "coordinator.py:handle",
                                "initial_seconds": 30,
                                "finalization_seconds": 30,
                            },
                        },
                        "run_lifecycle": {"max_wall_clock_hours": 1},
                        "generation_policy": {
                            "max_generations": 2,
                            "cohort_size": 1,
                            "per_generation_hours": 1,
                        },
                        "synthesis_trigger": {"enabled": False},
                        "pi_agent": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            spec = load_task_spec(task / "task.yaml")

            async def cohort(loop, gen_id):
                self.assertTrue((run / "initial.json").exists())
                (run / f"gen_{gen_id}").mkdir(exist_ok=True)
                finding = {
                    "id": f"finding-{gen_id}",
                    "generation_id": gen_id,
                    "peer_id": f"gen{gen_id}_peer0",
                    "finding_type": "challenge" if gen_id else "hypothesis",
                    "title": "Unresolved claim",
                    "content": f"Evidence from generation {gen_id}",
                    "metrics": {},
                }
                (loop.findings_dir / f"finding-{gen_id}.json").write_text(json.dumps(finding))
                if stop_after_cohort:
                    (run / "ORCHESTRATOR_SHUTDOWN").write_text("operator stop")
                return [{"peer_id": finding["peer_id"], "success": True}]

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
                loop = GenerationLoop(
                    spec, workspace=run, run_dir=run, local_mode=True, task_project_path=task
                )
                if stop_before_initial:
                    (run / "ORCHESTRATOR_SHUTDOWN").write_text("operator stop")
                result = asyncio.run(loop.run())
            if stop_before_initial or stop_after_cohort:
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["exit_condition"], "external_stop_signal")
                self.assertFalse((run / "finalize.json").exists())
                self.assertEqual((run / "initial.json").exists(), stop_after_cohort)
                self.assertEqual(result["generations_completed"], int(stop_after_cohort))
                if stop_after_cohort:
                    self.assertTrue((run / "gen_0" / "scoreless_evidence.json").exists())
                return
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["generations_completed"], 2)
            self.assertEqual(result["evaluation_status"], "not_configured")
            final = json.loads((run / "finalize.json").read_text())
            self.assertEqual(
                final["contents"], ["Evidence from generation 0", "Evidence from generation 1"]
            )
            self.assertEqual(final["types"], ["hypothesis", "challenge"])
            self.assertFalse((run / "orchestrator.lock").exists())
            self.assertEqual(result["task_delivery"]["status"], "completed")
            self.assertEqual(
                result["task_delivery"]["artifact_hashes"],
                {"finalize.json": hashlib.sha256((run / "finalize.json").read_bytes()).hexdigest()},
            )


class ScorelessLifecycleRecoveryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.task = self.root / "task"
        self.run = self.root / "run"
        self.task.mkdir()
        self.run.mkdir()
        self.now = time.time()
        self.started_at = self.now
        self._handler(
            "async def handle(ctx):\n"
            "    return {'status': 'completed', 'artifacts': [], 'summary': {}}\n"
        )
        (self.task / "task.yaml").write_text(
            yaml.safe_dump(
                {
                    "task_id": "lifecycle-recovery",
                    "research_loop": {
                        "mode": "scoreless",
                        "lifecycle": {
                            "entrypoint": "coordinator.py:handle",
                            "initial_seconds": 30,
                            "finalization_seconds": 30,
                            "after_generation": True,
                        },
                    },
                    "run_lifecycle": {"max_wall_clock_hours": 1},
                    "generation_policy": {
                        "max_generations": 3,
                        "cohort_size": 1,
                        "per_generation_hours": 1,
                    },
                    "synthesis_trigger": {"enabled": False},
                    "pi_agent": {"enabled": False},
                }
            ),
            encoding="utf-8",
        )
        self.spec = load_task_spec(self.task / "task.yaml")
        self.loop = SimpleNamespace(
            task_spec=self.spec,
            run_dir=self.run,
            _generations_completed=0,
            _lifecycle_phase="",
        )
        for name in ("_check_task_stop", "_frozen_findings_through", "_run_task_generation_review"):
            setattr(self.loop, name, MethodType(getattr(GenerationLoop, name), self.loop))
        self._restore_lifecycle()

    def _handler(self, source):
        (self.task / "coordinator.py").write_text(source, encoding="utf-8")

    def _restore_lifecycle(self):
        async def no_model_calls(*args, **kwargs):
            raise AssertionError("recovery tests must not dispatch model calls")

        self.loop._task_lifecycle = TaskLifecycle(
            self.spec,
            self.task,
            self.run,
            no_model_calls,
            clock=lambda: self.now,
            state_dir=self.root / "private",
        )

    def _findings(self, generation):
        write_scoreless_evidence_manifest(
            self.run,
            gen_id=generation,
            findings=[{"id": f"frozen-{generation}", "content": f"Evidence {generation}"}],
            evidence_cutoff_at=datetime.now(UTC).isoformat(),
            evidence_source_snapshot={},
        )

    async def test_resume_retries_pending_review_without_rewriting_committed_review(self):
        self._findings(0)
        self._findings(1)
        self._handler(
            "import json\n"
            "async def handle(ctx):\n"
            "    events = ctx.run_dir / 'review-events.json'\n"
            "    prior = json.loads(events.read_text()) if events.exists() else []\n"
            "    attempts = prior.count(ctx.generation_id)\n"
            "    events.write_text(json.dumps(prior + [ctx.generation_id]))\n"
            "    path = ctx.run_dir / f'review-{ctx.generation_id}.json'\n"
            "    path.write_text(json.dumps([item['id'] for item in ctx.findings]))\n"
            "    status = 'incomplete' if ctx.generation_id == 1 and attempts == 0 else 'completed'\n"
            "    return {'status': status, 'artifacts': [path.name], 'summary': {}}\n"
        )
        await task_lifecycle_execution.review_generation_if_enabled(
            self.loop, 0, self.started_at, 1
        )
        committed = (self.run / "review-0.json").read_bytes()
        await task_lifecycle_execution.review_generation_if_enabled(
            self.loop, 1, self.started_at, 2
        )
        self.assertEqual(self.loop._lifecycle_phase, "")
        self.assertEqual((self.run / "review-0.json").read_bytes(), committed)
        (self.run / "shared_findings").mkdir()
        (self.run / "shared_findings/changed.json").write_text('{"id":"new-unfrozen-input"}')

        self._restore_lifecycle()
        successor = await task_lifecycle_execution.review_completed_generations(
            self.loop, self.started_at, completed=2, start_generation=2
        )

        self.assertEqual(successor, 2)
        self.assertEqual(json.loads((self.run / "review-events.json").read_text()), [0, 1, 1])
        self.assertEqual(
            json.loads((self.run / "review-1.json").read_text()), ["frozen-0", "frozen-1"]
        )
        self.assertEqual((self.run / "review-0.json").read_bytes(), committed)
        self.assertEqual(self.loop._lifecycle_phase, "")

    async def test_missing_frozen_generation_prevents_review_callback(self):
        self._findings(0)
        self._handler(
            "async def handle(ctx):\n"
            "    (ctx.run_dir / 'unexpected-review').write_text('called')\n"
            "    return {'status': 'completed', 'artifacts': [], 'summary': {}}\n"
        )
        with self.assertRaisesRegex(RuntimeError, "generation 1 is unavailable"):
            await task_lifecycle_execution.review_generation_if_enabled(
                self.loop, 1, self.started_at, 2
            )
        self.assertFalse((self.run / "unexpected-review").exists())
        self.assertEqual(self.loop._lifecycle_phase, "")

    async def test_resume_verifies_terminal_review_before_accepting_completion(self):
        self._findings(0)
        self._findings(1)
        self._handler(
            "async def handle(ctx):\n"
            "    if ctx.phase == 'finalize':\n"
            "        return {'status': 'incomplete', 'artifacts': [], 'summary': {}}\n"
            "    name = f'review-{ctx.generation_id}.txt'\n"
            "    (ctx.run_dir / name).write_text('committed evidence')\n"
            "    return {'status': 'completed', 'artifacts': [name], "
            "'summary': {'research_complete': ctx.generation_id == 1}}\n"
        )
        await self.loop._run_task_generation_review(0)
        await self.loop._run_task_generation_review(1)
        for finalization_started in (False, True):
            with self.subTest(finalization_started=finalization_started):
                (self.run / "review-1.txt").write_text("committed evidence")
                if finalization_started:
                    with self.assertRaises(task_lifecycle_execution.TaskDeliveryIncomplete):
                        await task_lifecycle_execution.run_final_task_phase(
                            self.loop, self.started_at, 2
                        )
                (self.run / "review-1.txt").write_text("modified terminal decision evidence")
                self._restore_lifecycle()
                self.assertTrue(self.loop._task_lifecycle.research_completed)
                with self.assertRaisesRegex(ValueError, "artifact content changed"):
                    await task_lifecycle_execution.review_completed_generations(
                        self.loop, self.started_at, completed=2, start_generation=2
                    )

    async def test_incomplete_initial_delivery_keeps_artifact_and_failure_status(self):
        self._handler(
            "async def handle(ctx):\n"
            "    (ctx.run_dir / 'partial.txt').write_text('useful partial evidence')\n"
            "    return {'status': 'incomplete', 'artifacts': ['partial.txt'], 'summary': {'reason':'needs_review'}}\n"
        )
        with self.assertRaises(task_lifecycle_execution.TaskDeliveryIncomplete) as caught:
            await task_lifecycle_execution.run_initial_task_phase(self.loop, self.started_at)
        summary = {}
        task_lifecycle_execution.add_task_delivery_summary(
            self.loop, summary, caught.exception.result
        )
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["exit_code"], 1)
        self.assertEqual(summary["task_delivery"]["artifacts"], ["partial.txt"])
        self.assertEqual((self.run / "partial.txt").read_text(), "useful partial evidence")
        self.assertFalse(self.loop._task_lifecycle.initial_completed)
        self.assertEqual(self.loop._lifecycle_phase, "")

    async def test_incomplete_finalization_resume_does_not_restart_research_or_reviews(self):
        self._findings(0)
        self._handler(
            "async def handle(ctx):\n"
            "    (ctx.run_dir / f'{ctx.phase}-attempt').write_text('started')\n"
            "    return {'status': 'incomplete', 'artifacts': [], 'summary': {'reason':'needs_review'}}\n"
        )
        with self.assertRaises(task_lifecycle_execution.TaskDeliveryIncomplete):
            await task_lifecycle_execution.run_final_task_phase(self.loop, self.started_at, 1)
        self._restore_lifecycle()
        next_generation = await task_lifecycle_execution.review_completed_generations(
            self.loop, self.started_at, completed=1, start_generation=1
        )
        self.assertEqual(next_generation, 3)
        self.assertFalse((self.run / "review-attempt").exists())
        self.assertFalse(self.loop._task_lifecycle.finalization_completed)
        self.assertEqual(self.loop._lifecycle_phase, "")

    async def test_expired_research_stops_before_spending_finalization_reserve(self):
        self.now = self.started_at + 3590
        reason, report = task_lifecycle_execution.check_generation_start(
            self.loop, self.started_at, gen_id=1, completed=1
        )
        self.assertEqual(reason, "research_deadline")
        self.assertIsNone(report)
        self.assertEqual(self.loop._task_lifecycle.remaining_seconds(finalization=True), 10)

    async def test_expired_cohort_recovers_results_saved_during_cancellation(self):
        self.now = self.started_at + 3569.99
        interrupted = asyncio.Event()

        async def cohort(generation):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                gen_dir = self.run / f"gen_{generation}"
                gen_dir.mkdir()
                (gen_dir / "generation_results.json").write_text(
                    '[{"peer_id":"gen0_peer0","partial_evidence":"saved on interruption"}]'
                )
                interrupted.set()
                raise

        self.loop._run_generation = cohort
        results, deadline_reached = await task_lifecycle_execution.run_generation_with_deadline(
            self.loop, 0
        )
        self.assertTrue(interrupted.is_set())
        self.assertTrue(deadline_reached)
        self.assertEqual(
            results, [{"peer_id": "gen0_peer0", "partial_evidence": "saved on interruption"}]
        )

    async def test_runtime_timeout_without_research_deadline_is_not_swallowed(self):
        async def cohort(generation):
            raise TimeoutError("runtime transport timed out")

        self.loop._run_generation = cohort
        self.loop._task_lifecycle = SimpleNamespace(
            remaining_seconds=lambda: None, research_deadline_at=None
        )
        with self.assertRaisesRegex(TimeoutError, "runtime transport timed out"):
            await task_lifecycle_execution.run_generation_with_deadline(self.loop, 0)


if __name__ == "__main__":
    unittest.main()
