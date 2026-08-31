from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from praxist.core.protocol import AgentRunResult
from praxist.plugins.workflow_stages.research_loop.backend.task_lifecycle import TaskLifecycle


class TaskLifecycleTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task_path = self.root / "task"
        self.task_path.mkdir()
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.now = 1000.0
        self.spec = SimpleNamespace(
            research_loop={
                "mode": "scoreless",
                "lifecycle": {
                    "entrypoint": "coordinator.py:handle_lifecycle",
                    "initial_seconds": 60,
                    "finalization_seconds": 120,
                    "config": {"instruction": "retain evidence"},
                },
            },
            run_lifecycle=SimpleNamespace(max_wall_clock_hours=1),
        )
        self.requests = []
        self.handler("""
async def handle_lifecycle(context):
    return {"status": "completed", "artifacts": [], "summary": {}}
""")

    async def run_agent(self, prompt, **kwargs):
        self.requests.append((prompt, kwargs))
        return AgentRunResult(True, [], [], [], None, None, None)

    def lifecycle(self, **kwargs):
        return TaskLifecycle(
            self.spec,
            self.task_path,
            self.run_dir,
            self.run_agent,
            clock=kwargs.pop("clock", lambda: self.now),
            **kwargs,
        )

    def handler(self, source):
        (self.task_path / "coordinator.py").write_text(source, encoding="utf-8")

    def read_state(self):
        return json.loads((self.run_dir / "lifecycle" / "state.json").read_text())

    async def test_deadline_does_not_reset_on_resume_and_reserves_finalization(self):
        lifecycle = self.lifecycle()
        self.assertEqual(lifecycle.deadline_at, 4600)
        self.assertEqual(lifecycle.research_deadline_at, 4480)
        self.now = 1200
        resumed = self.lifecycle()
        self.assertEqual(resumed.deadline_at, 4600)
        self.assertEqual(resumed.remaining_seconds(), 3280)
        self.assertEqual(resumed.remaining_seconds(finalization=True), 3400)

    async def test_resolved_task_spec_runs_the_declared_lifecycle(self):
        import yaml

        from praxist.task_spec import load_task_spec

        descriptor = {
            "task_id": "evidence",
            "research_loop": self.spec.research_loop,
            "run_lifecycle": {"max_wall_clock_hours": 1},
        }
        task_yaml = self.task_path / "task.yaml"
        task_yaml.write_text(yaml.safe_dump(descriptor), encoding="utf-8")
        self.spec = load_task_spec(task_yaml)
        lifecycle = self.lifecycle()
        result = await lifecycle.run_phase("initial")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(lifecycle.research_deadline_at, 4480)

    async def test_raw_configuration_fallback_is_snapshotted(self):
        raw = {"research_loop": self.spec.research_loop}
        self.spec = SimpleNamespace(_raw=raw, run_lifecycle=self.spec.run_lifecycle)
        self.handler("""
async def handle_lifecycle(context):
    return {"status": "completed", "artifacts": [], "summary": {"value": context.config["instruction"]}}
""")
        lifecycle = self.lifecycle()
        raw["research_loop"]["lifecycle"]["config"]["instruction"] = "later mutation"
        result = await lifecycle.run_phase("initial")
        self.assertEqual(result["summary"], {"value": "retain evidence"})

    async def test_scoreless_without_callback_still_persists_deadline(self):
        self.spec.research_loop = {"mode": "scoreless"}
        lifecycle = self.lifecycle()
        self.assertFalse(lifecycle.enabled)
        self.assertEqual(lifecycle.started_at, 1000)
        self.now = 2000
        self.assertEqual(self.lifecycle().deadline_at, 4600)

    async def test_uncapped_lifecycle_has_no_phase_or_agent_timeout(self):
        self.spec.run_lifecycle.max_wall_clock_hours = None
        config = self.spec.research_loop["lifecycle"]
        config.pop("initial_seconds")
        config.pop("finalization_seconds")
        config["after_generation"] = True
        self.handler("""
async def handle_lifecycle(context):
    assert context.deadline_at is None
    assert context.phase_deadline_at is None
    await context.run_agent("continue research", role="review")
    return {"status": "completed", "artifacts": [], "summary": {}}
""")
        lifecycle = self.lifecycle()
        for phase, generation_id in (("initial", None), ("review", 0), ("finalize", None)):
            self.now += 1_000_000_000
            result = await lifecycle.run_phase(phase, generation_id=generation_id)
            self.assertEqual(result["status"], "completed")
        self.assertIsNone(lifecycle.deadline_at)
        self.assertIsNone(lifecycle.research_deadline_at)
        self.assertIsNone(lifecycle.remaining_seconds())
        self.assertIsNone(lifecycle.remaining_seconds(finalization=True))
        self.assertEqual([call[1]["timeout_seconds"] for call in self.requests], [None] * 3)
        state = self.read_state()
        self.assertIsNone(state["deadline_at"])
        self.assertTrue(all(item["phase_deadline_at"] is None for item in state["phases"].values()))
        self.assertNotIn("Infinity", json.dumps(state))
        self.assertIsNone(self.lifecycle().deadline_at)

    async def test_uncapped_finalization_retry_preserves_frozen_inputs(self):
        self.spec.run_lifecycle.max_wall_clock_hours = None
        config = self.spec.research_loop["lifecycle"]
        config["initial_seconds"] = config["finalization_seconds"] = None
        self.handler("""
import json
async def handle_lifecycle(context):
    output = context.run_dir / "delivery.json"
    status = "completed" if output.exists() else "incomplete"
    output.write_text(json.dumps(list(context.findings)))
    return {"status": status, "artifacts": ["delivery.json"], "summary": {}}
""")
        first = await self.lifecycle().run_phase("finalize", [{"content": "frozen"}])
        self.assertEqual(first["status"], "incomplete")
        frozen_digest = self.read_state()["phases"]["finalize"]["input_digest"]
        self.now += 1_000_000_000
        resumed = self.lifecycle()
        result = await resumed.run_phase("finalize", [{"content": "later input"}])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(resumed.finalization_started)
        self.assertEqual(resumed.started_at, 1000)
        self.assertIsNone(resumed.deadline_at)
        self.assertEqual(
            json.loads((self.run_dir / "delivery.json").read_text()), [{"content": "frozen"}]
        )
        self.assertEqual(self.read_state()["phases"]["finalize"]["input_digest"], frozen_digest)
        self.assertIsNone(self.read_state()["phases"]["finalize"]["phase_deadline_at"])

    async def test_uncapped_agent_execution_still_propagates_cancellation(self):
        self.spec.run_lifecycle.max_wall_clock_hours = None
        self.spec.research_loop["lifecycle"]["initial_seconds"] = None
        entered, stopped = asyncio.Event(), asyncio.Event()

        async def waiting_agent(prompt, **kwargs):
            self.assertIsNone(kwargs["timeout_seconds"])
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        self.run_agent = waiting_agent
        self.handler("""
async def handle_lifecycle(context):
    await context.run_agent("continue")
""")
        task = asyncio.create_task(self.lifecycle().run_phase("initial"))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stopped.is_set())
        state = self.read_state()["phases"]["initial"]
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(state["result"]["summary"], {"reason": "cancelled"})
        self.assertIsNone(state["phase_deadline_at"])

    async def test_optional_phase_and_request_caps_still_apply_without_total_deadline(self):
        self.spec.run_lifecycle.max_wall_clock_hours = None
        self.handler("""
async def handle_lifecycle(context):
    await context.run_agent("inspect", timeout_seconds=5)
    return {"status": "completed", "artifacts": [], "summary": {}}
""")
        lifecycle = self.lifecycle()
        result = await lifecycle.run_phase("initial")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.requests[0][1]["timeout_seconds"], 5)
        self.assertEqual(self.read_state()["phases"]["initial"]["phase_deadline_at"], 1060)
        self.assertIsNone(lifecycle.research_deadline_at)
        self.now = 5000
        result = await self.lifecycle().run_phase("finalize")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.read_state()["phases"]["finalize"]["phase_deadline_at"], 5120)

    async def test_bounded_run_with_null_phase_caps_uses_original_total_deadline(self):
        config = self.spec.research_loop["lifecycle"]
        config["initial_seconds"] = config["finalization_seconds"] = None
        lifecycle = self.lifecycle()
        self.assertEqual(lifecycle.research_deadline_at, 4600)
        await lifecycle.run_phase("initial")
        self.assertEqual(self.read_state()["phases"]["initial"]["phase_deadline_at"], 4600)
        self.now = 4500
        resumed = self.lifecycle()
        await resumed.run_phase("finalize")
        self.assertEqual(self.read_state()["phases"]["finalize"]["phase_deadline_at"], 4600)
        self.assertEqual(resumed.remaining_seconds(finalization=True), 100)

    async def test_research_completion_requires_a_committed_review_and_survives_resume(self):
        self.spec.research_loop["lifecycle"]["after_generation"] = True
        self.handler("""
async def handle_lifecycle(context):
    status = "completed"
    if context.phase == "review":
        marker = context.run_dir / "review-attempt"
        status = "completed" if marker.exists() else "incomplete"
        marker.write_text("reviewed")
    return {"status": status, "artifacts": [], "summary": {"research_complete": True}}
""")
        lifecycle = self.lifecycle()
        await lifecycle.run_phase("initial")
        self.assertFalse(lifecycle.research_completed)
        first = await lifecycle.run_phase("review", generation_id=0)
        self.assertEqual(first["status"], "incomplete")
        self.assertFalse(lifecycle.research_completed)
        second = await lifecycle.run_phase("review", generation_id=0)
        self.assertEqual(second["status"], "completed")
        self.assertTrue(lifecycle.research_completed)
        resumed = self.lifecycle()
        self.assertTrue(resumed.research_completed)
        self.assertEqual(await resumed.run_phase("review", generation_id=0), second)
        self.assertEqual(self.read_state()["phases"]["review_gen_0"]["attempts"], 2)

    async def test_final_delivery_summary_does_not_decide_research_completion(self):
        self.handler("""
async def handle_lifecycle(context):
    return {"status": "completed", "artifacts": [], "summary": {"research_complete": True}}
""")
        lifecycle = self.lifecycle()
        await lifecycle.run_phase("finalize")
        self.assertFalse(lifecycle.research_completed)

    async def test_review_research_completion_rejects_non_boolean_values(self):
        self.spec.research_loop["lifecycle"]["after_generation"] = True
        lifecycle = self.lifecycle()
        for value in ("true", 1, None):
            with self.subTest(value=value):
                self.handler(
                    "async def handle_lifecycle(context):\n"
                    "    return {'status': 'completed', 'artifacts': [], "
                    f"'summary': {{'research_complete': {value!r}}}}}\n"
                )
                result = await lifecycle.run_phase("review", generation_id=0)
                self.assertEqual(result["summary"], {"reason": "invalid_result"})
                self.assertFalse(lifecycle.research_completed)

    async def test_private_state_is_authoritative_over_public_mirror(self):
        private = self.root / "private"
        lifecycle = self.lifecycle(state_dir=private)
        public = self.run_dir / "lifecycle" / "state.json"
        public.write_text('{"deadline_at": 99999999}')
        self.now = 2000
        resumed = self.lifecycle(state_dir=private)
        self.assertEqual(resumed.deadline_at, lifecycle.deadline_at)
        self.assertEqual(private.stat().st_mode & 0o777, 0o700)

    async def test_private_phase_inputs_are_not_copied_to_public_mirror(self):
        self.spec.research_loop["lifecycle"]["config"] = {"private_input": "held_by_controller"}
        lifecycle = self.lifecycle(state_dir=self.root / "private")
        await lifecycle.run_phase("initial")
        public = (self.run_dir / "lifecycle" / "state.json").read_text()
        self.assertNotIn("held_by_controller", public)
        self.assertEqual(json.loads(public)["phases"]["initial"]["status"], "committed")

    async def test_corrupt_checkpoint_does_not_create_new_budget(self):
        lifecycle = self.lifecycle()
        original = self.read_state()
        for key, value in (
            ("schema_version", "unknown"),
            ("task_path", "/unrelated"),
            ("started_at", "bad"),
            ("deadline_at", None),
            ("deadline_at", 1),
            ("phases", []),
        ):
            with self.subTest(key=key, value=value):
                changed = dict(original, **{key: value})
                state_path = self.run_dir / "lifecycle" / "state.json"
                state_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    self.lifecycle()
                self.assertEqual(json.loads(state_path.read_text()), changed)
        self.assertEqual(lifecycle.deadline_at, 4600)

    async def test_changed_configuration_cannot_change_existing_phase_contract(self):
        self.lifecycle()
        self.spec.research_loop["lifecycle"]["finalization_seconds"] = 180
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.lifecycle()

    async def test_public_mirror_symlink_cannot_redirect_controller_write(self):
        outside = self.root / "outside-directory"
        outside.mkdir()
        (self.run_dir / "lifecycle").symlink_to(outside, target_is_directory=True)
        lifecycle = self.lifecycle(state_dir=self.root / "private")
        self.assertEqual(lifecycle.deadline_at, 4600)
        self.assertFalse((outside / "state.json").exists())

    async def test_completed_phase_is_not_repeated_and_agent_deadline_is_clamped(self):
        self.handler("""
async def handle_lifecycle(context):
    result = await context.run_agent("inspect", role="review", allowed_tools=["Read"], timeout_seconds=10000)
    assert result.success
    path = context.run_dir / "initial.json"
    path.write_text(context.config["instruction"])
    return {"status": "completed", "artifacts": ["initial.json"], "summary": {"ready": True}}
""")
        lifecycle = self.lifecycle()
        result = await lifecycle.run_phase("initial")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(lifecycle.initial_completed)
        self.assertEqual(
            self.requests,
            [("inspect", {"role": "review", "allowed_tools": ["Read"], "timeout_seconds": 60.0})],
        )
        self.now = 1050
        self.assertEqual(await self.lifecycle().run_phase("initial"), result)
        self.assertEqual(len(self.requests), 1)
        phase = self.read_state()["phases"]["initial"]
        self.assertEqual(phase["status"], "committed")
        self.assertEqual(len(phase["artifact_hashes"]["initial.json"]), 64)

    async def test_invalid_agent_role_preserves_partial_work_without_dispatch(self):
        self.handler("""
async def handle_lifecycle(context):
    (context.run_dir / "partial").write_text("retained")
    await context.run_agent("inspect", role="unconfigured")
    raise AssertionError("invalid role must not return")
""")
        result = await self.lifecycle().run_phase("initial")
        self.assertEqual(
            result["summary"], {"reason": "callback_failed", "error_type": "ValueError"}
        )
        self.assertEqual(self.requests, [])
        self.assertEqual((self.run_dir / "partial").read_text(), "retained")
        self.assertEqual(self.read_state()["phases"]["initial"]["status"], "incomplete")

    async def test_expired_agent_request_is_rejected_before_dispatch(self):
        clock_module = ModuleType("lifecycle_clock_fixture")
        clock_module.advance = lambda: setattr(self, "now", 2000)
        self.handler("""
from lifecycle_clock_fixture import advance
async def handle_lifecycle(context):
    advance()
    await context.run_agent("inspect")
    raise AssertionError("expired request must not return")
""")
        lifecycle = self.lifecycle()
        with patch.dict(sys.modules, {"lifecycle_clock_fixture": clock_module}):
            result = await lifecycle.run_phase("initial")
        self.assertEqual(result["summary"], {"reason": "deadline_exceeded"})
        self.assertEqual(self.requests, [])
        self.assertFalse(lifecycle.initial_completed)

    async def test_agent_result_arriving_after_phase_deadline_cannot_complete_delivery(self):
        async def late_result(prompt, **kwargs):
            self.requests.append((prompt, kwargs))
            self.now = 2000
            return AgentRunResult(True, [], [], [], None, None, None)

        self.run_agent = late_result
        self.handler("""
async def handle_lifecycle(context):
    (context.run_dir / "partial").write_text("retained")
    await context.run_agent("inspect")
    (context.run_dir / "delivered").write_text("must not publish")
    return {"status": "completed", "artifacts": ["delivered"], "summary": {}}
""")
        lifecycle = self.lifecycle()
        result = await lifecycle.run_phase("initial")
        self.assertEqual(result["summary"], {"reason": "deadline_exceeded"})
        self.assertEqual(len(self.requests), 1)
        self.assertEqual((self.run_dir / "partial").read_text(), "retained")
        self.assertFalse((self.run_dir / "delivered").exists())
        self.assertFalse(lifecycle.initial_completed)

    async def test_finalize_retries_frozen_findings_and_does_not_renew_phase_deadline(self):
        self.handler("""
import json
async def handle_lifecycle(context):
    marker = context.run_dir / "attempt"
    attempted = marker.exists()
    marker.write_text("attempted")
    output = context.run_dir / "final.json"
    output.write_text(json.dumps({"findings": context.findings, "deadline": context.phase_deadline_at}))
    return {"status": "completed" if attempted else "incomplete", "artifacts": ["final.json"], "summary": {}}
""")
        lifecycle = self.lifecycle()
        self.assertEqual(lifecycle.final_artifact_hashes(), {})
        self.now = 1200
        first = await lifecycle.run_phase("finalize", [{"id": "evidence", "content": "first"}])
        self.assertTrue(lifecycle.finalization_started)
        self.assertEqual(first["status"], "incomplete")
        self.assertEqual(lifecycle.final_artifact_hashes(), {})
        digest = self.read_state()["phases"]["finalize"]["input_digest"]
        self.now = 1240
        resumed = self.lifecycle()
        second = await resumed.run_phase("finalize", [{"id": "later"}])
        self.assertEqual(second["status"], "completed")
        self.assertTrue(resumed.finalization_completed)
        final = json.loads((self.run_dir / "final.json").read_text())
        self.assertEqual(
            final, {"findings": [{"id": "evidence", "content": "first"}], "deadline": 1320}
        )
        self.assertEqual(self.read_state()["phases"]["finalize"]["input_digest"], digest)

    async def test_changed_frozen_inputs_reject_retry_without_overwriting_partial_delivery(self):
        self.handler("""
async def handle_lifecycle(context):
    (context.run_dir / "partial").write_text(context.findings[0]["content"])
    return {"status": "incomplete", "artifacts": ["partial"], "summary": {}}
""")
        await self.lifecycle().run_phase("finalize", [{"id": "source", "content": "original"}])
        checkpoint = self.read_state()
        checkpoint["phases"]["finalize"]["inputs"]["findings"][0]["content"] = "substituted"
        state_path = self.run_dir / "lifecycle" / "state.json"
        state_path.write_text(json.dumps(checkpoint))
        self.now = 1040
        resumed = self.lifecycle()
        with self.assertRaisesRegex(ValueError, "frozen input digest"):
            await resumed.run_phase("finalize", [{"id": "new-input"}])
        self.assertEqual(resumed.deadline_at, 4600)
        self.assertEqual(self.read_state(), checkpoint)
        self.assertEqual((self.run_dir / "partial").read_text(), "original")
        self.assertEqual(resumed.final_artifact_hashes(), {})

    async def test_generation_reviews_are_distinct_and_committed_reviews_are_not_repeated(self):
        self.spec.research_loop["lifecycle"]["after_generation"] = True
        self.handler("""
import json
async def handle_lifecycle(context):
    assert context.phase == "review"
    output = context.run_dir / f"review-{context.generation_id}.json"
    assert not output.exists(), "committed review must not execute twice"
    output.write_text(json.dumps({"ids": [finding["id"] for finding in context.findings], "deadline": context.phase_deadline_at}))
    return {"status": "completed", "artifacts": [output.name], "summary": {"generation": context.generation_id}}
""")
        lifecycle = self.lifecycle()
        self.assertTrue(lifecycle.after_generation)
        first = await lifecycle.run_phase("review", [{"id": "first"}], generation_id=0)
        self.now = 2000
        second = await lifecycle.run_phase("review", [{"id": "second"}], generation_id=1)
        resumed = self.lifecycle()
        self.assertEqual(
            await resumed.run_phase("review", [{"id": "ignored"}], generation_id=0), first
        )
        self.assertEqual(await resumed.run_phase("review", [], generation_id=1), second)
        self.assertFalse(resumed.finalization_started)
        phases = self.read_state()["phases"]
        self.assertEqual(set(phases), {"review_gen_0", "review_gen_1"})
        self.assertEqual(
            [(event["phase"], event["generation_id"]) for event in self.read_state()["events"]],
            [("review", 0), ("review", 0), ("review", 1), ("review", 1)],
        )
        self.assertEqual(
            json.loads((self.run_dir / "review-0.json").read_text()),
            {"ids": ["first"], "deadline": 4480},
        )
        self.assertEqual(
            json.loads((self.run_dir / "review-1.json").read_text()),
            {"ids": ["second"], "deadline": 4480},
        )

    async def test_incomplete_generation_review_retries_original_inputs(self):
        self.spec.research_loop["lifecycle"]["after_generation"] = True
        self.handler("""
async def handle_lifecycle(context):
    marker = context.run_dir / "review-attempt"
    retried = marker.exists()
    marker.write_text(context.findings[0]["id"])
    return {"status": "completed" if retried else "incomplete", "artifacts": [marker.name], "summary": {"id": context.findings[0]["id"]}}
""")
        first = await self.lifecycle().run_phase("review", [{"id": "frozen"}], generation_id=2)
        self.assertEqual(first["status"], "incomplete")
        second = await self.lifecycle().run_phase("review", [{"id": "new"}], generation_id=2)
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["summary"], {"id": "frozen"})

    async def test_generation_review_cannot_consume_finalization_reserve(self):
        self.spec.research_loop["lifecycle"]["after_generation"] = True
        self.handler("""
async def handle_lifecycle(context):
    (context.run_dir / "unexpected-review").write_text("bad")
    return {"status": "completed", "artifacts": [], "summary": {}}
""")
        lifecycle = self.lifecycle()
        self.now = 4481
        result = await lifecycle.run_phase("review", generation_id=0)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["summary"]["reason"], "deadline_exceeded")
        self.assertEqual(lifecycle.remaining_seconds(finalization=True), 119)
        self.assertFalse((self.run_dir / "unexpected-review").exists())

    async def test_generation_review_is_opt_in_and_requires_valid_generation(self):
        lifecycle = self.lifecycle()
        self.assertFalse(lifecycle.after_generation)
        result = await lifecycle.run_phase("review", generation_id=0)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.read_state()["phases"], {})
        for generation_id in (None, -1, True, 1.5):
            with self.subTest(generation_id=generation_id), self.assertRaises(ValueError):
                await lifecycle.run_phase("review", generation_id=generation_id)

    async def test_invalid_phase_identity_does_not_create_a_phase_checkpoint(self):
        lifecycle = self.lifecycle()
        for phase, generation_id in (("unknown", None), ("initial", 0), ("finalize", 0)):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                await lifecycle.run_phase(phase, generation_id=generation_id)
        self.assertEqual(self.read_state()["phases"], {})
        self.assertEqual(self.read_state()["events"], [])
        self.assertEqual(self.requests, [])

    async def test_new_review_cannot_run_after_finalization_inputs_are_frozen(self):
        self.spec.research_loop["lifecycle"]["after_generation"] = True
        self.handler("""
async def handle_lifecycle(context):
    if context.phase == "review":
        (context.run_dir / "unexpected-review").write_text("bad")
    return {"status": "incomplete", "artifacts": [], "summary": {}}
""")
        lifecycle = self.lifecycle()
        await lifecycle.run_phase("finalize")
        review = await lifecycle.run_phase("review", generation_id=0)
        self.assertEqual(review["status"], "incomplete")
        self.assertEqual(review["summary"]["reason"], "finalization_started")
        self.assertFalse((self.run_dir / "unexpected-review").exists())
        self.assertEqual(set(self.read_state()["phases"]), {"finalize"})

    async def test_expired_phase_never_invokes_handler(self):
        self.handler("""
async def handle_lifecycle(context):
    (context.run_dir / "unexpected").write_text("bad")
    return {"status": "completed", "artifacts": [], "summary": {}}
""")
        lifecycle = self.lifecycle()
        self.now = 5000
        result = await lifecycle.run_phase("finalize")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["summary"]["reason"], "deadline_exceeded")
        self.assertFalse((self.run_dir / "unexpected").exists())

    async def test_phase_timeout_preserves_partial_output(self):
        self.spec.research_loop["lifecycle"]["initial_seconds"] = 0.01
        self.handler("""
import asyncio
async def handle_lifecycle(context):
    (context.run_dir / "partial").write_text("valuable")
    await asyncio.sleep(10)
    return {"status": "completed", "artifacts": [], "summary": {}}
""")
        result = await self.lifecycle().run_phase("initial")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["summary"]["reason"], "deadline_exceeded")
        self.assertEqual((self.run_dir / "partial").read_text(), "valuable")

    async def test_cancellation_propagates_with_recoverable_state(self):
        self.handler("""
import asyncio
async def handle_lifecycle(context):
    raise asyncio.CancelledError()
""")
        with self.assertRaises(asyncio.CancelledError):
            await self.lifecycle().run_phase("initial")
        state = self.read_state()["phases"]["initial"]
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(state["result"]["summary"]["reason"], "cancelled")

    async def test_checkpoint_failure_does_not_mask_cancellation(self):
        from praxist.plugins.workflow_stages.research_loop.backend import task_lifecycle

        self.handler("""
import asyncio
async def handle_lifecycle(context):
    raise asyncio.CancelledError()
""")
        lifecycle = self.lifecycle()
        original = task_lifecycle.atomic_write_json

        def fail_incomplete(path, data):
            if data["phases"]["initial"]["status"] == "incomplete":
                raise OSError("disk unavailable")
            return original(path, data)

        with (
            patch.object(task_lifecycle, "atomic_write_json", side_effect=fail_incomplete),
            self.assertRaises(asyncio.CancelledError),
        ):
            await lifecycle.run_phase("initial")

    async def test_late_completion_after_cancellation_is_incomplete(self):
        self.spec.research_loop["lifecycle"]["initial_seconds"] = 0.01
        self.handler("""
import asyncio
async def handle_lifecycle(context):
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        pass
    return {"status": "completed", "artifacts": [], "summary": {}}
""")
        import time

        lifecycle = self.lifecycle(clock=time.time)
        result = await lifecycle.run_phase("initial")
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(lifecycle.initial_completed)

    async def test_import_that_exhausts_phase_budget_does_not_start_callback(self):
        clock_module = ModuleType("lifecycle_clock_fixture")
        clock_module.advance = lambda: setattr(self, "now", 2000)
        self.handler("""
from lifecycle_clock_fixture import advance
advance()
async def handle_lifecycle(context):
    (context.run_dir / "unexpected").write_text("bad")
    return {"status": "completed", "artifacts": [], "summary": {}}
""")
        lifecycle = self.lifecycle()
        with patch.dict(sys.modules, {"lifecycle_clock_fixture": clock_module}):
            result = await lifecycle.run_phase("initial")
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse((self.run_dir / "unexpected").exists())

    async def test_artifact_validation_that_exhausts_budget_cannot_commit(self):
        self.handler("""
async def handle_lifecycle(context):
    (context.run_dir / "artifact").write_text("valuable")
    return {"status": "completed", "artifacts": ["artifact"], "summary": {}}
""")
        lifecycle = self.lifecycle()
        file_digest = hashlib.file_digest

        def slow_read(stream, digest):
            data = file_digest(stream, digest)
            self.now = 2000
            return data

        with patch.object(hashlib, "file_digest", slow_read):
            result = await lifecycle.run_phase("initial")
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(lifecycle.initial_completed)
        self.assertEqual((self.run_dir / "artifact").read_text(), "valuable")

    async def test_artifact_path_escape_and_symlink_are_rejected(self):
        (self.root / "outside").write_text("private")
        (self.run_dir / "link").symlink_to(self.root / "outside")
        for name in ("../outside", str(self.root / "outside"), "link"):
            with self.subTest(name=name):
                self.handler(f"""
async def handle_lifecycle(context):
    return {{"status": "completed", "artifacts": [{name!r}], "summary": {{}}}}
""")
                result = await self.lifecycle().run_phase("initial")
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["summary"]["reason"], "invalid_result")

    async def test_artifact_directory_swap_never_opens_private_file(self):
        await self.assert_artifact_swap_is_rejected("directory")

    async def test_artifact_leaf_swap_never_opens_private_file(self):
        await self.assert_artifact_swap_is_rejected("leaf")

    async def test_run_directory_swap_never_opens_private_file(self):
        await self.assert_artifact_swap_is_rejected("root")

    async def assert_artifact_swap_is_rejected(self, component):
        public = self.run_dir / "artifacts"
        public.mkdir()
        (public / "result.json").write_text("public evidence")
        private = self.root / "private"
        private.mkdir()
        if component == "root":
            (private / "artifacts").mkdir()
            secret = private / "artifacts/result.json"
        else:
            secret = private / "result.json"
        secret.write_text("controller-private bytes")
        secret_identity = (secret.stat().st_dev, secret.stat().st_ino)
        self.handler("""
async def handle_lifecycle(context):
    return {"status": "completed", "artifacts": ["artifacts/result.json"], "summary": {}}
""")
        lifecycle = self.lifecycle(state_dir=self.root / "control")
        swapped = False
        private_opened = False
        original_io_open, original_os_open = io.open, os.open

        def swap():
            nonlocal swapped
            if not swapped:
                if component == "root":
                    self.run_dir.rename(self.root / "retained-run")
                    self.run_dir.symlink_to(private, target_is_directory=True)
                elif component == "leaf":
                    leaf = public / "result.json"
                    leaf.rename(public / "retained-result.json")
                    leaf.symlink_to(secret)
                else:
                    public.rename(self.run_dir / "retained-artifacts")
                    public.symlink_to(private, target_is_directory=True)
                swapped = True

        def track(fd):
            nonlocal private_opened
            info = os.fstat(fd)
            private_opened |= (info.st_dev, info.st_ino) == secret_identity

        def unsafe_path_boundary(path, *args, **kwargs):
            if (
                not isinstance(path, int)
                and Path(path) == lifecycle.run_dir / "artifacts/result.json"
            ):
                swap()
            stream = original_io_open(path, *args, **kwargs)
            track(stream.fileno())
            return stream

        def descriptor_boundary(path, flags, *args, **kwargs):
            trigger = {"root": self.run_dir.name, "directory": "artifacts", "leaf": "result.json"}[
                component
            ]
            if path == trigger and kwargs.get("dir_fd") is not None:
                swap()
            fd = original_os_open(path, flags, *args, **kwargs)
            track(fd)
            return fd

        with (
            patch.object(io, "open", unsafe_path_boundary),
            patch.object(os, "open", descriptor_boundary),
        ):
            result = await lifecycle.run_phase("initial")
        self.assertTrue(swapped)
        self.assertFalse(private_opened)
        self.assertEqual(result["status"], "incomplete")

    async def test_artifact_special_file_is_rejected_without_blocking(self):
        os.mkfifo(self.run_dir / "pipe")
        self.handler("""
async def handle_lifecycle(context):
    return {"status": "completed", "artifacts": ["pipe"], "summary": {}}
""")
        result = await self.lifecycle().run_phase("initial")
        self.assertEqual(result["status"], "incomplete")

    async def test_committed_artifact_corruption_is_not_silently_reaccepted(self):
        self.handler("""
async def handle_lifecycle(context):
    (context.run_dir / "artifact").write_text("original")
    return {"status": "completed", "artifacts": ["artifact"], "summary": {}}
""")
        await self.lifecycle().run_phase("initial")
        (self.run_dir / "artifact").write_text("changed")
        with self.assertRaisesRegex(ValueError, "artifact"):
            await self.lifecycle().run_phase("initial")
        self.assertEqual((self.run_dir / "artifact").read_text(), "changed")

    async def test_local_imports_work_and_import_state_is_restored(self):
        (self.task_path / "local_helper_unique.py").write_text('VALUE = "local"\n')
        self.handler("""
async def handle_lifecycle(context):
    from local_helper_unique import VALUE
    return {"status": "completed", "artifacts": [], "summary": {"value": VALUE}}
""")
        old_path = list(sys.path)
        result = await self.lifecycle().run_phase("initial")
        self.assertEqual(result["summary"], {"value": "local"})
        self.assertEqual(sys.path, old_path)
        self.assertNotIn("local_helper_unique", sys.modules)

    async def test_cached_module_from_another_project_does_not_replace_local_helper(self):
        cached = ModuleType("local_helper_unique")
        cached.VALUE = "wrong project"
        (self.task_path / "local_helper_unique.py").write_text('VALUE = "this project"\n')
        self.handler("""
async def handle_lifecycle(context):
    from local_helper_unique import VALUE
    return {"status": "completed", "artifacts": [], "summary": {"value": VALUE}}
""")
        with patch.dict(sys.modules, {"local_helper_unique": cached}):
            result = await self.lifecycle().run_phase("initial")
            self.assertIs(sys.modules["local_helper_unique"], cached)
        self.assertEqual(result["summary"], {"value": "this project"})

    async def test_import_failure_is_incomplete_and_cleans_import_state(self):
        self.handler("""
raise RuntimeError("must not copy arbitrary exception details")
""")
        old_path = list(sys.path)
        result = await self.lifecycle().run_phase("initial")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(
            result["summary"], {"reason": "callback_failed", "error_type": "RuntimeError"}
        )
        self.assertEqual(sys.path, old_path)
        self.assertFalse(any(name.startswith("_praxist_task_lifecycle_") for name in sys.modules))

    async def test_nested_callback_cancellation_restores_imports_and_can_resume(self):
        nested = self.task_path / "hooks"
        nested.mkdir()
        (nested / "lifecycle_nested_helper.py").write_text('VALUE = "task evidence"\n')
        (nested / "coordinator.py").write_text("""
import asyncio
from lifecycle_nested_helper import VALUE
async def handle_lifecycle(context):
    partial = context.run_dir / "partial"
    if not partial.exists():
        partial.write_text(VALUE)
        raise asyncio.CancelledError()
    return {"status": "completed", "artifacts": ["partial"], "summary": {"value": VALUE}}
""")
        self.spec.research_loop["lifecycle"]["entrypoint"] = "hooks/coordinator.py:handle_lifecycle"
        cached = ModuleType("lifecycle_nested_helper")
        cached.VALUE = "unrelated project"
        original_path = list(sys.path)
        with patch.dict(sys.modules, {"lifecycle_nested_helper": cached}):
            with self.assertRaises(asyncio.CancelledError):
                await self.lifecycle().run_phase("initial")
            self.assertEqual(sys.path, original_path)
            self.assertIs(sys.modules["lifecycle_nested_helper"], cached)
            self.assertEqual((self.run_dir / "partial").read_text(), "task evidence")
            self.assertEqual(self.read_state()["phases"]["initial"]["status"], "incomplete")
            self.now = 1040
            resumed = await self.lifecycle().run_phase("initial")
            self.assertEqual(resumed["summary"], {"value": "task evidence"})
            self.assertEqual(sys.path, original_path)
            self.assertIs(sys.modules["lifecycle_nested_helper"], cached)
        self.assertEqual(self.read_state()["phases"]["initial"]["attempts"], 2)

    async def test_synchronous_handler_is_rejected_without_invocation(self):
        self.handler("""
def handle_lifecycle(context):
    (context.run_dir / "unexpected").write_text("bad")
    return {"status": "completed", "artifacts": [], "summary": {}}
""")
        result = await self.lifecycle().run_phase("initial")
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse((self.run_dir / "unexpected").exists())

    async def test_invalid_deliveries_are_not_committed(self):
        for payload in (
            None,
            {"status": "succeeded", "artifacts": [], "summary": {}},
            {"status": "completed", "artifacts": "file", "summary": {}},
            {"status": "completed", "artifacts": [7], "summary": {}},
            {"status": "completed", "artifacts": ["missing"], "summary": {}},
        ):
            with self.subTest(payload=payload):
                self.handler(f"async def handle_lifecycle(context):\n    return {payload!r}\n")
                result = await self.lifecycle().run_phase("initial")
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["summary"]["reason"], "invalid_result")

    async def test_invalid_budgets_are_rejected_before_checkpoint_creation(self):
        for name, value in (
            ("initial_seconds", True),
            ("initial_seconds", 0),
            ("initial_seconds", float("inf")),
            ("finalization_seconds", -1),
            ("finalization_seconds", 3600),
            ("config", []),
            ("after_generation", "true"),
        ):
            with self.subTest(name=name, value=value):
                original = dict(self.spec.research_loop["lifecycle"])
                self.spec.research_loop["lifecycle"][name] = value
                with self.assertRaises(ValueError):
                    self.lifecycle()
                self.assertFalse((self.run_dir / "lifecycle").exists())
                self.spec.research_loop["lifecycle"] = original
        self.spec.run_lifecycle.max_wall_clock_hours = 0
        with self.assertRaisesRegex(ValueError, "max_wall_clock_hours"):
            self.lifecycle()

    async def test_entrypoint_must_be_file_inside_explicit_task_root(self):
        outside = self.root / "outside.py"
        outside.write_text("raise AssertionError('outside source must never be loaded')\n")
        (self.task_path / "linked.py").symlink_to(outside)
        for entrypoint in (
            "../outside.py:handler",
            "os:system",
            "/tmp/code.py:handler",
            "missing.py:handler",
            "linked.py:handler",
            "coordinator.py:handler:extra",
            None,
        ):
            with self.subTest(entrypoint=entrypoint):
                self.spec.research_loop["lifecycle"]["entrypoint"] = entrypoint
                with self.assertRaises(ValueError):
                    self.lifecycle()
                self.assertFalse((self.run_dir / "lifecycle").exists())

    async def test_disabled_lifecycle_preserves_existing_mode_without_new_artifacts(self):
        self.spec.research_loop = {"mode": "metric"}
        lifecycle = self.lifecycle()
        self.assertFalse(lifecycle.enabled)
        self.assertIsNone(lifecycle.deadline_at)
        result = await lifecycle.run_phase("initial")
        self.assertEqual(
            result, {"status": "completed", "artifacts": [], "summary": {"lifecycle": "disabled"}}
        )
        self.assertFalse((self.run_dir / "lifecycle").exists())


if __name__ == "__main__":
    unittest.main()
