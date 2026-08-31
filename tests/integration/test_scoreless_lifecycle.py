"""Offline public-launcher coverage of scoreless research and delivery recovery."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml

from praxist.cli.status import pid_is_alive
from tests.helpers.paths import REPO_ROOT

_RUNTIME_SOURCE = """
import json
from pathlib import Path

from praxist.core.protocol import AgentEvent, AgentRunResult
from praxist.plugins.tools.evaluation_tools.adapter import _handle_share_finding


class EvidenceRuntime:
    runtime_ref = "agent_runtime:fake_runtime"

    async def execute(self, request, context):
        run = Path(request.runtime_options["run_dir"])
        role = request.runtime_options.get("execution_role", "research")
        prompt = request.prompt_ref["text"]
        entry = {"role": role, "prompt": prompt, "request_id": request.request_id,
                 "generation_id": context.env.get("GENERATION_ID"),
                 "timeout_seconds": request.timeout_seconds,
                 "tool_execution_timeout_seconds": request.runtime_options.get("tool_execution_timeout_seconds")}
        with (run / "fixture_runtime_calls.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\\n")
        if role in {"final", "review"}:
            inputs = json.loads(prompt)
            failed_once = run / "fixture_final_attempted"
            if role == "final" and inputs["fail_once"] and not failed_once.exists():
                failed_once.touch()
                return AgentRunResult(success=False, events=[], text_output_refs=[], tool_uses=[],
                                      error="temporary delivery failure", failover_reason=None, credential_ref=None)
            output = {"finding_ids": inputs["finding_ids"], "finding_types": inputs["finding_types"]}
        else:
            peer_id = context.env["PEER_ID"]
            gen_id = int(context.env["GENERATION_ID"])
            if gen_id > 0:
                reviewed = json.loads((run / f"review_gen_{gen_id - 1}.json").read_text())
                if not reviewed["finding_ids"]:
                    raise AssertionError("successor received an empty review")
                (run / f"review_seen_by_gen_{gen_id}.json").write_text(json.dumps(reviewed))
            kind = "hypothesis" if gen_id == 0 else "challenge"
            response = await _handle_share_finding({
                "finding_type": kind,
                "title": "Unresolved mechanism" if gen_id == 0 else "Competing explanation",
                "content": "Initial mechanism remains uncertain." if gen_id == 0 else "Counterevidence must remain available.",
                "peer_id": peer_id,
            })
            output = json.loads(response["content"][0]["text"])
            if output.get("status") != "shared":
                raise AssertionError(response)
        text = json.dumps(output)
        event = AgentEvent(
            event_id=request.request_id + "-final", run_id=request.run_id,
            agent_run_id=request.request_id, stage_id=request.stage_id,
            type="final_result", timestamp_ms=0, artifact_refs=[], credential_refs=[],
            payload={"success": True, "iteration_count": 1,
                     "legacy_output": {"text_outputs": [text], "tool_uses": []}},
        )
        return AgentRunResult(success=True, events=[event], text_output_refs=[], tool_uses=[],
                              error=None, failover_reason=None, credential_ref=None)


def create_runtime():
    return EvidenceRuntime()
"""


_TASK_SOURCE = """
import json


async def handle(context):
    if not (context.run_dir / "orchestrator.lock").is_file():
        raise AssertionError("delivery ran outside controller lifetime")
    if context.phase == "initial":
        path = context.run_dir / "initial.json"
        path.write_text(json.dumps({"status": "ready"}))
    elif context.phase == "review":
        path = context.run_dir / f"review_gen_{context.generation_id}.json"
        if path.exists():
            raise AssertionError("committed generation review was repeated")
        result = await context.run_agent(json.dumps({
            "finding_ids": sorted(f["id"] for f in context.findings),
            "finding_types": sorted(f["finding_type"] for f in context.findings),
            "fail_once": False,
        }), role="review", allowed_tools=[])
        if not result.success:
            raise RuntimeError(result.error)
        path.write_text(result.events[-1].payload["legacy_output"]["text_outputs"][0])
    else:
        findings = list(context.findings)
        if {f["finding_type"] for f in findings} != {"hypothesis", "challenge"}:
            raise AssertionError("terminal input lost scoreless findings")
        last_generation = context.config.get("complete_after_generation", 1)
        for generation in range(last_generation + 1):
            if not (context.run_dir / f"gen_{generation}" / "scoreless_evidence.json").is_file():
                raise AssertionError("terminal delivery preceded a generation boundary")
        result = await context.run_agent(json.dumps({
            "finding_ids": sorted(f["id"] for f in findings),
            "finding_types": sorted(f["finding_type"] for f in findings),
            "fail_once": context.config.get("fail_once", False),
        }), role="final", allowed_tools=[])
        if not result.success:
            raise RuntimeError(result.error)
        output = json.loads(result.events[-1].payload["legacy_output"]["text_outputs"][0])
        path = context.run_dir / "delivery.json"
        path.write_text(json.dumps(output))
    summary = {"phase": context.phase}
    if context.phase == "review" and "complete_after_generation" in context.config:
        summary["research_complete"] = context.generation_id >= context.config["complete_after_generation"]
    return {"status": "completed", "artifacts": [path.name], "summary": summary}
"""


class ScorelessLifecycleIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.task = self.root / "external_task"
        shutil.copytree(
            REPO_ROOT / "templates" / "tasks" / "toy_math",
            self.task,
            ignore=shutil.ignore_patterns("__pycache__", "experiments", ".pytest_cache"),
        )
        self.plugins = self.root / "fixture_plugins"
        shutil.copytree(REPO_ROOT / "tests" / "fixtures" / "plugins", self.plugins)
        (self.plugins / "agent_runtimes" / "fake_runtime" / "adapter.py").write_text(
            textwrap.dedent(_RUNTIME_SOURCE), encoding="utf-8"
        )
        (self.task / "task.py").write_text(textwrap.dedent(_TASK_SOURCE), encoding="utf-8")
        # Consent is stored outside XDG directories. Suppress this unrelated
        # observer in both real subprocesses without reading or changing it.
        bootstrap = self.root / "offline_bootstrap"
        bootstrap.mkdir()
        (bootstrap / "sitecustomize.py").write_text(
            "import os\n"
            "from pathlib import Path\n"
            "from praxist.infrastructure.product_usage import ProductUsageObserver\n"
            "from praxist.cli import product_usage\n"
            "from praxist.plugins.workflow_stages.research_loop.backend import runtime_environment\n"
            "ProductUsageObserver.create = classmethod(lambda cls, **kwargs: None)\n"
            "product_usage.prompt_for_consent_if_unset = lambda: None\n"
            "def isolated_runtime_path(value):\n"
            "    if str(value).startswith('/tmp/praxist_active_governor_uid'):\n"
            "        return Path(os.environ['GPU_GOVERNOR_POINTER_FILE'])\n"
            "    return Path(value)\n"
            "runtime_environment.Path = isolated_runtime_path\n",
            encoding="utf-8",
        )
        (self.task / "prompt_task.jinja2").write_text(
            "Retain unresolved hypotheses and counterevidence without assigning quality scores.\n",
            encoding="utf-8",
        )
        self.run_dir = self.root / "runs" / "scoreless"
        self.env = {
            key: os.environ[key]
            for key in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR", "TMPDIR")
            if key in os.environ
        }
        self.env.update(
            {
                "PYTHONPATH": os.pathsep.join((str(bootstrap), str(REPO_ROOT))),
                "PRAXIST_BUNDLED_PLUGIN_ROOTS": str(self.plugins),
                "PRAXIST_CONTROLLER_STATE_DIR": str(self.root / "private_controller"),
                "PRAXIST_STATE_DIR": str(self.root / "registry"),
                "XDG_CONFIG_HOME": str(self.root / "config"),
                "XDG_STATE_HOME": str(self.root / "state"),
                "XDG_DATA_HOME": str(self.root / "data"),
                "GPU_GOVERNOR_POINTER_FILE": str(self.root / "governor_pointer"),
            }
        )
        self.child_pids: list[int] = []
        self.addCleanup(self._stop_children)

    def _write_task(self, *, fail_once: bool = False, uncapped: bool = False) -> None:
        descriptor: dict[str, Any] = {
            "schema_version": 1,
            "task_version": "0.1.0",
            "task_id": "scoreless_fixture",
            "task_name": "Scoreless evidence fixture",
            "description_file": "description.md",
            "research_direction": "Retain unresolved claims and counterevidence.",
            "research_loop": {
                "mode": "scoreless",
                "lifecycle": {
                    "entrypoint": "task.py:handle",
                    "initial_seconds": 30,
                    "finalization_seconds": 30,
                    "after_generation": True,
                    "config": {"fail_once": fail_once},
                },
            },
            "run_lifecycle": {"max_wall_clock_hours": 1},
            "generation_policy": {
                "max_generations": 2,
                "cohort_size": 1,
                "per_generation_hours": 0.0005,
            },
            "synthesis_trigger": {
                "enabled": False,
                "min_findings": 1,
                "min_interval_minutes": 0.001,
                "max_interval_minutes": 0.01,
                "min_contributing_peers": 1,
            },
            "pi_agent": {"enabled": False},
            "dig_lite": {"enabled": False},
            "quality_diversity": {"enabled": False},
            "gems": {"enabled": False},
            "runtime_outputs": {"root": "experiments"},
            "runtime_environment": {"cwd": "task_project"},
            "praxist_plugins": {
                "workflow": {"stage": "workflow_stage:research_loop"},
                "panel": {"topology": "panel_topology:fake_two_round", "roles": ["role:fake_peer"]},
                "tools": ["tool_server:evaluation_tools"],
                "audit_rules": [],
                "evaluations": [],
                "graph_maintainers": [],
            },
        }
        if uncapped:
            descriptor["research_loop"]["lifecycle"].update(
                initial_seconds=None,
                finalization_seconds=None,
                config={"fail_once": fail_once, "complete_after_generation": 8},
            )
            descriptor["run_lifecycle"]["max_wall_clock_hours"] = None
            # Omitting max_generations must not reinstate the metric mode's
            # default eight-generation cap. Other ceilings explicitly use null.
            descriptor["generation_policy"] = {"cohort_size": 1, "per_generation_hours": None}
            descriptor["synthesis_trigger"].update(
                enabled=True,
                min_interval_minutes=0,
                max_interval_minutes=None,
                poll_interval_seconds=0.01,
                adaptive={"enabled": False},
            )
            descriptor["execution_policy"] = {"tool_execution_timeout_seconds": None}
        (self.task / "task.yaml").write_text(yaml.safe_dump(descriptor), encoding="utf-8")

    def _launch(self, *, resume: bool = False, exit_timeout: float = 40) -> dict:
        args = (
            ["resume", str(self.run_dir)]
            if resume
            else ["start", "--task-path", str(self.task), "--run-dir", str(self.run_dir)]
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "praxist",
                *args,
                "--runtime",
                "agent_runtime:fake_runtime",
                "--model-provider",
                "model_provider:fake_provider",
                "--model",
                "fake-deterministic",
                "--startup-timeout",
                "0",
                "--json",
            ],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        entry = json.loads(completed.stdout)
        self.child_pids.append(entry["pid"])
        deadline = time.monotonic() + exit_timeout
        while time.monotonic() < deadline:
            if not pid_is_alive(entry["pid"]):
                break
            time.sleep(0.05)
        else:
            self.fail("offline lifecycle did not exit:\n" + self._log())
        summary_path = self.run_dir / "run_summary.json"
        self.assertTrue(summary_path.is_file(), self._log())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def _log(self) -> str:
        path = self.run_dir / "logs" / "launcher.nohup.log"
        return path.read_text(encoding="utf-8")[-16000:] if path.exists() else "no launcher log"

    def _stop_children(self) -> None:
        for pid in self.child_pids:
            if pid_is_alive(pid):
                with suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGTERM)
                deadline = time.monotonic() + 5
                while pid_is_alive(pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                if pid_is_alive(pid):
                    with suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)

    def _calls(self) -> list[dict]:
        return [
            json.loads(line)
            for line in (self.run_dir / "fixture_runtime_calls.jsonl").read_text().splitlines()
        ]

    def _evidence(self, *, generations: int = 2) -> list[dict]:
        findings = []
        for generation in range(generations):
            manifest = json.loads(
                (self.run_dir / f"gen_{generation}" / "scoreless_evidence.json").read_text()
            )
            self.assertEqual(manifest["evidence_status"], "not_scored")
            self.assertTrue(manifest["findings"], self._log())
            findings.extend(manifest["findings"])
        return findings

    def _assert_committed_artifact_index(self, *, generations: int = 2) -> Path:
        by_path = {
            row["logical_path"]: row
            for line in (self.run_dir / "artifact_index.jsonl").read_text().splitlines()
            if (row := json.loads(line)).get("logical_path")
        }
        expected = {
            f"gen_{generation}/scoreless_evidence.json": "scoreless_evidence_manifest"
            for generation in range(generations)
        }
        expected["delivery.json"] = "task_delivery"
        for relative, artifact_type in expected.items():
            self.assertIn(relative, by_path)
            artifact = by_path[relative]
            self.assertEqual(artifact["artifact_type"], artifact_type)
            self.assertEqual(artifact["artifact_status"], "committed")
            self.assertTrue(artifact["runtime_fact_source"])
            archived_path = self.run_dir / artifact["payload_path"]
            archived_bytes = archived_path.read_bytes()
            wrapper = json.loads(archived_bytes)
            source_bytes = (self.run_dir / relative).read_bytes()
            self.assertEqual(wrapper["source_path"], relative)
            self.assertEqual(wrapper["source_sha256"], hashlib.sha256(source_bytes).hexdigest())
            self.assertEqual(wrapper["source_encoding"], "json")
            self.assertEqual(wrapper["payload"], json.loads(source_bytes))
            self.assertEqual(
                artifact["content_hash"], "sha256:" + hashlib.sha256(archived_bytes).hexdigest()
            )
        return self.run_dir / by_path["delivery.json"]["payload_path"]

    def _replay(self, *, success: bool = True) -> dict:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "praxist.run",
                "replay",
                str(self.run_dir),
                "--mode",
                "verify",
            ],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode, 0 if success else 1, completed.stderr + completed.stdout
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["success"], success, report["errors"])
        return report

    def test_public_start_retains_two_cohorts_before_terminal_delivery(self) -> None:
        self._write_task()
        summary = self._launch()
        self.assertEqual(summary["status"], "succeeded", self._log())
        self.assertEqual(summary["generations_completed"], 2)
        self.assertEqual(summary["evaluation_status"], "not_configured")
        self.assertEqual(summary["task_delivery"]["status"], "completed")
        findings = self._evidence()
        self.assertEqual({row["finding_type"] for row in findings}, {"hypothesis", "challenge"})
        for row in findings:
            # Legacy ingestion carries identity fields inside metrics; none
            # of these constitutes an objective, evaluation, or quality score.
            self.assertLessEqual(set(row.get("metrics", {})), {"generation_id", "id", "peer_id"})
        self.assertEqual((self.run_dir / "findings" / "frontier.jsonl").read_text(), "")
        delivery = json.loads((self.run_dir / "delivery.json").read_text())
        self.assertEqual(delivery["finding_ids"], sorted(row["id"] for row in findings))
        second_prompt = (self.run_dir / "gen_1" / "gen1_peer0_prompt.md").read_text()
        self.assertTrue(
            "Initial mechanism remains uncertain." in second_prompt,
            "second-cohort prompt omitted the retained first-cohort hypothesis",
        )
        reviewed = json.loads((self.run_dir / "review_gen_0.json").read_text())
        seen = json.loads((self.run_dir / "review_seen_by_gen_1.json").read_text())
        self.assertEqual(seen, reviewed)
        self.assertEqual(
            reviewed["finding_ids"],
            sorted(row["id"] for row in findings if int(row["generation_id"]) == 0),
        )
        identity = hashlib.sha256(str(self.run_dir).encode()).hexdigest()[:24]
        private_state = json.loads(
            (self.root / "private_controller" / identity / "state.json").read_text()
        )
        for generation in (0, 1):
            review = private_state["phases"][f"review_gen_{generation}"]
            self.assertEqual(review["status"], "committed")
            self.assertEqual(review["attempts"], 1)
        self.assertEqual(self._calls()[-1]["role"], "final")
        self.assertFalse((self.run_dir / "orchestrator.lock").exists())
        effective = yaml.safe_load((self.run_dir / "effective_task_spec.yaml").read_text())
        self.assertNotIn("evaluation", effective)
        self.assertNotIn("baselines", effective)
        archived_delivery = self._assert_committed_artifact_index()
        self._replay()
        original_delivery = archived_delivery.read_bytes()
        try:
            archived_delivery.write_bytes(original_delivery + b" ")
            corrupted = self._replay(success=False)
            self.assertTrue(
                any("artifact hash mismatch" in error for error in corrupted["errors"]),
                corrupted["errors"],
            )
        finally:
            archived_delivery.write_bytes(original_delivery)

    def test_public_resume_retries_pending_delivery_without_research_or_new_deadline(self) -> None:
        self._write_task(fail_once=True)
        first = self._launch()
        self.assertEqual(first["status"], "failed", self._log())
        self.assertEqual(first["exit_code"], 1)
        self.assertEqual(first["task_delivery"]["status"], "incomplete")
        self.assertFalse((self.run_dir / "delivery.json").exists())
        terminal_events = [
            json.loads(line)
            for line in (self.run_dir / "trajectory.jsonl").read_text().splitlines()
            if json.loads(line)["kind"] in {"workflow.stage_failed", "workflow.stage_succeeded"}
        ]
        self.assertEqual([event["kind"] for event in terminal_events], ["workflow.stage_failed"])
        self.assertEqual(terminal_events[0]["payload"]["implementation_backend"], "GenerationLoop")
        calls_before = self._calls()
        evidence_before = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.run_dir.glob("gen_*/scoreless_evidence.json")
        }
        public_state_path = self.run_dir / "lifecycle" / "state.json"
        original = json.loads(public_state_path.read_text())
        tampered = dict(original, deadline_at=original["deadline_at"] + 86400)
        public_state_path.write_text(json.dumps(tampered))
        resumed = self._launch(resume=True)
        self.assertEqual(resumed["status"], "succeeded", self._log())
        self.assertEqual(resumed["generations_completed"], 2)
        calls_after = self._calls()
        self.assertEqual(calls_after[: len(calls_before)], calls_before)
        self.assertEqual([call["role"] for call in calls_after[len(calls_before) :]], ["final"])
        self.assertEqual(
            evidence_before,
            {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.run_dir.glob("gen_*/scoreless_evidence.json")
            },
        )
        state = json.loads(public_state_path.read_text())
        self.assertEqual(state["started_at"], original["started_at"])
        self.assertEqual(state["deadline_at"], original["deadline_at"])
        self.assertEqual(
            state["phases"]["finalize"]["phase_deadline_at"],
            original["phases"]["finalize"]["phase_deadline_at"],
        )
        self.assertEqual(state["phases"]["initial"]["attempts"], 1)
        self.assertEqual(state["phases"]["finalize"]["attempts"], 2)
        self.assertEqual(state["phases"]["finalize"]["status"], "committed")
        self._assert_committed_artifact_index()

    def test_public_uncapped_run_passes_eight_generations_and_finalizes_on_task_review(
        self,
    ) -> None:
        self._write_task(uncapped=True)
        summary = self._launch(exit_timeout=120)
        self.assertEqual(summary["status"], "succeeded", self._log())
        self.assertEqual(summary["generations_completed"], 9)
        self.assertIsNone(summary["max_generations"])
        self.assertEqual(summary["exit_condition"], "research_complete")
        self.assertEqual(summary["task_delivery"]["status"], "completed")
        findings = self._evidence(generations=9)
        self.assertEqual({int(finding["generation_id"]) for finding in findings}, set(range(9)))
        delivery = json.loads((self.run_dir / "delivery.json").read_text())
        self.assertEqual(delivery["finding_ids"], sorted(finding["id"] for finding in findings))
        self.assertFalse((self.run_dir / "gen_9").exists(), "research continued after completion")
        self.assertTrue((self.run_dir / "review_seen_by_gen_8.json").is_file())

        def strict_json(text: str) -> dict:
            def reject_constant(value: str) -> None:
                self.fail(f"non-finite JSON constant was serialized: {value}")

            return json.loads(text, parse_constant=reject_constant)

        startup = strict_json((self.run_dir / "startup_config.json").read_text())
        self.assertIsNone(startup["canonical_args"]["generations"])
        self.assertEqual(startup["canonical_args"]["budget_policy"], "budget_policy:default_basic")
        self.assertTrue(startup["budget_authorization"]["uncapped"])
        effective = yaml.safe_load((self.run_dir / "effective_task_spec.yaml").read_text())
        self.assertIsNone(effective["generation_policy"].get("max_generations"))
        self.assertIsNone(effective["generation_policy"]["per_generation_hours"])
        self.assertIsNone(effective["run_lifecycle"]["max_wall_clock_hours"])
        self.assertIsNone(effective["synthesis_trigger"]["max_interval_minutes"])
        self.assertIsNone(effective["research_loop"]["lifecycle"]["initial_seconds"])
        self.assertIsNone(effective["research_loop"]["lifecycle"]["finalization_seconds"])

        state = strict_json((self.run_dir / "lifecycle/state.json").read_text())
        self.assertIsNone(state["deadline_at"])
        self.assertEqual(len(state["phases"]), 11)
        for phase in state["phases"].values():
            self.assertIsNone(phase["phase_deadline_at"])
            self.assertEqual(phase["status"], "committed")
            self.assertEqual(phase["attempts"], 1)
        for generation in range(9):
            review = state["phases"][f"review_gen_{generation}"]
            self.assertEqual(review["result"]["summary"]["research_complete"], generation == 8)

        ledger = [
            strict_json(line)
            for line in (self.run_dir / "budget_ledger.jsonl").read_text().splitlines()
        ]
        decisions = [row for row in ledger if row.get("kind") == "decision"]
        self.assertTrue(decisions)
        expected_budget = {"tokens": None, "wall_clock_seconds": None, "gpu_hours": None}
        self.assertEqual(decisions[0]["requested_budget"], expected_budget)
        self.assertEqual(decisions[0]["granted_budget"], expected_budget)
        self.assertEqual(decisions[0]["decision"], "grant")
        self.assertEqual(
            decisions[0]["request_record"]["expected_value"]["usage_estimate_status"], "unknown"
        )
        usage = [row for row in ledger if row.get("kind") == "usage"]
        self.assertTrue(usage)
        self.assertTrue(any(row["actual_usage"].get("wall_clock_seconds", 0) > 0 for row in usage))
        self.assertTrue(any(row.get("kind") == "usage_unknown" for row in ledger))
        for row in usage:
            self.assertNotIn(
                "tokens", row["actual_usage"], "unknown tokens were fabricated as zero"
            )

        calls = self._calls()
        self.assertEqual(calls[-1]["role"], "final")
        self.assertEqual(sum(call["role"] == "review" for call in calls), 9)
        self.assertEqual(sum(call["role"] == "final" for call in calls), 1)
        self.assertEqual(
            {int(call["generation_id"]) for call in calls if call["role"] == "research"},
            set(range(9)),
        )
        for call in calls:
            self.assertIsNone(call["timeout_seconds"])
            self.assertIsNone(call["tool_execution_timeout_seconds"])
        self._assert_committed_artifact_index(generations=9)
        self._replay()


if __name__ == "__main__":
    unittest.main()
