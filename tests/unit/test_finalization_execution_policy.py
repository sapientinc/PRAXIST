"""Final lifecycle calls retain task restrictions at real runtime dispatch."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from praxist.core.execution_policy import task_execution_policy_scope
from praxist.core.protocol import AgentRunRequest, AgentRunResult
from praxist.core.runtimes import AgentRuntimeExecutionContext
from praxist.plugins.agent_runtimes.codex_sdk._sandbox import sandbox_settings
from praxist.plugins.workflow_stages.research_loop.backend import agent as agent_module
from praxist.plugins.workflow_stages.research_loop.backend.generation_loop import GenerationLoop

_LIVE_TOOL = "mcp__live-evidence__search"


class FinalizationExecutionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.workspace = root / "task"
        self.run_dir = root / "run"
        self.workspace.mkdir()
        self.run_dir.mkdir()
        self.policy = {
            "sandbox_intent": {
                "filesystem": "workspace_write",
                "network": "on",
                "approval": "auto",
                "readable_roots": [str(self.workspace)],
                "writable_roots": [str(self.run_dir / "artifacts")],
                "denied_paths": [str(self.workspace / "sealed")],
            },
            "allowed_tools": ["Read", "Write", "Bash", "WebSearch", "WebFetch", _LIVE_TOOL],
            "model_by_role": {
                "final": {"model": "final-model", "reasoning_effort": "xhigh"},
                "review": {"model": "review-model", "reasoning_effort": "high"},
            },
        }
        self.loop = cast(
            GenerationLoop,
            SimpleNamespace(
                _lifecycle_phase="finalize",
                _peer_allowed_tools=["Read", "Write", "Bash", "WebSearch", _LIVE_TOOL],
                run_dir=self.run_dir,
                workspace=self.workspace,
                runtime_ref="agent_runtime:fake_runtime",
                mcp_servers={"live-evidence": object()},
                model="unbound-model",
                plugin_registry=None,
                task_spec=SimpleNamespace(agent=SimpleNamespace(reasoning_effort="low")),
            ),
        )
        self.requests: list[AgentRunRequest] = []
        self.contexts: list[AgentRuntimeExecutionContext] = []

        async def execute(
            request: AgentRunRequest, context: AgentRuntimeExecutionContext
        ) -> AgentRunResult:
            self.requests.append(request)
            self.contexts.append(context)
            return AgentRunResult(
                success=True,
                events=[],
                text_output_refs=[],
                tool_uses=[],
                error=None,
                failover_reason=None,
                credential_ref=None,
                terminal_status="succeeded",
            )

        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(task_execution_policy_scope(self.policy))
        self.enterContext(
            patch.object(
                agent_module, "runtime_for_ref", return_value=SimpleNamespace(execute=execute)
            )
        )

    def dispatch(self, role: str, allowed_tools: list[str]) -> AgentRunRequest:
        before = len(self.requests)
        result = asyncio.run(
            GenerationLoop._run_lifecycle_agent(
                self.loop,
                "inspect committed evidence",
                role=role,
                allowed_tools=allowed_tools,
                timeout_seconds=12.5,
            )
        )
        self.assertTrue(result.success)
        self.assertEqual(len(self.requests), before + 1)
        return self.requests[-1]

    def test_final_and_review_calls_keep_paths_and_exclude_live_tools(self) -> None:
        for role, effort in (("final", "xhigh"), ("review", "high")):
            with self.subTest(role=role):
                requested_tools = ["Read"] if role == "review" else ["Read", "Write", "Bash"]
                request = self.dispatch(role, requested_tools)
                self.assertEqual(request.tool_permissions.allowed_tools, requested_tools)
                self.assertNotIn(_LIVE_TOOL, request.tool_permissions.allowed_tools)
                if role == "review":
                    self.assertNotIn("Bash", request.tool_permissions.allowed_tools)
                self.assertEqual(self.contexts[-1].tool_servers, {})
                self.assertEqual(request.tool_servers, [])
                self.assertEqual(request.runtime_options["execution_role"], role)
                self.assertEqual(request.model_call.model, f"{role}-model")
                self.assertEqual(request.runtime_options["reasoning_effort"], effort)
                self.assertEqual(request.timeout_seconds, 13)
                self.assertEqual(
                    request.runtime_options["sandbox_intent"],
                    {
                        "filesystem": "read_only" if role == "review" else "workspace_write",
                        "network": "off",
                        "approval": "auto",
                        "readable_roots": sorted(
                            [str(self.workspace), str(self.run_dir / "artifacts")]
                        ),
                        "writable_roots": []
                        if role == "review"
                        else [str(self.run_dir / "artifacts")],
                        "denied_paths": [str(self.workspace / "sealed")],
                    },
                )
                self.assertTrue(request.runtime_options["require_task_sandbox_policy"])
                settings = sandbox_settings(request)
                self.assertEqual(
                    settings.sandbox, "read_only" if role == "review" else "workspace_write"
                )
                self.assertEqual(settings.permission_profile, "praxist_task")

    def test_explicit_empty_callback_tool_list_does_not_restore_peer_defaults(self) -> None:
        request = self.dispatch("final", [])
        self.assertEqual(request.tool_permissions.allowed_tools, [])
        sandbox = request.runtime_options["sandbox_intent"]
        assert isinstance(sandbox, dict)
        self.assertEqual(sandbox["network"], "off")

    def test_finalization_network_override_does_not_mutate_task_policy(self) -> None:
        original = deepcopy(self.policy)
        final = self.dispatch("final", ["Read"])
        self.loop._lifecycle_phase = "review"
        review = self.dispatch("review", ["Read", _LIVE_TOOL])
        for request, network in ((final, "off"), (review, "on")):
            sandbox = request.runtime_options["sandbox_intent"]
            assert isinstance(sandbox, dict)
            self.assertEqual(sandbox["network"], network)
        self.assertEqual(review.tool_permissions.allowed_tools, ["Read", _LIVE_TOOL])
        self.assertIn("live-evidence", self.contexts[-1].tool_servers)
        self.assertEqual(self.policy, original)
