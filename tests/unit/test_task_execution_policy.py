"""Task-wide execution restrictions survive every runtime dispatch."""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from praxist.core.execution_policy import (
    apply_task_execution_policy,
    task_execution_deadline_scope,
    task_execution_policy_scope,
    validate_task_execution_policy,
)
from praxist.core.protocol import AgentRunResult, ToolPermissionSet
from praxist.core.run_config import RunConfig
from praxist.core.runtimes import AgentRuntimeExecutionContext, execute_runtime
from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent


def _policy() -> dict:
    return {
        "sandbox_intent": {"filesystem": "workspace_write", "network": "off", "approval": "auto"},
        "allowed_tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "mcp__evidence__search"],
        "model_by_role": {
            "research": {"model": "research-model", "reasoning_effort": "xhigh"},
            "pi": {"model": "direction-model", "reasoning_effort": "xhigh"},
            "review": {"model": "review-model", "reasoning_effort": "high"},
            "final": {"model": "final-model", "reasoning_effort": "xhigh"},
        },
        "tool_execution_timeout_seconds": 28800,
    }


def _agent(workspace: Path, **kwargs) -> BaseAgent:
    return BaseAgent(
        name="arbitrary-agent-name",
        workspace=workspace,
        allowed_tools=[
            "Read",
            "Write",
            "Bash",
            "WebSearch",
            "mcp__evidence__search",
            "mcp__other__send",
        ],
        mcp_servers={},
        model="original-model",
        run_config=RunConfig(agent_runtime_ref="agent_runtime:fake_runtime"),
        **kwargs,
    )


class TaskExecutionPolicyTests(unittest.TestCase):
    def test_path_scope_intersection_cannot_expand_task_access_or_remove_denials(self):
        policy = _policy()
        policy["sandbox_intent"].update(
            {
                "readable_roots": ["/usr", "/workspace/task"],
                "writable_roots": ["/workspace/output"],
                "denied_paths": ["/workspace/output/private"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp, task_execution_policy_scope(policy):
            request = _agent(
                Path(tmp),
                runtime_sandbox_intent={
                    "filesystem": "full",
                    "network": "on",
                    "approval": "auto",
                    "readable_roots": ["/usr/bin", "/workspace"],
                    "writable_roots": ["/workspace/output/results", "/forbidden"],
                    "denied_paths": ["/workspace/output/results/secret"],
                },
            )._build_agent_run_request("work", {})
        sandbox = request.runtime_options["sandbox_intent"]
        assert isinstance(sandbox, dict)
        self.assertEqual(
            sandbox["readable_roots"], ["/usr/bin", "/workspace/output", "/workspace/task"]
        )
        self.assertEqual(sandbox["writable_roots"], ["/workspace/output/results"])
        self.assertEqual(
            sandbox["denied_paths"],
            ["/workspace/output/private", "/workspace/output/results/secret"],
        )

    def test_read_only_call_downgrades_task_writable_roots_to_readable(self):
        policy = _policy()
        policy["sandbox_intent"].update(
            {
                "readable_roots": ["/workspace/task"],
                "writable_roots": ["/workspace/output"],
                "denied_paths": ["/run/control"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp, task_execution_policy_scope(policy):
            request = _agent(Path(tmp), require_read_only_runtime=True)._build_agent_run_request(
                "work", {}
            )
        sandbox = request.runtime_options["sandbox_intent"]
        assert isinstance(sandbox, dict)
        self.assertEqual(sandbox["filesystem"], "read_only")
        self.assertEqual(sandbox["readable_roots"], ["/workspace/output", "/workspace/task"])
        self.assertEqual(sandbox["writable_roots"], [])
        self.assertEqual(sandbox["denied_paths"], ["/run/control"])

    def test_denied_ancestor_cannot_be_reopened_by_explicit_descendant_grant(self):
        policy = _policy()
        policy["sandbox_intent"].update(
            {
                "readable_roots": ["/run/control/config"],
                "writable_roots": ["/run/control/new"],
                "denied_paths": ["/run/control"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp, task_execution_policy_scope(policy):
            sandbox = (
                _agent(Path(tmp))
                ._build_agent_run_request("work", {})
                .runtime_options["sandbox_intent"]
            )
        assert isinstance(sandbox, dict)
        self.assertEqual(sandbox["readable_roots"], [])
        self.assertEqual(sandbox["writable_roots"], [])
        self.assertEqual(sandbox["denied_paths"], ["/run/control"])

    def test_per_call_path_scope_survives_a_coarse_task_policy(self):
        with tempfile.TemporaryDirectory() as tmp, task_execution_policy_scope(_policy()):
            sandbox = (
                _agent(
                    Path(tmp),
                    runtime_sandbox_intent={
                        "readable_roots": ["/workspace/task"],
                        "writable_roots": [],
                    },
                )
                ._build_agent_run_request("work", {})
                .runtime_options["sandbox_intent"]
            )
        assert isinstance(sandbox, dict)
        self.assertEqual(sandbox["readable_roots"], ["/workspace/task"])
        self.assertEqual(sandbox["writable_roots"], [])

    def test_path_lists_remain_structured_without_task_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = (
                _agent(
                    Path(tmp),
                    runtime_sandbox_intent={
                        "readable_roots": ["/workspace/task"],
                        "denied_paths": ["/run/control"],
                    },
                )
                ._build_agent_run_request("work", {})
                .runtime_options["sandbox_intent"]
            )
        assert isinstance(sandbox, dict)
        self.assertEqual(sandbox["readable_roots"], ["/workspace/task"])
        self.assertEqual(sandbox["denied_paths"], ["/run/control"])

    def test_path_scope_rejects_relative_traversal_and_pattern_paths(self):
        for value in ("/workspace", ["relative"], ["/workspace/../control"], ["/workspace/*"], [1]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_task_execution_policy({"sandbox_intent": {"readable_roots": value}})

    def test_policy_constrains_tools_sandbox_model_and_timeout_at_request_boundary(self):
        with tempfile.TemporaryDirectory() as tmp, task_execution_policy_scope(_policy()):
            request = _agent(Path(tmp))._build_agent_run_request("research", {})
        self.assertEqual(
            request.tool_permissions.allowed_tools,
            ["Read", "Write", "Bash", "mcp__evidence__search"],
        )
        self.assertEqual(
            request.runtime_options["sandbox_intent"],
            {"filesystem": "workspace_write", "network": "off", "approval": "auto"},
        )
        self.assertEqual(request.model_call.model, "research-model")
        self.assertEqual(request.runtime_options["reasoning_effort"], "xhigh")
        self.assertEqual(request.runtime_options["tool_execution_timeout_seconds"], 28800)

    def test_read_only_planner_cannot_regain_writes_shell_or_network(self):
        with tempfile.TemporaryDirectory() as tmp, task_execution_policy_scope(_policy()):
            agent = _agent(
                Path(tmp),
                execution_role="pi",
                require_no_shell_runtime=True,
                require_read_only_runtime=True,
                runtime_sandbox_intent={
                    "filesystem": "read_only",
                    "network": "on",
                    "approval": "always_ask",
                },
            )
            agent.allowed_tools = ["Read", "Glob", "Grep", "WebSearch"]
            request = agent._build_agent_run_request("plan", {})
        self.assertEqual(request.tool_permissions.allowed_tools, ["Read", "Glob", "Grep"])
        self.assertEqual(request.model_call.model, "direction-model")
        self.assertEqual(
            request.runtime_options["sandbox_intent"],
            {"filesystem": "read_only", "network": "off", "approval": "always_ask"},
        )
        self.assertTrue(request.runtime_options["require_no_shell_runtime"])

    def test_every_explicit_role_and_task_role_ref_selects_its_pinned_model(self):
        with tempfile.TemporaryDirectory() as tmp, task_execution_policy_scope(_policy()):
            for role, model in [
                ("research", "research-model"),
                ("pi", "direction-model"),
                ("review", "review-model"),
                ("final", "final-model"),
            ]:
                with self.subTest(role=role):
                    request = _agent(Path(tmp), execution_role=role)._build_agent_run_request(
                        "work", {}
                    )
                    self.assertEqual(request.model_call.model, model)
                    ref_request = replace(request, role_ref=f"task_role:{role}", runtime_options={})
                    self.assertEqual(
                        apply_task_execution_policy(ref_request).model_call.model, model
                    )

    def test_unknown_explicit_role_cannot_fall_back_to_unpinned_model(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            task_execution_policy_scope(_policy()),
            self.assertRaises(ValueError),
        ):
            _agent(Path(tmp), execution_role="unexpected")._build_agent_run_request("work", {})

    def test_runtime_adapters_map_explicit_xhigh_to_their_supported_maximum(self):
        from praxist.plugins.agent_runtimes.claude_sdk import adapter as claude
        from praxist.plugins.agent_runtimes.codex_sdk import adapter as codex

        with tempfile.TemporaryDirectory() as tmp, task_execution_policy_scope(_policy()):
            request = _agent(Path(tmp))._build_agent_run_request("work", {})
        effort = codex._reasoning_effort(request, SimpleNamespace(xhigh="sdk-xhigh"), {})
        self.assertEqual(effort, "sdk-xhigh")
        options = claude.LegacyClaudeRuntimeOptions(
            name="agent",
            allowed_tools=[],
            workspace=Path("."),
            mcp_servers={},
            model="model",
            permission_mode="default",
            reasoning_effort="xhigh",
        )
        self.assertEqual(claude._claude_reasoning_options(options)["effort"], "max")

    def test_denied_tools_and_empty_allow_list_stay_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = _agent(Path(tmp))._build_agent_run_request("work", {})
        with task_execution_policy_scope(_policy()):
            result = apply_task_execution_policy(
                replace(
                    request,
                    tool_permissions=ToolPermissionSet(mode="deny_list", denied_tools=["Bash"]),
                )
            )
            empty = apply_task_execution_policy(
                replace(request, tool_permissions=ToolPermissionSet(mode="allow_list"))
            )
        self.assertNotIn("Bash", result.tool_permissions.allowed_tools)
        self.assertNotIn("WebSearch", result.tool_permissions.allowed_tools)
        self.assertEqual(empty.tool_permissions.allowed_tools, [])

    def test_context_is_snapshotted_and_resets_without_mutating_requests(self):
        policy = _policy()
        with tempfile.TemporaryDirectory() as tmp:
            agent = _agent(Path(tmp))
            original = agent._build_agent_run_request("work", {})
            with task_execution_policy_scope(policy):
                policy["allowed_tools"].append("WebSearch")
                constrained = apply_task_execution_policy(original)
            unchanged = apply_task_execution_policy(original)
        self.assertNotIn("WebSearch", constrained.tool_permissions.allowed_tools)
        self.assertIn("WebSearch", original.tool_permissions.allowed_tools)
        self.assertIs(unchanged, original)

    def test_absent_policy_preserves_legacy_requests(self):
        self.assertEqual(validate_task_execution_policy(None), {})
        self.assertEqual(validate_task_execution_policy({}), {})
        with tempfile.TemporaryDirectory() as tmp:
            request = _agent(Path(tmp))._build_agent_run_request("work", {})
        self.assertEqual(request.model_call.model, "original-model")
        self.assertNotIn("sandbox_intent", request.runtime_options)
        self.assertIn("WebSearch", request.tool_permissions.allowed_tools)

    def test_policy_validation_rejects_unsafe_or_misspelled_settings(self):
        invalid = [
            [],
            {"unknown": True},
            {"allowed_tools": "Read"},
            {"allowed_tools": [""]},
            {"allowed_tools": ["mcp__*"]},
            {"tool_execution_timeout_seconds": 0},
            {"tool_execution_timeout_seconds": True},
            {"tool_execution_timeout_seconds": float("inf")},
            {"sandbox_intent": {"network": "sometimes"}},
            {
                "sandbox_intent": {
                    "filesystem": "full",
                    "network": "off",
                    "approval": "auto",
                    "typo": "value",
                }
            },
            {"model_by_role": {"research": {"model": "", "reasoning_effort": "xhigh"}}},
            {"model_by_role": {"research": {"model": "m", "reasoning_effort": "typo"}}},
            {"model_by_role": {"research": {"model": "m", "extra": "ignored"}}},
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_task_execution_policy(raw)

    def test_yaml_boolean_network_is_normalized_and_empty_allow_list_supported(self):
        policy = _policy()
        policy["sandbox_intent"]["network"] = False
        policy["allowed_tools"] = []
        result = validate_task_execution_policy(policy)
        self.assertEqual(result["sandbox_intent"]["network"], "off")
        self.assertEqual(result["allowed_tools"], [])

    def test_request_and_tool_timeouts_are_clamped_to_nested_deadline(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("praxist.core.execution_policy.time.time", return_value=100),
            task_execution_policy_scope(_policy()),
            task_execution_deadline_scope(150),
        ):
            with task_execution_deadline_scope(200):
                request = _agent(Path(tmp), runtime_timeout_seconds=1000)._build_agent_run_request(
                    "work", {}
                )
            with task_execution_deadline_scope(120):
                shorter = _agent(Path(tmp), runtime_timeout_seconds=5)._build_agent_run_request(
                    "work", {}
                )
            with task_execution_deadline_scope(99), self.assertRaises(TimeoutError):
                _agent(Path(tmp))._build_agent_run_request("work", {})
        self.assertEqual(request.timeout_seconds, 50)
        self.assertEqual(request.runtime_options["tool_execution_timeout_seconds"], 50)
        self.assertEqual(shorter.timeout_seconds, 5)
        self.assertEqual(shorter.runtime_options["tool_execution_timeout_seconds"], 5)


class TaskExecutionDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_hard_deadline_rejects_synchronous_dispatch_before_any_side_effect(self):
        result = AgentRunResult(
            success=True,
            events=[],
            text_output_refs=[],
            tool_uses=[],
            error=None,
            failover_reason=None,
            credential_ref=None,
        )
        calls = []

        def execute(*args):
            calls.append(args)
            return result

        with tempfile.TemporaryDirectory() as tmp:
            request = _agent(Path(tmp))._build_agent_run_request("work", {})
        for method in ("execute", "execute_sync"):
            with self.subTest(method=method):
                runtime = SimpleNamespace(**{method: execute})
                # Deterministic synchronous fixtures remain supported normally.
                actual = await execute_runtime(runtime, request, AgentRuntimeExecutionContext())
                self.assertIs(actual, result)
                calls.clear()
                with (
                    task_execution_deadline_scope(time.time() + 10),
                    self.assertRaisesRegex(TypeError, "async AgentRuntime.execute"),
                ):
                    await execute_runtime(runtime, request, AgentRuntimeExecutionContext())
                self.assertEqual(calls, [])

    async def test_claude_cannot_treat_task_tool_allowlist_as_autoapproval_only(self):
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        with (
            tempfile.TemporaryDirectory() as tmp,
            task_execution_policy_scope({"allowed_tools": ["Read"]}),
        ):
            request = _agent(Path(tmp))._build_agent_run_request("work", {})
        request = replace(request, agent_runtime_ref="agent_runtime:claude_sdk")
        runtime = adapter.ClaudeSdkAgentRuntime()
        with patch.object(
            runtime,
            "_execute_legacy_isolated",
            new=AsyncMock(side_effect=AssertionError("task allowlist must reject before SDK")),
        ):
            result = await runtime.execute(request, AgentRuntimeExecutionContext())
        self.assertFalse(result.success)
        assert result.error is not None
        self.assertIn("tool", result.error)

    async def test_normalized_callback_returns_the_single_policy_bound_runtime_result(self):
        captured = []
        expected = AgentRunResult(
            success=True,
            events=[],
            text_output_refs=[{"text": "delivered"}],
            tool_uses=[],
            error=None,
            failover_reason=None,
            credential_ref=None,
            usage={"input_tokens": 10},
            terminal_status="completed",
        )

        class Runtime:
            async def execute(self, request, context):
                captured.append(request)
                return expected

        with tempfile.TemporaryDirectory() as tmp, task_execution_policy_scope(_policy()):
            agent = _agent(Path(tmp), execution_role="final")
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.agent.runtime_for_ref",
                return_value=Runtime(),
            ):
                result = await agent.execute_normalized("final callback")
        self.assertIs(result, expected)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].model_call.model, "final-model")

    async def test_unsupported_runtime_rejects_task_sandbox_before_execution(self):
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        with tempfile.TemporaryDirectory() as tmp, task_execution_policy_scope(_policy()):
            request = _agent(Path(tmp))._build_agent_run_request("work", {})
        request = replace(request, agent_runtime_ref="agent_runtime:claude_sdk")
        runtime = adapter.ClaudeSdkAgentRuntime()
        with patch.object(
            runtime,
            "_execute_legacy_isolated",
            new=AsyncMock(side_effect=AssertionError("sandbox must reject before SDK")),
        ):
            result = await runtime.execute(request, AgentRuntimeExecutionContext())
        self.assertFalse(result.success)
        assert result.error is not None
        self.assertIn("sandbox", result.error)

    async def test_async_context_inheritance_restricts_direct_non_base_agent_dispatch(self):
        captured = []

        class Runtime:
            async def execute(self, request, context):
                captured.append(request)
                raise RuntimeError("captured-before-provider")

        with tempfile.TemporaryDirectory() as tmp:
            request = _agent(Path(tmp))._build_agent_run_request("work", {})
        with (
            task_execution_policy_scope(_policy()),
            self.assertRaisesRegex(RuntimeError, "captured-before-provider"),
        ):
            await asyncio.create_task(
                execute_runtime(Runtime(), request, AgentRuntimeExecutionContext())
            )
        self.assertEqual(captured[0].model_call.model, "research-model")
        self.assertNotIn("WebSearch", captured[0].tool_permissions.allowed_tools)
        self.assertEqual(captured[0].runtime_options["sandbox_intent"]["network"], "off")

    async def test_dispatch_cancels_at_deadline_even_when_adapter_ignores_timeout(self):
        cancelled = asyncio.Event()

        class Runtime:
            async def execute(self, request, context):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        with tempfile.TemporaryDirectory() as tmp:
            request = _agent(Path(tmp))._build_agent_run_request("work", {})
        with task_execution_deadline_scope(time.time() + 0.03), self.assertRaises(TimeoutError):
            await execute_runtime(Runtime(), request, AgentRuntimeExecutionContext())
        self.assertTrue(cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
