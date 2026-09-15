"""Typed Codex app-server notification normalization tests."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from praxist.core.protocol import (
    AgentRunRequest,
    CachePolicy,
    EnvPolicy,
    ModelCallSpec,
    ToolPermissionSet,
)
from praxist.plugins.agent_runtimes.codex_sdk._events import CodexEventCollector


@dataclass(frozen=True)
class NotificationPayload:
    """Pydantic-compatible stand-in for an official generated payload model."""

    data: dict[str, Any]

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class Notification:
    """Shape exposed by ``openai_codex.types.Notification``."""

    method: str
    payload: NotificationPayload


def _notification(method: str, payload: dict[str, Any]) -> Notification:
    return Notification(method=method, payload=NotificationPayload(payload))


def _request(*, request_id: str = "request-1") -> AgentRunRequest:
    return AgentRunRequest(
        request_id=request_id,
        run_id="run-1",
        stage_id="stage-1",
        role_ref=None,
        agent_runtime_ref="agent_runtime:codex_sdk",
        prompt_ref={"text": "Inspect the task."},
        system_prompt_ref=None,
        cwd="/tmp",
        model_profile_ref="model_profile:default",
        model_call=ModelCallSpec(
            profile_id="default",
            provider_ref="model_provider:openai",
            api_format="responses",
            model="gpt-5",
            parameters={},
            credential_ref=None,
        ),
        tool_permissions=ToolPermissionSet(),
        tool_servers=[],
        env_policy=EnvPolicy(),
        credential_ref=None,
        credential_mode="env",
        budget_grant_id=None,
        artifact_scope="run",
        timeout_seconds=60,
        cache_policy=CachePolicy(mode="deterministic_no_cache", frozen_prefix_hash=None),
    )


def _completed_turn(*, status: str = "completed", error: Any = None) -> Notification:
    turn: dict[str, Any] = {"id": "turn-1", "status": status, "items": []}
    if error is not None:
        turn["error"] = error
    return _notification("turn/completed", {"threadId": "thread-1", "turn": turn})


class CodexEventCollectorTest(unittest.TestCase):
    def test_assistant_reasoning_plan_and_file_change_notifications(self) -> None:
        collector = CodexEventCollector(_request())
        secret = "sk-or-v1-this-must-not-survive"

        collector.consume(
            _notification(
                "item/completed",
                {"item": {"id": "a1", "type": "agentMessage", "text": f"answer {secret}"}},
            )
        )
        collector.consume(
            _notification(
                "item/completed",
                {
                    "item": {
                        "id": "r1",
                        "type": "reasoning",
                        "summary": ["compare candidates"],
                        "content": ["inspect evidence"],
                    }
                },
            )
        )
        collector.consume(
            _notification(
                "item/completed",
                {"item": {"id": "p1", "type": "plan", "text": "run evaluator"}},
            )
        )
        collector.consume(
            _notification(
                "item/completed",
                {
                    "item": {
                        "id": "f1",
                        "type": "fileChange",
                        "status": "completed",
                        "changes": [{"path": "candidate.py", "kind": "update"}],
                    }
                },
            )
        )

        self.assertEqual(
            [event.type for event in collector.events[1:]],
            ["assistant_text", "reasoning", "plan", "file_change"],
        )
        assistant = collector.events[1].payload["text"]
        self.assertNotIn(secret, str(assistant))
        self.assertEqual(collector.text_outputs, [assistant])

    def test_shell_mcp_web_and_dynamic_tools_emit_started_and_completed_events(self) -> None:
        collector = CodexEventCollector(_request())
        items = [
            {
                "id": "shell-1",
                "type": "commandExecution",
                "command": "python evaluate.py",
                "cwd": "/tmp",
                "status": "completed",
                "exitCode": 0,
            },
            {
                "id": "mcp-1",
                "type": "mcpToolCall",
                "server": "evaluation-tools",
                "tool": "evaluate",
                "arguments": {"candidate": "a"},
                "status": "completed",
            },
            {
                "id": "web-1",
                "type": "webSearch",
                "query": "official benchmark metric",
            },
            {
                "id": "dynamic-1",
                "type": "dynamicToolCall",
                "namespace": "task",
                "tool": "inspect",
                "arguments": {"path": "result.json"},
                "status": "completed",
                "success": True,
            },
        ]

        for index, item in enumerate(items):
            collector.consume(
                _notification(
                    "item/started",
                    {"item": item, "startedAtMs": 100 + index},
                )
            )
            collector.consume(
                _notification(
                    "item/completed",
                    {"item": item, "completedAtMs": 200 + index},
                )
            )

        self.assertEqual(
            [event.type for event in collector.events[1:]],
            ["tool_use", "tool_result"] * 4,
        )
        self.assertEqual(
            [(record.server_name, record.tool_name) for record in collector.tool_uses],
            [
                ("codex_builtin", "shell"),
                ("evaluation-tools", "evaluate"),
                ("codex_builtin", "web_search"),
                ("task", "inspect"),
            ],
        )
        self.assertTrue(collector.tool_uses[0].success)
        self.assertTrue(collector.tool_uses[1].success)
        self.assertTrue(collector.tool_uses[2].success)
        self.assertTrue(collector.tool_uses[3].success)
        self.assertEqual(collector.tool_uses[0].started_at_ms, 100)
        self.assertEqual(collector.tool_uses[0].finished_at_ms, 200)

    def test_failed_tool_call_is_preserved_as_a_failed_record(self) -> None:
        collector = CodexEventCollector(_request())
        item = {
            "id": "shell-failed",
            "type": "commandExecution",
            "command": "false",
            "status": "failed",
            "exitCode": 1,
            "error": {"message": "command failed"},
        }
        collector.consume(_notification("item/started", {"item": item, "startedAtMs": 1}))
        collector.consume(_notification("item/completed", {"item": item, "completedAtMs": 2}))

        self.assertFalse(collector.tool_uses[0].success)
        self.assertEqual(collector.tool_uses[0].failover_reason, "runtime_error")
        self.assertEqual(collector.events[-1].payload["exit_code"], 1)

    def test_usage_and_successful_terminal_state_reach_result(self) -> None:
        collector = CodexEventCollector(_request())
        collector.consume(
            _notification(
                "thread/tokenUsage/updated",
                {
                    "tokenUsage": {
                        "total": {
                            "inputTokens": 11,
                            "cachedInputTokens": 7,
                            "outputTokens": 5,
                            "reasoningOutputTokens": 2,
                            "totalTokens": 18,
                        }
                    },
                    "cost": {"cost_usd": 0.125},
                },
            )
        )
        collector.consume(_notification("turn/started", {"turn": {"id": "turn-1"}}))
        collector.consume(_completed_turn())
        result = collector.result(relay_used=True)

        self.assertTrue(result.success)
        self.assertEqual(result.terminal_status, "completed")
        self.assertEqual(result.usage["cached_input_tokens"], 7.0)
        self.assertEqual(result.usage["total_tokens"], 18.0)
        self.assertEqual(result.cost, {"cost_usd": 0.125})
        self.assertEqual([event.type for event in result.events][-1], "final_result")
        self.assertTrue(result.events[-1].payload["relay_used"])

    def test_failed_terminal_state_is_not_reported_as_success(self) -> None:
        collector = CodexEventCollector(_request())
        collector.consume(
            _completed_turn(
                status="failed",
                error={"message": "provider rejected request"},
            )
        )
        result = collector.result()

        self.assertFalse(result.success)
        self.assertEqual(result.terminal_status, "failed")
        self.assertIn("provider rejected", result.error or "")
        self.assertEqual(result.failover_reason, "runtime_error")

    def test_retryable_error_is_warning_and_terminal_error_is_redacted(self) -> None:
        collector = CodexEventCollector(_request())
        secret = "sk-proj-do-not-record-this"
        collector.consume(
            _notification(
                "error",
                {"error": {"message": f"temporary failure {secret}"}, "willRetry": True},
            )
        )
        collector.consume(_completed_turn(status="failed", error={"message": f"fatal {secret}"}))
        result = collector.result()

        self.assertEqual(result.events[1].type, "runtime_warning")
        self.assertNotIn(secret, str(result.events[1].payload))
        self.assertNotIn(secret, result.error or "")

    def test_timeout_stop_and_transport_failures_have_distinct_terminal_flags(self) -> None:
        timed_out = CodexEventCollector(_request()).result(timed_out=True)
        stopped = CodexEventCollector(_request()).result(interrupted_by_stop=True)
        failed = CodexEventCollector(_request()).result(transport_error="transport disconnected")

        self.assertFalse(timed_out.success)
        self.assertTrue(timed_out.timed_out)
        self.assertEqual(timed_out.terminal_status, "timeout")
        self.assertEqual(timed_out.failover_reason, "timeout")
        self.assertTrue(stopped.success)
        self.assertTrue(stopped.cancelled)
        self.assertEqual(stopped.terminal_status, "cancelled")
        self.assertFalse(failed.success)
        self.assertEqual(failed.failover_reason, "runtime_error")

    def test_stop_and_timeout_override_a_late_completed_notification(self) -> None:
        stopped_collector = CodexEventCollector(_request())
        stopped_collector.consume(_completed_turn())
        timed_out_collector = CodexEventCollector(_request())
        timed_out_collector.consume(_completed_turn())

        self.assertEqual(
            stopped_collector.result(interrupted_by_stop=True).terminal_status,
            "cancelled",
        )
        self.assertEqual(
            timed_out_collector.result(timed_out=True).terminal_status,
            "timeout",
        )

    def test_unknown_notification_is_ignored_without_losing_lifecycle(self) -> None:
        collector = CodexEventCollector(_request(request_id="unsafe/request"))
        emitted = collector.consume(_notification("future/event", {"value": 1}))
        collector.consume(_completed_turn())
        result = collector.result()

        self.assertEqual(emitted, [])
        self.assertEqual(
            [event.type for event in result.events],
            ["agent_run_started", "final_result"],
        )
        self.assertTrue(all("/" not in event.event_id for event in result.events))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
