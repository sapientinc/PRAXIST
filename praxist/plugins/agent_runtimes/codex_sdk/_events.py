"""Normalize official Codex app-server notifications into Praxist events."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

from praxist.core.protocol import (
    AgentEvent,
    AgentRunRequest,
    AgentRunResult,
    JSONValue,
    ToolCallRecord,
)
from praxist.core.redaction import redact_json, redact_text
from praxist.core.runtimes import classify_runtime_failure

_TOOL_ITEM_TYPES = {
    "commandExecution",
    "dynamicToolCall",
    "mcpToolCall",
    "webSearch",
}


@dataclass(frozen=True)
class TerminalState:
    """Normalized terminal status reported by one Codex turn."""

    status: str
    error: str | None


class CodexEventCollector:
    """Accumulate one typed app-server turn without retaining provider objects."""

    def __init__(self, request: AgentRunRequest) -> None:
        self.request = request
        self.events: list[AgentEvent] = []
        self.text_outputs: list[str] = []
        self.tool_uses: list[ToolCallRecord] = []
        self.legacy_tool_uses: list[dict[str, Any]] = []
        self.usage: dict[str, float] = {}
        self.cost: dict[str, float] = {}
        self.terminal: TerminalState | None = None
        self._event_index = 0
        self._tool_started_at: dict[str, int] = {}
        self._started_monotonic = time.monotonic()
        self.emit(
            "agent_run_started",
            {"request": request.to_dict(), "runtime_ref": request.agent_runtime_ref},
        )

    def consume(self, notification: Any) -> list[AgentEvent]:
        """Consume one SDK notification and return newly emitted events."""

        before = len(self.events)
        method = str(getattr(notification, "method", "unknown"))
        payload = getattr(notification, "payload", None)
        payload_data = _model_dict(payload)
        if method == "item/started":
            self._item_started(_item_dict(payload), payload_data)
        elif method == "item/completed":
            self._item_completed(_item_dict(payload), payload_data)
        elif method == "thread/tokenUsage/updated":
            self.usage = _usage_dict(payload_data.get("tokenUsage"))
            self.cost = _cost_dict(payload_data.get("cost"))
            self.emit("usage", {"usage": self.usage, "cost": self.cost})
        elif method == "turn/started":
            self.emit("turn_started", {"turn_id": _turn_id(payload_data)})
        elif method == "turn/completed":
            turn = payload_data.get("turn")
            turn = turn if isinstance(turn, dict) else {}
            error = turn.get("error")
            error_text = _error_text(error)
            self.terminal = TerminalState(
                status=str(turn.get("status") or "unknown"),
                error=error_text,
            )
        elif method == "error":
            error_text = _error_text(payload_data.get("error")) or "Codex app-server error"
            self.emit(
                "runtime_warning" if payload_data.get("willRetry") else "runtime_error",
                {"error": error_text, "will_retry": bool(payload_data.get("willRetry"))},
            )
        return self.events[before:]

    def result(
        self,
        *,
        interrupted_by_stop: bool = False,
        timed_out: bool = False,
        transport_error: str | None = None,
        relay_used: bool = False,
    ) -> AgentRunResult:
        """Build the normalized terminal result from the collected turn."""

        terminal_status = (
            "timeout"
            if timed_out
            else "cancelled"
            if interrupted_by_stop
            else self.terminal.status
            if self.terminal
            else "unknown"
        )
        terminal_error = self.terminal.error if self.terminal else None
        error = transport_error or terminal_error
        if timed_out:
            error = "agent runtime timeout"
        elif not interrupted_by_stop and terminal_status != "completed" and not error:
            error = f"Codex turn ended with status {terminal_status}"
        success = interrupted_by_stop or (
            not timed_out and transport_error is None and terminal_status == "completed"
        )
        failover_reason = classify_runtime_failure(error, timed_out=timed_out)
        duration = max(0.0, time.monotonic() - self._started_monotonic)
        legacy_output = {
            "text_outputs": list(self.text_outputs),
            "tool_uses": list(self.legacy_tool_uses),
            "usage": dict(self.usage),
            "result_message": {
                "status": terminal_status,
                "interrupted_by_stop": interrupted_by_stop,
            },
        }
        self.emit(
            "final_result",
            {
                "success": success,
                "duration": duration,
                "iteration_count": len(self.tool_uses),
                "error": error,
                "failover_reason": failover_reason,
                "relay_used": relay_used,
                "legacy_output": legacy_output,
                "usage": dict(self.usage),
                "terminal_status": terminal_status,
                "timed_out": timed_out,
                "cancelled": interrupted_by_stop,
            },
        )
        return AgentRunResult(
            success=success,
            events=list(self.events),
            text_output_refs=[],
            tool_uses=list(self.tool_uses),
            error=error,
            failover_reason=failover_reason,
            credential_ref=self.request.credential_ref,
            usage=dict(self.usage),
            cost=dict(self.cost),
            terminal_status=terminal_status,
            timed_out=timed_out,
            cancelled=interrupted_by_stop,
        )

    def emit(self, event_type: str, payload: dict[str, Any]) -> AgentEvent:
        self._event_index += 1
        safe_value, _ = redact_json(payload)
        safe_payload: dict[str, JSONValue] = (
            cast(dict[str, JSONValue], safe_value)
            if isinstance(safe_value, dict)
            else {"value": safe_value}
        )
        safe_request_id = "".join(
            char if char.isalnum() or char in "_-" else "_" for char in self.request.request_id
        )
        event = AgentEvent(
            event_id=f"{safe_request_id}_event_{self._event_index:03d}",
            run_id=self.request.run_id,
            agent_run_id=self.request.request_id,
            stage_id=self.request.stage_id,
            type=event_type,
            payload=safe_payload,
            artifact_refs=[],
            credential_refs=[self.request.credential_ref] if self.request.credential_ref else [],
            timestamp_ms=int(time.time() * 1000),
        )
        self.events.append(event)
        return event

    def _item_started(self, item: dict[str, Any], payload: dict[str, Any]) -> None:
        item_type = str(item.get("type") or "unknown")
        item_id = str(item.get("id") or "")
        if item_type not in _TOOL_ITEM_TYPES:
            return
        started_at = _integer(payload.get("startedAtMs"), int(time.time() * 1000))
        self._tool_started_at[item_id] = started_at
        server_name, tool_name, arguments = _tool_identity(item)
        self.emit(
            "tool_use",
            {
                "tool_call_id": item_id,
                "server_name": server_name,
                "tool_name": tool_name,
                "input": arguments,
            },
        )

    def _item_completed(self, item: dict[str, Any], payload: dict[str, Any]) -> None:
        item_type = str(item.get("type") or "unknown")
        if item_type == "agentMessage":
            text = str(item.get("text") or "")
            if text:
                safe_text, _ = redact_text(text)
                self.text_outputs.append(safe_text)
                self.emit("assistant_text", {"text": safe_text})
            return
        if item_type == "fileChange":
            self.emit(
                "file_change",
                {"status": item.get("status"), "changes": item.get("changes") or []},
            )
            return
        if item_type == "reasoning":
            self.emit(
                "reasoning",
                {"summary": item.get("summary") or [], "content": item.get("content") or []},
            )
            return
        if item_type == "plan":
            self.emit("plan", {"text": str(item.get("text") or "")})
            return
        if item_type not in _TOOL_ITEM_TYPES:
            return

        item_id = str(item.get("id") or "")
        server_name, tool_name, arguments = _tool_identity(item)
        success = _tool_success(item)
        finished_at = _integer(payload.get("completedAtMs"), int(time.time() * 1000))
        record = ToolCallRecord(
            tool_call_id=item_id,
            server_name=server_name,
            tool_name=tool_name,
            started_at_ms=self._tool_started_at.pop(item_id, finished_at),
            finished_at_ms=finished_at,
            success=success,
            artifact_refs=[],
            failover_reason="none" if success else "runtime_error",
        )
        self.tool_uses.append(record)
        self.legacy_tool_uses.append({"tool": tool_name, "input": arguments})
        self.emit(
            "tool_result",
            {
                "tool_call_id": item_id,
                "server_name": server_name,
                "tool_name": tool_name,
                "success": success,
                "status": item.get("status"),
                "exit_code": item.get("exitCode"),
                "error": _error_text(item.get("error")),
            },
        )


def _model_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        result = dump(mode="json", by_alias=True, exclude_none=True)
        return result if isinstance(result, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _item_dict(payload: Any) -> dict[str, Any]:
    item = getattr(payload, "item", None)
    if item is not None:
        return _model_dict(item)
    raw = _model_dict(payload).get("item")
    return raw if isinstance(raw, dict) else {}


def _turn_id(payload: dict[str, Any]) -> str:
    turn = payload.get("turn")
    return str(turn.get("id") or "") if isinstance(turn, dict) else ""


def _tool_identity(item: dict[str, Any]) -> tuple[str, str, Any]:
    item_type = str(item.get("type") or "")
    if item_type == "mcpToolCall":
        return (
            str(item.get("server") or "mcp"),
            str(item.get("tool") or "mcp_tool"),
            item.get("arguments") or {},
        )
    if item_type == "dynamicToolCall":
        return (
            str(item.get("namespace") or "dynamic"),
            str(item.get("tool") or "dynamic_tool"),
            item.get("arguments") or {},
        )
    if item_type == "webSearch":
        return "codex_builtin", "web_search", {"query": item.get("query")}
    return "codex_builtin", "shell", {"command": item.get("command"), "cwd": item.get("cwd")}


def _tool_success(item: dict[str, Any]) -> bool:
    status = str(item.get("status") or "").lower()
    if item.get("error") not in (None, "", {}):
        return False
    exit_code = item.get("exitCode")
    if exit_code is not None:
        return _integer(exit_code, 1) == 0
    if item.get("success") is not None:
        return bool(item.get("success"))
    if item.get("type") == "webSearch" and not status:
        # A completed webSearch item has no status field in the current SDK.
        # Reaching item/completed without an explicit error is its success signal.
        return True
    return status in {"completed", "success", "succeeded"}


def _usage_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    total = value.get("total")
    total = total if isinstance(total, dict) else value
    mapping = {
        "input_tokens": "inputTokens",
        "cached_input_tokens": "cachedInputTokens",
        "output_tokens": "outputTokens",
        "reasoning_output_tokens": "reasoningOutputTokens",
        "total_tokens": "totalTokens",
    }
    return {
        key: float(total[source])
        for key, source in mapping.items()
        if isinstance(total.get(source), int | float)
    }


def _cost_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(val)
        for key, val in value.items()
        if isinstance(val, (int, float)) and not isinstance(val, bool)
    }


def _error_text(value: Any) -> str | None:
    if value in (None, "", {}):
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        text = str(value.get("message") or value.get("additionalDetails") or value)
    else:
        text = str(value)
    safe, _ = redact_text(text)
    return safe


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = ["CodexEventCollector", "TerminalState"]
