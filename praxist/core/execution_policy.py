"""Validated task execution restrictions applied at the runtime boundary."""

from __future__ import annotations

import math
import posixpath
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

from praxist.core.protocol import (
    AgentRunRequest,
    ApprovalIntent,
    FilesystemIntent,
    JSONValue,
    NetworkIntent,
    RuntimeSandboxIntent,
    ToolPermissionSet,
)

_POLICY_KEYS = frozenset(
    {"sandbox_intent", "allowed_tools", "model_by_role", "tool_execution_timeout_seconds"}
)
_EFFORTS = frozenset({"auto", "off", "low", "high", "max", "xhigh"})
_FILESYSTEM_ORDER = ("read_only", "workspace_write", "full")
_APPROVAL_ORDER = ("auto", "on_risk", "always_ask")
_NETWORK_TOOLS = frozenset({"WebSearch", "WebFetch", "web_search"})
_PATH_KEYS = frozenset({"readable_roots", "writable_roots", "denied_paths"})
_execution_policy: ContextVar[dict[str, Any] | None] = ContextVar(
    "praxist_task_execution_policy", default=None
)
_execution_deadline: ContextVar[float | None] = ContextVar(
    "praxist_task_execution_deadline", default=None
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping with string keys")
    return dict(value)


def _within(path: str, root: str) -> bool:
    return path == root or root == "/" or path.startswith(root + "/")


def _minimal_roots(paths: list[str]) -> list[str]:
    selected = sorted(set(paths))
    return [
        path
        for path in selected
        if not any(path != root and _within(path, root) for root in selected)
    ]


def _path_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of absolute paths")
    paths = []
    for path in value:
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or path.startswith("//")
            or any(char in path for char in "*?[]\0\n\r")
            or ".." in path.split("/")
        ):
            raise ValueError(f"{label} must contain absolute paths without patterns or traversal")
        paths.append(posixpath.normpath(path))
    return _minimal_roots(paths)


def _sandbox(value: Any) -> dict[str, Any]:
    raw = _mapping(value, "execution_policy.sandbox_intent")
    if set(raw) - {"filesystem", "network", "approval"} - _PATH_KEYS:
        raise ValueError("execution_policy.sandbox_intent has unknown fields")
    network = raw.get("network", "on")
    # YAML 1.1 readers parse the documented unquoted on/off spelling as bool.
    if isinstance(network, bool):
        network = "on" if network else "off"
    result: dict[str, Any] = RuntimeSandboxIntent(
        filesystem=cast(FilesystemIntent, raw.get("filesystem", "full")),
        network=cast(NetworkIntent, network),
        approval=cast(ApprovalIntent, raw.get("approval", "auto")),
    ).to_dict()
    if _PATH_KEYS.intersection(raw):
        for key in sorted(_PATH_KEYS):
            result[key] = _path_list(raw.get(key, []), f"sandbox_intent.{key}")
    return result


def _intersect_roots(left: list[str], right: list[str]) -> list[str]:
    return _minimal_roots(
        [a if _within(a, b) else b for a in left for b in right if _within(a, b) or _within(b, a)]
    )


def _path_scope(
    task: dict[str, Any], call: dict[str, Any], filesystem: str
) -> dict[str, JSONValue]:
    scopes = [scope for scope in (task, call) if _PATH_KEYS.intersection(scope)]
    if not scopes:
        return {}
    reads = _minimal_roots(scopes[0]["readable_roots"] + scopes[0]["writable_roots"])
    writes = list(scopes[0]["writable_roots"])
    for scope in scopes[1:]:
        reads = _intersect_roots(reads, scope["readable_roots"] + scope["writable_roots"])
        writes = _intersect_roots(writes, scope["writable_roots"])
    denied = _minimal_roots([path for scope in scopes for path in scope["denied_paths"]])
    # Native profiles choose the most-specific grant, so remove grants nested
    # below explicit denials rather than letting a child grant reopen a secret.
    reads = [path for path in reads if not any(_within(path, root) for root in denied)]
    writes = [path for path in writes if not any(_within(path, root) for root in denied)]
    if filesystem == "read_only":
        writes = []
    return {
        "readable_roots": cast(JSONValue, reads),
        "writable_roots": cast(JSONValue, writes),
        "denied_paths": cast(JSONValue, denied),
    }


def _positive_seconds(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or int(value) != value
    ):
        raise ValueError(f"{label} must be a positive whole number of seconds")
    return int(value)


def validate_task_execution_policy(raw: Any) -> dict[str, Any]:
    """Validate and detach an optional task-wide policy from mutable YAML input.

    Args:
        raw: The task's top-level ``execution_policy`` mapping, or ``None``.

    Returns:
        A normalized mapping. Empty mappings leave legacy execution unchanged.

    Raises:
        ValueError: A field is unknown, malformed, or cannot be enforced exactly.
    """
    if raw is None:
        return {}
    policy = _mapping(raw, "execution_policy")
    if set(policy) - _POLICY_KEYS:
        raise ValueError("execution_policy has unknown fields")
    result: dict[str, Any] = {}
    if "sandbox_intent" in policy:
        result["sandbox_intent"] = _sandbox(policy["sandbox_intent"])
    if "allowed_tools" in policy:
        tools = policy["allowed_tools"]
        if not isinstance(tools, list) or any(
            not isinstance(tool, str)
            or not tool.strip()
            or tool != tool.strip()
            or any(char in tool for char in "*?[]")
            for tool in tools
        ):
            raise ValueError("execution_policy.allowed_tools must list exact nonempty tool names")
        result["allowed_tools"] = list(dict.fromkeys(tools))
    if "model_by_role" in policy:
        roles = _mapping(policy["model_by_role"], "execution_policy.model_by_role")
        models: dict[str, dict[str, str]] = {}
        for role, value in roles.items():
            if not role.strip() or role != role.strip():
                raise ValueError("execution_policy model roles must be nonempty names")
            config = _mapping(value, f"execution_policy.model_by_role.{role}")
            if set(config) - {"model", "reasoning_effort"}:
                raise ValueError(f"execution_policy.model_by_role.{role} has unknown fields")
            model = config.get("model")
            if not isinstance(model, str) or not model.strip() or model != model.strip():
                raise ValueError(f"execution_policy.model_by_role.{role}.model must be nonempty")
            effort = config.get("reasoning_effort", "max")
            if not isinstance(effort, str) or effort not in _EFFORTS:
                raise ValueError(
                    f"execution_policy.model_by_role.{role} has invalid reasoning_effort"
                )
            models[role] = {"model": model, "reasoning_effort": effort}
        result["model_by_role"] = models
    if "tool_execution_timeout_seconds" in policy:
        result["tool_execution_timeout_seconds"] = (
            _positive_seconds(
                policy["tool_execution_timeout_seconds"],
                "execution_policy.tool_execution_timeout_seconds",
            )
            if policy["tool_execution_timeout_seconds"] is not None
            else None
        )
    return result


@contextmanager
def task_execution_policy_scope(policy: Mapping[str, Any] | None) -> Iterator[None]:
    """Bind an authoritative task policy to this context and its async children."""
    validated = validate_task_execution_policy(policy)
    token = _execution_policy.set(deepcopy(validated) if validated else None)
    try:
        yield
    finally:
        _execution_policy.reset(token)


@contextmanager
def task_execution_deadline_scope(deadline_at: float | None) -> Iterator[None]:
    """Bind a Unix deadline, preserving any stricter enclosing run deadline."""
    if deadline_at is not None and (
        isinstance(deadline_at, bool)
        or not isinstance(deadline_at, (int, float))
        or not math.isfinite(deadline_at)
    ):
        raise ValueError("execution deadline must be a finite Unix timestamp")
    current = _execution_deadline.get()
    effective = current if deadline_at is None else float(deadline_at)
    if current is not None and effective is not None:
        effective = min(current, effective)
    token = _execution_deadline.set(effective)
    try:
        yield
    finally:
        _execution_deadline.reset(token)


def task_execution_remaining_seconds() -> float | None:
    """Return the exact dispatch budget, rejecting an already-expired deadline."""
    deadline = _execution_deadline.get()
    if deadline is None:
        return None
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Task execution deadline expired before runtime dispatch")
    return remaining


def apply_task_execution_policy(
    request: AgentRunRequest, *, role: str | None = None
) -> AgentRunRequest:
    """Intersect a request with authoritative task restrictions without mutation.

    Explicit execution roles are separate from task skill identity. The role-ref
    fallback serves direct callers; normal workflow agents supply their role.
    Runtime adapters remain responsible for enforcing the resulting sandbox, and
    must reject ``require_task_sandbox_policy`` when they cannot honor it.
    """
    policy = _execution_policy.get()
    remaining = task_execution_remaining_seconds()
    if not policy and remaining is None:
        return request
    policy = policy or {}
    options = dict(request.runtime_options)
    permissions = request.tool_permissions
    model_call = request.model_call
    models = policy.get("model_by_role", {})
    selected_role = role or options.get("execution_role")
    if selected_role is None:
        role_ref = (request.role_ref or "research").rsplit(":", 1)[-1]
        selected_role = role_ref if role_ref in models else "research"
    if not isinstance(selected_role, str) or not selected_role:
        raise ValueError("execution_role must be a nonempty role name")
    if models and selected_role not in models:
        raise ValueError(f"Task execution policy has no model for role {selected_role!r}")
    options["execution_role"] = selected_role
    model = models.get(selected_role)
    if model is not None:
        model_call = replace(model_call, model=model["model"])
        options["reasoning_effort"] = model["reasoning_effort"]

    if "allowed_tools" in policy:
        options["require_task_tool_policy"] = True
        task_allowed = policy["allowed_tools"]
        if permissions.mode == "allow_list":
            allowed = [name for name in permissions.allowed_tools if name in task_allowed]
        elif permissions.mode in {"allow_all", "deny_list"}:
            allowed = list(task_allowed)
        else:
            raise ValueError(f"Unsupported tool permission mode {permissions.mode!r}")
        permissions = ToolPermissionSet(
            mode="allow_list",
            allowed_tools=[name for name in allowed if name not in permissions.denied_tools],
            denied_tools=list(permissions.denied_tools),
        )
    if "sandbox_intent" in policy:
        task_sandbox = policy["sandbox_intent"]
        call_sandbox = _sandbox(options.get("sandbox_intent", {}))
        filesystem = min(
            task_sandbox["filesystem"],
            call_sandbox["filesystem"],
            key=_FILESYSTEM_ORDER.index,
        )
        if options.get("require_read_only_runtime") or options.get("require_no_shell_runtime"):
            filesystem = "read_only"
        network = "off" if "off" in (task_sandbox["network"], call_sandbox["network"]) else "on"
        options["sandbox_intent"] = {
            "filesystem": filesystem,
            "network": network,
            "approval": max(
                task_sandbox["approval"], call_sandbox["approval"], key=_APPROVAL_ORDER.index
            ),
            **_path_scope(task_sandbox, call_sandbox, filesystem),
        }
        options["require_task_sandbox_policy"] = True
        if network == "off":
            permissions = replace(
                permissions,
                allowed_tools=[
                    name for name in permissions.allowed_tools if name not in _NETWORK_TOOLS
                ],
                denied_tools=list(
                    dict.fromkeys([*permissions.denied_tools, *sorted(_NETWORK_TOOLS)])
                ),
            )

    timeout = request.timeout_seconds
    if remaining is not None:
        deadline_timeout = max(1, math.ceil(remaining))
        timeout = (
            min(timeout, deadline_timeout)
            if timeout is not None and timeout > 0
            else deadline_timeout
        )
    tool_timeout = policy.get("tool_execution_timeout_seconds")
    existing_tool_timeout = options.get("tool_execution_timeout_seconds")
    if existing_tool_timeout is not None:
        existing = _positive_seconds(existing_tool_timeout, "tool_execution_timeout_seconds")
        tool_timeout = min(existing, tool_timeout) if tool_timeout is not None else existing
    if tool_timeout is not None:
        if timeout is not None and timeout > 0:
            tool_timeout = min(tool_timeout, timeout)
        options["tool_execution_timeout_seconds"] = cast(JSONValue, tool_timeout)
    return replace(
        request,
        model_call=model_call,
        tool_permissions=permissions,
        runtime_options=options,
        timeout_seconds=timeout,
    )
