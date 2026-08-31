"""Translate Praxist sandbox intent into official Codex SDK settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from praxist.core.protocol import AgentRunRequest, RuntimeSandboxIntent

_SHELL_TOOL_NAMES = frozenset({"Bash", "exec_command", "shell", "write_stdin"})
_READ_TOOL_NAMES = frozenset({"Read", "Glob", "Grep"})
_PATCH_TOOL_NAMES = frozenset({"Write", "Edit", "NotebookEdit", "apply_patch"})
_WEB_TOOL_NAMES = frozenset({"WebSearch", "WebFetch", "web_search"})
_IMAGE_TOOL_NAMES = frozenset({"view_image"})
_MULTI_AGENT_TOOL_NAMES = frozenset(
    {"Task", "spawn_agent", "send_input", "wait_agent", "close_agent"}
)
_PATH_PERMISSION_KEYS = frozenset({"readable_roots", "writable_roots", "denied_paths"})
_TASK_PERMISSION_PROFILE = "praxist_task"


@dataclass(frozen=True)
class CodexSandboxSettings:
    """SDK enum names and app-server overrides for one request."""

    approval_mode: str
    sandbox: str
    config: dict[str, object] = field(default_factory=dict)
    permission_profile: str | None = None


def sandbox_settings(request: AgentRunRequest) -> CodexSandboxSettings:
    """Translate a sandbox request into legacy settings or a named profile.

    Args:
        request: Runtime request containing the optional sandbox intent.

    Returns:
        SDK settings. When ``permission_profile`` is set, callers must omit
        the legacy ``sandbox`` argument because it overrides path restrictions.

    Raises:
        ValueError: An intent is malformed or cannot be enforced headlessly.
    """

    intent = _sandbox_intent(request)
    raw = request.runtime_options.get("sandbox_intent")
    path_mode = isinstance(raw, dict) and bool(_PATH_PERMISSION_KEYS.intersection(raw))
    if path_mode and request.runtime_options.get("require_read_only_runtime"):
        intent = replace(intent, filesystem="read_only")
    if intent.approval != "auto":
        raise ValueError(
            "codex_sdk is headless and cannot honor interactive sandbox approval "
            f"mode {intent.approval!r}"
        )
    if not path_mode and intent.filesystem == "full" and intent.network == "off":
        raise ValueError(
            "codex_sdk cannot disable network access in a full-access filesystem sandbox"
        )
    approval_mode = "deny_all"
    sandbox = {
        "read_only": "read_only",
        "workspace_write": "workspace_write",
        "full": "full_access",
    }[intent.filesystem]
    config = _builtin_tool_config(request, intent)
    if path_mode:
        assert isinstance(raw, dict)
        config.update(
            {
                "default_permissions": _TASK_PERMISSION_PROFILE,
                "permissions": {_TASK_PERMISSION_PROFILE: _path_permission_profile(raw, intent)},
            }
        )
        return CodexSandboxSettings(
            approval_mode=approval_mode,
            sandbox=sandbox,
            config=config,
            permission_profile=_TASK_PERMISSION_PROFILE,
        )
    if intent.filesystem == "workspace_write":
        config["sandbox_workspace_write"] = {"network_access": intent.network == "on"}
    return CodexSandboxSettings(
        approval_mode=approval_mode,
        sandbox=sandbox,
        config=config,
    )


def _path_permission_profile(
    raw: Mapping[str, object], intent: RuntimeSandboxIntent
) -> dict[str, object]:
    paths: dict[str, list[Path]] = {}
    for key in _PATH_PERMISSION_KEYS:
        values = raw.get(key, [])
        if not isinstance(values, list) or any(
            not isinstance(value, str)
            or not value.startswith("/")
            or value.startswith("//")
            or any(char in value for char in "*?[]\0\n\r")
            or ".." in value.split("/")
            for value in values
        ):
            raise ValueError(
                f"sandbox_intent.{key} must contain absolute paths without patterns or traversal"
            )
        paths[key] = [Path(value).resolve() for value in values]

    # Resolving aliases can introduce nested denials after core validation.
    # One ancestor denial covers its descendants; emitting both can prevent
    # Linux sandbox setup from mounting a child inside the denied ancestor.
    denied_roots: list[Path] = []
    for path in sorted(
        set(paths["denied_paths"]), key=lambda value: (len(value.parts), str(value))
    ):
        if not any(path.is_relative_to(parent) for parent in denied_roots):
            denied_roots.append(path)
    paths["denied_paths"] = denied_roots

    filesystem = {":root": "deny", ":minimal": "read"}
    for key, access in (
        ("readable_roots", "read"),
        ("writable_roots", "read" if intent.filesystem == "read_only" else "write"),
    ):
        for path in paths[key]:
            # Codex permits a more-specific grant to reopen a denied ancestor.
            # Task denials instead dominate every explicit descendant grant.
            if any(path.is_relative_to(denied) for denied in paths["denied_paths"]):
                continue
            filesystem[str(path)] = access
    for path in paths["denied_paths"]:
        filesystem[str(path)] = "deny"
    return {"filesystem": filesystem, "network": {"enabled": intent.network == "on"}}


def _builtin_tool_config(
    request: AgentRunRequest,
    intent: RuntimeSandboxIntent,
) -> dict[str, object]:
    permissions = request.tool_permissions
    if permissions.mode == "allow_all":
        allowed: set[str] | None = None
    elif permissions.mode == "allow_list":
        allowed = set(permissions.allowed_tools)
    elif permissions.mode == "deny_list":
        allowed = None
    else:
        raise ValueError(f"unsupported ToolPermissionSet mode {permissions.mode!r}")
    denied = set(permissions.denied_tools)

    def enabled(names: frozenset[str]) -> bool:
        selected = allowed is None or bool(allowed.intersection(names))
        return selected and not bool(denied.intersection(names))

    shell = enabled(_SHELL_TOOL_NAMES)
    read_only_shell = enabled(_READ_TOOL_NAMES) and intent.filesystem == "read_only"
    if (
        allowed is not None
        and allowed.intersection(_READ_TOOL_NAMES)
        and not (shell or read_only_shell)
    ):
        raise ValueError(
            "codex_sdk cannot expose Read/Glob/Grep without either Bash permission "
            "or a read-only filesystem sandbox"
        )
    return {
        "features": {
            "shell_tool": shell or read_only_shell,
            "multi_agent": enabled(_MULTI_AGENT_TOOL_NAMES),
        },
        "include_apply_patch_tool": enabled(_PATCH_TOOL_NAMES),
        "tools": {"view_image": enabled(_IMAGE_TOOL_NAMES)},
        "web_search": (
            "live" if intent.network == "on" and enabled(_WEB_TOOL_NAMES) else "disabled"
        ),
    }


def _sandbox_intent(request: AgentRunRequest) -> RuntimeSandboxIntent:
    raw = request.runtime_options.get("sandbox_intent") if request.runtime_options else None
    if isinstance(raw, dict):
        return RuntimeSandboxIntent(
            filesystem=str(raw.get("filesystem", "full")),  # type: ignore[arg-type]
            network=str(raw.get("network", "on")),  # type: ignore[arg-type]
            approval=str(raw.get("approval", "auto")),  # type: ignore[arg-type]
        )
    return RuntimeSandboxIntent(filesystem="full", network="on", approval="auto")


__all__ = ["CodexSandboxSettings", "sandbox_settings"]
