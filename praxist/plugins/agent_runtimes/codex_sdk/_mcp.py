"""Expose bundled Praxist tool servers to the Codex app-server over stdio."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

MCP_STDIO_MODULE = "praxist.plugins.tools._mcp_stdio"

MCP_SERVER_FACTORIES: dict[str, str] = {
    "frontier-tools": "praxist.plugins.tools.frontier_tools.adapter:create_frontier_tools_server",
    "evaluation-tools": (
        "praxist.plugins.tools.evaluation_tools.adapter:create_evaluation_tools_server"
    ),
    "memory-tools": "praxist.plugins.tools.memory_tools.adapter:create_memory_tools_server",
    "finding-graph-query": (
        "praxist.plugins.tools.finding_graph_query.adapter:create_finding_graph_query_server"
    ),
    "prior-work-tools": (
        "praxist.plugins.tools.prior_work_tools.adapter:create_prior_work_tools_server"
    ),
    "system-tools": "praxist.plugins.tools.system.adapter:create_system_tools_server",
    "brave-search": "praxist.plugins.tools.brave_search.adapter:create_brave_search_server",
    "browser": "praxist.plugins.tools.browser.adapter:create_browser_server",
    "arxiv": "praxist.plugins.tools.arxiv.adapter:create_arxiv_server",
    "pdf-reader": "praxist.plugins.tools.pdf_reader.adapter:create_pdf_reader_server",
    "literature-lookup": (
        "praxist.plugins.tools.literature_lookup.adapter:create_literature_lookup_server"
    ),
    "run-report": "praxist.plugins.tools.run_report.adapter:create_run_report_server",
}

MCP_SERVER_CREDENTIAL_ENV: dict[str, tuple[str, ...]] = {
    "brave-search": ("BRAVE_API_KEY",),
}

# These selected servers carry the durable research evidence contract. A turn
# without them cannot faithfully execute the role that requested their tools.
# Search, browser, literature, and report servers remain optional so an
# ancillary network failure does not discard otherwise valid work.
REQUIRED_MCP_SERVERS = frozenset(
    {
        "evaluation-tools",
        "finding-graph-query",
        "frontier-tools",
        "memory-tools",
        "prior-work-tools",
    }
)


@dataclass(frozen=True)
class McpConfiguration:
    """App-server configuration and any omitted-server diagnostics."""

    config: dict[str, object]
    warnings: tuple[str, ...]


def mcp_server_key(server_name: str) -> str:
    """Return the stable app-server configuration key for a tool server."""

    return server_name


def mcp_configuration(
    tool_servers: Iterable[dict[str, object]],
    *,
    env: Mapping[str, str] | None = None,
    credential_env: Mapping[str, str] | None = None,
    python_executable: str | None = None,
    allowed_tools: Iterable[str] | None = None,
    denied_tools: Iterable[str] | None = None,
    tool_execution_timeout_seconds: object | None = None,
    isolated_python: bool = False,
) -> McpConfiguration:
    """Translate selected tool servers to thread-scoped app-server config.

    Args:
        tool_servers: Resolved bundled or task-installed server descriptors.
        env: Scoped operational environment for each server.
        credential_env: Credentials selected only for declared server scopes.
        python_executable: Trusted interpreter containing installed tool packages.
        allowed_tools: Exact tool allowlist, or ``None`` for no allowlist.
        denied_tools: Tool denials that take precedence over the allowlist.
        tool_execution_timeout_seconds: Optional native MCP execution timeout.
            ``None`` omits the override, preserving the SDK's native default;
            it does not assert that native MCP calls have no timeout.
        isolated_python: Ignore analyst cwd, Python environment paths, and user
            site packages during Python startup. Tool packages must be installed
            in the trusted interpreter; its site-packages and editable source
            roots must be outside analyst-writable mounts.

    Returns:
        Server configuration and diagnostics for omitted descriptors.
    """

    if tool_execution_timeout_seconds is not None and (
        isinstance(tool_execution_timeout_seconds, bool)
        or not isinstance(tool_execution_timeout_seconds, (int, float))
        or not math.isfinite(tool_execution_timeout_seconds)
        or tool_execution_timeout_seconds <= 0
    ):
        raise ValueError("tool_execution_timeout_seconds must be a positive finite number")
    python_executable = python_executable or sys.executable
    restrict_to_allowed = allowed_tools is not None
    allowed = set(allowed_tools or ())
    denied = set(denied_tools or ())
    servers: dict[str, object] = {}
    warnings: list[str] = []
    seen: set[str] = set()
    for entry in tool_servers:
        if not isinstance(entry, dict):
            continue
        raw_name = entry.get("server_name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        server_name = raw_name.strip()
        if server_name in seen:
            continue
        seen.add(server_name)
        declared_factory = entry.get("factory")
        factory = (
            str(declared_factory).strip()
            if isinstance(declared_factory, str) and declared_factory.strip()
            else MCP_SERVER_FACTORIES.get(server_name)
        )
        if factory is None:
            warnings.append(f"tool server {server_name!r} has no bundled stdio adapter")
            continue
        enabled_tools = _enabled_tools(
            server_name,
            entry.get("tool_names"),
            allowed=allowed,
            denied=denied,
            restrict_to_allowed=restrict_to_allowed,
        )
        if restrict_to_allowed and not enabled_tools:
            continue
        key = mcp_server_key(server_name)
        # MCP subprocesses do not inherit the analyst shell sandbox. Prevent
        # writable cwd/PYTHONPATH modules from running before the trusted server.
        args = [*(["-I"] if isolated_python else []), "-m", MCP_STDIO_MODULE, factory]
        server_config: dict[str, object] = {
            "command": python_executable,
            "args": args,
            # Resolution and ToolPermissionSet filtering happen before this
            # config reaches Codex.  Approve only that already-scoped MCP
            # surface so headless peers do not stall on an app-server prompt.
            "default_tools_approval_mode": "approve",
            "required": server_name in REQUIRED_MCP_SERVERS,
            "startup_timeout_sec": 30,
        }
        if tool_execution_timeout_seconds is not None:
            server_config["tool_timeout_sec"] = tool_execution_timeout_seconds
        if enabled_tools:
            server_config["enabled_tools"] = enabled_tools
        server_env = dict(env or {})
        for key_name in MCP_SERVER_CREDENTIAL_ENV.get(server_name, ()):
            value = str((credential_env or {}).get(key_name) or "")
            if value:
                server_env[key_name] = value
        if server_env:
            server_config["env"] = server_env
        servers[key] = server_config
    return McpConfiguration({"mcp_servers": servers}, tuple(warnings))


def _enabled_tools(
    server_name: str,
    raw_tool_names: object,
    *,
    allowed: set[str],
    denied: set[str],
    restrict_to_allowed: bool,
) -> list[str]:
    declared = (
        [str(value) for value in raw_tool_names or ()]
        if isinstance(raw_tool_names, (list, tuple, set))
        else []
    )
    prefixes = (f"mcp__{server_name}__", f"mcp__{server_name.replace('-', '_')}__")
    if not declared and restrict_to_allowed:
        declared = sorted(
            {
                value[len(prefix) :]
                for value in allowed
                for prefix in prefixes
                if value.startswith(prefix) and value[len(prefix) :]
            }
        )
    if not declared:
        return []

    def selected(tool_name: str) -> bool:
        qualified = {f"{prefix}{tool_name}" for prefix in prefixes}
        if denied.intersection(qualified) or tool_name in denied:
            return False
        return (
            not restrict_to_allowed or bool(allowed.intersection(qualified)) or tool_name in allowed
        )

    return [tool_name for tool_name in declared if selected(tool_name)]


__all__ = [
    "MCP_SERVER_FACTORIES",
    "MCP_SERVER_CREDENTIAL_ENV",
    "REQUIRED_MCP_SERVERS",
    "MCP_STDIO_MODULE",
    "McpConfiguration",
    "mcp_configuration",
    "mcp_server_key",
]
