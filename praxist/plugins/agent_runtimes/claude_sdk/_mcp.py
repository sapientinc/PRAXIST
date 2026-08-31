"""Map explicit MCP execution deadlines to the pinned Claude CLI contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def mcp_execution_options(
    tool_servers: Mapping[str, Any],
    env: Mapping[str, str],
    timeout_seconds: float | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return copied SDK options with an explicit tool execution deadline.

    Args:
        tool_servers: Claude SDK or external MCP server descriptors.
        env: Environment explicitly passed to the Claude subprocess.
        timeout_seconds: Tool execution allowance, or None to omit Praxist's
            override. Existing server/environment limits and the CLI's finite
            native default still apply; None does not mean unlimited MCP work.

    Returns:
        Server descriptors and subprocess environment for ClaudeAgentOptions.

    Raises:
        ValueError: The requested deadline or a transport cannot be represented
            by the pinned CLI without silently clamping or ignoring the policy.
    """
    servers = dict(tool_servers)
    runtime_env = dict(env)
    if timeout_seconds is None:
        return servers, runtime_env
    # Claude CLI 2.1.228 clamps native timers to this millisecond range. Reject
    # unsupported allowances rather than silently shortening a queued call.
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 1 <= timeout_seconds <= 2147483.647
    ):
        raise ValueError(
            "claude_sdk tool_execution_timeout_seconds must be between 1 and 2147483.647"
        )
    timeout_ms = math.ceil(timeout_seconds * 1000)
    for name, descriptor in servers.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError(
                "claude_sdk cannot apply tool_execution_timeout_seconds to an opaque MCP descriptor"
            )
        server = dict(descriptor)
        transport = server.get("type", "stdio")
        if isinstance(transport, str) and transport in {"stdio", "http", "sse"}:
            # Per-server execution timeouts take precedence over the environment
            # and also extend the CLI's external-transport idle allowance.
            server["timeout"] = timeout_ms
        elif transport != "sdk":
            raise ValueError(
                "claude_sdk cannot apply tool_execution_timeout_seconds to an unsupported transport"
            )
        servers[name] = server
    # SDK descriptors do not accept a per-server timeout. MCP_TIMEOUT controls
    # startup instead, so leave it untouched and use the execution-specific env.
    runtime_env["MCP_TOOL_TIMEOUT"] = str(timeout_ms)
    return servers, runtime_env
