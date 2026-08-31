"""AgentRuntime registry-backed loader helpers."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from praxist.core import credentials
from praxist.core.execution_policy import (
    apply_task_execution_policy,
    task_execution_remaining_seconds,
)
from praxist.core.protocol import AgentRunRequest, AgentRunResult
from praxist.core.registry import (
    PluginLoader,
    PluginRef,
    PluginRegistry,
    PluginRoots,
    require_execution_plugin,
)

REASONING_EFFORT_POLICIES = frozenset({"auto", "off", "low", "high", "max", "xhigh"})


def effective_reasoning_effort(runtime_options: Mapping[str, Any] | None) -> str:
    """Resolve the task-level reasoning policy without assuming a provider API.

    ``premium_mode`` remains a compatibility alias for ``max`` when the new
    policy is absent or ``auto``. Runtime adapters own the provider-specific
    wire mapping.
    """

    options = runtime_options or {}
    raw = options.get("reasoning_effort", "max")
    policy = str(raw).strip().lower() if isinstance(raw, str) else "max"
    if policy not in REASONING_EFFORT_POLICIES:
        policy = "max"
    if policy == "auto" and bool(options.get("premium_mode")):
        return "max"
    return policy


@dataclass(frozen=True)
class AgentRuntimeExecutionContext:
    """Process-local handles shared by asynchronous AgentRuntime plugins.

    ``AgentRunRequest`` remains the complete serializable execution contract.
    This context carries only values that cannot safely or usefully be put in
    protocol JSON: instantiated tool servers, callbacks, stop polling, and the
    already-scoped environment prepared by the workflow stage.
    """

    tool_servers: Mapping[str, Any] = field(default_factory=dict)
    message_callback: Callable[[Any], None] | None = None
    stop_requested: Callable[[], bool] | None = None
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass
class RuntimeUsageCollector:
    """Aggregate runtime usage for the current workflow-stage context."""

    _usage: dict[str, float] = field(default_factory=dict)

    def add(self, usage: Mapping[str, float]) -> None:
        observed = {
            str(key): float(value)
            for key, value in usage.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if "total_tokens" not in observed:
            token_parts = ("input_tokens", "output_tokens")
            if any(key in observed for key in token_parts):
                observed["total_tokens"] = sum(observed.get(key, 0.0) for key in token_parts)
        for key, value in observed.items():
            self._usage[key] = self._usage.get(key, 0.0) + value

    def snapshot(self) -> dict[str, float]:
        return dict(self._usage)


_runtime_usage_collector: ContextVar[RuntimeUsageCollector | None] = ContextVar(
    "praxist_runtime_usage_collector",
    default=None,
)


@contextmanager
def collect_runtime_usage() -> Iterator[RuntimeUsageCollector]:
    """Collect usage from runtime executions in this asynchronous context."""

    collector = RuntimeUsageCollector()
    token = _runtime_usage_collector.set(collector)
    try:
        yield collector
    finally:
        _runtime_usage_collector.reset(token)


def runtime_for_ref(runtime_ref: str, registry: PluginRegistry | None = None) -> Any:
    """Resolve and instantiate an AgentRuntime implementation for a plugin reference."""
    if registry is None:
        loader = PluginLoader(PluginRoots.defaults(Path.cwd()))
        manifest = loader.resolve(
            [runtime_ref],
            run_id="runtime_conformance",
            root_task_ref=runtime_ref,
            enforce_bundled_execution=True,
        )
        registry = loader.load(manifest)
    require_execution_plugin(
        registry,
        runtime_ref,
        kind="agent_runtime",
    )
    parsed = PluginRef.parse(runtime_ref)
    runtime = registry.require(parsed.kind, parsed.name)
    if not callable(getattr(runtime, "execute", None)) and not callable(
        getattr(runtime, "execute_sync", None)
    ):
        raise TypeError(
            f"{runtime_ref} entrypoint did not return an AgentRuntime with execute() "
            "or execute_sync()"
        )
    observed_ref = getattr(runtime, "runtime_ref", runtime_ref)
    if observed_ref != runtime_ref:
        raise ValueError(f"{runtime_ref} entrypoint returned runtime_ref={observed_ref!r}")
    return runtime


def runtime_managed_credential_for_ref(
    runtime_ref: str,
    model_provider_ref: str,
    registry: PluginRegistry,
) -> credentials.CredentialRef | None:
    """Ask a runtime for optional non-environment authentication.

    This is an optional plugin extension. Core validates the returned redacted
    reference but remains independent of any runtime's authentication backend.
    """

    runtime = runtime_for_ref(runtime_ref, registry=registry)
    discover = getattr(runtime, "discover_managed_credential", None)
    if not callable(discover):
        return None
    credential = discover(model_provider_ref)
    if credential is None:
        return None
    if not isinstance(credential, credentials.CredentialRef):
        raise TypeError(
            f"{runtime_ref} discover_managed_credential() must return CredentialRef | None"
        )
    provider = credentials.provider_name_from_ref(model_provider_ref)
    if (
        credential.scope != "model_provider"
        or credential.provider != provider
        or credential.target_ref not in (None, model_provider_ref)
    ):
        raise ValueError(
            f"{runtime_ref} returned a managed credential outside {model_provider_ref}"
        )
    return credential


def resolve_model_credential_for_runtime(
    credential_set: credentials.CredentialSet,
    runtime_ref: str,
    model_provider_ref: str,
    registry: PluginRegistry,
    *,
    resolve_only: bool,
) -> tuple[credentials.CredentialSet, credentials.CredentialRef | None]:
    """Resolve env credentials before an optional runtime-managed fallback."""

    credential = credentials.find_model_provider_credential(credential_set, model_provider_ref)
    if credential is not None or resolve_only:
        return credential_set, credential
    managed = runtime_managed_credential_for_ref(runtime_ref, model_provider_ref, registry)
    if managed is not None:
        credential_set = credentials.CredentialSet(
            mode=credential_set.mode,
            credentials=[*credential_set.credentials, managed],
        )
    return credential_set, credentials.require_model_provider_credential(
        credential_set,
        model_provider_ref,
    )


async def execute_runtime(
    runtime: Any,
    request: AgentRunRequest,
    context: AgentRuntimeExecutionContext,
) -> AgentRunResult:
    """Execute one runtime through the common async contract.

    Production runtimes implement ``execute``. Deterministic fixture plugins
    may keep the smaller synchronous ``execute_sync`` contract used by
    conformance tests when no task deadline is active. A deadline requires
    an async ``execute`` implementation that cooperates with cancellation.
    """

    request = apply_task_execution_policy(request)
    remaining = task_execution_remaining_seconds()
    if remaining is None:
        return await _execute_runtime_unbounded(runtime, request, context)
    # Cancelling to_thread abandons its await without stopping the worker's
    # tools or file writes. Reject that contract before any runtime side effect.
    if not inspect.iscoroutinefunction(getattr(runtime, "execute", None)):
        raise TypeError("Task execution deadlines require an async AgentRuntime.execute() contract")
    async with asyncio.timeout(remaining):
        return await _execute_runtime_unbounded(runtime, request, context)


async def _execute_runtime_unbounded(
    runtime: Any,
    request: AgentRunRequest,
    context: AgentRuntimeExecutionContext,
) -> AgentRunResult:
    execute = getattr(runtime, "execute", None)
    if callable(execute):
        result = execute(request, context)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, AgentRunResult):
            _collect_result_usage(result)
            return result
        raise TypeError("AgentRuntime.execute() did not return AgentRunResult")

    execute_sync = getattr(runtime, "execute_sync", None)
    if not callable(execute_sync):
        raise TypeError("AgentRuntime implements neither execute() nor execute_sync()")
    result = await asyncio.to_thread(execute_sync, request)
    if not isinstance(result, AgentRunResult):
        raise TypeError("AgentRuntime.execute_sync() did not return AgentRunResult")
    _collect_result_usage(result)
    return result


def _collect_result_usage(result: AgentRunResult) -> None:
    collector = _runtime_usage_collector.get()
    if collector is not None:
        collector.add(result.usage)


def prompt_text_for_request(request: AgentRunRequest) -> str:
    """Return the normalized inline user prompt from an agent request."""

    if not isinstance(request.prompt_ref, dict):
        return ""
    text = request.prompt_ref.get("text")
    return text if isinstance(text, str) else ""


def system_prompt_text_for_request(request: AgentRunRequest) -> str | None:
    """Return the process-local system prompt carried by runtime options."""

    value = request.runtime_options.get("system_prompt") if request.runtime_options else None
    return value if isinstance(value, str) and value else None


def classify_runtime_failure(error: str | None, *, timed_out: bool = False) -> str:
    """Classify a redacted runtime error into the shared failover vocabulary."""

    if timed_out:
        return "timeout"
    if not error:
        return "none"
    lowered = error.lower()
    status_auth = re.search(r"(?<![0-9a-z])(?:401|403)(?![0-9a-z])", lowered)
    named_auth = re.search(
        r"(?<![0-9a-z])(?:authentication|authorization|unauthorized|oauth|auth)"
        r"(?=$|[^0-9a-z]|error|failed|failure|expired|invalid|token)",
        lowered,
    )
    if (
        status_auth
        or named_auth
        or any(token in lowered for token in ("api key", "invalid x-api-key", "user not found"))
    ):
        return "auth_error"
    if any(
        token in lowered
        for token in (
            "quota",
            "billing",
            "credit balance",
            "insufficient credits",
            "payment required",
        )
    ):
        return "quota_exhausted"
    if ("rate" in lowered and "limit" in lowered) or re.search(
        r"(?<![0-9a-z])429(?![0-9a-z])", lowered
    ):
        return "rate_limited"
    if any(token in lowered for token in ("timeout", "timed out", "deadline")):
        return "timeout"
    if "tool" in lowered and any(
        token in lowered for token in ("unavailable", "not installed", "not found")
    ):
        return "tool_unavailable"
    return "runtime_error"


def is_provider_access_error(error: str) -> bool:
    """Return whether a runtime failure reflects credentials or account quota."""

    return classify_runtime_failure(error) in {"auth_error", "quota_exhausted"}


async def close_runtime_for_ref(
    runtime_ref: str,
    registry: PluginRegistry | None,
) -> None:
    """Close a runtime's process-local resources when its workflow stage ends."""

    if registry is None:
        return
    runtime = runtime_for_ref(runtime_ref, registry=registry)
    close = getattr(runtime, "aclose", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def event_types_for_conformance(
    runtime_refs: list[str], request: AgentRunRequest
) -> dict[str, list[str]]:
    """Return the normalized runtime event types expected by conformance tests."""
    events: dict[str, list[str]] = {}
    for runtime_ref in runtime_refs:
        runtime = runtime_for_ref(runtime_ref)
        execute_sync = getattr(runtime, "execute_sync", None)
        if not callable(execute_sync):
            raise TypeError(f"{runtime_ref} does not provide offline conformance execution")
        result = execute_sync(request)
        if not isinstance(result, AgentRunResult):
            raise TypeError(f"{runtime_ref} execute_sync() did not return AgentRunResult")
        events[runtime_ref] = [event.type for event in result.events]
    return events
