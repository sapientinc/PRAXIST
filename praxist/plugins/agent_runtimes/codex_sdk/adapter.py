"""Official Codex Python SDK runtime backed by a long-lived app-server."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
import os
import threading
from collections.abc import Awaitable, Callable, Iterator, Mapping
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from praxist.core.credentials import CredentialRef
from praxist.core.protocol import AgentEvent, AgentRunRequest, AgentRunResult
from praxist.core.redaction import redact_text
from praxist.core.runtimes import (
    AgentRuntimeExecutionContext,
    effective_reasoning_effort,
    prompt_text_for_request,
    system_prompt_text_for_request,
)

from ._auth import (
    SUBSCRIPTION_ENV_KEYS,
    StagedChatgptHome,
    discover_chatgpt_credential,
    is_chatgpt_subscription_credential,
    operator_codex_home,
    resolve_codex_binary,
    stage_chatgpt_home,
    verify_chatgpt_login,
)
from ._events import CodexEventCollector
from ._mcp import mcp_configuration
from ._relay import RelayHandle, needs_relay, provider_key_var, provider_name, start_relay
from ._sandbox import sandbox_settings

logger = logging.getLogger(__name__)

_RUNTIME_REF = "agent_runtime:codex_sdk"
_STREAM_POLL_SECONDS = 0.25
_INTERRUPT_DRAIN_SECONDS = 5.0
_INTERRUPT_REQUEST_SECONDS = 2.0
_RESOURCE_CLOSE_SECONDS = 8.0
# The official async SDK delegates each blocking notification read to the
# process-wide default executor. A dedicated pool prevents a large peer cohort
# from starving unrelated Praxist I/O or its own interrupt/close requests.
_SDK_IO_WORKERS = 128
_MAX_ACTIVE_STREAMS = 112
_SUBSCRIPTION_RUNTIME_OVERRIDES = (
    'model_provider="openai"',
    "check_for_update_on_startup=false",
    'instructions=""',
    'developer_instructions=""',
    "features.apps=false",
    "features.hooks=false",
    "features.memories=false",
    "features.multi_agent=false",
    "features.multi_agent_v2=false",
    "features.plugin_sharing=false",
    "features.plugins=false",
    "features.remote_plugin=false",
    "features.shell_snapshot=false",
    "features.skill_mcp_dependency_install=false",
    "features.workspace_dependencies=false",
    "mcp_servers={}",
    "notify=[]",
    "project_doc_max_bytes=0",
    "project_doc_fallback_filenames=[]",
    "skills.bundled=[]",
    "skills.include_instructions=false",
)

_SAFE_PROCESS_ENV_KEYS = frozenset(
    {
        "ALL_PROXY",
        # Codex treats an empty override as an explicit blank runtime name.
        "CODEX_INTERNAL_ORIGINATOR_OVERRIDE",
        "CURL_CA_BUNDLE",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NO_PROXY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
    }
)
_MODEL_CREDENTIAL_ENV_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "DASHSCOPE_API_KEY",
        "DEEPSEEK_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "MOONSHOT_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ORCAROUTER_API_KEY",
        "XAI_API_KEY",
    }
)


@dataclass(frozen=True)
class _ClientKey:
    run_scope: str
    provider: str
    credential_digest: str
    reasoning_policy: str = "default"


@dataclass
class _ClientEntry:
    client: Any
    relay: RelayHandle | None
    provider_id: str
    relay_used: bool
    staged_chatgpt_home: StagedChatgptHome | None = None
    active_turns: int = 0
    unhealthy: bool = False
    closed: bool = False


@dataclass(frozen=True)
class _TurnOutcome:
    interrupted_by_stop: bool = False
    timed_out: bool = False
    client_unhealthy: bool = False


class _RuntimeStopped(Exception):
    pass


class _RuntimeTimedOut(Exception):
    pass


def create_runtime() -> CodexSdkRuntime:
    """Return the process-local Codex SDK runtime plugin instance."""

    return CodexSdkRuntime()


class CodexSdkRuntime:
    """Execute independent Codex threads through shared app-server clients."""

    runtime_ref = _RUNTIME_REF

    def __init__(self) -> None:
        self._clients: dict[_ClientKey, _ClientEntry] = {}
        self._closing = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client_lock: asyncio.Lock | None = None
        self._stream_slots: asyncio.Semaphore | None = None
        self._executor: ThreadPoolExecutor | None = None

    def discover_managed_credential(self, model_provider_ref: str) -> CredentialRef | None:
        """Return a redacted reference for a saved native ChatGPT login."""

        return discover_chatgpt_credential(model_provider_ref)

    async def execute(
        self,
        request: AgentRunRequest,
        context: AgentRuntimeExecutionContext,
    ) -> AgentRunResult:
        """Run one request and stream typed SDK notifications into Praxist."""

        collector = CodexEventCollector(request)
        self._notify(context, collector.events[0])
        deadline = (
            asyncio.get_running_loop().time() + request.timeout_seconds
            if request.timeout_seconds > 0
            else None
        )
        if _stop_requested(context):
            return self._finish(collector, context, interrupted_by_stop=True)
        validation_error = self._validate_request(request)
        if validation_error:
            return self._finish(collector, context, transport_error=validation_error)

        try:
            sandbox = sandbox_settings(request)
        except ValueError as exc:
            return self._finish(collector, context, transport_error=str(exc))
        if bool(request.runtime_options.get("require_read_only_runtime")) and (
            sandbox.sandbox != "read_only"
        ):
            return self._finish(
                collector,
                context,
                transport_error=(
                    "codex_sdk requires sandbox_intent.filesystem='read_only' "
                    "for a read-only runtime request"
                ),
            )

        key = _client_key(request, context.env)
        entry: _ClientEntry | None = None
        turn: Any | None = None
        slot_acquired = False
        rpc_started = False
        unhealthy = False
        try:
            entry = await self._await_controlled(
                self._acquire_client(key, request, context),
                context,
                deadline,
            )
            shell_env = _subprocess_env(request, context.env)
            mcp = mcp_configuration(
                request.tool_servers,
                env=shell_env,
                credential_env=_mcp_credential_env(request, context.env),
                allowed_tools=(
                    request.tool_permissions.allowed_tools
                    if request.tool_permissions.mode == "allow_list"
                    else None
                ),
                denied_tools=request.tool_permissions.denied_tools,
            )
            for warning in mcp.warnings:
                logger.warning("Codex SDK runtime: %s", warning)
            thread_config = _merge_config(
                mcp.config,
                sandbox.config,
                {
                    "shell_environment_policy": {
                        "inherit": "none",
                        "ignore_default_excludes": False,
                        "set": shell_env,
                    }
                },
            )
            sdk = _load_sdk()
            stream_slots = self._require_stream_slots()
            await self._await_controlled(stream_slots.acquire(), context, deadline)
            slot_acquired = True
            rpc_started = True
            thread = await self._await_controlled(
                self._call_resource(
                    entry.client.thread_start,
                    _ignore_late_resource,
                    approval_mode=getattr(sdk["ApprovalMode"], sandbox.approval_mode),
                    base_instructions=system_prompt_text_for_request(request),
                    config=thread_config,
                    cwd=request.cwd,
                    ephemeral=True,
                    model=_runtime_model(request),
                    model_provider=entry.provider_id,
                    sandbox=getattr(sdk["Sandbox"], sandbox.sandbox),
                    service_name="praxist",
                ),
                context,
                deadline,
            )
            output_schema = _output_schema(request)
            turn_prompt = prompt_text_for_request(request)
            turn_effort = _reasoning_effort(request, sdk["ReasoningEffort"], context.env)
            endpoint_schema = output_schema
            if _contains_non_strict_object_schema(output_schema):
                endpoint_schema = None
                logger.warning(
                    "Codex SDK runtime: omitting a known non-strict output schema from "
                    "the endpoint request; downstream prompt and task validation are unchanged"
                )
                warning = collector.emit(
                    "runtime_warning",
                    {
                        "error": "non-strict output schema omitted from endpoint request; "
                        "downstream prompt and task validation are unchanged",
                        "will_retry": False,
                    },
                )
                self._notify(context, warning)
            turn = await self._start_turn(
                thread,
                context,
                deadline,
                prompt=turn_prompt,
                effort=turn_effort,
                output_schema=endpoint_schema,
            )
            outcome = await self._consume_turn(
                turn,
                collector,
                context,
                deadline=deadline,
            )
            unhealthy = outcome.client_unhealthy
            return self._finish(
                collector,
                context,
                interrupted_by_stop=outcome.interrupted_by_stop,
                timed_out=outcome.timed_out,
                relay_used=entry.relay_used,
            )
        except _RuntimeStopped:
            unhealthy = rpc_started
            if turn is not None and not await self._interrupt(turn):
                unhealthy = True
            return self._finish(
                collector,
                context,
                interrupted_by_stop=True,
                relay_used=entry.relay_used if entry is not None else False,
            )
        except _RuntimeTimedOut:
            unhealthy = rpc_started
            if turn is not None and not await self._interrupt(turn):
                unhealthy = True
            return self._finish(
                collector,
                context,
                timed_out=True,
                relay_used=entry.relay_used if entry is not None else False,
            )
        except asyncio.CancelledError:
            unhealthy = rpc_started
            if turn is not None:
                unhealthy = not await self._interrupt(turn) or unhealthy
            raise
        except Exception as exc:  # noqa: BLE001 - normalize SDK/provider errors.
            unhealthy = True
            error, _ = redact_text(str(exc))
            return self._finish(
                collector,
                context,
                transport_error=error,
                relay_used=entry.relay_used if entry is not None else False,
            )
        finally:
            if slot_acquired:
                self._require_stream_slots().release()
            if entry is not None:
                await self._release_client(key, entry, unhealthy=unhealthy)

    def execute_sync(self, request: AgentRunRequest) -> AgentRunResult:
        """Return a deterministic transcript for offline conformance tests."""

        collector = CodexEventCollector(request)
        collector.text_outputs.append("codex_sdk_mocked_transcript")
        collector.emit("assistant_text", {"text": "codex_sdk_mocked_transcript"})
        from ._events import TerminalState

        collector.terminal = TerminalState("completed", None)
        return collector.result()

    async def aclose(self) -> None:
        """Close every app-server and relay owned by this runtime."""

        self._closing = True
        lock = self._client_lock
        if lock is None:
            entries = list(self._clients.values())
            self._clients.clear()
        else:
            async with lock:
                entries = list(self._clients.values())
                self._clients.clear()
        for entry in entries:
            await self._close_entry(entry)
        executor = self._executor
        self._executor = None
        if executor is not None:
            # A pathological SDK call may still be unwinding after its client
            # was closed. Never let runtime shutdown wait indefinitely for an
            # executor worker that Python cannot forcibly interrupt.
            executor.shutdown(wait=False, cancel_futures=True)
        self._loop = None
        self._client_lock = None
        self._stream_slots = None

    async def _start_turn(
        self,
        thread: Any,
        context: AgentRuntimeExecutionContext,
        deadline: float | None,
        *,
        prompt: str,
        effort: Any | None,
        output_schema: dict[str, Any] | None,
    ) -> Any:
        return await self._await_controlled(
            self._call_resource(
                thread.turn,
                _interrupt_late_turn,
                prompt,
                effort=effort,
                output_schema=output_schema,
            ),
            context,
            deadline,
        )

    def _validate_request(self, request: AgentRunRequest) -> str | None:
        if request.agent_runtime_ref != self.runtime_ref:
            return f"runtime mismatch: expected {self.runtime_ref}, got {request.agent_runtime_ref}"
        if bool(request.runtime_options.get("require_no_shell_runtime")):
            return (
                "codex_sdk cannot guarantee a shell-free tool surface; "
                "select a runtime with native read-only tools"
            )
        if (
            _uses_chatgpt_subscription(request)
            and provider_name(request.model_call.provider_ref) != "openai"
        ):
            return "codex_sdk ChatGPT authentication is valid only for the native OpenAI provider"
        return None

    async def _acquire_client(
        self,
        key: _ClientKey,
        request: AgentRunRequest,
        context: AgentRuntimeExecutionContext,
    ) -> _ClientEntry:
        self._bind_loop()
        lock = self._require_client_lock()
        async with lock:
            if self._closing:
                raise RuntimeError("CodexSdkRuntime is shutting down")
            existing = self._clients.get(key)
            if existing is not None and not existing.unhealthy and not existing.closed:
                existing.active_turns += 1
                return existing

            sdk = _load_sdk()
            provider = key.provider
            state_dir = _client_state_dir(request, key)
            relay: RelayHandle | None = None
            staged_chatgpt_home: StagedChatgptHome | None = None
            provider_id = "openai"
            config_overrides: tuple[str, ...] = ()
            subscription = _uses_chatgpt_subscription(request, context.env)
            client: Any | None = None
            try:
                codex_bin = _explicit_codex_binary(request)
                if subscription:
                    codex_bin = resolve_codex_binary(codex_bin)
                    operator_home = operator_codex_home()
                    if not operator_home.is_dir():
                        raise RuntimeError(
                            "Saved Codex ChatGPT authentication home is unavailable; "
                            "run `codex login`"
                        )
                    verify_chatgpt_login(codex_bin)
                    staged_chatgpt_home = stage_chatgpt_home(operator_home)
                    credential = request.credential_ref or request.model_call.credential_ref
                    if (
                        credential is None
                        or credential.key_id != staged_chatgpt_home.credential_key_id
                    ):
                        raise RuntimeError(
                            "Codex ChatGPT authentication changed after startup; restart the run"
                        )
                    codex_home = staged_chatgpt_home.path
                else:
                    codex_home = state_dir / "home"
                    codex_home.mkdir(parents=True, exist_ok=True)
                client_env = _client_process_env(
                    provider,
                    context.env,
                    codex_home,
                    subscription=subscription,
                )
                if subscription:
                    assert staged_chatgpt_home is not None
                    sqlite_home = state_dir / "sqlite"
                    sqlite_home.mkdir(parents=True, exist_ok=True)
                    log_dir = state_dir / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    config_overrides = (
                        f"sqlite_home={json.dumps(str(sqlite_home))}",
                        f"log_dir={json.dumps(str(log_dir))}",
                        "cli_auth_credentials_store="
                        f"{json.dumps(staged_chatgpt_home.credential_store)}",
                        *_SUBSCRIPTION_RUNTIME_OVERRIDES,
                    )
                if needs_relay(provider):
                    key_var = provider_key_var(provider)
                    relay_extra_params, relay_drop_params = _relay_reasoning_options(request)
                    relay = await self._call_resource(
                        start_relay,
                        lambda late: late.close() if late is not None else None,
                        provider=provider,
                        api_key=client_env.get(key_var, ""),
                        state_dir=state_dir,
                        upstream_session_id=(
                            _openrouter_session_id(request) if provider == "openrouter" else None
                        ),
                        upstream_extra_params=relay_extra_params,
                        drop_upstream_params=relay_drop_params,
                    )
                    provider_id = "praxist_relay"
                    config_overrides += (
                        'model_provider="praxist_relay"',
                        f"model_providers.praxist_relay.name={json.dumps('Praxist relay')}",
                        f"model_providers.praxist_relay.base_url={json.dumps(relay.base_url)}",
                        'model_providers.praxist_relay.wire_api="responses"',
                        f"model_providers.praxist_relay.env_key={json.dumps(key_var)}",
                        "model_providers.praxist_relay.requires_openai_auth=false",
                    )
                client_config = sdk["CodexConfig"](
                    codex_bin=codex_bin,
                    config_overrides=config_overrides,
                    cwd=request.cwd,
                    env=client_env,
                    client_name="praxist",
                    client_title="Praxist Codex SDK Runtime",
                )
                client = await self._call_resource(
                    sdk["Codex"],
                    lambda late: _close_late_client(late, relay, staged_chatgpt_home),
                    client_config,
                )
                if subscription:
                    account = await self._call_resource(
                        client.account,
                        _ignore_late_resource,
                        refresh_token=False,
                    )
                    if _codex_account_type(account) != "chatgpt":
                        raise RuntimeError(
                            "Codex app-server did not select ChatGPT authentication; "
                            "API-key fallback is disabled"
                        )
                if self._closing:
                    raise RuntimeError("CodexSdkRuntime is shutting down")
            except BaseException:
                if client is not None:
                    with contextlib.suppress(Exception):
                        await self._bounded_call(client.close, timeout=_RESOURCE_CLOSE_SECONDS)
                if relay is not None:
                    with contextlib.suppress(Exception):
                        await self._bounded_call(relay.close, timeout=_RESOURCE_CLOSE_SECONDS)
                if staged_chatgpt_home is not None:
                    staged_chatgpt_home.close()
                raise
            assert client is not None
            entry = _ClientEntry(
                client=client,
                relay=relay,
                provider_id=provider_id,
                relay_used=relay is not None,
                staged_chatgpt_home=staged_chatgpt_home,
                active_turns=1,
            )
            self._clients[key] = entry
            return entry

    async def _release_client(
        self,
        key: _ClientKey,
        entry: _ClientEntry,
        *,
        unhealthy: bool,
    ) -> None:
        lock = self._require_client_lock()
        close_entry = False
        async with lock:
            entry.active_turns = max(0, entry.active_turns - 1)
            entry.unhealthy = entry.unhealthy or unhealthy
            if entry.unhealthy and self._clients.get(key) is entry:
                self._clients.pop(key, None)
            close_entry = entry.unhealthy and entry.active_turns == 0
        if close_entry:
            await self._close_entry(entry)

    async def _consume_turn(
        self,
        turn: Any,
        collector: CodexEventCollector,
        context: AgentRuntimeExecutionContext,
        *,
        deadline: float | None,
        notify_events: bool = True,
    ) -> _TurnOutcome:
        consumer = asyncio.create_task(
            self._read_turn(turn, collector, context, notify_events=notify_events)
        )
        try:
            while not consumer.done():
                now = asyncio.get_running_loop().time()
                stop = _stop_requested(context)
                timed_out = deadline is not None and now >= deadline
                if stop or timed_out:
                    interrupted = await self._interrupt(turn)
                    drained = await _await_task(consumer, timeout=_INTERRUPT_DRAIN_SECONDS)
                    if not drained:
                        _observe_detached_task(consumer)
                    return _TurnOutcome(
                        interrupted_by_stop=stop and not timed_out,
                        timed_out=timed_out,
                        client_unhealthy=not interrupted or not drained,
                    )

                wait_for = _STREAM_POLL_SECONDS
                if deadline is not None:
                    wait_for = max(0.01, min(wait_for, deadline - now))
                await asyncio.wait({consumer}, timeout=wait_for)

            await consumer
            return _TurnOutcome(client_unhealthy=collector.terminal is None)
        except asyncio.CancelledError:
            await self._interrupt(turn)
            if not await _await_task(consumer, timeout=_INTERRUPT_DRAIN_SECONDS):
                _observe_detached_task(consumer)
            raise

    async def _read_turn(
        self,
        turn: Any,
        collector: CodexEventCollector,
        context: AgentRuntimeExecutionContext,
        *,
        notify_events: bool,
    ) -> None:
        stream: Iterator[Any] = turn.stream()
        try:
            while True:
                has_value, notification = await self._call(_next_item, stream)
                if not has_value:
                    return
                for event in collector.consume(notification):
                    if notify_events:
                        self._notify(context, event)
                if collector.terminal is not None:
                    return
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                await self._bounded_call(close, timeout=_RESOURCE_CLOSE_SECONDS)

    async def _interrupt(self, turn: Any) -> bool:
        try:
            await self._bounded_call(turn.interrupt, timeout=_INTERRUPT_REQUEST_SECONDS)
        except Exception:  # noqa: BLE001 - interruption is best-effort cleanup.
            return False
        return True

    async def _close_entry(self, entry: _ClientEntry) -> None:
        if entry.closed:
            return
        entry.closed = True
        with contextlib.suppress(Exception):
            await self._bounded_call(entry.client.close, timeout=_RESOURCE_CLOSE_SECONDS)
        if entry.relay is not None:
            with contextlib.suppress(Exception):
                await self._bounded_call(entry.relay.close, timeout=_RESOURCE_CLOSE_SECONDS)
        if entry.staged_chatgpt_home is not None:
            entry.staged_chatgpt_home.close()

    async def _await_controlled(
        self,
        awaitable: Awaitable[Any],
        context: AgentRuntimeExecutionContext,
        deadline: float | None,
    ) -> Any:
        """Await one SDK phase while enforcing the request-wide controls."""

        task = asyncio.ensure_future(awaitable)
        try:
            while not task.done():
                if _stop_requested(context):
                    raise _RuntimeStopped
                now = asyncio.get_running_loop().time()
                if deadline is not None and now >= deadline:
                    raise _RuntimeTimedOut
                wait_for = _STREAM_POLL_SECONDS
                if deadline is not None:
                    wait_for = max(0.01, min(wait_for, deadline - now))
                await asyncio.wait({task}, timeout=wait_for)
            return await task
        except (_RuntimeStopped, _RuntimeTimedOut, asyncio.CancelledError):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            raise

    async def _bounded_call(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        timeout: float,
        **kwargs: Any,
    ) -> Any:
        return await asyncio.wait_for(
            self._call(fn, *args, **kwargs),
            timeout=timeout,
        )

    async def _call_resource(
        self,
        fn: Callable[..., Any],
        late_cleanup: Callable[[Any | None], None],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run a blocking SDK call and clean up a value returned after cancellation."""

        concurrent = self._require_executor().submit(fn, *args, **kwargs)
        wrapped = asyncio.wrap_future(concurrent)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            wrapped.add_done_callback(_consume_asyncio_future)
            concurrent.add_done_callback(
                functools.partial(_cleanup_late_resource, cleanup=late_cleanup)
            )
            raise

    async def _call(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._require_executor(),
            functools.partial(fn, *args, **kwargs),
        )

    def _require_executor(self) -> ThreadPoolExecutor:
        executor = self._executor
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=_SDK_IO_WORKERS,
                thread_name_prefix="praxist-codex-sdk",
            )
            self._executor = executor
        return executor

    def _bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop and self._clients:
            raise RuntimeError("CodexSdkRuntime cannot share app-server clients across event loops")
        if self._loop is not loop:
            self._loop = loop
            self._client_lock = asyncio.Lock()
            self._stream_slots = asyncio.Semaphore(_MAX_ACTIVE_STREAMS)

    def _require_client_lock(self) -> asyncio.Lock:
        self._bind_loop()
        assert self._client_lock is not None
        return self._client_lock

    def _require_stream_slots(self) -> asyncio.Semaphore:
        self._bind_loop()
        assert self._stream_slots is not None
        return self._stream_slots

    def _finish(
        self,
        collector: CodexEventCollector,
        context: AgentRuntimeExecutionContext,
        **kwargs: Any,
    ) -> AgentRunResult:
        result = collector.result(**kwargs)
        self._notify(context, result.events[-1])
        return result

    @staticmethod
    def _notify(context: AgentRuntimeExecutionContext, event: AgentEvent) -> None:
        if context.message_callback is None:
            return
        try:
            context.message_callback(event)
        except Exception:  # noqa: BLE001 - observer cannot fail execution.
            logger.exception("Codex SDK runtime event callback failed for %s", event.type)


def _load_sdk() -> dict[str, Any]:
    try:
        from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
        from openai_codex.generated.v2_all import ReasoningEffort
    except ImportError as exc:
        raise RuntimeError(
            "openai-codex is required for agent_runtime:codex_sdk; install praxist[codex]"
        ) from exc
    return {
        "ApprovalMode": ApprovalMode,
        "Codex": Codex,
        "CodexConfig": CodexConfig,
        "ReasoningEffort": ReasoningEffort,
        "Sandbox": Sandbox,
    }


def available_chatgpt_models() -> tuple[str, ...]:
    """Return model identifiers advertised for the saved ChatGPT account."""

    verify_chatgpt_login()
    staged = stage_chatgpt_home(operator_codex_home())
    client: Any | None = None
    try:
        sdk = _load_sdk()
        config = sdk["CodexConfig"](
            codex_bin=resolve_codex_binary(),
            config_overrides=_SUBSCRIPTION_RUNTIME_OVERRIDES,
            env=_client_process_env("openai", os.environ, staged.path, subscription=True),
        )
        client = sdk["Codex"](config)
        response = client.models(include_hidden=False)
        return tuple(
            sorted(
                {
                    str(item.model).strip()
                    for item in getattr(response, "data", ())
                    if str(getattr(item, "model", "")).strip()
                }
            )
        )
    except Exception:  # noqa: BLE001 - never expose account or app-server details.
        raise RuntimeError("Unable to read the available ChatGPT Codex model catalog") from None
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()
        staged.close()


def verify_chatgpt_model_available(model: str) -> str:
    """Return the canonical model id or fail before a multi-peer launch."""

    requested = str(model).strip()
    if not requested:
        raise ValueError("model must be non-empty")
    by_name = {candidate.casefold(): candidate for candidate in available_chatgpt_models()}
    canonical = by_name.get(requested.casefold())
    if canonical is not None:
        return canonical
    available = ", ".join(sorted(by_name.values())) or "none reported"
    raise RuntimeError(
        f"ChatGPT account does not report model {requested!r} as available; "
        f"available models: {available}"
    )


def _output_schema(request: AgentRunRequest) -> dict[str, Any] | None:
    value = request.runtime_options.get("output_schema") if request.runtime_options else None
    return dict(value) if isinstance(value, dict) else None


def _contains_non_strict_object_schema(value: Any) -> bool:
    if isinstance(value, dict):
        schema_type = value.get("type")
        is_object = schema_type == "object" or (
            isinstance(schema_type, list) and "object" in schema_type
        )
        if is_object and value.get("additionalProperties") is not False:
            return True
        return any(_contains_non_strict_object_schema(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_non_strict_object_schema(item) for item in value)
    return False


def _state_dir(request: AgentRunRequest) -> Path:
    raw = request.runtime_options.get("run_dir") if request.runtime_options else None
    root = Path(str(raw)).expanduser() if isinstance(raw, str) and raw else Path(request.cwd)
    path = root / "runtime_state" / "codex_sdk"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _client_state_dir(request: AgentRunRequest, key: _ClientKey) -> Path:
    path = (
        _state_dir(request)
        / "clients"
        / f"{key.provider}-{key.credential_digest}-{key.reasoning_policy}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _openrouter_session_id(request: AgentRunRequest) -> str:
    """Return a stable, non-secret run scope for OpenRouter sticky routing."""

    run_dir = str(request.runtime_options.get("run_dir") or request.cwd)
    identity = "\0".join(
        (
            str(request.run_id),
            run_dir,
            str(request.model_call.provider_ref),
        )
    )
    return f"praxist-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}"


def _client_key(request: AgentRunRequest, env: Mapping[str, str]) -> _ClientKey:
    provider = provider_name(request.model_call.provider_ref)
    if _uses_chatgpt_subscription(request, env):
        credential = request.credential_ref or request.model_call.credential_ref
        identity = credential.key_id if credential is not None else ""
    else:
        key_var = provider_key_var(provider)
        identity = str(env.get(key_var) or "")
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16] if identity else "none"
    reasoning_policy = (
        effective_reasoning_effort(request.runtime_options)
        if provider in {"deepseek", "openrouter", "orcarouter"}
        else "default"
    )
    return _ClientKey(
        run_scope=str(_state_dir(request).resolve()),
        provider=provider,
        credential_digest=digest,
        reasoning_policy=reasoning_policy,
    )


def _client_process_env(
    provider: str,
    env: Mapping[str, str],
    codex_home: Path,
    *,
    subscription: bool = False,
) -> dict[str, str]:
    """Return app-server overrides with unrelated host credentials blanked."""

    key_var = provider_key_var(provider)
    key_value = str(env.get(key_var) or "")
    if subscription and provider != "openai":
        raise RuntimeError("ChatGPT authentication cannot be forwarded to a relay provider")
    if not subscription and not key_value:
        raise RuntimeError(f"{key_var} is required for Codex SDK provider {provider}")
    result = {key: "" for key in os.environ if key not in _SAFE_PROCESS_ENV_KEYS}
    result.update(
        {
            key: str(env[key])
            for key in _SAFE_PROCESS_ENV_KEYS
            if key in env and env[key] not in (None, "")
        }
    )
    if subscription:
        for key in SUBSCRIPTION_ENV_KEYS:
            result[key] = ""
        for key in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
            value = str(env.get(key) or os.environ.get(key) or "")
            if value:
                result[key] = value
    else:
        result.update({key_var: key_value, "OPENAI_API_KEY": key_value})
    result["CODEX_HOME"] = str(codex_home)
    return result


def _explicit_codex_binary(request: AgentRunRequest) -> str | None:
    """Return a request override; the SDK resolves its bundled default otherwise."""

    raw = request.runtime_options.get("codex_bin") if request.runtime_options else None
    value = str(raw).strip() if isinstance(raw, str) else ""
    return value if value and value != "codex" else None


def _uses_chatgpt_subscription(
    request: AgentRunRequest,
    env: Mapping[str, str] | None = None,
) -> bool:
    if env is not None and str(env.get("OPENAI_API_KEY") or "").strip():
        return False
    credential = request.credential_ref or request.model_call.credential_ref
    return is_chatgpt_subscription_credential(credential)


def _codex_account_type(response: Any) -> str | None:
    account = getattr(response, "account", None)
    value = getattr(account, "root", account)
    account_type = getattr(value, "type", None)
    return str(account_type) if account_type is not None else None


def _reasoning_effort(
    request: AgentRunRequest,
    effort_type: Any,
    env: Mapping[str, str],
) -> Any | None:
    policy = effective_reasoning_effort(request.runtime_options)
    if provider_name(request.model_call.provider_ref) == "deepseek":
        # The relay maps the task policy to DeepSeek's thinking contract.
        return None
    if policy == "off":
        return effort_type.none
    if policy in {"low", "high"}:
        return getattr(effort_type, policy)
    if policy == "max":
        return effort_type.xhigh
    if _uses_chatgpt_subscription(request, env):
        return effort_type.medium
    return None


def _relay_reasoning_options(
    request: AgentRunRequest,
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    """Return provider-native reasoning overrides owned by the private relay."""

    provider = provider_name(request.model_call.provider_ref)
    if provider not in {"deepseek", "openrouter", "orcarouter"}:
        return None, ()
    policy = effective_reasoning_effort(request.runtime_options)
    if policy == "auto":
        return None, ()
    if provider in {"openrouter", "orcarouter"}:
        effort = "none" if policy == "off" else policy
        return {"reasoning": {"effort": effort}}, ()
    if policy == "off":
        return {"thinking": {"type": "disabled"}}, ("reasoning_effort",)
    return {
        "thinking": {"type": "enabled"},
        "reasoning_effort": policy,
    }, ()


def _subprocess_env(
    request: AgentRunRequest,
    env: Mapping[str, str],
) -> dict[str, str]:
    """Return the scoped, non-secret task context for shell and MCP children."""

    exposed = set(request.env_policy.exposed_env_keys)
    task_keys = _task_runtime_env_keys(env)
    result = {
        str(key): str(value)
        for key, value in env.items()
        if key in exposed
        and key not in _MODEL_CREDENTIAL_ENV_KEYS
        and value not in (None, "")
        and (not _looks_secret(str(key)) or key in task_keys)
    }
    if str(result.get("PRAXIST_TASK_PYTHON") or "").strip():
        for key in ("PYTHONPATH", "PYTHONHOME"):
            if key not in task_keys:
                result.pop(key, None)
    return result


def _mcp_credential_env(
    request: AgentRunRequest,
    env: Mapping[str, str],
) -> dict[str, str]:
    """Return credentials explicitly scoped to task tools, never model providers."""

    provider_key = provider_key_var(provider_name(request.model_call.provider_ref))
    task_keys = _task_runtime_env_keys(env)
    task_keys.discard(provider_key)
    task_keys.discard("OPENAI_API_KEY")
    result = {key: str(env[key]) for key in task_keys if key in env and env[key] not in (None, "")}
    for key in ("BRAVE_API_KEY",):
        value = str(env.get(key) or os.environ.get(key) or "")
        if value:
            result[key] = value
    return result


def _task_runtime_env_keys(env: Mapping[str, str]) -> set[str]:
    return {
        item.strip()
        for item in str(env.get("PRAXIST_TASK_RUNTIME_ENV_KEYS") or "").split(",")
        if item.strip()
    }


def _runtime_model(request: AgentRunRequest) -> str | None:
    """Translate provider-specific aliases at the Codex relay boundary."""

    model = request.model_call.model or ""
    if provider_name(request.model_call.provider_ref) == "deepseek" and model.endswith("[1m]"):
        model = model[: -len("[1m]")]
    return model or None


def _looks_secret(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in ("KEY", "TOKEN", "PASSWORD", "SECRET"))


def _merge_config(*configs: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for config in configs:
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}  # type: ignore[dict-item]
            else:
                merged[key] = value
    return merged


def _stop_requested(context: AgentRuntimeExecutionContext) -> bool:
    if context.stop_requested is None:
        return False
    try:
        return bool(context.stop_requested())
    except Exception:  # noqa: BLE001 - stop polling is advisory.
        logger.debug("Codex SDK stop callback failed", exc_info=True)
        return False


def _next_item(iterator: Iterator[Any]) -> tuple[bool, Any | None]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


async def _await_task(task: asyncio.Task[Any], *, timeout: float) -> bool:
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        return True
    except TimeoutError:
        return False
    except asyncio.CancelledError:
        return True
    except Exception:  # noqa: BLE001 - draining consumes a terminal reader error.
        return True


def _observe_detached_task(task: asyncio.Task[Any]) -> None:
    def _consume(done: asyncio.Task[Any]) -> None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            done.result()

    task.add_done_callback(_consume)


def _cleanup_late_resource(
    future: ConcurrentFuture[Any],
    *,
    cleanup: Callable[[Any | None], None],
) -> None:
    value: Any | None = None
    with contextlib.suppress(Exception):
        value = future.result()
    with contextlib.suppress(Exception):
        cleanup(value)


def _ignore_late_resource(_value: Any | None) -> None:
    """Consume a late SDK value that does not own active work."""


def _interrupt_late_turn(turn: Any | None) -> None:
    """Interrupt a turn handle returned after its request already ended.

    SDK calls are blocking. Run this best-effort cleanup on a daemon thread so
    a pathological provider cannot pin the shared executor or process exit.
    """

    if turn is None or not callable(getattr(turn, "interrupt", None)):
        return

    def _run() -> None:
        with contextlib.suppress(Exception):
            turn.interrupt()

    threading.Thread(
        target=_run,
        name="praxist-codex-late-turn-interrupt",
        daemon=True,
    ).start()


def _consume_asyncio_future(future: asyncio.Future[Any]) -> None:
    with contextlib.suppress(asyncio.CancelledError, Exception):
        future.result()


def _close_late_client(
    client: Any | None,
    relay: RelayHandle | None,
    staged_chatgpt_home: StagedChatgptHome | None,
) -> None:
    if client is not None:
        with contextlib.suppress(Exception):
            client.close()
    if relay is not None:
        with contextlib.suppress(Exception):
            relay.close()
    if staged_chatgpt_home is not None:
        staged_chatgpt_home.close()


__all__ = [
    "CodexSdkRuntime",
    "available_chatgpt_models",
    "create_runtime",
    "verify_chatgpt_model_available",
]
