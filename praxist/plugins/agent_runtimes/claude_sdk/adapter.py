"""Claude Agent SDK runtime adapter.

This module owns the concrete Claude SDK import and event/message
normalization for the legacy research loop. ``BaseAgent`` remains as a
backward-compatible caller API, but the actual runtime execution lives here.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import shlex
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import TimeoutError as ConcurrentTimeoutError
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from praxist.core.protocol import (
    AgentEvent,
    AgentRunRequest,
    AgentRunResult,
    JSONValue,
    ToolCallRecord,
)
from praxist.core.redaction import redact_json, redact_text
from praxist.core.runtimes import (
    AgentRuntimeExecutionContext,
    classify_runtime_failure,
    effective_reasoning_effort,
    is_provider_access_error,
    prompt_text_for_request,
    system_prompt_text_for_request,
)
from praxist.plugins.agent_runtimes.claude_sdk._mcp import mcp_execution_options
from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
    _append_guard_warning,
    prepare_delete_guard_env,
    validate_tool_use,
)
from praxist.plugins.agent_runtimes.claude_sdk.liveness import (
    TERMINAL_BACKGROUND_TASK_STATUSES as _TERMINAL_BACKGROUND_TASK_STATUSES,
)
from praxist.plugins.agent_runtimes.claude_sdk.liveness import ClaudeSessionLiveness

logger = logging.getLogger(__name__)


_ALLOWED_SETTING_SOURCES = {"user", "project", "local"}

_CONVENTIONAL_EVALUATOR_FILENAMES = {
    "benchmark.py",
    "benchmark.sh",
    "eval.py",
    "eval.sh",
    "evaluate.py",
    "evaluate.sh",
    "evaluator.py",
    "evaluator.sh",
    "run_benchmark.py",
    "run_benchmark.sh",
    "run_eval.py",
    "run_eval.sh",
    "run_evaluation.py",
    "run_evaluation.sh",
    "run_evaluator.py",
    "run_evaluator.sh",
}
_SDK_STREAM_POLL_SECONDS = 1.0
_SDK_TERMINAL_TASK_IDLE_SECONDS = 5.0
_SDK_COMPLETED_TASK_GRACE_SECONDS = 30.0
# The pinned SDK can spend five seconds terminating its CLI process and another
# five seconds killing it.  Keep iterator cleanup inside the isolated worker,
# with enough outer budget for that SDK sequence to finish.
_SDK_QUERY_CLOSE_SECONDS = 12.0
_SDK_ISOLATED_SHUTDOWN_SECONDS = 15.0
_SDK_COOPERATIVE_YIELD_EVERY = 64
_SDK_LIVENESS_POLL_SECONDS = 30.0
_SDK_STALL_WARNING_SECONDS = 300.0
_SDK_AGGREGATED_SYSTEM_SUBTYPES = frozenset({"thinking_tokens"})
_BACKGROUND_TASK_CONTROL_TOOLS = {"agent", "task"}


def _generation_closing_signal_from_env(env: dict[str, str]) -> Path | None:
    """Return the active generation close sentinel for a peer runtime, if any."""

    run_dir_raw = env.get("PRAXIST_RUN_DIR") or env.get("AUTO_RESEARCH_RUN_DIR")
    if run_dir_raw:
        shutdown_signal = Path(run_dir_raw) / "ORCHESTRATOR_SHUTDOWN"
        try:
            if shutdown_signal.exists():
                return shutdown_signal
        except OSError:
            pass
    enabled = str(env.get("PRAXIST_LAUNCH_GUARD_ENABLED", "0")).strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    generation_raw = env.get("GENERATION_ID")
    if not run_dir_raw or generation_raw is None:
        return None
    try:
        generation_id = int(str(generation_raw).strip())
    except ValueError:
        return None
    generation_dir = Path(run_dir_raw) / f"gen_{generation_id}"
    for name in ("CLOSING_SIGNAL", "STOP_SIGNAL", "STOP_SIGNAL_POSTGEN"):
        signal = generation_dir / name
        try:
            if signal.exists():
                return signal
        except OSError:
            continue
    return None


def _closing_signal_bash_reason(command: str, env: dict[str, str]) -> str | None:
    """Deny new shell-launched work after close without blocking drain actions.

    This is a lifecycle rule, not a filesystem or security rule. It applies
    only while the current generation is draining and intentionally leaves
    findings, notebook/memory updates, and result inspection available through
    their normal tools.
    """

    signal = _generation_closing_signal_from_env(env)
    if signal is None:
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return None
    if not tokens:
        return None
    if _has_shell_background_operator(tokens):
        scope = (
            "run is shutting down"
            if signal.name == "ORCHESTRATOR_SHUTDOWN"
            else "generation is closing"
        )
        return (
            f"Praxist {scope}; new background work is frozen. Finish "
            "already-started work, publish findings, and exit."
        )
    declared_entrypoint = str(env.get("PRAXIST_EVALUATION_ENTRYPOINT") or "")
    if _command_launches_evaluator(command, declared_entrypoint):
        scope = (
            "run is shutting down"
            if signal.name == "ORCHESTRATOR_SHUTDOWN"
            else "generation is closing"
        )
        return (
            f"Praxist {scope}; launching a new evaluator is frozen. "
            "Finish already-started work, publish findings, and exit. "
            f"{signal.name} exists at {signal}."
        )
    return None


def _has_shell_background_operator(tokens: list[str]) -> bool:
    """Distinguish shell background control from file-descriptor redirection."""

    for index, token in enumerate(tokens):
        if token == "&" and not _ampersand_is_redirection(tokens, index):
            return True
    return False


def _ampersand_is_redirection(tokens: list[str], index: int) -> bool:
    previous = tokens[index - 1] if index else ""
    following = tokens[index + 1] if index + 1 < len(tokens) else ""
    return previous.endswith((">", "<")) or following.startswith(">")


def _evaluator_path_candidates(value: str) -> tuple[str, ...]:
    raw = value.strip().replace("\\", "/")
    return tuple(
        candidate.strip().strip("$(){}[]'\"").lstrip("./")
        for candidate in (raw, *raw.replace("$(", " ").split())
        if candidate.strip().strip("$(){}[]'\"").lstrip("./")
    )


def _matches_declared_evaluator_path(value: str, declared_entrypoint: str) -> bool:
    """Match only the evaluator entrypoint declared by the task contract."""

    declared = declared_entrypoint.strip().replace("\\", "/").lstrip("./")
    if not declared:
        return False
    return any(
        candidate == declared
        or candidate.endswith(f"/{declared}")
        or ("/" not in declared and Path(candidate).name == declared)
        for candidate in _evaluator_path_candidates(value)
    )


def _looks_like_evaluator_path(value: str, declared_entrypoint: str) -> bool:
    """Recognize a task's public evaluator or a conventional evaluator script."""

    if declared_entrypoint.strip():
        return _matches_declared_evaluator_path(value, declared_entrypoint)
    for normalized in _evaluator_path_candidates(value):
        if Path(normalized).name.lower() in _CONVENTIONAL_EVALUATOR_FILENAMES:
            return True
    return False


def _looks_like_env_assignment(value: str) -> bool:
    name, separator, _ = value.partition("=")
    return bool(
        separator
        and name
        and (name[0].isalpha() or name[0] == "_")
        and all(character.isalnum() or character == "_" for character in name)
    )


def _shell_command_segments(tokens: list[str]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    for index, token in enumerate(tokens):
        is_background = token == "&" and not _ampersand_is_redirection(tokens, index)
        if token in {"&&", "||", ";", "|"} or is_background:
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(token)
    return [segment for segment in segments if segment]


def _segment_launch_targets(segment: list[str]) -> list[str]:
    index = 0
    while index < len(segment) and _looks_like_env_assignment(segment[index]):
        index += 1
    while index < len(segment):
        command_name = Path(segment[index]).name.lower()
        if command_name in {"command", "exec", "nohup"}:
            index += 1
            continue
        if command_name == "env":
            index += 1
            while index < len(segment) and (
                segment[index].startswith("-") or _looks_like_env_assignment(segment[index])
            ):
                index += 1
            continue
        if command_name == "uv" and index + 1 < len(segment) and segment[index + 1] == "run":
            index += 2
            continue
        break
    if index >= len(segment):
        return []

    executable = segment[index]
    command_name = Path(executable).name.lower()
    if "python" in command_name:
        index += 1
        while index < len(segment):
            token = segment[index]
            if token == "-m" and index + 1 < len(segment):
                return [segment[index + 1]]
            if token == "-c":
                return []
            if token.startswith("-"):
                index += 1
                continue
            return [token]
        return []
    if command_name in {"bash", "dash", "sh", "zsh"}:
        for option_index in range(index + 1, len(segment)):
            if "c" in segment[option_index].lstrip("-") and segment[option_index].startswith("-"):
                if option_index + 1 < len(segment):
                    return _command_launch_targets(segment[option_index + 1])
                return []
        for token in segment[index + 1 :]:
            if not token.startswith("-"):
                return [token]
        return []
    return [executable]


def _command_launch_targets(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return []
    targets = [
        target
        for segment in _shell_command_segments(tokens)
        for target in _segment_launch_targets(segment)
    ]
    for token in tokens:
        if "$(" in token and ")" in token:
            inner = token[token.find("$(") + 2 : token.rfind(")")]
            targets.extend(_command_launch_targets(inner))
    return targets


def _command_launches_evaluator(command: str, declared_entrypoint: str) -> bool:
    return any(
        _looks_like_evaluator_path(target, declared_entrypoint)
        for target in _command_launch_targets(command)
    )


def _protected_evaluator_command(command: str, env: dict[str, str]) -> str | None:
    """Wrap one direct evaluator command in the existing protected-PID launcher.

    The wrapper recognizes the declared public evaluator (or a conventional
    evaluator script), then preserves the original Bash fragment inside the
    protected process group. This covers ordinary arguments, environment
    assignments, redirection, and intentional backgrounding without creating a
    second evaluator registry.
    """

    run_dir = str(env.get("PRAXIST_RUN_DIR") or env.get("AUTO_RESEARCH_RUN_DIR") or "").strip()
    peer_id = str(env.get("PRAXIST_PEER_ID") or env.get("PEER_ID") or "").strip()
    if not run_dir or not peer_id or "protected_pids" in command:
        return None
    declared_entrypoint = str(env.get("PRAXIST_EVALUATION_ENTRYPOINT") or "")
    if not _command_launches_evaluator(command, declared_entrypoint):
        return None
    # This rewrite is a compatibility fallback for a peer that ignored the
    # explicit scheduler command. Never collapse every evaluator invocation
    # into one central semantic ID. Proper task prompts supply a stable
    # scientific tag; here a command digest is the least surprising fallback.
    fallback_id = "auto-command-" + hashlib.sha256(command.strip().encode()).hexdigest()[:16]
    launcher = (
        f"{shlex.quote(sys.executable)} -m "
        "praxist.plugins.workflow_stages.research_loop.backend.protected_pids "
        f"launch --run-dir {shlex.quote(run_dir)} --peer {shlex.quote(peer_id)} "
        f"--tag {fallback_id} --work-class ordinary -- bash -lc "
        f"{shlex.quote(command.strip())}"
    )
    return launcher


def claude_setting_sources_from_env() -> list[str]:
    """Return the Claude Code settings scope for Praxist autonomous runs.

    Praxist passes MCP servers, permissions, cwd, model, and credentials explicitly
    through the SDK options. Defaulting to project settings lets unrelated
    Claude Code project context compete with the Praxist research task and can
    cause a peer to wait for human instructions. Local settings preserve
    operator-local overrides without loading project/user memory by default.
    """

    raw = os.environ.get("PRAXIST_CLAUDE_SETTING_SOURCES", "local").strip()
    if raw.lower() in {"", "none", "off", "disabled"}:
        return []
    sources: list[str] = []
    for item in raw.split(","):
        source = item.strip().lower()
        if not source:
            continue
        if source not in _ALLOWED_SETTING_SOURCES:
            logger.warning(
                "Ignoring unsupported PRAXIST_CLAUDE_SETTING_SOURCES item: %s",
                source,
            )
            continue
        if source not in sources:
            sources.append(source)
    return sources


@dataclass(frozen=True)
class LegacyClaudeRuntimeOptions:
    """Compatibility options used while legacy BaseAgent calls the Claude SDK runtime adapter."""

    name: str
    allowed_tools: list[str]
    workspace: Path
    mcp_servers: dict[str, Any]
    model: str
    permission_mode: str
    cli_path: str | None = None
    message_callback: Callable[[Any], None] | None = None
    system_prompt: str | None = None
    stop_check_fn: Callable[[], bool] | None = None
    premium_mode: bool = False
    env: dict[str, str] | None = None
    require_no_shell_runtime: bool = False
    runtime_timeout_seconds: int | None = None
    tool_execution_timeout_seconds: float | None = None
    liveness: ClaudeSessionLiveness | None = None
    model_provider_ref: str = ""
    reasoning_effort: str = "max"


def _claude_reasoning_options(options: LegacyClaudeRuntimeOptions) -> dict[str, Any]:
    """Map the provider-neutral policy onto the selected Claude wire API."""

    policy = effective_reasoning_effort(
        {
            "premium_mode": options.premium_mode,
            "reasoning_effort": options.reasoning_effort,
        }
    )
    if policy == "auto":
        return {}
    if policy == "off":
        return {"thinking": {"type": "disabled"}}
    if policy == "xhigh":
        policy = "max"
    provider = options.model_provider_ref.rsplit(":", 1)[-1].removesuffix("_alias").lower()
    if provider == "deepseek":
        # ClaudeAgentOptions requires budget_tokens for enabled thinking;
        # DeepSeek accepts the field for compatibility and ignores its value.
        thinking: dict[str, Any] = {"type": "enabled", "budget_tokens": 1024}
    else:
        thinking = {"type": "adaptive"}
    return {"thinking": thinking, "effort": policy}


@dataclass
class LegacyAgentResult:
    """Legacy-shaped terminal result returned to older research_loop callers."""

    success: bool
    output: dict[str, Any]
    duration: float
    iteration_count: int
    error: str | None = None
    usage: dict[str, float] = field(default_factory=dict)
    terminal_status: str | None = None
    timed_out: bool = False
    cancelled: bool = False


class ClaudeSdkAgentRuntime:
    """AgentRuntime adapter that drives Claude Code SDK and normalizes events for Praxist."""

    runtime_ref = "agent_runtime:claude_sdk"

    async def execute(
        self,
        request: AgentRunRequest,
        context: AgentRuntimeExecutionContext,
    ) -> AgentRunResult:
        """Execute a normalized AgentRunRequest via the Claude SDK adapter."""
        if request.agent_runtime_ref != self.runtime_ref:
            return _agent_run_result_from_legacy(
                request,
                LegacyAgentResult(
                    success=False,
                    output={},
                    duration=0.0,
                    iteration_count=0,
                    error=f"runtime mismatch: expected {self.runtime_ref}, got {request.agent_runtime_ref}",
                ),
            )

        runtime_options = request.runtime_options or {}
        if runtime_options.get("require_task_sandbox_policy") or runtime_options.get(
            "require_task_tool_policy"
        ):
            return _agent_run_result_from_legacy(
                request,
                LegacyAgentResult(
                    success=False,
                    output={},
                    duration=0.0,
                    iteration_count=0,
                    error="claude_sdk cannot enforce a task-wide sandbox or tool policy; select a runtime with sandbox and tool enforcement",
                    terminal_status="failed",
                ),
            )
        # Preserve raw timeout values for mcp_execution_options to validate;
        # coercion here would accidentally accept booleans or numeric strings.
        legacy_result = await self._execute_legacy_isolated(
            prompt_text_for_request(request),
            LegacyClaudeRuntimeOptions(
                name=request.request_id,
                allowed_tools=list(request.tool_permissions.allowed_tools),
                workspace=Path(request.cwd),
                mcp_servers=dict(context.tool_servers),
                model=request.model_call.model,
                model_provider_ref=request.model_call.provider_ref,
                permission_mode=str(runtime_options.get("permission_mode") or "bypassPermissions"),
                cli_path=str(runtime_options.get("cli_path") or "") or None,
                message_callback=context.message_callback,
                system_prompt=system_prompt_text_for_request(request),
                stop_check_fn=context.stop_requested,
                premium_mode=bool(runtime_options.get("premium_mode")),
                reasoning_effort=str(runtime_options.get("reasoning_effort") or "max"),
                env=dict(context.env),
                require_no_shell_runtime=bool(runtime_options.get("require_no_shell_runtime")),
                runtime_timeout_seconds=request.timeout_seconds or None,
                tool_execution_timeout_seconds=cast(
                    float | None, runtime_options.get("tool_execution_timeout_seconds")
                ),
            ),
        )
        return _agent_run_result_from_legacy(request, legacy_result)

    async def _execute_legacy_isolated(
        self,
        task: str,
        options: LegacyClaudeRuntimeOptions,
    ) -> LegacyAgentResult:
        """Run one Claude SDK session on its own event-loop thread.

        Claude SDK stream consumption, in-process MCP callbacks, and hooks can
        produce sustained bursts.  Keeping those bursts on the research-loop
        event loop lets one session delay generation timers and every other
        peer.  A daemon thread gives each session an independent event loop
        while retaining the existing in-process SDK/MCP implementation.
        """

        cancel_requested = threading.Event()
        parent_loop = asyncio.get_running_loop()
        original_stop_check = options.stop_check_fn
        liveness = ClaudeSessionLiveness()

        def stop_requested() -> bool:
            if cancel_requested.is_set():
                return True
            if original_stop_check is None:
                return False
            return bool(original_stop_check())

        def deliver_message(message: Any) -> None:
            callback = options.message_callback
            if cancel_requested.is_set() or callback is None:
                return
            delivered: ConcurrentFuture[None] = ConcurrentFuture()

            def invoke_callback() -> None:
                if delivered.done():
                    return
                if cancel_requested.is_set():
                    delivered.set_result(None)
                    return
                try:
                    callback(message)
                except BaseException as exc:  # noqa: BLE001 - preserve callback failures.
                    delivered.set_exception(exc)
                else:
                    delivered.set_result(None)

            try:
                parent_loop.call_soon_threadsafe(invoke_callback)
            except RuntimeError:
                return
            while not cancel_requested.is_set():
                try:
                    delivered.result(timeout=0.1)
                    return
                except ConcurrentTimeoutError:
                    continue
            delivered.cancel()

        worker_options = replace(
            options,
            stop_check_fn=stop_requested,
            message_callback=deliver_message,
            liveness=liveness,
        )
        concurrent: ConcurrentFuture[LegacyAgentResult] = ConcurrentFuture()
        worker_lock = threading.Lock()
        worker_loop: list[asyncio.AbstractEventLoop | None] = [None]
        worker_task: list[asyncio.Task[LegacyAgentResult] | None] = [None]

        def request_worker_cancel() -> None:
            cancel_requested.set()
            liveness.mark_closing()
            with worker_lock:
                loop = worker_loop[0]
                task = worker_task[0]
            if loop is not None and task is not None and not task.done():
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(task.cancel)

        def run_session() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task = loop.create_task(self.execute_legacy(task_text, worker_options))
            with worker_lock:
                worker_loop[0] = loop
                worker_task[0] = task
            if cancel_requested.is_set():
                task.cancel()
            try:
                result = loop.run_until_complete(task)
            except asyncio.CancelledError:
                result = LegacyAgentResult(
                    success=False,
                    output={},
                    duration=0.0,
                    iteration_count=0,
                    error="agent runtime cancelled",
                    terminal_status="cancelled",
                    cancelled=True,
                )
            except BaseException as exc:  # noqa: BLE001 - propagate worker failure.
                if not concurrent.done():
                    concurrent.set_exception(exc)
            else:
                if not concurrent.done():
                    concurrent.set_result(result)
            finally:
                liveness.mark_terminal()
                with worker_lock:
                    worker_task[0] = None
                    worker_loop[0] = None
                pending = [pending for pending in asyncio.all_tasks(loop) if not pending.done()]
                for pending_task in pending:
                    pending_task.cancel()
                if pending:
                    with contextlib.suppress(Exception):
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                with contextlib.suppress(Exception):
                    loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        task_text = task
        worker = threading.Thread(
            target=run_session,
            name=f"praxist-claude-sdk-{options.name}"[:120],
            daemon=True,
        )
        worker.start()
        wrapped = asyncio.wrap_future(concurrent)
        isolated_started = time.monotonic()
        isolated_deadline = (
            isolated_started + max(1, int(options.runtime_timeout_seconds))
            if options.runtime_timeout_seconds
            else None
        )
        try:
            warned_progress_at: float | None = None
            reported_active_state: tuple[str, float] | None = None
            while not wrapped.done():
                wait_seconds = _SDK_LIVENESS_POLL_SECONDS
                if isolated_deadline is not None:
                    wait_seconds = min(
                        wait_seconds,
                        max(0.0, isolated_deadline - time.monotonic()),
                    )
                await asyncio.wait({wrapped}, timeout=wait_seconds)
                if (
                    isolated_deadline is not None
                    and time.monotonic() >= isolated_deadline
                    and not wrapped.done()
                ):
                    request_worker_cancel()
                    logger.warning(
                        "Agent %s reached its configured runtime timeout in the isolated "
                        "supervisor; releasing the research-loop task.",
                        options.name,
                    )
                    try:
                        result = await asyncio.wait_for(
                            asyncio.shield(wrapped),
                            timeout=_SDK_ISOLATED_SHUTDOWN_SECONDS,
                        )
                    except asyncio.CancelledError:
                        raise
                    except TimeoutError:
                        return LegacyAgentResult(
                            success=False,
                            output={},
                            duration=time.monotonic() - isolated_started,
                            iteration_count=0,
                            error="agent runtime timeout",
                            terminal_status="timeout",
                            timed_out=True,
                        )
                    if result.timed_out:
                        return result
                    return replace(
                        result,
                        success=False,
                        error="agent runtime timeout",
                        terminal_status="timeout",
                        timed_out=True,
                        cancelled=False,
                    )
                now = time.monotonic()
                observation = liveness.observation()
                state = str(observation["session_state"])
                progress_at = float(observation["latest_progress_at"])
                state_started_at = float(observation["state_started_at"])
                if state in {"foreground_tool_running", "background_work_running"}:
                    active_state = (state, state_started_at)
                    if (
                        now - state_started_at >= _SDK_STALL_WARNING_SECONDS
                        and reported_active_state != active_state
                    ):
                        reported_active_state = active_state
                        tools = ",".join(observation["active_foreground_tools"]) or "-"
                        logger.info(
                            "Agent %s: Claude SDK liveness observed_state=%s for %.0fs; "
                            "active_tools=%s; active_background_tasks=%d; "
                            "worker_heartbeat_age=%.0fs.",
                            options.name,
                            state,
                            now - state_started_at,
                            tools,
                            observation["active_background_tasks"],
                            now - float(observation["tool_progress_at"]),
                        )
                elif (
                    state == "model_waiting"
                    and now - progress_at >= _SDK_STALL_WARNING_SECONDS
                    and warned_progress_at != progress_at
                ):
                    warned_progress_at = progress_at
                    logger.warning(
                        "Agent %s: Claude SDK liveness observed_state=model_waiting with no "
                        "complete message, partial stream event, or tool progress for %.0fs; "
                        "the isolated session remains bounded by its existing runtime and "
                        "generation deadlines.",
                        options.name,
                        now - progress_at,
                    )
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            request_worker_cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(wrapped),
                    timeout=_SDK_ISOLATED_SHUTDOWN_SECONDS,
                )
            except BaseException:  # noqa: BLE001 - preserve the caller's cancellation.
                logger.warning(
                    "Agent %s: isolated Claude SDK session did not close within %.1fs; "
                    "the daemon worker remains detached from research-loop progress.",
                    options.name,
                    _SDK_ISOLATED_SHUTDOWN_SECONDS,
                )
            raise

    def execute_sync(self, request: AgentRunRequest) -> AgentRunResult:
        """Return a deterministic transcript for offline fixture runtime conformance."""
        return _TranscriptRuntime(self.runtime_ref, "claude_sdk_mocked_transcript").execute_sync(
            request
        )

    async def execute_legacy(
        self, task: str, options: LegacyClaudeRuntimeOptions
    ) -> LegacyAgentResult:
        start_time = time.time()
        liveness = options.liveness or ClaudeSessionLiveness()
        iteration_count = 0
        messages: list[Any] = []
        system_event_counts: dict[str, int] = {}
        stream_message_count = 0
        query_iter: Any | None = None
        next_message_task: asyncio.Future[Any] | None = None

        async def _cancel_pending_message() -> None:
            nonlocal next_message_task
            pending = next_message_task
            next_message_task = None
            if pending is None or pending.done():
                return
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pending

        async def _close_query_iterator() -> None:
            nonlocal query_iter
            iterator = query_iter
            query_iter = None
            close = getattr(iterator, "aclose", None)
            if not callable(close):
                return
            try:
                await asyncio.wait_for(
                    cast(Awaitable[Any], close()), timeout=_SDK_QUERY_CLOSE_SECONDS
                )
            except TimeoutError:
                logger.warning(
                    "Agent %s: Claude SDK query iterator did not close within %.1fs.",
                    options.name,
                    _SDK_QUERY_CLOSE_SECONDS,
                )
            except Exception as exc:  # noqa: BLE001 - cleanup remains best-effort.
                redacted_error, _ = redact_text(str(exc))
                logger.warning(
                    "Agent %s: Claude SDK query iterator close failed: %s",
                    options.name,
                    redacted_error,
                )

        def collected_output() -> dict[str, Any]:
            output = extract_legacy_output(messages)
            if system_event_counts:
                output["sdk_system_event_counts"] = dict(sorted(system_event_counts.items()))
            return output

        try:
            sdk = _load_claude_sdk()
            if options.require_no_shell_runtime:
                runtime_env = dict(options.env or {})
            else:
                runtime_env = prepare_delete_guard_env(
                    options.env or {},
                    workspace=options.workspace,
                    agent_name=options.name,
                )
            mcp_servers, runtime_env = mcp_execution_options(
                options.mcp_servers, runtime_env, options.tool_execution_timeout_seconds
            )
            reasoning_options = _claude_reasoning_options(options)
            if (
                effective_reasoning_effort(
                    {
                        "premium_mode": options.premium_mode,
                        "reasoning_effort": options.reasoning_effort,
                    }
                )
                != "auto"
            ):
                runtime_env.pop("CLAUDE_CODE_EFFORT_LEVEL", None)

            async def _pre_tool_use_delete_guard(
                hook_input: Any, tool_use_id: str | None, context: Any
            ) -> dict[str, Any]:
                tool_name = _hook_value(hook_input, "tool_name")
                if options.require_no_shell_runtime:
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                "Praxist DIG-Lite read-only planner denied "
                                f"{tool_name or 'tool'}: shell and write tools are disabled "
                                "before selected_contract validation."
                            ),
                        }
                    }
                tool_input: dict[str, Any] = {}
                try:
                    tool_input = _hook_value(hook_input, "tool_input") or {}
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    if str(tool_name or "") == "Bash":
                        closing_reason = _closing_signal_bash_reason(
                            str(tool_input.get("command") or ""), runtime_env
                        )
                        if closing_reason:
                            logger.info(
                                "Agent %s: generation close denied Bash launch: %s",
                                options.name,
                                closing_reason,
                            )
                            return {
                                "hookSpecificOutput": {
                                    "hookEventName": "PreToolUse",
                                    "permissionDecision": "deny",
                                    "permissionDecisionReason": closing_reason,
                                }
                            }
                    protected_evaluator_command = (
                        _protected_evaluator_command(
                            str(tool_input.get("command") or ""), runtime_env
                        )
                        if str(tool_name or "") == "Bash"
                        else None
                    )
                    decision = validate_tool_use(
                        str(tool_name or ""),
                        tool_input,
                        env=runtime_env,
                        cwd=options.workspace,
                    )
                except Exception as exc:  # noqa: BLE001 - hook must stay available.
                    if str(tool_name or "") == "Bash":
                        message = (
                            "Praxist runtime guard validator hit an internal error while "
                            f"checking Bash; command is allowed with warning: {exc}"
                        )
                        logger.exception(
                            "Agent %s: delete guard validator warned after internal "
                            "error for Bash tool use.",
                            options.name,
                        )
                        _append_guard_warning(
                            env=runtime_env,
                            rule_id="validator_exception",
                            message=message,
                            command=str((tool_input or {}).get("command") or ""),
                            tool_name="Bash",
                        )
                        return {
                            "hookSpecificOutput": {
                                "hookEventName": "PreToolUse",
                                "permissionDecision": "allow",
                                "permissionDecisionReason": message,
                            }
                        }
                    logger.exception(
                        "Agent %s: delete guard validator failed; denying %s tool use.",
                        options.name,
                        tool_name,
                    )
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                "Praxist delete guard failed closed while validating "
                                f"{tool_name or 'tool'} use: {exc}"
                            ),
                        }
                    }
                if decision.allowed:
                    if getattr(decision, "warning", False):
                        logger.warning(
                            "Agent %s: delete guard warned for %s tool use: %s",
                            options.name,
                            tool_name,
                            decision.message,
                        )
                        _append_guard_warning(
                            env=runtime_env,
                            rule_id=decision.rule_id or "pretool_warning",
                            message=decision.message,
                            command=str(tool_input.get("command") or ""),
                            tool_name=str(tool_name or ""),
                        )
                        hook_output: dict[str, Any] = {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "allow",
                            "permissionDecisionReason": decision.message,
                        }
                        if protected_evaluator_command is not None:
                            hook_output["updatedInput"] = {
                                **tool_input,
                                "command": protected_evaluator_command,
                            }
                        return {"hookSpecificOutput": hook_output}
                    if protected_evaluator_command is not None:
                        return {
                            "hookSpecificOutput": {
                                "hookEventName": "PreToolUse",
                                "permissionDecision": "allow",
                                "permissionDecisionReason": (
                                    "Praxist registered this evaluator through the existing "
                                    "protected process-group launcher."
                                ),
                                "updatedInput": {
                                    **tool_input,
                                    "command": protected_evaluator_command,
                                },
                            }
                        }
                    return {}
                logger.warning(
                    "Agent %s: delete guard denied %s tool use: %s",
                    options.name,
                    tool_name,
                    decision.message,
                )
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": decision.message,
                    }
                }

            options_dict: dict[str, Any] = {
                "allowed_tools": options.allowed_tools,
                "system_prompt": options.system_prompt,
                "permission_mode": options.permission_mode,
                "cwd": str(options.workspace),
                "model": options.model,
                "mcp_servers": mcp_servers,
                "setting_sources": claude_setting_sources_from_env(),
                "betas": ["context-1m-2025-08-07"],
                "env": runtime_env,
                "include_partial_messages": True,
            }
            if options.require_no_shell_runtime:
                options_dict["tools"] = list(options.allowed_tools)
                options_dict["mcp_servers"] = {}
                options_dict["permission_mode"] = "default"
                options_dict["disallowed_tools"] = [
                    "Bash",
                    "Write",
                    "Edit",
                    "MultiEdit",
                    "NotebookEdit",
                ]
                no_shell_directive = (
                    "\n\nDIG-Lite read-only planner mode is active. You may only use "
                    "the listed read-only tools. Never request Bash, Write, Edit, "
                    "MultiEdit, NotebookEdit, shell commands, or plan-file writes. "
                    "Return the requested structured JSON object directly."
                )
                options_dict["system_prompt"] = (
                    (options.system_prompt or "") + no_shell_directive
                ).strip()
            options_dict["hooks"] = {
                "PreToolUse": [
                    sdk["HookMatcher"](
                        matcher=tool_name,
                        hooks=[_pre_tool_use_delete_guard],
                    )
                    for tool_name in ("Bash", "Write", "Edit", "MultiEdit", "NotebookEdit")
                ]
            }
            options_dict.update(reasoning_options)
            if options.cli_path:
                options_dict["cli_path"] = options.cli_path

            sdk_options = sdk["ClaudeAgentOptions"](**options_dict)

            query_iter = sdk["query"](prompt=task, options=sdk_options).__aiter__()
            deadline = (
                time.monotonic() + max(1, int(options.runtime_timeout_seconds))
                if options.runtime_timeout_seconds
                else None
            )
            terminal_background_since: float | None = None
            interrupted_by_stop = False

            def arm_terminal_background_timer_if_idle() -> None:
                nonlocal terminal_background_since
                if liveness.all_background_terminal() and not liveness.foreground_work_active():
                    terminal_background_since = terminal_background_since or time.monotonic()
                else:
                    terminal_background_since = None

            def stop_requested() -> bool:
                if options.stop_check_fn is None:
                    return False
                try:
                    return bool(options.stop_check_fn())
                except Exception as exc:  # noqa: BLE001 - stop hooks are best-effort.
                    logger.debug("Agent %s: stop_check_fn raised: %s", options.name, exc)
                    return False

            while True:
                if next_message_task is None:
                    next_message_task = asyncio.ensure_future(query_iter.__anext__())
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("agent runtime timeout")
                poll_seconds = max(0.01, float(_SDK_STREAM_POLL_SECONDS))
                if remaining is not None:
                    poll_seconds = min(poll_seconds, remaining)
                done, _pending = await asyncio.wait(
                    {next_message_task},
                    timeout=poll_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    liveness.record_active_work_heartbeat()
                    if stop_requested():
                        interrupted_by_stop = True
                        liveness.mark_closing()
                        logger.info(
                            "Agent %s: stop_check_fn returned True after %d tool uses; "
                            "exiting SDK message loop.",
                            options.name,
                            iteration_count,
                        )
                        break
                    if (
                        terminal_background_since is not None
                        and time.monotonic() - terminal_background_since
                        >= max(
                            0.01,
                            float(
                                _SDK_TERMINAL_TASK_IDLE_SECONDS
                                if liveness.any_background_failed()
                                else _SDK_COMPLETED_TASK_GRACE_SECONDS
                            ),
                        )
                    ):
                        logger.info(
                            "Agent %s: all observed background tasks are terminal and the "
                            "SDK stream stayed idle; closing the completed session.",
                            options.name,
                        )
                        liveness.mark_terminal()
                        break
                    continue
                completed_message_task = next_message_task
                next_message_task = None
                try:
                    message = completed_message_task.result()
                except StopAsyncIteration:
                    liveness.mark_terminal()
                    break
                class_name = type(message).__name__
                if class_name == "StreamEvent":
                    liveness.record_model_stream_progress()
                    if stop_requested():
                        interrupted_by_stop = True
                        liveness.mark_closing()
                        break
                    continue

                stream_message_count += 1
                liveness.record_complete_message()
                if (
                    class_name == "SystemMessage"
                    and str(getattr(message, "subtype", "") or "unknown")
                    in _SDK_AGGREGATED_SYSTEM_SUBTYPES
                ):
                    subtype = str(getattr(message, "subtype", "") or "unknown")
                    system_event_counts[subtype] = system_event_counts.get(subtype, 0) + 1
                    if stop_requested():
                        interrupted_by_stop = True
                        liveness.mark_closing()
                        break
                    if stream_message_count % _SDK_COOPERATIVE_YIELD_EVERY == 0:
                        await asyncio.sleep(0)
                    continue

                messages.append(message)
                if class_name in {
                    "TaskStartedMessage",
                    "TaskProgressMessage",
                    "TaskNotificationMessage",
                    "TaskUpdatedMessage",
                }:
                    task_payload = _background_task_message_payload(message)
                    task_id = str(task_payload.get("task_id") or "").strip()
                    status = str(task_payload.get("status") or "").strip().lower()
                    if task_id:
                        if not status and class_name in {
                            "TaskStartedMessage",
                            "TaskProgressMessage",
                        }:
                            status = "running"
                        if status:
                            liveness.record_background_status(task_id, status)
                    arm_terminal_background_timer_if_idle()

                log_msg = format_legacy_message(message, options.name)
                if log_msg:
                    logger.info(log_msg)

                if options.message_callback:
                    options.message_callback(message)

                if stop_requested():
                    interrupted_by_stop = True
                    liveness.mark_closing()
                    logger.info(
                        "Agent %s: stop_check_fn returned True after %d tool uses; "
                        "exiting SDK message loop.",
                        options.name,
                        iteration_count,
                    )
                    break

                if _is_instance(message, "ResultMessage", sdk):
                    liveness.mark_terminal()
                    break

                content_blocks = getattr(message, "content", []) or []
                if isinstance(content_blocks, (list, tuple)):
                    for content in content_blocks:
                        if _is_instance(content, "ToolUseBlock", sdk):
                            if _is_instance(message, "AssistantMessage", sdk):
                                iteration_count += 1
                            tool_name = str(getattr(content, "name", "") or "").strip().lower()
                            if tool_name not in _BACKGROUND_TASK_CONTROL_TOOLS:
                                tool_id = str(getattr(content, "id", "") or "").strip()
                                liveness.start_foreground_tool(tool_name, tool_id)
                                terminal_background_since = None
                        elif _is_instance(content, "ToolResultBlock", sdk):
                            tool_id = str(getattr(content, "tool_use_id", "") or "").strip()
                            liveness.finish_foreground_tool(tool_id)
                            arm_terminal_background_timer_if_idle()

                if stream_message_count % _SDK_COOPERATIVE_YIELD_EVERY == 0:
                    await asyncio.sleep(0)

            await _cancel_pending_message()
            output = collected_output()
            runtime_error = _legacy_result_error(output)
            terminal_background_only = _legacy_output_has_only_terminal_background_tasks(output)
            if terminal_background_only:
                output["terminal_background_only"] = True
            # Normal assistant output is research content, not an error
            # channel.  Classifying it for auth/billing keywords can turn a
            # valid Chair agenda into a runtime failure.  Only inspect the
            # SDK-declared terminal error envelope.
            if runtime_error and is_billing_error(runtime_error):
                redacted_text, _ = redact_text(runtime_error[:300])
                return LegacyAgentResult(
                    success=False,
                    output=output,
                    duration=time.time() - start_time,
                    iteration_count=iteration_count,
                    error=f"API error in output: {redacted_text}",
                )
            if runtime_error:
                return LegacyAgentResult(
                    success=False,
                    output=output,
                    duration=time.time() - start_time,
                    iteration_count=iteration_count,
                    error=runtime_error,
                )
            if _legacy_output_has_only_failed_background_tasks(output):
                return LegacyAgentResult(
                    success=False,
                    output=output,
                    duration=time.time() - start_time,
                    iteration_count=iteration_count,
                    error="all observed background tasks failed or stopped",
                )

            return LegacyAgentResult(
                success=True,
                output=output,
                duration=time.time() - start_time,
                iteration_count=iteration_count,
                usage=dict(output.get("usage") or {}),
                terminal_status="cancelled" if interrupted_by_stop else "completed",
                cancelled=interrupted_by_stop,
            )
        except asyncio.CancelledError:
            liveness.mark_closing()
            await _cancel_pending_message()
            logger.info("Agent %s cancelled after %d turns", options.name, iteration_count)
            raise
        except TimeoutError:
            liveness.mark_closing()
            await _cancel_pending_message()
            logger.warning(
                "Agent %s hit runtime timeout after %d turns", options.name, iteration_count
            )
            output = collected_output()
            return LegacyAgentResult(
                success=False,
                output=output,
                duration=time.time() - start_time,
                iteration_count=iteration_count,
                error="agent runtime timeout",
                usage=dict(output.get("usage") or {}),
                terminal_status="timeout",
                timed_out=True,
            )
        except Exception as exc:  # noqa: BLE001 - adapter reports runtime failure to caller.
            liveness.mark_closing()
            await _cancel_pending_message()
            redacted_error, _ = redact_text(str(exc))
            logger.error("Agent %s failed: %s", options.name, redacted_error)
            output = collected_output()
            if _legacy_output_has_only_terminal_background_tasks(output):
                output["terminal_background_only"] = True
            return LegacyAgentResult(
                success=False,
                output=output,
                duration=time.time() - start_time,
                iteration_count=iteration_count,
                error=redacted_error,
            )
        finally:
            liveness.mark_terminal()
            await _cancel_pending_message()
            await _close_query_iterator()


def is_billing_error(error_msg: str) -> bool:
    """Classify whether an error string represents provider billing or quota failure."""
    return is_provider_access_error(error_msg)


def _background_task_message_payload(message: Any) -> dict[str, Any]:
    patch = getattr(message, "patch", None)
    patch = patch if isinstance(patch, dict) else {}

    def field(name: str) -> Any:
        value = getattr(message, name, None)
        return patch.get(name, "") if value in (None, "") else value

    return {
        "task_id": field("task_id"),
        "status": field("status"),
        "output_file": field("output_file"),
        "summary": field("summary"),
    }


def format_legacy_message(message: Any, agent_name: str) -> str | None:
    """Format a Claude SDK stream message for legacy logs without leaking raw provider objects."""
    ts = datetime.now().strftime("%H:%M:%S")
    class_name = type(message).__name__
    if class_name == "AssistantMessage":
        parts = []
        for content in getattr(message, "content", []) or []:
            content_name = type(content).__name__
            if content_name == "TextBlock":
                text, _ = redact_text(str(getattr(content, "text", ""))[:200])
                parts.append(f"[{ts}] [{agent_name}] {text}")
            elif content_name == "ToolUseBlock":
                tool_input = getattr(content, "input", None) or {}
                detail = ""
                name = getattr(content, "name", "")
                if name == "Bash":
                    command, _ = redact_text(str(tool_input.get("command", ""))[:100])
                    detail = f" {command}"
                elif name in ("Read", "Write", "Edit"):
                    file_path, _ = redact_text(str(tool_input.get("file_path", "")))
                    detail = f" {file_path}"
                parts.append(f"[{ts}] [{agent_name}] -> {name}{detail}")
        return "\n".join(parts) if parts else None
    if class_name == "ResultMessage":
        return f"[{ts}] [{agent_name}] Done"
    if class_name in {
        "TaskStartedMessage",
        "TaskProgressMessage",
        "TaskNotificationMessage",
        "TaskUpdatedMessage",
    }:
        payload = _background_task_message_payload(message)
        task_id, _ = redact_text(str(payload["task_id"]))
        status, _ = redact_text(str(payload["status"] or class_name))
        summary, _ = redact_text(str(payload["summary"])[:200])
        suffix = f" {summary}" if summary else ""
        return f"[{ts}] [{agent_name}] background task {task_id}: {status}{suffix}"
    return None


def extract_legacy_output(messages: list[Any]) -> dict[str, Any]:
    """Extract text, thinking blocks, tool counts, and usage from Claude SDK stream messages."""
    output: dict[str, Any] = {"text_outputs": [], "tool_uses": []}
    for msg in messages:
        class_name = type(msg).__name__
        if class_name == "TaskNotificationMessage" or (
            class_name == "TaskUpdatedMessage"
            and str(_background_task_message_payload(msg).get("status") or "").lower()
            in {"completed", "failed", "killed", "stopped"}
        ):
            task_payload, _ = redact_json(_background_task_message_payload(msg))
            output.setdefault("background_tasks", []).append(task_payload)
        if class_name == "AssistantMessage":
            for content in getattr(msg, "content", []) or []:
                content_name = type(content).__name__
                if content_name == "TextBlock":
                    text, _ = redact_text(str(getattr(content, "text", "")))
                    output["text_outputs"].append(text)
                elif content_name == "ToolUseBlock":
                    tool_input, _ = redact_json(getattr(content, "input", None))
                    output["tool_uses"].append(
                        {"tool": getattr(content, "name", ""), "input": tool_input}
                    )
                elif content_name == "ThinkingBlock":
                    thinking_text = getattr(content, "thinking", None) or getattr(
                        content, "text", None
                    )
                    if isinstance(thinking_text, str) and thinking_text.strip():
                        text, _ = redact_text(thinking_text)
                        output.setdefault("thinking_outputs", []).append(text)
        if class_name == "ResultMessage":
            result, _ = redact_json(getattr(msg, "result", None))
            output["result_message"] = result
            if bool(getattr(msg, "is_error", False)):
                output["result_is_error"] = True
            errors = getattr(msg, "errors", None)
            if errors not in (None, "", [], {}, ()):
                redacted_errors, _ = redact_json(errors)
                output["result_errors"] = redacted_errors
            usage = _normalized_usage(getattr(msg, "usage", None))
            if usage:
                output["usage"] = usage
    return output


def _normalized_usage(value: Any) -> dict[str, float]:
    """Normalize Claude usage while preserving cache read/create semantics."""

    dump = getattr(value, "model_dump", None)
    if callable(dump):
        value = dump(mode="json", by_alias=True, exclude_none=True)
    if not isinstance(value, dict):
        return {}

    def first_number(*sources: str) -> float | None:
        for source in sources:
            raw = value.get(source)
            if isinstance(raw, (int, float)):
                return float(raw)
        return None

    uncached = first_number("input_tokens", "inputTokens")
    cache_read = first_number(
        "cache_read_input_tokens",
        "cached_input_tokens",
        "cachedInputTokens",
    )
    cache_creation = first_number(
        "cache_creation_input_tokens",
        "cacheCreationInputTokens",
    )
    output = first_number("output_tokens", "outputTokens")
    if all(item is None for item in (uncached, cache_read, cache_creation, output)):
        return {}

    inclusive_input = sum(item or 0.0 for item in (uncached, cache_read, cache_creation))
    usage: dict[str, float] = {
        "input_tokens": inclusive_input,
        "total_input_tokens": inclusive_input,
        "uncached_input_tokens": uncached or 0.0,
        "cached_input_tokens": cache_read or 0.0,
        "cache_read_input_tokens": cache_read or 0.0,
        "cache_creation_input_tokens": cache_creation or 0.0,
        "output_tokens": output or 0.0,
        "total_tokens": inclusive_input + (output or 0.0),
    }
    return usage


def _legacy_result_error(output: dict[str, Any]) -> str | None:
    """Return a redacted SDK-declared terminal error, if present."""

    if not bool(output.get("result_is_error")):
        return None
    parts: list[str] = []
    errors = output.get("result_errors")
    if isinstance(errors, (list, tuple, set)):
        parts.extend(str(item).strip() for item in errors if str(item).strip())
    elif errors not in (None, "", {}, []):
        parts.append(str(errors).strip())
    result = output.get("result_message")
    if result not in (None, "", {}, []):
        rendered = str(result).strip()
        if rendered and rendered not in parts:
            parts.append(rendered)
    message = "; ".join(parts) or "Claude SDK returned an error result"
    redacted, _ = redact_text(message[:1000])
    return redacted


def _legacy_output_has_only_failed_background_tasks(output: dict[str, Any]) -> bool:
    if not _legacy_output_has_only_terminal_background_tasks(output):
        return False
    tasks = cast(list[Any], output["background_tasks"])
    statuses = [
        str(task.get("status") or "").strip().lower() for task in tasks if isinstance(task, dict)
    ]
    return bool(statuses) and all(status in {"failed", "killed", "stopped"} for status in statuses)


def _legacy_output_has_only_terminal_background_tasks(output: dict[str, Any]) -> bool:
    tasks = output.get("background_tasks")
    if not isinstance(tasks, list) or not tasks:
        return False
    statuses = [
        str(task.get("status") or "").strip().lower() for task in tasks if isinstance(task, dict)
    ]
    if not statuses or any(status not in _TERMINAL_BACKGROUND_TASK_STATUSES for status in statuses):
        return False
    if any(str(text).strip() for text in output.get("text_outputs", []) or []):
        return False
    for item in output.get("tool_uses", []) or []:
        if item in (None, "", {}, []):
            continue
        if not isinstance(item, dict):
            return False
        if str(item.get("tool") or "").strip().lower() not in _BACKGROUND_TASK_CONTROL_TOOLS:
            return False
    return output.get("result_message") in (None, "", {}, [])


def _agent_run_result_from_legacy(
    request: AgentRunRequest, legacy_result: LegacyAgentResult
) -> AgentRunResult:
    credential_refs = [request.credential_ref] if request.credential_ref else []
    events: list[AgentEvent] = [
        _agent_event(
            request,
            1,
            "agent_run_started",
            {
                "request": request.to_dict(),
                "runtime_ref": request.agent_runtime_ref,
            },
            credential_refs,
        )
    ]
    event_index = 2
    for text in (
        legacy_result.output.get("text_outputs", [])
        if isinstance(legacy_result.output, dict)
        else []
    ):
        events.append(
            _agent_event(
                request,
                event_index,
                "assistant_text",
                {"text": str(text)},
                credential_refs,
            )
        )
        event_index += 1
    tool_records: list[ToolCallRecord] = []
    raw_tool_uses = (
        legacy_result.output.get("tool_uses", []) if isinstance(legacy_result.output, dict) else []
    )
    if isinstance(raw_tool_uses, list):
        for offset, tool_use in enumerate(raw_tool_uses, start=1):
            if not isinstance(tool_use, dict):
                continue
            tool_name = str(tool_use.get("tool", ""))
            tool_record = ToolCallRecord(
                tool_call_id=f"{request.request_id}_tool_{offset:03d}",
                server_name="legacy_claude_sdk",
                tool_name=tool_name,
                started_at_ms=0,
                finished_at_ms=0,
                success=True,
                artifact_refs=[],
                failover_reason="none",
            )
            tool_records.append(tool_record)
            events.append(
                _agent_event(
                    request,
                    event_index,
                    "tool_use",
                    {
                        "tool_call_id": tool_record.tool_call_id,
                        "server_name": tool_record.server_name,
                        "tool_name": tool_name,
                        "input": tool_use.get("input", {}),
                    },
                    credential_refs,
                )
            )
            event_index += 1

    failover_reason = classify_runtime_failure(
        legacy_result.error,
        timed_out=legacy_result.timed_out,
    )
    usage = dict(legacy_result.usage)
    if not usage and isinstance(legacy_result.output, dict):
        usage = _normalized_usage(legacy_result.output.get("usage"))
    terminal_status = legacy_result.terminal_status or (
        "completed" if legacy_result.success else "failed"
    )
    failure_context = _runtime_failure_context(request, relay_used=False)
    events.append(
        _agent_event(
            request,
            event_index,
            "final_result",
            {
                "success": legacy_result.success,
                "duration": legacy_result.duration,
                "iteration_count": legacy_result.iteration_count,
                "error": legacy_result.error,
                "failover_reason": failover_reason,
                "failure_context": failure_context if failover_reason != "none" else {},
                "legacy_output": legacy_result.output,
                "usage": usage,
                "terminal_status": terminal_status,
                "timed_out": legacy_result.timed_out,
                "cancelled": legacy_result.cancelled,
            },
            credential_refs,
        )
    )
    return AgentRunResult(
        success=legacy_result.success,
        events=events,
        text_output_refs=[],
        tool_uses=tool_records,
        error=legacy_result.error,
        failover_reason=failover_reason,
        credential_ref=request.credential_ref,
        usage=usage,
        terminal_status=terminal_status,
        timed_out=legacy_result.timed_out,
        cancelled=legacy_result.cancelled,
    )


def _agent_event(
    request: AgentRunRequest,
    event_index: int,
    event_type: str,
    payload: dict[str, Any],
    credential_refs: list[Any],
) -> AgentEvent:
    safe_request_id = "".join(
        ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in request.request_id
    )
    return AgentEvent(
        event_id=f"{safe_request_id}_event_{event_index:03d}",
        run_id=request.run_id,
        agent_run_id=request.request_id,
        stage_id=request.stage_id,
        type=event_type,
        payload=payload,
        artifact_refs=[],
        credential_refs=credential_refs,
        timestamp_ms=0,
    )


def _classify_legacy_failure(error: str | None) -> str:
    return classify_runtime_failure(error)


def _runtime_failure_context(request: AgentRunRequest, *, relay_used: bool) -> dict[str, Any]:
    """Return structured, redacted context for runtime/provider failures."""
    base_url = _provider_base_url_hint(request)
    credential = request.credential_ref or request.model_call.credential_ref
    return {
        "runtime_ref": request.agent_runtime_ref,
        "provider_ref": request.model_call.provider_ref,
        "api_format": request.model_call.api_format,
        "model": request.model_call.model,
        "base_url": base_url,
        "relay_used": relay_used,
        "credential_mode": request.credential_mode,
        "credential_key_id": credential.key_id if credential else None,
    }


def _provider_base_url_hint(request: AgentRunRequest) -> str:
    """Infer the provider route without reading raw secrets."""
    raw = request.runtime_options.get("provider_base_url") if request.runtime_options else None
    if isinstance(raw, str) and raw:
        return raw
    return ""


def _load_claude_sdk() -> dict[str, Any]:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        HookMatcher,
        query,
    )

    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            ToolResultBlock,
            ToolUseBlock,
        )
    except ImportError:  # pragma: no cover - old SDK variants.
        AssistantMessage = ResultMessage = ToolResultBlock = ToolUseBlock = None  # type: ignore[assignment]
    return {
        "ClaudeAgentOptions": ClaudeAgentOptions,
        "HookMatcher": HookMatcher,
        "query": query,
        "AssistantMessage": AssistantMessage,
        "ResultMessage": ResultMessage,
        "ToolResultBlock": ToolResultBlock,
        "ToolUseBlock": ToolUseBlock,
    }


def _hook_value(hook_input: Any, key: str) -> Any:
    if isinstance(hook_input, dict):
        return hook_input.get(key)
    return getattr(hook_input, key, None)


def _is_instance(value: Any, class_key: str, sdk: dict[str, Any]) -> bool:
    cls = sdk.get(class_key)
    if cls is not None and isinstance(value, cls):
        return True
    return type(value).__name__ == class_key


class _TranscriptRuntime:
    def __init__(self, runtime_ref: str, transcript_kind: str) -> None:
        self.runtime_ref = runtime_ref
        self.transcript_kind = transcript_kind

    def execute_sync(self, request: AgentRunRequest) -> AgentRunResult:
        agent_run_id = request.request_id
        credential_refs = [request.credential_ref] if request.credential_ref else []
        legacy_output: dict[str, JSONValue] = {
            "text_outputs": [f"deterministic {self.transcript_kind} response"],
            "tool_uses": [],
            "runtime_ref": self.runtime_ref,
        }
        events = [
            self._event(
                request,
                agent_run_id,
                1,
                "agent_run_started",
                {"runtime_ref": self.runtime_ref},
                credential_refs,
            ),
            self._event(
                request,
                agent_run_id,
                2,
                "assistant_text",
                {
                    "transcript_kind": self.transcript_kind,
                    "text": f"deterministic {self.transcript_kind} response",
                    "cache_hash": request.cache_policy.frozen_prefix_hash,
                },
                credential_refs,
            ),
            self._event(
                request,
                agent_run_id,
                3,
                "final_result",
                {
                    "success": True,
                    "duration": 0.0,
                    "iteration_count": 0,
                    "error": None,
                    "failover_reason": "none",
                    "legacy_output": legacy_output,
                    "result_kind": "finding",
                    "model_profile_ref": request.model_profile_ref,
                    "credential_key_id": request.credential_ref.key_id
                    if request.credential_ref
                    else None,
                },
                credential_refs,
            ),
        ]
        return AgentRunResult(
            success=True,
            events=events,
            text_output_refs=[],
            tool_uses=[],
            error=None,
            failover_reason="none",
            credential_ref=request.credential_ref,
        )

    def _event(
        self,
        request: AgentRunRequest,
        agent_run_id: str,
        event_index: int,
        event_type: str,
        payload: dict[str, JSONValue],
        credential_refs: list[Any],
    ) -> AgentEvent:
        safe_request_id = "".join(
            ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in request.request_id
        )
        return AgentEvent(
            event_id=f"{safe_request_id}_event_{event_index:03d}",
            run_id=request.run_id,
            agent_run_id=agent_run_id,
            stage_id=request.stage_id,
            type=event_type,
            payload=payload,
            artifact_refs=[],
            credential_refs=credential_refs,
            timestamp_ms=0,
        )


def create_runtime() -> ClaudeSdkAgentRuntime:
    """Manifest entrypoint that constructs the Claude SDK runtime plugin."""
    return ClaudeSdkAgentRuntime()
