"""Runtime-neutral autonomous research agent loop."""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
import time
import traceback
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from praxist.config import (
    S3_BUCKET as _S3_BUCKET_DEFAULT,
)
from praxist.config import (
    S3_RESULTS_PREFIX as _S3_RESULTS_PREFIX_DEFAULT,
)
from praxist.core.cache import build_cache_policy
from praxist.core.cloudflare import (
    CLOUDFLARE_ACCOUNT_VAR,
    CLOUDFLARE_BASE_URL_VAR,
    CLOUDFLARE_KEY_VAR,
    CLOUDFLARE_PROVIDER_REF,
)
from praxist.core.credentials import CredentialRef, provider_name_from_ref
from praxist.core.modeling import default_model_profile
from praxist.core.prompt_layout import (
    build_legacy_jinja_prompt_layout,
    write_prompt_layout_files,
)
from praxist.core.protocol import (
    AgentRunRequest,
    CachePolicy,
    EnvPolicy,
    JSONValue,
    ModelCallSpec,
    ToolPermissionSet,
)
from praxist.core.redaction import dumps_redacted, redact_text
from praxist.core.run_config import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_FINDINGS_POLL_INTERVAL_SECONDS,
    DEFAULT_FULL_AUTO_MAX_RUNTIME_SECONDS,
    DEFAULT_LOCAL_FINDINGS_DIR,
    DEFAULT_LOGS_DIR,
    DEFAULT_WORKSPACE_ROOT,
    RunConfig,
)
from praxist.core.runtime_guard_policy import (
    RESOURCE_GUARD_ENV_KEYS,
    TRUSTED_PROJECT_EXTRA_ROOTS_ENV,
)
from praxist.core.runtimes import (
    AgentRuntimeExecutionContext,
    execute_runtime,
    is_provider_access_error,
    runtime_for_ref,
)
from praxist.core.tool_servers import tool_server_for_ref
from praxist.plugins.workflow_stages.research_loop.backend.event_wait import (
    wait_for_filesystem_event,
)
from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler_client import (
    generation_advice,
    get_supply_lease,
    register_idle_supply,
    release_supply_lease,
    resource_supply_signal_path,
    unregister_idle_supply,
)
from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
    DEFAULT_MAX_HANDOFF_BYTES,
    DEFAULT_MAX_LEDGER_FILE_BYTES,
    DEFAULT_MAX_MEMORY_FILE_BYTES,
    NoOpPeerSessionMemory,
    PeerMemoryConfig,
    PeerSessionMemory,
    read_bounded_file_under_root_no_follow,
)
from praxist.plugins.workflow_stages.research_loop.provider_env import (
    DEEPSEEK_CLAUDE_DEFAULT_EFFORT,
    DEEPSEEK_CLAUDE_DEFAULT_HAIKU_MODEL,
    DEEPSEEK_CLAUDE_DEFAULT_MODEL,
    DEEPSEEK_CLAUDE_SDK_BASE_URL,
    normalize_openrouter_base_url,
)

logger = logging.getLogger(__name__)

PRAXIST_RESEARCH_RUNTIME_SYSTEM_PROMPT = """You are the execution runtime for a Praxist autonomous research agent.
The user prompt is a complete, immediate research task. Treat project memories,
skills, shell reminders, and other local Claude Code context as lower priority
than the Praxist task. Do not wait for a human instruction, do not ask what to do,
and do not summarize the prompt. Begin work by using the available tools."""

BOOTSTRAP_RETRY_DIRECTIVE = """# Praxist Bootstrap Recovery

Your previous response waited for a human instruction instead of executing the
Praxist research task. Begin now. Do not ask what to do. Use tools immediately:
read your peer notebook, inspect shared findings, query the frontier, and then
continue the research workflow."""

_BOOTSTRAP_WAIT_PATTERNS = (
    "what would you like me to do",
    "let me know how you'd like",
    "let me know what you'd like",
    "you haven't asked me",
    "haven't asked me",
    "wait for your instruction",
    "waiting for your instruction",
    "please tell me what",
)

_CONTEXT_EFFICIENCY_MODE_ENV = "PRAXIST_CONTEXT_EFFICIENCY_MODE"
_CONTEXT_EFFICIENCY_INTERVAL_ENV = "PRAXIST_CONTEXT_EFFICIENCY_MIN_SESSION_INTERVAL_SECONDS"
_CONTEXT_EFFICIENCY_MODES = frozenset({"auto", "lossless", "off"})
_CHATGPT_CREDENTIAL_PREFIX = "openai_compatible:codex_sdk:chatgpt:"
_LOSSLESS_MAX_SHARED_FINDINGS = 48
_LOSSLESS_MAX_MEMORY_PROMPT_CHARS = 24_000
_LOSSLESS_CONTINUATION_DIRECTIVE = """# Lossless Continuation Navigation

This is a continuation session. The complete task contract remains above and
all canonical artifacts remain available. Start from the supplied peer memory,
handoff, unseen finding IDs, and exact artifact references. Do not repeat broad
project-tree scans or reread unchanged task documents merely to reconstruct
context already present here. Reopen the exact original artifact whenever its
details changed, are uncertain, or are needed for a decision. Prefer compact
Praxist tool views first, and follow `full_result_ref` when the complete archived
tool output is required. This changes navigation order only; it does not permit
discarding evidence or relying on a summary when the source is needed."""


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def resolve_prompt(
    base_template_path: Path,
    task_prompt_path: Path | None,
    generation_template_path: Path | None,
    output_path: Path,
    context: dict[str, Any],
) -> str:
    """Render legacy prompt text.

    Compatibility wrapper for older callers. New research-loop callers should
    use ``resolve_prompt_with_layout`` so PromptLayout V1 manifests are
    produced alongside the rendered prompt.
    """
    content, _ = resolve_prompt_with_layout(
        base_template_path=base_template_path,
        task_prompt_path=task_prompt_path,
        generation_template_path=generation_template_path,
        output_path=output_path,
        context=context,
    )
    return content


def resolve_prompt_with_layout(
    base_template_path: Path,
    task_prompt_path: Path | None,
    generation_template_path: Path | None,
    output_path: Path,
    context: dict[str, Any],
    *,
    layout_output_path: Path | None = None,
    rendered_prompt_ref: dict[str, Any] | None = None,
    prompt_id: str | None = None,
    extra_dynamic_blocks: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Render prompt text and write a PromptLayout V1 manifest."""
    run_id = os.environ.get("PRAXIST_RUN_ID", "legacy_direct")
    layout = build_legacy_jinja_prompt_layout(
        base_template_path=base_template_path,
        task_prompt_path=task_prompt_path,
        generation_template_path=generation_template_path,
        context=context,
        run_id=run_id,
        stage_id=os.environ.get("PRAXIST_STAGE_ID", "research_loop"),
        prompt_id=prompt_id or output_path.stem,
        agent_runtime_ref=os.environ.get("PRAXIST_AGENT_RUNTIME_REF", "agent_runtime:claude_sdk"),
        model_provider_ref=os.environ.get("PRAXIST_MODEL_PROVIDER_REF", ""),
        repo_root=Path(os.environ.get("PRAXIST_WORKSPACE_ROOT", os.getcwd())),
        extra_dynamic_blocks=extra_dynamic_blocks,
    )
    manifest_path = layout_output_path or output_path.with_name(output_path.stem + "_layout.json")
    manifest = write_prompt_layout_files(
        layout=layout,
        prompt_path=output_path,
        manifest_path=manifest_path,
        rendered_prompt_ref=rendered_prompt_ref,
    )
    return layout.prompt_text, manifest


# ---------------------------------------------------------------------------
# Stop condition
# ---------------------------------------------------------------------------


class StopReason(Enum):
    """Reason enum describing why an autonomous agent session stopped."""

    TIMEOUT = "timeout"  # per-peer safety cap (max_runtime_seconds) hit
    USER_INTERRUPT = "user_interrupt"
    # v2026-05-04 R2#2 fix: orchestrator-initiated drain via STOP_SIGNAL
    # sentinel. Distinct from TIMEOUT so post-mortem can tell "trigger
    # fired (healthy gen-end)" from "peer ran the full safety cap".
    SYNTHESIS_TRIGGER = "synthesis_trigger"
    SYNTHESIS_CLOSING = "synthesis_closing"
    RUNTIME_EMPTY = "runtime_empty"
    RUNTIME_FAILURE = "runtime_failure"


API_BILLING_RETRY_INTERVAL = 1200  # 20 minutes
_MAX_CONSECUTIVE_EMPTY_SESSIONS = 2
_EMPTY_SESSION_RETRY_SECONDS = 5.0
_MAX_CONSECUTIVE_RUNTIME_FAILURES = 2
_RUNTIME_FAILURE_RETRY_SECONDS = 5.0


class StopChecker:
    """Timeout-based stop checker.

    v2026-05-04: also supports a `stop_signal_path` sentinel file. When
    the orchestrator's synthesis trigger fires, it writes that file and
    peers detect it on their next `check()` call (between SDK turns).
    This is the event-driven gen-end mechanism that supersedes the old
    fixed `per_generation_hours` timer.
    """

    def __init__(
        self,
        max_runtime: float,
        stop_signal_path: Optional["Path"] = None,
    ):
        self.max_runtime = max_runtime
        self.start_time = time.time()
        self.consecutive_errors = 0
        self.stop_signal_path = stop_signal_path

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def reset_start_time(self, start_time: float | None = None) -> None:
        """Align this peer with the orchestrator's shared generation clock."""

        self.start_time = time.time() if start_time is None else float(start_time)

    def check(self) -> StopReason | None:
        # Synthesis-trigger sentinel takes precedence (orchestrator-initiated).
        # R2#2 fix: return distinct SYNTHESIS_TRIGGER variant so callers
        # can distinguish trigger-initiated drain from genuine timeout.
        if self.stop_signal_path is not None:
            try:
                if self.stop_signal_path.exists():
                    return StopReason.SYNTHESIS_TRIGGER
            except (OSError, ValueError):
                # FS hiccup — fall through to timeout check
                pass
        if self.elapsed_time >= self.max_runtime:
            return StopReason.TIMEOUT
        return None

    def record_success(self):
        self.consecutive_errors = 0

    def record_error(self):
        self.consecutive_errors += 1


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Legacy autonomous-agent result shape returned by BaseAgent callers."""

    success: bool
    output: dict[str, Any]
    duration: float
    iteration_count: int
    error: str | None = None
    usage: dict[str, float] | None = None
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class BaseAgent:
    """Executes research tasks through the selected AgentRuntime plugin."""

    def __init__(
        self,
        name: str,
        allowed_tools: list[str],
        workspace: Path,
        mcp_servers: dict[str, Any],
        model: str = "",
        permission_mode: str = "acceptEdits",
        cli_path: str | None = None,
        message_callback: Callable | None = None,
        system_prompt: str | None = None,
        prompt_layout_manifest: dict[str, Any] | None = None,
        plugin_registry: Any | None = None,
        # R4-C2 fix: optional stop-check callable invoked between every
        # SDK message inside execute(). When it returns truthy, the
        # message loop exits early. This lets the synthesis trigger
        # actually interrupt a long-running session (for example, wait_for_file
        # rounds can take 30+ min in a single execute call).
        stop_check_fn: Callable[[], bool] | None = None,
        # Legacy compatibility switch. The provider-neutral reasoning_effort
        # policy below takes precedence when explicitly configured.
        premium_mode: bool = False,
        # Issue #75 batch 1: optional collected run-level configuration.
        # When provided, ``_build_agent_run_request`` and ``execute``
        # source PRAXIST_RUN_ID / PRAXIST_STAGE_ID / PRAXIST_ROLE_REF /
        # PRAXIST_AGENT_RUNTIME_REF / PRAXIST_BUDGET_GRANT_ID / PRAXIST_MODEL_PROFILE_REF
        # from this config instead of os.environ. ``None`` falls back to a
        # per-call ``RunConfig.from_environ(os.environ)`` so legacy callers
        # that mutate env after construction (a common test pattern) still
        # observe the latest values. New callers should build the config at
        # the CLI boundary and inject it explicitly — see
        # ``docs/concepts/config_discipline.md``.
        run_config: RunConfig | None = None,
        runtime_env_overrides: dict[str, str] | None = None,
        runtime_sandbox_intent: dict[str, str] | None = None,
        runtime_timeout_seconds: int | None = None,
        runtime_output_schema: dict[str, JSONValue] | None = None,
        require_no_shell_runtime: bool = False,
        require_read_only_runtime: bool = False,
        request_id: str | None = None,
        role_skill_sha256: str | None = None,
        reasoning_effort: str = "max",
    ):
        self.name = name
        self.allowed_tools = allowed_tools
        self.workspace = workspace
        self.mcp_servers = mcp_servers
        self.model = model or DEFAULT_AGENT_MODEL
        self.permission_mode = permission_mode
        self.cli_path = cli_path
        self.message_callback = message_callback
        self.system_prompt = (
            system_prompt if system_prompt is not None else PRAXIST_RESEARCH_RUNTIME_SYSTEM_PROMPT
        )
        self.prompt_layout_manifest = prompt_layout_manifest or None
        self.plugin_registry = plugin_registry
        self.stop_check_fn = stop_check_fn
        self.premium_mode = premium_mode
        self.reasoning_effort = reasoning_effort
        self._run_config_override: RunConfig | None = run_config
        self.runtime_env_overrides = dict(runtime_env_overrides or {})
        self.runtime_sandbox_intent = dict(runtime_sandbox_intent or {})
        self.runtime_timeout_seconds = runtime_timeout_seconds
        self.runtime_output_schema = dict(runtime_output_schema or {}) or None
        self.require_no_shell_runtime = bool(require_no_shell_runtime)
        self.require_read_only_runtime = bool(require_read_only_runtime)
        # A caller may reserve the identity of the next invocation so it can
        # prepare an invocation-owned output path. Consume it once; reusing a
        # BaseAgent still creates a fresh request boundary.
        self._next_request_id = str(request_id).strip() if request_id else None
        self.role_skill_sha256 = role_skill_sha256

    def _run_config(self) -> RunConfig:
        """Return the explicit ``RunConfig`` or build one from the live env.

        Centralizes the agent.py env-read surface — every per-call site
        that needs PRAXIST_RUN_ID / PRAXIST_STAGE_ID / PRAXIST_ROLE_REF /
        PRAXIST_AGENT_RUNTIME_REF / PRAXIST_BUDGET_GRANT_ID / PRAXIST_MODEL_PROFILE_REF
        reads through this method instead of touching ``os.environ``
        directly (issue #75 batch 1).
        """
        if self._run_config_override is not None:
            return self._run_config_override
        return RunConfig.from_environ(os.environ)

    async def execute(self, task: str) -> AgentResult:
        """Execute the agent task via the configured AgentRuntime adapter."""
        env = _scoped_legacy_provider_env()
        env.update({key: value for key, value in self.runtime_env_overrides.items() if value})
        request = self._build_agent_run_request(task, env)
        trajectory = _legacy_trajectory_writer()
        started_event_id = None
        model_call = _legacy_model_call_payload(self.model, run_config=self._run_config())
        if trajectory is not None:
            started = trajectory.emit(
                "agent.run_started",
                scope={"stage_id": "research_loop", "agent_name": self.name},
                actor={"type": "agent_runtime", "id": request.agent_runtime_ref},
                payload={
                    "agent_name": self.name,
                    "agent_runtime_ref": request.agent_runtime_ref,
                    "request": request.to_dict(),
                    "model": self.model,
                    "model_call": model_call,
                    "budget_grant_id": self._run_config().budget_grant_id,
                },
            )
            started_event_id = started.get("event_id")

        runtime = runtime_for_ref(request.agent_runtime_ref, registry=self.plugin_registry)
        has_async_execute = callable(getattr(runtime, "execute", None))
        runtime_result = await execute_runtime(
            runtime,
            request,
            AgentRuntimeExecutionContext(
                tool_servers=self.mcp_servers,
                message_callback=self.message_callback,
                stop_requested=self.stop_check_fn,
                env=env,
            ),
        )
        if not has_async_execute and self.message_callback is not None:
            for event in runtime_result.events:
                try:
                    self.message_callback(event)
                except Exception:
                    logger.exception(
                        "message_callback raised replaying event type=%s",
                        getattr(event, "type", "<unknown>"),
                    )
        final_payload = _runtime_final_payload(runtime_result)
        legacy_output = final_payload.get("legacy_output")
        if not isinstance(legacy_output, dict):
            legacy_output = {}
        duration = _float_payload(final_payload.get("duration"))
        iteration_count = _int_payload(final_payload.get("iteration_count"))
        if trajectory is not None:
            trajectory.emit(
                "agent.run_finished",
                scope={"stage_id": "research_loop", "agent_name": self.name},
                actor={"type": "agent_runtime", "id": request.agent_runtime_ref},
                payload={
                    "agent_name": self.name,
                    "success": runtime_result.success,
                    "duration": duration,
                    "iteration_count": iteration_count,
                    "agent_runtime_ref": request.agent_runtime_ref,
                    "request": request.to_dict(),
                    "model_call": model_call,
                    "budget_grant_id": self._run_config().budget_grant_id,
                    "output_summary": _legacy_output_summary(legacy_output),
                    "runtime_event_types": [event.type for event in runtime_result.events],
                },
                parent_event_ids=[started_event_id] if started_event_id else [],
            )
        return AgentResult(
            success=runtime_result.success,
            output=legacy_output,
            duration=duration,
            iteration_count=iteration_count,
            error=runtime_result.error,
            usage=dict(runtime_result.usage),
            request_id=request.request_id,
        )

    def _build_agent_run_request(self, task: str, env: dict[str, str]) -> AgentRunRequest:
        cfg = self._run_config()
        provider_ref = _legacy_model_provider_ref(self.model, run_config=cfg)
        credential_ref = _legacy_credential_ref(provider_ref, run_config=cfg)
        model_profile_ref = cfg.model_profile_ref or "cheap_peer"
        profile = default_model_profile(
            provider_ref, profile_id=model_profile_ref, model=self.model
        )
        model_call = profile_to_model_call(profile, credential_ref)
        task_hash = hashlib.sha256(task.encode("utf-8")).hexdigest()
        task_hash_ref = f"sha256:{task_hash}"
        cache_policy = self._cache_policy_for_request()
        scoped_refs = [credential_ref] if credential_ref else []
        layout_overlay = _prompt_layout_runtime_overlay(
            self.prompt_layout_manifest,
            actual_prompt_hash=task_hash_ref,
        )
        prompt_ref = {
            "kind": "legacy_inline_prompt",
            "sha256": task_hash_ref,
            "text": task,
        }
        if self.prompt_layout_manifest:
            prompt_ref = {
                "kind": "prompt_layout_v1",
                "sha256": task_hash_ref,
                "text": task,
                "layout_hash": self.prompt_layout_manifest.get("layout_hash"),
                "frozen_prefix_hash": self.prompt_layout_manifest.get("frozen_prefix_hash"),
                "dynamic_payload_hash": self.prompt_layout_manifest.get("dynamic_payload_hash"),
            }
            if layout_overlay:
                prompt_ref["runtime_overlay"] = layout_overlay
        prompt_layout_summary = _prompt_layout_runtime_summary(self.prompt_layout_manifest)
        if layout_overlay:
            prompt_layout_summary["runtime_overlay"] = layout_overlay
        runtime_options: dict[str, JSONValue] = {
            "permission_mode": self.permission_mode,
            "premium_mode": self.premium_mode,
            "reasoning_effort": self.reasoning_effort,
            "cli_path": self.cli_path or "",
            "legacy_base_agent_compat": True,
            "system_prompt": self.system_prompt or "",
            "prompt_layout": prompt_layout_summary,
            "runtime_env_overrides": {
                str(key): str(value) for key, value in self.runtime_env_overrides.items()
            },
            "provider_base_url": env.get("ANTHROPIC_BASE_URL") or env.get("OPENAI_BASE_URL") or "",
            "run_dir": str(cfg.run_dir) if cfg.run_dir else "",
            "codex_bin": cfg.codex_bin,
        }
        if self.runtime_sandbox_intent:
            runtime_options["sandbox_intent"] = {
                str(key): str(value) for key, value in self.runtime_sandbox_intent.items()
            }
        if self.runtime_output_schema is not None:
            runtime_options["output_schema"] = self.runtime_output_schema
        if self.require_no_shell_runtime:
            runtime_options["require_no_shell_runtime"] = True
        if self.require_read_only_runtime:
            runtime_options["require_read_only_runtime"] = True

        request_id = self._next_request_id or create_agent_request_id(self.name)
        self._next_request_id = None
        return AgentRunRequest(
            request_id=request_id,
            run_id=cfg.run_id or "legacy_direct",
            stage_id=cfg.stage_id or "research_loop",
            role_ref=cfg.role_ref,
            agent_runtime_ref=cfg.agent_runtime_ref or "agent_runtime:claude_sdk",
            prompt_ref=prompt_ref,
            system_prompt_ref=(
                {
                    "kind": "legacy_inline_system_prompt",
                    "sha256": "sha256:"
                    + hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest(),
                }
                if self.system_prompt
                else None
            ),
            cwd=str(self.workspace),
            model_profile_ref=model_profile_ref,
            model_call=model_call,
            tool_permissions=ToolPermissionSet(
                mode="allow_list", allowed_tools=list(self.allowed_tools)
            ),
            tool_servers=self._runtime_tool_server_descriptors(),
            env_policy=EnvPolicy(
                redaction_required=True,
                exposed_env_keys=sorted(env),
                scoped_credential_refs=scoped_refs,
            ),
            credential_ref=credential_ref,
            credential_mode=os.environ.get("PRAXIST_CREDENTIAL_MODE", "single"),
            budget_grant_id=cfg.budget_grant_id or None,
            artifact_scope="run",
            timeout_seconds=(
                int(self.runtime_timeout_seconds)
                if self.runtime_timeout_seconds is not None
                else _int_env("PRAXIST_AGENT_TIMEOUT_SECONDS", 0)
            ),
            cache_policy=cache_policy,
            runtime_options=runtime_options,
            role_skill_sha256=self.role_skill_sha256,
        )

    def _runtime_tool_server_descriptors(self) -> list[dict[str, Any]]:
        """Describe connected tool servers without exposing provider objects."""

        descriptors: dict[str, dict[str, Any]] = {
            name: {"server_name": name, "transport": "legacy_inprocess"}
            for name in sorted(self.mcp_servers)
        }
        if self.plugin_registry is None:
            return list(descriptors.values())
        for selected in self.plugin_registry.list("tool_server"):
            ref = f"tool_server:{selected.metadata.name}"
            try:
                spec = tool_server_for_ref(ref, self.plugin_registry)
            except (KeyError, TypeError, ValueError):
                continue
            entry = descriptors.get(spec.server_name)
            if entry is None:
                continue
            entry.update(
                {
                    "ref": spec.plugin_ref,
                    "factory": spec.factory or "",
                    "tool_names": list(spec.tool_names),
                    "requires_run_dir": spec.requires_run_dir,
                }
            )
        return list(descriptors.values())

    def _cache_policy_for_request(self):
        if self.prompt_layout_manifest:
            return CachePolicy(
                mode=str(self.prompt_layout_manifest.get("cache_mode") or "runtime_auto_cache"),
                frozen_prefix_hash=self.prompt_layout_manifest.get("frozen_prefix_hash"),
                cache_breakpoints=list(
                    self.prompt_layout_manifest.get("cache_breakpoints") or ["frozen_prefix"]
                ),
                runtime_cache_strategy=self.prompt_layout_manifest.get("runtime_cache_strategy"),
                provider_cache_strategy=self.prompt_layout_manifest.get("provider_cache_strategy"),
            )
        return build_cache_policy(
            frozen_prefix_parts={
                "agent": self.name,
                "allowed_tools": list(self.allowed_tools),
                "model": self.model,
                "system_prompt": self.system_prompt or "",
            }
        )


def _scoped_legacy_provider_env() -> dict[str, str]:
    provider_ref = os.environ.get("PRAXIST_MODEL_PROVIDER_REF", "")
    if provider_ref == "model_provider:openrouter":
        allowed = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "OPENROUTER_API_KEY")
    elif provider_ref == "model_provider:anthropic_messages":
        allowed = ("ANTHROPIC_API_KEY",)
    elif provider_ref == "model_provider:openai_compatible":
        allowed = ("OPENAI_API_KEY",)
    elif provider_ref == "model_provider:deepseek_alias":
        allowed = (
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL",
            "CLAUDE_CODE_EFFORT_LEVEL",
            "DEEPSEEK_API_KEY",
        )
    elif provider_ref == CLOUDFLARE_PROVIDER_REF:
        # Workers AI reaches the Codex runtime through the relay, which reads
        # the bearer from CLOUDFLARE_API_KEY/OPENAI_API_KEY and interpolates
        # the account-scoped upstream from CLOUDFLARE_ACCOUNT_ID.
        allowed = (
            CLOUDFLARE_KEY_VAR,
            "OPENAI_API_KEY",
            CLOUDFLARE_ACCOUNT_VAR,
            CLOUDFLARE_BASE_URL_VAR,
        )
    elif provider_ref == "model_provider:fake_provider":
        allowed = ()
    else:
        # Legacy direct callers predate provider selection. Preserve their
        # behavior outside the core-plugin launch path.
        allowed = (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_AUTH_TOKEN",
            "OPENROUTER_API_KEY",
        )
    env: dict[str, str] = {}
    for var in allowed:
        val = os.environ.get(var, "")
        if provider_ref == "model_provider:deepseek_alias":
            if var == "ANTHROPIC_BASE_URL":
                val = val or DEEPSEEK_CLAUDE_SDK_BASE_URL
            elif var == "ANTHROPIC_AUTH_TOKEN":
                val = os.environ.get("DEEPSEEK_API_KEY", "") or val
            elif var in (
                "ANTHROPIC_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
            ):
                val = val or DEEPSEEK_CLAUDE_DEFAULT_MODEL
            elif var in ("ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"):
                val = val or DEEPSEEK_CLAUDE_DEFAULT_HAIKU_MODEL
            elif var == "CLAUDE_CODE_EFFORT_LEVEL":
                val = val or DEEPSEEK_CLAUDE_DEFAULT_EFFORT
        if val:
            if provider_ref == "model_provider:openrouter" and var == "ANTHROPIC_BASE_URL":
                val = normalize_openrouter_base_url(val)
            env[var] = val
    if provider_ref == CLOUDFLARE_PROVIDER_REF:
        cloudflare_key = os.environ.get(CLOUDFLARE_KEY_VAR, "")
        if cloudflare_key:
            env[CLOUDFLARE_KEY_VAR] = cloudflare_key
            env["OPENAI_API_KEY"] = cloudflare_key
    if provider_ref == "model_provider:deepseek_alias":
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if deepseek_key:
            env["DEEPSEEK_API_KEY"] = deepseek_key
            env["ANTHROPIC_AUTH_TOKEN"] = deepseek_key
            env["ANTHROPIC_BASE_URL"] = (
                os.environ.get("DEEPSEEK_ANTHROPIC_BASE_URL") or DEEPSEEK_CLAUDE_SDK_BASE_URL
            )
    for var in (*_legacy_runtime_env_keys(), *_task_runtime_extra_env_keys()):
        val = os.environ.get(var, "")
        if val:
            env[var] = val
    return env


def _legacy_runtime_env_keys() -> tuple[str, ...]:
    """Return non-secret operational env allowed through the legacy SDK bridge."""

    return (
        "PRAXIST_RUN_DIR",
        "PRAXIST_RUN_ID",
        "PRAXIST_STAGE_ID",
        "PRAXIST_EVALUATION_ENTRYPOINT",
        "PRAXIST_EVALUATION_ENTRYPOINT_PATH",
        "PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT",
        "PRAXIST_EXPERIMENT_SCHEDULER_CONFIG",
        "PEER_ID",
        "GENERATION_ID",
        "PRAXIST_AGENT_RUNTIME_REF",
        "PRAXIST_WORKSPACE_ROOT",
        "PRAXIST_TASK_PROJECT_PATH",
        "PRAXIST_RUNNER_PYTHON",
        TRUSTED_PROJECT_EXTRA_ROOTS_ENV,
        "PRAXIST_TASK_PYTHON",
        "PRAXIST_TASK_VENV",
        "PRAXIST_TASK_WRITABLE_ROOTS",
        "PRAXIST_TASK_SHELL_PREFIX",
        "PRAXIST_TASK_RUNTIME_ENV_KEYS",
        "PRAXIST_MODEL_PROVIDER_REF",
        "PRAXIST_MODEL_CREDENTIAL_KEY_ID",
        "PRAXIST_BUDGET_GRANT_ID",
        "PRAXIST_BUDGET_REQUEST_ID",
        "PRAXIST_BASELINE_CACHE_DIR",
        "PRAXIST_LAUNCH_GUARD_ENABLED",
        "LOCAL_STORE_DIR",
        "LOCAL_FINDINGS_DIR",
        "AUTO_RESEARCH_RUN_DIR",
        "FRONTIER_DIR",
        "PROTECTED_PIDS_DIR",
        "PRAXIST_MAX_PARALLEL_RUNS_PER_PEER",
        "GPU_GOVERNOR_DIR",
        "GPU_GOVERNOR_MAX_PER_GPU",
        *RESOURCE_GUARD_ENV_KEYS,
        "GPU_GOVERNOR_POINTER_FILE",
        "LOGS_DIR",
        "PRIMARY_METRIC",
        "METRIC_DIRECTION",
        "ANCHOR_METRICS",
        "REQUIRES_TIER",
        "PRAXIST_PROTECTED_CHILD_PATHS",
        "PRAXIST_DATA_DIR",
        "PRAXIST_DATASETS_DIR",
        "PRAXIST_DATA_ROOT",
        "VIRTUAL_ENV",
        "PATH",
        "PYTHONPATH",
    )


def _task_runtime_extra_env_keys() -> tuple[str, ...]:
    raw = os.environ.get("PRAXIST_TASK_RUNTIME_ENV_KEYS", "")
    keys: list[str] = []
    for item in raw.split(","):
        key = item.strip()
        if key and key.replace("_", "").isalnum() and not key[0].isdigit():
            keys.append(key)
    return tuple(dict.fromkeys(keys))


def profile_to_model_call(profile, credential_ref: CredentialRef | None) -> ModelCallSpec:
    """Convert a ModelProfile into the ModelCallSpec used by runtime adapters."""
    return ModelCallSpec(
        profile_id=profile.profile_id,
        provider_ref=profile.provider_ref,
        api_format=profile.api_format,
        model=profile.model,
        parameters=dict(profile.default_parameters),
        credential_ref=credential_ref,
    )


def _legacy_model_provider_ref(model: str, *, run_config: RunConfig | None = None) -> str:
    """Resolve a model provider ref for ``model``.

    Issue #75 batch 2: prefer an explicit ``run_config.model_provider_ref``
    when one was supplied; otherwise fall back to the historical env-read
    path so existing tests / out-of-band callers that mutate
    ``PRAXIST_MODEL_PROVIDER_REF`` between construction and call continue to
    work. The legacy ``"/" in model`` heuristic is preserved — its
    removal is a behavioral change that belongs in a separate PR.
    """
    if run_config is not None and run_config.model_provider_ref:
        return run_config.model_provider_ref
    provider_ref = os.environ.get("PRAXIST_MODEL_PROVIDER_REF", "")
    if provider_ref:
        return provider_ref
    if "/" in model:
        return "model_provider:openrouter"
    if model.startswith("deepseek"):
        return "model_provider:deepseek_alias"
    if model.startswith(("gpt-", "o")):
        return "model_provider:openai_compatible"
    return "model_provider:anthropic_messages"


def _legacy_credential_ref(
    provider_ref: str, *, run_config: RunConfig | None = None
) -> CredentialRef | None:
    """Build a ``CredentialRef`` for the active model-provider credential.

    Issue #75 batch 2: source the key id from ``run_config`` when
    supplied; otherwise fall back to ``PRAXIST_MODEL_CREDENTIAL_KEY_ID``.
    Returns ``None`` when no credential is selected (credential-less
    runs such as resolve-only smoke tests).
    """
    if run_config is not None:
        key_id = run_config.model_credential_key_id
    else:
        key_id = os.environ.get("PRAXIST_MODEL_CREDENTIAL_KEY_ID", "")
    if not key_id:
        return None
    return CredentialRef(
        scope="model_provider",
        provider=provider_name_from_ref(provider_ref),
        target_ref=provider_ref,
        key_id=key_id,
        source="startup",
    )


def _legacy_model_call_payload(
    model: str, *, run_config: RunConfig | None = None
) -> dict[str, str]:
    """Trajectory bookend payload describing the model the agent ran with."""
    if run_config is not None:
        credential_key_id = run_config.model_credential_key_id
    else:
        credential_key_id = os.environ.get("PRAXIST_MODEL_CREDENTIAL_KEY_ID", "")
    return {
        "provider_ref": _legacy_model_provider_ref(model, run_config=run_config),
        "model": model,
        "credential_ref": credential_key_id,
    }


def _legacy_output_summary(output: dict[str, Any]) -> dict[str, Any]:
    tool_uses = output.get("tool_uses") if isinstance(output, dict) else None
    if not isinstance(tool_uses, list):
        tool_uses = []
    return {
        "tool_uses": tool_uses[:50],
    }


def _prompt_layout_runtime_summary(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {}
    return {
        "schema_version": manifest.get("schema_version"),
        "layout_hash": manifest.get("layout_hash"),
        "frozen_prefix_hash": manifest.get("frozen_prefix_hash"),
        "dynamic_payload_hash": manifest.get("dynamic_payload_hash"),
        "cache_mode": manifest.get("cache_mode"),
        "runtime_cache_strategy": manifest.get("runtime_cache_strategy"),
        "provider_cache_strategy": manifest.get("provider_cache_strategy"),
        "cache_usage_status": manifest.get("cache_usage_status"),
    }


def _prompt_layout_runtime_overlay(
    manifest: dict[str, Any] | None,
    *,
    actual_prompt_hash: str,
) -> dict[str, Any]:
    if not manifest:
        return {}
    base_hash = manifest.get("rendered_prompt_hash")
    if not base_hash or base_hash == actual_prompt_hash:
        return {}
    return {
        "overlay_kind": "peer_local_memory_or_bootstrap_runtime_block",
        "base_rendered_prompt_hash": base_hash,
        "runtime_composed_prompt_hash": actual_prompt_hash,
        "base_layout_hash": manifest.get("layout_hash"),
        "note": (
            "PromptLayout hashes describe the rendered base peer prompt; "
            "the runtime prompt includes an additional bounded session-local overlay."
        ),
    }


def _with_bootstrap_retry_directive(task_prompt: str) -> str:
    return task_prompt.rstrip() + "\n\n" + BOOTSTRAP_RETRY_DIRECTIVE.strip() + "\n"


def _legacy_trajectory_writer():
    run_dir = os.environ.get("PRAXIST_RUN_DIR")
    if not run_dir:
        return None
    try:
        from praxist.core.trajectory import TrajectoryWriter

        return TrajectoryWriter(Path(run_dir), os.environ.get("PRAXIST_RUN_ID", Path(run_dir).name))
    except Exception:
        return None


def _runtime_final_payload(runtime_result) -> dict[str, Any]:
    for event in reversed(runtime_result.events):
        if event.type == "final_result":
            return dict(event.payload)
    return {}


def create_agent_request_id(agent_name: str) -> str:
    """Return a fresh filesystem-safe identity for one agent invocation."""

    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in agent_name)
    return f"legacy_{safe}_{uuid.uuid4().hex[:8]}"


def _float_payload(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_payload(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _merge_numeric_usage(
    current: dict[str, float],
    observed: dict[str, float],
) -> dict[str, float]:
    """Add normalized runtime usage counters without inventing missing values."""

    merged = dict(current)
    for key, value in observed.items():
        if isinstance(value, (int, float)):
            merged[str(key)] = merged.get(str(key), 0.0) + float(value)
    return merged


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _lossless_context_efficiency_enabled() -> bool:
    """Select lossless session batching without changing direct DeepSeek runs."""

    provider_ref = os.environ.get("PRAXIST_MODEL_PROVIDER_REF", "").strip()
    if provider_ref == "model_provider:deepseek_alias":
        return False
    mode = os.environ.get(_CONTEXT_EFFICIENCY_MODE_ENV, "auto").strip().lower()
    if mode not in _CONTEXT_EFFICIENCY_MODES:
        mode = "auto"
    if mode == "off":
        return False
    if mode == "lossless":
        return True
    if provider_ref == "model_provider:openrouter":
        return True
    return bool(
        os.environ.get("PRAXIST_AGENT_RUNTIME_REF", "").strip() == "agent_runtime:codex_sdk"
        and provider_ref == "model_provider:openai_compatible"
        and os.environ.get("PRAXIST_MODEL_CREDENTIAL_KEY_ID", "")
        .strip()
        .startswith(_CHATGPT_CREDENTIAL_PREFIX)
    )


def _resolve_peer_memory_dirs(
    logs_dir: Path,
    generation_id: int,
    *,
    prefer_env_run_dir: bool = False,
) -> tuple[Path, Path]:
    """Resolve run/gen roots for peer-local memory across legacy log layouts."""

    generation_names = {f"gen_{generation_id}", f"gen{generation_id}"}
    if logs_dir.parent.name in generation_names:
        return logs_dir.parent.parent, logs_dir.parent
    env_run_dir = os.environ.get("PRAXIST_RUN_DIR")
    if prefer_env_run_dir and env_run_dir:
        run_dir = Path(env_run_dir)
        return run_dir, run_dir / f"gen_{generation_id}"
    run_dir = logs_dir.parent
    return run_dir, run_dir / f"gen_{generation_id}"


def _peer_memory_upload_limit(path: Path) -> int | None:
    """Return the upload cap for known peer-memory artifacts, or None to skip."""

    name = path.name
    if name == "experiment_ledger.jsonl":
        return DEFAULT_MAX_LEDGER_FILE_BYTES
    if name in {"session_handoff.md", "session_auto_handoff.md"}:
        return DEFAULT_MAX_HANDOFF_BYTES
    if name in {
        "peer_state.yaml",
        "seen_shared_findings.json",
        "memory_prompt.md",
        "session_prompt_manifest.json",
    }:
        return DEFAULT_MAX_MEMORY_FILE_BYTES
    if name.startswith("memory_prompt_") and name.endswith(".md"):
        return DEFAULT_MAX_MEMORY_FILE_BYTES
    if name.startswith("session_prompt_manifest_") and name.endswith(".json"):
        return DEFAULT_MAX_MEMORY_FILE_BYTES
    return None


# ---------------------------------------------------------------------------
# Autonomous agent loop (single peer, multi-session)
# ---------------------------------------------------------------------------


class AutonomousAgentLoop:
    """
    Agent loop for a single peer within a generation cohort.

    The first iteration starts immediately. Later sessions are event-driven:
    a peer waits for new shared findings or the STOP_SIGNAL before sending
    another full prompt to the agent runtime. Lower-level graph/render/result
    files are intentionally not session wakeups; long experiment completion is
    handled inside a running session by tools such as wait_for_file.
    """

    def __init__(
        self,
        peer_id: str,
        generation_id: int,
        task_prompt: str,
        workspace: Path | None = None,
        max_runtime_seconds: int | None = None,
        logs_dir: Path | None = None,
        findings_dir: Path | None = None,
        s3_bucket: str | None = None,
        model: str = "",
        local_mode: bool = False,
        mcp_servers: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        # v2026-05-04: orchestrator can supply a sentinel-file path that
        # the StopChecker watches; when present, this peer exits gracefully
        # at the next check (used by the synthesis trigger to coordinate
        # event-driven gen termination).
        stop_signal_path: Path | None = None,
        # Session-boundary drain signal: finish the current SDK session and
        # avoid opening another one while the trigger waits for protected
        # training/evaluation subprocesses to complete. This intentionally does
        # not feed into StopChecker/BaseAgent, so in-flight work is not
        # interrupted by CLOSING_SIGNAL.
        closing_signal_path: Path | None = None,
        # Legacy compatibility switch passed to BaseAgent.
        premium_mode: bool = False,
        prompt_layout_manifest: dict[str, Any] | None = None,
        plugin_registry: Any | None = None,
        peer_memory_config: PeerMemoryConfig | None = None,
        role_ref: str | None = None,
        role_skill_sha256: str | None = None,
        reasoning_effort: str = "max",
    ):
        self.peer_id = peer_id
        self.generation_id = generation_id
        self.task_prompt = task_prompt
        self.run_id = os.getenv("RUN_ID") or str(uuid.uuid4())
        self.premium_mode = premium_mode
        self.reasoning_effort = reasoning_effort
        self.prompt_layout_manifest = prompt_layout_manifest or None
        self.plugin_registry = plugin_registry
        self.role_ref = role_ref
        self.role_skill_sha256 = role_skill_sha256

        self.workspace = Path(workspace) if workspace else Path(DEFAULT_WORKSPACE_ROOT)
        self.logs_dir = Path(logs_dir) if logs_dir else Path(DEFAULT_LOGS_DIR)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.findings_path = self.logs_dir.parent / "findings.json"

        self.max_runtime_seconds = max_runtime_seconds or DEFAULT_FULL_AUTO_MAX_RUNTIME_SECONDS
        # #75 batch 8a: env reads moved to the BaseAgent boundary. The
        # config re-export is the final fallback for tests that patch
        # ``config.S3_BUCKET`` / ``config.S3_RESULTS_PREFIX`` directly.
        self.s3_bucket = s3_bucket or os.environ.get("S3_BUCKET") or _S3_BUCKET_DEFAULT
        self.s3_prefix = (
            f"{os.environ.get('S3_RESULTS_PREFIX') or _S3_RESULTS_PREFIX_DEFAULT}"
            f"{self.peer_id}/{self.run_id}/"
        )
        self.model = model or DEFAULT_AGENT_MODEL
        self.local_mode = local_mode
        self.mcp_servers = mcp_servers or {}
        self.allowed_tools = allowed_tools or [
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Glob",
            "Grep",
            "WebSearch",
            "WebFetch",
        ]

        self.stop_signal_path = Path(stop_signal_path) if stop_signal_path else None
        self.closing_signal_path = Path(closing_signal_path) if closing_signal_path else None
        self.stop_checker = StopChecker(
            max_runtime=self.max_runtime_seconds,
            stop_signal_path=self.stop_signal_path,
        )

        # Findings sync:
        #   Server mode: each peer is a separate RunPod container with its own
        #   filesystem — it needs a per-peer FindingsSync to pull sibling-peer
        #   findings from the HTTP server into the local findings dir.
        #   Local mode: all peers share one process AND one shared_findings
        #   directory AND one SQLite file. The orchestrator (GenerationLoop)
        #   already runs a single FindingsSync daemon against that shared
        #   state, so a per-peer daemon would be 5–6× redundant work on the
        #   same files every 60 s and contend the same SQLite _DB_LOCK. Skip.
        self._findings_dir = (
            Path(findings_dir) if findings_dir else Path(DEFAULT_LOCAL_FINDINGS_DIR)
        )
        self.run_dir, self.gen_dir = _resolve_peer_memory_dirs(
            self.logs_dir,
            self.generation_id,
            prefer_env_run_dir=logs_dir is None,
        )
        self.resource_supply_signal_path = resource_supply_signal_path(self.gen_dir, self.peer_id)
        self._seen_resource_supply_signal_id = ""
        self._active_resource_supply_lease_id = ""
        self.lossless_context_efficiency = _lossless_context_efficiency_enabled()
        self.context_efficiency_min_session_interval_seconds = (
            max(0, _int_env(_CONTEXT_EFFICIENCY_INTERVAL_ENV, 300))
            if self.lossless_context_efficiency
            else 0
        )
        memory_config = peer_memory_config
        if self.lossless_context_efficiency:
            if memory_config is None:
                memory_config = PeerMemoryConfig(
                    max_shared_findings=_LOSSLESS_MAX_SHARED_FINDINGS,
                    max_prompt_chars=_LOSSLESS_MAX_MEMORY_PROMPT_CHARS,
                    track_finding_content_versions=True,
                )
            else:
                memory_config = replace(
                    memory_config,
                    track_finding_content_versions=True,
                )
        self.peer_memory: PeerSessionMemory | NoOpPeerSessionMemory
        try:
            self.peer_memory = PeerSessionMemory(
                run_dir=self.run_dir,
                gen_dir=self.gen_dir,
                peer_id=self.peer_id,
                generation_id=self.generation_id,
                findings_dir=self._findings_dir,
                config=memory_config or PeerMemoryConfig(),
            )
        except Exception as exc:
            logger.warning(
                "[%s] peer-local memory initialization failed; continuing without memory: %s",
                self.peer_id,
                exc,
            )
            self.peer_memory = NoOpPeerSessionMemory()
        self.findings_sync = None
        if not local_mode:
            try:
                from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync import (
                    FindingsSync,
                )

                self.findings_sync = FindingsSync(
                    findings_dir=self._findings_dir,
                    poll_interval=DEFAULT_FINDINGS_POLL_INTERVAL_SECONDS,
                    local_mode=False,
                )
            except Exception as e:
                logger.warning(f"Could not init findings sync: {e}")

        self.session_count = 0
        self.runtime_usage: dict[str, float] = {}
        if self.lossless_context_efficiency:
            logger.info(
                "[%s] lossless context efficiency enabled; finding-only wakeups "
                "are batched for up to %ss",
                self.peer_id,
                self.context_efficiency_min_session_interval_seconds,
            )

    def _compose_session_task_prompt(self, *, session_id: str) -> str:
        try:
            prompt = self.peer_memory.compose_session_prompt(
                self.task_prompt,
                session_id=session_id,
                session_index=self.session_count,
            )
        except Exception as exc:
            logger.warning(
                "[%s] peer-local memory prompt build failed; continuing with base prompt: %s",
                self.peer_id,
                exc,
            )
            prompt = self.task_prompt
        if self.session_count == 0:
            try:
                advice = generation_advice(self.peer_id, self.generation_id)
            except Exception as exc:  # noqa: BLE001 - first-wave advice is advisory.
                logger.debug("[%s] generation advice unavailable: %s", self.peer_id, exc)
                advice = {}
            first_wave = str(advice.get("first_wave", ""))
            if first_wave in {"direct_mature", "explore"}:
                target = int(advice.get("mature_target", 0) or 0)
                instruction = (
                    "Begin with one already justified direct mature/full-protocol evaluation "
                    "from your assigned research contract; do not wait for every scout to finish."
                    if first_wave == "direct_mature"
                    else "Begin with the exploration or scout work in your assigned research contract."
                )
                prompt = (
                    prompt.rstrip()
                    + "\n\n# Generation First-Wave Allocation\n\n"
                    + f"The evidence controller targets {target} mature result(s) this generation. "
                    + instruction
                    + " This allocation selects an evidence class, not a scientific hypothesis: "
                    "do not invent filler work or override the PI/peer research contract.\n"
                )
        supply = self._read_resource_supply_signal()
        signal_id = str(supply.get("lease_id", ""))
        if signal_id and signal_id != self._seen_resource_supply_signal_id:
            self._seen_resource_supply_signal_id = signal_id
            self._active_resource_supply_lease_id = signal_id
            profiles = ", ".join(str(item) for item in supply.get("admissible_profiles", []))
            priority = str(supply.get("priority", "frontier_followup"))
            priority_text = (
                "Prioritize one existing mature top-up/direct full-evidence plan and submit it "
                "with work class `mature`."
                if priority == "mature"
                else "Prefer an existing Pareto-relevant follow-up; otherwise use an already planned scout."
            )
            prompt = (
                prompt.rstrip()
                + "\n\n# Runtime Resource Supply\n\n"
                + "The central scheduler granted this peer one short-lived idle-capacity lease "
                + (f"for currently admissible profile(s): {profiles}. " if profiles else ". ")
                + "The lease expiry is the deadline for submitting one existing plan, not a limit "
                "on the runtime of an experiment admitted before expiry; normal generation launch "
                "windows and Closing/Stop still apply. Act promptly rather than spending the lease "
                "window searching for a new idea. " + "Re-read "
                f"`{self.resource_supply_signal_path}` immediately before acting. If the signal "
                "is still active and your existing "
                "research plan already contains a justified next evaluation, submit at most one "
                f"such experiment through the central scheduler. {priority_text} Follow the current "
                "research plan, evidence priorities, and exploration commitments; the resource signal "
                "does not choose the hypothesis or implementation. Do not invent filler work, duplicate an experiment, lower "
                "the evaluation protocol, or bypass Closing/Stop merely to consume hardware. If no "
                "justified experiment is ready, continue analysis or publish results without "
                "submitting work.\n"
            )
        if self.lossless_context_efficiency and self.session_count > 0:
            prompt = prompt.rstrip() + "\n\n" + _LOSSLESS_CONTINUATION_DIRECTIVE + "\n"
        return prompt

    def _read_resource_supply_signal(self) -> dict[str, Any]:
        payload = read_bounded_file_under_root_no_follow(
            self.resource_supply_signal_path,
            self.gen_dir,
            max_bytes=64 * 1024,
        )
        if payload is None:
            return {}
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        try:
            generation_id = int(parsed.get("generation_id", -1))
            expires_at = float(parsed.get("expires_at", 0.0))
        except (TypeError, ValueError):
            return {}
        if (
            not str(parsed.get("lease_id", ""))
            or str(parsed.get("peer_id", "")) != self.peer_id
            or generation_id != self.generation_id
            or expires_at <= time.time()
        ):
            return {}
        try:
            canonical = get_supply_lease(
                self.peer_id,
                self.generation_id,
                str(parsed["lease_id"]),
            )
        except Exception as exc:  # noqa: BLE001 - an unverified locator cannot wake a peer.
            logger.debug("[%s] supply lease verification failed: %s", self.peer_id, exc)
            return {}
        return canonical

    def _resource_supply_signal_pending(self) -> bool:
        payload = self._read_resource_supply_signal()
        signal_id = str(payload.get("lease_id", ""))
        return bool(signal_id and signal_id != self._seen_resource_supply_signal_id)

    def _record_peer_memory_session(
        self,
        *,
        session_id: str,
        result: "AgentResult | None",
        log_file: Path,
        error: BaseException | None,
    ) -> None:
        try:
            self.peer_memory.record_session_result(
                session_id=session_id,
                result=result,
                log_file=log_file,
                error=error,
            )
        except Exception as exc:
            logger.warning(
                "[%s] peer-local memory session recording failed; preserving session outcome: %s",
                self.peer_id,
                exc,
            )

    def _closing_signal_present(self) -> bool:
        if self.closing_signal_path is None:
            return False
        try:
            return self.closing_signal_path.exists()
        except (OSError, ValueError):
            return False

    def _session_event_watch_paths(self, *, productive: bool = True) -> list[Path]:
        gen_dir = self.logs_dir.parent
        if not productive:
            paths = [self.stop_signal_path or (gen_dir / "STOP_SIGNAL")]
            if self.closing_signal_path is not None:
                paths.append(self.closing_signal_path)
            return paths
        paths = [
            self._findings_dir,
        ]
        if self.stop_signal_path is not None:
            paths.append(self.stop_signal_path)
        if self.closing_signal_path is not None:
            paths.append(self.closing_signal_path)
        paths.append(self.resource_supply_signal_path)
        return paths

    def _is_next_session_event(self, path: Path, *, productive: bool = True) -> bool:
        path = Path(path)
        name = path.name
        if name in {
            "STOP_SIGNAL",
            "CLOSING_SIGNAL",
            "ORCHESTRATOR_SHUTDOWN",
        }:
            return True
        if path == self.resource_supply_signal_path:
            return productive and self._resource_supply_signal_pending()
        if not productive:
            return False
        if name.startswith("shared_store.db"):
            return False
        if name.endswith((".log", ".tmp", ".lock", "-wal", "-shm")):
            return False

        try:
            path.relative_to(self._findings_dir)
            if path.suffix.lower() != ".json":
                return False
            if not self.lossless_context_efficiency:
                return True
            return self.peer_memory.should_wake_for_shared_finding(path)
        except ValueError:
            pass
        return False

    def _is_shared_finding_path(self, path: Path) -> bool:
        try:
            Path(path).relative_to(self._findings_dir)
        except ValueError:
            return False
        return Path(path).suffix.lower() == ".json"

    def _finding_only_wakeup(self, result: Any) -> bool:
        paths = getattr(result, "paths", ()) or ()
        return bool(
            getattr(result, "reason", "") == "filesystem_event"
            and paths
            and all(self._is_shared_finding_path(Path(path)) for path in paths)
        )

    @staticmethod
    def _session_was_productive(result: Any) -> bool:
        if result is None:
            return True
        try:
            return int(getattr(result, "iteration_count", 0) or 0) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _session_was_empty(result: Any) -> bool:
        if result is None or not getattr(result, "success", False):
            return False
        try:
            if int(getattr(result, "iteration_count", 0) or 0) > 0:
                return False
        except (TypeError, ValueError):
            return False
        output = getattr(result, "output", {}) or {}
        if not isinstance(output, dict):
            return False
        text_outputs = output.get("text_outputs")
        if isinstance(text_outputs, list) and any(
            str(value).strip() for value in text_outputs if value is not None
        ):
            return False
        for key in ("tool_uses", "background_tasks"):
            value = output.get(key)
            if isinstance(value, list) and any(item not in (None, "", [], {}) for item in value):
                return False
        result_message = output.get("result_message")
        if isinstance(result_message, str):
            return not result_message.strip()
        if isinstance(result_message, (dict, list, tuple, set)):
            return not result_message
        return result_message is None

    @staticmethod
    def _session_needs_immediate_followup(result: Any) -> bool:
        if AutonomousAgentLoop._session_was_empty(result):
            return True
        output = getattr(result, "output", {}) or {}
        return bool(
            getattr(result, "success", False)
            and isinstance(output, dict)
            and output.get("terminal_background_only") is True
        )

    @staticmethod
    def _session_was_bootstrap_wait(result: Any) -> bool:
        if result is None or not getattr(result, "success", False):
            return False
        try:
            if int(getattr(result, "iteration_count", 0) or 0) > 0:
                return False
        except (TypeError, ValueError):
            return False
        output = getattr(result, "output", {}) or {}
        if not isinstance(output, dict):
            return False
        text_outputs = output.get("text_outputs", [])
        if not isinstance(text_outputs, list):
            return False
        combined = "\n".join(str(text) for text in text_outputs).lower()
        if not combined.strip():
            return False
        return any(pattern in combined for pattern in _BOOTSTRAP_WAIT_PATTERNS)

    async def _wait_for_next_session_event(self, *, productive: bool = True) -> None:
        """Wait before opening another runtime session.

        This is the main token-control point. A completed Claude/Codex session
        is no longer followed by an immediate fresh prompt. We only wake on a
        filesystem event that could change the peer's next useful action, a stop
        signal, or a deliberately sparse heartbeat fallback.
        """
        idle_env = (
            "PRAXIST_AGENT_EVENT_IDLE_SECONDS"
            if productive
            else "PRAXIST_AGENT_UNPRODUCTIVE_IDLE_SECONDS"
        )
        idle_seconds = max(60, _int_env(idle_env, 900))
        stop_check_seconds = max(5, _int_env("PRAXIST_AGENT_STOP_CHECK_SECONDS", 30))
        supply_check_seconds = max(1, _int_env("PRAXIST_AGENT_SUPPLY_CHECK_SECONDS", 5))

        await self._release_active_supply_lease(declined=True)

        if productive:
            try:
                await asyncio.to_thread(
                    register_idle_supply,
                    self.peer_id,
                    self.generation_id,
                )
            except Exception as exc:  # noqa: BLE001 - normal event wait remains available.
                logger.debug("[%s] idle supply registration failed: %s", self.peer_id, exc)
        else:
            try:
                await asyncio.to_thread(
                    unregister_idle_supply,
                    self.peer_id,
                    self.generation_id,
                )
            except Exception as exc:  # noqa: BLE001 - normal cooldown remains available.
                logger.debug("[%s] idle supply unregister failed: %s", self.peer_id, exc)

        if self._resource_supply_signal_pending():
            logger.info("[%s] resource supply signal is pending", self.peer_id)
            return

        def _stop_check() -> bool:
            return (
                self.stop_checker.check() is not None
                or self._closing_signal_present()
                or (productive and self._resource_supply_signal_pending())
            )

        result = await wait_for_filesystem_event(
            self._session_event_watch_paths(productive=productive),
            timeout_seconds=idle_seconds,
            stop_check=_stop_check,
            recursive=True,
            max_dirs=max(64, _int_env("PRAXIST_AGENT_EVENT_MAX_WATCH_DIRS", 2048)),
            fallback_interval_seconds=idle_seconds,
            stop_check_interval_seconds=min(stop_check_seconds, supply_check_seconds)
            if productive
            else stop_check_seconds,
            event_filter=lambda p: self._is_next_session_event(
                p,
                productive=productive,
            ),
        )
        finding_batch_waited = False
        if (
            productive
            and self.lossless_context_efficiency
            and self.context_efficiency_min_session_interval_seconds > 0
            and self._finding_only_wakeup(result)
        ):
            remaining = max(
                0.0,
                float(self.context_efficiency_min_session_interval_seconds)
                - float(result.elapsed_seconds),
            )
            if remaining > 0:
                finding_batch_waited = True
                logger.info(
                    "[%s] collecting finding burst for up to %.1fs before the next session",
                    self.peer_id,
                    remaining,
                )
                control_paths = self._session_event_watch_paths(productive=False)
                control_paths.append(self.resource_supply_signal_path)
                result = await wait_for_filesystem_event(
                    control_paths,
                    timeout_seconds=remaining,
                    stop_check=_stop_check,
                    recursive=False,
                    max_dirs=64,
                    fallback_interval_seconds=remaining,
                    stop_check_interval_seconds=min(
                        stop_check_seconds,
                        supply_check_seconds,
                    ),
                    event_filter=lambda p: (
                        self._is_next_session_event(
                            p,
                            productive=True,
                        )
                        and not self._is_shared_finding_path(Path(p))
                    ),
                )
        if productive and not self._resource_supply_signal_pending():
            try:
                await asyncio.to_thread(
                    unregister_idle_supply,
                    self.peer_id,
                    self.generation_id,
                )
            except Exception as exc:  # noqa: BLE001 - normal session wake remains valid.
                logger.debug("[%s] post-wait supply unregister failed: %s", self.peer_id, exc)
        if finding_batch_waited and result.reason in {
            "timeout",
            "fallback_elapsed",
            "no_watch_paths",
        }:
            logger.info(
                "[%s] finding burst collection complete; opening one continuation session",
                self.peer_id,
            )
        elif result.reason == "filesystem_event":
            sample = ", ".join(result.paths[:3])
            suffix = f" ({sample})" if sample else ""
            logger.info(
                "[%s] next-session event after %.1fs%s",
                self.peer_id,
                result.elapsed_seconds,
                suffix,
            )
        elif not productive:
            logger.info(
                "[%s] unproductive-session cooldown elapsed after %.1fs (reason=%s, inotify=%s)",
                self.peer_id,
                result.elapsed_seconds,
                result.reason,
                result.used_inotify,
            )
        elif result.reason == "stop" and self._resource_supply_signal_pending():
            logger.info(
                "[%s] next-session wait found a pending resource supply lease after %.1fs",
                self.peer_id,
                result.elapsed_seconds,
            )
        elif result.reason == "stop":
            logger.info(
                "[%s] next-session wait ended by stop signal after %.1fs",
                self.peer_id,
                result.elapsed_seconds,
            )
        else:
            logger.info(
                "[%s] next-session heartbeat after %.1fs (reason=%s, inotify=%s)",
                self.peer_id,
                result.elapsed_seconds,
                result.reason,
                result.used_inotify,
            )

    def _create_agent(self, session_id: str, message_callback=None) -> BaseAgent:
        # R4-C2 fix: pass a stop_check callback so the BaseAgent's
        # SDK message loop can exit early when STOP_SIGNAL fires.
        # We delegate to the existing StopChecker which already knows
        # about both safety-cap timeout AND the synthesis sentinel.
        def _stop_check():
            sr = self.stop_checker.check()
            return sr is not None

        runtime_env_overrides = {
            "PRAXIST_PEER_ID": self.peer_id,
            "PEER_ID": self.peer_id,
            "GENERATION_ID": str(self.generation_id),
            "PRAXIST_LOGICAL_GENERATION_ID": os.environ.get(
                "PRAXIST_LOGICAL_GENERATION_ID",
                str(self.generation_id),
            ),
            "PRAXIST_GEMS_CYCLE": os.environ.get("PRAXIST_GEMS_CYCLE", "0"),
            "AUTO_RESEARCH_RUN_DIR": str(self.run_dir),
            "PRAXIST_LAUNCH_GUARD_ENABLED": os.environ.get(
                "PRAXIST_LAUNCH_GUARD_ENABLED",
                "0",
            ),
        }
        if self._active_resource_supply_lease_id:
            runtime_env_overrides["PRAXIST_RESOURCE_SUPPLY_LEASE_ID"] = (
                self._active_resource_supply_lease_id
            )
        for key in _legacy_runtime_env_keys():
            value = os.environ.get(key)
            if value:
                runtime_env_overrides.setdefault(key, value)

        run_config = None
        if self.role_ref:
            run_config = RunConfig.from_environ(
                os.environ,
                overrides={"role_ref": self.role_ref},
            )

        return BaseAgent(
            name=f"{self.peer_id}-{session_id}",
            allowed_tools=self.allowed_tools,
            workspace=self.workspace,
            mcp_servers=self.mcp_servers,
            model=self.model,
            cli_path=shutil.which("claude") or None,
            message_callback=message_callback,
            prompt_layout_manifest=self.prompt_layout_manifest,
            plugin_registry=self.plugin_registry,
            stop_check_fn=_stop_check,
            premium_mode=self.premium_mode,
            reasoning_effort=self.reasoning_effort,
            runtime_env_overrides=runtime_env_overrides,
            runtime_timeout_seconds=max(1, int(self._remaining_runtime_seconds())),
            run_config=run_config,
            role_skill_sha256=self.role_skill_sha256,
        )

    async def run(self) -> dict[str, Any]:
        """Run the autonomous agent loop."""
        mode_str = "local" if self.local_mode else "server"
        logger.info(
            f"\n{'=' * 60}\n"
            f"Autonomous Agent Loop ({mode_str} mode)\n"
            f"  Peer ID: {self.peer_id}\n"
            f"  Generation: {self.generation_id}\n"
            f"  Run ID: {self.run_id}\n"
            f"  Max Runtime: {self.max_runtime_seconds / 3600:.1f}h\n"
            f"{'=' * 60}"
        )

        # Initial findings sync
        if self.findings_sync:
            try:
                count = self.findings_sync.sync_once()
                logger.info(f"Fetched {count} finding(s) from server")
            except Exception as e:
                logger.warning(f"Initial findings fetch failed: {e}")
            try:
                self.findings_sync.start()
            except Exception as e:
                logger.warning(f"Could not start findings sync: {e}")

        stop_reason = None
        consecutive_empty_sessions = 0

        while True:
            stop_reason = self.stop_checker.check()
            if stop_reason:
                logger.info(f"Stopping: {stop_reason.value}")
                break
            if self._closing_signal_present():
                stop_reason = StopReason.SYNTHESIS_CLOSING
                logger.info("Stopping before next session: synthesis closing signal present")
                break

            try:
                session_result = await self._run_session()
                if session_result is not None:
                    self.runtime_usage = _merge_numeric_usage(
                        self.runtime_usage,
                        session_result.usage or {},
                    )
                self.session_count += 1
                self.stop_checker.record_success()
                if not self.local_mode:
                    await self._sync_to_s3()
                if self._closing_signal_present():
                    stop_reason = StopReason.SYNTHESIS_CLOSING
                    logger.info("Stopping after current session: synthesis closing signal present")
                    break
                if self._session_needs_immediate_followup(session_result):
                    consecutive_empty_sessions += 1
                    if consecutive_empty_sessions >= _MAX_CONSECUTIVE_EMPTY_SESSIONS:
                        stop_reason = StopReason.RUNTIME_EMPTY
                        logger.warning(
                            "[%s] runtime returned %d consecutive sessions without a "
                            "conclusive assistant result; "
                            "ending this peer without delaying generation close.",
                            self.peer_id,
                            consecutive_empty_sessions,
                        )
                        break
                    await asyncio.sleep(max(0.0, float(_EMPTY_SESSION_RETRY_SECONDS)))
                    continue
                consecutive_empty_sessions = 0
                await self._wait_for_next_session_event(
                    productive=self._session_was_productive(session_result),
                )

            except KeyboardInterrupt:
                logger.info("Interrupted")
                stop_reason = StopReason.USER_INTERRUPT
                break

            except Exception as e:
                error_msg = str(e)

                if is_provider_access_error(error_msg):
                    await self._release_active_supply_lease()
                    # API billing/auth error — pause and retry periodically
                    logger.error(
                        f"[{self.peer_id}] API billing error detected: {error_msg}\n"
                        f"  Pausing all activity. Will retry every "
                        f"{API_BILLING_RETRY_INTERVAL // 60} minutes."
                    )
                    while True:
                        sr = self.stop_checker.check()
                        if sr:
                            stop_reason = sr
                            break
                        if self._closing_signal_present():
                            stop_reason = StopReason.SYNTHESIS_CLOSING
                            break
                        # R1#9 fix: poll the stop signal every 30s during the
                        # billing pause (was: 1200s blanket sleep that ignored
                        # the synthesis trigger STOP_SIGNAL).
                        slept = 0.0
                        billing_check_interval = 30.0
                        while slept < API_BILLING_RETRY_INTERVAL:
                            sr = self.stop_checker.check()
                            if sr:
                                stop_reason = sr
                                break
                            if self._closing_signal_present():
                                stop_reason = StopReason.SYNTHESIS_CLOSING
                                break
                            await asyncio.sleep(billing_check_interval)
                            slept += billing_check_interval
                        if stop_reason:
                            break
                        # R2#8 fix: re-check STOP_SIGNAL one more time
                        # right before invoking _run_session(). The inner
                        # 30s-granularity loop above can let a STOP_SIGNAL
                        # that fires in the last second slip through.
                        sr = self.stop_checker.check()
                        if sr:
                            stop_reason = sr
                            break
                        if self._closing_signal_present():
                            stop_reason = StopReason.SYNTHESIS_CLOSING
                            break
                        logger.info(f"[{self.peer_id}] Retrying after billing pause...")
                        try:
                            retry_result = await self._run_session()
                            if retry_result is not None:
                                self.runtime_usage = _merge_numeric_usage(
                                    self.runtime_usage,
                                    retry_result.usage or {},
                                )
                            logger.info(f"[{self.peer_id}] Billing restored — resuming.")
                            self.session_count += 1
                            self.stop_checker.record_success()
                            break  # Exit billing wait loop, resume normal loop
                        except Exception as retry_err:
                            if is_provider_access_error(str(retry_err)):
                                logger.info(
                                    f"[{self.peer_id}] Still billing error. "
                                    f"Next retry in "
                                    f"{API_BILLING_RETRY_INTERVAL // 60}m."
                                )
                            else:
                                # Different error — fall back to normal handling
                                logger.error(f"Session error: {retry_err}")
                                self.stop_checker.record_error()
                                break
                    if stop_reason:
                        break
                else:
                    logger.error(f"Session error: {e}")
                    traceback.print_exc()
                    self.stop_checker.record_error()
                    stop_reason = self.stop_checker.check()
                    if stop_reason:
                        logger.info(f"Stopping after session failure: {stop_reason.value}")
                        break
                    if self._closing_signal_present():
                        stop_reason = StopReason.SYNTHESIS_CLOSING
                        logger.info(
                            "Stopping after session failure: synthesis closing signal present"
                        )
                        break
                    if self.stop_checker.consecutive_errors >= _MAX_CONSECUTIVE_RUNTIME_FAILURES:
                        stop_reason = StopReason.RUNTIME_FAILURE
                        logger.warning(
                            "[%s] runtime failed %d consecutive sessions; ending this peer "
                            "without delaying generation close.",
                            self.peer_id,
                            self.stop_checker.consecutive_errors,
                        )
                        break
                    await asyncio.sleep(max(0.0, float(_RUNTIME_FAILURE_RETRY_SECONDS)))

        await self._release_active_supply_lease()

        if self.findings_sync:
            try:
                self.findings_sync.stop()
            except Exception as e:
                # Adv-R1.3 fix: was silently swallowed, hiding thread leaks.
                # Now log so we know if the background sync thread didn't
                # actually stop (could leave SQLite connection / FD leaked).
                logger.warning(
                    "findings_sync.stop() failed: %s — background thread may "
                    "still be alive (resource leak)",
                    e,
                )

        if not self.local_mode:
            await self._sync_to_s3()

        result: dict[str, Any] = {
            "peer_id": self.peer_id,
            "generation_id": self.generation_id,
            "run_id": self.run_id,
            "sessions": self.session_count,
            "duration_seconds": self.stop_checker.elapsed_time,
            "stop_reason": stop_reason.value if stop_reason else "unknown",
        }
        if self.runtime_usage:
            result["runtime_usage"] = dict(self.runtime_usage)
            total_tokens = self.runtime_usage.get("total_tokens")
            if total_tokens is not None:
                result["total_tokens"] = total_tokens

        logger.info(
            f"\n{'=' * 60}\n"
            f"Done: {result['sessions']} sessions, "
            f"{result['duration_seconds'] / 3600:.1f}h, "
            f"reason={result['stop_reason']}\n"
            f"{'=' * 60}"
        )

        return result

    async def _release_active_supply_lease(self, *, declined: bool = False) -> None:
        lease_id = self._active_resource_supply_lease_id
        if not lease_id:
            return
        try:
            await asyncio.to_thread(
                release_supply_lease,
                lease_id,
                self.peer_id,
                declined=declined,
                reason="session_ended_without_submission" if declined else "peer_session_finished",
            )
        except Exception as exc:  # noqa: BLE001 - supply feedback is advisory.
            logger.debug("[%s] resource supply lease release failed: %s", self.peer_id, exc)
        else:
            self._active_resource_supply_lease_id = ""

    def _remaining_runtime_seconds(self) -> float:
        return max(0.0, float(self.max_runtime_seconds) - self.stop_checker.elapsed_time)

    async def _run_session(self) -> AgentResult:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{self.session_count:03d}_{timestamp}"
        log_file = self.logs_dir / f"{session_id}.log"
        session_task_prompt = self._compose_session_task_prompt(session_id=session_id)

        logger.info(f"[Session {self.session_count}] {session_id}")

        # buffering=1 → line-buffered text mode. Every ``write(...)`` that
        # ends in ``\n`` (which all of ours do) flushes the Python-side
        # buffer to the OS immediately, so SIGKILL / OOM / hard crashes
        # leave the log truncated at the last completed line rather than
        # losing the header and the ``# Ended:`` marker.
        with open(log_file, "w", buffering=1) as log_f:
            log_f.write(
                f"# Session {session_id}\n"
                f"# Peer: {self.peer_id}\n"
                f"# Generation: {self.generation_id}\n"
                f"# Started: {datetime.now().isoformat()}\n\n"
            )

            def message_callback(message):
                ts = datetime.now().strftime("%H:%M:%S")
                log_f.write(f"\n[{ts}] {type(message).__name__}\n")
                # claude_sdk streams native Claude message objects
                # (``.content`` is a list of content blocks with ``.text``
                # or ``.name`` + ``.input``). Runtime-neutral adapters stream
                # ``AgentEvent`` records (``.type`` + ``.payload``); the
                # branch below captures both shapes.
                if hasattr(message, "content"):
                    for content in message.content:
                        if hasattr(content, "text"):
                            text, _ = redact_text(str(content.text))
                            log_f.write(f"{text}\n")
                        elif hasattr(content, "name"):
                            log_f.write(f"Tool: {content.name}\n")
                            if hasattr(content, "input"):
                                input_str = dumps_redacted(content.input, indent=2)
                                if len(input_str) > 1000:
                                    input_str = input_str[:1000] + "... [truncated]"
                                log_f.write(f"Input: {input_str}\n")
                elif hasattr(message, "type") and hasattr(message, "payload"):
                    log_f.write(f"event_type: {message.type}\n")
                    payload = message.payload
                    if isinstance(payload, dict):
                        # ``assistant_text`` events carry the model's reply
                        # under ``payload["text"]`` — surface that directly
                        # so the log reads like a transcript instead of a
                        # blob of JSON-encoded payload.
                        text = payload.get("text") if message.type == "assistant_text" else None
                        if isinstance(text, str) and text:
                            redacted, _ = redact_text(text)
                            log_f.write(f"{redacted}\n")
                        else:
                            payload_str = dumps_redacted(payload, indent=2)
                            if len(payload_str) > 1000:
                                payload_str = payload_str[:1000] + "... [truncated]"
                            log_f.write(f"payload: {payload_str}\n")

            result: AgentResult | None = None
            session_error: BaseException | None = None
            try:
                agent = self._create_agent(session_id, message_callback=message_callback)
                result = await agent.execute(task=session_task_prompt)
                if self._session_was_bootstrap_wait(result):
                    first_usage = dict(result.usage or {})
                    log_f.write(
                        "\n# Bootstrap failure: runtime waited for a human "
                        "instruction with zero tool calls. Retrying once with "
                        "an explicit Praxist start directive.\n"
                    )
                    log_f.flush()
                    retry_agent = self._create_agent(
                        f"{session_id}_bootstrap_retry",
                        message_callback=message_callback,
                    )
                    result = await retry_agent.execute(
                        task=_with_bootstrap_retry_directive(session_task_prompt)
                    )
                    result.usage = _merge_numeric_usage(
                        first_usage,
                        result.usage or {},
                    )
                    if self._session_was_bootstrap_wait(result):
                        result = AgentResult(
                            success=False,
                            output=result.output,
                            duration=result.duration,
                            iteration_count=result.iteration_count,
                            error=(
                                "bootstrap failure: runtime still waited for "
                                "human instruction after explicit retry"
                            ),
                            usage=dict(result.usage or {}),
                        )

                log_f.write(
                    f"\n# Result: success={result.success}, "
                    f"duration={result.duration:.1f}s, "
                    f"tools={result.iteration_count}\n"
                )
                if result.error:
                    log_f.write(f"# Error: {result.error}\n")

                if not result.success:
                    self.runtime_usage = _merge_numeric_usage(
                        self.runtime_usage,
                        result.usage or {},
                    )
                    raise RuntimeError(f"Agent failed: {result.error}")
                return result

            except Exception as e:
                session_error = e
                log_f.write(f"\n# ERROR: {e}\n{traceback.format_exc()}")
                raise
            finally:
                self._record_peer_memory_session(
                    session_id=session_id,
                    result=result,
                    log_file=log_file,
                    error=session_error,
                )
                log_f.write(f"\n# Ended: {datetime.now().isoformat()}\n")

        logger.info(f"[Session {self.session_count}] Completed")
        raise AssertionError("session completed without an AgentResult")

    async def _sync_to_s3(self):
        try:
            from praxist.infrastructure.s3_utils import upload_file_to_s3
        except ImportError:
            return

        try:
            if self.findings_path.exists():
                upload_file_to_s3(
                    file_path=self.findings_path,
                    s3_key=f"{self.s3_prefix}findings.json",
                    bucket_name=self.s3_bucket,
                    content_type="application/json",
                )
            for log_file in self.logs_dir.glob("session_*.log"):
                upload_file_to_s3(
                    file_path=log_file,
                    s3_key=f"{self.s3_prefix}logs/{log_file.name}",
                    bucket_name=self.s3_bucket,
                    content_type="text/plain",
                )
            memory_dir = getattr(self.peer_memory, "memory_dir", None)
            if memory_dir is not None:
                try:
                    ensure_memory_dir = getattr(self.peer_memory, "_ensure_memory_dir", None)
                    if callable(ensure_memory_dir):
                        ensure_memory_dir()
                except OSError as exc:
                    logger.warning("Skipping unsafe peer memory artifact tree: %s", exc)
                    memory_dir = None
            if memory_dir is not None:
                memory_root = Path(memory_dir)
                try:
                    root_stat = memory_root.lstat()
                except OSError:
                    root_stat = None
                if (
                    root_stat is None
                    or stat.S_ISLNK(root_stat.st_mode)
                    or not stat.S_ISDIR(root_stat.st_mode)
                ):
                    logger.warning("Skipping unsafe peer memory artifact root: %s", memory_root)
                    return
                memory_root_resolved = memory_root.resolve(strict=True)
                for memory_file in sorted(memory_root.iterdir()):
                    try:
                        st = memory_file.lstat()
                    except OSError:
                        continue
                    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                        continue
                    try:
                        memory_file.resolve(strict=True).relative_to(memory_root_resolved)
                    except (OSError, ValueError):
                        continue
                    limit = _peer_memory_upload_limit(memory_file)
                    if limit is None or st.st_size > limit:
                        continue
                    payload = read_bounded_file_under_root_no_follow(
                        memory_file,
                        memory_root,
                        max_bytes=limit,
                    )
                    if payload is None:
                        continue
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            prefix="praxist-peer-memory-",
                            suffix=memory_file.suffix or ".txt",
                            delete=False,
                        ) as tmp:
                            tmp_path = Path(tmp.name)
                            tmp.write(payload)
                        upload_file_to_s3(
                            file_path=tmp_path,
                            s3_key=f"{self.s3_prefix}memory/{memory_file.relative_to(memory_root).as_posix()}",
                            bucket_name=self.s3_bucket,
                            content_type="text/plain",
                        )
                    finally:
                        if tmp_path is not None:
                            with suppress(OSError):
                                tmp_path.unlink()
        except Exception as e:
            logger.warning(f"S3 sync failed: {e}")
