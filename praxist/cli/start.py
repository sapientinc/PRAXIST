"""``praxist start`` — canonical registry-backed Praxist research launcher.

The command writes a registry entry consumed by the remaining lifecycle
subcommands and is the only supported detached launch path.

Agent system selection uses ``PRAXIST_AGENT_SYSTEM`` (or ``--agent-system``)
to pick which agent runtime the launched run will use. The rest of the
defaults (provider, runtime ref, credential env var) cascade from that
choice.

* ``claude_sdk`` (default) — in-process Anthropic SDK runtime.
  Prefers ``model_provider:deepseek_alias`` and ``DEEPSEEK_API_KEY``
  when the DeepSeek direct key is available; otherwise falls back to
  ``model_provider:openrouter`` with ``OPENROUTER_API_KEY`` and then
  ``model_provider:anthropic_messages`` with ``ANTHROPIC_API_KEY``.
* ``codex_sdk`` — official Codex Python SDK/app-server runtime. Uses the
  explicitly configured provider, otherwise prefers available DeepSeek and
  OpenRouter credentials before its native OpenAI provider. The codex_sdk
  runtime plugin then auto-starts its own ``codex-relay`` sidecar
  for non-OpenAI providers, so ``praxist start`` does not start the
  relay itself.

What this subcommand does, in order:

1. Resolve agent system, then defaults that cascade from it: model,
   provider, runtime ref, task path, run dir, frontier strategy.
2. Credential precheck: require the active provider's env credential, except
   for native OpenAI through ``codex_sdk`` where child startup may discover a
   saved ChatGPT login. Raw secrets are never printed.
3. Build ``python -m praxist.run run …`` argv and a
   ``<run_dir>/logs/launcher.nohup.log`` log path.
4. Spawn the child with ``start_new_session=True`` (cross-platform
   equivalent of ``nohup setsid``), stdio redirected to the log file
   and ``stdin`` connected to ``/dev/null`` so the child cannot inherit
   the operator's controlling terminal.
5. Persist a :class:`RegistryEntry` and print a redacted operator hint.

What this subcommand intentionally does **not** do:

* Parse or print raw API keys.  Credential resolution remains in
  Python core, not in this launcher.
* Start ``codex-relay``.  When the run uses ``agent_runtime:codex_sdk``
  with a non-OpenAI provider, the codex_sdk plugin manages its own
  private run/provider-scoped sidecar relay — see
  :mod:`praxist.plugins.agent_runtimes.codex_sdk._relay`.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from praxist.cli._env import (
    AGENT_SYSTEM_TO_RUNTIME_REF,
    AGENT_SYSTEM_VALUES,
    CODEX_NATIVE_DEFAULT_MODEL,
    PROVIDER_KEY_MAP,
    PROVIDER_REF_FOR_SHORT_NAME,
    default_provider_for_agent_system,
    getenv,
)
from praxist.cli._setup_common import (
    load_cli_environment,
    normalize_runtime_selection,
    selected_config_file,
)
from praxist.cli.registry import (
    SCHEMA_VERSION,
    STATE_RUNNING,
    STATE_STOPPED,
    RegistryEntry,
    RegistryError,
    create_entry,
    entry_is_local,
    entry_lock,
    entry_path,
    entry_process_epoch_matches,
    local_host_identity,
    process_identity_matches,
    process_start_token,
    read_entry,
    registry_lock,
    remove_entry,
    write_entry,
)
from praxist.cli.status import pid_is_alive, read_ps_table, registry_command_matches
from praxist.task_spec import load_task_spec

if TYPE_CHECKING:  # pragma: no cover - import only for static type checkers
    from collections.abc import Callable

DEFAULT_TASK_PATH = "."
"""Default task project path: the directory where ``praxist start`` is invoked."""

_CONVENTIONAL_BASELINE_RESULT_FILENAMES = (
    "results.jsonl",
    "baseline_results.jsonl",
    "results.json",
    "baseline_results.json",
    "summary.json",
)
_MAX_BASELINE_RESULT_BYTES = 4 * 1024 * 1024

DEFAULT_FRONTIER_STRATEGY = "auto"
"""Default ``--frontier-strategy`` value."""

DEFAULT_AGENT_SYSTEM = "claude_sdk"
"""Default ``PRAXIST_AGENT_SYSTEM`` for runs launched by ``praxist start``.

``praxist start`` launches a Peer-style python run that talks to a model
provider directly; ``claude_sdk`` is the in-process Anthropic SDK
runtime and the production default per AGENTS.md §7.
"""

OPENROUTER_PROVIDER_REF = PROVIDER_REF_FOR_SHORT_NAME["openrouter"]
ANTHROPIC_PROVIDER_REF = PROVIDER_REF_FOR_SHORT_NAME["anthropic"]
OPENAI_PROVIDER_REF = PROVIDER_REF_FOR_SHORT_NAME["openai"]
DEEPSEEK_PROVIDER_REF = PROVIDER_REF_FOR_SHORT_NAME["deepseek"]

OPENROUTER_DEFAULT_MODEL = "anthropic/claude-opus-4.7"
ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-7"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-pro[1m]"

# Back-compat alias retained so callers that imported the old constant
# from previous PRs keep working until they migrate to
# AGENT_SYSTEM_TO_RUNTIME_REF[DEFAULT_AGENT_SYSTEM].
DEFAULT_RUNTIME_REF = AGENT_SYSTEM_TO_RUNTIME_REF[DEFAULT_AGENT_SYSTEM]
DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0
CODEX_NATIVE_BLOCKED_ENV = (
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "CODEX_ACCESS_TOKEN",
    "OPENAI_BASE_URL",
    "PRAXIST_CODEX_BIN",
    "MODEL_PROVIDER_REF",
    "PRAXIST_MODEL_PROVIDER_REF",
    "PRAXIST_LLM_PROVIDER",
    "PRAXIST_AGENT_RUNTIME_REF",
    "RUNTIME_REF",
    "MODEL",
    "PRAXIST_MODEL",
)


class StartError(RuntimeError):
    """Raised when ``praxist start`` cannot proceed (bad args, missing creds, …)."""


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``praxist start`` subcommand on the parent parser."""
    parser = subparsers.add_parser(
        "start",
        help="Launch a new Praxist research run (registry-backed).",
        description=(
            "Async launcher: starts ``python -m praxist.run run`` in a "
            "new session, redirects stdout/stderr to a run-local log file, "
            "and writes a registry entry under $PRAXIST_STATE_DIR/runs/.\n\n"
            "Pass --task-path / --model / --model-provider to override the "
            "resolved task and runtime configuration."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task-path",
        dest="task_path",
        default=None,
        help="Task project directory (default: $TASK_PATH or the current directory).",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Config file to load (default: $PRAXIST_CONFIG_FILE or the user config).",
    )
    parser.add_argument(
        "--agent-system",
        dest="agent_system",
        choices=AGENT_SYSTEM_VALUES,
        default=None,
        help=(
            "Agent system the launched run will use. "
            "Default: $PRAXIST_AGENT_SYSTEM if set, else "
            f"{DEFAULT_AGENT_SYSTEM!r}. "
            "Recognised values: claude_sdk (default), codex_sdk."
        ),
    )
    parser.add_argument(
        "--runtime",
        dest="runtime_ref",
        default=None,
        help=(
            "Explicit ``agent_runtime:*`` plugin ref. Wins over the agent-system mapping when set."
        ),
    )
    parser.add_argument(
        "--codex-native",
        action="store_true",
        help=(
            "Use codex_sdk with native OpenAI and saved ChatGPT login, ignoring "
            "API-key and custom-endpoint settings from process/config/task env."
        ),
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dir",
        default=None,
        help="Explicit run directory (default: <task>/experiments/run_<ts>_<task>).",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume an existing run directory instead of requiring fresh artifacts.",
    )
    parser.add_argument(
        "--resume-from",
        dest="resume_from",
        default=None,
        help=(
            "Path to an existing run directory to resume. Equivalent to --run-dir <path> --resume."
        ),
    )
    parser.add_argument(
        "--resume-policy",
        dest="resume_policy",
        default="completed_generation",
        choices=["completed_generation"],
        help="Resume policy forwarded to praxist.run.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="Model name forwarded to the runtime; defaults depend on provider.",
    )
    parser.add_argument(
        "--model-provider",
        dest="model_provider_ref",
        default=None,
        help=(
            "Provider plugin ref (e.g. model_provider:deepseek_alias). "
            "Default cascades from agent system: claude_sdk → "
            "deepseek_alias when DEEPSEEK_API_KEY is set, then openrouter "
            "when OPENROUTER_API_KEY is set, then anthropic_messages; "
            "codex_sdk follows the same credential-aware selection and falls "
            "back to openai_compatible."
        ),
    )
    parser.add_argument(
        "--strategy",
        dest="frontier_strategy",
        default=DEFAULT_FRONTIER_STRATEGY,
        choices=("auto", "mixed", "explore", "exploit"),
        help="Frontier strategy (auto|mixed|explore|exploit).",
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        type=_positive_int,
        default=None,
        help="Cohort size override (exported as COHORT_SIZE).",
    )
    parser.add_argument(
        "--generations",
        dest="generations",
        type=_positive_int,
        default=None,
        help="Maximum generations override (exported as MAX_GENERATIONS).",
    )
    parser.add_argument(
        "--server",
        dest="server",
        action="store_true",
        help="Disable --local mode (server mode).",
    )
    parser.add_argument(
        "--daemonize",
        dest="daemonize",
        action="store_true",
        help=(
            "Double-fork the launcher before spawning so the workload "
            "survives when the launching shell's process tree is "
            "reaped. Required for sandboxed launcher contexts "
            "(agent tool shells, CI runners, Docker ``--init``). "
            "The default ``start_new_session=True`` "
            "path is fine for a normal terminal."
        ),
    )
    parser.add_argument(
        "--startup-timeout",
        type=_nonnegative_finite_float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
        help=(
            "Seconds to wait for startup artifacts before returning. A live "
            "run that exceeds the deadline remains in 'starting' state "
            f"(default {DEFAULT_STARTUP_TIMEOUT_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit one JSON document on stdout instead of the operator table.",
    )
    parser.set_defaults(func=cmd_start)


def cmd_start(args: argparse.Namespace) -> int:
    """Handler for ``praxist start``."""
    try:
        resolved_task_path = _resolve_task_path(args.task_path)
        load_cli_environment(
            resolved_task_path,
            config_file=selected_config_file(args.config_file),
        )
        _validate_task_project(resolved_task_path)
        if not args.as_json:
            _offer_product_usage_consent()
        entry = launch_run(
            task_path=str(resolved_task_path),
            agent_system=args.agent_system,
            runtime_ref=args.runtime_ref,
            run_dir=args.run_dir,
            model=args.model,
            model_provider_ref=args.model_provider_ref,
            frontier_strategy=args.frontier_strategy,
            cohort=args.cohort,
            generations=args.generations,
            server=args.server,
            daemonize=args.daemonize,
            resume=args.resume,
            resume_from=args.resume_from,
            resume_policy=args.resume_policy,
            startup_timeout=args.startup_timeout,
            codex_native=args.codex_native,
        )
    except (OSError, StartError) as exc:
        sys.stderr.write(f"praxist start: {exc}\n")
        return 1
    if args.as_json:
        sys.stdout.write(json.dumps(entry.to_dict(), indent=2) + "\n")
    else:
        _write_operator_hint(entry)
    return 1 if entry.state == STATE_STOPPED or entry.extra.get("startup_state") == "failed" else 0


def _offer_product_usage_consent() -> None:
    """Offer unset consent in the foreground before the detached child starts."""

    try:
        from praxist.cli.product_usage import prompt_for_consent_if_unset

        prompt_for_consent_if_unset()
    except Exception:
        # Product-usage setup must never become a launch prerequisite.
        return


def launch_run(
    *,
    task_path: str | None,
    agent_system: str | None = None,
    runtime_ref: str | None = None,
    run_dir: str | None,
    model: str | None,
    model_provider_ref: str | None,
    frontier_strategy: str,
    cohort: str | None,
    generations: str | None,
    server: bool,
    daemonize: bool = False,
    resume: bool = False,
    resume_from: str | None = None,
    resume_policy: str = "completed_generation",
    startup_timeout: float = 0.0,
    now: _dt.datetime | None = None,
    spawn: Callable[..., subprocess.Popen[bytes]] | None = None,
    daemon_spawn: Callable[..., int] | None = None,
    _lifecycle_locked: bool = False,
    _defer_startup_wait: bool = False,
    codex_native: bool = False,
) -> RegistryEntry:
    """Resolve defaults, spawn the run, persist the registry entry.

    ``now``, ``spawn`` and ``daemon_spawn`` are injectable for tests;
    production callers leave them as ``None`` so the wall clock,
    :class:`subprocess.Popen` and :func:`_spawn_daemonized` are used.
    ``daemonize=True`` selects the double-fork path so the workload
    survives a sandboxed launcher's process-tree reaper (agent tool
    shells, CI runners, Docker ``--init``).
    """
    now = now or _dt.datetime.now(_dt.UTC)
    production_launch = spawn is None and daemon_spawn is None
    spawn = spawn or _default_spawn
    daemon_spawn = daemon_spawn or _spawn_daemonized

    if codex_native:
        _select_codex_native_mode(
            agent_system=agent_system,
            runtime_ref=runtime_ref,
            model_provider_ref=model_provider_ref,
        )
        _sanitize_codex_native_environment()
        agent_system = "codex_sdk"
        runtime_ref = "agent_runtime:codex_sdk"
        model_provider_ref = OPENAI_PROVIDER_REF
        model = model or CODEX_NATIVE_DEFAULT_MODEL
    resolved_agent_system, resolved_runtime_ref = _resolve_runtime_selection(
        agent_system, runtime_ref
    )
    resolved_task_path = _resolve_task_path(task_path)
    provider_ref = _resolve_provider_ref(model_provider_ref, resolved_agent_system)
    _precheck_credentials(provider_ref, resolved_runtime_ref)
    resolved_model = _resolve_model(model, provider_ref, resolved_agent_system)
    resume_from_path = Path(resume_from).expanduser().resolve() if resume_from else None
    if resume_from_path is not None and run_dir:
        explicit_run_dir = Path(run_dir).expanduser().resolve()
        if explicit_run_dir != resume_from_path:
            raise StartError(
                "--resume-from and --run-dir refer to different directories; "
                "pass only --resume-from or make them identical"
            )
    if resume and resume_from_path is None and not run_dir:
        raise StartError("--resume requires --run-dir or --resume-from.")
    resolved_run_dir = resume_from_path or _resolve_run_dir(run_dir, resolved_task_path, now)
    resume_enabled = bool(resume or resume_from_path is not None)
    if not resume_enabled and resolved_run_dir.exists():
        blocking_paths = [
            path for path in resolved_run_dir.iterdir() if not _is_ignorable_precreated_path(path)
        ]
        if blocking_paths:
            has_artifacts = any(
                (resolved_run_dir / rel).exists()
                for rel in (
                    "run.json",
                    "startup_config.json",
                    "trajectory.jsonl",
                    "budget_ledger.jsonl",
                    "artifact_index.jsonl",
                    "run_summary.json",
                    "plugin_resolution.json",
                    "effective_task_spec.yaml",
                    "model_profiles.json",
                    "cache_policy.json",
                    "task_project_manifest.json",
                    "credentials_redacted.json",
                    "findings",
                    "frontier",
                    "agendas",
                    "memory",
                    "gems",
                    "gen_0",
                )
            )
            if has_artifacts:
                raise StartError(
                    f"fresh run directory is not empty: {resolved_run_dir}. "
                    "Choose another --run-dir or use --resume-from."
                )
            raise StartError(
                f"fresh run directory is not empty: {resolved_run_dir}. Choose another --run-dir."
            )
    run_id = resolved_run_dir.name
    startup_baseline = _startup_artifact_signatures(resolved_run_dir)
    log_file = resolved_run_dir / "logs" / "launcher.nohup.log"
    (resolved_run_dir / "logs").mkdir(parents=True, exist_ok=True)

    command = _build_command(
        task_path=resolved_task_path,
        run_dir=resolved_run_dir,
        model=resolved_model,
        provider_ref=provider_ref,
        runtime_ref=resolved_runtime_ref,
        frontier_strategy=frontier_strategy,
        server=server,
        resume=resume_enabled,
        resume_from=resume_from_path,
        resume_policy=resume_policy,
    )

    env = _build_env(
        cohort=cohort,
        generations=generations,
        agent_system=resolved_agent_system,
    )
    reservation_extra = {
        "agent_system": resolved_agent_system,
        "daemonized": "1" if daemonize else "0",
        "monitor_command": monitor_command(run_id),
        "monitor_mode": "foreground",
        "resume": "1" if resume_enabled else "0",
        "resume_policy": resume_policy if resume_enabled else "",
        "resume_from": str(resume_from_path or "") if resume_enabled else "",
        "cohort": str(cohort or ""),
        "generations": str(generations or ""),
        "server": "1" if server else "0",
        "startup_state": "starting",
        "auth_mode": "codex-native" if codex_native else "configured-provider",
        **local_host_identity(),
    }
    reservation = RegistryEntry(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        pid=0,
        parent_pid=os.getpid(),
        run_dir=str(resolved_run_dir),
        log_file=str(log_file),
        task_path=str(resolved_task_path),
        model=resolved_model,
        model_provider_ref=provider_ref,
        runtime_ref=resolved_runtime_ref,
        command=tuple(command),
        command_prefix=_command_prefix(command),
        started_at=now.isoformat(),
        state=STATE_RUNNING,
        extra=reservation_extra,
    )
    consumed_shutdown_fence: tuple[Path, bytes] | None = None
    try:
        try:
            with contextlib.ExitStack() as locks:
                if not _lifecycle_locked:
                    locks.enter_context(registry_lock())
                    locks.enter_context(entry_lock(run_id))
                if resume_enabled:
                    _validate_resume_registry_slot(reservation)
                    try:
                        consumed_shutdown_fence = _consume_shutdown_fence(resolved_run_dir)
                    except OSError as exc:
                        raise StartError(
                            f"could not reopen stopped run {resolved_run_dir}: {exc}"
                        ) from exc
                owns_reservation = False
                if not resume_enabled:
                    try:
                        create_entry(reservation)
                    except RegistryError as exc:
                        raise StartError(
                            f"run_id {run_id!r} is already being launched or exists; "
                            "choose a distinct --run-dir."
                        ) from exc
                    owns_reservation = True

                try:
                    if daemonize:
                        pid = daemon_spawn(command, log_file, env)
                    else:
                        pid = _spawn_child(spawn, command, log_file, env)
                except BaseException:
                    if owns_reservation:
                        remove_entry(run_id)
                    raise

                extra = {
                    **reservation_extra,
                    "startup_state": (
                        "starting" if production_launch and startup_timeout > 0 else "running"
                    ),
                }
                token = process_start_token(pid)
                if token:
                    extra["process_start_token"] = token
                entry = replace(
                    reservation,
                    pid=pid,
                    extra=extra,
                )
                try:
                    write_entry(entry)
                except OSError as exc:
                    with contextlib.suppress(OSError):
                        os.kill(pid, signal.SIGTERM)
                    if owns_reservation:
                        remove_entry(run_id)
                    raise StartError(f"could not persist run registry entry: {exc}") from exc
        except RegistryError as exc:
            raise StartError(str(exc)) from exc
        if production_launch and startup_timeout > 0 and not _defer_startup_wait:
            entry = _wait_for_startup(entry, startup_timeout, startup_baseline)
    except BaseException:
        try:
            _restore_shutdown_fence(consumed_shutdown_fence)
        except OSError as restore_exc:
            raise StartError(
                f"resume launch failed and its shutdown fence could not be restored: {restore_exc}"
            ) from restore_exc
        raise
    if entry.state == STATE_STOPPED or entry.extra.get("startup_state") == "failed":
        try:
            _restore_shutdown_fence(consumed_shutdown_fence)
        except OSError as exc:
            raise StartError(
                f"could not restore shutdown fence after failed startup: {exc}"
            ) from exc
    return entry


def _consume_shutdown_fence(run_dir: Path) -> tuple[Path, bytes] | None:
    """Consume the existing stop fence while the caller owns lifecycle locking."""

    path = Path(run_dir) / "ORCHESTRATOR_SHUTDOWN"
    if not path.exists():
        return None
    payload = path.read_bytes()
    path.unlink()
    return path, payload


def _restore_shutdown_fence(fence: tuple[Path, bytes] | None) -> None:
    """Restore a consumed stop fence without overwriting a newer stop."""

    if fence is None:
        return
    path, payload = fence
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        return


def _select_codex_native_mode(
    *,
    agent_system: str | None,
    runtime_ref: str | None,
    model_provider_ref: str | None,
) -> None:
    """Validate explicit selections before Codex-native defaults are applied."""

    if agent_system not in (None, "codex_sdk"):
        raise StartError("--codex-native cannot be combined with a non-codex_sdk agent system")
    if runtime_ref not in (None, "", "agent_runtime:codex_sdk"):
        raise StartError("--codex-native requires agent_runtime:codex_sdk")
    if model_provider_ref not in (None, "", OPENAI_PROVIDER_REF):
        raise StartError("--codex-native requires model_provider:openai_compatible")


def _sanitize_codex_native_environment() -> None:
    """Keep saved-login mode isolated from API-key and endpoint configuration."""

    for key in CODEX_NATIVE_BLOCKED_ENV:
        os.environ.pop(key, None)


def _validate_resume_registry_slot(reservation: RegistryEntry) -> None:
    """Reject a resume that would overwrite another live or unrelated run."""

    path = entry_path(reservation.run_id)
    try:
        existing = read_entry(reservation.run_id)
    except RegistryError as exc:
        if path.exists():
            raise StartError(
                f"run id {reservation.run_id!r} has an unreadable registry entry: {exc}"
            ) from exc
        return
    existing_dir = Path(existing.run_dir).expanduser().resolve()
    requested_dir = Path(reservation.run_dir).expanduser().resolve()
    if existing_dir != requested_dir:
        raise StartError(
            f"run id {reservation.run_id!r} is registered for {existing_dir}, not {requested_dir}"
        )
    if entry_is_local(existing) is False:
        raise StartError(
            f"run id {reservation.run_id!r} belongs to "
            f"{existing.extra.get('hostname', 'another host')!r}; resume it there"
        )
    if entry_process_epoch_matches(existing) is False:
        return
    identity = process_identity_matches(existing)
    if identity is False:
        return
    if identity is True:
        if pid_is_alive(existing.pid):
            raise StartError(
                f"run id {reservation.run_id!r} still appears to be running; stop it first"
            )
        return
    live = read_ps_table().get(existing.pid)
    if live is not None:
        _ppid, _etime, command = live
        if registry_command_matches(existing, command):
            raise StartError(
                f"run id {reservation.run_id!r} still appears to be running; stop it first"
            )
        return
    if pid_is_alive(existing.pid):
        raise StartError(
            f"run id {reservation.run_id!r} has a live controller that cannot be "
            "verified from the process table; refusing to overwrite it"
        )


def _resolve_agent_system(raw: str | None) -> str:
    """Resolve ``--agent-system`` to a value in :data:`AGENT_SYSTEM_VALUES`.

    ``praxist start`` defaults to ``claude_sdk``, the production runtime per
    AGENTS.md §7.
    """
    value = raw or getenv("PRAXIST_AGENT_SYSTEM", "")
    value = value.strip().lower() or DEFAULT_AGENT_SYSTEM
    if value not in AGENT_SYSTEM_VALUES:
        raise StartError(
            f"unknown PRAXIST_AGENT_SYSTEM={value!r}; expected one of {AGENT_SYSTEM_VALUES}."
        )
    return value


def _resolve_runtime_selection(
    agent_system: str | None,
    runtime_ref: str | None,
) -> tuple[str, str]:
    """Resolve and cross-check the public agent-system/runtime pair."""
    try:
        return normalize_runtime_selection(
            agent_system=agent_system,
            runtime_ref=runtime_ref,
            default_agent_system=DEFAULT_AGENT_SYSTEM,
        )
    except ValueError as exc:
        raise StartError(str(exc)) from exc


def _resolve_runtime_ref(raw: str | None, agent_system: str) -> str:
    """Resolve the ``agent_runtime:*`` plugin ref for ``agent_system``."""
    if raw:
        return raw
    from_env = getenv("PRAXIST_AGENT_RUNTIME_REF", "").strip() or getenv("RUNTIME_REF", "").strip()
    if from_env:
        return from_env
    return AGENT_SYSTEM_TO_RUNTIME_REF[agent_system]


def _resolve_task_path(raw: str | None) -> Path:
    """Resolve ``--task-path`` to an absolute, existing directory."""
    candidate = raw or getenv("TASK_PATH", "") or Path.cwd()
    path = Path(candidate).expanduser().resolve()
    if not path.is_dir():
        raise StartError(
            f"task project directory not found: {path}. "
            "Pass --task-path <task-project> or set $TASK_PATH."
        )
    return path


def _validate_task_project(task_path: Path) -> None:
    """Parse the task contract before any background process is created."""
    task_spec = task_path / "task.yaml"
    if not task_spec.is_file():
        raise StartError(f"task project is missing required file: {task_spec}")
    try:
        loaded = load_task_spec(task_spec)
    except (OSError, TypeError, ValueError) as exc:
        raise StartError(f"task project validation failed: {exc}") from exc
    _warn_unwired_baseline_assets(task_path, loaded)


def _warn_unwired_baseline_assets(task_path: Path, task_spec: object) -> None:
    """Warn when measured-looking baseline assets are not declared by the task."""

    if list(getattr(task_spec, "baselines", []) or []):
        return
    assets = [
        path
        for name in _CONVENTIONAL_BASELINE_RESULT_FILENAMES
        if _baseline_asset_has_measurement(path := task_path / "assets" / "baselines" / name)
    ]
    if not assets:
        return
    paths = ", ".join(str(path) for path in assets)
    sys.stderr.write(
        "WARNING: task.yaml declares no baselines, but parseable measured-looking "
        f"baseline data exists at: {paths}. These files are not trusted or used "
        "for comparison/promotion until verified values are explicitly declared "
        "under task.yaml:baselines.\n"
    )


def _baseline_asset_has_measurement(path: Path) -> bool:
    """Recognize compact conventional result files without trusting their values."""

    try:
        size = path.stat().st_size
        if size <= 0 or size > _MAX_BASELINE_RESULT_BYTES or not path.is_file():
            return False
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            rows = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                rows.append(json.loads(line))
            payload: object = rows
        else:
            payload = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return _has_finite_numeric_value(payload)


def _has_finite_numeric_value(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(float(value))
        except (OverflowError, ValueError):
            return False
    if isinstance(value, dict):
        return any(_has_finite_numeric_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_finite_numeric_value(item) for item in value)
    return False


def _resolve_provider_ref(raw: str | None, agent_system: str) -> str:
    """Resolve ``--model-provider`` based on agent system and env hints.

    Resolution order:

    1. Explicit ``--model-provider`` flag.
    2. ``PRAXIST_MODEL_PROVIDER_REF`` canonical env selector.
    3. ``MODEL_PROVIDER_REF`` compatibility environment alias.
    4. ``PRAXIST_LLM_PROVIDER`` short name mapped through
       :data:`PROVIDER_REF_FOR_SHORT_NAME`.
    5. Available credential default: DeepSeek, then OpenRouter, then the
       selected runtime's native provider.
    """
    if raw:
        return raw
    from_env = (
        getenv("PRAXIST_MODEL_PROVIDER_REF", "").strip() or getenv("MODEL_PROVIDER_REF", "").strip()
    )
    if from_env:
        return from_env
    short_name = getenv("PRAXIST_LLM_PROVIDER", "").strip().lower()
    if short_name:
        return PROVIDER_REF_FOR_SHORT_NAME.get(short_name, f"model_provider:{short_name}")
    if getenv("DEEPSEEK_API_KEY", ""):
        return PROVIDER_REF_FOR_SHORT_NAME["deepseek"]
    if getenv("OPENROUTER_API_KEY", ""):
        return OPENROUTER_PROVIDER_REF
    if agent_system == "codex_sdk":
        default_short = default_provider_for_agent_system(agent_system)
        return PROVIDER_REF_FOR_SHORT_NAME.get(default_short, OPENAI_PROVIDER_REF)
    return ANTHROPIC_PROVIDER_REF


def _resolve_model(raw: str | None, provider_ref: str, agent_system: str) -> str:
    """Resolve ``--model`` through explicit, environment, and provider defaults.

    Provider defaults are independent of the selected runtime so both SDK
    runtimes receive the same explicit model contract.
    """
    if raw:
        return raw
    from_env = getenv("PRAXIST_MODEL", "").strip() or getenv("MODEL", "").strip()
    if from_env:
        return from_env
    if provider_ref == OPENROUTER_PROVIDER_REF:
        return OPENROUTER_DEFAULT_MODEL
    if provider_ref == DEEPSEEK_PROVIDER_REF:
        return DEEPSEEK_DEFAULT_MODEL
    if provider_ref == ANTHROPIC_PROVIDER_REF:
        return ANTHROPIC_DEFAULT_MODEL
    return ""


def _resolve_run_dir(raw: str | None, task_path: Path, now: _dt.datetime) -> Path:
    """Resolve ``--run-dir`` through the standard CLI precedence."""
    if raw:
        return Path(raw).expanduser().resolve()
    from_env = getenv("RUN_DIR", "").strip()
    if from_env:
        return Path(from_env).expanduser().resolve()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    return task_path / "experiments" / f"run_{timestamp}_{task_path.name}"


def _precheck_credentials(provider_ref: str, runtime_ref: str | None = None) -> None:
    """Refuse to start if the active provider has no launcher-visible auth.

    Walks the inverse of :data:`PROVIDER_REF_FOR_SHORT_NAME`: from the
    resolved provider ref, recover the short name, look up the
    expected env var in :data:`PROVIDER_KEY_MAP`, and refuse if the
    var is empty. Native OpenAI through Codex defers to child startup because
    the runtime may provide saved ChatGPT authentication. Raw secrets are not
    printed.
    """
    if runtime_ref == "agent_runtime:codex_sdk" and provider_ref == OPENAI_PROVIDER_REF:
        # The child startup performs the bounded ChatGPT-login probe through
        # the selected runtime plugin. API credentials still win there.
        return
    short_name = _provider_short_name(provider_ref)
    if not short_name:
        # Unknown provider plugin ref — let the runtime fail loudly
        # rather than refusing in the launcher.  Operators using
        # custom provider plugins should know what env var they need.
        return
    env_var = PROVIDER_KEY_MAP.get(short_name)
    if env_var is None:
        return
    if not getenv(env_var, ""):
        raise StartError(
            f"provider credential missing: {env_var} must be set for "
            f"{provider_ref}. Export the key (or set $PRAXIST_LLM_PROVIDER "
            "to a provider you do have credentials for) and retry."
        )


def _provider_short_name(provider_ref: str) -> str | None:
    """Inverse of :data:`PROVIDER_REF_FOR_SHORT_NAME` (None if unknown)."""
    for short_name, ref in PROVIDER_REF_FOR_SHORT_NAME.items():
        if ref == provider_ref:
            return short_name
    return None


def _build_command(
    *,
    task_path: Path,
    run_dir: Path,
    model: str,
    provider_ref: str,
    runtime_ref: str,
    frontier_strategy: str,
    server: bool,
    resume: bool = False,
    resume_from: Path | None = None,
    resume_policy: str = "completed_generation",
) -> list[str]:
    """Build the ``python -m praxist.run run …`` argv list."""
    workspace = Path(__file__).resolve().parents[2]
    command: list[str] = [
        sys.executable,
        "-m",
        "praxist.run",
        "run",
        "--task-path",
        str(task_path),
        "--workspace",
        str(workspace),
        "--run-dir",
        str(run_dir),
        "--runtime",
        runtime_ref,
        "--model-provider",
        provider_ref,
        "--frontier-strategy",
        frontier_strategy,
    ]
    # ``--model`` is positional-free in praxist.run but the runtime
    # plugin treats an empty string as "use my default", so we only
    # emit the flag when we actually have a concrete model name.
    if model:
        command += ["--model", model]
    if resume_from is not None:
        command += ["--resume-from", str(resume_from)]
    elif resume:
        command.append("--resume")
    if resume:
        command += ["--resume-policy", resume_policy]
    if not server:
        command.append("--local")
    return command


def _build_env(
    *,
    cohort: str | None,
    generations: str | None,
    agent_system: str,
) -> dict[str, str]:
    """Build the spawn environment, layering CLI overrides on top of os.environ.

    The child reads ``PRAXIST_AGENT_SYSTEM`` from its environment when the
    research_loop stage decides how to spawn Peers; we propagate the
    resolved value so explicit ``--agent-system`` wins over any stale
    value the operator might have in their shell.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # #167: the spawn redirects the child's stdout / stderr to
    # ``<run_dir>/logs/launcher.nohup.log``. When stdout/stderr are
    # not a TTY, CPython block-buffers them in ~8 KB chunks — so
    # ``logger.info`` output accumulates and the operator's ``tail -f``
    # on the log file sees long quiet stretches followed by sudden
    # bursts. Forcing unbuffered I/O makes the log actually stream
    # while the workload runs. ``setdefault`` lets operators override
    # in either direction (e.g. set to empty to opt out).
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PRAXIST_AGENT_SYSTEM"] = agent_system
    if cohort:
        env["COHORT_SIZE"] = str(cohort)
    if generations:
        env["MAX_GENERATIONS"] = str(generations)
    return env


def _command_prefix(command: Sequence[str]) -> str:
    """Return the ``ps``-comparable prefix used by the stop-time TOCTOU guard.

    We deliberately keep this to the first three argv elements
    (``python -m praxist.run``).  ``ps`` truncates command lines on
    some platforms; using the head keeps the comparison robust while
    still being specific enough to reject an unrelated recycled PID.
    """
    return " ".join(command[:3])


def _spawn_child(
    spawn: Callable[..., subprocess.Popen[bytes]],
    command: Sequence[str],
    log_file: Path,
    env: dict[str, str],
) -> int:
    """Spawn the child process detached from the operator terminal.

    ``start_new_session=True`` is the cross-platform equivalent of
    ``nohup setsid``: the child becomes its own session/process-group
    leader, so the operator's terminal closing does not deliver SIGHUP.
    ``stdin`` is connected to ``/dev/null`` to break TTY inheritance.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("ab") as log_handle, open(os.devnull, "rb") as devnull:
        proc = spawn(
            list(command),
            stdin=devnull,
            stdout=log_handle,
            stderr=log_handle,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    return int(proc.pid)


def _default_spawn(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
    """Default spawn callable; thin wrapper so tests can substitute."""
    return subprocess.Popen(*args, **kwargs)  # type: ignore[call-overload]  # pragma: no cover - production-only path; tests inject their own spawn


# Bytes prefix the grandchild writes to the status pipe on a failed
# pre-exec setup. The original parent treats it as a fatal launch
# error; absent (clean) status data is parsed as the workload PID.
_DAEMON_ERR_PREFIX = b"ERR:"


def _spawn_daemonized(
    command: Sequence[str],
    log_file: Path,
    env: Mapping[str, str],
) -> int:
    """Double-fork + ``execvpe`` so the workload survives a sandboxed launcher.

    Issue #99 follow-up: ``Popen(start_new_session=True)`` alone is not
    enough when the launching process tree is reaped by an outer
    sandbox (agent tool shells, CI runners, Docker ``--init``). The
    classic UNIX double-fork
    re-parents the workload to PID 1 so the sandbox's reaper can't
    reach it.

    Lineage on success::

        original parent (``praxist start``)
          │ fork()
          ├── first child  → setsid()
          │     │ fork()
          │     ├── middle child  → _exit(0)
          │     └── grandchild  → execvpe() → workload (same PID)
          └── reads PID from pipe, returns

    Status pipe contract: the grandchild writes ``<pid>\\n`` to the
    pipe just before ``execvpe``.  Any pre-exec failure writes
    ``ERR:<message>\\n`` instead.  The original parent reads from the
    pipe, raises :class:`StartError` on an ``ERR:`` payload, or
    returns the parsed PID otherwise.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Windows-specific path
        raise StartError("--daemonize requires a POSIX host") from exc

    read_fd, write_fd = os.pipe()

    pid = os.fork()
    if pid > 0:
        # Original parent — read the daemonized workload PID and reap
        # the first child (which exits almost immediately after its
        # own second fork).
        os.close(write_fd)
        try:
            with os.fdopen(read_fd, "rb") as pipe:
                data = pipe.read()
        except OSError as exc:
            raise StartError(f"daemonize: pipe read failed: {exc}") from exc
        with contextlib.suppress(OSError):  # pragma: no cover - middle child already reaped
            os.waitpid(pid, 0)
        if not data:
            raise StartError("daemonize: grandchild exited before reporting workload PID")
        # The grandchild reports ``<pid>\n`` before ``execvpe``. On
        # exec success ``FD_CLOEXEC`` on ``write_fd`` closes the pipe,
        # so the parent sees only the PID line. On exec failure the
        # grandchild appends an ``ERR:<msg>\n`` line — scan for it
        # rather than only checking ``startswith``, because the PID
        # line lands first.
        text = data.decode(errors="replace")
        for line in text.splitlines():
            if line.startswith(_DAEMON_ERR_PREFIX.decode()):
                message = line[len(_DAEMON_ERR_PREFIX) :].strip()
                raise StartError(f"daemonize: {message}")
        first_line = text.splitlines()[0] if text else ""
        try:
            return int(first_line.strip())
        except ValueError as exc:
            raise StartError(f"daemonize: invalid PID report {data!r}") from exc

    # First child — become session leader and second-fork.
    os.close(read_fd)
    try:
        os.setsid()
        middle_pid = os.fork()
        if middle_pid > 0:
            # Middle child exits immediately, leaving the grandchild
            # orphaned to PID 1.
            os._exit(0)

        # Grandchild — set up stdio, report PID, exec the workload.
        os.chdir("/")
        os.umask(0o022)

        log_fd = os.open(
            str(log_file),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        null_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null_fd, 0)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        os.close(null_fd)
        os.close(log_fd)

        # Report PID (which becomes the workload's PID after exec)
        # to the original parent BEFORE the exec attempt.
        os.write(write_fd, f"{os.getpid()}\n".encode())
        # Set ``FD_CLOEXEC`` on ``write_fd`` so a successful exec
        # auto-closes it → parent reads EOF and returns the PID.
        # A failing exec keeps write_fd open inside the ``except``
        # below so we can append an ``ERR:`` line for the parent.
        flags = fcntl.fcntl(write_fd, fcntl.F_GETFD)
        fcntl.fcntl(write_fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)

        os.execvpe(command[0], list(command), dict(env))
    except BaseException as exc:  # pragma: no cover - reached only on exec failure
        try:
            # Clear CLOEXEC so the write actually reaches the parent.
            cur_flags = fcntl.fcntl(write_fd, fcntl.F_GETFD)
            fcntl.fcntl(write_fd, fcntl.F_SETFD, cur_flags & ~fcntl.FD_CLOEXEC)
            os.write(write_fd, _DAEMON_ERR_PREFIX + f"{exc}\n".encode())
            os.close(write_fd)
        except OSError:
            pass
        os._exit(2)
    # Unreachable: execvpe replaces the process image on success and
    # the except branch above calls os._exit on failure.
    os._exit(0)  # pragma: no cover - belt-and-braces; see above


def _write_operator_hint(entry: RegistryEntry) -> None:
    """Print the launch summary on stderr (decoration) per the CLI contract."""
    agent_system = entry.extra.get("agent_system", "")
    daemonized = entry.extra.get("daemonized") == "1"
    if entry.state == STATE_STOPPED:
        heading = "=== Praxist run stopped during startup ==="
    else:
        heading = {
            "running": "=== Praxist run launched ===",
            "completed": "=== Praxist run completed during startup ===",
            "failed": "=== Praxist run failed during startup ===",
        }.get(
            entry.extra.get("startup_state", "running"),
            "=== Praxist run initialization pending ===",
        )
    sys.stderr.write(heading + "\n")
    sys.stderr.write(f"run_id       : {operator_text(entry.run_id)}\n")
    sys.stderr.write(f"pid          : {entry.pid}\n")
    sys.stderr.write(f"task_path    : {operator_text(entry.task_path)}\n")
    sys.stderr.write(f"run_dir      : {operator_text(entry.run_dir)}\n")
    sys.stderr.write(f"log_file     : {operator_text(entry.log_file)}\n")
    if agent_system:
        sys.stderr.write(f"agent_system : {operator_text(agent_system)}\n")
    sys.stderr.write(f"runtime      : {operator_text(entry.runtime_ref)}\n")
    sys.stderr.write(f"model        : {operator_text(entry.model or '(runtime default)')}\n")
    sys.stderr.write(f"provider     : {operator_text(entry.model_provider_ref)}\n")
    if daemonized:
        sys.stderr.write("daemonize    : on (double-fork; workload re-parented to init)\n")
    # #167: surface the two log streams the operator actually wants.
    # ``launcher.nohup.log`` carries the orchestrator's stdout/stderr
    # (INFO logging, warnings, the synthesis trigger heartbeat). For
    # control-plane events (per-peer sessions, finding promotions,
    # generation transitions) ``trajectory.jsonl`` is the canonical
    # replayable feed — it's append-only and structured, so
    # ``tail -f | jq`` is the right shape.
    trajectory_path = f"{entry.run_dir}/trajectory.jsonl"
    sys.stderr.write("\nMonitor with:\n")
    sys.stderr.write(
        f"  {operator_text(shlex.join(('tail', '-f', entry.log_file)))}"
        "                # orchestrator stdout/stderr\n"
    )
    sys.stderr.write(
        f"  {operator_text(shlex.join(('tail', '-f', trajectory_path)))}"
        "    # control-plane events (JSONL)\n"
    )
    sys.stderr.write(
        "  praxist status                                                  # run state\n"
    )
    sys.stderr.write(
        f"  {operator_text(monitor_command(entry.run_id))}"
        "              # fullscreen read-only TUI\n"
    )
    sys.stderr.write("Stop with:\n")
    sys.stderr.write(f"  {operator_text(shlex.join(('praxist', 'stop', entry.run_id)))}\n")
    sys.stdout.write(f"{entry.run_id}\n")


def _wait_for_startup(
    entry: RegistryEntry,
    timeout: float,
    baseline: dict[str, tuple[int, int, int, int] | None],
) -> RegistryEntry:
    """Wait until this launch enters the research-loop workflow stage."""
    deadline = time.monotonic() + max(0.0, timeout)
    run_dir = Path(entry.run_dir)
    run_json = run_dir / "run.json"
    while time.monotonic() < deadline:
        if _startup_failed(run_dir, baseline):
            failed = _persist_startup_state(entry, "failed")
            return failed
        if _startup_stage_started(run_dir, baseline):
            state = _state_from_run_json(run_json)
            updated = _persist_startup_state(entry, state)
            return updated
        if not _pid_is_alive(entry.pid):
            failed = _persist_startup_state(entry, "failed")
            if failed.state == STATE_STOPPED:
                return failed
            detail = _tail_text(Path(entry.log_file), limit=8)
            suffix = f"\nLast launcher log lines:\n{detail}" if detail else ""
            raise StartError(
                "background process exited before startup completed; "
                f"inspect {entry.log_file}.{suffix}"
            )
        time.sleep(0.2)
    return _current_launch_entry(entry)


def _state_from_run_json(path: Path) -> str:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return STATE_RUNNING
    status = str(raw.get("status") or "").strip().lower()
    if status in {"succeeded", "completed"}:
        return "completed"
    if status in {"failed", "error"}:
        return "failed"
    return "running"


def _with_startup_state(entry: RegistryEntry, state: str) -> RegistryEntry:
    return replace(entry, extra={**entry.extra, "startup_state": state})


def _same_launch_instance(expected: RegistryEntry, current: RegistryEntry) -> bool:
    """Match one launch while retaining schema-v1 PID-only compatibility."""

    if current.pid != expected.pid:
        return False
    expected_token = expected.extra.get("process_start_token", "").strip()
    current_token = current.extra.get("process_start_token", "").strip()
    return not (expected_token and current_token) or expected_token == current_token


def _persist_startup_state(entry: RegistryEntry, state: str) -> RegistryEntry:
    """Update startup state without reviving a concurrently stopped run."""

    try:
        with entry_lock(entry.run_id):
            current = read_entry(entry.run_id)
            if not _same_launch_instance(entry, current):
                raise StartError(
                    f"run registry controller changed during startup for {entry.run_id!r}"
                )
            if current.state != STATE_RUNNING:
                return current
            updated = _with_startup_state(current, state)
            write_entry(updated)
            return updated
    except RegistryError as exc:
        raise StartError(str(exc)) from exc


def _current_launch_entry(entry: RegistryEntry) -> RegistryEntry:
    """Return current registry truth for this launch after a bounded wait."""

    try:
        with entry_lock(entry.run_id):
            current = read_entry(entry.run_id)
    except RegistryError as exc:
        raise StartError(str(exc)) from exc
    if not _same_launch_instance(entry, current):
        raise StartError(f"run registry controller changed during startup for {entry.run_id!r}")
    return current


def _startup_artifact_signatures(
    run_dir: Path,
) -> dict[str, tuple[int, int, int, int] | None]:
    return {
        name: _file_signature(run_dir / name)
        for name in ("run.json", "startup_config.json", "trajectory.jsonl")
    }


def _startup_failed(
    run_dir: Path,
    baseline: dict[str, tuple[int, int, int, int] | None],
) -> bool:
    run_json = run_dir / "run.json"
    observed = _file_signature(run_json)
    if observed is None or observed == baseline.get("run.json"):
        return False
    return _state_from_run_json(run_json) == "failed"


def _startup_stage_started(
    run_dir: Path,
    baseline: dict[str, tuple[int, int, int, int] | None],
) -> bool:
    """Return True when this launch appended the workflow-start event."""
    path = run_dir / "trajectory.jsonl"
    baseline_signature = baseline.get("trajectory.jsonl")
    try:
        observed = path.stat()
        offset = 0
        if baseline_signature is not None and observed.st_ino == baseline_signature[2]:
            offset = baseline_signature[3]
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            lines = handle.readlines()
    except OSError:
        return False
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("kind") == "workflow.stage_started":
            return True
    return False


def _file_signature(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_size


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _tail_text(path: Path, *, limit: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:])


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_finite_float(value: str) -> float:
    import math

    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def monitor_command(run_id: str) -> str:
    """Return a shell-safe command that opens the monitor for ``run_id``."""

    return shlex.join(("praxist", "--monitor", "--run-id", run_id))


def operator_text(value: object) -> str:
    """Return terminal-safe text for human-facing CLI decorations."""

    output: list[str] = []
    for char in str(value):
        codepoint = ord(char)
        if char in "\t\r\n":
            output.append(" ")
        elif (
            codepoint < 0x20
            or 0x7F <= codepoint <= 0x9F
            or codepoint in {0x061C, 0x200E, 0x200F}
            or 0x202A <= codepoint <= 0x202E
            or 0x2066 <= codepoint <= 0x2069
        ):
            continue
        else:
            output.append(char)
    return "".join(output)


def _is_ignorable_precreated_path(path: Path) -> bool:
    if path.is_file():
        return path.name in {".DS_Store", ".gitkeep"}
    if path.is_dir():
        if path.name == "logs":
            return all(child.name in {".gitkeep", "launcher.nohup.log"} for child in path.iterdir())
        return not any(path.iterdir())
    return False
