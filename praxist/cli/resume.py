"""``praxist resume`` - continue an interrupted Praxist research run.

This subcommand is a thin lifecycle wrapper around ``praxist start``.  It
recovers the original run settings from either the Praxist registry or a
run directory's ``startup_config.json``, then launches
``python -m praxist.run run --resume-from <run_dir>`` through the
same registry-backed launcher used by ``praxist start``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from praxist.cli import start
from praxist.cli._env import agent_system_for_runtime_ref
from praxist.cli._setup_common import load_cli_environment, selected_config_file
from praxist.cli.registry import (
    STATE_STOPPED,
    RegistryEntry,
    RegistryError,
    entry_is_local,
    entry_lock,
    entry_path,
    entry_process_epoch_matches,
    list_entries,
    process_identity_matches,
    read_entry,
    registry_lock,
)
from praxist.cli.status import pid_is_alive, read_ps_table, registry_command_matches
from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
    ensure_resumable_run_dir,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static type checkers
    from collections.abc import Callable


class ResumeError(RuntimeError):
    """Raised when ``praxist resume`` cannot determine a safe resume target."""


@dataclass(frozen=True)
class ResumeTarget:
    """Resolved target and launch defaults for a resumed run."""

    run_dir: Path
    task_path: str
    model: str | None = None
    model_provider_ref: str | None = None
    runtime_ref: str | None = None
    frontier_strategy: str = start.DEFAULT_FRONTIER_STRATEGY
    agent_system: str | None = None
    cohort: str | None = None
    generations: str | None = None
    server: bool = False
    source: str = "run_dir"
    source_run_id: str = ""
    codex_native: bool = False


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``praxist resume`` subcommand on the parent parser."""

    parser = subparsers.add_parser(
        "resume",
        help="Resume an interrupted Praxist run.",
        description=(
            "Continue an existing Praxist run directory from its last safe "
            "completed generation boundary.  The target may be a registry "
            "run_id from ``praxist status`` or a direct experiments/run_* path."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        help="Registry run_id or path to an existing Praxist run directory.",
    )
    parser.add_argument(
        "--task-path",
        dest="task_path",
        default=None,
        help="Override task project path when resuming from a run directory.",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Config file to load (default: $PRAXIST_CONFIG_FILE or the user config).",
    )
    parser.add_argument(
        "--agent-system",
        dest="agent_system",
        choices=start.AGENT_SYSTEM_VALUES,
        default=None,
        help="Override agent system for the resumed launch.",
    )
    parser.add_argument(
        "--runtime",
        dest="runtime_ref",
        default=None,
        help="Override agent_runtime plugin ref.",
    )
    parser.add_argument(
        "--codex-native",
        action="store_true",
        default=None,
        help="Resume in Codex-native saved-login mode without provider API keys.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="Override model name.",
    )
    parser.add_argument(
        "--model-provider",
        dest="model_provider_ref",
        default=None,
        help="Override model_provider plugin ref.",
    )
    parser.add_argument(
        "--strategy",
        dest="frontier_strategy",
        choices=("auto", "mixed", "explore", "exploit"),
        default=None,
        help="Override frontier strategy.",
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        type=start._positive_int,
        default=None,
        help="Cohort size override (exported as COHORT_SIZE).",
    )
    parser.add_argument(
        "--generations",
        dest="generations",
        type=start._positive_int,
        default=None,
        help="Maximum generations override (exported as MAX_GENERATIONS).",
    )
    parser.add_argument(
        "--server",
        dest="server",
        action="store_true",
        default=None,
        help="Disable --local mode (server mode).",
    )
    parser.add_argument(
        "--daemonize",
        dest="daemonize",
        action="store_true",
        help="Use the same double-fork daemon launch path as praxist start.",
    )
    parser.add_argument(
        "--resume-policy",
        dest="resume_policy",
        default="completed_generation",
        choices=["completed_generation"],
        help="Resume policy forwarded to praxist.run.",
    )
    parser.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help=(
            "Allow resume only when an old registry entry's process ownership "
            "cannot be verified. It never overrides a verified live controller."
        ),
    )
    parser.add_argument(
        "--startup-timeout",
        type=start._nonnegative_finite_float,
        default=start.DEFAULT_STARTUP_TIMEOUT_SECONDS,
        help="Seconds to wait for resume startup artifacts.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit one JSON document on stdout instead of the operator summary.",
    )
    parser.set_defaults(func=cmd_resume)


def cmd_resume(args: argparse.Namespace) -> int:
    """Handler for ``praxist resume``."""

    try:
        entry = resume_run(
            target=args.target,
            task_path=args.task_path,
            agent_system=args.agent_system,
            runtime_ref=args.runtime_ref,
            model=args.model,
            model_provider_ref=args.model_provider_ref,
            frontier_strategy=args.frontier_strategy,
            cohort=args.cohort,
            generations=args.generations,
            server=args.server,
            daemonize=args.daemonize,
            resume_policy=args.resume_policy,
            force=args.force,
            startup_timeout=args.startup_timeout,
            config_file=selected_config_file(args.config_file),
            codex_native=args.codex_native,
        )
    except (ResumeError, start.StartError) as exc:
        sys.stderr.write(f"praxist resume: {exc}\n")
        return 1

    if args.as_json:
        sys.stdout.write(json.dumps(entry.to_dict(), indent=2) + "\n")
    else:
        _write_resume_hint(entry)
    return 1 if entry.state == STATE_STOPPED or entry.extra.get("startup_state") == "failed" else 0


def resume_run(
    *,
    target: str,
    task_path: str | None = None,
    agent_system: str | None = None,
    runtime_ref: str | None = None,
    model: str | None = None,
    model_provider_ref: str | None = None,
    frontier_strategy: str | None = None,
    cohort: str | None = None,
    generations: str | None = None,
    server: bool | None = None,
    daemonize: bool = False,
    resume_policy: str = "completed_generation",
    force: bool = False,
    startup_timeout: float = 0.0,
    config_file: Path | None = None,
    codex_native: bool | None = None,
    spawn: Callable[..., subprocess.Popen[bytes]] | None = None,
    daemon_spawn: Callable[..., int] | None = None,
) -> RegistryEntry:
    """Resolve ``target`` and launch a resumed run."""

    initial = resolve_resume_target(target, force=force)
    try:
        with registry_lock(), entry_lock(initial.run_dir.name):
            entry, startup_baseline, consumed_shutdown_fence = _resume_locked(
                target=target,
                task_path=task_path,
                agent_system=agent_system,
                runtime_ref=runtime_ref,
                model=model,
                model_provider_ref=model_provider_ref,
                frontier_strategy=frontier_strategy,
                cohort=cohort,
                generations=generations,
                server=server,
                daemonize=daemonize,
                resume_policy=resume_policy,
                force=force,
                startup_timeout=startup_timeout,
                config_file=config_file,
                codex_native=codex_native,
                spawn=spawn,
                daemon_spawn=daemon_spawn,
            )
        if spawn is None and daemon_spawn is None and startup_timeout > 0:
            try:
                entry = start._wait_for_startup(entry, startup_timeout, startup_baseline)
            except BaseException:
                _restore_consumed_shutdown_fence(consumed_shutdown_fence)
                raise
            if entry.state == STATE_STOPPED or entry.extra.get("startup_state") == "failed":
                _restore_consumed_shutdown_fence(consumed_shutdown_fence)
        return entry
    except (RegistryError, ValueError) as exc:
        raise ResumeError(str(exc)) from exc


def _resume_locked(
    *,
    target: str,
    task_path: str | None,
    agent_system: str | None,
    runtime_ref: str | None,
    model: str | None,
    model_provider_ref: str | None,
    frontier_strategy: str | None,
    cohort: str | None,
    generations: str | None,
    server: bool | None,
    daemonize: bool,
    resume_policy: str,
    force: bool,
    startup_timeout: float,
    config_file: Path | None,
    codex_native: bool | None,
    spawn: Callable[..., subprocess.Popen[bytes]] | None,
    daemon_spawn: Callable[..., int] | None,
) -> tuple[
    RegistryEntry,
    dict[str, tuple[int, int, int, int] | None],
    tuple[Path, bytes] | None,
]:
    """Revalidate and launch while the run lifecycle lock is held."""

    resolved = resolve_resume_target(target, force=force)
    if (
        codex_native is True
        and not resolved.codex_native
        and (
            resolved.runtime_ref != "agent_runtime:codex_sdk"
            or resolved.model_provider_ref != start.OPENAI_PROVIDER_REF
        )
    ):
        raise ResumeError(
            "--codex-native cannot change an existing run's canonical "
            "runtime or model provider. Start a new Codex-native run instead."
        )
    launch_task_path = task_path or resolved.task_path
    if not launch_task_path:
        raise ResumeError(
            "could not infer task project path; pass --task-path or resume by registry run_id"
        )
    load_cli_environment(
        Path(launch_task_path).expanduser().resolve(),
        config_file=config_file,
    )
    effective_cohort = cohort if cohort is not None else resolved.cohort
    effective_generations = generations if generations is not None else resolved.generations
    effective_server = resolved.server if server is None else server
    effective_agent_system, effective_runtime = _resume_runtime_selection(
        agent_system=agent_system,
        runtime_ref=runtime_ref,
        inherited_agent_system=resolved.agent_system,
        inherited_runtime_ref=resolved.runtime_ref,
    )
    effective_codex_native = resolved.codex_native if codex_native is None else codex_native
    startup_baseline = start._startup_artifact_signatures(resolved.run_dir)
    try:
        consumed_shutdown_fence = start._consume_shutdown_fence(resolved.run_dir)
    except OSError as exc:
        raise ResumeError(f"could not reopen stopped run {resolved.run_dir}: {exc}") from exc
    try:
        entry = start.launch_run(
            task_path=launch_task_path,
            agent_system=effective_agent_system,
            runtime_ref=effective_runtime,
            run_dir=str(resolved.run_dir),
            model=model if model is not None else resolved.model,
            model_provider_ref=(
                model_provider_ref
                if model_provider_ref is not None
                else resolved.model_provider_ref
            ),
            frontier_strategy=frontier_strategy or resolved.frontier_strategy,
            cohort=effective_cohort,
            generations=effective_generations,
            server=effective_server,
            daemonize=daemonize,
            resume=True,
            resume_from=str(resolved.run_dir),
            resume_policy=resume_policy,
            spawn=spawn,
            daemon_spawn=daemon_spawn,
            startup_timeout=startup_timeout,
            _lifecycle_locked=True,
            _defer_startup_wait=True,
            codex_native=effective_codex_native,
        )
    except BaseException:
        _restore_consumed_shutdown_fence(consumed_shutdown_fence)
        raise
    return entry, startup_baseline, consumed_shutdown_fence


def _restore_consumed_shutdown_fence(fence: tuple[Path, bytes] | None) -> None:
    """Restore a consumed stop fence without overwriting a newer stop."""

    if fence is None:
        return
    try:
        start._restore_shutdown_fence(fence)
    except OSError as exc:
        path, _payload = fence
        raise ResumeError(f"could not restore shutdown fence at {path}: {exc}") from exc


def resolve_resume_target(target: str, *, force: bool = False) -> ResumeTarget:
    """Resolve ``target`` as a registry run_id or run directory."""

    raw = str(target or "").strip()
    if not raw:
        raise ResumeError("expected <run_id-or-run-dir>")
    entry = _registry_entry_for_target(raw)
    if entry is not None:
        _validate_resume_entry(entry, force=force)
        return _target_from_registry(entry)
    run_dir = Path(raw).expanduser().resolve()
    matching = _registry_entries_for_run_dir(run_dir)
    if len(matching) > 1:
        raise ResumeError(
            f"multiple registry entries refer to {run_dir}; resume by an explicit run_id"
        )
    if matching:
        _validate_resume_entry(matching[0], force=force)
        return _target_from_registry(matching[0])
    try:
        same_id_entry = read_entry(run_dir.name)
    except RegistryError as exc:
        if entry_path(run_dir.name).exists():
            raise ResumeError(
                f"run id {run_dir.name!r} already has an unreadable registry entry: {exc}"
            ) from exc
        same_id_entry = None
    if same_id_entry is not None:
        existing_dir = Path(same_id_entry.run_dir).expanduser().resolve()
        if existing_dir != run_dir:
            raise ResumeError(
                f"run id {run_dir.name!r} is already registered for {existing_dir}; "
                "choose the registered run or rename the unregistered run directory"
            )
    return _target_from_run_dir(run_dir)


def _registry_entry_for_target(raw: str) -> RegistryEntry | None:
    if "/" in raw or raw.startswith(".") or raw.startswith("~"):
        return None
    try:
        return read_entry(raw)
    except (RegistryError, ValueError):
        return None


def _registry_entries_for_run_dir(run_dir: Path) -> list[RegistryEntry]:
    wanted = run_dir.expanduser().resolve()
    matches: list[RegistryEntry] = []
    for entry in list_entries():
        try:
            candidate = Path(entry.run_dir).expanduser().resolve()
        except OSError:
            continue
        if candidate == wanted:
            matches.append(entry)
    return matches


def _validate_resume_entry(entry: RegistryEntry, *, force: bool) -> None:
    if entry_is_local(entry) is False:
        raise ResumeError(
            f"run {entry.run_id!r} belongs to "
            f"{entry.extra.get('hostname', 'another host')!r}; resume it there"
        )
    liveness = _entry_liveness(entry)
    if liveness == "verified-live":
        force_note = " --force cannot bypass this check." if force else ""
        raise ResumeError(
            f"run {entry.run_id!r} still appears to be running; stop it first.{force_note}"
        )
    if liveness == "unknown" and not force:
        raise ResumeError(
            f"run {entry.run_id!r} still appears to be running, but its process "
            "ownership cannot be verified; stop it first or pass --force "
            "after confirming no controller is active"
        )


def _target_from_registry(entry: RegistryEntry) -> ResumeTarget:
    run_dir = Path(entry.run_dir).expanduser().resolve()
    _ensure_resumable(run_dir)
    auth_mode = entry.extra.get("auth_mode", "").strip()
    return ResumeTarget(
        run_dir=run_dir,
        task_path=entry.task_path,
        model=entry.model or None,
        model_provider_ref=entry.model_provider_ref or None,
        runtime_ref=entry.runtime_ref or None,
        frontier_strategy=_frontier_strategy_from_entry(entry),
        agent_system=entry.extra.get("agent_system") or None,
        cohort=entry.extra.get("cohort") or None,
        generations=entry.extra.get("generations") or None,
        server=entry.extra.get("server") == "1",
        source="registry",
        source_run_id=entry.run_id,
        codex_native=(
            auth_mode == "codex-native"
            or (not auth_mode and _run_dir_used_codex_native(run_dir, {}))
        ),
    )


def _target_from_run_dir(run_dir: Path) -> ResumeTarget:
    _ensure_resumable(run_dir)
    startup_config = _read_json_object(run_dir / "startup_config.json")
    canonical_args = startup_config.get("canonical_args")
    canonical_args = canonical_args if isinstance(canonical_args, dict) else {}
    task_project = startup_config.get("task_project")
    task_project = task_project if isinstance(task_project, dict) else {}
    task_path = str(canonical_args.get("task_path") or task_project.get("path") or "").strip()
    if "server" in canonical_args:
        server = _optional_bool(canonical_args.get("server"), default=False)
    else:
        resume_identity = startup_config.get("resume_identity")
        resume_identity = resume_identity if isinstance(resume_identity, dict) else {}
        local_mode = startup_config.get("local_mode", resume_identity.get("local_mode", True))
        server = not _optional_bool(local_mode, default=True)
    return ResumeTarget(
        run_dir=run_dir,
        task_path=task_path,
        model=_optional_str(canonical_args.get("model")),
        model_provider_ref=_optional_str(canonical_args.get("model_provider")),
        runtime_ref=_optional_str(canonical_args.get("runtime")),
        frontier_strategy=(
            _optional_str(canonical_args.get("frontier_strategy"))
            or start.DEFAULT_FRONTIER_STRATEGY
        ),
        cohort=_optional_str(canonical_args.get("cohort")),
        generations=_optional_str(canonical_args.get("generations")),
        server=server,
        source="run_dir",
        codex_native=_run_dir_used_codex_native(run_dir, canonical_args),
    )


def _entry_appears_live(entry: RegistryEntry) -> bool:
    return _entry_liveness(entry) != "stale"


def _entry_liveness(entry: RegistryEntry) -> str:
    """Return ``verified-live``, ``stale``, or ``unknown`` for resume safety."""

    if entry_process_epoch_matches(entry) is False:
        return "stale"
    identity = process_identity_matches(entry)
    if identity is False:
        return "stale"
    if identity is True:
        return "verified-live" if pid_is_alive(entry.pid) else "stale"
    ps_rows = read_ps_table()
    live = ps_rows.get(entry.pid)
    if live is not None:
        _ppid, _etime, command = live
        if not registry_command_matches(entry, command):
            return "stale"
        return "unknown"
    if not pid_is_alive(entry.pid):
        return "stale"
    return "unknown"


def _resume_runtime_selection(
    *,
    agent_system: str | None,
    runtime_ref: str | None,
    inherited_agent_system: str | None,
    inherited_runtime_ref: str | None,
) -> tuple[str | None, str | None]:
    """Apply explicit resume overrides without reviving conflicting defaults."""

    if runtime_ref is not None:
        runtime_agent = agent_system_for_runtime_ref(runtime_ref)
        return agent_system or runtime_agent or inherited_agent_system, runtime_ref
    if agent_system is not None:
        return agent_system, None
    return inherited_agent_system, inherited_runtime_ref


def _frontier_strategy_from_entry(entry: RegistryEntry) -> str:
    command = list(entry.command)
    for index, value in enumerate(command):
        if value == "--frontier-strategy" and index + 1 < len(command):
            return command[index + 1]
        if value.startswith("--frontier-strategy="):
            return value.split("=", 1)[1]
    return start.DEFAULT_FRONTIER_STRATEGY


def _ensure_resumable(run_dir: Path) -> None:
    try:
        ensure_resumable_run_dir(run_dir)
    except ValueError as exc:
        raise ResumeError(str(exc)) from exc


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeError(
            f"could not read resume artifact {path}: {exc}. "
            "Manual inspection is required to verify or repair run artifacts before resuming."
        ) from exc
    if not isinstance(payload, dict):
        raise ResumeError(
            f"resume artifact must be a JSON object: {path}. "
            "Manual inspection is required to verify or repair run artifacts before resuming."
        )
    return payload


def _run_dir_used_codex_native(
    run_dir: Path,
    canonical_args: dict[str, object],
) -> bool:
    """Recover saved-login intent from canonical startup evidence.

    Newer launch metadata may persist the mode directly. Existing runs already
    record the redacted runtime-managed ChatGPT credential, which is sufficient
    to preserve that mode without adding another artifact.
    """

    if "codex_native" in canonical_args:
        return _optional_bool(canonical_args.get("codex_native"), default=False)
    try:
        payload = json.loads((run_dir / "credentials_redacted.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    profiles = payload.get("credential_profiles") if isinstance(payload, dict) else None
    if not isinstance(profiles, list):
        return False
    return any(
        isinstance(profile, dict)
        and profile.get("provider") == "openai_compatible"
        and profile.get("source") == "runtime_session"
        and ":chatgpt:" in str(profile.get("key_id") or "")
        for profile in profiles
    )


def _optional_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_bool(value: object, *, default: bool) -> bool:
    """Decode booleans persisted by older string- or JSON-based launchers."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _write_resume_hint(entry: RegistryEntry) -> None:
    failed = entry.extra.get("startup_state") == "failed"
    heading = (
        "=== Praxist resume failed during startup ===" if failed else "=== Praxist run resumed ==="
    )
    sys.stderr.write(heading + "\n")
    sys.stderr.write(f"run_id:  {start.operator_text(entry.run_id)}\n")
    sys.stderr.write(f"pid:     {entry.pid}\n")
    sys.stderr.write(f"run_dir: {start.operator_text(entry.run_dir)}\n")
    sys.stderr.write(f"log:     {start.operator_text(entry.log_file)}\n")
    if failed:
        sys.stderr.write(
            "next:    inspect the launcher log and repair the reported startup error\n"
        )
    else:
        sys.stderr.write(f"watch:   {start.operator_text(start.monitor_command(entry.run_id))}\n")
        sys.stderr.write("next:    praxist status\n")
