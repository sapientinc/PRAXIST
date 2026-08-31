"""``praxist status`` — list known Praxist experiment runs.

CLI lifecycle Phase 2.5: merges the structured run registry written by
``praxist start`` with the cross-platform ``ps``-scan fallback inherited
from Phase 1.  Rows surface a ``source`` tag so operators can tell
where each row came from:

* ``registry`` — the run was launched by ``praxist start`` and the PID is
  still alive (validated lazily via ``kill -0``).
* ``ps-only`` — a matching live process that has no registry entry. This
  surfaces direct Python invocations or external tooling that bypasses
  ``praxist start``.
* ``stale`` — a registry entry whose recorded PID is no longer
  running (OOM, ``kill -9``, host reboot).  Entries are not rewritten
  to disk by ``praxist status``; explicit cleanup goes through
  ``praxist stop --gc`` (#166).

Output discipline (per :mod:`praxist.cli` package docstring): data
goes to ``stdout``, decorations / hints go to ``stderr``.  Default
output is a plain-text table; ``--json`` emits one JSON document on
stdout for downstream tooling.

The pattern set ``_PRAXIST_CONTROLLER_PATTERNS`` is shared conceptually with the
``praxist stop --all --ps-scan-only`` fallback so the read-side (status) and
bulk stop path agree on what counts as a Praxist run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from praxist.cli.registry import (
    STATE_RUNNING,
    STATE_STALE,
    STATE_STOPPED,
    RegistryEntry,
    entry_is_local,
    entry_process_epoch_matches,
    iter_entries_with_errors,
    process_identity_matches,
)
from praxist.plugins.workflow_stages.research_loop.backend.orchestrator_status import (
    read_effective_orchestrator_status,
)
from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
    PeerHealthSnapshot,
    collect_peer_memory_health,
)
from praxist.task_spec import TaskSpec, load_task_spec

# A process-table scan can establish ownership only for an explicit top-level
# Praxist controller. Task evaluators, trainers, agent runtimes, and research-loop
# children may also belong to unrelated projects on a shared host. Their
# ownership is established through registry ancestry or scheduler receipts.
_PRAXIST_CONTROLLER_PATTERNS = (
    r"^(?:\S*/)?python(?:\d+(?:\.\d+)*)?(?:\s+-u)?\s+-m\s+"
    r"praxist\.run\s+run(?:\s|$)",
)

SOURCE_REGISTRY = "registry"
SOURCE_PS_ONLY = "ps-only"
SOURCE_STALE = "stale"
SOURCE_REMOTE = "remote"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
STATE_INCONSISTENT = "status_inconsistent"

_RUN_DIR_HINT_RE = re.compile(r"--run-dir(?:[=\s]+)(\S+)")
_LAST_PS_ERROR = ""


@dataclass(frozen=True)
class StatusRow:
    """One row of the merged status view.

    Fields populated only by the registry source (``run_id``,
    ``task_path``, ``model``, ``model_provider_ref``) are ``None`` for
    ``ps-only`` rows.  ``etime`` and ``ppid`` come from the live ``ps``
    table when the process is alive and are ``"-"`` / ``0`` for stale
    registry entries.

    #165 progress fields (``generation`` / ``findings_total`` /
    ``updated_at``) are sourced from
    ``<run_dir>/orchestrator_status.json`` — the periodic snapshot the
    research_loop status writer emits. When the JSON is missing,
    unreadable, or malformed, the fields are ``None`` and render as
    ``"-"``. Stale rows surface the last-known snapshot so the
    operator can see where the run got to before it died.
    """

    pid: int
    ppid: int
    etime: str
    command: str
    run_dir: str | None
    source: str
    state: str
    run_id: str | None = None
    task_path: str | None = None
    model: str | None = None
    model_provider_ref: str | None = None
    started_at: str | None = None
    # #165 progress columns.
    generation: int | None = None
    findings_total: int | None = None
    updated_at: str | None = None
    peer_health_summary: dict[str, int] | None = None
    peers: list[dict[str, object]] = field(default_factory=list)
    extras: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``praxist status`` subcommand on the parent parser."""
    parser = subparsers.add_parser(
        "status",
        help="List known Praxist experiment runs.",
        description=(
            "Merge the run registry written by ``praxist start`` with a "
            "cross-platform ``ps`` scan to list every Praxist run the operator "
            "should know about.\n\n"
            "Rows are tagged with their source: ``registry`` (managed run, "
            "PID alive), ``ps-only`` (matching process without a registry "
            "entry — e.g. started through a direct Python invocation), or "
            "``stale`` (registry entry whose PID is gone)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit one JSON document on stdout instead of the plain-text table.",
    )
    parser.add_argument("--run-id", default=None, help="Show only this registry run id.")
    parser.add_argument("--task-path", default=None, help="Show runs for this task directory.")
    parser.add_argument("--active", action="store_true", help="Show only live local runs.")
    parser.add_argument("--latest", action="store_true", help="Show only the newest matching run.")
    parser.set_defaults(func=cmd_status)


def cmd_status(args: argparse.Namespace) -> int:
    """Handler for ``praxist status``."""
    errors: list[str] = []
    rows = collect_status_rows(errors=errors)
    rows = _filter_rows(
        rows,
        run_id=args.run_id,
        task_path=args.task_path,
        active=args.active,
        latest=args.latest,
    )
    if args.as_json:
        sys.stdout.write(json.dumps([row.to_dict() for row in rows], indent=2) + "\n")
    else:
        sys.stdout.write(format_status_table(rows))
        if not rows:
            sys.stderr.write("(no Praxist experiment processes found)\n")
    for error in errors:
        sys.stderr.write(f"praxist status: registry warning: {_terminal_safe(error)}\n")
    return 0


def collect_status_rows(
    *,
    errors: list[str] | None = None,
    include_peer_health: bool = True,
    process_probe_timeout: float = 10.0,
) -> list[StatusRow]:
    """Build the merged registry + ps-scan view.

    Order of operations:

    1. Read the registry.
    2. Read the ``ps`` table.
    3. For each registry entry: if the PID is alive, emit a
       ``source=registry`` row enriched with ``ps`` data; else emit a
       ``source=stale`` row carrying only what the registry knows.
    4. For each ``ps`` match not already claimed by a registry entry,
       emit a ``source=ps-only`` row.

    Read-only consumers may disable peer enrichment while retaining the complete
    registry and process view.
    """
    cmdline_by_pid = _read_ps_table(timeout_seconds=process_probe_timeout)
    ps_error = _LAST_PS_ERROR
    if ps_error and errors is not None:
        errors.append(ps_error)
    excluded = _self_ancestor_pids(cmdline_by_pid)
    regexes = [re.compile(pattern) for pattern in _PRAXIST_CONTROLLER_PATTERNS]

    claimed_pids: set[int] = set()
    rows: list[StatusRow] = []

    entries: list[RegistryEntry] = []
    for entry, error in iter_entries_with_errors():
        if error is not None:
            if errors is not None:
                errors.append(error)
            continue
        if entry is not None:
            entries.append(entry)

    for entry in entries:
        if entry_is_local(entry) is False:
            rows.append(_row_from_registry_remote(entry))
            continue
        if ps_error and _pid_is_alive(entry.pid):
            rows.append(_row_from_registry_unknown(entry, ps_error))
            claimed_pids.add(entry.pid)
            continue
        live = _validate_registry_pid(entry, cmdline_by_pid)
        if live is not None:
            ppid, etime, command = live
            rows.append(
                _row_from_registry_live(
                    entry,
                    ppid,
                    etime,
                    command,
                    include_peer_health=include_peer_health,
                )
            )
            claimed_pids.add(entry.pid)
        else:
            rows.append(
                _row_from_registry_stale(
                    entry,
                    include_peer_health=include_peer_health,
                )
            )

    for pid, (ppid, etime, command) in cmdline_by_pid.items():
        if pid in excluded or pid in claimed_pids:
            continue
        if not any(regex.search(command) for regex in regexes):
            continue
        ps_run_dir = _extract_run_dir(command)
        ps_gen, ps_findings, ps_updated = _read_orchestrator_progress(ps_run_dir)
        ps_peer_summary, ps_peers = _peer_health_fields(
            ps_run_dir,
            None,
            ps_gen,
            enabled=include_peer_health,
        )
        rows.append(
            StatusRow(
                pid=pid,
                ppid=ppid,
                etime=etime,
                command=command,
                run_dir=ps_run_dir,
                source=SOURCE_PS_ONLY,
                state=STATE_RUNNING,
                generation=ps_gen,
                findings_total=ps_findings,
                updated_at=ps_updated,
                peer_health_summary=ps_peer_summary,
                peers=ps_peers,
            )
        )

    rows.sort(key=lambda row: (_source_sort_key(row.source), row.pid))
    return rows


def format_status_table(rows: Iterable[StatusRow]) -> str:
    """Render ``rows`` as a plain-text table suitable for terminal display.

    #165: ``GEN`` / ``FINDINGS`` / ``UPDATED`` columns surface
    orchestrator progress so the operator can tell *at a glance*
    whether a run is making progress — not just whether the process
    is alive.
    """
    rows = list(rows)
    headers = (
        "PID",
        "SOURCE",
        "STATE",
        "AGE",
        "GEN",
        "FINDINGS",
        "PEERS",
        "UPDATED",
        "RUN_ID",
        "RUN_DIR",
        "COMMAND",
    )
    table_rows: list[tuple[str, ...]] = []
    for row in rows:
        table_rows.append(
            (
                str(row.pid),
                row.source,
                row.state,
                row.etime if row.etime else "-",
                str(row.generation) if row.generation is not None else "-",
                str(row.findings_total) if row.findings_total is not None else "-",
                _format_peer_health_summary(row.peer_health_summary),
                _short_updated(row.updated_at),
                _terminal_safe(row.run_id or "-"),
                _terminal_safe(_truncate(row.run_dir or "-", 64)),
                _terminal_safe(_truncate(row.command, 64)),
            )
        )
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for index, cell in enumerate(table_row):
            widths[index] = max(widths[index], len(cell))
    lines: list[str] = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)).rstrip()
    ]
    for table_row in table_rows:
        lines.append(
            "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(table_row)).rstrip()
        )
    return "\n".join(lines) + "\n" if lines else ""


def _read_orchestrator_progress(
    run_dir: str | None,
) -> tuple[int | None, int | None, str | None]:
    """Read the effective orchestrator snapshot if available.

    Returns ``(current_generation, findings_total, updated_at)`` from the
    effective periodic/final snapshot written by the research_loop status writer
    (#165). Any error path — missing run_dir, missing file, unreadable,
    malformed JSON, missing fields — collapses to ``(None, None, None)``
    so the call site can render ``"-"`` without branching.

    Stale rows benefit too: the JSON is the last snapshot the
    orchestrator wrote before death, so the operator still sees where
    the run got to.
    """
    if not run_dir:
        return None, None, None
    payload = read_effective_orchestrator_status(Path(run_dir))
    if not payload:
        return None, None, None
    lifecycle = str(payload.get("exit_condition") or payload.get("status") or "").lower()
    terminal = bool(lifecycle) and lifecycle not in {"in_progress", "running"}
    gen_raw = (
        payload.get("generations_completed") if terminal else payload.get("current_generation")
    )
    if not isinstance(gen_raw, int):
        gen_raw = payload.get("current_generation")
    findings_raw = payload.get("findings_total")
    updated_raw = payload.get("updated_at")
    gen = int(gen_raw) if isinstance(gen_raw, int) else None
    findings = int(findings_raw) if isinstance(findings_raw, int) else None
    updated = updated_raw if isinstance(updated_raw, str) and updated_raw else None
    return gen, findings, updated


def _row_from_registry_live(
    entry: RegistryEntry,
    ppid: int,
    etime: str,
    command: str,
    *,
    include_peer_health: bool = True,
) -> StatusRow:
    generation, findings_total, updated_at = _read_orchestrator_progress(entry.run_dir)
    registry_state = entry.extra.get("startup_state") or entry.state
    state = STATE_INCONSISTENT if entry.state == STATE_STOPPED else registry_state
    peer_summary, peers = _peer_health_fields(
        entry.run_dir,
        entry.task_path,
        generation,
        enabled=include_peer_health,
    )
    return StatusRow(
        pid=entry.pid,
        ppid=ppid,
        etime=etime,
        command=command,
        run_dir=entry.run_dir,
        source=SOURCE_REGISTRY,
        state=state,
        run_id=entry.run_id,
        task_path=entry.task_path,
        model=entry.model,
        model_provider_ref=entry.model_provider_ref,
        started_at=entry.started_at,
        generation=generation,
        findings_total=findings_total,
        updated_at=updated_at,
        peer_health_summary=peer_summary,
        peers=peers,
        extras=(
            {"registry_state": STATE_STOPPED, "process_state": STATE_RUNNING}
            if state == STATE_INCONSISTENT
            else {}
        ),
    )


def _row_from_registry_stale(
    entry: RegistryEntry,
    *,
    include_peer_health: bool = True,
) -> StatusRow:
    # We do not rewrite the registry on stale detection — explicit
    # cleanup goes through ``praxist stop --gc`` (#166). The row shape
    # simply reports what disk says plus a "stale" tag.
    generation, findings_total, updated_at = _read_orchestrator_progress(entry.run_dir)
    peer_summary, peers = _peer_health_fields(
        entry.run_dir,
        entry.task_path,
        generation,
        enabled=include_peer_health,
    )
    return StatusRow(
        pid=entry.pid,
        ppid=0,
        etime="-",
        command=" ".join(entry.command),
        run_dir=entry.run_dir,
        source=SOURCE_STALE,
        state=_terminal_registry_state(entry),
        run_id=entry.run_id,
        task_path=entry.task_path,
        model=entry.model,
        model_provider_ref=entry.model_provider_ref,
        started_at=entry.started_at,
        generation=generation,
        findings_total=findings_total,
        updated_at=updated_at,
        peer_health_summary=peer_summary,
        peers=peers,
    )


def _row_from_registry_remote(entry: RegistryEntry) -> StatusRow:
    generation, findings_total, updated_at = _read_orchestrator_progress(entry.run_dir)
    return StatusRow(
        pid=entry.pid,
        ppid=0,
        etime="-",
        command=" ".join(entry.command),
        run_dir=entry.run_dir,
        source=SOURCE_REMOTE,
        state="remote",
        run_id=entry.run_id,
        task_path=entry.task_path,
        model=entry.model,
        model_provider_ref=entry.model_provider_ref,
        started_at=entry.started_at,
        generation=generation,
        findings_total=findings_total,
        updated_at=updated_at,
        extras={
            "hostname": entry.extra.get("hostname", ""),
            "boot_id": entry.extra.get("boot_id", ""),
        },
    )


def _row_from_registry_unknown(entry: RegistryEntry, reason: str) -> StatusRow:
    generation, findings_total, updated_at = _read_orchestrator_progress(entry.run_dir)
    return StatusRow(
        pid=entry.pid,
        ppid=0,
        etime="-",
        command=" ".join(entry.command),
        run_dir=entry.run_dir,
        source=SOURCE_REGISTRY,
        state="unknown",
        run_id=entry.run_id,
        task_path=entry.task_path,
        model=entry.model,
        model_provider_ref=entry.model_provider_ref,
        started_at=entry.started_at,
        generation=generation,
        findings_total=findings_total,
        updated_at=updated_at,
        extras={"probe_error": reason},
    )


def _terminal_registry_state(entry: RegistryEntry) -> str:
    """Prefer a terminal artifact status over the generic stale label."""
    startup_state = entry.extra.get("startup_state", "")
    if startup_state == STATE_FAILED:
        return STATE_FAILED
    if entry.state not in {STATE_RUNNING, STATE_STALE}:
        return entry.state
    for filename in ("run_summary.json", "run.json"):
        try:
            payload = json.loads((Path(entry.run_dir) / filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").strip().lower()
        if status in {"succeeded", "completed"}:
            return STATE_COMPLETED
        if status in {"failed", "error"}:
            return STATE_FAILED
    return STATE_STALE


def _read_peer_health(
    run_dir: str | None,
    task_path: str | None,
    generation: int | None,
    *,
    scan_result_artifacts: bool = True,
) -> PeerHealthSnapshot:
    """Read peer-memory health for ``run_dir`` without letting status fail."""

    if not run_dir:
        return _empty_peer_health(generation)
    task_spec = _load_task_spec_for_status(task_path)
    primary_metric = "metric_value"
    direction = "maximize"
    baselines: list[object] = []
    if task_spec is not None:
        primary_metric = str(getattr(task_spec.evaluation, "primary_metric", "") or primary_metric)
        direction = str(getattr(task_spec.evaluation, "direction", "") or direction)
        baselines = list(getattr(task_spec, "baselines", []) or [])
    try:
        return collect_peer_memory_health(
            run_dir=Path(run_dir),
            generation_id=generation,
            primary_metric=primary_metric,
            direction="minimize" if direction == "minimize" else "maximize",
            baselines=baselines,
            scan_result_artifacts=scan_result_artifacts,
            task_spec=task_spec,
        )
    except Exception:
        return _empty_peer_health(generation)


def _load_task_spec_for_status(task_path: str | None) -> TaskSpec | None:
    if not task_path:
        return None
    try:
        path = Path(task_path).expanduser()
        task_yaml = path if path.name == "task.yaml" else path / "task.yaml"
        if not task_yaml.exists():
            return None
        return load_task_spec(task_yaml)
    except Exception:
        return None


def _empty_peer_health(generation: int | None) -> PeerHealthSnapshot:
    return PeerHealthSnapshot(
        generation_id=generation,
        summary={"red": 0, "yellow": 0, "green": 0},
        peers=[],
    )


def _peer_health_fields(
    run_dir: str | None,
    task_path: str | None,
    generation: int | None,
    *,
    enabled: bool,
) -> tuple[dict[str, int] | None, list[dict[str, object]]]:
    if not enabled:
        return None, []
    snapshot = _read_peer_health(run_dir, task_path, generation)
    return snapshot.summary, [peer.to_dict() for peer in snapshot.peers]


def _format_peer_health_summary(summary: dict[str, int] | None) -> str:
    if not summary:
        return "-"
    red = int(summary.get("red", 0) or 0)
    yellow = int(summary.get("yellow", 0) or 0)
    green = int(summary.get("green", 0) or 0)
    if red + yellow + green == 0:
        return "-"
    return f"R{red}/Y{yellow}/G{green}"


def _validate_registry_pid(
    entry: RegistryEntry, cmdline_by_pid: dict[int, tuple[int, str, str]]
) -> tuple[int, str, str] | None:
    """Return the live ``ps`` row for ``entry.pid`` if still ours.

    Compares against ``entry.command_prefix`` to defend against PID
    recycling: if the PID belongs to a different process now, we treat
    the registry entry as stale rather than report a foreign command
    under the Praxist row.
    """
    if entry_process_epoch_matches(entry) is False:
        return None
    live = cmdline_by_pid.get(entry.pid)
    if live is None:
        if not _pid_is_alive(entry.pid):
            return None
        # PID is alive but ``ps`` didn't list it — likely a permission
        # boundary (different user, container, …).  Treat as stale for
        # status purposes; we'd refuse to signal it from stop anyway.
        return None
    ppid, etime, command = live
    if process_identity_matches(entry) is False:
        return None
    if not registry_command_matches(entry, command):
        return None
    return ppid, etime, command


def registry_command_matches(entry: RegistryEntry, command: str) -> bool:
    """Return whether a live controller command belongs to ``entry``.

    A short executable prefix cannot distinguish two Praxist runs after PID reuse.
    The controller must also carry the exact recorded run directory. A command
    that cannot be tokenized is unknown and therefore cannot grant ownership.
    """

    try:
        argv = shlex.split(command)
        prefix_argv = shlex.split(entry.command_prefix) if entry.command_prefix else []
    except ValueError:
        return False
    if prefix_argv:
        if len(argv) < len(prefix_argv) or argv[1 : len(prefix_argv)] != prefix_argv[1:]:
            return False
        if not _executable_alias_matches(prefix_argv[0], argv[0]):
            return False
    recorded_argv = [str(value) for value in entry.command]

    def run_dir_value(values: list[str]) -> str | None:
        for index, value in enumerate(values):
            if value == "--run-dir" and index + 1 < len(values):
                return values[index + 1]
            if value.startswith("--run-dir="):
                return value.split("=", 1)[1]
        return None

    recorded_run_dir = run_dir_value(recorded_argv)
    # Schema-v1 registry files written by early releases did not always retain
    # the full argv. Preserve their prefix-only lifecycle compatibility while
    # enforcing run-specific identity for every current ``praxist start`` record.
    if recorded_run_dir is None:
        return True
    live_run_dir = run_dir_value(argv)
    expected = str(Path(recorded_run_dir).expanduser().resolve())
    if live_run_dir is not None and str(Path(live_run_dir).expanduser().resolve()) == expected:
        return True

    # ``ps`` returns a flat display string, not the original argv. Some
    # implementations omit quoting, so a path containing spaces cannot be
    # reconstructed with shlex. Accept only the exact recorded path followed
    # by end-of-command or another option; this does not weaken PID-reuse
    # protection for prefix-related paths.
    for candidate in {str(recorded_run_dir), expected}:
        for marker in ("--run-dir ", "--run-dir="):
            needle = f"{marker}{candidate}"
            start = command.find(needle)
            while start >= 0:
                suffix = command[start + len(needle) :]
                if not suffix or (suffix[0].isspace() and suffix.lstrip().startswith("-")):
                    return True
                start = command.find(needle, start + 1)
    return False


def _executable_alias_matches(recorded: str, live: str) -> bool:
    """Compare one executable token without relaxing the remaining command."""

    if recorded == live:
        return True
    recorded_path = _resolved_executable_path(recorded)
    live_path = _resolved_executable_path(live)
    try:
        return os.path.samefile(recorded_path, live_path)
    except OSError:
        return recorded_path == live_path


def _resolved_executable_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        resolved = shutil.which(value)
        if resolved:
            candidate = Path(resolved)
    return candidate.resolve(strict=False)


def _pid_is_alive(pid: int) -> bool:
    """Cross-platform liveness check via ``kill -0``.

    Returns False when the PID does not exist; True when it exists and
    we have permission to signal it; True (conservative) when the PID
    exists but EPERM blocks us — the caller decides what that means.
    """
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


def _source_sort_key(source: str) -> int:
    """Sort registry rows above ps-only above stale (most-relevant first)."""
    return {
        SOURCE_REGISTRY: 0,
        SOURCE_PS_ONLY: 1,
        SOURCE_REMOTE: 2,
        SOURCE_STALE: 3,
    }.get(source, 4)


def _filter_rows(
    rows: list[StatusRow],
    *,
    run_id: str | None,
    task_path: str | None,
    active: bool,
    latest: bool,
) -> list[StatusRow]:
    filtered = rows
    if run_id:
        filtered = [row for row in filtered if row.run_id == run_id]
    if task_path:
        expected = str(Path(task_path).expanduser().resolve())
        filtered = [
            row
            for row in filtered
            if row.task_path and str(Path(row.task_path).expanduser().resolve()) == expected
        ]
    if active:
        filtered = [
            row
            for row in filtered
            if row.source in {SOURCE_REGISTRY, SOURCE_PS_ONLY}
            and row.state in {STATE_RUNNING, STATE_INCONSISTENT, "starting"}
        ]
    if latest and filtered:
        filtered = [
            max(
                filtered,
                key=lambda row: (row.started_at or "", row.run_id or "", row.pid),
            )
        ]
    return filtered


def _terminal_safe(value: str) -> str:
    """Remove control characters that could rewrite an operator terminal."""
    return "".join(
        char
        for char in str(value)
        if (char >= " " or char == "\t")
        and char
        not in {
            "\u202a",
            "\u202b",
            "\u202c",
            "\u202d",
            "\u202e",
            "\u2066",
            "\u2067",
            "\u2068",
            "\u2069",
        }
    )


def _read_ps_table(*, timeout_seconds: float = 10.0) -> dict[int, tuple[int, str, str]]:
    """Run ``ps`` and return ``{pid: (ppid, etime, command)}``.

    ``ps -axo pid,ppid,etime,command`` is portable across macOS, Linux,
    and BSD; ``etime`` formats as ``[[DD-]hh:]mm:ss`` which is human
    readable. We do not parse it — it round-trips to the table and
    JSON output unchanged so downstream callers can apply their own
    formatting if needed.
    """
    global _LAST_PS_ERROR
    _LAST_PS_ERROR = ""
    ps_bin = shutil.which("ps")
    if ps_bin is None:
        _LAST_PS_ERROR = "process probe unavailable: ps was not found on PATH"
        return {}
    try:
        result = subprocess.run(
            [ps_bin, "-ww", "-axo", "pid,ppid,etime,command"],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.1, float(timeout_seconds)),
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        _LAST_PS_ERROR = f"process probe failed: {exc}"
        return {}
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        _LAST_PS_ERROR = f"process probe failed: {detail}"
        return {}
    rows: dict[int, tuple[int, str, str]] = {}
    lines = result.stdout.splitlines()
    for line in lines[1:]:
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        etime = parts[2]
        command = parts[3]
        rows[pid] = (ppid, etime, command)
    return rows


def _self_ancestor_pids(rows: dict[int, tuple[int, str, str]]) -> set[int]:
    """Return PIDs that should never appear in the output (this process tree)."""
    try:
        excluded: set[int] = {os.getpid(), os.getppid()}
        current = os.getppid()
        while current and current != 1 and current in rows:
            parent = rows[current][0]
            if parent in excluded:
                break
            excluded.add(parent)
            current = parent
        return excluded
    except OSError:  # pragma: no cover - posix syscall guard.
        return set()


def _extract_run_dir(command: str) -> str | None:
    """Best-effort extraction of ``--run-dir`` from a process command line."""
    match = _RUN_DIR_HINT_RE.search(command)
    return match.group(1) if match else None


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def _short_updated(value: str | None) -> str:
    """Compact a full ISO 8601 timestamp for the ``UPDATED`` column.

    Returns ``-`` when ``value`` is missing. Otherwise strips the
    timezone suffix and microseconds so the column stays narrow:
    ``2026-05-22T01:30:15+00:00`` → ``2026-05-22 01:30:15``. Falls
    back to the original string if parsing fails — better to render
    the raw value than a misleading slice.
    """
    if not value:
        return "-"
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return value
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Internal helpers retained for the rest of the CLI lifecycle commands.
# ---------------------------------------------------------------------------


def praxist_process_regexes() -> list[re.Pattern[str]]:
    """Return controller-only patterns shared with ``praxist stop``."""
    return [re.compile(pattern) for pattern in _PRAXIST_CONTROLLER_PATTERNS]


def read_ps_table() -> dict[int, tuple[int, str, str]]:
    """Public ``_read_ps_table`` alias for use by ``praxist stop``."""
    return _read_ps_table()


def self_ancestor_pids(rows: dict[int, tuple[int, str, str]]) -> set[int]:
    """Public ``_self_ancestor_pids`` alias for use by ``praxist stop``."""
    return _self_ancestor_pids(rows)


def pid_is_alive(pid: int) -> bool:
    """Public ``_pid_is_alive`` alias for use by ``praxist stop``."""
    return _pid_is_alive(pid)
