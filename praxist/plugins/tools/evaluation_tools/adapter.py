"""
Generalized evaluation MCP tools.

Replaces the W2S-specific evaluate_predictions / PGR tools with a
task-agnostic metrics logging system.

Dual-mode:
  - Local mode (single server): reads/writes shared SQLite directly
  - Server mode (multi-machine): POSTs to orchestrator HTTP API
"""

import asyncio
import json
import logging
import math
import os
import re
import tempfile
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from praxist.core.execution_guards import BudgetedActionGuard
from praxist.plugins.tools.result_envelope import (
    coerce_inline_limit,
    read_tool_result_ref,
    with_tool_output_envelope,
)
from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    DERIVED_VIEW,
    attach_artifact_semantics,
)
from praxist.plugins.workflow_stages.research_loop.backend.event_wait import (
    wait_for_filesystem_event,
)
from praxist.plugins.workflow_stages.research_loop.backend.tools.http_utils import (
    async_http_get,
    async_http_post,
    get_server_url,
)

try:
    from claude_agent_sdk import create_sdk_mcp_server, tool
except ImportError:
    tool = None
    create_sdk_mcp_server = None

# R2#1 fix (v2026-05-04): pattern matches the canonical peer_id form
# `gen{N}_peer{i}` so finding/metric records can derive their generation
# from the peer_id (which is set at peer construction and immutable for
# the peer's lifetime), bypassing the GENERATION_ID-via-os.environ race
# at gen boundaries.
# R5#5 fix: case-insensitive so common LLM-formatted IDs (Gen0_Peer0,
# GEN1_PEER3) still parse. Trim whitespace before matching.
_PEER_ID_GEN_RE = re.compile(r"^gen(\d+)_peer\d+$", re.IGNORECASE)
_GEN_DIR_RE = re.compile(r"^gen[_-]?(\d+)$", re.IGNORECASE)
_CLAUDE_SESSION_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_CLAUDE_BACKGROUND_TASK_ID_RE = re.compile(r"^[a-z0-9]{6,64}$", re.IGNORECASE)


def _gen_id_from_peer_id(peer_id: str) -> int | None:
    """Extract gen_id from a canonical peer_id like 'gen2_peer3'.

    Case-insensitive (R5#5 fix); whitespace-tolerant. Returns None if the
    peer_id is empty or doesn't match the canonical pattern. Callers
    should fall back to GENERATION_ID env in that case.
    """
    if not peer_id:
        return None
    m = _PEER_ID_GEN_RE.match(peer_id.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except (ValueError, TypeError):
        return None


def _gen_id_from_wait_paths(paths: list[str] | tuple[str, ...]) -> int | None:
    """Return a generation id only when stable path components agree."""

    generation_ids: set[int] = set()
    for raw_path in paths:
        try:
            parts = Path(str(raw_path)).parts
        except (OSError, TypeError, ValueError):
            continue
        for part in parts:
            gen_id = _gen_id_from_peer_id(part)
            if gen_id is None:
                match = _GEN_DIR_RE.fullmatch(part)
                if match:
                    gen_id = int(match.group(1))
            if gen_id is not None:
                generation_ids.add(gen_id)
    return next(iter(generation_ids)) if len(generation_ids) == 1 else None


def _gen_id_from_wait_context(
    args: dict[str, Any], paths: list[str] | tuple[str, ...] = ()
) -> int | None:
    """Resolve the generation whose STOP_SIGNAL this tool call should honor.

    Historical ``gen_N/STOP_SIGNAL`` files remain in a run directory forever.
    A long ``wait_for_file`` from gen N+1 must not drain just because gen N
    closed earlier. Prefer the immutable peer id when available; fall back to an
    explicit generation id or the runtime env only when needed.
    """

    caller_peer_id = args.get("peer_id", "") or ""
    gen_id = _gen_id_from_peer_id(caller_peer_id)
    if gen_id is not None:
        return gen_id

    raw_generation = args.get("generation_id")
    try:
        return int(raw_generation)
    except (TypeError, ValueError):
        pass

    path_gen_id = _gen_id_from_wait_paths(paths)
    if path_gen_id is not None:
        return path_gen_id

    env_gen_id = _gen_id_from_peer_id(_get_env("PEER_ID", ""))
    if env_gen_id is not None:
        return env_gen_id
    try:
        return int(_get_env("GENERATION_ID", ""))
    except (TypeError, ValueError):
        return None


logger = logging.getLogger(__name__)


def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _wait_for_file_hard_cap_seconds() -> int:
    """Return the task-tunable wait_for_file ceiling.

    Long formal evaluations can legitimately exceed the historical 6h clamp.
    Keep a bounded default and allow tasks to raise/lower it through the
    runtime environment without changing tool-call prompts.
    """
    default = 18 * 3600
    raw = _get_env("PRAXIST_WAIT_FOR_FILE_MAX_SECONDS", str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(60, min(value, 24 * 3600))


def _claude_runtime_task_output_id(path: str | Path) -> str | None:
    """Identify Claude runtime-owned background-task output files.

    ``tasks/<task-id>.output`` is a transcript surface, not a completion
    sentinel: a successful command may legitimately write zero bytes.  Keep
    the detector deliberately narrow so ordinary task-owned ``*.output`` files
    retain normal ``wait_for_file`` semantics.
    """

    try:
        candidate = Path(path)
        if candidate.suffix != ".output" or candidate.parent.name != "tasks":
            return None
        task_id = candidate.stem
        session_dir = candidate.parent.parent
        if not _CLAUDE_BACKGROUND_TASK_ID_RE.fullmatch(task_id):
            return None
        if not _CLAUDE_SESSION_ID_RE.fullmatch(session_dir.name):
            return None
        if not any(parent.name.startswith("claude-") for parent in session_dir.parents):
            return None
        return task_id
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _is_local_mode() -> bool:
    return _get_env("LOCAL_MODE", "false").lower() in ("1", "true", "yes")


def _env_path(key: str) -> Path | None:
    raw = _get_env(key, "")
    if not raw:
        return None
    try:
        return Path(raw).expanduser()
    except (OSError, RuntimeError, ValueError):
        return None


def _run_dir_from_runtime_env() -> Path | None:
    for key in ("PRAXIST_RUN_DIR", "LOCAL_STORE_DIR", "AUTO_RESEARCH_RUN_DIR", "RUN_DIR"):
        path = _env_path(key)
        if path is not None:
            return path
    for key in ("LOGS_DIR", "LOCAL_FINDINGS_DIR", "FRONTIER_DIR"):
        path = _env_path(key)
        if path is not None:
            return path.parent
    return None


def _logs_dir_from_runtime_env() -> Path:
    path = _env_path("LOGS_DIR")
    if path is not None:
        return path
    run_dir = _run_dir_from_runtime_env()
    if run_dir is not None:
        return run_dir / "logs"
    return Path("logs")


def _findings_dir_from_runtime_env() -> Path:
    path = _env_path("LOCAL_FINDINGS_DIR")
    if path is not None:
        return path
    run_dir = _run_dir_from_runtime_env()
    if run_dir is not None:
        return run_dir / "shared_findings"
    return Path("shared_findings")


def _run_id_from_args_or_env(args: dict[str, Any]) -> str:
    raw = args.get("run_id") or _get_env("PRAXIST_RUN_ID", "")
    if raw:
        return str(raw)
    run_dir = _run_dir_from_runtime_env()
    return run_dir.name if run_dir is not None else ""


def _generation_id_from_context(peer_id: str) -> int:
    gen_id_from_peer = _gen_id_from_peer_id(peer_id)
    if gen_id_from_peer is not None:
        return gen_id_from_peer
    try:
        return int(_get_env("GENERATION_ID", "0"))
    except (TypeError, ValueError):
        return 0


def _text_result(data: Any) -> dict[str, Any]:
    """Wrap a JSON-serializable value as an MCP text content response."""
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps({"error": msg})}], "is_error": True}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def _handle_log_experiment_metrics(args: dict[str, Any]) -> dict[str, Any]:
    """Log experiment metrics."""
    run_id = _run_id_from_args_or_env(args)
    if not run_id:
        return _error_result(
            "log_experiment_metrics: 'run_id' is required when PRAXIST_RUN_ID/run dir context is absent"
        )
    variant_name = str(args.get("variant_name", "")).strip()
    if not variant_name:
        return _error_result("log_experiment_metrics: 'variant_name' is required")
    if "metrics" not in args:
        return _error_result("log_experiment_metrics: 'metrics' is required")
    metrics_raw = args["metrics"]
    notes = args.get("notes", "")
    step = args.get("step", 0)

    try:
        metrics_dict = json.loads(metrics_raw) if isinstance(metrics_raw, str) else metrics_raw
    except json.JSONDecodeError:
        return _error_result(f"Invalid JSON in metrics: {metrics_raw}")
    if not isinstance(metrics_dict, dict):
        return _error_result("log_experiment_metrics: 'metrics' must be a JSON object")

    caller_peer_id = args.get("peer_id", "") or _get_env("PEER_ID", "") or ""
    # R2#1 fix (v2026-05-04): derive generation_id from caller_peer_id
    # when it has the canonical "gen{N}_peer{i}" form. This prevents the
    # GENERATION_ID-via-os.environ race where a still-draining gen-N peer
    # submits a finding AFTER the orchestrator has set GENERATION_ID=N+1.
    # The peer_id is captured at peer construction (immutable for the peer's
    # lifetime), so it accurately identifies the gen the finding belongs
    # to regardless of orchestrator-side env mutations.
    record = {
        "run_id": run_id,
        "variant_name": variant_name,
        "metrics": metrics_dict,
        "notes": notes,
        "step": step,
        "peer_id": caller_peer_id,
        "generation_id": _generation_id_from_context(caller_peer_id),
        "timestamp": datetime.now().isoformat(),
    }

    # Always persist to local JSONL (append-only log)
    logs_dir = _logs_dir_from_runtime_env()
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = logs_dir / "metrics_log.jsonl"
    with open(metrics_log, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    # Write to shared SQLite (local mode) or POST to server
    if _is_local_mode():
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
                init_db,
                insert_metric,
            )

            init_db()
            insert_metric(record)
        except Exception as e:
            logger.debug(f"SQLite write failed: {e}")
    else:
        try:
            server_url = get_server_url()
            await async_http_post(
                f"{server_url}/api/metrics",
                json_data=record,
                timeout=10,
            )
        except Exception as e:
            logger.debug(f"Could not post metrics to server: {e}")

    return _text_result(
        {
            "status": "recorded",
            "run_id": run_id,
            "metrics": metrics_dict,
        }
    )


async def _handle_share_finding(args: dict[str, Any]) -> dict[str, Any]:
    """Share a research finding with other agents.

    peer_id resolution: caller-supplied ``peer_id`` arg wins. This is needed
    because the MCP server is hosted in-process in the orchestrator, which
    runs multiple peers concurrently as asyncio tasks — they share one
    ``os.environ`` so ``_get_env("PEER_ID")`` can't distinguish the caller.
    The prompt tells each peer its own id; passing it here preserves it
    through to SQLite.
    """
    missing_required = [
        key for key in ("finding_type", "title", "content") if not str(args.get(key, "")).strip()
    ]
    if missing_required:
        return _error_result(f"share_finding: missing required fields: {missing_required}")
    finding_type = str(args["finding_type"])
    title = str(args["title"])
    content = str(args["content"])
    metrics_raw = args.get("metrics", "{}")
    variant_name = args.get("variant_name", "")
    notes = args.get("notes", "")
    caller_peer_id = args.get("peer_id", "") or _get_env("PEER_ID", "") or ""
    links_raw = args.get("links", "")
    design_dimensions_raw = args.get("design_dimensions", "")

    valid_types = {"result", "hypothesis", "insight", "challenge", "error"}
    if finding_type not in valid_types:
        return _error_result(f"finding_type must be one of {valid_types}")

    try:
        metrics_dict = json.loads(metrics_raw) if isinstance(metrics_raw, str) else metrics_raw
    except json.JSONDecodeError:
        metrics_dict = {}

    # Parse optional graph-edge declarations (Finding Graph MVP Step 5).
    # Accepted shapes:
    #   "" (empty) → no links
    #   JSON string "[{target_finding_id, edge_type, rationale?}, ...]"
    #   already a list (SDK passes through)
    links_list = []
    if links_raw:
        try:
            parsed = json.loads(links_raw) if isinstance(links_raw, str) else links_raw
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and item.get("target_finding_id"):
                        links_list.append(
                            {
                                "target_finding_id": str(item["target_finding_id"]),
                                "edge_type": str(item.get("edge_type", "related_to")),
                                "rationale": str(item.get("rationale", "")),
                            }
                        )
        except (json.JSONDecodeError, TypeError):
            links_list = []

    import uuid

    # R9-1 fix: parse design_dimensions JSON. If valid flat dict, attach
    # to finding. Without this kwarg in the tool schema, peers following
    # the prompt example silently drop diversity metadata, making the
    # entire diversity-overlap signal `no_data`.
    design_dimensions = None
    if design_dimensions_raw:
        try:
            parsed = (
                json.loads(design_dimensions_raw)
                if isinstance(design_dimensions_raw, str)
                else design_dimensions_raw
            )
            if isinstance(parsed, dict):
                # Coerce values to strings (the spec says "short strings").
                design_dimensions = {
                    str(k): str(v) for k, v in parsed.items() if isinstance(k, str)
                }
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                f"share_finding: could not parse design_dimensions={design_dimensions_raw!r}"
            )

    # R2#1 fix: derive gen_id from canonical peer_id (immutable per peer)
    gen_id_from_peer = _gen_id_from_peer_id(caller_peer_id)
    # R7-1 fix: parse `extra` JSON if supplied. Peers stamp peer_role +
    # target_hypothesis here so the next PI synthesis + frontier filter
    # can audit role-specific contributions.
    extra_raw = args.get("extra", "")
    extra_dict: dict[str, Any] = {}
    if isinstance(extra_raw, dict):
        extra_dict = extra_raw
    elif isinstance(extra_raw, str) and extra_raw.strip():
        try:
            parsed = json.loads(extra_raw)
            if isinstance(parsed, dict):
                extra_dict = parsed
            else:
                logger.warning(
                    "share_finding: `extra` must JSON-decode to dict, got %s",
                    type(parsed).__name__,
                )
        except json.JSONDecodeError as e:
            logger.warning(
                "share_finding: could not parse extra=%r: %s",
                extra_raw,
                e,
            )
    # R9-M4 fix: warn when extra is empty + the gen has an active agenda.
    # Peers are instructed to stamp peer_role + target_hypothesis; missing
    # extra silently breaks the frontier role-exclusion filter (R6#3).
    # Resolve through the same run-context helper used by metric/finding writes
    # so direct bridge calls and resumed runs inspect the same agenda root.
    if not extra_dict and gen_id_from_peer is not None:
        local_store_dir = _run_dir_from_runtime_env()
        if local_store_dir is not None:
            try:
                from pathlib import Path as _Path

                agenda_path = (
                    _Path(local_store_dir)
                    / "agendas"
                    / f"research_agenda_gen{gen_id_from_peer}.yaml"
                )
                if agenda_path.exists():
                    logger.warning(
                        "share_finding(peer_id=%s): `extra` is empty but a PI "
                        "agenda exists for gen %d. Peers should stamp "
                        'extra={"peer_role":..., "target_hypothesis":...} '
                        "so the frontier filter can audit role compliance.",
                        caller_peer_id,
                        gen_id_from_peer,
                    )
            except Exception:
                pass
    from praxist.plugins.workflow_stages.research_loop.backend.effective_config import (
        strip_effective_config_fields,
    )

    strip_effective_config_fields(metrics_dict)
    strip_effective_config_fields(extra_dict)
    finding = {
        "id": str(uuid.uuid4()),
        "finding_type": finding_type,
        "title": title,
        "content": content,
        "metrics": metrics_dict,
        "variant_name": variant_name,
        "notes": notes,
        "peer_id": caller_peer_id,
        "generation_id": _generation_id_from_context(caller_peer_id),
        "timestamp": datetime.now().isoformat(),
    }
    if design_dimensions:
        finding["design_dimensions"] = design_dimensions
    if links_list:
        # Stash in extra so findings_ingest + graph builder can see it.
        finding["links"] = links_list
    # R7-1 fix: preserve extra dict at top level so the row-mapper's
    # "flatten extra into top-level" pattern (local_store._row_to_finding)
    # exposes peer_role + target_hypothesis + any peer-supplied keys
    # to downstream consumers (frontier.promote, PI synthesis prompt).
    if extra_dict:
        finding["extra"] = extra_dict

    # Save to filesystem (always — agents read this directory directly)
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync import (
            save_finding_to_dir,
        )

        findings_dir = _findings_dir_from_runtime_env()
        save_finding_to_dir(finding, findings_dir)
    except Exception as e:
        logger.warning(f"Could not save finding to filesystem: {e}")

    # Write to shared SQLite (local mode) or POST to server
    if _is_local_mode():
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
                init_db,
                insert_finding,
            )

            init_db()
            insert_finding(finding)
        except Exception as e:
            logger.debug(f"SQLite write failed: {e}")

        # Immediately run the rule engine against this one finding so
        # edges — especially agent-declared `links` — become visible
        # before the next 120s maintainer cycle. Without this,
        # share_finding(links=[...]) → immediate get_finding_neighbors
        # round-trips return nothing, surprising agents that expect
        # their declared edges to be instantly queryable.
        try:
            from praxist.plugins.graph_maintainers.finding_graph_mvp.engine import (
                FindingGraphBuilder,
            )
            from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
                get_all_findings,
                insert_edges_batch,
            )

            all_findings = get_all_findings()
            builder = FindingGraphBuilder(all_findings)
            proposed = builder.propose_edges_for(finding)
            proposed = [e for e in proposed if e["confidence"] >= builder.MIN_CONFIDENCE]
            if proposed:
                inserted = insert_edges_batch(proposed)
                logger.debug(
                    "share_finding: materialized %d/%d edges immediately for %s",
                    inserted,
                    len(proposed),
                    finding["id"],
                )
        except Exception as e:
            # Never let graph-materialization failure break the share,
            # but log at WARNING with traceback so silent rule-engine
            # regressions are visible in operator logs instead of
            # disappearing into DEBUG. The 120s maintainer cycle will
            # eventually pick up missed edges, so correctness is
            # preserved — but we still want to know when it happens.
            logger.warning(
                "Immediate graph materialization failed for finding %s: %s",
                finding.get("id", "?"),
                e,
                exc_info=True,
            )
    else:
        try:
            server_url = get_server_url()
            await async_http_post(
                f"{server_url}/api/findings/share",
                json_data=finding,
                timeout=30,
            )
        except Exception as e:
            logger.debug(f"Could not post finding to server: {e}")

    summary = content[:500] + "..." if len(content) > 500 else content

    return _text_result(
        {
            "status": "shared",
            "finding_id": finding["id"],
            "type": finding_type,
            "title": title,
            "summary": summary,
        }
    )


async def _handle_wait_for_file(args: dict[str, Any]) -> dict[str, Any]:
    guard = BudgetedActionGuard.from_env(
        action_type="tool.wait_for_file",
        actor_ref="tool_server:evaluation_tools",
        metadata={
            "tool_name": "wait_for_file",
            "path_count": len([p for p in str(args.get("path", "")).split(",") if p.strip()]),
        },
    )
    guard.start()
    status = "succeeded"
    try:
        result = await _handle_wait_for_file_impl(args)
        if isinstance(result, dict) and result.get("is_error"):
            status = "failed"
        return result
    except Exception:
        status = "failed"
        raise
    finally:
        guard.finish(
            actual_usage={},
            expected_units=("wall_clock_seconds",),
            status=status,
            reason="wait_for_file_wall_clock_usage",
        )


async def _handle_wait_for_file_impl(args: dict[str, Any]) -> dict[str, Any]:
    """Wait until a file (or files) exist + meet a minimum-size / contains-text
    condition, OR until timeout_seconds elapses. Returns immediately when the
    requested condition becomes true.

    Why this exists: peers supervising long-running evaluation subprocesses otherwise
    poll via repeated `ls` / `cat` / `sleep` Bash calls, each of which
    re-burns ~1-2K LLM context tokens per round-trip. A long evidence run
    can incur many such polls and waste substantial context. This single tool call
    blocks for the whole wait inside the orchestrator process and returns
    once with the result — one round-trip for the entire wait.

    Runtime boundary:
      - Runtime-owned ``tasks/<id>.output`` files are never interpreted as
        completion sentinels.  Empty output is valid; the runtime's structured
        task notification owns completion and exit status.

    Hardening (R1 review):
      - Path confined to allowed roots (workspace / run_dir / /tmp): a
        malicious or buggy peer cannot use the contains_text boolean as a
        substring oracle on /etc/shadow etc.
      - Regular-file check (S_ISREG): rejects FIFO (would block forever),
        block devices (would exfil + OOM), directories (false-positive).
      - File read capped at 4 MiB and run via asyncio.to_thread so a slow
        disk doesn't stall the orchestrator's event loop.
      - Path count capped at 32; wait fallback is low-frequency and
        inotify-backed on Linux; min_bytes floored at 1.
    """
    import stat as _stat

    raw_path = args.get("path", "")
    if not raw_path:
        return _error_result("wait_for_file: 'path' is required")
    paths_raw = [p.strip() for p in raw_path.split(",") if p.strip()]
    if not paths_raw:
        return _error_result("wait_for_file: no valid paths after split")
    if len(paths_raw) > 32:
        return _error_result(f"wait_for_file: too many paths ({len(paths_raw)} > 32)")

    # Path confinement: resolve each path and require it to live under one
    # of the allowed roots. Defends against peers using contains_text as a
    # substring oracle on host files (e.g. /etc/shadow).
    # R2-5 hardening: reject obviously dangerous roots (operator-misconfig
    # like LOCAL_STORE_DIR=/ or =/etc would otherwise re-open the bypass).
    DANGEROUS_ROOTS = {
        Path("/"),
        Path("/etc"),
        Path("/proc"),
        Path("/sys"),
        Path("/var"),
        Path("/usr"),
        Path("/root"),
        Path("/home"),
        Path("/boot"),
        Path("/dev"),
        Path("/run"),
    }
    allowed_roots: list[Path] = []
    for env_var in (
        "PRAXIST_RUN_DIR",
        "AUTO_RESEARCH_RUN_DIR",
        "LOCAL_STORE_DIR",
        "LOGS_DIR",
        "LOCAL_FINDINGS_DIR",
        "GPU_GOVERNOR_DIR",
        "PROTECTED_PIDS_DIR",
    ):
        v = os.environ.get(env_var)
        if v:
            try:
                resolved = Path(v).resolve()
                if resolved in DANGEROUS_ROOTS:
                    logger.warning(
                        "wait_for_file: ignoring dangerous %s=%s",
                        env_var,
                        v,
                    )
                    continue
                allowed_roots.append(resolved)
            except (OSError, RuntimeError):
                pass
    try:
        repo_root = Path(__file__).resolve().parents[3]
        if repo_root not in DANGEROUS_ROOTS:
            allowed_roots.append(repo_root)
    except IndexError:
        pass
    try:
        temp_root = Path(tempfile.gettempdir()).resolve()
        if temp_root not in DANGEROUS_ROOTS:
            allowed_roots.append(temp_root)
    except (OSError, RuntimeError):
        pass
    allowed_roots.append(Path("/tmp").resolve())

    def _path_allowed(rp: Path) -> bool:
        for root in allowed_roots:
            try:
                rp.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    paths: list[str] = []
    rejected: list[str] = []
    for p in paths_raw:
        # R3-Issue4: reject paths containing control characters (newline,
        # tab, NUL, etc.) — almost always a peer CSV-formatting bug.
        if any(ord(ch) < 0x20 for ch in p):
            rejected.append(repr(p))
            continue
        try:
            rp = Path(p).resolve()
        except (OSError, RuntimeError):
            rejected.append(p)
            continue
        if not _path_allowed(rp):
            rejected.append(p)
            continue
        paths.append(str(rp))
    if rejected:
        return _error_result(
            f"wait_for_file: paths outside allowed roots (workspace/tmp): "
            f"{rejected[:5]}{'...' if len(rejected) > 5 else ''}"
        )
    # R2-6: dedupe duplicates from CSV while preserving order.
    deduped = list(dict.fromkeys(paths))
    dedup_count = len(paths) - len(deduped)
    paths = deduped

    runtime_task_outputs = [
        (path, task_id)
        for path in paths
        if (task_id := _claude_runtime_task_output_id(path)) is not None
    ]
    if runtime_task_outputs:
        return _text_result(
            {
                "status": "runtime_task_notification_required",
                "completion_inferred": False,
                "task_ids": [task_id for _path, task_id in runtime_task_outputs],
                "runtime_output_paths": [path for path, _task_id in runtime_task_outputs],
                "deduped_count": dedup_count,
                "hint": (
                    "This is a runtime-managed background-task transcript, not a "
                    "completion sentinel. Do not call wait_for_file again for this "
                    "path: successful tasks may leave it empty. Return control to "
                    "the agent runtime and consume its structured task notification, "
                    "which carries task status and exit information."
                ),
            }
        )

    try:
        timeout_s = int(args.get("timeout_seconds", 3600))
    except (TypeError, ValueError):
        timeout_s = 3600
    if timeout_s <= 0:
        return _error_result(f"wait_for_file: timeout_seconds must be > 0, got {timeout_s}")
    try:
        # Backwards-compatible input name. On Linux this no longer drives a
        # short polling loop; it is only the minimum fallback cadence when
        # inotify is unavailable.
        poll_s = max(2, int(args.get("poll_interval_seconds", 5)))
    except (TypeError, ValueError):
        poll_s = 5
    try:
        # Floor at 1 to force "non-empty" semantics. min_bytes=0 was a
        # footgun: directories pass `os.stat`'s size check trivially.
        min_bytes = max(1, int(args.get("min_bytes", 1)))
    except (TypeError, ValueError):
        min_bytes = 1
    contains_text = args.get("contains_text", "") or ""
    if len(contains_text) > 1024:
        return _error_result(
            f"wait_for_file: contains_text too long ({len(contains_text)} > 1024 chars)"
        )
    mode = (args.get("mode", "any") or "any").lower()
    if mode not in ("any", "all"):
        mode = "any"

    current_generation_id = _gen_id_from_wait_context(args, paths)
    signal_run_dirs = sorted(_candidate_run_dirs_for_wait_paths(paths))
    hard_cap_seconds = _wait_for_file_hard_cap_seconds()
    generation_budget_seconds = _generation_wait_budget_seconds(
        signal_run_dirs,
        current_generation_id,
    )
    effective_cap_seconds = hard_cap_seconds
    if generation_budget_seconds is not None:
        effective_cap_seconds = min(effective_cap_seconds, generation_budget_seconds)
    timeout_clamped_to_hard_cap = timeout_s > hard_cap_seconds
    timeout_clamped_to_generation_deadline = (
        generation_budget_seconds is not None and timeout_s > generation_budget_seconds
    )
    timeout_s = min(timeout_s, effective_cap_seconds)
    timing_metadata = {
        "timeout_clamped_to_hard_cap": timeout_clamped_to_hard_cap,
        "timeout_hard_cap_seconds": hard_cap_seconds,
        "timeout_clamped_to_generation_deadline": timeout_clamped_to_generation_deadline,
        "generation_budget_seconds": generation_budget_seconds,
    }
    MAX_SCAN_BYTES = 64 * 1024 * 1024
    SCAN_CHUNK_BYTES = 1024 * 1024

    def _check_one_blocking(p: str) -> bool:
        try:
            lst = os.lstat(p)
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            return False
        is_symlink = _stat.S_ISLNK(lst.st_mode)
        if is_symlink:
            try:
                target = Path(p).resolve(strict=True)
            except (FileNotFoundError, OSError, RuntimeError):
                return False
            if not _path_allowed(target):
                return False
            try:
                st = os.stat(p)
            except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
                return False
        else:
            st = lst
        if not _stat.S_ISREG(st.st_mode):
            return False  # rejects FIFO, devices, directories, and unsafe symlink targets
        if st.st_size < min_bytes:
            return False
        if contains_text:
            needle = contains_text.encode("utf-8", errors="replace")
            try:
                flags = os.O_RDONLY | os.O_CLOEXEC
                if not is_symlink:
                    flags |= os.O_NOFOLLOW
                fd = os.open(p, flags)
                opened_st = os.fstat(fd)
                if not _stat.S_ISREG(opened_st.st_mode) or opened_st.st_size < min_bytes:
                    os.close(fd)
                    return False
                if is_symlink:
                    with suppress(OSError, RuntimeError):
                        opened_target = Path(os.readlink(f"/proc/self/fd/{fd}")).resolve(
                            strict=False
                        )
                        if not _path_allowed(opened_target):
                            os.close(fd)
                            return False
                try:
                    with os.fdopen(fd, "rb") as f:
                        scanned = 0
                        overlap = b""
                        keep = max(0, len(needle) - 1)
                        while scanned < MAX_SCAN_BYTES:
                            chunk = f.read(min(SCAN_CHUNK_BYTES, MAX_SCAN_BYTES - scanned))
                            if not chunk:
                                break
                            scanned += len(chunk)
                            haystack = overlap + chunk
                            if needle in haystack:
                                return True
                            overlap = haystack[-keep:] if keep else b""
                        return False
                except (OSError, UnicodeDecodeError):
                    return False
            except (OSError, ValueError):
                # ELOOP from O_NOFOLLOW, or path vanished between lstat and open
                return False
        return True

    # R3-Issue7 fix: bounded thread pool so a hung NFS path can't starve
    # the orchestrator's other async tools. Lazily-initialized module-global.
    global _WAIT_FOR_FILE_EXECUTOR
    if _WAIT_FOR_FILE_EXECUTOR is None:
        from concurrent.futures import ThreadPoolExecutor

        _WAIT_FOR_FILE_EXECUTOR = ThreadPoolExecutor(
            max_workers=16,  # R4-N2: 16 worker headroom for occasional leaks.
            thread_name_prefix="wait_for_file",
        )
    loop = asyncio.get_event_loop()

    async def _check_one(p: str) -> bool:
        # R4-N2: keep per-stat timeout (else a single hung NFS path wedges
        # the whole gather indefinitely past the outer deadline). On
        # timeout the asyncio coroutine is cancelled but the underlying
        # thread leaks — bounded by the 16-worker pool, which can absorb
        # many unique hung paths before saturation.
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(_WAIT_FOR_FILE_EXECUTOR, _check_one_blocking, p),
                timeout=max(2.0, poll_s * 2),
            )
        except TimeoutError:
            return False

    # Stop-signal-honor fix (2026-05-05): poll STOP_SIGNAL in the same cycle
    # so long waits don't delay synthesis. Generation-scope fix (2026-05-31):
    # old gen_N/STOP_SIGNAL files are permanent history. A gen N+1 peer must
    # only drain for gen_(N+1)/STOP_SIGNAL or run-level ORCHESTRATOR_SHUTDOWN.
    # If the caller generation cannot be resolved, keep the legacy broad
    # behavior so non-peer callers remain conservative.
    _signal_watch_paths: list[Path] = []
    for rd in signal_run_dirs:
        _signal_watch_paths.append(rd)
        if current_generation_id is not None:
            _signal_watch_paths.append(rd / f"gen_{current_generation_id}")

    def _stop_signal_present() -> bool:
        for rd in signal_run_dirs:
            try:
                if (rd / "ORCHESTRATOR_SHUTDOWN").exists():
                    return True
                if current_generation_id is None:
                    if any(sig.exists() for sig in rd.glob("gen_*/STOP_SIGNAL")):
                        return True
                elif (rd / f"gen_{current_generation_id}" / "STOP_SIGNAL").exists():
                    return True
            except Exception:
                continue
        return False

    def _closing_signal_present() -> bool:
        if current_generation_id is None:
            return False
        for rd in signal_run_dirs:
            try:
                if (rd / f"gen_{current_generation_id}" / "CLOSING_SIGNAL").exists():
                    return True
            except Exception:
                continue
        return False

    def _wake_signal_present() -> bool:
        return _stop_signal_present() or _closing_signal_present()

    t0 = time.monotonic()
    deadline = t0 + timeout_s
    last_status_log = t0  # log first only after a real 60s elapses
    event_fallback_s = max(60, poll_s)
    while True:
        # Run the per-path checks concurrently to keep latency low when
        # multiple paths are present.
        check_results = await asyncio.gather(*(_check_one(p) for p in paths))
        matches = [p for p, ok in zip(paths, check_results, strict=False) if ok]
        # R3-Issue9: include `deduped_count` in ALL response branches for
        # consistent observability.
        condition_ready = (mode == "any" and bool(matches)) or (
            mode == "all" and len(matches) == len(paths)
        )
        if condition_ready:
            return _text_result(
                {
                    "status": "ready",
                    "elapsed_seconds": round(time.monotonic() - t0, 2),
                    "matched_paths": matches,
                    "missing_paths": []
                    if mode == "all"
                    else [p for p in paths if p not in matches],
                    "deduped_count": dedup_count,
                    **timing_metadata,
                }
            )
        # Stop-signal-honor fix: short-circuit if the orchestrator wrote
        # STOP_SIGNAL while we were waiting. Ready results are returned first
        # so completed evidence is not mislabeled as aborted.
        if signal_run_dirs and _stop_signal_present():
            return _text_result(
                {
                    "status": "aborted_by_stop_signal",
                    "elapsed_seconds": round(time.monotonic() - t0, 2),
                    "matched_paths": matches,
                    "missing_paths": [p for p in paths if p not in matches],
                    "deduped_count": dedup_count,
                    "generation_id": current_generation_id,
                    **timing_metadata,
                    "hint": (
                        "Synthesis trigger fired (STOP_SIGNAL detected); "
                        "wait_for_file exited so the peer can drain. Any later "
                        "background results should be treated as late/quarantined "
                        "signals until revalidated."
                    ),
                }
            )
        if signal_run_dirs and _closing_signal_present():
            return _text_result(
                {
                    "status": "released_for_generation_closing",
                    "elapsed_seconds": round(time.monotonic() - t0, 2),
                    "matched_paths": matches,
                    "missing_paths": [p for p in paths if p not in matches],
                    "deduped_count": dedup_count,
                    "generation_id": current_generation_id,
                    "completion_inferred": False,
                    **timing_metadata,
                    "hint": (
                        "The generation is closing, so this passive file wait was "
                        "released. No evaluator or background process was stopped. "
                        "Inspect already-published results, publish remaining evidence, "
                        "and let scheduler-owned work drain naturally."
                    ),
                }
            )
        now = time.monotonic()
        if now - last_status_log > 60:
            logger.info(
                "wait_for_file: %d/%d ready, elapsed=%.0fs, mode=%s",
                len(matches),
                len(paths),
                now - t0,
                mode,
            )
            last_status_log = now
        if time.monotonic() >= deadline:
            generation_deadline_elapsed = generation_budget_seconds == 0
            return _text_result(
                {
                    "status": (
                        "generation_deadline_elapsed" if generation_deadline_elapsed else "timeout"
                    ),
                    "elapsed_seconds": round(time.monotonic() - t0, 2),
                    "matched_paths": matches,
                    "missing_paths": [p for p in paths if p not in matches],
                    "deduped_count": dedup_count,
                    **timing_metadata,
                    "hint": (
                        "The central generation deadline elapsed; record any available "
                        "evidence and return control to the runtime."
                        if generation_deadline_elapsed
                        else "If you expected the file by now: "
                        "(1) verify the background subprocess actually started "
                        "(`jobs` or `ps -ef | grep <your-script>`); "
                        "(2) check the subprocess log for errors; "
                        "(3) confirm the output filename schema matches what "
                        "your runner produces."
                    ),
                }
            )
        # R2-2 fix: clamp the wait to the remaining time so timeout fires
        # promptly. Prefer filesystem events; fall back only at low
        # frequency on platforms without inotify.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            continue  # let the next iteration's deadline check fire
        watch_paths = [Path(p) for p in paths]
        watch_paths.extend(_signal_watch_paths)
        target_paths = {Path(p).resolve() for p in paths}

        # Bind target_paths via default arg so each loop iteration captures
        # its own set rather than the late-bound closure variable (B023).
        def _is_wait_target_event(path: Path, _targets: set[Path] = target_paths) -> bool:
            try:
                if Path(path).resolve() in _targets:
                    return True
            except (OSError, RuntimeError):
                pass
            return Path(path).name in {"STOP_SIGNAL", "CLOSING_SIGNAL"}

        await wait_for_filesystem_event(
            watch_paths,
            timeout_seconds=min(event_fallback_s, remaining),
            stop_check=_wake_signal_present if signal_run_dirs else None,
            recursive=False,
            max_dirs=128,
            fallback_interval_seconds=min(event_fallback_s, remaining),
            stop_check_interval_seconds=min(30.0, max(2.0, float(poll_s))),
            event_filter=_is_wait_target_event,
        )


def _candidate_run_dirs_for_wait_paths(paths: list[str]) -> list[Path]:
    """Infer run directories from arbitrary task-local wait paths.

    Current task projects can write under any explicit external ``--run-dir``.
    Stop-signal detection therefore keys off run artifacts and run-shaped
    directories instead of the legacy ``experiments_tracking`` path.
    """

    candidates: dict[str, Path] = {}
    for raw in paths:
        try:
            path = Path(str(raw)).expanduser()
        except (OSError, TypeError, ValueError):
            continue
        for parent in (path, *path.parents):
            try:
                resolved = parent.resolve(strict=False)
            except OSError:
                continue
            if _looks_like_run_dir(resolved):
                candidates[str(resolved)] = resolved
                break
    return list(candidates.values())


def _generation_wait_budget_seconds(run_dirs: list[Path], generation_id: int | None) -> int | None:
    """Read the remaining generation budget from the scheduler's existing state.

    The central scheduler owns this deadline.  ``wait_for_file`` only consumes
    the already-materialized fact, avoiding a second deadline registry or
    per-peer process-environment guess.
    """

    if generation_id is None:
        return None
    deadlines: list[float] = []
    for run_dir in run_dirs:
        status_path = run_dir / "resource_scheduler" / "status.json"
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            raw_deadlines = payload.get("generation_deadlines")
            if not isinstance(raw_deadlines, dict):
                continue
            deadline = float(raw_deadlines.get(str(generation_id)))
            if deadline > 0 and math.isfinite(deadline):
                deadlines.append(deadline)
        except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not deadlines:
        return None
    return max(0, int(min(deadlines) - time.time()))


def _looks_like_run_dir(path: Path) -> bool:
    """Return True when ``path`` has Praxist run-dir markers or naming shape."""

    try:
        if (path / "run.json").exists() or (path / "startup_config.json").exists():
            return True
        if (path / "shared_store.db").exists() or (path / "trajectory.jsonl").exists():
            return True
        return path.name.startswith("run_") and (
            (path / "logs").exists()
            or (path / "frontier").exists()
            or (path / "shared_findings").exists()
            or any(path.glob("gen_*"))
        )
    except OSError:
        return False


async def _handle_get_leaderboard(args: dict[str, Any]) -> dict[str, Any]:
    """Get leaderboard of results, optionally filtered by generation."""
    # R7 M1 fix: defensive int-coercion. claude_agent_sdk preserves
    # declared types, but a malformed peer input or future SDK regression
    # would leave us comparing a str to an int → silent fall-through to
    # filesystem mode via the broad except.
    # R8 M1 fix: clamp top_k to a non-negative value. Negative top_k
    # produces `entries[:-1]` which silently drops the LAST entry — the
    # user-visible failure mode is "leaderboard always shows N-1 results
    # for any direction=minimize task". Defensive clamp at handler entry.
    try:
        generation = int(args.get("generation", -1))
    except (TypeError, ValueError):
        generation = -1
    try:
        top_k = max(0, int(args.get("top_k", 20)))
    except (TypeError, ValueError):
        top_k = 20
    inline_limit = coerce_inline_limit(args.get("inline_limit", top_k), default=top_k)

    # Local mode: query SQLite directly
    if _is_local_mode():
        return _text_result(
            _envelope_leaderboard(
                json.loads(_sqlite_leaderboard(generation, top_k)),
                inline_limit,
                generation=generation,
            )
        )

    # R6 Issue 1 fix: in non-local (server) mode, prefer the local
    # SQLite Pareto path when ANCHOR_METRICS is configured AND the
    # SQLite store is reachable. The legacy HTTP server returns a
    # single-axis ranking only — using it when Pareto data is
    # available silently regresses the feature for non-local deployments.
    # Fall through to HTTP only if Pareto is unconfigured or SQLite is
    # unavailable.
    anchors = _parse_anchor_metrics_env()
    if anchors:
        try:
            return _text_result(
                _envelope_leaderboard(
                    json.loads(_sqlite_leaderboard(generation, top_k)),
                    inline_limit,
                    generation=generation,
                )
            )
        except Exception as e:
            logger.warning(
                "server-mode Pareto path failed (%s); falling through to HTTP",
                e,
            )

    # Server mode: try HTTP, fall back to local
    try:
        server_url = get_server_url()
        params = {"top_k": top_k}
        if generation >= 0:
            params["generation"] = generation
        result = await async_http_get(
            f"{server_url}/api/leaderboard",
            params=params,
            timeout=15,
        )
        # R2 MAJ-2 fix: the legacy server returns
        # `{"entries": [...]}` with no `mode` field. Inject
        # `mode="server_legacy"` so peers following the prompt's
        # "branch on mode" guidance don't KeyError. Pareto fields
        # (`pareto_front`, `dominated_top`, `best_in`) are absent in
        # this branch — peers see the legacy single-axis ranking.
        # R3 Issue 6: if the legacy server itself returns an `error`
        # payload (downstream timeout, etc.), don't paper over it —
        # fall through to local SQLite so peers get a real leaderboard.
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"server returned error: {result.get('error')!r}")
        # R4 Issue 5 fix: if the legacy server returns a non-dict
        # (some old endpoints emit bare lists), wrap it so peers doing
        # `response["mode"]` per the prompt don't crash.
        if not isinstance(result, dict):
            result = {
                "mode": "server_legacy",
                "entries": result if isinstance(result, list) else [],
            }
        elif "mode" not in result:
            result = {"mode": "server_legacy", **result}
        return _text_result(_envelope_leaderboard(result, inline_limit, generation=generation))
    except Exception:
        return _text_result(
            _envelope_leaderboard(
                json.loads(_sqlite_leaderboard(generation, top_k)),
                inline_limit,
                generation=generation,
            )
        )


def _envelope_leaderboard(payload: Any, inline_limit: int, *, generation: int | None = None) -> Any:
    """Attach a bounded-output envelope to leaderboard responses."""

    if not isinstance(payload, dict):
        return payload
    payload = attach_artifact_semantics(
        payload,
        role=DERIVED_VIEW,
        stage="leaderboard_tool_response",
        generation_id=generation if generation is not None and generation >= 0 else None,
        actor="tool_server:evaluation_tools",
        derived_from=["shared_store.db", "shared_findings/*", "frontier/frontier_manifest.json"],
        canonical_sources=[
            "shared_store.db",
            "shared_findings/*",
            "frontier/frontier_manifest.json",
        ],
        runtime_fact_source=False,
        notes=(
            "Leaderboards are bounded derived views for agents/operators. "
            "Promotion truth remains owned by result evidence and frontier state."
        ),
    )
    return with_tool_output_envelope(
        payload,
        tool_name="get_leaderboard",
        list_fields=("pareto_front", "dominated_top", "entries"),
        inline_limit=inline_limit,
    )


async def _handle_read_tool_result(args: dict[str, Any]) -> dict[str, Any]:
    """Read a bounded chunk from a full tool-result artifact."""

    ref = str(args.get("ref", "")).strip()
    if not ref:
        return _error_result("read_tool_result: 'ref' is required")
    try:
        offset = int(args.get("offset", 0))
    except (TypeError, ValueError):
        offset = 0
    try:
        max_chars = int(args.get("max_chars", 4000))
    except (TypeError, ValueError):
        max_chars = 4000
    try:
        return _text_result(read_tool_result_ref(ref, offset=offset, max_chars=max_chars))
    except ValueError as exc:
        return _error_result(str(exc))


# ---------------------------------------------------------------------------
# Tool definitions (new SDK API)
# ---------------------------------------------------------------------------

log_experiment_metrics = None
share_finding = None
get_leaderboard = None
wait_for_file = None
read_tool_result = None

# R3-Issue7 / R4-N2: dedicated, bounded thread pool for wait_for_file's
# blocking I/O. Bumped from 4 to 16 workers (R4-N2): per-stat asyncio.wait_for
# leaks underlying threads; small pool starves under stale-NFS scenarios.
# 16 is a generous headroom for typical multi-peer usage and lets a few
# leaked threads coexist before degradation.
_WAIT_FOR_FILE_EXECUTOR = None

if tool is not None:
    log_experiment_metrics = tool(
        "log_experiment_metrics",
        "Log experiment metrics. Metrics should be a JSON string like "
        '{"metric_value": 0.78, "secondary_metric": 0.03}. Pass your '
        "peer_id (e.g. 'gen0_peer3') — the orchestrator hosts this tool "
        "in-process for all sibling peers so env-based attribution won't work.",
        {
            "run_id": str,
            "variant_name": str,
            "metrics": str,
            "notes": str,
            "step": int,
            "peer_id": str,
        },
    )(_handle_log_experiment_metrics)

    share_finding = tool(
        "share_finding",
        "Share a research finding with other agents. "
        "finding_type must be one of: result, hypothesis, insight, error. "
        "Pass your own peer_id (e.g. 'gen0_peer3') so the finding is "
        "attributed correctly — the orchestrator hosts this tool in-process "
        "for all sibling peers so it cannot infer the caller from env. "
        "OPTIONAL links: JSON string of explicit Finding Graph edges, e.g. "
        '\'[{"target_finding_id": "abc-uuid", "edge_type": "supports", '
        '"rationale": "independent replication result"}]\'. Valid edge_types: '
        "related_to, derived_from, updates, supports, challenges. "
        "Use when your new finding explicitly builds on, updates, supports, "
        "or challenges a previous finding whose id you know. "
        "OPTIONAL design_dimensions (R9-1 fix): JSON string of a flat dict "
        "mapping the task's diversity-dimension axes to short strings, e.g. "
        '\'{"computational_mechanism": "two-step correction", '
        '"adaptation_strategy": "adaptive threshold schedule"}\'. The system '
        "uses this for diversity comparison only; it does NOT affect ranking. "
        "Without this, the diversity-overlap signal is `no_data` for your finding. "
        "OPTIONAL extra (R7-1 fix, v2026-05-04): JSON string of arbitrary "
        "metadata. Peers MUST stamp `peer_role` and `target_hypothesis` here "
        "so the next PI synthesis + frontier promotion can credit role-specific "
        'contributions, e.g. \'{"peer_role":"theorist","target_hypothesis":"H_g3_01"}\'. '
        "In legacy single-metric frontier mode, theorist/falsifier findings "
        "are excluded from the deployable frontier candidate pool. In "
        "task-configured lane mode, role filtering is lane-local, so "
        "falsifier/control findings can still enter diagnostic lanes when "
        "the task explicitly allows them.",
        {
            "finding_type": str,
            "title": str,
            "content": str,
            "metrics": str,
            "variant_name": str,
            "notes": str,
            "peer_id": str,
            "links": str,
            "design_dimensions": str,
            "extra": str,
        },
    )(_handle_share_finding)

    get_leaderboard = tool(
        "get_leaderboard",
        "Return the multi-metric Pareto-frontier leaderboard for `result`-"
        "type findings (Praxist v2026-05-01+). The response has TWO "
        "lists: `pareto_front` (variants that no other variant beats on "
        "every configured axis — primary_metric + anchor_metrics) and "
        "`dominated_top` (next-best variants for context). Each entry "
        "carries `best_in` listing the axes where it is #1 on the front. "
        "When the task spec defines no anchor metrics, falls back to a "
        "legacy single-metric ranking (`mode: single_metric`). Use this to "
        "see task-owned tradeoffs across configured metrics instead of a "
        "single-number ranking. Args: `generation` (int, -1 for all), "
        "`top_k` (int, max dominated entries returned).",
        {
            "generation": int,
            "top_k": int,
            "inline_limit": int,
        },
    )(_handle_get_leaderboard)

    read_tool_result = tool(
        "read_tool_result",
        "Read a bounded chunk from a full tool-result artifact returned in "
        "`_tool_output.full_result_ref`. Use this instead of re-running a "
        "large query or catting raw JSON. Args: ref (the tool_result:* ref), "
        "offset (default 0), max_chars (default 4000, hard-capped). The "
        "response includes next_offset when more text remains.",
        {
            "ref": str,
            "offset": int,
            "max_chars": int,
        },
    )(_handle_read_tool_result)

    wait_for_file = tool(
        "wait_for_file",
        "Block until a file (or one/all of several files) exists and meets a "
        "size/content condition, OR until timeout/closing. Use this only for a "
        "task-owned result or progress file whose readiness contract is explicitly "
        "documented. Never use it on a runtime-managed `tasks/<id>.output` path: "
        "empty output can be a successful task, and the runtime's structured task "
        "notification is the completion source of truth. One MCP "
        "round-trip replaces dozens of Bash poll calls and saves LLM context "
        "on long evidence runs. "
        "Args: 'path' (file path, or comma-separated list); "
        "'timeout_seconds' (int, default 3600, capped by "
        "PRAXIST_WAIT_FOR_FILE_MAX_SECONDS; default cap 18h, bounded to 24h); "
        "'poll_interval_seconds' (legacy fallback lower bound); "
        "'min_bytes' (int, default 1 — file must be non-empty); "
        "'contains_text' (str, optional substring that file content must contain); "
        "'mode' ('any' or 'all', default 'any' — return on first match vs wait for all). "
        "Returns JSON with readiness/release status, elapsed_seconds, matched_paths, missing_paths.",
        {
            "path": str,
            "timeout_seconds": int,
            "poll_interval_seconds": int,
            "min_bytes": int,
            "contains_text": str,
            "mode": str,
        },
    )(_handle_wait_for_file)


# ---------------------------------------------------------------------------
# Leaderboard helpers (used by both MCP handler and external callers)
# ---------------------------------------------------------------------------


def _parse_anchor_metrics_env() -> list[dict[str, str]]:
    """Read ANCHOR_METRICS env var as JSON list of {name, direction}.

    Empty / malformed → []. The leaderboard falls back to a single-axis
    ranking when no anchors are configured (preserves compatibility with
    older task specs that don't define `evaluation.anchor_metrics`).
    """
    raw = _get_env("ANCHOR_METRICS", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("ANCHOR_METRICS env var is not valid JSON: %s", e)
        return []
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        d = item.get("direction", "maximize")
        if d not in ("maximize", "minimize"):
            # Round 1 N1 fix: drop with warning instead of silently
            # flipping to maximize (parity with the spec-layer parser
            # in task_spec._normalize_anchor_metrics, which also drops).
            logger.warning(
                "ANCHOR_METRICS axis %r has invalid direction %r — dropped "
                "(must be 'maximize' or 'minimize').",
                name,
                d,
            )
            continue
        out.append({"name": name, "direction": d})
    return out


def _sqlite_leaderboard(generation: int, top_k: int) -> str:
    """Build leaderboard from shared SQLite store.

    When ANCHOR_METRICS env is set, returns a Pareto-frontier leaderboard
    across primary + anchor metrics (Praxist v2026-05-01+). Peers
    receive a structured response with the non-dominated frontier
    explicitly separated from dominated entries, so a single task metric
    cannot monopolize the ranking when other configured axes matter.
    """
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
            get_leaderboard as db_leaderboard,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
            get_pareto_leaderboard,
            init_db,
        )

        init_db()

        gen_id = generation if generation >= 0 else None
        primary_metric = _get_env("PRIMARY_METRIC", "metric_value")
        direction = _get_env("METRIC_DIRECTION", "maximize")
        anchors = _parse_anchor_metrics_env()

        if anchors:
            # R7 M2 fix: default to "false" to match
            # EvaluationSpec.requires_tier default. The orchestrator
            # always sets the env explicitly per task spec, so live
            # behavior is unchanged; this only affects tools that call
            # _sqlite_leaderboard outside GenerationLoop.run().
            requires_tier = _get_env("REQUIRES_TIER", "false").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            result = get_pareto_leaderboard(
                primary_metric=primary_metric,
                direction=direction,
                anchor_metrics=anchors,
                generation_id=gen_id,
                top_k_dominated=top_k,
                requires_tier=requires_tier,
            )
            return json.dumps(
                {
                    "mode": "pareto",
                    "axes": result["axes"],
                    "n_total": result["n_total"],
                    "n_pareto": result["n_pareto"],
                    "n_dominated_total": result.get("n_dominated_total"),
                    "n_excluded_missing_axis": result.get("n_excluded_missing_axis", {}),
                    "best_in": result["best_in"],
                    "pareto_front": result["pareto_front"],
                    "dominated_top": result["dominated_top"],
                    "source": "sqlite",
                    "note": (
                        "Pareto front = variants that NO other variant beats on "
                        "every axis. Each entry's `best_in` lists axes it leads. "
                        "`dominated_top` is truncated to top-K by primary; the "
                        "full count is in `n_dominated_total`."
                    ),
                },
                indent=2,
            )

        # Single-axis legacy fallback (no anchors configured)
        entries = db_leaderboard(
            primary_metric=primary_metric,
            direction=direction,
            generation_id=gen_id,
            top_k=top_k,
        )
        return json.dumps(
            {
                "mode": "single_metric",
                "primary_metric": primary_metric,
                "direction": direction,
                "entries": [
                    {
                        "variant_name": e.get("variant_name", ""),
                        "metrics": e.get("metrics", {}),
                        "generation_id": e.get("generation_id"),
                        "peer_id": e.get("peer_id", ""),
                        "title": e.get("title", ""),
                    }
                    for e in entries
                ],
                "source": "sqlite",
            },
            indent=2,
        )
    except Exception as e:
        # R5 M1 fix: log at WARNING with exc_info instead of DEBUG. The
        # SQLite path is the canonical Pareto surface — silent fallback
        # to filesystem mode on a real bug masks the regression for
        # hours of peer compute. Surface it loudly so operators notice.
        logger.warning(
            "SQLite leaderboard failed, falling back to filesystem: %s", e, exc_info=True
        )
        return _filesystem_leaderboard(generation, top_k)


def _filesystem_leaderboard(generation: int, top_k: int) -> str:
    """Fallback: build leaderboard from filesystem JSON files.

    Used when the SQLite store is unreachable. M2 fix (review round 1):
    always emit a `mode` field so peers can branch on response shape
    (the prompt instructs them to check `mode` first).
    """
    findings_dir = _findings_dir_from_runtime_env()
    if not findings_dir.exists():
        return json.dumps(
            {
                "mode": "filesystem_fallback",
                "entries": [],
                "note": "No findings directory found",
            }
        )

    # R4 Issue 2 fix: filesystem fallback fires only when SQLite is
    # unhealthy — peers need *some* leaderboard signal, not an empty
    # list. Apply only the same coarse filters as the legacy code
    # (finding_type=="result", generation match) and use the new
    # `degraded_filtering` flag to flag the relaxed contract. The
    # cost of a few stale entries leaking through is low compared
    # to leaving peers with zero leaderboard data when storage is down.
    primary_metric = _get_env("PRIMARY_METRIC", "metric_value")
    direction = _get_env("METRIC_DIRECTION", "maximize")
    entries = []
    for f_path in findings_dir.glob("*.json"):
        try:
            with open(f_path) as f:
                finding = json.load(f)
            # R8 M2 fix: accept both `result` and `insight` (parity with
            # the SQLite Pareto path's R6 Issue 3 fix). Otherwise
            # SQLite-vs-fallback silently disagree on insight findings.
            if finding.get("finding_type") not in ("result", "insight"):
                continue
            if generation >= 0 and finding.get("generation_id") != generation:
                continue
            # Best-effort fallback does not interpret tier labels. Tier is
            # opaque task metadata; filtering relies on generic promotion flags
            # below, matching the core frontier contract.
            m = finding.get("metrics") or {}
            t_raw = m.get("tier")
            if t_raw is None:
                t_raw = (finding.get("details") or {}).get("tier")
            if t_raw is None:
                t_raw = finding.get("tier")
            # R5 m2/m5 fix: include `title`, `tier`, and
            # `promotion_eligible` in the entry so peers can post-filter
            # in degraded mode and so `e["title"]` doesn't KeyError.
            elig = m.get("promotion_eligible")
            if elig is None:
                elig = (finding.get("details") or {}).get("promotion_eligible")
            if elig is None:
                elig = finding.get("promotion_eligible")
            entries.append(
                {
                    "variant_name": finding.get("variant_name", ""),
                    "metrics": finding.get("metrics", {}),
                    "title": finding.get("title", ""),
                    # R6 Issue 6 fix: preserve tier as-is (incl. int/None)
                    # so peers can distinguish "malformed" from "missing"
                    # in degraded mode.
                    "tier": t_raw,
                    "promotion_eligible": elig,
                    "generation_id": finding.get("generation_id"),
                    "peer_id": finding.get("peer_id", ""),
                }
            )
        except (json.JSONDecodeError, KeyError):
            continue

    # R5 M2 fix: sort by primary metric before truncating so the
    # `entries[:top_k]` slice is the actual top-K (not a glob-arbitrary
    # K). The prompt promises "ranked by primary metric only" for this
    # mode — was previously broken because filesystem glob is arbitrary.
    rev = direction == "maximize"

    def _sort_key(e):
        v = (e.get("metrics") or {}).get(primary_metric)
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
            # Sort missing-primary entries to the end (regardless of dir)
            return (1, 0.0, str(e.get("variant_name") or ""))
        return (0, -float(v) if rev else float(v), str(e.get("variant_name") or ""))

    entries.sort(key=_sort_key)

    return json.dumps(
        {
            "mode": "filesystem_fallback",
            "entries": entries[:top_k],
            "source": "filesystem",
            "degraded_filtering": True,
            "note": (
                "SQLite store unreachable; this is a degraded response — "
                "tier labels are opaque and missing-tier / "
                "promotion_eligible=False entries may leak through. "
                "Each entry now carries `tier` and `promotion_eligible` so "
                "peers can post-filter. Treat results as advisory until "
                "SQLite recovers."
            ),
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# MCP server factory
# ---------------------------------------------------------------------------


def create_evaluation_tools_server():
    """Create MCP server for evaluation tools."""
    if create_sdk_mcp_server is None or tool is None:
        raise ImportError("claude_agent_sdk is required for MCP tools")
    return create_sdk_mcp_server(
        "evaluation-tools",
        tools=[
            log_experiment_metrics,
            share_finding,
            get_leaderboard,
            wait_for_file,
            read_tool_result,
        ],
    )


def create_tool_plugin() -> dict[str, object]:
    """Manifest entrypoint that exposes evaluation and experiment logging tools."""
    return {
        "tool_server_ref": "tool_server:evaluation_tools",
        "server_name": "evaluation-tools",
        "factory": "praxist.plugins.tools.evaluation_tools.adapter:create_evaluation_tools_server",
        "tool_names": [
            "log_experiment_metrics",
            "share_finding",
            "get_leaderboard",
            "wait_for_file",
            "read_tool_result",
        ],
        "visibility": ["peer", "panel"],
        "required_capability": "tool_server.evaluation_tools",
        "handlers": {
            "log_experiment_metrics": "praxist.plugins.tools.evaluation_tools.adapter:_handle_log_experiment_metrics",
            "share_finding": "praxist.plugins.tools.evaluation_tools.adapter:_handle_share_finding",
            "get_leaderboard": "praxist.plugins.tools.evaluation_tools.adapter:_handle_get_leaderboard",
            "wait_for_file": "praxist.plugins.tools.evaluation_tools.adapter:_handle_wait_for_file",
            "read_tool_result": "praxist.plugins.tools.evaluation_tools.adapter:_handle_read_tool_result",
        },
    }
