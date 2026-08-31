"""Peer-local structured memory for multi-session research loops.

This module keeps bounded, task-agnostic state between sessions of the same
peer. It intentionally stores summaries and ledgers rather than raw transcripts:
the goal is to preserve continuity without turning every later session into a
long-context replay of prior chat.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from praxist.core.redaction import JSONValue, redact_json, redact_text
from praxist.core.storage import atomic_write_bytes, open_append_file, open_readonly_file
from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
    iter_result_summary_paths,
)
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import is_scoreless

MEMORY_HEADER = "Praxist Peer-Local Structured Memory"
DEFAULT_MAX_RECENT_EXPERIMENTS = 5
DEFAULT_MAX_SHARED_FINDINGS = 6
DEFAULT_MAX_PROMPT_CHARS = 12_000
DEFAULT_MAX_MEMORY_FILE_BYTES = 256_000
DEFAULT_MAX_LEDGER_FILE_BYTES = 512_000
DEFAULT_MAX_HANDOFF_BYTES = 64_000
DEFAULT_MAX_EXTERNAL_JSON_BYTES = 2_000_000
DEFAULT_MAX_SESSION_SNAPSHOTS = 8
_SAFE_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_CANONICAL_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PEER_HEALTH_RED = "red"
PEER_HEALTH_YELLOW = "yellow"
PEER_HEALTH_GREEN = "green"
PEER_HEALTH_VALUES = (PEER_HEALTH_RED, PEER_HEALTH_YELLOW, PEER_HEALTH_GREEN)


@dataclass(frozen=True)
class PeerMemoryConfig:
    """Runtime knobs for bounded peer-local memory."""

    enabled: bool = True
    max_recent_experiments: int = DEFAULT_MAX_RECENT_EXPERIMENTS
    max_shared_findings: int = DEFAULT_MAX_SHARED_FINDINGS
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS
    max_session_snapshots: int = DEFAULT_MAX_SESSION_SNAPSHOTS
    track_finding_content_versions: bool = False


@dataclass(frozen=True)
class PeerHealthStatus:
    """Read-only status aggregate for one peer's memory and evidence artifacts."""

    peer_id: str
    generation_id: int
    health: str
    health_reason: str
    research_state: str = ""
    active_variant: str = ""
    last_session_id: str = ""
    last_session_success: bool | None = None
    last_updated_utc: str = ""
    baseline_status: str = "unknown"
    primary_metric: str = ""
    best_metric_value: float | None = None
    baseline_metric_value: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PeerHealthSnapshot:
    """Peer-level health view for a generation."""

    generation_id: int | None
    summary: dict[str, int]
    peers: list[PeerHealthStatus]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "summary": dict(self.summary),
            "peers": [peer.to_dict() for peer in self.peers],
            "warnings": list(self.warnings),
        }


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=0o600)


def _read_json(
    path: Path,
    default: Any,
    *,
    max_bytes: int = DEFAULT_MAX_EXTERNAL_JSON_BYTES,
) -> Any:
    try:
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
            return default
        with open_readonly_file(path) as stream:
            opened_st = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_st.st_mode) or opened_st.st_size > max_bytes:
                return default
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            return default
        return json.loads(data.decode("utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return default
    return default


def read_bounded_file_under_root_no_follow(
    path: Path,
    root: Path,
    *,
    max_bytes: int,
) -> bytes | None:
    """Read a regular file under ``root`` without following path symlinks."""

    if not hasattr(os, "O_NOFOLLOW"):
        return None
    if os.open not in getattr(os, "supports_dir_fd", set()):
        return None
    root = Path(root)
    path = Path(path)
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    if (
        not rel.parts
        or any(part in {"", ".", ".."} for part in rel.parts)
        or path.is_absolute() != root.is_absolute()
    ):
        return None
    nofollow = os.O_NOFOLLOW
    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    file_flags = os.O_RDONLY | nofollow | os.O_NONBLOCK
    fd = -1
    current_fd = -1
    try:
        current_fd = os.open(root, dir_flags)
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            return None
        for component in rel.parts[:-1]:
            next_fd = os.open(component, dir_flags, dir_fd=current_fd)
            if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                os.close(next_fd)
                return None
            os.close(current_fd)
            current_fd = next_fd
        fd = os.open(rel.parts[-1], file_flags, dir_fd=current_fd)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size > max_bytes:
            return None
        data = os.read(fd, max_bytes + 1)
        if len(data) > max_bytes:
            return None
        return data
    except OSError:
        return None
    finally:
        if fd != -1:
            with suppress(OSError):
                os.close(fd)
        if current_fd != -1:
            with suppress(OSError):
                os.close(current_fd)


def _read_json_under_root(
    path: Path,
    root: Path,
    default: Any,
    *,
    max_bytes: int = DEFAULT_MAX_EXTERNAL_JSON_BYTES,
) -> Any:
    data = read_bounded_file_under_root_no_follow(path, root, max_bytes=max_bytes)
    if data is None:
        return default
    try:
        return json.loads(data.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def _write_yaml(path: Path, data: Any) -> None:
    _atomic_write_text(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def _shorten(value: Any, *, limit: int = 220) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _redact_for_memory(value: Any, *, limit: int = 220) -> str:
    text, _ = redact_text(str(value or ""))
    return _shorten(text, limit=limit)


def _redact_value(value: Any) -> Any:
    redacted, _ = redact_json(_json_compatible(value))
    return redacted


def _finding_seen_key(finding_id: Any) -> str:
    raw = str(finding_id or "")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _finding_version_seen_key(finding_id: Any, payload: dict[str, Any]) -> str:
    source_hash = _finding_source_result_hash(payload)
    if source_hash is not None:
        normalized = dict(payload)
        normalized.pop("timestamp", None)
        artifact_semantics = normalized.get("artifact_semantics")
        if isinstance(artifact_semantics, dict):
            normalized_semantics = dict(artifact_semantics)
            normalized_semantics.pop("created_at", None)
            normalized["artifact_semantics"] = normalized_semantics
        version_payload = json.dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        version = f"source_result_sha256:{source_hash}\0{version_payload}"
    else:
        version = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    raw = f"{finding_id}\0{version}"
    return f"sha256v2:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _finding_source_result_hash(payload: dict[str, Any]) -> str | None:
    metrics = payload.get("metrics")
    artifact_semantics = payload.get("artifact_semantics")
    auto_materialized = isinstance(metrics, dict) and (
        metrics.get("auto_materialized_from_result_artifact") is True
    )
    derived_result_reference = (
        isinstance(artifact_semantics, dict)
        and artifact_semantics.get("role") == "derived_view"
        and artifact_semantics.get("stage") == "result_finding_reference"
    )
    if not (auto_materialized or derived_result_reference):
        return None
    for container in (
        payload,
        metrics,
        payload.get("extra"),
        payload.get("details"),
        payload.get("current_aggregate"),
    ):
        if not isinstance(container, dict):
            continue
        value = container.get("source_result_sha256")
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value.strip()):
            return value.strip().lower()
    return None


def _explicit_finding_id(payload: dict[str, Any]) -> str | None:
    for key in ("finding_id", "id", "finding_key"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _json_compatible(value: Any) -> JSONValue:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    return str(value)


def _metric_subset(source: Any, *, max_items: int = 10) -> dict[str, Any]:
    """Return a compact task-agnostic metric subset from an arbitrary mapping."""

    if not isinstance(source, dict):
        return {}
    preferred_fragments = (
        "score",
        "metric",
        "loss",
        "accuracy",
        "objective",
        "value",
        "rank",
        "cell",
        "seed",
        "status",
        "violation",
        "stage",
        "tier",
    )
    compact: dict[str, Any] = {}
    for key, value in source.items():
        key_text = str(key)
        lower = key_text.lower()
        if not any(fragment in lower for fragment in preferred_fragments):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key_text] = value
        elif isinstance(value, (list, tuple)):
            compact[key_text] = [_shorten(item, limit=80) for item in value[:4]]
        elif isinstance(value, dict):
            nested = _metric_subset(value, max_items=4)
            if nested:
                compact[key_text] = nested
        if len(compact) >= max_items:
            break
    return compact


def _extract_variant_name(path: Path, payload: dict[str, Any]) -> str:
    for key in ("variant_name", "variant", "name", "candidate", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return path.parent.name


def _matches_peer(variant_name: str, peer_id: str, generation_id: int) -> bool:
    lower_variant = variant_name.lower()
    lower_peer = peer_id.lower()
    match = re.fullmatch(r"gen_?(\d+)_peer_?(\d+)", lower_peer)
    if match:
        gen, peer = match.groups()
        if int(gen) != int(generation_id):
            return False
        return bool(
            re.search(
                rf"(?<![a-z0-9])gen_?{int(gen)}_peer_?{int(peer)}(?![0-9])",
                lower_variant,
            )
        )
    escaped = re.escape(lower_peer)
    return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", lower_variant))


def _safe_path_component(value: str, *, fallback: str = "peer") -> str:
    raw = str(value or "").strip()
    unsafe = (
        raw != str(value or "")
        or not raw
        or Path(raw).is_absolute()
        or "/" in raw
        or "\\" in raw
        or any(part == ".." for part in re.split(r"[\\/]+", raw))
        or _CANONICAL_PATH_COMPONENT_RE.fullmatch(raw) is None
    )
    if unsafe:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return f"unsafe-{digest}"
    return raw or fallback


def _safe_result_summary_paths(run_dir: Path) -> list[Path]:
    """Return supported result summary paths without following directory symlinks."""

    results_dir = Path(run_dir) / "results"
    try:
        root_stat = results_dir.lstat()
    except OSError:
        return []
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        return []
    try:
        results_root = results_dir.resolve(strict=True)
    except OSError:
        return []

    paths: dict[str, Path] = {}

    def _add_if_safe(path: Path) -> None:
        try:
            file_stat = path.lstat()
        except OSError:
            return
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            return
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(results_root)
        except (OSError, ValueError):
            return
        paths[str(resolved)] = path

    for path in iter_result_summary_paths(run_dir):
        _add_if_safe(path)

    return sorted(paths.values(), key=lambda path: str(path))


def collect_peer_memory_health(
    *,
    run_dir: Path,
    generation_id: int | None,
    primary_metric: str = "metric_value",
    direction: str = "maximize",
    baselines: list[Any] | tuple[Any, ...] | None = None,
    scan_result_artifacts: bool = True,
    task_spec: Any = None,
) -> PeerHealthSnapshot:
    """Return peer-level health derived from peer memory and result summaries.

    This is a read-only operator/status helper. It deliberately consumes the
    peer-local structured memory surface instead of letting CLI code parse those
    files ad hoc. High-frequency read-only views may skip the recursive result
    reconciliation and rely on the peer-owned ``recent_result_artifacts``
    summary; the default retains the complete status/diagnostic scan.
    An explicitly scoreless task reports operational session health and marks
    numerical baseline comparisons as not applicable.
    """

    run_dir = Path(run_dir).expanduser().resolve(strict=False)
    resolved_generation, gen_dir = _resolve_generation_dir(run_dir, generation_id)
    if gen_dir is None or resolved_generation is None:
        return PeerHealthSnapshot(
            generation_id=resolved_generation,
            summary=_empty_health_summary(),
            peers=[],
            warnings=["generation directory unavailable"],
        )

    scoreless = is_scoreless(task_spec)
    if scoreless:
        primary_metric = ""
    baseline_value = (
        None
        if scoreless
        else _baseline_threshold(
            baselines or [],
            primary_metric=primary_metric,
            direction=direction,
        )
    )
    peer_roots = _safe_peer_roots(gen_dir)
    scanned_by_peer = (
        _scan_result_artifacts_for_peers(
            run_dir=run_dir,
            peer_ids=[peer_root.name for peer_root in peer_roots],
            generation_id=resolved_generation,
            primary_metric=primary_metric,
        )
        if scan_result_artifacts and peer_roots and not scoreless
        else {}
    )
    peers: list[PeerHealthStatus] = []
    warnings: list[str] = []
    for peer_root in peer_roots:
        peer = _collect_one_peer_health(
            run_dir=run_dir,
            peer_root=peer_root,
            generation_id=resolved_generation,
            primary_metric=primary_metric,
            direction=direction,
            baseline_value=baseline_value,
            result_artifacts=scanned_by_peer.get(peer_root.name, []),
            scoreless=scoreless,
        )
        peers.append(peer)
    peers.sort(key=lambda peer: peer.peer_id)
    return PeerHealthSnapshot(
        generation_id=resolved_generation,
        summary=summarize_peer_health(peers),
        peers=peers,
        warnings=warnings,
    )


def summarize_peer_health(peers: list[PeerHealthStatus]) -> dict[str, int]:
    """Count peer health levels in a stable JSON shape."""

    summary = _empty_health_summary()
    for peer in peers:
        if peer.health in summary:
            summary[peer.health] += 1
    return summary


def _empty_health_summary() -> dict[str, int]:
    return {PEER_HEALTH_RED: 0, PEER_HEALTH_YELLOW: 0, PEER_HEALTH_GREEN: 0}


def _resolve_generation_dir(
    run_dir: Path, generation_id: int | None
) -> tuple[int | None, Path | None]:
    candidates: list[tuple[int, Path]] = []
    if generation_id is not None:
        try:
            gen = int(generation_id)
        except (TypeError, ValueError):
            return None, None
        candidates = [(gen, run_dir / f"gen_{gen}"), (gen, run_dir / f"gen{gen}")]
    else:
        try:
            children = sorted(run_dir.iterdir(), key=lambda item: item.name)
        except OSError:
            children = []
        for child in children:
            match = re.fullmatch(r"gen_?(\d+)", child.name)
            if match:
                candidates.append((int(match.group(1)), child))
        candidates.sort(key=lambda item: item[0], reverse=True)

    for gen, path in candidates:
        try:
            st = path.lstat()
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                continue
            path.resolve(strict=True).relative_to(run_dir.resolve(strict=True))
        except (OSError, ValueError):
            continue
        return gen, path
    return generation_id, None


def _safe_peer_roots(gen_dir: Path) -> list[Path]:
    peers_root = gen_dir / "peers"
    try:
        root_stat = peers_root.lstat()
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            return []
        peers_resolved = peers_root.resolve(strict=True)
        children = sorted(peers_root.iterdir(), key=lambda item: item.name)
    except OSError:
        return []
    peer_roots: list[Path] = []
    for child in children:
        try:
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISDIR(child_stat.st_mode):
                continue
            child.resolve(strict=True).relative_to(peers_resolved)
        except (OSError, ValueError):
            continue
        peer_roots.append(child)
    return peer_roots


def _collect_one_peer_health(
    *,
    run_dir: Path,
    peer_root: Path,
    generation_id: int,
    primary_metric: str,
    direction: str,
    baseline_value: float | None,
    result_artifacts: list[dict[str, Any]],
    scoreless: bool = False,
) -> PeerHealthStatus:
    peer_id = peer_root.name
    memory_dir = peer_root / "memory"
    state, state_error = _read_peer_state(memory_dir / "peer_state.yaml", run_dir)
    ledger_rows, ledger_warning = _read_recent_ledger(
        memory_dir / "experiment_ledger.jsonl", run_dir
    )
    artifacts = [] if scoreless else list(result_artifacts)
    if not scoreless:
        artifacts.extend(_state_result_artifacts(state, primary_metric=primary_metric))
    best_value = _best_metric_value(
        [artifact.get("metric_value") for artifact in artifacts],
        direction=direction,
    )
    best_variant = next(
        (
            str(artifact.get("variant_name") or "")
            for artifact in artifacts
            if artifact.get("variant_name")
        ),
        "",
    )
    warnings = [warning for warning in (state_error, ledger_warning) if warning]
    last_ledger = ledger_rows[-1] if ledger_rows else {}

    last_success = state.get("last_session_success")
    if not isinstance(last_success, bool):
        ledger_success = last_ledger.get("success")
        last_success = ledger_success if isinstance(ledger_success, bool) else None

    research_state = _redact_for_memory(state.get("research_state", ""))
    active_variant = _redact_for_memory(state.get("active_variant", "")) or best_variant
    baseline_status = (
        "not_applicable"
        if scoreless
        else _baseline_status(
            best_metric_value=best_value,
            baseline_value=baseline_value,
            direction=direction,
        )
    )
    health, reason = _peer_health_level(
        state_error=state_error,
        state=state,
        last_session_success=last_success,
        baseline_status=baseline_status,
        scoreless=scoreless,
    )
    return PeerHealthStatus(
        peer_id=peer_id,
        generation_id=generation_id,
        health=health,
        health_reason=reason,
        research_state=research_state,
        active_variant=active_variant,
        last_session_id=_redact_for_memory(
            state.get("last_session_id", last_ledger.get("session_id", ""))
        ),
        last_session_success=last_success,
        last_updated_utc=_redact_for_memory(state.get("last_updated_utc", "")),
        baseline_status=baseline_status,
        primary_metric=str(primary_metric or ""),
        best_metric_value=best_value,
        baseline_metric_value=baseline_value,
        warnings=warnings,
    )


def _read_peer_state(path: Path, run_dir: Path) -> tuple[dict[str, Any], str | None]:
    try:
        st = path.lstat()
    except FileNotFoundError:
        return {}, "missing peer_state.yaml"
    except OSError as exc:
        return {}, f"peer_state.yaml unreadable: {exc}"
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        return {}, "unsafe peer_state.yaml"
    if st.st_size > DEFAULT_MAX_MEMORY_FILE_BYTES:
        return {}, "peer_state.yaml too large"
    data = read_bounded_file_under_root_no_follow(
        path,
        run_dir,
        max_bytes=DEFAULT_MAX_MEMORY_FILE_BYTES,
    )
    if data is None:
        return {}, "peer_state.yaml unreadable"
    try:
        loaded = yaml.safe_load(data.decode("utf-8", errors="replace"))
    except (yaml.YAMLError, TypeError, ValueError):
        return {}, "peer_state.yaml malformed"
    if not isinstance(loaded, dict):
        return {}, "peer_state.yaml malformed"
    redacted = _redact_value(loaded)
    if not isinstance(redacted, dict):
        return {}, "peer_state.yaml malformed"
    return redacted, None


def _read_recent_ledger(path: Path, run_dir: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        st = path.lstat()
    except FileNotFoundError:
        return [], None
    except OSError as exc:
        return [], f"experiment_ledger.jsonl unreadable: {exc}"
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        return [], "unsafe experiment_ledger.jsonl"
    if st.st_size > DEFAULT_MAX_LEDGER_FILE_BYTES:
        return [], "experiment_ledger.jsonl too large"
    data = read_bounded_file_under_root_no_follow(
        path,
        run_dir,
        max_bytes=DEFAULT_MAX_LEDGER_FILE_BYTES,
    )
    if data is None:
        return [], "experiment_ledger.jsonl unreadable"
    rows: list[dict[str, Any]] = []
    warning: str | None = None
    for line in data.decode("utf-8", errors="replace").splitlines()[-25:]:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            warning = "experiment_ledger.jsonl has malformed rows"
            continue
        if isinstance(row, dict):
            redacted = _redact_value(row)
            if isinstance(redacted, dict):
                rows.append(redacted)
    return rows, warning


def _scan_peer_result_artifacts(
    *,
    run_dir: Path,
    peer_id: str,
    generation_id: int,
    primary_metric: str,
) -> list[dict[str, Any]]:
    return _scan_result_artifacts_for_peers(
        run_dir=run_dir,
        peer_ids=[peer_id],
        generation_id=generation_id,
        primary_metric=primary_metric,
    ).get(peer_id, [])


def _scan_result_artifacts_for_peers(
    *,
    run_dir: Path,
    peer_ids: list[str],
    generation_id: int,
    primary_metric: str,
) -> dict[str, list[dict[str, Any]]]:
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
            result_summary_variant_name,
        )
    except Exception:
        return {}
    results_root = run_dir / "results"
    artifacts: dict[str, list[tuple[float, dict[str, Any]]]] = {peer_id: [] for peer_id in peer_ids}
    for path in _safe_result_summary_paths(run_dir):
        payload = _read_json_under_root(path, results_root, None)
        if not isinstance(payload, dict):
            continue
        try:
            variant_name = result_summary_variant_name(path, payload, run_dir)
        except Exception:
            variant_name = _extract_variant_name(path, payload)
        metric_value = _extract_metric_value(payload, primary_metric)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        artifact = {
            "variant_name": variant_name,
            "path": str(path),
            "metric_value": metric_value,
        }
        for peer_id in peer_ids:
            if _matches_peer(variant_name, peer_id, generation_id):
                artifacts[peer_id].append((mtime, artifact))
    return {
        peer_id: [artifact for _, artifact in sorted(rows, key=lambda item: item[0], reverse=True)]
        for peer_id, rows in artifacts.items()
    }


def _state_result_artifacts(state: dict[str, Any], *, primary_metric: str) -> list[dict[str, Any]]:
    raw = state.get("recent_result_artifacts")
    if not isinstance(raw, list):
        return []
    artifacts: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics")
        metric_value = _extract_metric_value(metrics, primary_metric)
        artifacts.append(
            {
                "variant_name": item.get("variant_name") or item.get("summary") or "",
                "metric_value": metric_value,
            }
        )
    return artifacts


def _extract_metric_value(payload: Any, primary_metric: str) -> float | None:
    if not primary_metric:
        primary_metric = "metric_value"
    raw = _find_nested_key(payload, primary_metric, max_depth=4)
    if raw is None and primary_metric != "metric_value":
        raw = _find_nested_key(payload, "metric_value", max_depth=4)
    return _finite_float(raw)


def _find_nested_key(source: Any, key: str, *, max_depth: int) -> Any:
    if max_depth < 0 or not isinstance(source, dict):
        return None
    if key in source:
        return source.get(key)
    for nested_key in ("metrics", "current_aggregate", "aggregate", "summary", "result"):
        nested = source.get(nested_key)
        found = _find_nested_key(nested, key, max_depth=max_depth - 1)
        if found is not None:
            return found
    return None


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _baseline_threshold(
    baselines: list[Any] | tuple[Any, ...],
    *,
    primary_metric: str,
    direction: str,
) -> float | None:
    values: list[float] = []
    fallback_values: list[float] = []
    for baseline in baselines:
        metric_name = str(
            _get_mapping_or_attr(baseline, "metric_name", default=primary_metric) or primary_metric
        )
        value = _finite_float(
            _get_mapping_or_attr(
                baseline,
                "metric_value",
                default=_get_mapping_or_attr(baseline, "expected_acc", default=None),
            )
        )
        if value is None:
            continue
        fallback_values.append(value)
        if not primary_metric or metric_name == primary_metric:
            values.append(value)
    selected = values or fallback_values
    if not selected:
        return None
    return min(selected) if direction == "minimize" else max(selected)


def _get_mapping_or_attr(value: Any, key: str, *, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _best_metric_value(values: list[Any], *, direction: str) -> float | None:
    numeric = [value for value in (_finite_float(item) for item in values) if value is not None]
    if not numeric:
        return None
    return min(numeric) if direction == "minimize" else max(numeric)


def _baseline_status(
    *,
    best_metric_value: float | None,
    baseline_value: float | None,
    direction: str,
) -> str:
    if baseline_value is None:
        return "baseline_unavailable"
    if best_metric_value is None:
        return "no_primary_metric_result"
    reached = (
        best_metric_value <= baseline_value
        if direction == "minimize"
        else best_metric_value >= baseline_value
    )
    return "reached_baseline" if reached else "below_baseline"


def _peer_health_level(
    *,
    state_error: str | None,
    state: dict[str, Any],
    last_session_success: bool | None,
    baseline_status: str,
    scoreless: bool = False,
) -> tuple[str, str]:
    if state_error:
        return PEER_HEALTH_RED, state_error
    if last_session_success is False:
        return PEER_HEALTH_RED, "last session failed"
    if state.get("last_error"):
        return PEER_HEALTH_RED, "last session recorded an error"
    research_state = str(state.get("research_state", "")).strip().lower()
    if any(token in research_state for token in ("fail", "error", "blocked", "fault")):
        return PEER_HEALTH_RED, f"research_state={research_state}"
    if scoreless:
        if last_session_success is True:
            return PEER_HEALTH_GREEN, "last session completed; research quality not evaluated"
        return PEER_HEALTH_YELLOW, "no completed session recorded; research quality not evaluated"
    if baseline_status == "reached_baseline":
        return PEER_HEALTH_GREEN, "baseline reached"
    if baseline_status == "below_baseline":
        return PEER_HEALTH_YELLOW, "below baseline"
    if baseline_status == "no_primary_metric_result":
        return PEER_HEALTH_YELLOW, "no primary metric result yet"
    return PEER_HEALTH_YELLOW, "baseline unavailable"


class PeerSessionMemory:
    """Maintain and render bounded state for one peer across sessions."""

    def __init__(
        self,
        *,
        run_dir: Path,
        gen_dir: Path,
        peer_id: str,
        generation_id: int,
        findings_dir: Path,
        config: PeerMemoryConfig | None = None,
        task_spec: Any = None,
    ) -> None:
        self._declared_run_dir = Path(run_dir).expanduser()
        self.run_dir = self._declared_run_dir.resolve(strict=False)
        self.gen_dir = Path(gen_dir).expanduser().resolve(strict=False)
        self.peer_id = peer_id
        self.safe_peer_id = _safe_path_component(peer_id)
        self.generation_id = generation_id
        self.findings_dir = Path(findings_dir).expanduser()
        self.config = config or PeerMemoryConfig()
        self.task_spec = task_spec
        self.peers_root = self.gen_dir / "peers"
        self.peer_root = self.peers_root / self.safe_peer_id
        self.memory_dir = self.peer_root / "memory"
        try:
            self.peer_root.relative_to(self.peers_root)
            self.memory_dir.relative_to(self.peers_root)
        except ValueError as exc:
            raise ValueError(f"unsafe peer memory path for peer_id={peer_id!r}") from exc
        self.state_path = self.memory_dir / "peer_state.yaml"
        self.ledger_path = self.memory_dir / "experiment_ledger.jsonl"
        self.handoff_path = self.memory_dir / "session_handoff.md"
        self.auto_handoff_path = self.memory_dir / "session_auto_handoff.md"
        self.seen_findings_path = self.memory_dir / "seen_shared_findings.json"
        self.prompt_snapshot_path = self.memory_dir / "memory_prompt.md"
        self.prompt_manifest_path = self.memory_dir / "session_prompt_manifest.json"
        self._last_prompt_finding_keys: list[str] = []

    def initialize(self) -> None:
        """Create the peer's initial state before any long prelaunch stage."""

        if not self.config.enabled:
            return
        self._ensure_memory_dir()
        self._load_or_initialize_state()

    def _ensure_dir_no_symlink(self, path: Path) -> None:
        try:
            base_stat = self._declared_run_dir.lstat()
        except FileNotFoundError:
            self._declared_run_dir.mkdir(parents=True, exist_ok=True)
            base_stat = self._declared_run_dir.lstat()
        if stat.S_ISLNK(base_stat.st_mode) or not stat.S_ISDIR(base_stat.st_mode):
            raise OSError(f"unsafe peer memory run directory: {self._declared_run_dir}")
        base = self.run_dir.resolve(strict=True)
        target = path.expanduser().resolve(strict=False)
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise OSError(f"unsafe peer memory directory outside run: {path}") from exc
        current = base
        for component in target.relative_to(base).parts:
            current = current / component
            try:
                st = current.lstat()
            except FileNotFoundError:
                current.mkdir()
                st = current.lstat()
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                raise OSError(f"unsafe peer memory directory: {current}")

    def _ensure_memory_dir(self) -> None:
        self._ensure_dir_no_symlink(self.peers_root)
        self._ensure_dir_no_symlink(self.peer_root)
        self._ensure_dir_no_symlink(self.memory_dir)
        resolved_root = self.memory_dir.resolve(strict=True)
        resolved_root.relative_to(self.peers_root.resolve(strict=True))

    def _safe_existing_file(self, path: Path, root: Path) -> bool:
        try:
            st = path.lstat()
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            return False
        try:
            path.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (OSError, ValueError):
            return False
        return True

    def _open_existing_file_no_follow(self, path: Path, root: Path) -> int | None:
        if not self._safe_existing_file(path, root):
            return None
        try:
            with open_readonly_file(path) as stream:
                return os.dup(stream.fileno())
        except OSError:
            return None

    def _read_existing_file_no_follow(
        self,
        path: Path,
        root: Path,
        *,
        default: str = "",
        max_bytes: int = DEFAULT_MAX_MEMORY_FILE_BYTES,
        tail: bool = False,
    ) -> str:
        fd = self._open_existing_file_no_follow(path, root)
        if fd is None:
            return default
        try:
            st = os.fstat(fd)
            if st.st_size > max_bytes and not tail:
                return default
            if tail and st.st_size > max_bytes:
                os.lseek(fd, -max_bytes, os.SEEK_END)
                data = os.read(fd, max_bytes)
                text = data.decode("utf-8", errors="replace")
                if "\n" in text:
                    text = text.split("\n", 1)[1]
                return text
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = os.read(fd, min(65_536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > max_bytes:
                return default
            return data.decode("utf-8", errors="replace")
        except OSError:
            return default
        finally:
            with suppress(OSError):
                os.close(fd)

    def _assert_safe_write_target(self, path: Path) -> None:
        self._ensure_memory_dir()
        rel = path.relative_to(self.memory_dir)
        if any(part in {"", ".", ".."} for part in rel.parts) or path.parent != self.memory_dir:
            raise OSError(f"unsafe peer memory file path: {path}")
        if path.exists() and not self._safe_existing_file(path, self.memory_dir):
            raise OSError(f"unsafe peer memory file: {path}")

    def _write_memory_text(self, path: Path, text: str) -> None:
        self._assert_safe_write_target(path)
        _atomic_write_text(path, text)

    def _append_memory_text(self, path: Path, text: str) -> None:
        self._assert_safe_write_target(path)
        with open_append_file(path, mode=0o600) as stream:
            stream.write(text.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())

    def _read_memory_text(
        self,
        path: Path,
        default: str = "",
        *,
        max_bytes: int = DEFAULT_MAX_MEMORY_FILE_BYTES,
        tail: bool = False,
    ) -> str:
        try:
            self._ensure_memory_dir()
            path.relative_to(self.memory_dir)
            return self._read_existing_file_no_follow(
                path,
                self.memory_dir,
                default=default,
                max_bytes=max_bytes,
                tail=tail,
            )
        except (OSError, ValueError):
            return default

    def _read_memory_json(self, path: Path, default: Any) -> Any:
        text = self._read_memory_text(path, "")
        if not text:
            return default
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return default

    def _read_memory_yaml(self, path: Path, default: Any) -> Any:
        text = self._read_memory_text(path, "")
        if not text:
            return default
        try:
            loaded = yaml.safe_load(text)
        except (yaml.YAMLError, TypeError, ValueError):
            return default
        return default if loaded is None else loaded

    def _write_memory_yaml(self, path: Path, data: Any) -> None:
        self._write_memory_text(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    def _append_memory_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        self._append_memory_text(path, f"{line}\n")

    def _load_memory_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in self._read_memory_text(
            path,
            "",
            max_bytes=DEFAULT_MAX_LEDGER_FILE_BYTES,
            tail=True,
        ).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
        return rows

    def _read_peer_yaml(self, path: Path, default: Any) -> Any:
        try:
            self._ensure_dir_no_symlink(self.peers_root)
            if not self.peer_root.exists():
                return default
            self._ensure_dir_no_symlink(self.peer_root)
            path.relative_to(self.peer_root)
            text = self._read_existing_file_no_follow(
                path,
                self.peer_root,
                default="",
                max_bytes=DEFAULT_MAX_MEMORY_FILE_BYTES,
            )
            if not text:
                return default
            loaded = yaml.safe_load(text)
        except (OSError, ValueError, yaml.YAMLError, TypeError):
            return default
        return default if loaded is None else loaded

    def compose_session_prompt(
        self,
        task_prompt: str,
        *,
        session_id: str,
        session_index: int,
    ) -> str:
        """Append a bounded memory block to the base task prompt."""

        if not self.config.enabled:
            return task_prompt
        block = self.build_prompt_block(session_id=session_id, session_index=session_index)
        if not block:
            return task_prompt
        composed = f"{task_prompt.rstrip()}\n\n{block}\n"
        safe_session_id = _safe_path_component(session_id, fallback="session")
        manifest = {
            "session_id": session_id,
            "safe_session_id": safe_session_id,
            "session_index": session_index,
            "base_prompt_sha256": hashlib.sha256(task_prompt.encode("utf-8")).hexdigest(),
            "memory_block_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
            "composed_prompt_sha256": hashlib.sha256(composed.encode("utf-8")).hexdigest(),
            "memory_block_chars": len(block),
            "composed_prompt_chars": len(composed),
            "updated_utc": _utc_now(),
        }
        try:
            serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            self._write_memory_text(self.prompt_manifest_path, serialized)
            self._write_memory_text(
                self.memory_dir / f"session_prompt_manifest_{safe_session_id}.json",
                serialized,
            )
            self._prune_session_snapshots("session_prompt_manifest_*.json")
        except OSError:
            pass
        return composed

    def build_prompt_block(self, *, session_id: str, session_index: int) -> str:
        self._ensure_memory_dir()
        self._last_prompt_finding_keys = []
        state = self._load_or_initialize_state()
        contract = self._load_selected_contract()
        recent_experiments = self._recent_experiments()
        new_findings = self._new_shared_findings()
        surfaced_finding_lines: list[tuple[tuple[str, ...], str]] = []

        lines = [
            f"## {MEMORY_HEADER}",
            "",
            "This block is bounded cross-session state for this peer. It is not a raw transcript.",
            "Use it to preserve continuity, avoid repeating failed work, and refresh from new shared findings.",
            "",
            f"- peer_id: `{self.peer_id}`",
            f"- generation_id: `{self.generation_id}`",
            f"- session_id: `{session_id}`",
            f"- session_index: `{session_index}`",
            f"- memory_dir: `{self.memory_dir}`",
            "",
            "### Required Memory Discipline",
            "- Read and update `peer_state.yaml`, `experiment_ledger.jsonl`, and `session_handoff.md` as your work progresses.",
            "- Record every meaningful experiment, smoke/scout/eval, failure, or abandoned branch in the ledger.",
            "- Before ending a session, write a concise handoff with current state, open processes, blockers, and next action.",
            "- Before continuing the same direction, run an anti-anchoring check: what evidence says pivot, simplify, falsify, or continue?",
            "- Prefer structured summaries over copying raw chat or long logs into memory.",
            "",
            "### Current Peer State",
        ]
        lines.extend(self._yaml_bullets(_redact_value(state), max_lines=18))

        if contract:
            lines.extend(["", "### DIG / Selected Contract Snapshot"])
            lines.extend(self._yaml_bullets(_redact_value(contract), max_lines=20))

        if recent_experiments:
            lines.extend(["", "### Recent Experiment Ledger"])
            for item in recent_experiments:
                safe_item = _redact_value(item)
                summary = _shorten(
                    safe_item.get("summary") or safe_item.get("variant_name") or safe_item
                )
                metrics = safe_item.get("metrics")
                metric_text = f" metrics={_shorten(metrics, limit=180)}" if metrics else ""
                lines.append(f"- {summary}{metric_text}")
        else:
            lines.extend(
                [
                    "",
                    "### Recent Experiment Ledger",
                    "- No peer-local experiment ledger entries yet. Create one after the first meaningful action.",
                ]
            )

        if new_findings:
            lines.extend(["", "### New Shared Findings Since Last Session"])
            for item in new_findings:
                safe_item = _redact_value(item)
                title = _shorten(safe_item.get("title") or safe_item.get("finding_id"))
                finding_type = _shorten(safe_item.get("finding_type") or safe_item.get("type"))
                producer = _shorten(safe_item.get("producer_ref") or safe_item.get("producer"))
                finding_line = (
                    f"- `{safe_item['finding_id']}` type={finding_type or 'unknown'} "
                    f"producer={producer or 'unknown'}: {title}"
                )
                if is_scoreless(self.task_spec) and safe_item.get("content"):
                    finding_line += "\n  Evidence: " + _shorten(safe_item["content"], limit=1200)
                lines.append(finding_line)
                raw_seen_keys = item.get("_seen_keys")
                seen_keys = (
                    tuple(str(value) for value in raw_seen_keys if str(value))
                    if isinstance(raw_seen_keys, list)
                    else ()
                )
                surfaced_finding_lines.append((seen_keys, finding_line))
        else:
            lines.extend(["", "### New Shared Findings Since Last Session", "- None detected."])

        handoff = self._load_handoff()
        if handoff:
            lines.extend(["", "### Previous Session Handoff", _shorten(handoff, limit=1400)])

        lines.extend(
            [
                "",
                "### Anti-Anchoring Check",
                "- Name one reason to continue the current mechanism.",
                "- Name one reason to pivot, ablate, or simplify.",
                "- If you choose to continue, state the cheapest evidence that could falsify it.",
            ]
        )

        block = "\n".join(lines).strip()
        if len(block) > self.config.max_prompt_chars:
            block = block[: self.config.max_prompt_chars - 80].rstrip()
            block += "\n\n[Peer-local memory block truncated by configured character budget.]"
        self._write_memory_text(self.prompt_snapshot_path, block + "\n")
        safe_session_id = _safe_path_component(session_id, fallback="session")
        self._write_memory_text(
            self.memory_dir / f"memory_prompt_{safe_session_id}.md", block + "\n"
        )
        self._prune_session_snapshots("memory_prompt_*.md")
        self._last_prompt_finding_keys = [
            seen_key
            for seen_keys, finding_line in surfaced_finding_lines
            if finding_line in block
            for seen_key in seen_keys
        ]
        return block

    def _prune_session_snapshots(self, pattern: str) -> None:
        limit = max(1, int(self.config.max_session_snapshots or 1))
        try:
            self._ensure_memory_dir()
            candidates: list[tuple[float, Path]] = []
            for path in self.memory_dir.glob(pattern):
                try:
                    st = path.lstat()
                except OSError:
                    continue
                if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                    continue
                if path.name in {"memory_prompt.md", "session_prompt_manifest.json"}:
                    continue
                candidates.append((st.st_mtime, path))
            candidates.sort(key=lambda item: (item[0], item[1].name), reverse=True)
            for _, path in candidates[limit:]:
                with suppress(OSError):
                    path.unlink()
        except (OSError, ValueError):
            return

    def record_session_result(
        self,
        *,
        session_id: str,
        result: Any | None,
        log_file: Path | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Persist a minimal handoff and ledger row after each session."""

        if not self.config.enabled:
            return
        self._ensure_memory_dir()
        success = bool(getattr(result, "success", False)) if result is not None else False
        output = getattr(result, "output", {}) if result is not None else {}
        text_outputs = []
        if isinstance(output, dict):
            raw = output.get("text_outputs", [])
            if isinstance(raw, list):
                text_outputs = [_redact_for_memory(item, limit=220) for item in raw[-3:]]
        error_text = _redact_for_memory(error or getattr(result, "error", ""), limit=240)
        summary = "session completed" if success else "session failed or ended with error"
        if text_outputs:
            summary = text_outputs[-1]
        row: dict[str, Any] = {
            "timestamp": _utc_now(),
            "session_id": session_id,
            "summary": summary,
            "success": success,
            "iteration_count": getattr(result, "iteration_count", None)
            if result is not None
            else None,
            "duration": getattr(result, "duration", None) if result is not None else None,
            "error": error_text,
            "log_file": str(log_file) if log_file else "",
            "metrics": {},
        }
        self._append_memory_jsonl(self.ledger_path, row)

        state = self._load_or_initialize_state()
        state["last_session_id"] = session_id
        state["last_updated_utc"] = _utc_now()
        state["session_count_recorded"] = int(state.get("session_count_recorded") or 0) + 1
        state["last_session_success"] = success
        state["last_session_summary"] = summary
        if error_text:
            state["last_error"] = error_text
        elif success and is_scoreless(self.task_spec):
            state.pop("last_error", None)
        latest_artifacts = self._scan_recent_result_artifacts()
        if latest_artifacts:
            state["recent_result_artifacts"] = _redact_value(
                latest_artifacts[: self.config.max_recent_experiments]
            )
        self._write_memory_yaml(self.state_path, _redact_value(state))

        auto_handoff = "\n".join(
            [
                f"# Automatic Session Status: {session_id}",
                "",
                f"- updated_utc: `{_utc_now()}`",
                f"- success: `{success}`",
                f"- summary: {summary}",
                f"- error: {error_text or 'none'}",
                f"- log_file: `{log_file}`" if log_file else "- log_file: unavailable",
                "",
                "## Next Session Checklist",
                "- Re-read this handoff and the experiment ledger before repeating work.",
                "- Check whether new shared findings changed the next best action.",
                "- Run the anti-anchoring check before extending the same branch.",
            ]
        ).strip()
        self._write_memory_text(self.auto_handoff_path, auto_handoff + "\n")

        existing_handoff = self._load_handoff()
        audit_marker = "<!-- PRAXIST_AUTO_SESSION_STATUS -->"
        if existing_handoff.startswith("# Automatic Session Status"):
            manual_handoff = ""
        else:
            manual_handoff = existing_handoff.split(audit_marker, 1)[0].rstrip()
        combined_handoff = manual_handoff
        if combined_handoff:
            combined_handoff = f"{combined_handoff}\n\n{audit_marker}\n\n{auto_handoff}\n"
        else:
            combined_handoff = f"{audit_marker}\n\n{auto_handoff}\n"
        self._write_memory_text(self.handoff_path, combined_handoff)
        if self._session_consumed_prompt(result):
            self._mark_prompt_findings_seen()

    def _session_consumed_prompt(self, result: Any | None) -> bool:
        if result is None:
            return False
        success = bool(getattr(result, "success", False))
        try:
            if int(getattr(result, "iteration_count", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
        output = getattr(result, "output", {})
        if not isinstance(output, dict):
            return False
        tool_uses = output.get("tool_uses")
        if isinstance(tool_uses, list) and bool(tool_uses):
            return True
        if not success:
            return False
        text_outputs = output.get("text_outputs")
        return isinstance(text_outputs, list) and any(str(item).strip() for item in text_outputs)

    def _load_or_initialize_state(self) -> dict[str, Any]:
        state = self._read_memory_yaml(self.state_path, {})
        if not isinstance(state, dict):
            state = {}
        if not state:
            state = {
                "peer_id": self.peer_id,
                "generation_id": self.generation_id,
                "created_utc": _utc_now(),
                "research_state": "initializing",
                "current_hypothesis": "",
                "open_questions": [],
                "known_dead_ends": [],
                "active_variant": "",
                "session_count_recorded": 0,
            }
            self._write_memory_yaml(self.state_path, state)
        return state

    def _load_selected_contract(self) -> dict[str, Any]:
        path = self.peer_root / "dig" / "selected_contract.yaml"
        contract = self._read_peer_yaml(path, {})
        if not isinstance(contract, dict):
            return {}
        keep_keys = (
            "variant_name",
            "diversity_cell",
            "semantic_family",
            "parent_lineage",
            "mechanism_hypothesis",
            "why_selected",
            "expected_metric_signature",
            "ablation_hooks",
            "fail_fast_checks",
        )
        return {key: contract[key] for key in keep_keys if key in contract}

    def _recent_experiments(self) -> list[dict[str, Any]]:
        rows = self._load_memory_jsonl(self.ledger_path)
        artifact_rows = [
            {
                "summary": artifact.get("variant_name"),
                "metrics": artifact.get("metrics"),
            }
            for artifact in self._scan_recent_result_artifacts()
        ]
        combined = rows + artifact_rows
        return combined[-self.config.max_recent_experiments :]

    def _scan_recent_result_artifacts(self) -> list[dict[str, Any]]:
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
                result_summary_variant_name,
            )
        except Exception:
            return []
        artifacts: list[tuple[float, dict[str, Any]]] = []
        results_root = self.run_dir / "results"
        for path in _safe_result_summary_paths(self.run_dir):
            payload = _read_json_under_root(path, results_root, None)
            if not isinstance(payload, dict):
                continue
            try:
                variant_name = result_summary_variant_name(path, payload, self.run_dir)
            except Exception:
                variant_name = _extract_variant_name(path, payload)
            if not _matches_peer(variant_name, self.peer_id, self.generation_id):
                continue
            metrics: dict[str, Any] = {}
            metrics.update(_metric_subset(payload))
            for nested_key in ("current_aggregate", "aggregate", "summary", "metrics"):
                if nested_key in payload:
                    metrics.update(_metric_subset(payload.get(nested_key)))
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            artifacts.append(
                (
                    mtime,
                    {
                        "variant_name": variant_name,
                        "path": str(path),
                        "metrics": metrics,
                    },
                )
            )
        artifacts.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in artifacts[: self.config.max_recent_experiments]]

    def _new_shared_findings(self) -> list[dict[str, Any]]:
        seen = self._load_seen_finding_keys(migrate=True)
        try:
            findings_stat = self.findings_dir.lstat()
        except OSError:
            return []
        if stat.S_ISLNK(findings_stat.st_mode) or not stat.S_ISDIR(findings_stat.st_mode):
            return []
        try:
            findings_root = self.findings_dir.resolve(strict=True)
        except OSError:
            return []
        findings: list[tuple[float, dict[str, Any]]] = []
        for path in self.findings_dir.glob("*.json"):
            if path.is_symlink():
                continue
            try:
                path.resolve(strict=True).relative_to(findings_root)
            except (OSError, ValueError):
                continue
            payload = _read_json_under_root(path, self.findings_dir, None)
            if not isinstance(payload, dict):
                continue
            explicit_finding_id = _explicit_finding_id(payload)
            finding_id = explicit_finding_id or path.stem
            legacy_seen_key = _finding_seen_key(finding_id)
            if self.config.track_finding_content_versions:
                seen_key = (
                    _finding_version_seen_key(explicit_finding_id, payload)
                    if explicit_finding_id is not None
                    else ""
                )
                seen_keys_to_record = (
                    [seen_key, legacy_seen_key] if explicit_finding_id is not None else []
                )
            else:
                seen_key = legacy_seen_key
                seen_keys_to_record = [legacy_seen_key]
            if seen_key and seen_key in seen:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            findings.append(
                (
                    mtime,
                    {
                        "finding_id": finding_id,
                        "title": payload.get("title") or payload.get("summary"),
                        "finding_type": payload.get("finding_type") or payload.get("type"),
                        "producer_ref": payload.get("producer_ref") or payload.get("producer"),
                        **(
                            {"content": _redact_for_memory(payload.get("content"), limit=1200)}
                            if is_scoreless(self.task_spec)
                            else {}
                        ),
                        "_seen_keys": seen_keys_to_record,
                    },
                )
            )
        findings.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in findings[: self.config.max_shared_findings]]

    def should_wake_for_shared_finding(self, path: Path) -> bool:
        """Return false only when a safely read finding was already consumed."""

        payload_bytes = read_bounded_file_under_root_no_follow(
            Path(path),
            self.findings_dir,
            max_bytes=DEFAULT_MAX_EXTERNAL_JSON_BYTES,
        )
        if payload_bytes is None:
            return True
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return True
        if not isinstance(payload, dict):
            return True
        finding_id = _explicit_finding_id(payload)
        if finding_id is None:
            return True
        try:
            seen_key = (
                _finding_version_seen_key(finding_id, payload)
                if self.config.track_finding_content_versions
                else _finding_seen_key(finding_id)
            )
            return seen_key not in self._load_seen_finding_keys()
        except (OSError, TypeError, ValueError):
            return True

    def _mark_prompt_findings_seen(self) -> None:
        if not self._last_prompt_finding_keys:
            return
        seen = self._load_seen_finding_keys(migrate=True)
        seen.update(self._last_prompt_finding_keys)
        self._write_memory_text(
            self.seen_findings_path,
            json.dumps(sorted(seen), indent=2, ensure_ascii=False) + "\n",
        )

    def _load_seen_finding_keys(self, *, migrate: bool = False) -> set[str]:
        raw_items = self._read_memory_json(self.seen_findings_path, [])
        if not isinstance(raw_items, list):
            raw_items = []
        seen: set[str] = set()
        changed = False
        for item in raw_items:
            text = str(item)
            if re.fullmatch(r"sha256(?:v2)?:[0-9a-f]{64}", text):
                seen.add(text)
            else:
                seen.add(_finding_seen_key(text))
                changed = True
        if migrate and changed:
            self._write_memory_text(
                self.seen_findings_path,
                json.dumps(sorted(seen), indent=2, ensure_ascii=False) + "\n",
            )
        return seen

    def _load_handoff(self) -> str:
        try:
            redacted, _ = redact_text(
                self._read_memory_text(
                    self.handoff_path,
                    "",
                    max_bytes=DEFAULT_MAX_HANDOFF_BYTES,
                    tail=True,
                )
            )
            return redacted.strip()
        except OSError:
            return ""
        return ""

    def _yaml_bullets(self, data: dict[str, Any], *, max_lines: int) -> list[str]:
        if not data:
            return ["- unavailable"]
        text = yaml.safe_dump(_redact_value(data), sort_keys=False, allow_unicode=True).splitlines()
        bullets = [f"    {line}" for line in text[:max_lines]]
        if len(text) > max_lines:
            bullets.append("    ...")
        return ["```yaml", *bullets, "```"]


class NoOpPeerSessionMemory:
    """Fallback used when memory initialization fails before a session starts."""

    def compose_session_prompt(
        self,
        task_prompt: str,
        *,
        session_id: str,
        session_index: int,
    ) -> str:
        _ = (session_id, session_index)
        return task_prompt

    def should_wake_for_shared_finding(self, path: Path) -> bool:
        _ = path
        return True

    def record_session_result(
        self,
        *,
        session_id: str,
        result: Any | None,
        log_file: Path | None = None,
        error: BaseException | None = None,
    ) -> None:
        _ = (session_id, result, log_file, error)
