"""
Baseline cache with provenance metadata + boot-time validation.

Problem this solves:
    A peer caught a cached baseline metric that disagreed with a fresh
    measurement enough to invalidate leaderboard comparisons until it was
    remeasured.

Design:
  - task_spec.yaml stays read-only. Packaged baseline values there are the
    historical reference for the task's primary metric.
  - Fresh measurements by peers are recorded in a run-scoped cache when
    ``PRAXIST_BASELINE_CACHE_DIR`` is set:
        <run_dir>/baseline_cache/<task_id>/baselines.jsonl
  - Each entry carries provenance: measured_at, code_hash, hardware,
    dataset_hash, baseline name, metric_name/metric_value, plus optional
    mean/std/seeds. ``accuracy`` remains as a backwards-compatible alias for
    older task packs and caches.
  - At run boot, `validate_cache(task_spec, current_code_hash)` marks entries
    stale when:
      * `code_hash` differs AND the diff touches baseline-relevant files, OR
      * entry is older than STALE_AFTER_DAYS (default 30).
  - Stale entries are NOT deleted — they are marked with `is_stale: True` and
    `stale_reason`, and the validator returns a summary the orchestrator can
    log and surface to peers via the prompt context.

Peer-facing guidance lives in the prompt (see prompt_base.jinja2 "Known
pitfalls"): if a baseline number looks surprising, remeasure before comparing.
"""

from __future__ import annotations

import contextlib

try:
    import fcntl
except ImportError:
    fcntl = None

import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Generator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .tools.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _flock(path: Path) -> Generator[None, None, None]:
    """Hold an exclusive fcntl.flock on `path` for the duration of the block.

    Used to serialize read-modify-write cycles on the cache JSONL file so
    two peers' record_measurement calls don't clobber each other. Creates
    `<path>.lock` as the lock target so we don't depend on the cache file
    existing yet.

    O_CLOEXEC ensures child processes spawned while we hold the flock don't
    inherit the lock — avoids the "child inherits fd, parent closes, lock
    still held" hazard that would deadlock future callers.

    NOT REENTRANT. Calling `record_measurement` or `validate_cache` while
    already inside another `_flock(...)` block on the same cache path will
    deadlock: fcntl.flock is per-open-file-description, and a second open
    of the same lock file gets a fresh OFD whose LOCK_EX will block forever
    on the outer holder. No current call path creates reentrance. Don't
    add one — if you need nested access, pass entries through the call
    chain instead of re-acquiring the lock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o644)
    try:
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# Entries older than this default are considered stale unless re-validated.
STALE_AFTER_DAYS = 30

# Files whose changes we treat as possibly invalidating baselines.
_BASELINE_RELEVANT_SUBSTRINGS = (
    "baseline/",
    "entrypoint",
    "train.py",  # compatibility with older ML task packs
    "dataset",
    "data/",
    "eval/",
)

_CURATED_BASELINE_NAME_KEYS = (
    "name",
    "baseline",
    "baseline_name",
    "optimizer",
    "method",
    "variant_name",
)

_CURATED_BASELINE_METRIC_PRIORITY = (
    "metric_value",
    "deterministic_score",
    "score",
    "value",
    "test_accuracy",
    "accuracy",
)

_CURATED_BASELINE_NUMERIC_EXCLUDE = {
    "seed",
    "seeds",
    "n_seeds",
    "epochs",
    "wall_time_s",
    "wall_time_seconds",
    "runtime_seconds",
}


@dataclass
class CachedBaseline:
    """One cached baseline measurement with provenance.

    `name`, `metric_name`, and `metric_value` are the load-bearing fields; the
    remaining fields are provenance and optional aggregate statistics.
    `accuracy` is retained as an alias for existing caches. `is_stale` is set
    by the validator, not the writer.
    """

    name: str
    accuracy: float | None = None
    metric_name: str = "metric_value"
    metric_value: float | None = None
    direction: str = ""
    measured_at: str = ""
    code_hash: str = ""
    hardware: str = ""
    dataset_hash: str = ""
    mean: float | None = None
    std: float | None = None
    seeds: list[int] = field(default_factory=list)
    epochs: int | None = None
    notes: str = ""
    is_stale: bool = False
    stale_reason: str = ""

    def __post_init__(self) -> None:
        if self.metric_value is None:
            self.metric_value = self.accuracy
        if self.accuracy is None:
            self.accuracy = self.metric_value
        if self.metric_value is not None:
            try:
                self.metric_value = float(self.metric_value)
            except (TypeError, ValueError):
                self.metric_value = None
        if self.accuracy is not None:
            try:
                self.accuracy = float(self.accuracy)
            except (TypeError, ValueError):
                self.accuracy = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CachedBaseline:
        # Be tolerant of unknown fields so old cache files don't break on upgrade.
        if not isinstance(d, dict):
            raise TypeError(f"baseline cache row must be a JSON object, got {type(d).__name__}")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        # Dataclass fields with default_factory are NOT applied when the key
        # is present with value None. Coerce mutable-container defaults so
        # a legacy cache with `"seeds": null` doesn't TypeError on iteration.
        if filtered.get("seeds") is None:
            filtered["seeds"] = []
        return cls(**filtered)


def get_current_code_hash(repo_root: Path | None = None) -> str:
    """Return the current git HEAD short hash, or empty string if unavailable."""
    cwd = str(repo_root) if repo_root else None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ""


def get_changed_files_since(
    ref_hash: str,
    repo_root: Path | None = None,
) -> list[str]:
    """Return list of files changed since `ref_hash` relative to HEAD.

    Returns empty list if ref is unknown or git is unavailable.
    """
    if not ref_hash:
        return []
    cwd = str(repo_root) if repo_root else None
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", f"{ref_hash}...HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return [line for line in out.decode().splitlines() if line]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []


def _default_cache_path(workspace: Path, task_id: str) -> Path:
    override = os.environ.get("PRAXIST_BASELINE_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve() / task_id / "baselines.jsonl"
    return Path(tempfile.gettempdir()) / "praxist" / "baseline_cache" / task_id / "baselines.jsonl"


def load_cache(
    task_id: str,
    workspace: Path,
) -> list[CachedBaseline]:
    """Load all cached baseline entries for a task. Missing file → [].

    Each JSONL line is one entry; malformed lines are skipped with a warning.
    """
    path = _default_cache_path(workspace, task_id)
    if not path.exists():
        return []
    entries: list[CachedBaseline] = []
    try:
        with open(path) as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entries.append(CachedBaseline.from_dict(json.loads(raw)))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        "baseline_cache: skipping malformed line %d of %s: %s",
                        lineno,
                        path,
                        e,
                    )
    except OSError as e:
        logger.warning("baseline_cache: could not read %s: %s", path, e)
    return entries


def save_cache(
    task_id: str,
    workspace: Path,
    entries: list[CachedBaseline],
) -> Path:
    """Atomically write cached baselines as JSONL.

    Uses tmp+rename via atomic_io.atomic_write_json wrapper for safety.
    Entries are written newest-first by measured_at for human readability.
    """
    path = _default_cache_path(workspace, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _key(e: CachedBaseline):
        # Primary: measured_at (newer first). Secondary: name for stable
        # ordering on tie. Reverse applies to the tuple elementwise, so we
        # negate name-ordering by using a sentinel inverse sort via sorted()
        # twice: first stable-sort by name ascending, then descending by
        # measured_at. Python's sort is stable, so the name order is kept
        # within equal timestamps.
        return e.measured_at or ""

    sorted_entries = sorted(entries, key=lambda e: e.name)
    sorted_entries = sorted(sorted_entries, key=_key, reverse=True)

    # atomic_write_json writes JSON; we want JSONL, so do it manually with
    # the same tmp+rename pattern.
    import tempfile

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w") as f:
            for e in sorted_entries:
                f.write(json.dumps(asdict(e), default=str))
                f.write("\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                pass
        # Chmod the tmp BEFORE replace so dest appears atomically with 0o644.
        # Chmod-after-replace has a microsecond window at 0o600 where a
        # concurrent reader gets PermissionError. Matches atomic_write_json.
        with contextlib.suppress(OSError):
            os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
        return path
    except BaseException:
        # BaseException (not just Exception) so KeyboardInterrupt /
        # CancelledError / SystemExit also trigger tmp cleanup.
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise


def record_measurement(
    task_id: str,
    workspace: Path,
    name: str,
    accuracy: float,
    *,
    metric_name: str = "metric_value",
    metric_value: float | None = None,
    direction: str = "",
    code_hash: str = "",
    hardware: str = "",
    dataset_hash: str = "",
    mean: float | None = None,
    std: float | None = None,
    seeds: list[int] | None = None,
    epochs: int | None = None,
    notes: str = "",
) -> CachedBaseline:
    """Append a fresh measurement to the cache.

    This is the entrypoint that baseline-measurement scripts should call.
    ``accuracy`` is the legacy positional metric value; new task packs should
    also pass ``metric_name`` and may pass ``metric_value`` explicitly.
    Replaces the nearest-matching prior entry (same `name`, same `code_hash`,
    same `epochs`) so repeated re-measurements in-place don't bloat the file.

    The whole load-modify-save cycle is serialized under an fcntl.flock so
    two peers running concurrently don't lose each other's writes.
    """
    cache_path = _default_cache_path(workspace, task_id)
    with _flock(cache_path):
        entries = load_cache(task_id, workspace)
        new = CachedBaseline(
            name=name,
            accuracy=accuracy,
            metric_name=metric_name or "metric_value",
            metric_value=accuracy if metric_value is None else metric_value,
            direction=direction,
            measured_at=datetime.now(UTC).isoformat(),
            code_hash=code_hash or get_current_code_hash(),
            hardware=hardware,
            dataset_hash=dataset_hash,
            mean=mean,
            std=std,
            seeds=list(seeds or []),
            epochs=epochs,
            notes=notes,
        )

        # Replace in-place if same (name, code_hash, epochs) exists.
        def _same_slot(e: CachedBaseline) -> bool:
            return e.name == new.name and e.code_hash == new.code_hash and e.epochs == new.epochs

        kept = [e for e in entries if not _same_slot(e)]
        kept.append(new)
        save_cache(task_id, workspace, kept)
        return new


@dataclass
class ValidationReport:
    """Summary returned by validate_cache. Safe to log and to JSON-serialize."""

    task_id: str
    total: int
    fresh: int
    stale: int
    missing_baselines: list[str] = field(default_factory=list)
    missing_runtime_cache_baselines: list[str] = field(default_factory=list)
    curated_baseline_names: list[str] = field(default_factory=list)
    curated_entries: list[dict[str, Any]] = field(default_factory=list)
    stale_entries: list[dict[str, Any]] = field(default_factory=list)
    fresh_entries: list[dict[str, Any]] = field(default_factory=list)


def _curated_baseline_name(row: dict[str, Any]) -> str:
    for key in _CURATED_BASELINE_NAME_KEYS:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return ""


def _curated_numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, dict):
        value = value.get("mean", value.get("value"))
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _curated_baseline_metric(row: dict[str, Any]) -> tuple[str, float | None]:
    for key in _CURATED_BASELINE_METRIC_PRIORITY:
        value = _curated_numeric_value(row.get(key))
        if value is not None:
            return key, value
    for key, raw in row.items():
        if key in _CURATED_BASELINE_NAME_KEYS or key in _CURATED_BASELINE_NUMERIC_EXCLUDE:
            continue
        value = _curated_numeric_value(raw)
        if value is not None:
            return str(key), value
    return "metric_value", None


def load_curated_baseline_entries(path: Path | None) -> list[dict[str, Any]]:
    """Load task-packaged curated baseline rows from a JSONL asset.

    The runtime cache tracks freshness provenance for re-measured baselines.
    Task projects may also ship curated baseline evidence under assets, which
    remains valid baseline data even when no runtime cache entry exists yet.
    This parser is intentionally tolerant: non-row metadata records such as
    ``{"_protocol": ...}`` are ignored, and unknown row fields are preserved.
    """
    if path is None or not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "baseline_cache: skipping malformed curated baseline line %d of %s: %s",
                        lineno,
                        path,
                        e,
                    )
                    continue
                if not isinstance(row, dict):
                    continue
                name = _curated_baseline_name(row)
                if not name:
                    continue
                row = dict(row)
                row.setdefault("name", str(name))
                metric_name, metric_value = _curated_baseline_metric(row)
                row.setdefault("metric_name", metric_name)
                if "metric_value" not in row and metric_value is not None:
                    row["metric_value"] = metric_value
                row.setdefault("source", str(path))
                entries.append(row)
    except OSError as e:
        logger.warning("baseline_cache: could not read curated baselines %s: %s", path, e)
    return entries


def validate_cache(
    task_id: str,
    workspace: Path,
    expected_baseline_names: list[str],
    *,
    current_code_hash: str | None = None,
    stale_after_days: int = STALE_AFTER_DAYS,
    repo_root: Path | None = None,
    curated_entries: list[dict[str, Any]] | None = None,
) -> ValidationReport:
    """Validate cache entries, marking stale ones. Persists `is_stale` back.

    Parameters
    ----------
    task_id : str
        Used to locate the cache file.
    workspace : Path
        Compatibility parameter for direct callers. Real runs set
        ``PRAXIST_BASELINE_CACHE_DIR`` so cache files are written under run_dir.
    expected_baseline_names : list[str]
        The baselines the task_spec expects. Names not present in the cache
        are reported in `missing_baselines`.
    current_code_hash : str, optional
        If provided, entries whose `code_hash` differs AND whose diff
        touches baseline-relevant files are marked stale. If omitted, the
        current HEAD hash is computed via git.
    stale_after_days : int
        Entries older than this are marked stale regardless of code_hash.
    repo_root : Path, optional
        Override the git repo root for code-hash comparisons.
    curated_entries : list[dict], optional
        Task-packaged baseline evidence. These rows do not count as fresh
        runtime cache entries, but they do satisfy baseline availability so
        `missing_baselines` only means "not available anywhere".

    Returns
    -------
    ValidationReport

    Side effects
    ------------
    If any entry's `is_stale` flag changed, the cache file is rewritten.
    """
    # Resolve the current code hash OUTSIDE the flock — git invocation can
    # take up to 5 seconds and we must not block concurrent record_measurement
    # calls for that long. Same for the per-code-hash git diff: pre-compute
    # the set of changed files for each distinct historical code_hash we
    # might encounter.
    if current_code_hash is None:
        current_code_hash = get_current_code_hash(repo_root)

    # Pre-read entries without the lock to discover which historical hashes
    # we need changed-files for. A concurrent writer could change this set
    # between this read and the lock acquisition; that's OK — any newly
    # added entry will have the current_code_hash (no diff needed) and
    # will be a fresh entry by definition.
    pre_entries = load_cache(task_id, workspace)
    changed_files_by_hash: dict[str, list[str]] = {}
    if current_code_hash:
        for e in pre_entries:
            if (
                e.code_hash
                and e.code_hash != current_code_hash
                and e.code_hash not in changed_files_by_hash
            ):
                changed_files_by_hash[e.code_hash] = get_changed_files_since(e.code_hash, repo_root)

    cache_path = _default_cache_path(workspace, task_id)
    with _flock(cache_path):
        entries = load_cache(task_id, workspace)

        cutoff = datetime.now(UTC) - timedelta(days=stale_after_days)
        changed_any = False

        fresh_entries: list[dict[str, Any]] = []
        stale_entries: list[dict[str, Any]] = []

        for e in entries:
            prior_stale = e.is_stale
            prior_reason = e.stale_reason
            e.is_stale = False
            e.stale_reason = ""

            # Check age
            age_stale = False
            if e.measured_at:
                try:
                    ts = datetime.fromisoformat(e.measured_at)
                    # Normalize to UTC so comparison against a UTC cutoff is
                    # well-defined for any timezone we ingest.
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    else:
                        ts = ts.astimezone(UTC)
                    if ts < cutoff:
                        age_stale = True
                except ValueError:
                    # Unparseable timestamp → treat as stale conservatively.
                    age_stale = True

            if age_stale:
                e.is_stale = True
                e.stale_reason = f"older than {stale_after_days} days"

            # Check code hash relevance using pre-computed changed_files.
            # If an entry was added between the pre-read and lock acquisition
            # and its code_hash isn't in our map, fall back to a synchronous
            # git call (rare, short path).
            if (
                not e.is_stale
                and current_code_hash
                and e.code_hash
                and e.code_hash != current_code_hash
            ):
                if e.code_hash in changed_files_by_hash:
                    changed_files = changed_files_by_hash[e.code_hash]
                else:
                    # Entry appeared between pre-read and lock; compute now.
                    changed_files = get_changed_files_since(e.code_hash, repo_root)
                    changed_files_by_hash[e.code_hash] = changed_files
                if any(
                    any(sub in f for sub in _BASELINE_RELEVANT_SUBSTRINGS) for f in changed_files
                ):
                    e.is_stale = True
                    e.stale_reason = (
                        f"code_hash {e.code_hash} → {current_code_hash} "
                        f"touches baseline-relevant files"
                    )

            if e.is_stale or prior_stale:
                stale_entries.append(asdict(e))
            else:
                fresh_entries.append(asdict(e))

            if e.is_stale != prior_stale or e.stale_reason != prior_reason:
                changed_any = True

        if changed_any:
            save_cache(task_id, workspace, entries)

    have_runtime_names = {e.name for e in entries if not e.is_stale}
    curated_entries = list(curated_entries or [])
    curated_names = sorted(
        {_curated_baseline_name(row) for row in curated_entries if _curated_baseline_name(row)}
    )
    have_available_names = have_runtime_names | set(curated_names)
    missing_runtime = [n for n in expected_baseline_names if n not in have_runtime_names]
    missing = [n for n in expected_baseline_names if n not in have_available_names]

    return ValidationReport(
        task_id=task_id,
        total=len(entries),
        fresh=sum(1 for e in entries if not e.is_stale),
        stale=sum(1 for e in entries if e.is_stale),
        missing_baselines=missing,
        missing_runtime_cache_baselines=missing_runtime,
        curated_baseline_names=curated_names,
        curated_entries=curated_entries,
        stale_entries=stale_entries,
        fresh_entries=fresh_entries,
    )


def write_report_for_peers(
    report: ValidationReport,
    run_dir: Path,
) -> Path:
    """Write a per-run summary peers can read via the `findings_dir` context.

    The file lives at `<run_dir>/baseline_cache_status.json` and is consumed
    by prompt_base.jinja2 to surface staleness warnings.
    """
    path = run_dir / "baseline_cache_status.json"
    payload = {
        "task_id": report.task_id,
        "total": report.total,
        "fresh": report.fresh,
        "stale": report.stale,
        "missing_baselines": report.missing_baselines,
        "missing_runtime_cache_baselines": report.missing_runtime_cache_baselines,
        "curated_baseline_names": report.curated_baseline_names,
        "curated_entries": report.curated_entries,
        "fresh_entries": report.fresh_entries,
        "stale_entries": report.stale_entries,
    }
    atomic_write_json(path, payload)
    return path
