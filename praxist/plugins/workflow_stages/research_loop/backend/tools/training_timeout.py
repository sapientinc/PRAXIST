"""
Long-running subprocess timeout utilities for tiered_eval-style pipelines.

Designed to be imported by per-task ``tiered_eval.py`` modules so that
all tasks share the same timeout / partial-summary semantics.

Three improvements over a naive "kill at fixed wallclock":

1. **Hard-cap raised** (default 120 min, was 90 min) — gives an extra
   33% wallclock to all variants. Most timeouts in a prior run
   killed variants very near completion; the extra 30 min would have rescued
   them outright.

2. **Progress-aware completion grace**: when the hard cap fires, if the
   variant is in the last 10% of its training schedule
   (``current_epoch / total_epochs >= 0.9``), the timeout is **removed
   entirely** for that subprocess — the training is allowed to run to
   natural completion. This protects the most expensive case (where the
   variant is almost done, with sunk-cost ~5-10 min already invested).

3. **Partial-summary on partial cell failure**: at the tier-evaluation
   layer, a tier is no longer aborted just because 1 cell timed out.
   If failure rate < 30%, the tier emits a summary computed on the
   surviving cells, with ``partial=True`` flag. Frontier promotion
   applies a 0.95× discount to partial-summary variants so they are
   ranked below clean variants of equal raw FF.

Usage in a per-task ``tiered_eval.py``::

    from praxist.plugins.workflow_stages.research_loop.backend.tools.training_timeout import (
        TimeoutPolicy, monitor_subprocess_with_grace,
        should_emit_partial_summary,
    )

    policy = TimeoutPolicy()  # defaults: 120 min cap, 0.9 grace threshold
    rc = monitor_subprocess_with_grace(
        proc, log_path=cell_dir / "run.log",
        total_epochs=500, policy=policy,
    )

    # Then at tier-eval time:
    abort, reason = should_emit_partial_summary(failed_cells=1, total=10)
    if not abort:
        # emit summary with partial=True
        ...
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from praxist.core.execution_guards import emit_resource_event_from_env

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Policy dataclass — task-tunable timeout knobs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeoutPolicy:
    """Timeout configuration for a single training subprocess.

    Attributes:
        hard_cap_seconds: Absolute hardest wall-clock the subprocess is
            allowed BEFORE checking grace. Default 120 min (was 90 min
            pre-2026-04-30).
        grace_progress_threshold: Fraction of total_epochs above which the
            grace mechanism activates. Default 0.9 — i.e. if at the
            hard-cap, the training has already completed ≥ 90% of its
            schedule, the timeout is REMOVED. The training is then
            allowed to finish naturally, BOUNDED by
            ``grace_max_extension_seconds`` and stall-detection.
        grace_max_extension_seconds: Upper bound on additional wall-clock
            after grace activates. Even under grace, no single subprocess
            holds its slot beyond ``hard_cap + grace_max_extension``
            seconds. Default 30 min — enough to finish a near-complete
            task at typical observed speeds, without indefinitely starving
            siblings on the slot pool.
        grace_stall_max_polls: Number of consecutive grace-period polls
            with NO epoch advance before forcing a kill. Detects deadlocks
            / NaN loops / log-stuck conditions during the grace window.
            Default 5 (= 5 * 30s = 2.5 min of stall).
        grace_check_interval_seconds: How often to poll progress. Default
            30s; also the resolution of the eventual timeout.
        kill_signal: Signal sent on timeout-with-no-grace. Default SIGTERM.
        kill_grace_seconds: How long after SIGTERM to wait before SIGKILL.

    The defaults assume a moderately long task with parseable progress logs.
    Tasks with very different schedules should override.
    """

    hard_cap_seconds: int = 120 * 60
    grace_progress_threshold: float = 0.9
    grace_max_extension_seconds: int = 30 * 60
    grace_stall_max_polls: int = 5
    grace_check_interval_seconds: int = 30
    kill_signal: int = signal.SIGTERM
    kill_grace_seconds: int = 10
    # M3 from review round 2: log tail read for progress parsing must
    # cover several recent progress lines. Tasks with very verbose per-step
    # logs should raise this further.
    log_tail_read_bytes: int = 64 * 1024


# -----------------------------------------------------------------------------
# Subprocess monitoring with progress-aware grace
# -----------------------------------------------------------------------------

# Default regex matches common epoch-style progress logs ("Epoch 123/500").
# Tasks with different log formats must pass a custom ``epoch_pattern``.
_DEFAULT_EPOCH_PATTERN = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")


def parse_current_epoch(
    log_path: Path,
    epoch_pattern: re.Pattern = _DEFAULT_EPOCH_PATTERN,
    expected_total: int | None = None,
    tail_bytes: int | None = None,
) -> tuple[int, int] | None:
    """Parse the latest "Epoch X/Y" line from a training log.

    Returns:
        (current_epoch, total_epochs) or None if no parseable line found.

    The log is read from the END, since training logs are append-only.
    Tail size is configurable via ``tail_bytes`` (default 64 KB; M3 from
    review round 2 — 8 KB was too small for verbose progress logs).

    Returns the **highest** valid epoch found in the tail (M2 from
    review round 2 — must be monotonic; previously took "last in
    tail" which could regress on interleaved log lines).

    Failure modes (file missing, encoding issues, log not yet flushed)
    return None — caller should treat None as "progress unknown" and
    apply the hard cap conservatively.

    If ``expected_total`` is provided, only matches whose total equals
    ``expected_total`` are considered. This prevents an eval-loop's
    "Epoch 1/10" or LR-warmup's "Epoch 1/20" from being mistaken for
    training progress.
    """
    # Round 3 M7 fix: when tail_bytes is None, derive from
    # TimeoutPolicy default to keep the standalone function in sync
    # with policy changes. Caller-supplied non-None value still wins.
    if tail_bytes is None:
        tail_bytes = TimeoutPolicy.__dataclass_fields__["log_tail_read_bytes"].default
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            seek_back = min(size, max(1024, int(tail_bytes)))
            f.seek(size - seek_back)
            tail = f.read().decode("utf-8", errors="replace")
    except (OSError, FileNotFoundError):
        return None

    matches = epoch_pattern.findall(tail)
    if not matches:
        return None

    # Two-pass match selection (round 4 C1 fix):
    # Pass 1: prefer matches whose `tot` equals expected_total (strictest).
    # Pass 2: if pass 1 yields nothing AND expected_total was given,
    #         fall back to highest-`tot` match (still bounded by
    #         monotonic max-cur logic). This rescues tasks whose log
    #         format slightly disagrees with the schedule (e.g. epoch
    #         499 vs 500 due to off-by-one rounding, or a custom log
    #         format that uses step-counts instead of epochs).
    # Always take MAX cur within the chosen tot.
    parsed: list[tuple[int, int]] = []
    for m in matches:
        try:
            cur, tot = int(m[0]), int(m[1])
        except (ValueError, IndexError):
            continue
        if tot <= 0:
            continue
        parsed.append((cur, tot))

    if not parsed:
        return None

    if expected_total is not None:
        strict = [(c, t) for c, t in parsed if t == expected_total]
        if strict:
            best_cur, best_total = max(strict, key=lambda x: x[0])
            return (best_cur, best_total)
        # Fallback: pick the highest-tot match (likely the actual
        # training schedule; eval/warmup sub-loops have lower tot).
        best_total = max(t for _, t in parsed)
        candidates = [(c, t) for c, t in parsed if t == best_total]
        best_cur, _ = max(candidates, key=lambda x: x[0])
        return (best_cur, best_total)

    # No filter: max-cur over all parsed.
    best_cur, best_total = max(parsed, key=lambda x: x[0])
    return (best_cur, best_total)


def monitor_subprocess_with_grace(
    proc: subprocess.Popen,
    log_path: Path,
    total_epochs: int,
    policy: TimeoutPolicy | None = None,
    epoch_pattern: re.Pattern = _DEFAULT_EPOCH_PATTERN,
) -> int:
    """Wait for ``proc`` with progress-aware timeout + bounded grace.

    Behavior (per user spec 2026-04-30, with safety bounds added in
    review round 1 fixes):

    1. Poll ``proc`` every ``policy.grace_check_interval_seconds``.
    2. At ``elapsed >= policy.hard_cap_seconds``:
       - Parse ``current_epoch`` from log (filtered by ``total_epochs``
         to ignore eval/warmup sub-loops).
       - If ``current_epoch / total_epochs >= grace_progress_threshold``
         (default 0.9), GRANT GRACE: subprocess gets up to
         ``grace_max_extension_seconds`` more wall-clock to finish.
       - Otherwise: kill process group (SIGTERM → SIGKILL).
    3. During grace:
       - Poll continues. If no epoch advance for
         ``grace_stall_max_polls`` consecutive polls, force kill
         (deadlock / NaN-loop detection).
       - At ``elapsed >= hard_cap + grace_max_extension``, force kill
         (absolute upper bound).

    Args:
        proc: Subprocess started with ``start_new_session=True`` so we
            can kill its process group cleanly.
        log_path: Path to the training log (used for progress parsing).
        total_epochs: Total epochs the training is expected to run.
            MUST be > 0 (rejected as ValueError otherwise — this avoids
            silent grace-denial when the caller forgot to populate it).
        policy: TimeoutPolicy. Uses defaults if None.
        epoch_pattern: Regex matching "Epoch X/Y". Override for tasks
            with different progress-log formats.

    Returns:
        Subprocess exit code. Negative codes:
        - -1: SIGTERM sent (timeout, grace not granted or denied)
        - -2: SIGKILL sent (escalated after SIGTERM grace expired)
        - -3: Force-killed during grace (stall or absolute cap)
        Positive or zero: natural exit (including grace-extended
        natural exit).

    Raises:
        ValueError: if total_epochs <= 0.
    """
    if policy is None:
        policy = TimeoutPolicy()

    # M1 fix (review round 2): require a meaningful schedule. Tasks
    # with total_epochs < 10 give degenerate grace semantics (e.g.
    # current=1/total=1 → fraction=1.0, immediate grace). Floor at 10
    # to keep "10% remaining" interpretation rigorous.
    MIN_TOTAL_EPOCHS = 10
    if not isinstance(total_epochs, int) or total_epochs < MIN_TOTAL_EPOCHS:
        raise ValueError(
            f"total_epochs must be int >= {MIN_TOTAL_EPOCHS} (got "
            f"{total_epochs!r}). Below this floor the 90%-grace "
            f"threshold is degenerate. For ultra-short schedules use "
            f"a custom TimeoutPolicy with grace disabled."
        )

    start = time.time()
    grace_active = False  # set True once hard_cap crossed with grace granted
    grace_started_at: float | None = None
    last_observed_epoch: int | None = None
    stall_polls = 0  # consecutive polls with no epoch advance during grace

    def _kill_group(reason_label: str, escalate_code: int) -> int:
        """Kill the process group, return appropriate negative exit code.
        Re-checks proc.poll() before signaling to handle TOCTOU (the
        subprocess may have just exited, and PID may have been reused).
        """
        emit_resource_event_from_env(
            "resource.training_timeout_abort",
            action_type="training_timeout",
            actor_ref="resource_guard:training_timeout",
            payload={
                "pid": getattr(proc, "pid", None),
                "reason": reason_label,
                "escalate_code": escalate_code,
                "log_path": str(log_path),
                "total_epochs": total_epochs,
            },
            severity="warning",
        )
        # TOCTOU guard: re-check just before killing.
        rc_now = proc.poll()
        if rc_now is not None:
            logger.info(
                "Subprocess exited just before kill (%s); using natural rc=%d", reason_label, rc_now
            )
            return rc_now

        if not hasattr(os, "killpg"):
            # Non-POSIX fallback: use proc.terminate() / proc.kill()
            # which the subprocess module routes by PID safely.
            logger.warning("os.killpg unavailable; using proc.terminate() (%s)", reason_label)
            try:
                proc.terminate()
                try:
                    return proc.wait(timeout=policy.kill_grace_seconds)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                    return escalate_code
            except (ProcessLookupError, PermissionError):
                # Process gone; collect real rc.
                rc_after = proc.poll()
                return rc_after if rc_after is not None else escalate_code

        try:
            os.killpg(proc.pid, policy.kill_signal)
            try:
                return proc.wait(timeout=policy.kill_grace_seconds)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
                return escalate_code
        except (ProcessLookupError, PermissionError):
            # Process gone between poll and killpg (TOCTOU). Collect rc.
            rc_after = proc.poll()
            if rc_after is not None:
                return rc_after
            return escalate_code

    while True:
        rc = proc.poll()
        if rc is not None:
            elapsed_min = (time.time() - start) / 60
            if grace_active:
                logger.info(
                    "Subprocess finished under grace at %.1f min "
                    "(hard cap %.0f min, grace threshold %.0f%%)",
                    elapsed_min,
                    policy.hard_cap_seconds / 60,
                    policy.grace_progress_threshold * 100,
                )
            return rc

        elapsed = time.time() - start

        if grace_active:
            # Bounded grace: check absolute extension cap + stall.
            grace_elapsed = time.time() - (grace_started_at or time.time())
            if grace_elapsed >= policy.grace_max_extension_seconds:
                logger.error(
                    "Subprocess exceeded grace extension (%d s) at total "
                    "elapsed %.0f min. Force-killing.",
                    policy.grace_max_extension_seconds,
                    elapsed / 60,
                )
                return _kill_group("grace-extension-exhausted", -3)

            # Stall check: progress must advance during grace.
            # Round 4 M6 fix: a persistent log-read failure (disk full,
            # truncation, encoding) means progress=None on every poll,
            # so the stall counter previously never advanced — leaving
            # only the absolute extension cap as a safety net. Now we
            # increment stall_polls EVEN when progress is None
            # (treated as a stall poll) so disk-failure deadlocks are
            # caught within ``grace_stall_max_polls`` cycles.
            progress = parse_current_epoch(
                log_path,
                epoch_pattern,
                expected_total=total_epochs,
                tail_bytes=policy.log_tail_read_bytes,
            )
            if progress is None:
                stall_polls += 1
                if stall_polls >= policy.grace_stall_max_polls:
                    logger.error(
                        "Subprocess in grace period: no parseable "
                        "progress for %d consecutive polls (log=%s). "
                        "Force-killing — likely log-write failure / "
                        "stall.",
                        stall_polls,
                        log_path,
                    )
                    return _kill_group("grace-stall-no-progress", -3)
            else:
                cur_epoch = progress[0]
                if last_observed_epoch is None or cur_epoch > last_observed_epoch:
                    last_observed_epoch = cur_epoch
                    stall_polls = 0
                else:
                    stall_polls += 1
                    if stall_polls >= policy.grace_stall_max_polls:
                        logger.error(
                            "Subprocess stalled at epoch %d for %d "
                            "consecutive polls during grace. "
                            "Force-killing.",
                            cur_epoch,
                            stall_polls,
                        )
                        return _kill_group("grace-stall", -3)
            time.sleep(policy.grace_check_interval_seconds)
            continue

        if elapsed >= policy.hard_cap_seconds:
            # Hard cap reached. Decide grace vs kill.
            progress = parse_current_epoch(
                log_path,
                epoch_pattern,
                expected_total=total_epochs,
                tail_bytes=policy.log_tail_read_bytes,
            )
            if progress is None:
                logger.error(
                    "Subprocess hit hard cap (%.0f min) but progress parsing "
                    "failed (log=%s). Killing.",
                    elapsed / 60,
                    log_path,
                )
                return _kill_group("hard-cap-no-progress", -1)

            current, total = progress
            fraction = current / total
            if fraction >= policy.grace_progress_threshold:
                logger.warning(
                    "Subprocess hit hard cap (%.0f min) at epoch %d/%d "
                    "= %.1f%% (>= %.0f%%). GRANTING GRACE: up to %d more "
                    "min before force-kill.",
                    elapsed / 60,
                    current,
                    total,
                    fraction * 100,
                    policy.grace_progress_threshold * 100,
                    policy.grace_max_extension_seconds // 60,
                )
                emit_resource_event_from_env(
                    "resource.training_timeout_grace_granted",
                    action_type="training_timeout",
                    actor_ref="resource_guard:training_timeout",
                    payload={
                        "pid": getattr(proc, "pid", None),
                        "current_epoch": current,
                        "total_epochs": total,
                        "progress_fraction": fraction,
                        "grace_max_extension_seconds": policy.grace_max_extension_seconds,
                    },
                    severity="warning",
                )
                grace_active = True
                grace_started_at = time.time()
                last_observed_epoch = current
                time.sleep(policy.grace_check_interval_seconds)
                continue

            logger.error(
                "Subprocess hit hard cap (%.0f min) at epoch %d/%d "
                "= %.1f%% (< %.0f%% grace threshold). Killing.",
                elapsed / 60,
                current,
                total,
                fraction * 100,
                policy.grace_progress_threshold * 100,
            )
            return _kill_group("hard-cap-below-threshold", -1)

        time.sleep(policy.grace_check_interval_seconds)


# -----------------------------------------------------------------------------
# Partial-summary policy
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PartialSummaryPolicy:
    """Policy for emitting partial summaries when some cells fail.

    Attributes:
        max_failure_rate: Tier abort threshold. If the fraction of
            failed cells > this value, the tier IS aborted (no partial
            summary). Default 0.3 (30% failure tolerated).
        frontier_discount: Multiplicative discount applied to a
            partial-summary variant's metric for frontier ranking
            purposes. Default 0.95 (5% penalty), so a partial FF=+6.0
            ranks as +5.7 against clean variants.
    """

    max_failure_rate: float = 0.3
    frontier_discount: float = 0.95


def should_emit_partial_summary(
    failed_cells: int,
    total_cells: int,
    policy: PartialSummaryPolicy | None = None,
) -> tuple[bool, str]:
    """Decide whether a tier should emit a (possibly partial) summary.

    Args:
        failed_cells: Number of cells that failed (timeout / crash / etc).
        total_cells: Total task-declared evaluation units in the run.
        policy: PartialSummaryPolicy. Uses defaults if None.

    Returns:
        (should_emit, reason) tuple.
        - should_emit=True: emit a summary, possibly partial. Caller
          should set ``partial=True`` flag on the summary if
          failed_cells > 0.
        - should_emit=False: tier aborted, do not emit summary. The
          ``reason`` string explains why.

    Decision rule:
        - 0 failures: emit clean summary.
        - 0 < failure_rate <= policy.max_failure_rate: emit partial.
        - failure_rate > policy.max_failure_rate: abort.
    """
    if policy is None:
        policy = PartialSummaryPolicy()
    if total_cells <= 0:
        return False, "total_cells <= 0"
    if failed_cells <= 0:
        return True, "ok"
    if failed_cells >= total_cells:
        return False, "all cells failed"
    failure_rate = failed_cells / total_cells
    if failure_rate > policy.max_failure_rate:
        return False, (
            f"failure rate {failure_rate:.0%} > {policy.max_failure_rate:.0%} max — aborting tier"
        )
    return True, (
        f"partial: {failed_cells}/{total_cells} cells failed "
        f"({failure_rate:.0%}, within {policy.max_failure_rate:.0%} budget)"
    )


def apply_frontier_discount(
    metric_value: float,
    is_partial: bool,
    policy: PartialSummaryPolicy | None = None,
) -> float:
    """Apply the partial-summary frontier discount.

    For ranking purposes, partial-summary variants get a multiplicative
    discount so they rank below clean variants of nominally equal metric.
    Returns the original ``metric_value`` if not partial.
    """
    if not is_partial:
        return metric_value
    if policy is None:
        policy = PartialSummaryPolicy()
    return metric_value * policy.frontier_discount
