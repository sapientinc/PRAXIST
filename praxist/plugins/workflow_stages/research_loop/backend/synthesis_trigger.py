"""v2026-05-04 Synthesis trigger — event-driven generation termination.

Replaces the old `per_generation_hours` fixed timer. A generation runs
until information density (findings count + contributing peers) reaches
a threshold OR an explicitly configured safety cap fires. A null maximum
interval keeps evidence and cohort-completion events active without a timer cap.

When close criteria are met while protected evaluations are still running, the
orchestrator first writes `<gen_dir>/CLOSING_SIGNAL`. Peers use this as a
session-boundary drain signal: finish the current runtime session, do not open a
new one. Protected work always drains naturally. Once it has drained, the
configured agent grace bounds passive waits and final note publication before
the trigger writes `<gen_dir>/STOP_SIGNAL`. Peers watch that file via
`StopChecker` (see
praxist/plugins/workflow_stages/research_loop/backend/agent.py) and exit
gracefully at their next safe checkpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

from praxist.plugins.workflow_stages.research_loop.backend.event_wait import (
    FileEventWaitResult,
    wait_for_filesystem_event,
)
from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
    durable_promotion_exclusion,
    evidence_maturity_snapshot,
    has_explicit_false_completion,
    missing_required_ratio_telemetry,
    normalize_maturity_policy,
    resolve_result_snapshot_producers,
    task_authorizes_descriptive_maturity,
)
from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
    result_artifact_key as _result_artifact_key,
)
from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
    result_snapshot_key as _result_snapshot_key,
)
from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
    same_result_snapshot as _same_result_snapshot,
)

logger = logging.getLogger(__name__)

STOP_SIGNAL_FILENAME = "STOP_SIGNAL"
CLOSING_SIGNAL_FILENAME = "CLOSING_SIGNAL"
STOP_SIGNAL_POSTGEN_FILENAME = "STOP_SIGNAL_POSTGEN"  # written if peers finish before trigger
_INTERNAL_SYNTHETIC_PRODUCER_IDS = frozenset({"gems_agent", "tiered_eval_auto"})
_INTERNAL_SYNTHETIC_PRODUCER_PATTERN = re.compile(
    r"gen(?P<generation>\d+)_(?:late_signal|protected_jobs|result_artifact|unknown_peer)"
)


def _is_internal_synthetic_producer_id(
    value: object,
    *,
    generation_id: int | None = None,
) -> bool:
    peer_id = str(value or "").strip()
    if peer_id in _INTERNAL_SYNTHETIC_PRODUCER_IDS:
        return True
    match = _INTERNAL_SYNTHETIC_PRODUCER_PATTERN.fullmatch(peer_id)
    if match is None:
        return False
    return generation_id is None or int(match.group("generation")) == generation_id


@dataclass
class TriggerSnapshot:
    """Snapshot of a single trigger evaluation, for logging + diagnostics."""

    fired: bool
    reason: str  # "info_density" | "safety_cap" | "not_yet"
    findings_count: int
    minutes_since_start: float
    contributing_peers: int
    evidence_units: float = 0.0
    formal_result_peers: int = 0
    mature_result_peers: int = 0
    mature_result_count: int = 0
    required_mature_result_peers: int = 0
    active_protected_pids: int = 0
    active_generation_work: int = 0
    assessment_started: bool = False


@dataclass
class AdaptiveSynthesisPolicy:
    """Optional evidence-aware generation-close policy.

    The fixed synthesis trigger counts raw findings. That is appropriate for
    short generic tasks, but long-running evaluation tasks can produce many
    low-evidence notes while complete task-defined evaluations are still running. This
    policy preserves the default fixed behavior unless explicitly enabled by
    the task spec.
    """

    enabled: bool = False
    min_evidence_units: float = 0.0
    min_formal_result_peers: int = 0
    min_interval_floor_minutes: float = 0.0
    max_interval_ceiling_minutes: float = 0.0
    drain_grace_minutes: float = 5.0
    evidence_weights: dict[str, float] | None = None
    smoke_weight: float = 0.25
    result_finding_weight: float = 1.0

    @classmethod
    def from_raw(cls, raw: object | None) -> AdaptiveSynthesisPolicy:
        if raw is None:
            return cls()
        if isinstance(raw, AdaptiveSynthesisPolicy):
            return raw
        if not isinstance(raw, dict):
            logger.warning(
                "synthesis_trigger.adaptive must be an object; got %s — disabled",
                type(raw).__name__,
            )
            return cls()
        weights_raw = raw.get("evidence_weights")
        weights: dict[str, float] = {}
        if isinstance(weights_raw, dict):
            for key, value in weights_raw.items():
                try:
                    weights[str(key)] = float(value)
                except (TypeError, ValueError):
                    logger.warning(
                        "Ignoring invalid adaptive evidence weight %r=%r",
                        key,
                        value,
                    )
        return cls(
            enabled=_truthy(raw.get("enabled", False)),
            min_evidence_units=_float_or_default(raw.get("min_evidence_units"), 0.0),
            min_formal_result_peers=max(
                0,
                int(_float_or_default(raw.get("min_formal_result_peers"), 0.0)),
            ),
            min_interval_floor_minutes=_float_or_default(
                raw.get("min_interval_floor_minutes"),
                0.0,
            ),
            max_interval_ceiling_minutes=_float_or_default(
                raw.get("max_interval_ceiling_minutes"),
                0.0,
            ),
            drain_grace_minutes=max(
                0.0,
                _float_or_default(raw.get("drain_grace_minutes"), cls.drain_grace_minutes),
            ),
            evidence_weights=weights or None,
            smoke_weight=_float_or_default(raw.get("smoke_weight"), 0.25),
            result_finding_weight=_float_or_default(raw.get("result_finding_weight"), 1.0),
        )


def _float_or_default(value: object, default: float) -> float:
    if not isinstance(value, int | float | str):
        return float(default)
    try:
        parsed = float(value)
    except ValueError:
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return parsed


def _int_or_default(value: object, default: int) -> int:
    return int(_float_or_default(value, float(default)))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _falsey(value: object) -> bool:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return value == 0
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no", "n", "off"}
    return False


def _payload_has_validation_only_marker(payload: dict[str, object]) -> bool:
    if any(
        _truthy(payload.get(key))
        for key in (
            "validation_only",
            "validation_only_result",
            "late_after_generation_boundary",
        )
    ):
        return True
    for key in ("artifact_signal_status", "late_result_policy", "durability_scope"):
        token = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(payload.get(key) or "").strip().lower(),
        ).strip("_")
        if token in {
            "late_after_generation_boundary",
            "late_quarantined_protected_job",
            "quarantined_signal",
            "validation_signal_only",
            "validation_only",
        }:
            return True
    return False


def _payload_has_hard_non_mature_status(payload: dict[str, object]) -> bool:
    finding_type = str(payload.get("finding_type") or "").strip().lower()
    if finding_type and finding_type != "result":
        return True
    if _payload_has_validation_only_marker(payload):
        return True
    raw_protocol_violation_count = payload.get("protocol_integrity_violation_count")
    if isinstance(raw_protocol_violation_count, bool):
        protocol_violation_count = 0.0
    elif isinstance(raw_protocol_violation_count, (int, float)):
        protocol_violation_count = float(raw_protocol_violation_count)
    elif isinstance(raw_protocol_violation_count, str):
        try:
            protocol_violation_count = float(raw_protocol_violation_count)
        except ValueError:
            protocol_violation_count = 0.0
    else:
        protocol_violation_count = 0.0
    if protocol_violation_count > 0:
        return True

    if has_explicit_false_completion(payload):
        return True

    for key in (
        "incomplete_eval",
        "is_incomplete_eval",
        "summary_only",
        "is_summary_only",
        "unscored_artifact",
        "suspect",
        "suspect_protocol",
        # Legacy input alias only; new task outputs should use suspect_protocol.
        "suspect_fixed_weight_eval",
        "protocol_integrity_failed",
    ):
        if _truthy(payload.get(key)):
            return True

    if payload.get("protocol_integrity_passed") is False:
        return True

    status_text = " ".join(
        str(payload.get(key) or "")
        for key in (
            "status",
            "final_status",
            "tier_status",
            "result_status",
            "protocol_integrity_status",
        )
    ).lower()
    status_tokens = set(token for token in re.split(r"[^a-z0-9]+", status_text) if token)
    if (
        {"scored", "complete", "false"}.issubset(status_tokens)
        or {"not", "scored", "complete"}.issubset(status_tokens)
        or {"not", "complete"}.issubset(status_tokens)
    ):
        return True
    return any(
        token in status_text
        for token in (
            "failed",
            "crashed",
            "incomplete",
            "running",
            "invalid",
            "error",
            "summary_only",
            "unscored",
            "not_scored",
        )
    )


def _payload_has_soft_non_mature_label(payload: dict[str, object]) -> bool:
    if not str(payload.get("finding_type") or "").strip() and not _payload_has_completion_marker(
        payload
    ):
        return True
    return _payload_has_explicit_soft_non_mature_marker(payload)


def _payload_has_explicit_soft_non_mature_marker(payload: dict[str, object]) -> bool:
    for key in (
        "partial",
        "partial_cohort",
        "partial_eval",
        "is_partial_eval",
        "scout_only",
        "is_scout_eval",
        "is_smoke_eval",
        "smoke_only",
        "capped",
        "is_capped",
        "result_capped",
    ):
        if _truthy(payload.get(key)):
            return True
    status_text = " ".join(
        str(payload.get(key) or "")
        for key in (
            "status",
            "final_status",
            "tier_status",
            "result_status",
            "protocol_integrity_status",
        )
    ).lower()
    status_tokens = set(token for token in re.split(r"[^a-z0-9]+", status_text) if token)
    return bool(
        status_tokens
        & {
            "partial",
            "scout",
            "smoke",
            "preliminary",
            "prelim",
            "cheap_probe",
            "capped",
        }
    )


def _payload_has_completion_marker(payload: dict[str, object]) -> bool:
    for key in ("scored_complete", "complete_eval", "is_complete_eval"):
        if _truthy(payload.get(key)):
            return True
    explicit_statuses = {
        "complete_eval",
        "full_evaluation",
        "scored_complete",
    }
    return any(
        str(payload.get(key) or "").strip().lower().replace("-", "_").replace(" ", "_")
        in explicit_statuses
        for key in ("status", "final_status", "tier_status", "result_status")
    )


def _json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _merged_evidence_payload(
    metrics: dict[str, object],
    extra: dict[str, object],
) -> dict[str, object]:
    nested_extra = extra.get("extra") if isinstance(extra.get("extra"), dict) else {}
    merged = {**nested_extra, **extra, **metrics}
    for source in (metrics, extra, nested_extra):
        aggregate = source.get("current_aggregate")
        if isinstance(aggregate, dict):
            for key, value in aggregate.items():
                merged.setdefault(key, value)
    return merged


def _variant_identity(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return ""


def _json_array_from_path(path: Path) -> list[object]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SynthesisTrigger:
    """Watches findings/state changes and decides when to fire synthesis.

    Thread-/asyncio-safe in the sense that it's read-only against SQLite
    (uses short-lived connections) and the only write is the STOP_SIGNAL
    sentinel file (atomic rename).
    """

    def __init__(
        self,
        run_dir: Path,
        gen_dir: Path,
        gen_id: int,
        gen_start_time: float,
        min_findings: int = 30,
        min_interval_minutes: float = 120.0,
        max_interval_minutes: float | None = 240.0,
        min_contributing_peers: int = 3,
        poll_interval_seconds: int = 30,
        adaptive_policy: object | None = None,
        maturity_policy: object | None = None,
        mature_quorum_fraction: float = 0.0,
        cohort_size: int = 0,
        store_db_filename: str = "shared_store.db",
        # R2#5 fix: optional callback invoked before each SQLite evaluation
        # to force any pending filesystem→SQLite sync (FindingsSync). This
        # closes the up-to-60s lag between a peer writing a JSON finding
        # and that finding becoming visible to the trigger SQL query.
        pre_eval_sync_callback: Callable[[], None] | None = None,
        # #148: optional callback that returns the number of cohort
        # peers still running. When it drops to 0 after the warmup
        # window, ``evaluate()`` fires with reason ``cohort_drained``.
        # Closes the "all peers died with errors, ``synthesis_trigger``
        # waits another 240 min before timing out" deadlock.
        cohort_active_peers_callback: Callable[[], int] | None = None,
        cohort_drain_warmup_seconds: float = 60.0,
        # #75 batch 9 (config discipline): explicit overrides for the
        # two env-fallbacks below. When the orchestrator wires the
        # values through (e.g. cohort_runner forwards run_dir as
        # ``local_store_dir`` and the heartbeat from the task spec),
        # the env reads disappear entirely. Defaults preserve legacy
        # callers that still drive these via env between construction
        # and use.
        local_store_dir: Path | str | None = None,
        event_heartbeat_seconds: int | None = None,
        started_peer_ids: Collection[str] | None = None,
        cohort_completed_event: asyncio.Event | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.gen_dir = Path(gen_dir)
        self.gen_id = gen_id
        self.gen_start_time = _float_or_default(gen_start_time, time.time())
        self.min_findings = _int_or_default(min_findings, 30)
        self.min_interval_minutes = _float_or_default(min_interval_minutes, 120.0)
        self.max_interval_minutes = (
            None if max_interval_minutes is None else _float_or_default(max_interval_minutes, 240.0)
        )
        self.min_contributing_peers = _int_or_default(min_contributing_peers, 3)
        self.adaptive_policy = AdaptiveSynthesisPolicy.from_raw(adaptive_policy)
        self.maturity_policy = normalize_maturity_policy(maturity_policy)
        try:
            parsed_quorum = float(mature_quorum_fraction)
            self.mature_quorum_fraction = (
                min(1.0, max(0.0, parsed_quorum)) if math.isfinite(parsed_quorum) else 0.0
            )
        except (TypeError, ValueError):
            self.mature_quorum_fraction = 0.0
        self.cohort_size = max(0, _int_or_default(cohort_size, 0))
        self.started_peer_ids = (
            {
                peer_id
                for value in started_peer_ids
                if (peer_id := str(value or "").strip())
                and re.fullmatch(rf"gen{self.gen_id}_peer\d+", peer_id)
            }
            if started_peer_ids is not None
            else None
        )
        if self.cohort_size > 0:
            self.min_contributing_peers = min(
                self.min_contributing_peers,
                self.cohort_size,
            )
        self.required_mature_result_peers = (
            int(math.ceil(self.cohort_size * self.mature_quorum_fraction))
            if self.mature_quorum_fraction > 0 and self.cohort_size > 0
            else 0
        )
        if self.adaptive_policy.enabled:
            if self.adaptive_policy.min_interval_floor_minutes > 0:
                self.min_interval_minutes = max(
                    self.min_interval_minutes,
                    self.adaptive_policy.min_interval_floor_minutes,
                )
            if (
                self.max_interval_minutes is not None
                and self.adaptive_policy.max_interval_ceiling_minutes > 0
            ):
                self.max_interval_minutes = max(
                    self.max_interval_minutes,
                    self.adaptive_policy.max_interval_ceiling_minutes,
                )
        # Compatibility knob: older task specs call this poll_interval_seconds.
        # The event-driven trigger treats it as a lower bound only; the default
        # heartbeat is intentionally sparse so trigger supervision does not
        # become a high-frequency background loop.
        # #75 batch 9 (config discipline): heartbeat comes from the
        # explicit kwarg or the built-in 900s default; no env read.
        # Orchestrators that want to override pass
        # ``event_heartbeat_seconds`` directly. The legacy
        # ``PRAXIST_SYNTHESIS_EVENT_HEARTBEAT_SECONDS`` env knob was
        # operator-facing but unused by any in-tree caller; the same
        # knob is reachable today by setting the task spec's
        # ``synthesis_trigger.poll_interval_seconds``.
        heartbeat = (
            _int_or_default(event_heartbeat_seconds, 900)
            if event_heartbeat_seconds is not None
            else 900
        )
        self.poll_interval_seconds = float(
            max(60.0, _float_or_default(poll_interval_seconds, 30.0), heartbeat)
        )
        # R1#7 fix: honor LOCAL_STORE_DIR env (set by generation_loop.run);
        # falls back to run_dir for backwards compat.
        # #75 batch 9: ``local_store_dir`` is now the only source. The
        # orchestrator (cohort_runner) threads it from ``loop.run_dir``;
        # tests construct with an explicit value. No env read.
        store_root = str(local_store_dir) if local_store_dir is not None else str(self.run_dir)
        self.db_path = Path(store_root) / store_db_filename

        self._stop_signal_path = self.gen_dir / STOP_SIGNAL_FILENAME
        self._closing_signal_path = self.gen_dir / CLOSING_SIGNAL_FILENAME
        self._signal_lock = threading.Lock()
        self._fired = False
        self._closing = False
        self._snapshots: list[TriggerSnapshot] = []
        self._pre_eval_sync_callback = pre_eval_sync_callback
        self._cohort_active_peers_callback = cohort_active_peers_callback
        self._cohort_completed_event = cohort_completed_event
        self._cohort_drain_warmup_seconds = float(cohort_drain_warmup_seconds)
        self._drain_started_at: float | None = None
        self._assessment_started = False
        self._warned_noncanonical_peer_ids: set[str] = set()
        self._warned_missing_required_ratios = False

    def _query_gen_state(self) -> tuple[int, int]:
        """Return (findings_count, contributing_peers) for this gen.

        Synchronous SQLite read; intended to be invoked via
        `asyncio.to_thread` so it doesn't block the event loop
        (R1#8 fix). Returns (0, 0) if the DB does not yet exist
        (e.g. early in gen 0 before first finding).

        R1#3 fix: COUNT(DISTINCT peer_id) excludes empty/NULL peer_id rows
        — empty stamps from misconfigured peers would otherwise collapse
        to a single-peer count and mask real cohort participation.
        """
        try:
            if not self.db_path.exists():
                return 0, 0
        except OSError:
            return 0, 0
        conn = None
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                timeout=2.0,
            )
            rows = list(
                conn.execute(
                    "SELECT peer_id, COUNT(*) FROM findings WHERE generation_id = ? GROUP BY peer_id",
                    (self.gen_id,),
                )
            )
            n_findings = sum(int(row[1] or 0) for row in rows)
            distinct_ids = {str(row[0] or "").strip() for row in rows if str(row[0] or "").strip()}
            canonical_ids = {
                peer_id
                for value in distinct_ids
                if (peer_id := self._canonical_peer_id(value)) is not None
            }
            n_canonical = len(canonical_ids)
            malformed_ids = {
                peer_id
                for peer_id in distinct_ids - canonical_ids
                if not _is_internal_synthetic_producer_id(
                    peer_id,
                    generation_id=self.gen_id,
                )
            }
            new_malformed_ids = malformed_ids - self._warned_noncanonical_peer_ids
            if new_malformed_ids:
                self._warned_noncanonical_peer_ids.update(new_malformed_ids)
                logger.warning(
                    "synthesis_trigger gen %d: %d new non-canonical peer_id "
                    "stamp(s) detected (canonical=%d): %s. Possible cross-gen "
                    "leakage or misconfigured peer.",
                    self.gen_id,
                    len(new_malformed_ids),
                    n_canonical,
                    ", ".join(sorted(new_malformed_ids)),
                )
            # Use canonical count for trigger gating (stricter)
            return n_findings, n_canonical
        except sqlite3.Error as e:
            logger.debug("synthesis_trigger: SQLite read failed (%s); treating as 0 findings", e)
            return 0, 0
        except Exception as e:  # R1#10 catch-all for OSError, MemoryError, etc.
            logger.warning(
                "synthesis_trigger: unexpected error in "
                "_query_gen_state (%s); treating as 0 findings",
                type(e).__name__,
            )
            return 0, 0
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

    def _query_adaptive_state(self) -> tuple[float, int]:
        """Return ``(evidence_units, formal_result_peers)`` for this generation.

        Evidence is derived from result-like metrics/finding records and
        deduplicated by peer plus variant/run id so repeated notebook notes do
        not prematurely close a generation. Malformed JSON is ignored because
        the fixed trigger remains the compatibility path.
        """
        if not self.adaptive_policy.enabled:
            return 0.0, 0
        try:
            if not self.db_path.exists():
                return 0.0, 0
        except OSError:
            return 0.0, 0

        best_by_key: dict[tuple[str, str], float] = {}
        artifact_records: list[tuple[tuple[str, str, str], float, set[str]]] = []
        formal_peers: set[str] = set()
        attributed_auto_results = 0

        conn = None
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                timeout=2.0,
            )
            conn.row_factory = sqlite3.Row
            metric_rows = list(
                conn.execute(
                    "SELECT run_id, variant_name, metrics, peer_id "
                    "FROM metrics WHERE generation_id = ?",
                    (self.gen_id,),
                )
            )
            finding_rows = list(
                conn.execute(
                    "SELECT finding_type, variant_name, title, metrics, extra, peer_id "
                    "FROM findings WHERE generation_id = ?",
                    (self.gen_id,),
                )
            )
            metric_payloads = [_json_object(row["metrics"]) for row in metric_rows]
            finding_records = []
            for row in finding_rows:
                metrics = _json_object(row["metrics"])
                extra = _json_object(row["extra"])
                finding_records.append(
                    (
                        row,
                        _merged_evidence_payload(metrics, extra),
                        {"metrics": metrics, "extra": extra},
                    )
                )
            resolved_artifacts = resolve_result_snapshot_producers(
                [
                    *(
                        (_result_snapshot_key(payload), row["variant_name"])
                        for row, payload in zip(metric_rows, metric_payloads, strict=True)
                    ),
                    *(
                        (_result_snapshot_key(identity_payload), row["variant_name"])
                        for row, _payload, identity_payload in finding_records
                    ),
                ]
            )
            metric_artifacts = resolved_artifacts[: len(metric_rows)]
            finding_artifacts = resolved_artifacts[len(metric_rows) :]
            finding_records = [
                (row, payload, artifact)
                for (row, payload, _identity_payload), artifact in zip(
                    finding_records,
                    finding_artifacts,
                    strict=True,
                )
            ]
            non_durable_artifacts = {
                artifact
                for payload, artifact in [
                    *zip(metric_payloads, metric_artifacts, strict=True),
                    *((payload, artifact) for _row, payload, artifact in finding_records),
                ]
                if self._payload_is_explicitly_non_durable(payload)
                and artifact is not None
                and bool(artifact[0])
            }

            def record_evidence(
                payload: dict[str, object],
                artifact: tuple[str, str, str] | None,
                fallback_key: tuple[str, str],
                units: float,
                peer_id: str | None,
            ) -> None:
                non_durable = self._payload_is_explicitly_non_durable(payload)
                if artifact is None:
                    best_by_key[fallback_key] = max(
                        units,
                        best_by_key.get(fallback_key, 0.0),
                    )
                    if (
                        not non_durable
                        and units >= self.adaptive_policy.result_finding_weight
                        and peer_id is not None
                    ):
                        formal_peers.add(peer_id)
                    return
                artifact_is_non_durable = any(
                    _same_result_snapshot(artifact, non_durable_artifact)
                    for non_durable_artifact in non_durable_artifacts
                )
                if artifact_is_non_durable and not non_durable:
                    return
                for index, (known_artifact, known_units, known_peers) in enumerate(
                    artifact_records
                ):
                    if not _same_result_snapshot(artifact, known_artifact):
                        continue
                    peers = set(known_peers)
                    if peer_id is not None and not non_durable:
                        peers.add(peer_id)
                    artifact_records[index] = (
                        known_artifact,
                        max(known_units, units),
                        peers,
                    )
                    return
                artifact_records.append(
                    (
                        artifact,
                        units,
                        {peer_id} if peer_id is not None and not non_durable else set(),
                    )
                )

            for row, payload, artifact in zip(
                metric_rows,
                metric_payloads,
                metric_artifacts,
                strict=True,
            ):
                units = self._evidence_units_from_payload(payload, row["run_id"])
                if units <= 0:
                    continue
                key = (
                    str(row["peer_id"] or ""),
                    str(row["variant_name"] or row["run_id"] or ""),
                )
                record_evidence(
                    payload,
                    artifact,
                    key,
                    units,
                    self._canonical_peer_id(row["peer_id"]),
                )

            variant_to_peers = self._canonical_variant_peer_map(finding_rows)
            for row, payload, artifact in finding_records:
                units = self._evidence_units_from_payload(payload, row["title"])
                if units <= 0:
                    continue
                key = (
                    str(row["peer_id"] or ""),
                    str(row["variant_name"] or row["title"] or ""),
                )
                peer_id = self._canonical_peer_id(row["peer_id"])
                if (
                    peer_id is None
                    and str(row["finding_type"] or "") == "result"
                    and _truthy(payload.get("auto_materialized_from_result_artifact"))
                    and str(row["peer_id"] or "").strip() != f"gen{self.gen_id}_unknown_peer"
                ):
                    peer_id = self._infer_auto_result_peer(row, variant_to_peers)
                    if peer_id is not None:
                        attributed_auto_results += 1
                record_evidence(payload, artifact, key, units, peer_id)

            for artifact, units, peers in artifact_records:
                artifact_is_non_durable = any(
                    _same_result_snapshot(artifact, non_durable_artifact)
                    for non_durable_artifact in non_durable_artifacts
                )
                if (
                    not artifact_is_non_durable
                    and units >= self.adaptive_policy.result_finding_weight
                    and peers
                ):
                    formal_peers.add(sorted(peers)[0])
        except sqlite3.Error as e:
            logger.debug("adaptive synthesis state read failed (%s); treating as 0", e)
            return 0.0, 0
        except Exception as e:
            logger.warning(
                "adaptive synthesis state read raised %s; treating as 0",
                type(e).__name__,
            )
            return 0.0, 0
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

        if attributed_auto_results:
            logger.debug(
                "adaptive synthesis gen %d: attributed %d tiered_eval_auto "
                "result(s) to canonical peer evidence for formal gate",
                self.gen_id,
                attributed_auto_results,
            )
        return float(
            sum(best_by_key.values()) + sum(units for _artifact, units, _peers in artifact_records)
        ), len(formal_peers)

    def mature_result_count(self, *, synchronize: bool = True) -> int:
        """Return unique mature experiment results for scheduler feedback.

        This uses the same effort/coverage policy and the same canonical store
        as synthesis.  Variant-bearing evidence is counted by peer plus variant;
        legacy variant-less evidence contributes at most one result per peer.
        """

        if synchronize and self._pre_eval_sync_callback is not None:
            try:
                self._pre_eval_sync_callback()
            except Exception as exc:  # noqa: BLE001 - stale evidence is advisory here.
                logger.debug("mature result pre-sync failed: %s", exc)
        _mature_peers, _mature_peer_variants, result_count = self._query_mature_evidence_details()
        return result_count

    def mature_peer_count(self, *, synchronize: bool = True) -> int:
        """Return distinct mature peers for the legacy peer-quorum contract."""

        return len(self.mature_peer_ids(synchronize=synchronize))

    def mature_peer_ids(self, *, synchronize: bool = True) -> set[str]:
        """Return canonical distinct peer identities with mature evidence."""

        if synchronize and self._pre_eval_sync_callback is not None:
            try:
                self._pre_eval_sync_callback()
            except Exception as exc:  # noqa: BLE001 - stale evidence is advisory here.
                logger.debug("mature peer pre-sync failed: %s", exc)
        mature_peers, mature_peer_variants = self._query_mature_evidence_state()
        mature_peers.update(peer_id for peer_id, _variant_id in mature_peer_variants)
        return mature_peers

    def _query_mature_state(self) -> int:
        """Return distinct peers with mature-enough result evidence.

        Ratio fields are authoritative when present. Legacy evidence units are
        used only as compatibility fallback when the task has not opted into
        a strict ratio gate.
        """

        mature_peers, mature_peer_variants = self._query_mature_evidence_state()
        mature_peers.update(peer_id for peer_id, _variant_id in mature_peer_variants)
        return len(mature_peers)

    def _query_mature_evidence_state(
        self,
    ) -> tuple[set[str], set[tuple[str, str]]]:
        """Return canonical mature peer and peer/variant identities."""

        mature_peers, mature_peer_variants, _result_count = self._query_mature_evidence_details()
        return mature_peers, mature_peer_variants

    def _query_mature_evidence_details(
        self,
    ) -> tuple[set[str], set[tuple[str, str]], int]:
        """Return mature identities plus the number of distinct results."""

        try:
            if not self.db_path.exists():
                return set(), set(), 0
        except OSError:
            return set(), set(), 0

        mature_artifacts_by_peer: dict[str, list[tuple[str, str, str] | None]] = {}
        mature_artifacts_by_variant: dict[tuple[str, str], list[tuple[str, str, str] | None]] = {}
        unattributed_mature_artifacts: list[tuple[str, str, str]] = []
        validation_artifacts: set[tuple[str, str, str]] = set()
        missing_required_ratios: set[str] = set()

        def result_key(
            payload: dict[str, object],
            *,
            snapshot: tuple[str, str, str] | None,
            source: str,
            row_index: int,
            source_identity: object = "",
            source_variant: object = "",
        ) -> tuple[str, str, str] | None:
            if snapshot is not None:
                return snapshot
            artifact = _result_artifact_key(payload)
            source_token = str(source_identity or "").strip()
            explicit_metric_run = (
                source == "metric" and source_token and source_token != self.run_dir.name
            )
            metric_identity = json.dumps(
                [source_token, str(source_variant or "").strip()],
                ensure_ascii=True,
                separators=(",", ":"),
            )
            if artifact is None:
                return (
                    (f"__metric_run__:{metric_identity}", "", "") if explicit_metric_run else None
                )
            if explicit_metric_run:
                return f"__metric_run__:{metric_identity}", artifact[0], artifact[1]
            return f"__unverified_{source}_{row_index}", artifact[0], artifact[1]

        conn = None
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                timeout=2.0,
            )
            conn.row_factory = sqlite3.Row
            finding_rows = list(
                conn.execute(
                    "SELECT finding_type, variant_name, title, metrics, extra, peer_id "
                    "FROM findings WHERE generation_id = ?",
                    (self.gen_id,),
                )
            )
            variant_to_peers = self._canonical_variant_peer_map(finding_rows)

            try:
                metric_rows = list(
                    conn.execute(
                        "SELECT run_id, variant_name, metrics, peer_id "
                        "FROM metrics WHERE generation_id = ?",
                        (self.gen_id,),
                    )
                )
            except sqlite3.Error as e:
                logger.debug(
                    "mature synthesis metrics table read failed (%s); "
                    "falling back to findings rows",
                    e,
                )
                metric_rows = []

            metric_payloads = [_json_object(row["metrics"]) for row in metric_rows]
            finding_record_data = [
                (
                    _json_object(row["metrics"]),
                    _json_object(row["extra"]),
                )
                for row in finding_rows
            ]
            resolved_artifacts = resolve_result_snapshot_producers(
                [
                    *(
                        (_result_snapshot_key(payload), row["variant_name"])
                        for row, payload in zip(metric_rows, metric_payloads, strict=True)
                    ),
                    *(
                        (
                            _result_snapshot_key({"metrics": metrics, "extra": extra}),
                            row["variant_name"],
                        )
                        for row, (metrics, extra) in zip(
                            finding_rows,
                            finding_record_data,
                            strict=True,
                        )
                    ),
                ]
            )
            metric_artifacts = resolved_artifacts[: len(metric_rows)]
            finding_artifacts = resolved_artifacts[len(metric_rows) :]

            for row_index, (row, payload, snapshot) in enumerate(
                zip(metric_rows, metric_payloads, metric_artifacts, strict=True)
            ):
                explicitly_signal_only = _payload_has_hard_non_mature_status(
                    payload
                ) or _payload_has_explicit_soft_non_mature_marker(payload)
                if not explicitly_signal_only and (
                    _payload_has_completion_marker(payload) or _result_artifact_key(payload)
                ):
                    missing_required_ratios.update(
                        missing_required_ratio_telemetry(payload, self.maturity_policy)
                    )
                if self._payload_is_explicitly_non_durable(payload):
                    if snapshot is not None and snapshot[0]:
                        validation_artifacts.add(snapshot)
                    continue
                peer_id = self._canonical_peer_id(row["peer_id"])
                if peer_id is None:
                    continue
                variant_id = _variant_identity(row["variant_name"], row["run_id"])
                if self._payload_is_mature(payload, row["run_id"]):
                    if variant_id:
                        pair = (peer_id, variant_id)
                        mature_artifacts_by_variant.setdefault(pair, []).append(
                            result_key(
                                payload,
                                snapshot=snapshot,
                                source="metric",
                                row_index=row_index,
                                source_identity=row["run_id"],
                                source_variant=row["variant_name"],
                            )
                        )
                    else:
                        mature_artifacts_by_peer.setdefault(peer_id, []).append(
                            result_key(
                                payload,
                                snapshot=snapshot,
                                source="metric",
                                row_index=row_index,
                                source_identity=row["run_id"],
                                source_variant=row["variant_name"],
                            )
                        )

            for row_index, (row, record_data, snapshot) in enumerate(
                zip(finding_rows, finding_record_data, finding_artifacts, strict=True)
            ):
                finding_type = str(row["finding_type"] or "")
                metrics, extra = record_data
                payload = {
                    **_merged_evidence_payload(metrics, extra),
                    "finding_type": finding_type,
                    "variant_name": str(row["variant_name"] or ""),
                    "title": str(row["title"] or ""),
                }
                artifact_key = result_key(
                    {"metrics": metrics, "extra": extra},
                    snapshot=snapshot,
                    source="finding",
                    row_index=row_index,
                )
                explicitly_signal_only = _payload_has_hard_non_mature_status(
                    payload
                ) or _payload_has_explicit_soft_non_mature_marker(payload)
                if finding_type == "result" and not explicitly_signal_only:
                    missing_required_ratios.update(
                        missing_required_ratio_telemetry(payload, self.maturity_policy)
                    )
                if self._payload_is_explicitly_non_durable(payload):
                    if snapshot is not None and snapshot[0]:
                        validation_artifacts.add(snapshot)
                    continue
                peer_id = self._canonical_peer_id(row["peer_id"])
                if (
                    peer_id is None
                    and finding_type == "result"
                    and _truthy(payload.get("auto_materialized_from_result_artifact"))
                    and str(row["peer_id"] or "").strip() != f"gen{self.gen_id}_unknown_peer"
                ):
                    peer_id = self._infer_auto_result_peer(row, variant_to_peers)
                is_mature = self._payload_is_mature(payload, row["title"])
                if peer_id is None:
                    if (
                        finding_type == "result"
                        and _truthy(payload.get("auto_materialized_from_result_artifact"))
                        and is_mature
                        and artifact_key is not None
                    ):
                        unattributed_mature_artifacts.append(artifact_key)
                    continue
                variant_id = _variant_identity(row["variant_name"])
                if is_mature:
                    if variant_id:
                        pair = (peer_id, variant_id)
                        mature_artifacts_by_variant.setdefault(pair, []).append(artifact_key)
                    else:
                        mature_artifacts_by_peer.setdefault(peer_id, []).append(artifact_key)
        except sqlite3.Error as e:
            logger.debug("mature synthesis state read failed (%s); treating as 0", e)
            return set(), set(), 0
        except Exception as e:
            logger.warning(
                "mature synthesis state read raised %s; treating as 0",
                type(e).__name__,
            )
            return set(), set(), 0
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

        if missing_required_ratios and not self._warned_missing_required_ratios:
            self._warned_missing_required_ratios = True
            logger.warning(
                "generation %d has result evidence without required finite %s; "
                "the evidence remains available as a validation signal but cannot "
                "satisfy durable frontier, Gems, or mature-close eligibility. "
                "Validate an evaluator-produced summary with praxist resolve "
                "--result-summary before the next launch.",
                self.gen_id,
                ", ".join(sorted(missing_required_ratios)),
            )

        mature_peer_variants: set[tuple[str, str]] = set()
        counted_artifacts: list[tuple[str, str, str]] = []
        mature_result_count = 0

        def same_counted_result(
            left: tuple[str, str, str],
            right: tuple[str, str, str],
        ) -> bool:
            if left[0].startswith("__metric_run__:") or right[0].startswith("__metric_run__:"):
                return left == right
            return _same_result_snapshot(left, right)

        def prefer_attributed_snapshots(
            artifacts: list[tuple[str, str, str] | None],
        ) -> list[tuple[str, str, str] | None]:
            """Drop an unattributed duplicate only inside one peer/variant group."""

            attributed_coordinates = {
                (artifact[1], artifact[2])
                for artifact in artifacts
                if artifact is not None and artifact[0]
            }
            return [
                artifact
                for artifact in artifacts
                if artifact is None
                or artifact[0]
                or (artifact[1], artifact[2]) not in attributed_coordinates
            ]

        for pair, artifacts in sorted(mature_artifacts_by_variant.items()):
            artifacts = prefer_attributed_snapshots(artifacts)
            surviving = [
                artifact
                for artifact in artifacts
                if artifact is None
                or not any(
                    _same_result_snapshot(artifact, validation_artifact)
                    for validation_artifact in validation_artifacts
                )
            ]
            identified = [artifact for artifact in surviving if artifact is not None]
            has_new_result = any(artifact is None for artifact in surviving) or any(
                not any(
                    same_counted_result(artifact, counted_artifact)
                    for counted_artifact in counted_artifacts
                )
                for artifact in identified
            )
            if not has_new_result:
                continue
            mature_peer_variants.add(pair)
            new_artifacts: list[tuple[str, str, str]] = []
            for artifact in identified:
                if any(
                    same_counted_result(artifact, known_artifact)
                    for known_artifact in [*counted_artifacts, *new_artifacts]
                ):
                    continue
                new_artifacts.append(artifact)
            mature_result_count += len(new_artifacts) or int(
                any(artifact is None for artifact in surviving)
            )
            for artifact in identified:
                if not any(
                    same_counted_result(artifact, counted_artifact)
                    for counted_artifact in counted_artifacts
                ):
                    counted_artifacts.append(artifact)
        mature_peers: set[str] = set()
        for peer_id, artifacts in sorted(mature_artifacts_by_peer.items()):
            artifacts = prefer_attributed_snapshots(artifacts)
            surviving = [
                artifact
                for artifact in artifacts
                if artifact is None
                or not any(
                    _same_result_snapshot(artifact, validation_artifact)
                    for validation_artifact in validation_artifacts
                )
            ]
            identified = [artifact for artifact in surviving if artifact is not None]
            if any(artifact is None for artifact in surviving) or any(
                not any(
                    same_counted_result(artifact, counted_artifact)
                    for counted_artifact in counted_artifacts
                )
                for artifact in identified
            ):
                mature_peers.add(peer_id)
                new_artifacts = []
                for artifact in identified:
                    if any(
                        same_counted_result(artifact, known_artifact)
                        for known_artifact in [*counted_artifacts, *new_artifacts]
                    ):
                        continue
                    new_artifacts.append(artifact)
                mature_result_count += len(new_artifacts) or int(
                    any(artifact is None for artifact in surviving)
                )
                for artifact in identified:
                    if not any(
                        same_counted_result(artifact, counted_artifact)
                        for counted_artifact in counted_artifacts
                    ):
                        counted_artifacts.append(artifact)

        for artifact in unattributed_mature_artifacts:
            if any(artifact[1:] == counted_artifact[1:] for counted_artifact in counted_artifacts):
                continue
            counted_artifacts.append(artifact)
            mature_result_count += 1
        return mature_peers, mature_peer_variants, mature_result_count

    def _payload_is_explicitly_non_durable(self, payload: dict[str, object]) -> bool:
        result_payload = dict(payload)
        result_payload.pop("finding_type", None)
        if _payload_has_hard_non_mature_status(result_payload):
            return True
        if durable_promotion_exclusion(result_payload) is not None:
            return True
        snapshot = evidence_maturity_snapshot(result_payload, self.maturity_policy)
        task_stage_is_complete = snapshot.get(
            "mature_enough"
        ) is True and task_authorizes_descriptive_maturity(
            result_payload,
            self.maturity_policy,
            maturity=snapshot,
        )
        if (
            _payload_has_explicit_soft_non_mature_marker(result_payload)
            and not task_stage_is_complete
        ):
            return True
        return snapshot.get("mature_enough") is False

    def _payload_is_mature(self, payload: dict[str, object], label: object) -> bool:
        if _payload_has_hard_non_mature_status(payload):
            return False
        snapshot = evidence_maturity_snapshot(payload, self.maturity_policy)
        decision = snapshot.get("mature_enough")
        if decision is not None:
            return bool(decision)
        if _payload_has_soft_non_mature_label(payload):
            return False
        if self.maturity_policy.get("require_ratio_gate"):
            return False
        return (
            self._evidence_units_from_payload(payload, label)
            >= self.adaptive_policy.result_finding_weight
        )

    def _canonical_variant_peer_map(self, rows: list[sqlite3.Row]) -> dict[str, set[str]]:
        """Map variant aliases from canonical peer findings to peer ids.

        This is a narrow evidence-gating fallback. It lets the trigger count
        evaluator-written result references as formal peer evidence
        only when their variant name uniquely matches a canonical `genN_peer*`
        finding. It does not mutate findings, frontier data, or PI inputs.
        """
        mapping: dict[str, set[str]] = {}
        for row in rows:
            peer_id = self._canonical_peer_id(row["peer_id"])
            if peer_id is None:
                continue
            for alias in self._variant_aliases(row["variant_name"], row["title"]):
                mapping.setdefault(alias, set()).add(peer_id)
        return mapping

    def _canonical_peer_id(self, value: object) -> str | None:
        match = re.fullmatch(rf"gen{self.gen_id}_peer(\d+)", str(value or "").strip())
        if match is None:
            return None
        candidate = f"gen{self.gen_id}_peer{int(match.group(1))}"
        if self.started_peer_ids is not None:
            return candidate if candidate in self.started_peer_ids else None
        peer_index = int(match.group(1))
        if self.cohort_size > 0 and peer_index >= self.cohort_size:
            return None
        return candidate

    def _infer_auto_result_peer(
        self,
        row: sqlite3.Row,
        variant_to_peers: dict[str, set[str]],
    ) -> str | None:
        """Infer the owning peer for an evaluator result reference if unique."""
        prefix = f"gen{self.gen_id}_peer"
        for alias in self._variant_aliases(row["variant_name"], row["title"]):
            if alias.startswith(prefix):
                suffix = alias[len(prefix) :]
                digits = ""
                for char in suffix:
                    if char.isdigit():
                        digits += char
                    else:
                        break
                if digits:
                    inferred = self._canonical_peer_id(f"{prefix}{digits}")
                    if inferred is not None:
                        return inferred
            peers = variant_to_peers.get(alias)
            if peers is not None and len(peers) == 1:
                inferred = self._canonical_peer_id(next(iter(peers)))
                if inferred is not None:
                    return inferred
        return None

    @staticmethod
    def _variant_aliases(*values: object) -> set[str]:
        aliases: set[str] = set()
        for value in values:
            text = str(value or "").strip().lower()
            if not text:
                continue
            candidates = {
                text,
                text.split(":", 1)[0],
            }
            for candidate in candidates:
                cleaned = candidate.strip().replace("-", "_").replace(" ", "_")
                cleaned = cleaned.strip("_")
                if not cleaned or cleaned in {"test", "baseline"}:
                    continue
                aliases.add(cleaned)
        return aliases

    def _evidence_units_from_payload(self, payload: dict[str, object], label: object) -> float:
        weights = self.adaptive_policy.evidence_weights or {}
        stage = str(
            payload.get("tier_reached")
            or payload.get("completed_tier")
            or payload.get("tier")
            or payload.get("evidence_stage")
            or payload.get("stage")
            or ""
        ).strip()
        units = weights.get(stage, 0.0)
        stage_snapshot = evidence_maturity_snapshot(
            {
                key: payload[key]
                for key in (
                    "evidence_stage",
                    "eval_stage",
                    "stage",
                    "tier",
                    "tier_reached",
                    "completed_tier",
                    "candidate_tier",
                )
                if key in payload
            },
            {**self.maturity_policy, "require_ratio_gate": False},
        )
        weighted_preliminary = (
            stage_snapshot.get("mature_enough") is False
            and stage_snapshot.get("maturity_basis") == "task_configured_stage"
        )
        if (
            _payload_has_hard_non_mature_status(payload)
            and not _payload_has_validation_only_marker(payload)
            and not weighted_preliminary
        ):
            return 0.0
        if not self._payload_is_explicitly_non_durable(payload) and (
            evidence_maturity_snapshot(payload, self.maturity_policy).get("mature_enough") is True
            or _payload_has_completion_marker(payload)
        ):
            units = max(units, self.adaptive_policy.result_finding_weight)
        label_text = str(label or "").lower()
        is_smoke = _truthy(payload.get("is_smoke_eval")) or "smoke" in label_text
        if is_smoke and units > 0:
            units = min(units, self.adaptive_policy.smoke_weight)
        return max(0.0, float(units))

    def _protected_entry_belongs_to_generation(self, entry: dict[str, object]) -> bool:
        peer_id = str(entry.get("peer_id") or "")
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.protected_pids import (
                _generation_id_from_peer_id,
            )

            if _generation_id_from_peer_id(peer_id) == self.gen_id:
                return True
        except Exception:
            pass
        return peer_id.startswith(f"gen{self.gen_id}_peer") or peer_id.startswith(
            f"gen{self.gen_id}/peer"
        )

    def _active_protected_pid_count(self) -> int:
        try:
            from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

            return sum(
                1
                for entry in protected_pids.list_active_jobs(run_dir=self.run_dir)
                if self._protected_entry_belongs_to_generation(
                    {"peer_id": getattr(entry, "peer_id", "")}
                )
            )
        except Exception as exc:  # noqa: BLE001 - trigger telemetry is advisory.
            logger.debug("synthesis_trigger: protected process scan failed: %s", exc)
            return 0

    def evaluate(self) -> TriggerSnapshot:
        """Check trigger conditions ONCE; do not fire (caller fires).

        R1#11 fix: info_density wins ties with safety_cap (cleaner
        diagnostics: prefer the success label when both conditions hold).
        R2#5 fix: invoke optional pre-eval sync callback to absorb any
        pending filesystem→SQLite findings before reading. Callback
        failures are non-fatal — the SQL read just sees stale state.
        """
        if self._pre_eval_sync_callback is not None:
            try:
                self._pre_eval_sync_callback()
            except Exception as e:
                logger.debug(
                    "synthesis_trigger: pre_eval_sync_callback raised (non-fatal): %s",
                    e,
                )
        now = time.time()
        minutes_since_start = (now - self.gen_start_time) / 60.0
        findings_count, contributing_peers = self._query_gen_state()
        evidence_units, formal_result_peers = self._query_adaptive_state()
        mature_result_peers = self._query_mature_state()
        mature_result_count = self.mature_result_count(synchronize=False)
        active_protected_pids = self._active_protected_pid_count()
        active_generation_work = 0
        active_generation_work_known = False
        if self._cohort_active_peers_callback is not None:
            try:
                active_generation_work = max(0, int(self._cohort_active_peers_callback()))
                active_generation_work_known = True
            except Exception as e:  # noqa: BLE001 - non-fatal.
                logger.debug(
                    "synthesis_trigger: cohort_active_peers_callback raised during audit: %s",
                    e,
                )
                active_generation_work = 0
        required_mature = self.required_mature_result_peers

        fixed_info_density_ready = (
            findings_count >= self.min_findings
            and minutes_since_start >= self.min_interval_minutes
            and contributing_peers >= self.min_contributing_peers
        )

        adaptive_ready = False
        if self.adaptive_policy.enabled:
            enough_evidence = (
                self.adaptive_policy.min_evidence_units <= 0
                or evidence_units >= self.adaptive_policy.min_evidence_units
            )
            enough_formal_peers = (
                self.adaptive_policy.min_formal_result_peers <= 0
                or formal_result_peers >= self.adaptive_policy.min_formal_result_peers
            )
            adaptive_ready = (
                enough_evidence
                and enough_formal_peers
                and minutes_since_start >= self.min_interval_minutes
                and contributing_peers >= self.min_contributing_peers
            )
        mature_quorum_ready = (
            required_mature > 0
            and mature_result_peers >= required_mature
            and minutes_since_start >= self.min_interval_minutes
        )

        # A configured mature quorum defines normal completion. Fixed/adaptive
        # evidence can begin a close assessment, but cannot bypass it. The
        # optional safety cap remains the only unconditional timer escape hatch.
        normal_completion_ready = (
            (
                mature_quorum_ready
                if required_mature > 0
                else (fixed_info_density_ready or adaptive_ready)
            )
            and active_generation_work <= 0
            and active_protected_pids <= 0
        )
        if (
            self.max_interval_minutes is not None
            and minutes_since_start >= self.max_interval_minutes
            and not normal_completion_ready
        ):
            return TriggerSnapshot(
                fired=True,
                reason="safety_cap",
                findings_count=findings_count,
                minutes_since_start=minutes_since_start,
                contributing_peers=contributing_peers,
                evidence_units=evidence_units,
                formal_result_peers=formal_result_peers,
                mature_result_peers=mature_result_peers,
                mature_result_count=mature_result_count,
                required_mature_result_peers=required_mature,
                active_protected_pids=active_protected_pids,
                active_generation_work=active_generation_work,
                assessment_started=self._assessment_started,
            )

        # Fixed/adaptive evidence starts a close assessment. With a mature
        # quorum configured, it is deliberately not itself a success path.
        assessment_ready = fixed_info_density_ready or adaptive_ready or mature_quorum_ready
        if not self.closing or active_protected_pids > 0 or active_generation_work <= 0:
            self._drain_started_at = None
        elif assessment_ready:
            if self._drain_started_at is None:
                self._drain_started_at = now
            drain_grace_seconds = self.adaptive_policy.drain_grace_minutes * 60.0
            if now - self._drain_started_at >= drain_grace_seconds:
                return TriggerSnapshot(
                    fired=True,
                    reason="closing_agent_drain_deadline",
                    findings_count=findings_count,
                    minutes_since_start=minutes_since_start,
                    contributing_peers=contributing_peers,
                    evidence_units=evidence_units,
                    formal_result_peers=formal_result_peers,
                    mature_result_peers=mature_result_peers,
                    mature_result_count=mature_result_count,
                    required_mature_result_peers=required_mature,
                    active_protected_pids=active_protected_pids,
                    active_generation_work=active_generation_work,
                    assessment_started=self._assessment_started,
                )
        if assessment_ready:
            reason = (
                "mature_quorum"
                if mature_quorum_ready
                else "adaptive_evidence"
                if adaptive_ready
                else "info_density"
            )
            ready_snapshot = TriggerSnapshot(
                fired=False,
                reason=reason,
                findings_count=findings_count,
                minutes_since_start=minutes_since_start,
                contributing_peers=contributing_peers,
                evidence_units=evidence_units,
                formal_result_peers=formal_result_peers,
                mature_result_peers=mature_result_peers,
                mature_result_count=mature_result_count,
                required_mature_result_peers=required_mature,
                active_protected_pids=active_protected_pids,
                active_generation_work=active_generation_work,
                assessment_started=True,
            )
            if required_mature > 0 and not mature_quorum_ready:
                if (
                    active_generation_work_known
                    and active_generation_work <= 0
                    and active_protected_pids <= 0
                    and (now - self.gen_start_time) >= self._cohort_drain_warmup_seconds
                ):
                    ready_snapshot.fired = True
                    ready_snapshot.reason = "cohort_drained_insufficient_mature"
                    return ready_snapshot
                assessment_started = self.begin_assessment(ready_snapshot)
                if assessment_started is None:
                    self._assessment_started = False
                    ready_snapshot.assessment_started = False
                    ready_snapshot.reason = "assessment_fence_retry"
                    return ready_snapshot
                self._assessment_started = assessment_started
                ready_snapshot.assessment_started = assessment_started
                if not assessment_started:
                    if (
                        active_generation_work_known
                        and active_generation_work <= 0
                        and active_protected_pids <= 0
                    ):
                        ready_snapshot.fired = True
                        ready_snapshot.reason = "cohort_drained_insufficient_mature"
                        return ready_snapshot
                    # Legacy execution has no scheduler assessment fence. Keep
                    # its established soft-close/drain behavior instead of
                    # claiming that ordinary admission was stopped.
                    self.begin_closing(ready_snapshot)
                    self._assessment_started = True
                    ready_snapshot.assessment_started = True
                    ready_snapshot.reason = (
                        "draining_active_evals"
                        if active_protected_pids > 0
                        else "assessment_draining"
                    )
                    return ready_snapshot
                ready_snapshot.reason = "assessment_mature_topup"
                return ready_snapshot
            if active_generation_work > 0 or active_protected_pids > 0:
                self.begin_closing(ready_snapshot)
                self._assessment_started = True
                return TriggerSnapshot(
                    fired=False,
                    reason=(
                        "draining_active_evals"
                        if active_protected_pids > 0
                        else "assessment_draining"
                    ),
                    findings_count=findings_count,
                    minutes_since_start=minutes_since_start,
                    contributing_peers=contributing_peers,
                    evidence_units=evidence_units,
                    formal_result_peers=formal_result_peers,
                    mature_result_peers=mature_result_peers,
                    mature_result_count=mature_result_count,
                    required_mature_result_peers=required_mature,
                    active_protected_pids=active_protected_pids,
                    active_generation_work=active_generation_work,
                    assessment_started=True,
                )
            ready_snapshot.fired = True
            return ready_snapshot
        self._drain_started_at = None

        # Cohort drained: every peer the orchestrator launched is
        # terminal. Without this exit, a cohort where peers crash
        # early sits idle until ``safety_cap`` expires (default 240
        # min) — see #148.  The callback is opt-in (legacy callers
        # without peer-state visibility leave it None and keep the
        # info_density / safety_cap-only behaviour).
        if (
            active_generation_work_known
            and active_generation_work <= 0
            and (now - self.gen_start_time) >= self._cohort_drain_warmup_seconds
        ):
            return TriggerSnapshot(
                fired=True,
                reason=(
                    "cohort_drained_insufficient_mature"
                    if required_mature > 0 and mature_result_peers < required_mature
                    else "cohort_drained"
                ),
                findings_count=findings_count,
                minutes_since_start=minutes_since_start,
                contributing_peers=contributing_peers,
                evidence_units=evidence_units,
                formal_result_peers=formal_result_peers,
                mature_result_peers=mature_result_peers,
                mature_result_count=mature_result_count,
                required_mature_result_peers=required_mature,
                active_protected_pids=active_protected_pids,
                active_generation_work=active_generation_work,
                assessment_started=self._assessment_started,
            )

        return TriggerSnapshot(
            fired=False,
            reason="not_yet",
            findings_count=findings_count,
            minutes_since_start=minutes_since_start,
            contributing_peers=contributing_peers,
            evidence_units=evidence_units,
            formal_result_peers=formal_result_peers,
            mature_result_peers=mature_result_peers,
            mature_result_count=mature_result_count,
            required_mature_result_peers=required_mature,
            active_protected_pids=active_protected_pids,
            active_generation_work=active_generation_work,
            assessment_started=self._assessment_started,
        )

    async def evaluate_async(self) -> TriggerSnapshot:
        """R1#8 fix: evaluate() in a thread to avoid blocking event loop."""
        return await asyncio.to_thread(self.evaluate)

    def _write_signal_atomic(self, path: Path, payload: str) -> None:
        """Atomically write a sentinel and fsync the parent directory."""
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(path)
        try:
            dir_fd = os.open(str(self.gen_dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass  # best effort; some FS don't support dir fsync

    def begin_closing(self, snapshot: TriggerSnapshot) -> None:
        """Write CLOSING_SIGNAL once close criteria are met but evals drain.

        Unlike STOP_SIGNAL, this is not an interrupt signal for an in-flight
        agent runtime. It only prevents peers from opening another session while
        protected subprocesses finish and the trigger waits to fire.
        """
        if self._closing:
            return
        # Freeze the canonical launch queue before publishing the filesystem
        # signal.  This closes the submit-vs-sentinel race while existing
        # scheduler-owned process groups continue to drain normally.
        try:
            from .experiment_scheduler_client import freeze_generation

            freeze_generation(self.gen_id, snapshot.reason)
        except Exception as exc:  # noqa: BLE001 - the sentinel remains the legacy fallback.
            logger.warning("synthesis_trigger: central scheduler freeze failed: %s", exc)
        payload = (
            f"trigger_reason={snapshot.reason}\n"
            f"gen_id={self.gen_id}\n"
            f"findings_count={snapshot.findings_count}\n"
            f"minutes_since_start={snapshot.minutes_since_start:.2f}\n"
            f"contributing_peers={snapshot.contributing_peers}\n"
            f"evidence_units={snapshot.evidence_units:.2f}\n"
            f"result_evidence_peers={snapshot.formal_result_peers}\n"
            f"formal_result_peers={snapshot.formal_result_peers}\n"
            f"mature_result_peers={snapshot.mature_result_peers}\n"
            f"mature_result_count={snapshot.mature_result_count}\n"
            f"required_mature_result_peers={snapshot.required_mature_result_peers}\n"
            f"active_protected_pids={snapshot.active_protected_pids}\n"
            f"active_generation_work={snapshot.active_generation_work}\n"
            f"assessment_started={int(snapshot.assessment_started)}\n"
            f"closing_started_at={time.time():.0f}\n"
        )
        self._write_signal_atomic(self._closing_signal_path, payload)
        self._closing = True
        if snapshot.active_protected_pids <= 0 and snapshot.active_generation_work > 0:
            self._drain_started_at = time.time()
        logger.info(
            "synthesis_trigger: CLOSING for gen %d (reason=%s, findings=%d, "
            "peers=%d, evidence=%.2f, result_evidence_peers=%d, mature_results=%d, mature_peers=%d/%d, "
            "active_work=%d, active_evals=%d, "
            "elapsed=%.1f min). Closing signal written to %s",
            self.gen_id,
            snapshot.reason,
            snapshot.findings_count,
            snapshot.contributing_peers,
            snapshot.evidence_units,
            snapshot.formal_result_peers,
            snapshot.mature_result_count,
            snapshot.mature_result_peers,
            snapshot.required_mature_result_peers,
            snapshot.active_generation_work,
            snapshot.active_protected_pids,
            snapshot.minutes_since_start,
            self._closing_signal_path,
        )

    def begin_assessment(self, snapshot: TriggerSnapshot) -> bool | None:
        """Fence ordinary starts while mature evidence debt remains."""

        if self._assessment_started:
            return True
        try:
            from .experiment_scheduler_client import begin_assessment

            if not begin_assessment(self.gen_id, snapshot.reason):
                return False
        except Exception as exc:  # noqa: BLE001 - legacy tasks still reach the safety cap.
            logger.warning("synthesis_trigger: central assessment fence failed: %s", exc)
            return None
        logger.info(
            "synthesis_trigger: ASSESSMENT gen %d mature=%d/%d; "
            "ordinary admission stopped, mature top-ups remain eligible",
            self.gen_id,
            snapshot.mature_result_count,
            snapshot.required_mature_result_peers,
        )
        return True

    def fire(self, snapshot: TriggerSnapshot) -> None:
        """Write the STOP_SIGNAL sentinel file (atomic rename + fsync).

        R1#12 fix: fsync the parent directory after rename so NFS / shared
        FS makes the file visible to peer watchers immediately.
        R1#19 fix: explicit utf-8 encoding on write_text.
        """
        with self._signal_lock:
            if self._fired:
                return
            # Every close path, including safety-cap and wall-time fallbacks,
            # fences the canonical launch queue before STOP_SIGNAL publication.
            try:
                from .experiment_scheduler_client import freeze_generation

                freeze_generation(self.gen_id, snapshot.reason)
            except Exception as exc:  # noqa: BLE001 - filesystem sentinel remains fallback.
                logger.warning("synthesis_trigger: central scheduler freeze failed: %s", exc)
            payload = (
                f"trigger_reason={snapshot.reason}\n"
                f"gen_id={self.gen_id}\n"
                f"findings_count={snapshot.findings_count}\n"
                f"minutes_since_start={snapshot.minutes_since_start:.2f}\n"
                f"contributing_peers={snapshot.contributing_peers}\n"
                f"evidence_units={snapshot.evidence_units:.2f}\n"
                f"result_evidence_peers={snapshot.formal_result_peers}\n"
                f"formal_result_peers={snapshot.formal_result_peers}\n"
                f"mature_result_peers={snapshot.mature_result_peers}\n"
                f"mature_result_count={snapshot.mature_result_count}\n"
                f"required_mature_result_peers={snapshot.required_mature_result_peers}\n"
                f"active_protected_pids={snapshot.active_protected_pids}\n"
                f"active_generation_work={snapshot.active_generation_work}\n"
                f"assessment_started={int(snapshot.assessment_started)}\n"
                f"fired_at={time.time():.0f}\n"
            )
            self._write_signal_atomic(self._stop_signal_path, payload)
            self._fired = True
        logger.info(
            "synthesis_trigger: FIRED for gen %d (reason=%s, findings=%d, "
            "peers=%d, evidence=%.2f, result_evidence_peers=%d, mature_results=%d, mature_peers=%d/%d, "
            "elapsed=%.1f min). "
            "Stop signal written to %s",
            self.gen_id,
            snapshot.reason,
            snapshot.findings_count,
            snapshot.contributing_peers,
            snapshot.evidence_units,
            snapshot.formal_result_peers,
            snapshot.mature_result_count,
            snapshot.mature_result_peers,
            snapshot.required_mature_result_peers,
            snapshot.minutes_since_start,
            self._stop_signal_path,
        )

    def fire_deadline(self, reason: str = "generation_wall_timeout") -> None:
        """Publish the configured hard generation deadline without querying shared state.

        This path is safe for the independent watchdog thread.  It deliberately
        avoids SQLite, filesystem scans, and asyncio callbacks that may be part
        of the stalled control path.  ``-1`` marks live-work telemetry as
        unavailable rather than incorrectly reporting that no work exists.
        """

        self.fire(
            TriggerSnapshot(
                fired=True,
                reason=reason,
                findings_count=-1,
                minutes_since_start=max(0.0, (time.time() - self.gen_start_time) / 60.0),
                contributing_peers=-1,
                evidence_units=-1.0,
                formal_result_peers=-1,
                mature_result_peers=-1,
                mature_result_count=-1,
                required_mature_result_peers=self.required_mature_result_peers,
                active_protected_pids=-1,
                active_generation_work=-1,
                assessment_started=self._assessment_started,
            )
        )

    def write_postgen_marker(self, snapshot: TriggerSnapshot) -> None:
        """R1#6 fix: when peers finish before trigger fires, write a SEPARATE
        sentinel (STOP_SIGNAL_POSTGEN) to record termination context without
        misleadingly claiming the trigger fired.
        """
        if self._fired:
            return
        marker_path = self.gen_dir / STOP_SIGNAL_POSTGEN_FILENAME
        try:
            payload = (
                f"trigger_reason=peers_finished_before_trigger\n"
                f"snapshot_reason={snapshot.reason}\n"
                f"gen_id={self.gen_id}\n"
                f"findings_count={snapshot.findings_count}\n"
                f"minutes_since_start={snapshot.minutes_since_start:.2f}\n"
                f"contributing_peers={snapshot.contributing_peers}\n"
                f"evidence_units={snapshot.evidence_units:.2f}\n"
                f"result_evidence_peers={snapshot.formal_result_peers}\n"
                f"formal_result_peers={snapshot.formal_result_peers}\n"
                f"mature_result_peers={snapshot.mature_result_peers}\n"
                f"mature_result_count={snapshot.mature_result_count}\n"
                f"required_mature_result_peers={snapshot.required_mature_result_peers}\n"
                f"active_protected_pids={snapshot.active_protected_pids}\n"
                f"active_generation_work={snapshot.active_generation_work}\n"
                f"assessment_started={int(snapshot.assessment_started)}\n"
                f"written_at={time.time():.0f}\n"
            )
            marker_path.write_text(payload, encoding="utf-8")
        except OSError as e:
            logger.debug("postgen marker write failed: %s", e)

    @property
    def stop_signal_path(self) -> Path:
        return self._stop_signal_path

    @property
    def closing_signal_path(self) -> Path:
        return self._closing_signal_path

    @property
    def fired(self) -> bool:
        return self._fired

    @property
    def closing(self) -> bool:
        if self._closing:
            return True
        try:
            return self._closing_signal_path.exists()
        except (OSError, ValueError):
            return False

    @property
    def assessment_started(self) -> bool:
        return self._assessment_started

    def _watch_paths(self) -> list[Path]:
        return [
            self.run_dir / "shared_findings",
            self.run_dir / "protected_pids",
            self.db_path,
        ]

    def _is_trigger_event(self, path: Path) -> bool:
        path = Path(path)
        db_name = self.db_path.name
        if path.name in {db_name, f"{db_name}-wal", f"{db_name}-shm"}:
            return True
        return path.suffix.lower() == ".json" and not path.name.endswith(".tmp")

    def _seconds_until_next_timer_check(self, snap: TriggerSnapshot) -> float:
        if snap.reason in {"draining_active_evals", "assessment_draining"}:
            return max(1.0, min(60.0, self.poll_interval_seconds))
        elapsed_seconds = max(0.0, time.time() - self.gen_start_time)
        min_interval_seconds = self.min_interval_minutes * 60.0
        timer_seconds = (
            self.poll_interval_seconds
            if self.max_interval_minutes is None
            else max(1.0, self.max_interval_minutes * 60.0 - elapsed_seconds)
        )
        has_fixed_density_except_time = (
            snap.findings_count >= self.min_findings
            and snap.contributing_peers >= self.min_contributing_peers
        )
        has_adaptive_density_except_time = False
        if self.adaptive_policy.enabled:
            enough_evidence = (
                self.adaptive_policy.min_evidence_units <= 0
                or snap.evidence_units >= self.adaptive_policy.min_evidence_units
            )
            enough_formal_peers = (
                self.adaptive_policy.min_formal_result_peers <= 0
                or snap.formal_result_peers >= self.adaptive_policy.min_formal_result_peers
            )
            has_adaptive_density_except_time = (
                enough_evidence
                and enough_formal_peers
                and snap.contributing_peers >= self.min_contributing_peers
            )
        has_mature_quorum_except_time = (
            snap.required_mature_result_peers > 0
            and snap.mature_result_peers >= snap.required_mature_result_peers
        )
        has_density_except_time = (
            has_fixed_density_except_time
            or has_adaptive_density_except_time
            or has_mature_quorum_except_time
        )
        if has_density_except_time and elapsed_seconds < min_interval_seconds:
            timer_seconds = min(timer_seconds, min_interval_seconds - elapsed_seconds)
        return max(1.0, min(timer_seconds, self.poll_interval_seconds))

    async def _wait_for_change(
        self, wait_seconds: float, abort_requested: Callable[[], bool]
    ) -> FileEventWaitResult:
        filesystem_wait = wait_for_filesystem_event(
            self._watch_paths(),
            timeout_seconds=wait_seconds,
            stop_check=abort_requested,
            recursive=True,
            max_dirs=512,
            fallback_interval_seconds=wait_seconds,
            stop_check_interval_seconds=30.0,
            event_filter=self._is_trigger_event,
        )
        if self._cohort_completed_event is None:
            return await filesystem_wait
        # Peer completion is independent of filesystem activity. Wake this
        # evaluation loop rather than evaluating shared trigger state concurrently.
        filesystem_task = asyncio.create_task(filesystem_wait)
        completed_task = asyncio.create_task(self._cohort_completed_event.wait())
        try:
            done, _ = await asyncio.wait(
                {filesystem_task, completed_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if completed_task in done:
                self._cohort_completed_event.clear()
            if filesystem_task in done:
                return filesystem_task.result()
            return FileEventWaitResult(reason="cohort_completed", elapsed_seconds=0.0)
        finally:
            for task in (filesystem_task, completed_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(filesystem_task, completed_task, return_exceptions=True)

    async def wait_until_fire(
        self,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> TriggerSnapshot:
        """Async loop: wait for relevant events until the trigger fires.

        If `abort_event` is supplied and gets set, returns the most recent
        non-fired snapshot without firing (allows external cancellation).

        R1#10 fix: the loop catches all exceptions per-iteration so a
        single transient FS / SQLite hiccup cannot kill the trigger task.
        """
        n = 0
        last_snap = TriggerSnapshot(
            fired=False,
            reason="not_yet",
            findings_count=0,
            minutes_since_start=0.0,
            contributing_peers=0,
        )
        while True:
            n += 1
            if abort_event is not None and abort_event.is_set():
                logger.info(
                    "synthesis_trigger: wait_until_fire aborted via event "
                    "(gen %d, elapsed %.1f min)",
                    self.gen_id,
                    (time.time() - self.gen_start_time) / 60.0,
                )
                return last_snap
            try:
                snap = await self.evaluate_async()
                last_snap = snap
                if snap.fired:
                    self.fire(snap)
                    return snap
                if n == 1 or n % 4 == 0:
                    cap_display = (
                        "unbounded"
                        if self.max_interval_minutes is None
                        else f"{self.max_interval_minutes:.1f}"
                    )
                    if self.adaptive_policy.enabled:
                        logger.info(
                            "synthesis_trigger: gen %d waiting — "
                            "findings=%d/%d, peers=%d/%d, "
                            "evidence=%.2f/%.2f, result_evidence_peers=%d/%d, "
                            "mature_peers=%d/%d, active_evals=%d, active_work=%d, elapsed=%.1f/%.1f min "
                            "(cap %s, reason=%s)",
                            self.gen_id,
                            snap.findings_count,
                            self.min_findings,
                            snap.contributing_peers,
                            self.min_contributing_peers,
                            snap.evidence_units,
                            self.adaptive_policy.min_evidence_units,
                            snap.formal_result_peers,
                            self.adaptive_policy.min_formal_result_peers,
                            snap.mature_result_peers,
                            snap.required_mature_result_peers,
                            snap.active_protected_pids,
                            snap.active_generation_work,
                            snap.minutes_since_start,
                            self.min_interval_minutes,
                            cap_display,
                            snap.reason,
                        )
                    else:
                        logger.info(
                            "synthesis_trigger: gen %d waiting — findings=%d/%d, "
                            "peers=%d/%d, result_evidence_peers=%d, mature_peers=%d/%d, "
                            "active_evals=%d, active_work=%d, elapsed=%.1f/%.1f min (cap %s)",
                            self.gen_id,
                            snap.findings_count,
                            self.min_findings,
                            snap.contributing_peers,
                            self.min_contributing_peers,
                            snap.formal_result_peers,
                            snap.mature_result_peers,
                            snap.required_mature_result_peers,
                            snap.active_protected_pids,
                            snap.active_generation_work,
                            snap.minutes_since_start,
                            self.min_interval_minutes,
                            cap_display,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # R1#10 fix: log + continue; do NOT let the trigger die
                logger.warning(
                    "synthesis_trigger: event iteration failed (%s: %s); continuing",
                    type(e).__name__,
                    e,
                )

            wait_seconds = self._seconds_until_next_timer_check(last_snap)

            def _abort_requested() -> bool:
                return bool(abort_event is not None and abort_event.is_set())

            wait_result = await self._wait_for_change(wait_seconds, _abort_requested)
            if wait_result.reason == "stop":
                logger.info(
                    "synthesis_trigger: wait aborted via event (gen %d, elapsed %.1f min)",
                    self.gen_id,
                    (time.time() - self.gen_start_time) / 60.0,
                )
                return last_snap

    async def poll_until_fire(
        self,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> TriggerSnapshot:
        """Compatibility wrapper for older callers.

        The implementation is event-driven; the legacy method name is kept so
        direct tests or old scripts do not break.
        """
        return await self.wait_until_fire(abort_event=abort_event)


def stop_signal_present(gen_dir: Path) -> bool:
    """Peer-side check: does the STOP_SIGNAL sentinel exist for this gen?

    Peers call this from `StopChecker.check()` to detect orchestrator-initiated
    early termination (synthesis trigger fired). Wrapped in a separate function
    so the agent.py module doesn't need to import the whole SynthesisTrigger.
    """
    try:
        return (Path(gen_dir) / STOP_SIGNAL_FILENAME).exists()
    except (OSError, ValueError):
        return False


def closing_signal_present(gen_dir: Path) -> bool:
    """Peer-side check: does the CLOSING_SIGNAL drain sentinel exist?"""
    try:
        return (Path(gen_dir) / CLOSING_SIGNAL_FILENAME).exists()
    except (OSError, ValueError):
        return False
