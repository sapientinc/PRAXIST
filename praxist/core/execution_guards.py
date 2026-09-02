"""Budget and observability helpers for execution resource guards."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praxist.core.ledgers import BudgetLedger
from praxist.core.run_config import RunConfig
from praxist.core.trajectory import TrajectoryWriter


class ResourceBudgetError(RuntimeError):
    """Raised before a high-cost action starts without an approved grant."""


@dataclass(frozen=True)
class BudgetedActionReport:
    """Lifecycle report for one budget-guarded runtime, tool, GPU, or evaluation action."""

    recorded: bool
    usage_record_id: str | None = None
    unknown_record_id: str | None = None
    warning: str | None = None
    actual_usage: dict[str, float] = field(default_factory=dict)
    unknown_units: list[str] = field(default_factory=list)


class BudgetedActionGuard:
    """Record per-action usage without making observability a result killer."""

    def __init__(
        self,
        *,
        run_dir: Path | None,
        run_id: str,
        stage_id: str,
        actor_ref: str,
        action_type: str,
        budget_grant_id: str | None = None,
        request_id: str | None = None,
        require_budget_grant: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir) if run_dir else None
        self.run_id = run_id
        self.stage_id = stage_id
        self.actor_ref = actor_ref
        self.action_type = action_type
        self.budget_grant_id = budget_grant_id
        self.request_id = request_id
        self.require_budget_grant = require_budget_grant
        self.metadata = dict(metadata or {})
        self._started = time.monotonic()

    @classmethod
    def from_env(
        cls,
        *,
        action_type: str,
        actor_ref: str,
        require_budget_grant: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> BudgetedActionGuard:
        run_dir = os.environ.get("PRAXIST_RUN_DIR") or ""
        resolved_run_dir = Path(run_dir) if run_dir else None
        run_id = os.environ.get("PRAXIST_RUN_ID") or (
            resolved_run_dir.name if resolved_run_dir else "legacy_direct"
        )
        return cls(
            run_dir=resolved_run_dir,
            run_id=run_id,
            stage_id=os.environ.get("PRAXIST_STAGE_ID") or "research_loop",
            actor_ref=actor_ref,
            action_type=action_type,
            budget_grant_id=os.environ.get("PRAXIST_BUDGET_GRANT_ID") or None,
            request_id=os.environ.get("PRAXIST_BUDGET_REQUEST_ID") or None,
            require_budget_grant=require_budget_grant,
            metadata=metadata,
        )

    @classmethod
    def from_run_config(
        cls,
        run_config: RunConfig,
        *,
        action_type: str,
        actor_ref: str,
        require_budget_grant: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> BudgetedActionGuard:
        """Build a guard from an explicit :class:`RunConfig` (issue #75 batch 4).

        Sibling of :meth:`from_env` for callers that have already
        constructed a ``RunConfig`` at the CLI boundary. Preserves the
        same defaults (``stage_id="research_loop"``, ``run_id`` derived
        from ``run_dir.name`` when explicit ``run_id`` is empty,
        ``budget_grant_id`` / ``request_id`` ``None`` when empty) so the
        two constructors produce equivalent guards from equivalent
        inputs.
        """
        run_dir = run_config.run_dir
        explicit_run_id = run_config.run_id or ""
        run_id = explicit_run_id or (run_dir.name if run_dir else "legacy_direct")
        return cls(
            run_dir=run_dir,
            run_id=run_id,
            stage_id=run_config.stage_id or "research_loop",
            actor_ref=actor_ref,
            action_type=action_type,
            budget_grant_id=run_config.budget_grant_id or None,
            request_id=run_config.budget_request_id or None,
            require_budget_grant=require_budget_grant,
            metadata=metadata,
        )

    def start(self) -> None:
        if self.require_budget_grant and not self.budget_grant_id:
            self._emit_event(
                "resource.action_denied",
                severity="error",
                payload={"reason": "missing_budget_grant", **self.metadata},
            )
            raise ResourceBudgetError(f"{self.action_type} requires an approved budget grant")
        if self.budget_grant_id and self.run_dir is not None:
            try:
                ledger = BudgetLedger(self.run_dir, self.run_id)
                ledger.require_active_grant(self.budget_grant_id)
                exhausted = ledger.get_exhausted_units(self.budget_grant_id)
                if exhausted:
                    raise ResourceBudgetError(
                        f"Budget exhausted for units: {', '.join(sorted(exhausted))}"
                    )
            except Exception as exc:  # noqa: BLE001 - preflight should report the real reason.
                is_overrun = isinstance(exc, ResourceBudgetError)
                self._emit_event(
                    "resource.action_denied"
                    if self.require_budget_grant or is_overrun
                    else "resource.action_budget_warning",
                    severity="error" if self.require_budget_grant or is_overrun else "warning",
                    payload={
                        "reason": "budget_overrun" if is_overrun else "invalid_budget_grant",
                        "error": str(exc),
                        **self.metadata,
                    },
                )
                if self.require_budget_grant or is_overrun:
                    raise ResourceBudgetError(str(exc)) from exc
        self._emit_event("resource.action_started", payload=self.metadata)

    def finish(
        self,
        *,
        actual_usage: dict[str, float] | None = None,
        expected_units: list[str] | tuple[str, ...] = (),
        status: str = "succeeded",
        reason: str = "action_usage",
        metadata: dict[str, Any] | None = None,
    ) -> BudgetedActionReport:
        elapsed = max(0.0, time.monotonic() - self._started)
        payload = {
            "status": status,
            "elapsed_seconds": elapsed,
            **self.metadata,
            **dict(metadata or {}),
        }
        if self.run_dir is None or not self.budget_grant_id:
            self._emit_event(
                "resource.action_finished", payload={**payload, "budget_recorded": False}
            )
            return BudgetedActionReport(
                recorded=False, warning="missing run_dir or budget_grant_id"
            )

        usage = _finite_nonnegative_usage(actual_usage or {})
        try:
            ledger = BudgetLedger(self.run_dir, self.run_id)
            grant = ledger.require_active_grant(self.budget_grant_id)
            approved = grant.get("granted_budget") or {}
            if not isinstance(approved, dict):
                raise ValueError(f"invalid granted budget for {self.budget_grant_id}")
            usage = {unit: value for unit, value in usage.items() if unit in approved or unit in ("cost_usd", "neurons")}
            if "wall_clock_seconds" in expected_units and "wall_clock_seconds" in approved:
                usage.setdefault("wall_clock_seconds", elapsed)
            unknown_units = [
                unit
                for unit in sorted({str(item) for item in expected_units})
                if unit in approved and unit not in usage and _positive_amount(approved.get(unit))
            ]
            usage_record = None
            unknown_record = None
            if usage:
                usage_record = ledger.append_usage(
                    request_id=self.request_id or grant.get("request_id"),
                    grant_id=self.budget_grant_id,
                    actor_ref=self.actor_ref,
                    stage_id=self.stage_id,
                    action_type=self.action_type,
                    actual_usage=usage,
                    reason=reason,
                )
            if unknown_units:
                unknown_record = ledger.append_usage_unknown(
                    request_id=self.request_id or grant.get("request_id"),
                    grant_id=self.budget_grant_id,
                    actor_ref=self.actor_ref,
                    stage_id=self.stage_id,
                    action_type=self.action_type,
                    unknown_units=unknown_units,
                    reason=f"{reason}_unknown",
                )
            self._emit_event(
                "resource.action_finished",
                payload={
                    **payload,
                    "budget_recorded": bool(usage_record or unknown_record),
                    "actual_usage": usage,
                    "unknown_units": unknown_units,
                },
            )
            return BudgetedActionReport(
                recorded=bool(usage_record or unknown_record),
                usage_record_id=usage_record.get("record_id") if usage_record else None,
                unknown_record_id=unknown_record.get("record_id") if unknown_record else None,
                actual_usage=usage,
                unknown_units=unknown_units,
            )
        except Exception as exc:  # noqa: BLE001 - late accounting failures are warnings.
            self._emit_event(
                "resource.action_usage_warning",
                severity="warning",
                payload={**payload, "error": str(exc)},
            )
            return BudgetedActionReport(recorded=False, warning=str(exc), actual_usage=usage)

    def abort(self, *, reason: str, metadata: dict[str, Any] | None = None) -> None:
        self._emit_event(
            "resource.action_aborted",
            severity="warning",
            payload={"reason": reason, **self.metadata, **dict(metadata or {})},
        )

    def _emit_event(self, kind: str, *, payload: dict[str, Any], severity: str = "info") -> None:
        if self.run_dir is None:
            return
        try:
            TrajectoryWriter(self.run_dir, self.run_id).emit(
                kind,
                severity=severity,
                scope={"stage_id": self.stage_id},
                actor={"type": "resource_guard", "id": self.actor_ref},
                payload={
                    "action_type": self.action_type,
                    "budget_grant_id": self.budget_grant_id or "",
                    **payload,
                },
            )
        except Exception:
            return


def record_budgeted_action_from_env(
    *,
    action_type: str,
    actor_ref: str,
    actual_usage: dict[str, float] | None = None,
    expected_units: list[str] | tuple[str, ...] = (),
    status: str = "succeeded",
    reason: str = "action_usage",
    metadata: dict[str, Any] | None = None,
) -> BudgetedActionReport:
    """Record action usage from PRAXIST_* environment variables used by legacy subprocesses."""
    guard = BudgetedActionGuard.from_env(
        action_type=action_type, actor_ref=actor_ref, metadata=metadata
    )
    guard.start()
    return guard.finish(
        actual_usage=actual_usage,
        expected_units=expected_units,
        status=status,
        reason=reason,
    )


def emit_resource_event_from_env(
    kind: str,
    *,
    action_type: str,
    actor_ref: str,
    payload: dict[str, Any] | None = None,
    severity: str = "info",
) -> None:
    """Emit a resource lifecycle trajectory event from the current scoped execution environment."""
    guard = BudgetedActionGuard.from_env(
        action_type=action_type, actor_ref=actor_ref, metadata=payload
    )
    guard._emit_event(kind, payload=payload or {}, severity=severity)


def record_budgeted_action_from_run_config(
    run_config: RunConfig,
    *,
    action_type: str,
    actor_ref: str,
    actual_usage: dict[str, float] | None = None,
    expected_units: list[str] | tuple[str, ...] = (),
    status: str = "succeeded",
    reason: str = "action_usage",
    metadata: dict[str, Any] | None = None,
) -> BudgetedActionReport:
    """``record_budgeted_action_from_env`` sibling that takes a :class:`RunConfig` (issue #75 batch 4)."""
    guard = BudgetedActionGuard.from_run_config(
        run_config,
        action_type=action_type,
        actor_ref=actor_ref,
        metadata=metadata,
    )
    guard.start()
    return guard.finish(
        actual_usage=actual_usage,
        expected_units=expected_units,
        status=status,
        reason=reason,
    )


def emit_resource_event_from_run_config(
    run_config: RunConfig,
    kind: str,
    *,
    action_type: str,
    actor_ref: str,
    payload: dict[str, Any] | None = None,
    severity: str = "info",
) -> None:
    """``emit_resource_event_from_env`` sibling that takes a :class:`RunConfig` (issue #75 batch 4)."""
    guard = BudgetedActionGuard.from_run_config(
        run_config,
        action_type=action_type,
        actor_ref=actor_ref,
        metadata=payload,
    )
    guard._emit_event(kind, payload=payload or {}, severity=severity)


def gpu_hours_since(started_at: str, *, finished_at: datetime | None = None) -> float | None:
    """Convert elapsed wall-clock time and GPU count into gpu_hours accounting units."""
    try:
        start = datetime.fromisoformat(started_at)
    except (TypeError, ValueError):
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    end = finished_at or datetime.now(UTC)
    elapsed = max(0.0, (end - start).total_seconds())
    return elapsed / 3600.0


def _finite_nonnegative_usage(values: dict[str, float]) -> dict[str, float]:
    usage: dict[str, float] = {}
    for unit, raw in values.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            usage[str(unit)] = value
    return usage


def _positive_amount(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False
