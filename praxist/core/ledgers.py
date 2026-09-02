"""C5 append-only ledger helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from praxist.core.budget import ALLOWED_BUDGET_UNITS
from praxist.core.protocol import BudgetDecision, BudgetRequest
from praxist.core.storage import append_jsonl, read_jsonl, utc_now


class BudgetLedger:
    """Append-only BudgetLedger over ``budget_ledger.jsonl``."""

    schema_version = "praxist.budget_ledger.v1"

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.path = self.run_dir / "budget_ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append_request(
        self,
        request: BudgetRequest,
        *,
        actor_ref: str,
        stage_id: str,
        action_type: str,
        reason: str,
        source_event_ids: list[str] | None = None,
        artifact_refs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        _validate_budget_amounts(request.requested, f"request {request.request_id}")
        record = self._base_record(
            kind="request",
            request_id=request.request_id,
            grant_id=None,
            actor_ref=actor_ref,
            stage_id=stage_id,
            action_type=action_type,
            reason=reason,
            source_event_ids=source_event_ids or [],
            artifact_refs=artifact_refs or [],
        )
        record.update(
            {
                "request_record": asdict(request),
                "requested_budget": dict(request.requested),
                "decision": None,
                "granted_budget": None,
                "actual_usage": None,
            }
        )
        append_jsonl(self.path, record)
        return record

    def append_decision(
        self,
        request: BudgetRequest,
        decision: BudgetDecision,
        *,
        actor_ref: str,
        stage_id: str,
        action_type: str,
        reason: str,
        source_event_ids: list[str] | None = None,
        artifact_refs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        _validate_budget_amounts(request.requested, f"request {request.request_id}")
        grant_id = decision.grant.grant_id if decision.grant else None
        if grant_id and grant_id in self.active_grants():
            raise ValueError(f"Duplicate budget grant: {grant_id}")
        if decision.grant:
            _validate_budget_amounts(decision.grant.approved, f"grant {grant_id}")
        record = self._base_record(
            kind="decision",
            request_id=request.request_id,
            grant_id=grant_id,
            actor_ref=actor_ref,
            stage_id=stage_id,
            action_type=action_type,
            reason=reason,
            source_event_ids=source_event_ids or [],
            artifact_refs=artifact_refs or [],
        )
        record.update(
            {
                "request_record": asdict(request),
                "requested_budget": dict(request.requested),
                "decision": decision.decision,
                "granted_budget": decision.grant.approved if decision.grant else None,
                "actual_usage": None,
                "decision_record": decision.to_dict(),
            }
        )
        append_jsonl(self.path, record)
        return record

    def append_usage(
        self,
        *,
        request_id: str | None,
        grant_id: str | None,
        actor_ref: str,
        stage_id: str,
        action_type: str,
        actual_usage: dict[str, float],
        reason: str,
        source_event_ids: list[str] | None = None,
        artifact_refs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        overrun_details: list[dict[str, Any]] = []
        if grant_id:
            overrun_details = self._require_usage_within_grant(grant_id, actual_usage)
        record = self._base_record(
            kind="usage",
            request_id=request_id,
            grant_id=grant_id,
            actor_ref=actor_ref,
            stage_id=stage_id,
            action_type=action_type,
            reason=reason,
            source_event_ids=source_event_ids or [],
            artifact_refs=artifact_refs or [],
        )
        record.update(
            {
                "requested_budget": {},
                "decision": None,
                "granted_budget": None,
                "actual_usage": dict(actual_usage),
            }
        )
        if overrun_details:
            record["budget_overrun"] = True
            record["overrun_units"] = sorted({str(detail["unit"]) for detail in overrun_details})
            record["overrun_details"] = overrun_details
        append_jsonl(self.path, record)
        return record

    def append_usage_unknown(
        self,
        *,
        request_id: str | None,
        grant_id: str,
        actor_ref: str,
        stage_id: str,
        action_type: str,
        unknown_units: list[str],
        reason: str,
        source_event_ids: list[str] | None = None,
        artifact_refs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        grant = self.require_active_grant(grant_id)
        approved = grant.get("granted_budget") or {}
        if not isinstance(approved, dict):
            raise ValueError(f"Budget grant has invalid approved budget: {grant_id}")
        _validate_budget_amounts(approved, f"grant {grant_id}")
        normalized_units = sorted({str(unit) for unit in unknown_units if str(unit)})
        for unit in normalized_units:
            if unit not in approved:
                raise ValueError(
                    f"Budget usage_unknown unit not approved by grant {grant_id}: {unit}"
                )
        record = self._base_record(
            kind="usage_unknown",
            request_id=request_id,
            grant_id=grant_id,
            actor_ref=actor_ref,
            stage_id=stage_id,
            action_type=action_type,
            reason=reason,
            source_event_ids=source_event_ids or [],
            artifact_refs=artifact_refs or [],
        )
        record.update(
            {
                "requested_budget": {},
                "decision": None,
                "granted_budget": None,
                "actual_usage": None,
                "unknown_units": normalized_units,
            }
        )
        append_jsonl(self.path, record)
        return record

    def records(self) -> list[dict[str, Any]]:
        records, errors = read_jsonl(self.path)
        if errors:
            raise ValueError(f"BudgetLedger read errors: {json.dumps(errors)}")
        return records

    def active_grants(self) -> dict[str, dict[str, Any]]:
        grants: dict[str, dict[str, Any]] = {}
        for record in self.records():
            if record.get("kind") != "decision":
                continue
            grant_id = record.get("grant_id")
            if isinstance(grant_id, str) and record.get("granted_budget") is not None:
                grants[grant_id] = record
        return grants

    def require_active_grant(self, grant_id: str) -> dict[str, Any]:
        grants = self.active_grants()
        if grant_id not in grants:
            raise ValueError(f"Budget grant not found: {grant_id}")
        return grants[grant_id]

    def get_exhausted_units(self, grant_id: str) -> list[str]:
        grant = self.require_active_grant(grant_id)
        approved = grant.get("granted_budget") or {}
        if not isinstance(approved, dict):
            raise ValueError(f"Budget grant has invalid approved budget: {grant_id}")

        totals = self._usage_totals_by_grant().get(grant_id, {})
        exhausted_units = []
        for unit, raw_amount in approved.items():
            try:
                allowed = float(raw_amount)
                used = float(totals.get(unit, 0.0))
                if used >= allowed and allowed > 0:
                    exhausted_units.append(str(unit))
            except (TypeError, ValueError):
                continue
        return exhausted_units

    def _require_usage_within_grant(
        self, grant_id: str, actual_usage: dict[str, float]
    ) -> list[dict[str, Any]]:
        grant = self.require_active_grant(grant_id)
        approved = grant.get("granted_budget") or {}
        if not isinstance(approved, dict):
            raise ValueError(f"Budget grant has invalid approved budget: {grant_id}")
        _validate_budget_amounts(approved, f"grant {grant_id}")
        totals = self._usage_totals_by_grant().get(grant_id, {})
        overrun_details: list[dict[str, Any]] = []
        for unit, raw_amount in actual_usage.items():
            amount = float(raw_amount)
            if not math.isfinite(amount) or amount < 0:
                raise ValueError(
                    f"Budget usage must be finite and non-negative for {unit}: {raw_amount}"
                )
            if unit not in approved:
                raise ValueError(f"Budget usage unit not approved by grant {grant_id}: {unit}")
            approved_amount = float(approved[unit])
            new_total = float(totals.get(unit, 0.0)) + amount
            if new_total > approved_amount:
                overrun_details.append(
                    {
                        "unit": unit,
                        "approved": approved_amount,
                        "previous_total": float(totals.get(unit, 0.0)),
                        "recorded_amount": amount,
                        "new_total": new_total,
                    }
                )
        return overrun_details

    def _usage_totals_by_grant(self) -> dict[str, dict[str, float]]:
        totals: dict[str, dict[str, float]] = {}
        for record in self.records():
            if record.get("kind") != "usage":
                continue
            grant_id = record.get("grant_id")
            actual_usage = record.get("actual_usage") or {}
            if not isinstance(grant_id, str) or not isinstance(actual_usage, dict):
                continue
            grant_totals = totals.setdefault(grant_id, {})
            for unit, raw_amount in actual_usage.items():
                try:
                    grant_totals[unit] = grant_totals.get(unit, 0.0) + float(raw_amount)
                except (TypeError, ValueError):
                    continue
        return totals

    def _base_record(
        self,
        *,
        kind: str,
        request_id: str | None,
        grant_id: str | None,
        actor_ref: str,
        stage_id: str,
        action_type: str,
        reason: str,
        source_event_ids: list[str],
        artifact_refs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": f"budget_{kind}_{len(self.records()) + 1:06d}",
            "run_id": self.run_id,
            "timestamp": utc_now(),
            "kind": kind,
            "request_id": request_id,
            "grant_id": grant_id,
            "actor_ref": actor_ref,
            "stage_id": stage_id,
            "action_type": action_type,
            "reason": reason,
            "source_event_ids": source_event_ids,
            "artifact_refs": artifact_refs,
        }


def _validate_budget_amounts(values: dict[str, Any], label: str) -> None:
    for unit, raw_value in values.items():
        if unit not in ALLOWED_BUDGET_UNITS:
            raise ValueError(f"Budget {label} has unsupported unit: {unit}")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Budget {label} has non-numeric amount for {unit}: {raw_value}"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Budget {label} has invalid amount for {unit}: {raw_value}")
