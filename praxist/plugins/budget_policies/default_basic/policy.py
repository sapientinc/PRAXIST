"""Internal dogfood budget policy plugin."""

from __future__ import annotations

from praxist.core.budget import _invalid_budget_units
from praxist.core.protocol import BudgetDecision, BudgetGrant, BudgetRequest


class DefaultBasicBudgetPolicy:
    """Grant strong stage budgets without silent downscope."""

    def decide(self, request: BudgetRequest, *, allow_uncapped: bool = False) -> BudgetDecision:
        """Decide a request, requiring controller authorization for uncapped units.

        Args:
            request: Budget request, whose metadata cannot authorize uncapped use.
            allow_uncapped: Trusted controller permission derived from task config.

        Returns:
            A grant, denial, or review decision without changing requested limits.
        """
        invalid_units = _invalid_budget_units(request.requested)
        if invalid_units:
            return BudgetDecision(
                decision="deny",
                reason_codes=["invalid_budget_value", *invalid_units],
                grant=None,
                review_target=None,
            )
        if request.expected_value.get("impossible"):
            return BudgetDecision(
                decision="deny",
                reason_codes=["impossible_request"],
                grant=None,
                review_target=None,
            )
        requested = request.requested
        uncapped = any(amount is None for amount in requested.values())
        if uncapped and not allow_uncapped:
            return BudgetDecision(
                decision="deny",
                reason_codes=["uncapped_requires_controller_authorization"],
                grant=None,
                review_target=None,
            )
        tokens = requested.get("tokens", 0.0)
        wall_clock = requested.get("wall_clock_seconds", 0.0)
        gpu_hours = requested.get("gpu_hours", 0.0)
        confidence = str(request.expected_value.get("confidence", "weak"))
        if (
            str(request.expected_value.get("confidence", "weak")) == "strong"
            and request.evidence_refs
        ):
            return BudgetDecision(
                decision="grant",
                reason_codes=[
                    "controller_authorized_uncapped_grant"
                    if uncapped
                    else "strong_evidence_internal_grant"
                ],
                grant=BudgetGrant(
                    grant_id=f"grant_{request.request_id}",
                    approved=dict(request.requested),
                    conditions=["record_actual_usage"]
                    if uncapped
                    else ["record_actual_usage", "abort_on_no_signal"],
                    expires_at_generation=None,
                ),
            )
        if (
            tokens is not None
            and wall_clock is not None
            and gpu_hours is not None
            and tokens <= 5_000
            and wall_clock <= 120
            and gpu_hours <= 0
        ):
            return BudgetDecision(
                decision="grant",
                reason_codes=["cheap_probe_auto_grant"],
                grant=BudgetGrant(
                    grant_id=f"grant_{request.request_id}",
                    approved=dict(requested),
                    conditions=[],
                    expires_at_generation=None,
                ),
            )
        return BudgetDecision(
            decision="require_review",
            reason_codes=[
                "weak_evidence_over_budget" if confidence == "weak" else "budget_requires_review"
            ],
            grant=None,
            review_target="chair",
        )


def create_policy() -> DefaultBasicBudgetPolicy:
    """Manifest entrypoint that returns the default basic budget policy."""
    return DefaultBasicBudgetPolicy()
