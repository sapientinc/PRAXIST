"""Budget policy registry dispatch and shared unit validation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol

from praxist.core.protocol import BudgetDecision, BudgetRequest
from praxist.core.registry import (
    PluginLoader,
    PluginRef,
    PluginRegistry,
    PluginRoots,
    require_execution_plugin,
)

ALLOWED_BUDGET_UNITS = {"tokens", "wall_clock_seconds", "gpu_hours"}


class BudgetPolicy(Protocol):
    """Protocol for deterministic budget policies that turn requests into grants, denials, or review decisions."""

    def decide(self, request: BudgetRequest, *, allow_uncapped: bool = False) -> BudgetDecision: ...


def policy_for_ref(policy_ref: str, registry: PluginRegistry | None = None) -> BudgetPolicy:
    """Return the bundled or registry-backed BudgetPolicy implementation for a plugin reference."""
    registry = registry or _load_single_budget_registry(policy_ref)
    require_execution_plugin(
        registry,
        policy_ref,
        kind="budget_policy",
        capability="budget.decide",
    )
    parsed = PluginRef.parse(policy_ref)
    plugin = registry.require(parsed.kind, parsed.name)
    if not hasattr(plugin, "decide"):
        raise TypeError(f"{policy_ref} entrypoint did not return a BudgetPolicy with decide()")
    return plugin


def _load_single_budget_registry(ref: str) -> PluginRegistry:
    parsed = PluginRef.parse(ref)
    if parsed.kind != "budget_policy":
        raise ValueError(f"Budget policy ref must use kind budget_policy: {ref}")
    loader = PluginLoader(PluginRoots.defaults())
    manifest = loader.resolve(
        [ref],
        run_id="budget_policy_spec",
        root_task_ref=ref,
        enforce_bundled_execution=True,
    )
    return loader.load(manifest)


def _invalid_budget_units(values: Mapping[str, float | None]) -> list[str]:
    invalid = []
    for unit, raw_value in values.items():
        if unit not in ALLOWED_BUDGET_UNITS:
            invalid.append(str(unit))
            continue
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            invalid.append(str(unit))
            continue
        if not math.isfinite(value) or value < 0:
            invalid.append(str(unit))
    return invalid
