"""Generation cohort execution for the research-loop stage."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import threading
import time
from typing import Any, cast

from praxist.core.role_skills import RoleSkill
from praxist.plugins.workflow_stages.research_loop.backend.agent import (
    AutonomousAgentLoop,
    resolve_prompt_with_layout,
)
from praxist.plugins.workflow_stages.research_loop.backend.dig.cohort_allocator import (
    allocate_cohort_qd_contracts,
)
from praxist.plugins.workflow_stages.research_loop.backend.dig.prompts import (
    implementation_contract_block,
)
from praxist.plugins.workflow_stages.research_loop.backend.dig.runner import (
    run_dig_lite,
)
from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
    PeerSessionMemory,
)
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import is_scoreless
from praxist.plugins.workflow_stages.research_loop.backend.tools.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

_PEER_DRAIN_GRACE_SECONDS = 120  # grace period for peers to exit after STOP_SIGNAL


def _effective_generation_cap_seconds(
    synthesis_trigger_config: Any,
    *,
    per_peer_safety_seconds: float | None,
) -> float | None:
    """Return the same effective close horizon enforced by the trigger."""

    if not bool(getattr(synthesis_trigger_config, "enabled", True)):
        return per_peer_safety_seconds
    if getattr(synthesis_trigger_config, "max_interval_minutes", 0.0) is None:
        return per_peer_safety_seconds

    from praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger import (
        AdaptiveSynthesisPolicy,
    )

    fixed_minutes = max(
        0.0,
        float(getattr(synthesis_trigger_config, "max_interval_minutes", 0.0) or 0.0),
    )
    adaptive = AdaptiveSynthesisPolicy.from_raw(getattr(synthesis_trigger_config, "adaptive", {}))
    adaptive_minutes = max(0.0, adaptive.max_interval_ceiling_minutes) if adaptive.enabled else 0.0
    trigger_horizon = max(60.0, max(fixed_minutes, adaptive_minutes) * 60.0)
    return (
        trigger_horizon
        if per_peer_safety_seconds is None
        else min(per_peer_safety_seconds, trigger_horizon)
    )


def _start_generation_deadline_watchdog(
    trigger: Any,
    *,
    deadline: float,
    gen_id: int,
) -> tuple[threading.Event, threading.Thread]:
    """Start an out-of-loop watchdog for the configured generation deadline."""

    stop = threading.Event()

    def watch() -> None:
        if stop.wait(max(0.0, deadline - time.time())):
            return
        logger.warning(
            "Generation %d reached its hard deadline outside the research-loop event loop; "
            "publishing the existing generation wall-time stop signal.",
            gen_id,
        )
        try:
            trigger.fire_deadline()
        except Exception as exc:  # noqa: BLE001 - main-loop timeout path remains available.
            logger.error(
                "Generation %d deadline watchdog could not publish STOP_SIGNAL: %s",
                gen_id,
                exc,
            )

    thread = threading.Thread(
        target=watch,
        name=f"praxist-generation-{gen_id}-deadline",
        daemon=True,
    )
    thread.start()
    return stop, thread


def _load_dig_stage_status(dig_dir: Any) -> dict[str, Any]:
    """Best-effort DIG stage status for failure summaries."""

    try:
        path = dig_dir / "dig_stage_status.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - status is diagnostic only.
        return {}
    return payload if isinstance(payload, dict) else {}


def _protected_job_belongs_to_generation(peer_id: str, gen_id: int) -> bool:
    """Return True when a protected PID entry belongs to ``gen_id``."""

    value = str(peer_id or "")
    with contextlib.suppress(Exception):
        from praxist.plugins.workflow_stages.research_loop.backend.protected_pids import (
            _generation_id_from_peer_id,
        )

        if _generation_id_from_peer_id(value) == gen_id:
            return True
    return value.startswith(f"gen{gen_id}_peer") or value.startswith(f"gen{gen_id}/peer")


def _active_generation_work_count(
    peer_tasks: list[asyncio.Task[Any]],
    *,
    run_dir: Any,
    gen_id: int,
) -> int:
    """Count active peer tasks plus current-generation protected background jobs."""

    active_peer_tasks = sum(1 for task in peer_tasks if not task.done())
    try:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        protected_jobs = [
            entry
            for entry in protected_pids.list_active_jobs(run_dir=run_dir)
            if _protected_job_belongs_to_generation(entry.peer_id, gen_id)
        ]
    except Exception as exc:  # noqa: BLE001 - trigger liveness must be best-effort.
        logger.debug("cohort active-work check could not read protected_pids: %s", exc)
        protected_jobs = []
    return active_peer_tasks + len(protected_jobs)


def _closing_trigger_wait_seconds(trigger: Any, protected_jobs: list[Any]) -> float:
    """Return a bounded wait for a closing trigger to flush current work."""

    adaptive_policy = getattr(trigger, "adaptive_policy", None)
    drain_minutes = float(getattr(adaptive_policy, "drain_grace_minutes", 0.0) or 0.0)
    wait_seconds = max(60.0, drain_minutes * 60.0 + 120.0)
    protected_eta_seconds = [
        max(0, int(getattr(job, "eta_seconds", 0) or 0)) for job in protected_jobs
    ]
    if protected_eta_seconds:
        wait_seconds = max(wait_seconds, min(max(protected_eta_seconds) + 120.0, 3600.0))
    return wait_seconds


def _dig_diagnostic_quota(dig_config: Any, cohort_size: int) -> int:
    """Return the max diagnostic/control DIG slots for this cohort."""

    innovation = getattr(dig_config, "innovation", None)
    if innovation is None or not bool(getattr(innovation, "enabled", False)):
        return cohort_size
    max_peers = max(0, int(getattr(innovation, "max_diagnostic_peers", 2) or 0))
    fraction = max(0.0, float(getattr(innovation, "max_diagnostic_fraction", 0.20) or 0.0))
    fraction_cap = int(cohort_size * fraction)
    if fraction > 0 and cohort_size > 0 and fraction_cap == 0:
        fraction_cap = 1
    if max_peers <= 0:
        return min(cohort_size, fraction_cap)
    if fraction_cap <= 0:
        return min(cohort_size, max_peers)
    return min(cohort_size, max_peers, fraction_cap)


def _ctx_requests_diagnostic_slot(ctx: dict[str, Any], dig_config: Any) -> bool:
    """Return True when the agenda explicitly asks this peer for diagnostic work."""

    peer_id = str(ctx.get("peer_id") or "")
    agenda = ctx.get("research_agenda")
    contracts = agenda.get("peer_contracts") if isinstance(agenda, dict) else {}
    contract = contracts.get(peer_id) if isinstance(contracts, dict) else None
    if not isinstance(contract, dict):
        return False
    for key in ("diagnostic_only", "is_diagnostic", "requires_diagnostic_slot"):
        value = contract.get(key)
        if value is True or str(value or "").strip().lower() in {"1", "true", "yes", "on"}:
            return True
    innovation = getattr(dig_config, "innovation", None)
    diagnostic_labels = {
        str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
        for item in [
            *(getattr(innovation, "diagnostic_intents", None) or []),
            *(getattr(innovation, "diagnostic_mechanism_families", None) or []),
        ]
        if str(item or "").strip()
    }
    values: list[str] = []
    for key in (
        "role",
        "intent",
        "intent_preference",
        "next_step_intent",
        "mechanism_family",
        "innovation_surface",
    ):
        raw = contract.get(key)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw)
        elif raw is not None:
            values.append(str(raw))
    normalized_values = {
        value.strip().lower().replace("-", "_").replace(" ", "_")
        for value in values
        if value.strip()
    }
    return bool(normalized_values & diagnostic_labels)


def _assign_dig_selection_policies(
    contexts: dict[int, dict[str, Any]],
    *,
    dig_config: Any,
    cohort_size: int,
) -> dict[int, dict[str, Any]]:
    """Assign cohort-level DIG intent slots before peers plan independently.

    DIG still creates candidate pools with falsifiers, but the selected contract
    should be forward-moving for most peers.  Diagnostic slots are first granted
    to explicit agenda requests, then to the final peer indices as a stable
    fallback for generation-0 cohorts with no agenda.
    """

    innovation = getattr(dig_config, "innovation", None)
    if (
        dig_config is None
        or innovation is None
        or not bool(getattr(innovation, "enabled", False))
        or not bool(getattr(innovation, "enforce_forward_slots", True))
    ):
        return {
            index: {
                "intent_slot": "unconstrained",
                "reason": "DIG innovation cohort balancing disabled.",
            }
            for index in contexts
        }

    quota = _dig_diagnostic_quota(dig_config, cohort_size)
    requested = [
        index for index, ctx in contexts.items() if _ctx_requests_diagnostic_slot(ctx, dig_config)
    ]
    diagnostic_indices: list[int] = []
    for index in requested:
        if index not in diagnostic_indices and len(diagnostic_indices) < quota:
            diagnostic_indices.append(index)
    for index in range(max(0, cohort_size - quota), cohort_size):
        if (
            index in contexts
            and index not in diagnostic_indices
            and len(diagnostic_indices) < quota
        ):
            diagnostic_indices.append(index)
    diagnostic_set = set(diagnostic_indices)
    policies: dict[int, dict[str, Any]] = {}
    for index in contexts:
        slot = "diagnostic" if index in diagnostic_set else "forward_innovation"
        policies[index] = {
            "intent_slot": slot,
            "cohort_size": cohort_size,
            "peer_index": index,
            "max_diagnostic_peers": quota,
            "diagnostic_indices": sorted(diagnostic_set),
            "reason": (
                "Reserved diagnostic DIG slot."
                if slot == "diagnostic"
                else "Forward-innovation DIG slot; diagnostic candidates are coverage only unless no forward candidate survives."
            ),
        }
    return policies


def _generation_feature_enabled(config: Any, generation_id: int, *, fallback: bool) -> bool:
    resolver = getattr(config, "enabled_for_generation", None)
    if callable(resolver):
        try:
            return bool(resolver(generation_id))
        except (TypeError, ValueError):
            return fallback
    return fallback


def _clear_current_generation_qd_views(gen_dir: Any, gen_id: int) -> None:
    """Remove stale QD views before rebuilding the current generation."""

    paths = [gen_dir / "dig_cohort_allocation.yaml"]
    peers_dir = gen_dir / "peers"
    if peers_dir.is_dir():
        paths.extend(peers_dir.glob(f"gen{gen_id}_peer*/dig/cohort_qd_override.yaml"))
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("could not remove stale current-generation QD view %s: %s", path, exc)


def clear_generation_runtime_signals(gen_dir: Any) -> None:
    """Remove transient generation controls before rerunning its cohort."""

    from praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger import (
        CLOSING_SIGNAL_FILENAME,
        STOP_SIGNAL_FILENAME,
        STOP_SIGNAL_POSTGEN_FILENAME,
    )

    stop_signal_path = gen_dir / STOP_SIGNAL_FILENAME
    closing_signal_path = gen_dir / CLOSING_SIGNAL_FILENAME
    for stale in (
        stop_signal_path,
        closing_signal_path,
        gen_dir / STOP_SIGNAL_POSTGEN_FILENAME,
        stop_signal_path.with_suffix(".tmp"),
        closing_signal_path.with_suffix(".tmp"),
    ):
        try:
            stale.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("could not remove stale sentinel %s: %s", stale, exc)


async def run_generation_cohort(loop: Any, gen_id: int) -> list[dict[str, Any]]:
    """Run one generation cohort using the owning ``GenerationLoop`` state."""
    gp = loop.task_spec.generation_policy
    cohort_size = gp.cohort_size
    per_peer_safety_seconds = (
        None if gp.per_generation_hours is None else gp.per_generation_hours * 3600
    )
    lifecycle = getattr(loop, "_task_lifecycle", None)

    def _remaining_wait_seconds(requested: float) -> float:
        remaining = lifecycle.remaining_seconds() if lifecycle is not None else None
        return max(0.0, requested if remaining is None else min(requested, remaining))

    def _research_deadline_expired() -> bool:
        remaining = lifecycle.remaining_seconds() if lifecycle is not None else None
        return remaining is not None and remaining <= 0

    remaining = lifecycle.remaining_seconds() if lifecycle is not None else None
    if remaining is not None:
        if remaining <= 0:
            return []
        per_peer_safety_seconds = (
            remaining
            if per_peer_safety_seconds is None
            else min(per_peer_safety_seconds, remaining)
        )

    os.environ["GENERATION_ID"] = str(gen_id)
    logical_gen_id = gen_id
    gems_context = {}
    if hasattr(loop, "gems"):
        logical_gen_id = loop.gems.logical_generation(gen_id)
        gems_context = loop.gems.prompt_context(gen_id)
    os.environ["PRAXIST_LOGICAL_GENERATION_ID"] = str(logical_gen_id)
    os.environ["PRAXIST_GEMS_CYCLE"] = str(gems_context.get("cycle_index", 0) or 0)

    gen_dir = loop.run_dir / f"gen_{gen_id}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    # The orchestrator owns canonical containers. Creating the neutral
    # generation parent before peer sessions start lets a peer-owned evaluator
    # create only its own descendant without broadening runtime-guard access.
    (loop.run_dir / "results" / f"gen_{gen_id}").mkdir(parents=True, exist_ok=True)
    maybe_peer_mcp_servers = getattr(loop, "_peer_mcp_servers", None)
    peer_mcp_servers = cast(
        dict[str, Any],
        loop.mcp_servers if maybe_peer_mcp_servers is None else maybe_peer_mcp_servers,
    )

    from praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger import (
        CLOSING_SIGNAL_FILENAME,
        STOP_SIGNAL_FILENAME,
        STOP_SIGNAL_POSTGEN_FILENAME,
        SynthesisTrigger,
    )

    clear_generation_runtime_signals(gen_dir)
    stop_signal_path = gen_dir / STOP_SIGNAL_FILENAME
    closing_signal_path = gen_dir / CLOSING_SIGNAL_FILENAME

    logger.info(
        f"\n{'=' * 60}\n"
        f"Generation {gen_id} — cohort_size={cohort_size}, "
        f"event-driven (synthesis_trigger)\n"
        f"{'=' * 60}"
    )

    prebuilt_contexts: dict[int, dict[str, Any]] = {}
    for i in range(cohort_size):
        peer_id = f"gen{gen_id}_peer{i}"
        prebuilt_contexts[i] = {
            **loop._build_prompt_context(gen_id, i, cohort_size),
            "peer_id": peer_id,
        }
    dig_config = getattr(loop, "dig_lite_config", None)
    scoreless = is_scoreless(loop.task_spec)
    dig_active = not scoreless and _generation_feature_enabled(
        dig_config,
        gen_id,
        fallback=bool(dig_config is not None and getattr(dig_config, "enabled", False)),
    )
    qd_config = getattr(loop, "quality_diversity_config", None)
    legacy_qd = bool(dig_active)
    qd_active = not scoreless and _generation_feature_enabled(
        qd_config,
        gen_id,
        fallback=legacy_qd,
    )
    cohort_qd_config = (
        getattr(qd_config, "cohort", None)
        if qd_config is not None
        else getattr(dig_config, "cohort_qd", None)
    )
    cohort_qd_active = bool(dig_active and qd_active and getattr(cohort_qd_config, "enabled", True))
    _clear_current_generation_qd_views(gen_dir, gen_id)
    dig_selection_policies = _assign_dig_selection_policies(
        prebuilt_contexts,
        dig_config=dig_config if dig_active else None,
        cohort_size=cohort_size,
    )

    async def _run_peer_dig(i: int) -> dict[str, Any]:
        peer_id = f"gen{gen_id}_peer{i}"
        ctx = dict(prebuilt_contexts[i])

        dig_result = None
        if dig_active:
            assert dig_config is not None
            try:
                PeerSessionMemory(
                    run_dir=loop.run_dir,
                    gen_dir=gen_dir,
                    peer_id=peer_id,
                    generation_id=gen_id,
                    findings_dir=loop.findings_dir,
                ).initialize()
            except Exception as exc:  # noqa: BLE001 - peer launch retains its existing fallback.
                logger.warning(
                    "Could not initialize peer memory before DIG for %s: %s",
                    peer_id,
                    exc,
                )
            dig_dir = gen_dir / "peers" / peer_id / "dig"
            max_attempts = min(
                10,
                max(1, int(getattr(dig_config, "max_attempts", 10) or 10)),
            )
            total_budget_seconds = max(
                30.0,
                float(getattr(dig_config, "max_total_runtime_minutes", 10) or 10) * 60.0,
            )
            phase_cap_seconds = max(
                30.0,
                float(getattr(dig_config, "planner_max_runtime_minutes", 10) or 10) * 60.0,
            )
            attempt_cap_minutes = float(getattr(dig_config, "attempt_max_runtime_minutes", 0) or 0)
            attempt_cap_seconds = (
                max(30.0, attempt_cap_minutes * 60.0) if attempt_cap_minutes > 0 else None
            )
            dig_started_at = time.monotonic()
            dig_deadline = dig_started_at + total_budget_seconds
            dig_errors: list[dict[str, Any]] = []
            for attempt in range(1, max_attempts + 1):
                remaining_budget = dig_deadline - time.monotonic()
                if remaining_budget <= 0:
                    dig_errors.append(
                        {
                            "attempt": attempt,
                            "error_type": "DIGTotalTimeout",
                            "error": (
                                "DIG total runtime budget exhausted before next attempt; "
                                f"total_budget_seconds={total_budget_seconds:.1f}"
                            ),
                            "written_at": time.time(),
                            "elapsed_seconds": time.monotonic() - dig_started_at,
                        }
                    )
                    break
                attempt_budget = max(
                    1.0,
                    min(
                        remaining_budget,
                        attempt_cap_seconds
                        if attempt_cap_seconds is not None
                        else remaining_budget,
                    ),
                )
                try:
                    dig_result = await run_dig_lite(
                        ctx={
                            **ctx,
                            "agent_runtime_ref": str(
                                getattr(loop, "runtime_ref", "") or "agent_runtime:claude_sdk"
                            ),
                            "dig_attempt": attempt,
                            "dig_max_attempts": max_attempts,
                            "dig_selection_policy": dig_selection_policies.get(i, {}),
                            "dig_remaining_budget_seconds": int(attempt_budget),
                            "dig_total_budget_seconds": int(total_budget_seconds),
                            "dig_attempt_timeout_seconds": int(attempt_budget),
                            "dig_phase_timeout_seconds": int(phase_cap_seconds),
                        },
                        config=dig_config,
                        dig_dir=dig_dir,
                        workspace=loop.workspace,
                        model=loop.model,
                        mcp_servers=peer_mcp_servers,
                        plugin_registry=loop.plugin_registry,
                        premium_mode=bool(
                            getattr(
                                getattr(loop.task_spec, "agent", None),
                                "premium_mode",
                                False,
                            )
                        ),
                        reasoning_effort=str(
                            getattr(
                                getattr(loop.task_spec, "agent", None),
                                "reasoning_effort",
                                "max",
                            )
                        ),
                        quality_diversity_enabled=qd_active,
                    )
                    logger.info(
                        "DIG-Lite succeeded for %s on attempt %d/%d",
                        peer_id,
                        attempt,
                        max_attempts,
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - retry; fallback if exhausted.
                    logger.warning(
                        "DIG-Lite attempt %d/%d failed before launching %s: %s",
                        attempt,
                        max_attempts,
                        peer_id,
                        exc,
                    )
                    stage_status = _load_dig_stage_status(dig_dir)
                    error_message = str(exc) or type(exc).__name__
                    dig_errors.append(
                        {
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": error_message,
                            "written_at": time.time(),
                            "elapsed_seconds": time.monotonic() - dig_started_at,
                            "attempt_budget_seconds": attempt_budget,
                            "phase_timeout_seconds": phase_cap_seconds,
                            "total_budget_seconds": total_budget_seconds,
                            "last_phase": stage_status.get("last_phase"),
                            "last_status": stage_status.get("last_status"),
                            "stage_event_count": len(stage_status.get("events") or []),
                        }
                    )

            if dig_result is not None:
                return {
                    "peer_index": i,
                    "peer_id": peer_id,
                    "ctx": ctx,
                    "dig_result": dig_result,
                    "prelaunch_result": None,
                }
            else:
                dig_dir.mkdir(parents=True, exist_ok=True)
                failure_summary = {
                    "peer_id": peer_id,
                    "generation_id": gen_id,
                    "dig_lite_enabled": True,
                    "max_attempts": max_attempts,
                    "max_total_runtime_minutes": float(
                        getattr(dig_config, "max_total_runtime_minutes", 10) or 10
                    ),
                    "planner_max_runtime_minutes": float(
                        getattr(dig_config, "planner_max_runtime_minutes", 10) or 10
                    ),
                    "attempt_max_runtime_minutes": attempt_cap_minutes,
                    "phase_timeout_seconds": phase_cap_seconds,
                    "elapsed_seconds": time.monotonic() - dig_started_at,
                    "fallback_to_direct_on_failure": bool(
                        getattr(dig_config, "fallback_to_direct_on_failure", True)
                    ),
                    "stage_status": _load_dig_stage_status(dig_dir),
                    "errors": dig_errors,
                }
                with open(dig_dir / "dig_failure_summary.json", "w", encoding="utf-8") as f:
                    json.dump(failure_summary, f, indent=2, default=str)
                if bool(getattr(dig_config, "fallback_to_direct_on_failure", True)):
                    logger.error(
                        "DIG-Lite failed %d/%d attempts for %s; falling back to direct peer prompt.",
                        len(dig_errors),
                        max_attempts,
                        peer_id,
                    )
                    ctx["dig_lite"] = {
                        "enabled": True,
                        "status": "fallback_to_direct_after_failed_attempts",
                        "dig_dir": str(dig_dir),
                        "attempt_count": len(dig_errors),
                    }
                else:
                    logger.error(
                        "DIG-Lite failed for %s and fallback is disabled; suppressing peer launch.",
                        peer_id,
                    )
                    return {
                        "peer_index": i,
                        "peer_id": peer_id,
                        "ctx": ctx,
                        "dig_result": None,
                        "peer": None,
                        "prelaunch_result": {
                            "peer_id": peer_id,
                            "generation_id": gen_id,
                            "success": False,
                            "error": f"DIG-Lite failed after {max_attempts} attempts",
                            "phase": "dig_lite",
                            "dig_lite": failure_summary,
                        },
                    }

        return {
            "peer_index": i,
            "peer_id": peer_id,
            "ctx": ctx,
            "dig_result": dig_result,
            "prelaunch_result": None,
        }

    def _build_peer_from_prelaunch(prepared: dict[str, Any]) -> dict[str, Any]:
        peer_id = str(prepared["peer_id"])
        ctx = dict(prepared["ctx"])
        dig_result = prepared.get("dig_result")
        extra_dynamic_blocks: list[dict[str, Any]] = []

        if dig_result is not None:
            ctx["dig_lite"] = dig_result.to_prompt_context(loop.run_dir)
            if getattr(dig_config, "inject_contract_into_prompt", True):
                extra_dynamic_blocks.append(
                    {
                        "block_id": "dig_lite_selected_contract",
                        "renderer": "static_text",
                        "source_path": str(dig_result.selected_contract_path),
                        "text": implementation_contract_block(
                            dig_result.selected_contract.to_dict(),
                            dig_result.selected_contract_path,
                        ),
                    }
                )

        role_resolver = getattr(loop, "peer_role_skill_for_context", None)
        peer_role_skill = cast(
            RoleSkill | None,
            (
                role_resolver(ctx)
                if callable(role_resolver)
                else getattr(loop, "peer_role_skill", None)
            ),
        )
        role_ref_resolver = getattr(loop, "peer_role_ref_for_context", None)
        peer_role_ref = cast(
            str | None,
            (
                role_ref_resolver(ctx)
                if callable(role_ref_resolver)
                else (
                    peer_role_skill.role_ref
                    if peer_role_skill is not None
                    else getattr(loop, "peer_role_ref", None)
                )
            ),
        )
        if peer_role_skill is not None:
            extra_dynamic_blocks.insert(
                0,
                {
                    "block_id": "task_peer_role_contract",
                    "renderer": "static_text",
                    "source_path": str(peer_role_skill.plugin_path / "skill.md"),
                    "text": (
                        "## Selected peer RoleSkill contract\n\n"
                        f"Role: `{peer_role_skill.role_ref}`\n\n"
                        f"{peer_role_skill.skill_markdown.strip()}"
                    ),
                },
            )

        prompt_output = gen_dir / f"{peer_id}_prompt.md"
        prompt_layout_path = gen_dir / f"{peer_id}_prompt_layout.json"
        prompt_text, prompt_layout_manifest = resolve_prompt_with_layout(
            base_template_path=loop.base_template,
            task_prompt_path=loop.task_prompt_path,
            generation_template_path=loop.gen_template,
            output_path=prompt_output,
            layout_output_path=prompt_layout_path,
            context=ctx,
            prompt_id=f"gen_{gen_id}/{peer_id}",
            extra_dynamic_blocks=extra_dynamic_blocks,
            start_command_text=(
                "Begin by reading the assigned task, research notebook, and prior retained "
                "findings. Use that evidence to plan and carry out your assigned research; "
                "retain useful findings without metric ranking."
                if scoreless
                else None
            ),
        )
        prompt_layout_manifest = loop._persist_prompt_layout_artifacts(
            prompt_text=prompt_text,
            prompt_path=prompt_output,
            manifest=prompt_layout_manifest,
            manifest_path=prompt_layout_path,
            peer_id=peer_id,
            gen_id=gen_id,
        )

        peer_logs = gen_dir / peer_id
        peer_logs.mkdir(parents=True, exist_ok=True)
        peer = AutonomousAgentLoop(
            peer_id=peer_id,
            generation_id=gen_id,
            task_prompt=prompt_text,
            workspace=loop.workspace,
            max_runtime_seconds=(
                None
                if per_peer_safety_seconds is None
                else max(1, math.ceil(per_peer_safety_seconds))
            ),
            logs_dir=peer_logs,
            findings_dir=loop.findings_dir,
            model=loop.model,
            local_mode=loop.local_mode,
            mcp_servers=peer_mcp_servers,
            allowed_tools=list(loop._peer_allowed_tools),
            stop_signal_path=stop_signal_path,
            closing_signal_path=closing_signal_path,
            premium_mode=bool(
                getattr(getattr(loop.task_spec, "agent", None), "premium_mode", False)
            ),
            reasoning_effort=str(
                getattr(getattr(loop.task_spec, "agent", None), "reasoning_effort", "max")
            ),
            prompt_layout_manifest=prompt_layout_manifest,
            plugin_registry=loop.plugin_registry,
            role_ref=peer_role_ref,
            role_skill_sha256=(
                peer_role_skill.content_hash if peer_role_skill is not None else None
            ),
            task_spec=loop.task_spec,
        )
        return {"peer_id": peer_id, "peer": peer, "prelaunch_result": None}

    peers = []
    started_peer_ids: list[str] = []
    prelaunch_results: list[dict[str, Any]] = []
    prepare_tasks = [asyncio.create_task(_run_peer_dig(i)) for i in range(cohort_size)]
    prepared_peers = await asyncio.gather(*prepare_tasks)
    successful_dig_results = {
        int(prepared["peer_index"]): prepared["dig_result"]
        for prepared in prepared_peers
        if prepared.get("dig_result") is not None
    }
    if cohort_qd_active:
        assert dig_config is not None
        successful_dig_results = allocate_cohort_qd_contracts(
            dig_results=successful_dig_results,
            config=dig_config,
            gen_dir=gen_dir,
        )
        for prepared in prepared_peers:
            index = int(prepared.get("peer_index", -1))
            if index in successful_dig_results:
                prepared["dig_result"] = successful_dig_results[index]
    for prepared in prepared_peers:
        prelaunch_result = prepared.get("prelaunch_result")
        if prelaunch_result is not None:
            prelaunch_results.append(prelaunch_result)
            continue
        prepared = _build_peer_from_prelaunch(prepared)
        peer = prepared.get("peer")
        if peer is None:
            continue
        started_peer_ids.append(str(prepared.get("peer_id") or getattr(peer, "peer_id", "")))
        peers.append(peer)

    # DIG/prelaunch planning has its own bounded budget. The generation close
    # horizon governs the peer implementation/evaluation loop that starts here.
    gen_start_time = time.time()
    for peer in peers:
        stop_checker = getattr(peer, "stop_checker", None)
        reset_start_time = getattr(stop_checker, "reset_start_time", None)
        if callable(reset_start_time):
            reset_start_time(gen_start_time)
    st_cfg = loop.task_spec.synthesis_trigger
    scheduler = getattr(loop, "_experiment_scheduler", None)
    generation_cap_seconds = _effective_generation_cap_seconds(
        st_cfg,
        per_peer_safety_seconds=per_peer_safety_seconds,
    )
    remaining = lifecycle.remaining_seconds() if lifecycle is not None else None
    if remaining is not None:
        generation_cap_seconds = (
            max(0.0, remaining)
            if generation_cap_seconds is None
            else min(generation_cap_seconds, max(0.0, remaining))
        )
        per_peer_safety_seconds = (
            max(0.0, remaining)
            if per_peer_safety_seconds is None
            else min(per_peer_safety_seconds, max(0.0, remaining))
        )
    watchdog_cap_seconds = generation_cap_seconds if st_cfg.enabled else per_peer_safety_seconds
    if scheduler is not None:
        scheduler.open_generation(
            gen_id,
            deadline=(
                None if generation_cap_seconds is None else gen_start_time + generation_cap_seconds
            ),
            cohort_size=len(peers),
            peer_ids=started_peer_ids,
        )

    def _trigger_pre_sync() -> None:
        if loop._findings_sync is not None:
            with contextlib.suppress(Exception):
                sync_blocking = getattr(loop._findings_sync, "sync_once_blocking", None)
                if callable(sync_blocking):
                    sync_blocking(timeout=2.0)
                else:
                    loop._findings_sync.sync_once()

    # #148: wrap peer coroutines in tasks first so the synthesis
    # trigger can observe their done-state. With the legacy
    # ``asyncio.gather(*[peer.run() for peer in peers])`` the only
    # done-signal the orchestrator had was ``gather`` returning,
    # which by definition only fires when every peer is terminal.
    # Exposing the live peer-task list to the trigger lets it fire
    # ``cohort_drained`` once the last peer exits, replacing the up-
    # to-240-min ``safety_cap`` wait that used to follow a fully
    # crashed cohort.
    peer_tasks = [asyncio.create_task(peer.run()) for peer in peers]

    def _active_peers() -> int:
        return _active_generation_work_count(
            peer_tasks,
            run_dir=loop.run_dir,
            gen_id=gen_id,
        )

    configured_min_contributing_peers = max(0, int(st_cfg.min_contributing_peers))
    launched_peer_count = len(peers)
    effective_min_contributing_peers = (
        min(configured_min_contributing_peers, max(1, launched_peer_count))
        if configured_min_contributing_peers > 0
        else 0
    )
    if effective_min_contributing_peers != configured_min_contributing_peers:
        logger.info(
            "synthesis_trigger gen %d: clamped min_contributing_peers %d -> %d for launched_peers=%d",
            gen_id,
            configured_min_contributing_peers,
            effective_min_contributing_peers,
            launched_peer_count,
        )

    cohort_completed = asyncio.Event()
    trigger = SynthesisTrigger(
        run_dir=loop.run_dir,
        gen_dir=gen_dir,
        gen_id=gen_id,
        gen_start_time=gen_start_time,
        min_findings=st_cfg.min_findings,
        min_interval_minutes=st_cfg.min_interval_minutes,
        max_interval_minutes=st_cfg.max_interval_minutes,
        min_contributing_peers=effective_min_contributing_peers,
        poll_interval_seconds=st_cfg.poll_interval_seconds,
        adaptive_policy=getattr(st_cfg, "adaptive", {}),
        maturity_policy=getattr(getattr(loop.task_spec, "evaluation", None), "maturity_policy", {}),
        mature_quorum_fraction=getattr(st_cfg, "mature_quorum_fraction", 0.0),
        cohort_size=launched_peer_count,
        started_peer_ids=started_peer_ids,
        pre_eval_sync_callback=_trigger_pre_sync,
        cohort_active_peers_callback=_active_peers,
        cohort_completed_event=cohort_completed,
        # #75 batch 9: thread the store root explicitly so the trigger
        # never reads LOCAL_STORE_DIR from os.environ. ``run_dir`` is
        # the in-process answer; operators that want to override the
        # path still set ``LOCAL_STORE_DIR`` before launching ``praxist
        # start`` and ``startup.py`` propagates it via the same
        # ``loop.run_dir``.
        local_store_dir=loop.run_dir,
    )

    if scheduler is not None:
        supply_fraction = float(
            getattr(getattr(scheduler, "settings", None), "mature_supply_fraction", 0.0) or 0.0
        )
        supply_redundancy = float(
            getattr(getattr(scheduler, "settings", None), "mature_supply_redundancy", 0.0) or 0.0
        )
        configured_target = 0
        if supply_fraction > 0 and supply_redundancy > 0:
            configured_target = max(
                1,
                int(math.ceil(launched_peer_count * supply_fraction)),
            )
            # A task-owned hard close quorum may deliberately require more
            # mature evidence than the default supply target. Supplying less
            # would make that explicit close contract impossible to satisfy.
            configured_target = max(
                configured_target,
                trigger.required_mature_result_peers,
            )
        scheduler.configure_generation_maturity(
            gen_id,
            cohort_size=launched_peer_count,
            mature_target=configured_target,
            count_callback=(
                trigger.mature_peer_count
                if trigger.required_mature_result_peers > 0
                else trigger.mature_result_count
            ),
            identity_callback=(
                trigger.mature_peer_ids if trigger.required_mature_result_peers > 0 else None
            ),
            distinct_peer_commitments=trigger.required_mature_result_peers > 0,
        )

    trigger_done = asyncio.Event()

    async def _trigger_task() -> None:
        try:
            if not st_cfg.enabled:
                logger.info("synthesis_trigger disabled by config; peers use their runtime policy.")
                return
            await trigger.wait_until_fire(abort_event=trigger_done)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - peers still finish or safety-cap.
            logger.error(
                "synthesis_trigger task raised (gen %d): %s — "
                "peers will run to safety cap and orchestrator will still "
                "write a STOP_SIGNAL_POSTGEN marker.",
                gen_id,
                e,
            )

    trigger_task = asyncio.create_task(_trigger_task())

    # When the synthesis trigger fires, give peers a grace period to exit
    # on their own (they check STOP_SIGNAL between sessions), then cancel.
    async def _cancel_peers_after_trigger() -> None:
        if not st_cfg.enabled:
            if per_peer_safety_seconds is None:
                logger.info(
                    "synthesis_trigger disabled for gen %d with no peer runtime cap; "
                    "waiting for peers to finish or operator cancellation.",
                    gen_id,
                )
                await trigger_done.wait()
                return
            logger.info(
                "synthesis_trigger disabled for gen %d; peers run to generation runtime cap %.0fs.",
                gen_id,
                per_peer_safety_seconds,
            )
            await asyncio.sleep(_remaining_wait_seconds(max(1.0, float(per_peer_safety_seconds))))
        else:
            completed, _ = await asyncio.wait(
                {trigger_task},
                timeout=(
                    None
                    if per_peer_safety_seconds is None
                    else _remaining_wait_seconds(max(1.0, float(per_peer_safety_seconds)))
                ),
            )
            if trigger_task in completed:
                logger.info(
                    "synthesis_trigger fired for gen %d — waiting %d s for peers to exit gracefully",
                    gen_id,
                    _PEER_DRAIN_GRACE_SECONDS,
                )
                await asyncio.sleep(_remaining_wait_seconds(_PEER_DRAIN_GRACE_SECONDS))
            else:
                logger.warning(
                    "Generation %d reached peer runtime cap %.0fs before synthesis_trigger fired; "
                    "writing STOP_SIGNAL and draining active peers.",
                    gen_id,
                    per_peer_safety_seconds,
                )
                try:
                    if _research_deadline_expired():
                        trigger.fire_deadline()
                    else:
                        snap = await trigger.evaluate_async()
                        snap.fired = True
                        snap.reason = "generation_wall_timeout"
                        trigger.fire(snap)
                except Exception as e:  # noqa: BLE001 - cancellation still proceeds.
                    logger.warning(
                        "Could not write generation wall-time STOP_SIGNAL for gen %d: %s",
                        gen_id,
                        e,
                    )
                await asyncio.sleep(_remaining_wait_seconds(_PEER_DRAIN_GRACE_SECONDS))
        try:
            from .experiment_scheduler_client import freeze_generation

            freeze_generation(gen_id, "generation_drain")
        except Exception as exc:  # noqa: BLE001 - legacy signal/cancellation still proceeds.
            logger.warning("Could not freeze experiment queue for gen %d drain: %s", gen_id, exc)
        if st_cfg.enabled and not trigger.fired and not stop_signal_path.exists():
            logger.warning(
                "Generation %d reached drain without a STOP_SIGNAL; active peers will be cancelled.",
                gen_id,
            )
        active = [t for t in peer_tasks if not t.done()]
        if active:
            logger.warning(
                "Cancelling %d peer task(s) that did not exit after STOP_SIGNAL",
                len(active),
            )
            for t in active:
                t.cancel()

    _cancel_task = asyncio.create_task(_cancel_peers_after_trigger())
    deadline_watchdog = (
        None
        if watchdog_cap_seconds is None
        else _start_generation_deadline_watchdog(
            trigger,
            deadline=gen_start_time + watchdog_cap_seconds,
            gen_id=gen_id,
        )
    )

    def _peer_task_results() -> list[Any]:
        collected: list[Any] = []
        for peer, task in zip(peers, peer_tasks, strict=True):
            if not task.done():
                collected.append(
                    {
                        **(getattr(peer, "last_result", None) or {}),
                        "peer_id": peer.peer_id,
                        "generation_id": gen_id,
                        "success": False,
                        "status": "late_quarantined_peer_task",
                        "late_result_policy": "quarantined_signal",
                        "promotion_eligible": False,
                        "error": "peer task still active after bounded cancellation",
                    }
                )
                continue
            if task.cancelled():
                partial = getattr(peer, "last_result", None)
                collected.append(
                    partial
                    if isinstance(partial, dict)
                    else RuntimeError("peer task cancelled at generation boundary")
                )
                continue
            try:
                collected.append(task.result())
            except Exception as exc:  # noqa: BLE001 - preserve per-peer failure.
                collected.append(exc)
            except BaseException as exc:  # noqa: BLE001 - normalize cancellation-like failures.
                collected.append(
                    RuntimeError(f"peer task ended unexpectedly: {type(exc).__name__}: {exc}")
                )
        return collected

    def _active_protected_jobs(*, raise_on_error: bool = False) -> list[Any]:
        try:
            from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

            return [
                entry
                for entry in protected_pids.list_active_jobs(run_dir=loop.run_dir)
                if _protected_job_belongs_to_generation(entry.peer_id, gen_id)
            ]
        except Exception as exc:  # noqa: BLE001 - visibility beats silent loss.
            if raise_on_error:
                raise
            logger.debug(
                "Generation %d could not read protected background jobs for closing wait: %s",
                gen_id,
                exc,
            )
            return []

    def _late_protected_job_records() -> list[dict[str, Any]]:
        try:
            jobs = _active_protected_jobs(raise_on_error=True)
        except Exception as exc:  # noqa: BLE001 - visibility beats silent loss.
            logger.warning(
                "Generation %d could not read protected background jobs at boundary: %s",
                gen_id,
                exc,
            )
            return [
                {
                    "peer_id": f"gen{gen_id}_protected_jobs",
                    "generation_id": gen_id,
                    "success": True,
                    "status": "late_protected_jobs_unknown",
                    "late_result_policy": "quarantined_signal",
                    "promotion_eligible": False,
                    "message": (
                        "Praxist could not confirm protected background job state at the "
                        "generation boundary. Treat any later artifacts from this "
                        "generation as late/quarantined signals until revalidated."
                    ),
                    "error": str(exc),
                }
            ]
        records: list[dict[str, Any]] = []
        for job in jobs:
            records.append(
                {
                    "peer_id": getattr(job, "peer_id", "") or f"gen{gen_id}_unknown_peer",
                    "generation_id": gen_id,
                    "success": True,
                    "status": "late_quarantined_protected_job",
                    "late_result_policy": "quarantined_signal",
                    "promotion_eligible": False,
                    "pid": int(getattr(job, "pid", 0) or 0),
                    "tag": str(getattr(job, "tag", "") or ""),
                    "eta_seconds": int(getattr(job, "eta_seconds", 0) or 0),
                    "started_at": str(getattr(job, "started_at", "") or ""),
                    "message": (
                        "Protected background work was still active at the generation "
                        "boundary. Keep this as a visible late signal, but do not treat "
                        "future artifacts from the job as mature promotion evidence "
                        "until a later agent revalidates them."
                    ),
                }
            )
        if records:
            logger.warning(
                "Generation %d recorded %d protected background job(s) as late/quarantined signals",
                gen_id,
                len(records),
            )
        return records

    cancelled_error: asyncio.CancelledError | None = None

    def _interrupt_cohort_tasks() -> None:
        for task in [*peer_tasks, trigger_task, _cancel_task]:
            if not task.done():
                task.cancel()
        try:
            trigger.fire_deadline(
                "research_deadline_reached" if _research_deadline_expired() else "cohort_cancelled"
            )
        except Exception as error:  # noqa: BLE001 - peer cancellation must still happen.
            logger.warning("Could not publish cancellation signal for gen %d: %s", gen_id, error)

    try:
        if peer_tasks:
            while not _cancel_task.done() and any(not task.done() for task in peer_tasks):
                active_waits = [task for task in peer_tasks if not task.done()]
                await asyncio.wait(
                    [*active_waits, _cancel_task],
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            _done, pending = await asyncio.wait(peer_tasks, timeout=_remaining_wait_seconds(10.0))
            if pending:
                logger.warning(
                    "Generation %d still has %d peer task(s) after drain; recording them as boundary errors and continuing.",
                    gen_id,
                    len(pending),
                )
                for task in pending:
                    task.cancel()
                await asyncio.wait(pending, timeout=_remaining_wait_seconds(2.0))
    except asyncio.CancelledError as exc:
        # The outer run deadline cancels this cohort. Freeze admission and
        # request peer interruption before allowing finalization to proceed.
        # Completed peer output is serialized below before cancellation escapes.
        cancelled_error = exc
        _interrupt_cohort_tasks()
        if peer_tasks:
            try:
                await asyncio.wait(peer_tasks, timeout=_remaining_wait_seconds(10.0))
            except asyncio.CancelledError as repeated:
                cancelled_error = repeated
                _interrupt_cohort_tasks()
    finally:
        if all(task.done() for task in peer_tasks):
            cohort_completed.set()
        if deadline_watchdog is not None:
            deadline_watchdog_stop, deadline_watchdog_thread = deadline_watchdog
            deadline_watchdog_stop.set()
            deadline_watchdog_thread.join(timeout=_remaining_wait_seconds(2.0))

        async def _bounded_task_wait(task: asyncio.Task[Any], seconds: float) -> bool:
            nonlocal cancelled_error
            try:
                done, _ = await asyncio.wait({task}, timeout=_remaining_wait_seconds(seconds))
            except asyncio.CancelledError as exc:
                # Cancellation can arrive after all peers have finished, while
                # this function is already in its cleanup block. Retain those
                # results with the same contract as an interrupted cohort.
                cancelled_error = exc
                _interrupt_cohort_tasks()
                task.cancel()
                try:
                    done, _ = await asyncio.wait({*peer_tasks, task}, timeout=0.0)
                except asyncio.CancelledError as repeated:
                    cancelled_error = repeated
                    _interrupt_cohort_tasks()
                    done = {candidate for candidate in [*peer_tasks, task] if candidate.done()}
            if task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    task.result()
                return True
            return False

        async def _cancel_trigger_task() -> None:
            trigger_done.set()
            if cancelled_error is not None or _research_deadline_expired():
                trigger_task.cancel()
            if not await _bounded_task_wait(trigger_task, 10.0):
                trigger_task.cancel()
                await _bounded_task_wait(trigger_task, 2.0)

        pending_trigger_drain = bool(
            getattr(trigger, "closing", False) or getattr(trigger, "assessment_started", False)
        )
        if (
            st_cfg.enabled
            and pending_trigger_drain
            and not trigger.fired
            and cancelled_error is None
            and not _research_deadline_expired()
        ):
            wait_seconds = _remaining_wait_seconds(
                _closing_trigger_wait_seconds(trigger, _active_protected_jobs())
            )
            logger.info(
                "synthesis_trigger closing active for gen %d; waiting up to %.0fs "
                "for final STOP_SIGNAL before generation boundary.",
                gen_id,
                wait_seconds,
            )
            if not await _bounded_task_wait(trigger_task, wait_seconds):
                logger.warning(
                    "synthesis_trigger still closing after %.0fs for gen %d; "
                    "cancelling trigger task and falling back to post-gen evaluation.",
                    wait_seconds,
                    gen_id,
                )
                await _cancel_trigger_task()
        else:
            await _cancel_trigger_task()
        try:
            from .experiment_scheduler_client import freeze_generation

            freeze_generation(gen_id, "peers_finished")
        except Exception as exc:  # noqa: BLE001 - post-gen marker remains legacy fallback.
            logger.warning("Could not freeze experiment queue after gen %d peers: %s", gen_id, exc)
        _cancel_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await _bounded_task_wait(_cancel_task, 2.0)

        if not trigger.fired:
            try:
                if cancelled_error is not None or _research_deadline_expired():
                    trigger.fire_deadline(
                        "research_deadline_reached"
                        if _research_deadline_expired()
                        else "cohort_cancelled"
                    )
                else:
                    remaining = lifecycle.remaining_seconds() if lifecycle is not None else None
                    if remaining is None:
                        snap = await trigger.evaluate_async()
                    else:
                        evaluation_task = asyncio.create_task(trigger.evaluate_async())
                        if not await _bounded_task_wait(evaluation_task, remaining):
                            evaluation_task.cancel()
                            await _bounded_task_wait(evaluation_task, 0.0)
                            trigger.fire_deadline("research_deadline_reached")
                            raise TimeoutError(
                                "post-generation evaluation reached research deadline"
                            )
                        if evaluation_task.cancelled():
                            raise TimeoutError("post-generation evaluation interrupted")
                        snap = evaluation_task.result()
                    if st_cfg.enabled and snap.fired:
                        trigger.fire(snap)
                    else:
                        trigger.write_postgen_marker(snap)
            except asyncio.CancelledError as exc:
                # An unbounded evaluation can be interrupted after the cohort
                # has entered cleanup. Commit peer outputs before propagating it.
                cancelled_error = exc
                _interrupt_cohort_tasks()
            except Exception as e:  # noqa: BLE001 - preserve a postmortem marker.
                logger.debug("post-gen trigger evaluation failed: %s", e)
                try:
                    marker = gen_dir / STOP_SIGNAL_POSTGEN_FILENAME
                    marker.write_text(
                        f"trigger_reason=postgen_eval_raised\n"
                        f"gen_id={gen_id}\n"
                        f"exception_type={type(e).__name__}\n"
                        f"exception={str(e)[:200]}\n"
                        f"written_at={time.time():.0f}\n",
                        encoding="utf-8",
                    )
                except Exception:
                    pass

    results = _peer_task_results()
    gen_results = list(prelaunch_results)
    for peer_id, result in zip(started_peer_ids, results, strict=False):
        if isinstance(result, Exception):
            logger.error("Peer %s raised: %s", peer_id, result)
            gen_results.append(
                {
                    "peer_id": peer_id,
                    "generation_id": gen_id,
                    "success": False,
                    "error": str(result),
                }
            )
        elif isinstance(result, dict):
            gen_results.append(result)
        else:
            gen_results.append({"peer_id": peer_id, "generation_id": gen_id, "result": result})
    gen_results.extend(_late_protected_job_records())

    results_file = gen_dir / "generation_results.json"
    atomic_write_json(results_file, gen_results)

    if cancelled_error is not None:
        raise cancelled_error
    return gen_results
