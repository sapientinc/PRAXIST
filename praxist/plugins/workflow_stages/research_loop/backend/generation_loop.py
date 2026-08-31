"""
Generation loop — orchestrates multi-generation, multi-agent research.

Each generation spawns a cohort of parallel agent peers. After a generation
completes, the Frontier Store promotes top-K results and seeds the next
generation with frontier context.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from praxist.core.execution_policy import task_execution_policy_scope
from praxist.core.role_skills import RoleSkill
from praxist.core.run_config import DEFAULT_AGENT_MODEL, DEFAULT_WORKSPACE_ROOT
from praxist.core.tool_servers import (
    DEFAULT_RESEARCH_TOOL_SERVER_REFS,
    build_legacy_mcp_servers,
    peer_mcp_context,
)
from praxist.plugins.workflow_stages.research_loop.backend import (
    task_lifecycle_execution as task_execution,
)
from praxist.plugins.workflow_stages.research_loop.backend.baseline_runtime import (
    validate_baseline_cache_for_run,
)
from praxist.plugins.workflow_stages.research_loop.backend.cohort_runner import (
    run_generation_cohort,
)
from praxist.plugins.workflow_stages.research_loop.backend.dig.config import (
    DIGLiteConfig,
    QualityDiversityConfig,
)
from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
    collect_loop_findings,
    result_artifact_options_from_task_spec,
)
from praxist.plugins.workflow_stages.research_loop.backend.frontier import FrontierStore
from praxist.plugins.workflow_stages.research_loop.backend.gems import GemsManager
from praxist.plugins.workflow_stages.research_loop.backend.generation_boundary import (
    complete_generation_boundary,
    record_completed_generation_observation,
)
from praxist.plugins.workflow_stages.research_loop.backend.generation_resume import (
    prepare_resume_for_sidecars,
    prime_resume_boundary_evidence_cutoff,
    recover_pending_gems_reset_for_resume,
)
from praxist.plugins.workflow_stages.research_loop.backend.generation_resume import (
    repair_inferred_boundaries_for_resume as repair_inferred_gems_boundary_markers,
)
from praxist.plugins.workflow_stages.research_loop.backend.orchestrator_runtime import (
    enter_orchestrator_runtime_scope,
)
from praxist.plugins.workflow_stages.research_loop.backend.orchestrator_status import (
    OrchestratorSnapshot,
)
from praxist.plugins.workflow_stages.research_loop.backend.prompt_artifacts import (
    compact_artifact_ref as _compact_artifact_ref,
)
from praxist.plugins.workflow_stages.research_loop.backend.prompt_artifacts import (
    persist_prompt_layout_artifacts,
)
from praxist.plugins.workflow_stages.research_loop.backend.prompt_context import (
    build_prompt_context,
)
from praxist.plugins.workflow_stages.research_loop.backend.prompt_strategy import (
    _GENERIC_DIVERSITY_DIMENSIONS_DEFAULT,
    _build_axis_assignment_block,
    _build_diversity_penalty_block,
    _format_dimensions_block,
    _generate_variant_hint,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_memory_update import (
    update_research_memory_post_gen,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_topology import (
    LegacyResearchTopologyExecutor,
)
from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
    append_resume_event,
    canonical_completed_generation_count,
    load_generation_results,
)
from praxist.plugins.workflow_stages.research_loop.backend.run_lifecycle import (
    evaluate_run_stop_gate,
    write_run_stop_report,
)
from praxist.plugins.workflow_stages.research_loop.backend.run_report import (
    generate_loop_boundary_report,
)
from praxist.plugins.workflow_stages.research_loop.backend.run_summary import (
    write_run_summary,
)
from praxist.plugins.workflow_stages.research_loop.backend.runtime_environment import (
    configure_runtime_environment,
    initialize_local_store_if_needed,
)
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import is_scoreless
from praxist.plugins.workflow_stages.research_loop.backend.sidecars import (
    close_sidecars_and_runtime,
    start_sidecars,
)
from praxist.plugins.workflow_stages.research_loop.backend.status_snapshot import (
    build_orchestrator_status_snapshot,
)
from praxist.plugins.workflow_stages.research_loop.backend.task_lifecycle import TaskLifecycle
from praxist.plugins.workflow_stages.research_loop.lifecycle import ResearchRunLifecycleObserver
from praxist.plugins.workflow_stages.research_loop.peer_roles import (
    DEFAULT_TOPOLOGY_REF,
    PeerRoleSelector,
    index_peer_role_skills,
    resolve_peer_role_rotation,
    resolve_topology_peer_info,
)
from praxist.task_spec import TaskSpec

logger = logging.getLogger(__name__)


_recover_pending_gems_reset_for_resume = recover_pending_gems_reset_for_resume
_DEFAULT_TOPOLOGY_REF = DEFAULT_TOPOLOGY_REF
_index_peer_role_skills = index_peer_role_skills
_resolve_peer_role_rotation = resolve_peer_role_rotation
_resolve_topology_peer_info = resolve_topology_peer_info

__all__ = [
    "GenerationLoop",
    "_GENERIC_DIVERSITY_DIMENSIONS_DEFAULT",
    "_build_axis_assignment_block",
    "_build_diversity_penalty_block",
    "_compact_artifact_ref",
    "_format_dimensions_block",
    "_generate_variant_hint",
]


class GenerationLoop:
    """
    Top-level orchestrator for multi-generation research.

    Lifecycle:
        for gen_id in range(max_generations):
            1. Build prompt with frontier context
            2. Spawn cohort of parallel agents
            3. Wait for all to complete
            4. Promote top-K to frontier
            5. Check plateau / termination
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        workspace: Path | None = None,
        run_dir: Path | None = None,
        local_mode: bool = False,
        model: str = "",
        frontier_strategy: str = "auto",
        tool_server_refs: list[str] | tuple[str, ...] | None = None,
        plugin_registry: Any | None = None,
        # Issue #75 batch 3: threaded down to PIAgent → run_panel → BasePI
        # so ``task_role:*`` resolution doesn't fall back to
        # ``PRAXIST_TASK_PROJECT_PATH`` from ``os.environ``.
        task_project_path: Path | None = None,
        # Forwarded to prompt context for runtime attribution. Tool guidance
        # remains runtime-neutral; each runtime exposes the same MCP surface.
        runtime_ref: str = "",
        resume: bool = False,
        resume_policy: str = "completed_generation",
        run_lifecycle_observer: ResearchRunLifecycleObserver | None = None,
        peer_role_ref: str | None = None,
        peer_role_skill: RoleSkill | None = None,
        peer_role_skills: tuple[RoleSkill, ...] = (),
    ):
        # Default strategy is "auto": gen 0 free-explores for diversity,
        # then gen >= 1 follows PI-directed per-peer role contracts. Pass
        # "mixed" / "exploit" / "explore" explicitly only for legacy
        # debugging or operator overrides.
        self.task_spec = task_spec
        self.workspace = Path(workspace) if workspace else Path(DEFAULT_WORKSPACE_ROOT)
        self.local_mode = local_mode
        self.model = model or DEFAULT_AGENT_MODEL
        self.frontier_strategy = frontier_strategy
        self.plugin_registry = plugin_registry
        self.task_project_path: Path | None = (
            Path(task_project_path) if task_project_path is not None else None
        )
        self.peer_role_ref = peer_role_ref
        self.peer_role_skill = peer_role_skill
        self.peer_role_skills = tuple(peer_role_skills)
        self.runtime_ref: str = runtime_ref or ""
        self.resume = bool(resume)
        self.resume_policy = resume_policy
        self.run_lifecycle_observer = run_lifecycle_observer
        self._task_lifecycle: TaskLifecycle | None = None
        self._lifecycle_phase = ""

        gp = task_spec.generation_policy

        # Run directory for tracking. Startup is responsible for selecting a
        # task-local external run_dir before the workflow stage is constructed.
        if not run_dir:
            raise ValueError("GenerationLoop requires an explicit external run_dir")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot task_spec into the run dir so downstream tools (graph
        # viz, dashboard, post-run analysis scripts) have a self-contained,
        # stable reference to primary_metric / direction / aux_metrics /
        # baselines without reaching back into tasks/<id>/ — where the
        # file may have been edited between runs. This is the canonical
        # answer to "which task was this run actually configured for?"
        try:
            import yaml as _yaml

            with open(self.run_dir / "task_spec.yaml", "w") as _f:
                _yaml.safe_dump(
                    task_spec._raw,
                    _f,
                    sort_keys=False,
                    allow_unicode=True,
                )
        except Exception as _e:
            logger.warning(f"failed to snapshot task_spec into run_dir: {_e}")

        # Per-run output directories
        self.results_dir = self.run_dir / "results"
        self.variants_dir = self.run_dir / "variants"
        self.findings_dir = self.run_dir / "shared_findings"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.variants_dir.mkdir(parents=True, exist_ok=True)
        self.findings_dir.mkdir(parents=True, exist_ok=True)

        risk_frontier_cfg = {}
        raw_task_spec = getattr(task_spec, "_raw", {}) or {}
        self.dig_lite_config = DIGLiteConfig.from_raw(
            raw_task_spec.get("dig_lite") if isinstance(raw_task_spec, dict) else None,
            scoreless=is_scoreless(task_spec),
        )
        self.quality_diversity_config = QualityDiversityConfig.from_task_spec(
            raw_task_spec, dig_config=self.dig_lite_config
        )
        self.dig_lite_config.cohort_qd = self.quality_diversity_config.cohort
        if isinstance(raw_task_spec, dict):
            raw_risk_cfg = raw_task_spec.get("risk_violating_frontier") or {}
            if isinstance(raw_risk_cfg, dict):
                risk_frontier_cfg = raw_risk_cfg
        risk_frontier_threshold = risk_frontier_cfg.get("min_primary_metric")
        try:
            risk_frontier_threshold = (
                float(risk_frontier_threshold) if risk_frontier_threshold is not None else None
            )
        except (TypeError, ValueError):
            logger.warning(
                "risk_violating_frontier.min_primary_metric=%r is not numeric; "
                "risk-violating repair candidates disabled",
                risk_frontier_threshold,
            )
            risk_frontier_threshold = None

        # Task-normalized secondary metrics retain multi-axis frontier behavior.
        result_options = result_artifact_options_from_task_spec(task_spec)
        self.frontier = FrontierStore(
            base_dir=self.run_dir / "frontier",
            promote_top_k=gp.promote_top_k,
            primary_metric=task_spec.evaluation.primary_metric,
            metric_direction=task_spec.evaluation.direction,
            anchor_metrics=getattr(task_spec.evaluation, "anchor_metrics", None) or [],
            frontier_lanes=getattr(task_spec.evaluation, "frontier_lanes", None) or [],
            validation_signal_metrics=result_options.get("result_scoring_metric_keys") or [],
            result_cell_metric_derivations=result_options.get("result_cell_metric_derivations")
            or [],
            result_metric_aliases=result_options.get("result_metric_aliases") or {},
            require_tier=getattr(task_spec.evaluation, "requires_tier", False),
            maturity_policy=getattr(task_spec.evaluation, "maturity_policy", None),
            risk_violating_frontier_enabled=bool(risk_frontier_cfg.get("enabled", False)),
            risk_violating_primary_threshold=risk_frontier_threshold,
        )
        self.gems = GemsManager(
            run_dir=self.run_dir,
            task_spec=self.task_spec,
            frontier=self.frontier,
            local_mode=self.local_mode,
        )

        # Prompt templates
        loop_dir = Path(__file__).parent
        default_base_template = loop_dir / "prompt_base.jinja2"
        base_template_resolver = getattr(task_spec, "get_prompt_base_path", None)
        if callable(base_template_resolver):
            self.base_template = base_template_resolver(default_base_template)
        else:
            self.base_template = default_base_template
        default_generation_template = loop_dir / "prompt_generation.jinja2"
        generation_template_resolver = getattr(task_spec, "get_prompt_generation_path", None)
        if callable(generation_template_resolver):
            self.gen_template = generation_template_resolver(default_generation_template)
        else:
            self.gen_template = default_generation_template
        self.task_prompt_path = task_spec.get_prompt_task_path()

        mp_cfg = getattr(self.task_spec, "multi_pi", None)
        self.tool_server_refs = tuple(tool_server_refs or DEFAULT_RESEARCH_TOOL_SERVER_REFS)
        tool_server_build = build_legacy_mcp_servers(
            self.tool_server_refs,
            run_dir=self.run_dir,
            local_mode=self.local_mode,
            multi_pi_enabled=bool(mp_cfg is not None and getattr(mp_cfg, "enabled", False)),
            registry=plugin_registry,
        )
        self.mcp_servers: dict[str, Any] = tool_server_build.servers
        self.tool_server_registry_refs = tool_server_build.refs
        for item in tool_server_build.unavailable:
            logger.warning(
                "MCP %s server unavailable: %s",
                item["server_name"],
                item["reason"],
            )
        for item in tool_server_build.skipped:
            logger.debug(
                "MCP %s server skipped: %s",
                item["server_name"],
                item["reason"],
            )
        self._peer_mcp_servers, self._peer_allowed_tools = peer_mcp_context(
            self.mcp_servers,
            tool_refs=self.tool_server_refs,
            registry=plugin_registry,
        )
        logger.info("MCP servers registered: %s", sorted(self.mcp_servers.keys()))

        # Orchestrator status writer wiring — set in run().
        # The status writer daemon and the main async loop both touch
        # `_current_generation` / `_generations_completed`. Python int
        # assignment is GIL-atomic, but *pair* reads can observe a
        # logically-inconsistent pair mid-transition (e.g. "gen 0 running,
        # 1 completed"). `_state_lock` guards pair updates and pair reads.
        self._status_writer: Any | None = None
        self._run_started_at: str | None = None
        self._generations_completed: int = 0
        self._current_generation: int = 0
        self._state_lock = threading.Lock()

        # Bidirectional filesystem↔SQLite sync daemon — started in run().
        self._findings_sync = None
        self._boundary_evidence_cutoff: tuple[int, datetime, dict[str, str]] | None = None
        # Finding Graph maintainer (sidecar index over shared_findings).
        # Advisory, no behavioral change if disabled.
        self._graph_maintainer = None

        # Issue #83 + #85: resolve the panel topology's optional peer-role
        # rotation (Gen 0 contract injection) and peer-role descriptions
        # (peer prompt template's role explanation block) once at construction
        # time. Both fall back to their historical defaults — empty rotation
        # and bundled five-bullet vocab — when the topology doesn't declare
        # them or resolution fails. ``panel_topology_ref`` defaults to the
        # legacy ref when the TaskSpec didn't carry an override, so legacy
        # tasks still get the same plugin manifest they always did.
        self._panel_topology_ref: str = (
            getattr(task_spec, "panel_topology_ref", "") or _DEFAULT_TOPOLOGY_REF
        )
        rotation, descriptions = _resolve_topology_peer_info(
            plugin_registry, topology_ref=self._panel_topology_ref
        )
        self._peer_role_rotation: tuple[str, ...] = rotation
        self._peer_role_descriptions: dict[str, str] = descriptions
        self._peer_role_selector = PeerRoleSelector(
            run_dir=self.run_dir,
            task_spec=self.task_spec,
            role_rotation=rotation,
            default_role_skill=self.peer_role_skill,
            role_skills=self.peer_role_skills,
        )
        self._topology_executor = LegacyResearchTopologyExecutor(
            lambda: run_generation_cohort,
            resolve_each_call=True,
        )

        logger.info(
            f"GenerationLoop initialized: task={task_spec.task_id}, "
            f"max_gen={gp.max_generations}, cohort={gp.cohort_size}, "
            f"strategy={frontier_strategy}"
        )

    def _build_prompt_context(
        self, gen_id: int, peer_index: int, cohort_size: int
    ) -> dict[str, Any]:
        """Build template context for prompt rendering."""
        return build_prompt_context(
            task_spec=self.task_spec,
            workspace=self.workspace,
            run_dir=self.run_dir,
            results_dir=self.results_dir,
            variants_dir=self.variants_dir,
            findings_dir=self.findings_dir,
            frontier=self.frontier,
            local_mode=self.local_mode,
            gen_id=gen_id,
            peer_index=peer_index,
            cohort_size=cohort_size,
            strategy=self._strategy_for_gen(gen_id),
            peer_role_rotation=self._peer_role_rotation,
            peer_role_descriptions=self._peer_role_descriptions,
            available_tool_server_names=self.mcp_servers,
            runtime_ref=self.runtime_ref,
            gems_context=self.gems.prompt_context(
                gen_id,
                peer_index=peer_index,
                cohort_size=cohort_size,
            ),
            logical_gen_id=self.gems.logical_generation(gen_id),
        )

    def peer_role_skill_for_context(self, context: dict[str, Any]) -> RoleSkill | None:
        return self._peer_role_selector.skill_for_context(context)

    def peer_role_ref_for_context(self, context: dict[str, Any]) -> str | None:
        return self._peer_role_selector.ref_for_context(context)

    def peer_role_ref_for(self, gen_id: int, peer_index: int) -> str | None:
        return self._peer_role_selector.ref_for_peer(gen_id, peer_index)

    def _strategy_for_gen(self, gen_id: int) -> str:
        """v2026-05-04: 3-cycle annealing is REPLACED by event-driven
        synthesis (PI agent assigns roles dynamically).

        Returned values:
          - gen 0: "explore" — uses must_explore_axes for initial diversity
            seeding (PI has no prior data to synthesize from yet).
          - gen ≥ 1: "pi_directed" — peers get role contracts from the PI
            agenda; the legacy explore/mixed/exploit phase no longer applies.

        Explicit override (frontier_strategy != "auto") still respected
        for backwards compatibility / debugging.
        """
        if self.frontier_strategy != "auto":
            return self.frontier_strategy
        logical_gen_id = self.gems.logical_generation(gen_id)
        if logical_gen_id == 0:
            return "explore"
        return "pi_directed"

    def _persist_prompt_layout_artifacts(
        self,
        *,
        prompt_text: str,
        prompt_path: Path,
        manifest: dict[str, Any],
        manifest_path: Path,
        peer_id: str,
        gen_id: int,
    ) -> dict[str, Any]:
        """Persist PromptLayout V1 artifacts on a best-effort basis."""
        return persist_prompt_layout_artifacts(
            run_dir=self.run_dir,
            prompt_text=prompt_text,
            prompt_path=prompt_path,
            manifest=manifest,
            manifest_path=manifest_path,
            peer_id=peer_id,
            gen_id=gen_id,
        )

    async def _run_generation(self, gen_id: int) -> list[dict[str, Any]]:
        """Run a single generation cohort and persist generation results."""
        return await self._topology_executor.execute_generation(self, gen_id)

    def _check_plateau(self) -> bool:
        """Retained compatibility stub; generation count is budget-driven."""
        return False

    async def run(self) -> dict[str, Any]:
        """Run all phases with the startup-resolved task execution policy."""
        with task_execution_policy_scope(getattr(self.task_spec, "execution_policy", {})):
            return await self._run_research_loop()

    # Bound delegates preserve the lifecycle entrypoints used by callbacks.
    _run_lifecycle_agent = task_execution.run_lifecycle_agent
    _check_task_stop = task_execution.check_task_stop
    _frozen_findings_through = task_execution.frozen_findings_through
    _run_task_generation_review = task_execution.run_task_generation_review

    async def _run_research_loop(self) -> dict[str, Any]:
        """Run research and optional delivery while holding lifecycle authority."""
        gp = self.task_spec.generation_policy
        start_time = time.time()
        resume_plan = None
        task_delivery: dict[str, Any] | None = None

        runtime_scope = enter_orchestrator_runtime_scope(
            run_dir=self.run_dir,
            resume=self.resume,
            logger=logger,
        )
        try:
            start_time = task_execution.initialize_task_lifecycle(
                self,
                start_time,
                configure_runtime_environment,
                initialize_local_store_if_needed,
                validate_baseline_cache_for_run,
                logger,
            )

            if self.resume:
                prime_resume_boundary_evidence_cutoff(self, max_generations=gp.max_generations)
                resume_plan = prepare_resume_for_sidecars(
                    self.run_dir,
                    max_generations=gp.max_generations,
                    pi_enabled=bool(self.task_spec.pi_agent.enabled),
                    policy=self.resume_policy,
                )
            start_sidecars(self, resume_plan=resume_plan)
        except BaseException:
            close_sidecars_and_runtime(self, "error", runtime_scope)
            raise

        all_results: list[list[dict[str, Any]]] = []
        run_stop_report: dict[str, Any] | None = None
        # Default to "error" so any non-Exception termination
        # (asyncio.CancelledError, KeyboardInterrupt, SystemExit — all
        # BaseException subclasses that our except does NOT catch) ends up
        # with the right status label. Success paths explicitly set
        # a lifecycle exit condition before the finally fires.
        exit_condition = "error"

        try:
            # v2026-05-04: PI agent runs between every pair of gens.
            # Lazy import to avoid module-load cycles.
            from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent

            pi_cfg = self.task_spec.pi_agent
            pi_agent = None
            if pi_cfg.enabled:
                # v2026-05-05: route to Multi-PI panel when configured.
                # Default config has multi_pi.enabled=False → identical to
                # v2026-05-04 single-PI behavior.
                mp_cfg = getattr(self.task_spec, "multi_pi", None)
                rm_cfg = getattr(self.task_spec, "research_memory", None)
                use_panel = bool(mp_cfg and getattr(mp_cfg, "enabled", False))
                if use_panel:
                    logger.info(
                        "PI: using Multi-PI panel (panel_mode_default=%s, auto_escalate=%s)",
                        getattr(mp_cfg, "panel_mode_default", "full"),
                        getattr(mp_cfg, "auto_escalate_to_high_stakes", True),
                    )
                pi_agent = PIAgent(
                    task_spec=self.task_spec,
                    run_dir=self.run_dir,
                    workspace=self.workspace,
                    cohort_size=gp.cohort_size,
                    model=self.model,
                    max_runtime_minutes=pi_cfg.max_runtime_minutes,
                    strict=pi_cfg.strict,
                    mcp_servers=self.mcp_servers,
                    local_mode=self.local_mode,
                    use_multi_pi_panel=use_panel,
                    multi_pi_config=mp_cfg,
                    research_memory_config=rm_cfg,
                    premium_mode=self.task_spec.agent.premium_mode,
                    reasoning_effort=str(getattr(self.task_spec.agent, "reasoning_effort", "max")),
                    task_project_path=self.task_project_path,
                    panel_topology_ref=self._panel_topology_ref,
                    plugin_registry=self.plugin_registry,
                    peer_role_rotation=self._peer_role_rotation,
                    local_store_dir=self.run_dir,
                    quality_diversity_config=self.quality_diversity_config,
                    diversity_dimensions=(
                        getattr(self.task_spec.evaluation, "diversity_dimensions", None) or []
                    ),
                )

            await task_execution.run_initial_task_phase(self, start_time)
            start_generation = 0
            if self.resume:
                assert resume_plan is not None
                append_resume_event(
                    self.run_dir,
                    {
                        "event": "resume_plan",
                        "plan": resume_plan.to_dict(),
                    },
                )
                for warning in resume_plan.warnings:
                    logger.warning("resume: %s", warning)
                logger.info("resume plan: %s", resume_plan.to_dict())
                for repair in repair_inferred_gems_boundary_markers(
                    self,
                    max_generations=gp.max_generations,
                    pi_enabled=bool(pi_cfg.enabled),
                ):
                    append_resume_event(
                        self.run_dir,
                        {
                            "event": "inferred_boundary_marker_repaired",
                            **repair,
                        },
                    )

                for prior_gen in range(resume_plan.completed_generations):
                    all_results.append(load_generation_results(self.run_dir, prior_gen))
                with self._state_lock:
                    self._generations_completed = len(all_results)
                    self._current_generation = max(0, len(all_results) - 1)

                if resume_plan.has_pending_boundary:
                    raw_pending_gen = resume_plan.pending_boundary_generation
                    if raw_pending_gen is None:
                        raise RuntimeError("resume plan reported pending boundary without gen id")
                    pending_gen = int(raw_pending_gen)
                    logger.info(
                        "resume: completing pending generation boundary for gen %d",
                        pending_gen,
                    )
                    if pending_gen >= len(all_results):
                        all_results.append(load_generation_results(self.run_dir, pending_gen))
                    recovered_gems = recover_pending_gems_reset_for_resume(
                        self,
                        pending_gen=pending_gen,
                    )
                    if recovered_gems.triggered:
                        logger.warning(
                            "resume: recovered pending Gems reset for gen %d "
                            "(reset_count=%d, admitted=%d)",
                            pending_gen,
                            recovered_gems.reset_count,
                            recovered_gems.admitted_count,
                        )
                        record_completed_generation_observation(
                            self,
                            gen_id=pending_gen,
                            generation_results=all_results[pending_gen],
                        )
                    else:
                        await task_execution.complete_boundary_with_deadline(
                            self,
                            pending_gen,
                            pi_agent,
                            pi_cfg,
                            all_results[pending_gen],
                            complete_generation_boundary,
                        )
                    with self._state_lock:
                        self._generations_completed = max(
                            self._generations_completed,
                            pending_gen + 1,
                        )
                        self._current_generation = pending_gen
                    start_generation = pending_gen + 1
                else:
                    start_generation = resume_plan.start_generation

            start_generation = await task_execution.review_completed_generations(
                self, start_time, len(all_results), start_generation
            )
            for gen_id in task_execution.generation_indices(self, start_generation):
                stop_exit, stop_report = task_execution.check_generation_start(
                    self, start_time, gen_id, len(all_results)
                )
                if stop_exit:
                    exit_condition, run_stop_report = stop_exit, stop_report
                    break
                with self._state_lock:
                    # Round 4 M5 fix: atomize the (current, completed)
                    # pair update. Previously two separate lock
                    # acquisitions allowed a status writer thread to
                    # observe the inconsistent pair (current=N+1,
                    # completed=N) between the two assignments. Now
                    # both fields are updated together at the start of
                    # each iteration AND together at the end.
                    self._current_generation = gen_id
                gen_results, deadline_reached = await task_execution.run_generation_with_deadline(
                    self, gen_id
                )
                all_results.append(gen_results)
                await task_execution.complete_boundary_with_deadline(
                    self, gen_id, pi_agent, pi_cfg, gen_results, complete_generation_boundary
                )
                with self._state_lock:
                    self._generations_completed = canonical_completed_generation_count(self.run_dir)
                generate_loop_boundary_report(self, generation_id=gen_id)
                await task_execution.review_generation_if_enabled(
                    self, gen_id, start_time, len(all_results)
                )
                if deadline_reached:
                    exit_condition = "research_deadline"
                    break

                # No plateau check in v2026-05-04 — the run goes the full
                # max_generations, and the final cohort's frontier is the
                # output. Plateau detection was removed because it only
                # watched the primary axis; multi-axis frontier progress
                # was being thrown away.
            else:
                run_stop_report = task_execution.generation_limit_report(
                    self, start_time, len(all_results)
                )
                exit_condition = str(run_stop_report["exit_condition"])
            task_delivery = await task_execution.run_final_task_phase(
                self, start_time, len(all_results)
            )
        except task_execution.TaskDeliveryIncomplete as exc:
            task_delivery = exc.result
            exit_condition = str(task_delivery.get("exit_condition") or "delivery_incomplete")
            if isinstance(task_delivery.get("run_stop_report"), dict):
                run_stop_report = task_delivery["run_stop_report"]
        except (KeyboardInterrupt, SystemExit) as _stop_exc:
            stop_decision = evaluate_run_stop_gate(
                task_spec=self.task_spec,
                run_dir=self.run_dir,
                run_started_at_seconds=start_time,
                next_generation=self._current_generation,
                generations_completed=len(all_results),
            )
            if stop_decision.should_stop:
                run_stop_report = write_run_stop_report(self.run_dir, stop_decision)
                exit_condition = stop_decision.exit_condition
            else:
                exit_condition = "interrupted"
            try:
                write_run_summary(
                    self.run_dir / "run_summary.json",
                    {
                        "task_id": self.task_spec.task_id,
                        "task_name": self.task_spec.task_name,
                        "generations_completed": canonical_completed_generation_count(self.run_dir),
                        "max_generations": gp.max_generations,
                        "exit_condition": exit_condition,
                        "error_type": type(_stop_exc).__name__,
                        "error_message": str(_stop_exc)[:500],
                        "run_dir": str(self.run_dir),
                        "frontier_summary": self.frontier.get_summary(),
                        "run_stop_report": run_stop_report,
                        "stop_signal_evidence": (
                            run_stop_report.get("signal_evidence") if run_stop_report else None
                        ),
                        "gems": self.gems.load_state() if self.gems.enabled else {},
                    },
                )
                logger.warning(
                    "run_summary.json written under exit_condition=%s after orchestrator interruption",
                    exit_condition,
                )
            except Exception as _write_exc:
                logger.error(
                    "could not write interrupted run_summary.json: %s",
                    _write_exc,
                )
            raise
        except Exception as _run_exc:
            exit_condition = "error"
            # R9-H2 fix: write a degenerate run_summary.json BEFORE
            # re-raising, so downstream tooling (deliver.py, dashboards)
            # always sees an end-of-run record even on partial failure.
            try:
                write_run_summary(
                    self.run_dir / "run_summary.json",
                    {
                        "task_id": self.task_spec.task_id,
                        "task_name": self.task_spec.task_name,
                        "generations_completed": canonical_completed_generation_count(self.run_dir),
                        "max_generations": gp.max_generations,
                        "exit_condition": exit_condition,
                        "error_type": type(_run_exc).__name__,
                        "error_message": str(_run_exc)[:500],
                        "run_dir": str(self.run_dir),
                        "frontier_summary": self.frontier.get_summary(),
                        "gems": self.gems.load_state() if self.gems.enabled else {},
                    },
                )
                logger.warning(
                    "run_summary.json written under exit_condition=error "
                    "after exception in main loop"
                )
            except Exception as _write_exc:
                logger.error(
                    "could not write degenerate run_summary.json on error path: %s",
                    _write_exc,
                )
            raise
        finally:
            close_sidecars_and_runtime(self, exit_condition, runtime_scope)

        total_duration = time.time() - start_time
        stop_reason = run_stop_report.get("reason") if run_stop_report else None
        stop_next_generation = run_stop_report.get("next_generation") if run_stop_report else None
        stop_elapsed_seconds = run_stop_report.get("elapsed_seconds") if run_stop_report else None
        stop_signal_evidence = run_stop_report.get("signal_evidence") if run_stop_report else None

        # R4-M5 fix: enrich the final summary so downstream tooling and
        # human readers can answer "what did the last gen actually
        # contribute?" without having to scrape gen_<N>/ subdirs.
        committed_generations = canonical_completed_generation_count(self.run_dir)
        last_gen_idx = committed_generations - 1
        last_gen_findings_count = 0
        last_gen_promoted_count = 0
        last_gen_pi_skipped = False
        last_gen_pi_reason = ""
        if last_gen_idx >= 0:
            try:
                last_gen_findings = self._collect_findings_for_generation(
                    last_gen_idx,
                    do_ingest=False,
                )
                last_gen_findings_count = len(last_gen_findings)
                last_gen_promoted = (
                    self.frontier.get_summary_for_generation(last_gen_idx)
                    if hasattr(self.frontier, "get_summary_for_generation")
                    else []
                )
                last_gen_promoted_count = len(last_gen_promoted) if last_gen_promoted else 0
            except Exception as e:
                logger.debug("final summary: last-gen stats failed: %s", e)
            if gp.max_generations is not None and last_gen_idx == gp.max_generations - 1:
                last_gen_pi_skipped = True
                last_gen_pi_reason = "is_last_gen (no successor to plan for)"

        summary = {
            "task_id": self.task_spec.task_id,
            "task_name": self.task_spec.task_name,
            "generations_completed": committed_generations,
            "max_generations": gp.max_generations,
            "total_duration_seconds": total_duration,
            "frontier_summary": self.frontier.get_summary(),
            "run_dir": str(self.run_dir),
            "exit_condition": exit_condition,
            "stop_reason": stop_reason,
            "stop_next_generation": stop_next_generation,
            "stop_elapsed_seconds": stop_elapsed_seconds,
            "stop_signal_evidence": stop_signal_evidence,
            "run_stop_report": run_stop_report,
            "last_gen_index": last_gen_idx,
            "last_gen_findings_count": last_gen_findings_count,
            "last_gen_promoted_count": last_gen_promoted_count,
            "last_gen_pi_skipped": last_gen_pi_skipped,
            "last_gen_pi_skip_reason": last_gen_pi_reason,
            "gems": self.gems.load_state() if self.gems.enabled else {},
        }

        task_execution.add_task_delivery_summary(self, summary, task_delivery)
        write_run_summary(self.run_dir / "run_summary.json", summary)
        generate_loop_boundary_report(self, generation_id=last_gen_idx, final=True)

        logger.info(
            f"\n{'#' * 60}\n"
            f"# Run complete: {committed_generations} generations, "
            f"{total_duration / 3600:.1f}h total, exit={exit_condition}\n"
            f"{'#' * 60}"
        )

        return summary

    # ------------------------------------------------------------------
    # Orchestrator status snapshot (I-8)
    # ------------------------------------------------------------------

    def _build_status_snapshot(self) -> OrchestratorSnapshot:
        """Compose the orchestrator status snapshot for the status writer."""
        with self._state_lock:
            current_gen = self._current_generation
            gens_completed = canonical_completed_generation_count(self.run_dir)

        try:
            findings = self._collect_findings_for_generation(current_gen, do_ingest=False)
        except Exception:
            findings = []

        return build_orchestrator_status_snapshot(
            run_started_at=self._run_started_at,
            run_dir=self.run_dir,
            task_spec=self.task_spec,
            frontier=self.frontier,
            current_gen=current_gen,
            gens_completed=gens_completed,
            frontier_strategy=self.frontier_strategy,
            strategy_for_gen=self._strategy_for_gen,
            findings=findings,
            gems_context=self.gems.prompt_context(current_gen),
        )

    def _update_research_memory_post_gen(
        self,
        gen_id: int,
        findings: list[dict[str, Any]],
        promoted: list[Any],
    ) -> None:
        """Update research_memory ledgers with this generation's outputs."""
        update_research_memory_post_gen(
            run_dir=self.run_dir,
            gen_id=gen_id,
            findings=findings,
            promoted=promoted,
            evaluation=self.task_spec.evaluation,
        )

    def _collect_findings_for_generation(
        self,
        gen_id: int,
        *,
        do_ingest: bool = True,
    ) -> list[dict[str, Any]]:
        """Collect all findings for a generation from the unified store."""
        return collect_loop_findings(self, gen_id, do_ingest=do_ingest)

    def _collect_findings_for_boundary(
        self,
        gen_id: int,
        *,
        evidence_cutoff: datetime,
        evidence_source_snapshot: dict[str, str],
    ) -> list[dict[str, Any]]:
        self._boundary_evidence_cutoff = (
            int(gen_id),
            evidence_cutoff,
            dict(evidence_source_snapshot),
        )
        return self._collect_findings_for_generation(gen_id)
