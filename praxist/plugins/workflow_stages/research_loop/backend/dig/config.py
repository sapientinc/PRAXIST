"""Configuration objects for DIG-Lite."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _bool_value(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in {"1", "true", "yes", "on", "y"}:
            return True
        if token in {"0", "false", "no", "off", "n"}:
            return False
    return default


def _int_value(raw: Any, default: int, minimum: int | None = None) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        return max(minimum, value)
    return value


def _optional_limit(raw: Any, default: int, minimum: int, *, scoreless: bool) -> int | None:
    if scoreless and raw is None:
        return None
    return _int_value(raw, default, minimum)


def _float_value(raw: Any, default: float, minimum: float | None = None) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(default)
    if minimum is not None:
        return max(minimum, value)
    return value


def _str_list(raw: Any, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    values = raw if isinstance(raw, list) else [raw]
    out: list[str] = []
    for item in values:
        try:
            text = str(item).strip()
        except (TypeError, ValueError):
            continue
        if text:
            out.append(text)
    return out


def _dict_list(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _str_dict(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        key_text = str(key or "").strip()
        value_text = str(value or "").strip()
        if key_text and value_text:
            out[key_text] = value_text
    return out


@dataclass
class DIGCritiqueConfig:
    """Scoring lenses used by the DIG candidate critique phase."""

    scale: str = "1_to_5"
    lenses: list[str] = field(
        default_factory=lambda: [
            "mechanism_plausibility",
            "implementability",
            "diagnostic_clarity",
            "diversity_value",
            "shortcut_risk",
            "silent_bug_risk",
            "compute_risk",
        ]
    )


@dataclass
class DIGDiversityConfig:
    """Quality-diversity selection settings for DIG candidates."""

    cell_fields: list[str] = field(
        default_factory=lambda: ["mechanism_family", "intervention_surface", "intent"]
    )
    selection: str = "best_within_lane"
    reject_near_duplicate: bool = True
    duplicate_threshold: float = 0.82
    allow_adjacent_lane_fallback: bool = True


@dataclass
class DIGInnovationConfig:
    """Cohort-level guardrails that keep DIG from over-selecting diagnostics.

    DIG candidate pools should contain falsifiers, but most peer contracts in a
    broad research generation should still move an implementation mechanism
    forward.  These settings are deliberately generic: tasks can override the
    intent/family labels without changing core code.
    """

    enabled: bool = True
    enforce_forward_slots: bool = True
    max_diagnostic_fraction: float = 0.20
    max_diagnostic_peers: int = 2
    diagnostic_score_penalty: float = 4.0
    forward_intents: list[str] = field(
        default_factory=lambda: ["exploit", "explore", "bridge", "repair", "anti_mainline"]
    )
    diagnostic_intents: list[str] = field(
        default_factory=lambda: ["falsify", "diagnose", "ablate", "audit"]
    )
    diagnostic_mechanism_families: list[str] = field(
        default_factory=lambda: [
            "diagnostic_falsifier",
            "diagnostic",
            "falsifier",
            "ablation",
        ]
    )


@dataclass
class DIGCohortQDTargetGroup:
    """Task-owned soft target group used by the cohort-level QD allocator."""

    name: str
    min_peers: int = 0
    fields: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class DIGCohortQDConfig:
    """Cohort-level quality-diversity allocation settings.

    This layer stays generic.  Task projects can define target keyword groups,
    but Praxist core does not hard-code domain metrics or algorithm names.
    """

    enabled: bool = True
    min_distinct_mechanism_families: int = 0
    min_distinct_intervention_surfaces: int = 0
    max_same_diversity_cell_peers: int = 1
    max_same_mechanism_family_peers: int = 0
    max_same_mechanism_family_fraction: float = 0.34
    max_same_intervention_surface_peers: int = 0
    max_same_intervention_surface_fraction: float = 0.50
    max_same_intent_peers: int = 0
    max_same_intent_fraction: float = 0.60
    max_same_semantic_family_peers: int = 0
    max_same_semantic_family_fraction: float = 0.34
    max_same_parent_lineage_peers: int = 0
    max_same_parent_lineage_fraction: float = 0.50
    quality_weight: float = 1.0
    novelty_weight: float = 1.5
    lane_fit_bonus: float = 1.0
    local_selection_bonus: float = 0.25
    risk_penalty_weight: float = 0.5
    target_keyword_bonus: float = 2.0
    target_keyword_groups: list[DIGCohortQDTargetGroup] = field(default_factory=list)
    semantic_label_groups: list[DIGCohortQDTargetGroup] = field(default_factory=list)
    parent_lineage_label_groups: list[DIGCohortQDTargetGroup] = field(default_factory=list)
    novelty_axis_label_groups: list[DIGCohortQDTargetGroup] = field(default_factory=list)
    label_synonyms: dict[str, str] = field(default_factory=dict)


@dataclass
class DIGContractConfig:
    """Validation requirements for the locked DIG implementation contract."""

    require_selected_contract: bool = True
    require_ablation_hooks: bool = True
    min_ablation_hooks: int = 1
    min_rejected_alternatives: int = 3
    require_forbidden_changes: bool = True
    require_expected_metric_signature: bool = True
    require_fail_fast_checks: bool = True


@dataclass
class DIGWriteGateConfig:
    """Write and shell restrictions applied before DIG unlocks implementation."""

    enabled: bool = True
    allow_writes_only_under_dig_dir_before_unlock: bool = True
    block_shell_before_unlock: bool = True
    block_variant_dir_before_unlock: bool = True
    block_results_dir_before_unlock: bool = True


@dataclass
class DIGLiteConfig:
    """Top-level DIG-Lite configuration loaded from task-spec raw config."""

    enabled: bool = True
    generation_scope: str = "initial_only"
    phase: str = "pre_code_only"
    experiments_allowed: bool = False
    shell_allowed: bool = False
    code_writes_allowed: bool = False
    result_writes_allowed: bool = False
    candidate_count: int = 8
    min_mechanism_families: int = 4
    min_intervention_surfaces: int = 3
    max_refinement_rounds: int | None = 1
    planner_allowed_tools: list[str] = field(default_factory=lambda: ["Read", "Grep", "Glob"])
    planner_permission_mode: str = "acceptEdits"
    planner_max_runtime_minutes: int | None = 10
    attempt_max_runtime_minutes: int | None = 0
    allowed_file_rules: list[str] = field(default_factory=list)
    disallowed_file_rules: list[str] = field(default_factory=list)
    max_total_runtime_minutes: int | None = 40
    max_attempts: int | None = 10
    fallback_to_direct_on_failure: bool = True
    strict: bool = True
    inject_contract_into_prompt: bool = True
    critique: DIGCritiqueConfig = field(default_factory=DIGCritiqueConfig)
    diversity: DIGDiversityConfig = field(default_factory=DIGDiversityConfig)
    innovation: DIGInnovationConfig = field(default_factory=DIGInnovationConfig)
    cohort_qd: DIGCohortQDConfig = field(default_factory=DIGCohortQDConfig)
    contract: DIGContractConfig = field(default_factory=DIGContractConfig)
    write_gate: DIGWriteGateConfig = field(default_factory=DIGWriteGateConfig)

    def enabled_for_generation(self, generation_id: int) -> bool:
        """Return whether DIG runs for an absolute generation id."""

        if not self.enabled:
            return False
        return self.generation_scope == "all" or int(generation_id) == 0

    @classmethod
    def from_raw(cls, raw: Any, *, scoreless: bool = False) -> DIGLiteConfig:
        """Parse DIG settings, leaving omitted scoreless limits unbounded.

        Args:
            raw: Task-local DIG configuration mapping.
            scoreless: Whether absent or null execution limits mean no cap.

        Returns:
            Normalized configuration with legacy metric defaults preserved.
        """
        if not isinstance(raw, dict):
            raw = {"enabled": False}

        critique_raw = raw.get("critique") if isinstance(raw.get("critique"), dict) else {}
        diversity_raw = raw.get("diversity") if isinstance(raw.get("diversity"), dict) else {}
        innovation_raw = raw.get("innovation") if isinstance(raw.get("innovation"), dict) else {}
        cohort_qd_raw = raw.get("cohort_qd") if isinstance(raw.get("cohort_qd"), dict) else {}
        contract_raw = raw.get("contract") if isinstance(raw.get("contract"), dict) else {}
        write_gate_raw = raw.get("write_gate") if isinstance(raw.get("write_gate"), dict) else {}

        defaults = cls()
        generation_scope = str(raw.get("generation_scope") or defaults.generation_scope)
        generation_scope = generation_scope.strip().lower().replace("-", "_")
        generation_scope = {
            "initial": "initial_only",
            "gen0": "initial_only",
            "gen_0": "initial_only",
            "first": "initial_only",
            "all_generations": "all",
        }.get(generation_scope, generation_scope)
        if generation_scope not in {"initial_only", "all"}:
            generation_scope = defaults.generation_scope
        return cls(
            enabled=_bool_value(raw.get("enabled"), defaults.enabled),
            generation_scope=generation_scope,
            phase=str(raw.get("phase") or defaults.phase),
            experiments_allowed=_bool_value(
                raw.get("experiments_allowed"), defaults.experiments_allowed
            ),
            shell_allowed=_bool_value(raw.get("shell_allowed"), defaults.shell_allowed),
            code_writes_allowed=_bool_value(
                raw.get("code_writes_allowed"), defaults.code_writes_allowed
            ),
            result_writes_allowed=_bool_value(
                raw.get("result_writes_allowed"), defaults.result_writes_allowed
            ),
            candidate_count=_int_value(raw.get("candidate_count"), defaults.candidate_count, 1),
            min_mechanism_families=_int_value(
                raw.get("min_mechanism_families"), defaults.min_mechanism_families, 1
            ),
            min_intervention_surfaces=_int_value(
                raw.get("min_intervention_surfaces"), defaults.min_intervention_surfaces, 1
            ),
            max_refinement_rounds=_optional_limit(
                raw.get("max_refinement_rounds"), 1, 0, scoreless=scoreless
            ),
            planner_allowed_tools=_str_list(
                raw.get("planner_allowed_tools"), defaults.planner_allowed_tools
            ),
            planner_permission_mode=str(
                raw.get("planner_permission_mode") or defaults.planner_permission_mode
            ),
            planner_max_runtime_minutes=_optional_limit(
                raw.get("planner_max_runtime_minutes"),
                10,
                1,
                scoreless=scoreless,
            ),
            attempt_max_runtime_minutes=_optional_limit(
                raw.get("attempt_max_runtime_minutes"),
                0,
                0,
                scoreless=scoreless,
            ),
            allowed_file_rules=_str_list(
                raw.get("allowed_file_rules"), defaults.allowed_file_rules
            ),
            disallowed_file_rules=_str_list(
                raw.get("disallowed_file_rules"), defaults.disallowed_file_rules
            ),
            max_total_runtime_minutes=_optional_limit(
                raw.get("max_total_runtime_minutes"),
                40,
                1,
                scoreless=scoreless,
            ),
            max_attempts=_optional_limit(raw.get("max_attempts"), 10, 1, scoreless=scoreless),
            fallback_to_direct_on_failure=_bool_value(
                raw.get("fallback_to_direct_on_failure"),
                defaults.fallback_to_direct_on_failure,
            ),
            strict=_bool_value(raw.get("strict"), defaults.strict),
            inject_contract_into_prompt=_bool_value(
                raw.get("inject_contract_into_prompt"), defaults.inject_contract_into_prompt
            ),
            critique=DIGCritiqueConfig(
                scale=str(critique_raw.get("scale") or defaults.critique.scale),
                lenses=_str_list(critique_raw.get("lenses"), defaults.critique.lenses),
            ),
            diversity=DIGDiversityConfig(
                cell_fields=_str_list(
                    diversity_raw.get("cell_fields"), defaults.diversity.cell_fields
                ),
                selection=str(diversity_raw.get("selection") or defaults.diversity.selection),
                reject_near_duplicate=_bool_value(
                    diversity_raw.get("reject_near_duplicate"),
                    defaults.diversity.reject_near_duplicate,
                ),
                duplicate_threshold=_float_value(
                    diversity_raw.get("duplicate_threshold"),
                    defaults.diversity.duplicate_threshold,
                    0.0,
                ),
                allow_adjacent_lane_fallback=_bool_value(
                    diversity_raw.get("allow_adjacent_lane_fallback"),
                    defaults.diversity.allow_adjacent_lane_fallback,
                ),
            ),
            innovation=DIGInnovationConfig(
                enabled=_bool_value(innovation_raw.get("enabled"), defaults.innovation.enabled),
                enforce_forward_slots=_bool_value(
                    innovation_raw.get("enforce_forward_slots"),
                    defaults.innovation.enforce_forward_slots,
                ),
                max_diagnostic_fraction=_float_value(
                    innovation_raw.get("max_diagnostic_fraction"),
                    defaults.innovation.max_diagnostic_fraction,
                    0.0,
                ),
                max_diagnostic_peers=_int_value(
                    innovation_raw.get("max_diagnostic_peers"),
                    defaults.innovation.max_diagnostic_peers,
                    0,
                ),
                diagnostic_score_penalty=_float_value(
                    innovation_raw.get("diagnostic_score_penalty"),
                    defaults.innovation.diagnostic_score_penalty,
                    0.0,
                ),
                forward_intents=_str_list(
                    innovation_raw.get("forward_intents"),
                    defaults.innovation.forward_intents,
                ),
                diagnostic_intents=_str_list(
                    innovation_raw.get("diagnostic_intents"),
                    defaults.innovation.diagnostic_intents,
                ),
                diagnostic_mechanism_families=_str_list(
                    innovation_raw.get("diagnostic_mechanism_families"),
                    defaults.innovation.diagnostic_mechanism_families,
                ),
            ),
            cohort_qd=DIGCohortQDConfig(
                enabled=_bool_value(cohort_qd_raw.get("enabled"), defaults.cohort_qd.enabled),
                min_distinct_mechanism_families=_int_value(
                    cohort_qd_raw.get("min_distinct_mechanism_families"),
                    defaults.cohort_qd.min_distinct_mechanism_families,
                    0,
                ),
                min_distinct_intervention_surfaces=_int_value(
                    cohort_qd_raw.get("min_distinct_intervention_surfaces"),
                    defaults.cohort_qd.min_distinct_intervention_surfaces,
                    0,
                ),
                max_same_diversity_cell_peers=_int_value(
                    cohort_qd_raw.get("max_same_diversity_cell_peers"),
                    defaults.cohort_qd.max_same_diversity_cell_peers,
                    0,
                ),
                max_same_mechanism_family_peers=_int_value(
                    cohort_qd_raw.get("max_same_mechanism_family_peers"),
                    defaults.cohort_qd.max_same_mechanism_family_peers,
                    0,
                ),
                max_same_mechanism_family_fraction=_float_value(
                    cohort_qd_raw.get("max_same_mechanism_family_fraction"),
                    defaults.cohort_qd.max_same_mechanism_family_fraction,
                    0.0,
                ),
                max_same_intervention_surface_peers=_int_value(
                    cohort_qd_raw.get("max_same_intervention_surface_peers"),
                    defaults.cohort_qd.max_same_intervention_surface_peers,
                    0,
                ),
                max_same_intervention_surface_fraction=_float_value(
                    cohort_qd_raw.get("max_same_intervention_surface_fraction"),
                    defaults.cohort_qd.max_same_intervention_surface_fraction,
                    0.0,
                ),
                max_same_intent_peers=_int_value(
                    cohort_qd_raw.get("max_same_intent_peers"),
                    defaults.cohort_qd.max_same_intent_peers,
                    0,
                ),
                max_same_intent_fraction=_float_value(
                    cohort_qd_raw.get("max_same_intent_fraction"),
                    defaults.cohort_qd.max_same_intent_fraction,
                    0.0,
                ),
                max_same_semantic_family_peers=_int_value(
                    cohort_qd_raw.get("max_same_semantic_family_peers"),
                    defaults.cohort_qd.max_same_semantic_family_peers,
                    0,
                ),
                max_same_semantic_family_fraction=_float_value(
                    cohort_qd_raw.get("max_same_semantic_family_fraction"),
                    defaults.cohort_qd.max_same_semantic_family_fraction,
                    0.0,
                ),
                max_same_parent_lineage_peers=_int_value(
                    cohort_qd_raw.get("max_same_parent_lineage_peers"),
                    defaults.cohort_qd.max_same_parent_lineage_peers,
                    0,
                ),
                max_same_parent_lineage_fraction=_float_value(
                    cohort_qd_raw.get("max_same_parent_lineage_fraction"),
                    defaults.cohort_qd.max_same_parent_lineage_fraction,
                    0.0,
                ),
                quality_weight=_float_value(
                    cohort_qd_raw.get("quality_weight"),
                    defaults.cohort_qd.quality_weight,
                    0.0,
                ),
                novelty_weight=_float_value(
                    cohort_qd_raw.get("novelty_weight"),
                    defaults.cohort_qd.novelty_weight,
                    0.0,
                ),
                lane_fit_bonus=_float_value(
                    cohort_qd_raw.get("lane_fit_bonus"),
                    defaults.cohort_qd.lane_fit_bonus,
                    0.0,
                ),
                local_selection_bonus=_float_value(
                    cohort_qd_raw.get("local_selection_bonus"),
                    defaults.cohort_qd.local_selection_bonus,
                    0.0,
                ),
                risk_penalty_weight=_float_value(
                    cohort_qd_raw.get("risk_penalty_weight"),
                    defaults.cohort_qd.risk_penalty_weight,
                    0.0,
                ),
                target_keyword_bonus=_float_value(
                    cohort_qd_raw.get("target_keyword_bonus"),
                    defaults.cohort_qd.target_keyword_bonus,
                    0.0,
                ),
                target_keyword_groups=[
                    DIGCohortQDTargetGroup(
                        name=str(item.get("name") or "").strip(),
                        min_peers=_int_value(item.get("min_peers"), 0, 0),
                        fields=_str_list(item.get("fields"), []),
                        keywords=_str_list(item.get("keywords"), []),
                    )
                    for item in _dict_list(cohort_qd_raw.get("target_keyword_groups"))
                    if str(item.get("name") or "").strip()
                ],
                semantic_label_groups=[
                    DIGCohortQDTargetGroup(
                        name=str(item.get("name") or "").strip(),
                        min_peers=0,
                        fields=_str_list(item.get("fields"), []),
                        keywords=_str_list(item.get("keywords"), []),
                    )
                    for item in _dict_list(cohort_qd_raw.get("semantic_label_groups"))
                    if str(item.get("name") or "").strip()
                ],
                parent_lineage_label_groups=[
                    DIGCohortQDTargetGroup(
                        name=str(item.get("name") or "").strip(),
                        min_peers=0,
                        fields=_str_list(item.get("fields"), []),
                        keywords=_str_list(item.get("keywords"), []),
                    )
                    for item in _dict_list(cohort_qd_raw.get("parent_lineage_label_groups"))
                    if str(item.get("name") or "").strip()
                ],
                novelty_axis_label_groups=[
                    DIGCohortQDTargetGroup(
                        name=str(item.get("name") or "").strip(),
                        min_peers=0,
                        fields=_str_list(item.get("fields"), []),
                        keywords=_str_list(item.get("keywords"), []),
                    )
                    for item in _dict_list(cohort_qd_raw.get("novelty_axis_label_groups"))
                    if str(item.get("name") or "").strip()
                ],
                label_synonyms=_str_dict(cohort_qd_raw.get("label_synonyms")),
            ),
            contract=DIGContractConfig(
                require_selected_contract=_bool_value(
                    contract_raw.get("require_selected_contract"),
                    defaults.contract.require_selected_contract,
                ),
                require_ablation_hooks=_bool_value(
                    contract_raw.get("require_ablation_hooks"),
                    defaults.contract.require_ablation_hooks,
                ),
                min_ablation_hooks=_int_value(
                    contract_raw.get("min_ablation_hooks"),
                    defaults.contract.min_ablation_hooks,
                    0,
                ),
                min_rejected_alternatives=_int_value(
                    contract_raw.get("min_rejected_alternatives"),
                    defaults.contract.min_rejected_alternatives,
                    0,
                ),
                require_forbidden_changes=_bool_value(
                    contract_raw.get("require_forbidden_changes"),
                    defaults.contract.require_forbidden_changes,
                ),
                require_expected_metric_signature=_bool_value(
                    contract_raw.get("require_expected_metric_signature"),
                    defaults.contract.require_expected_metric_signature,
                ),
                require_fail_fast_checks=_bool_value(
                    contract_raw.get("require_fail_fast_checks"),
                    defaults.contract.require_fail_fast_checks,
                ),
            ),
            write_gate=DIGWriteGateConfig(
                enabled=_bool_value(write_gate_raw.get("enabled"), defaults.write_gate.enabled),
                allow_writes_only_under_dig_dir_before_unlock=_bool_value(
                    write_gate_raw.get("allow_writes_only_under_dig_dir_before_unlock"),
                    defaults.write_gate.allow_writes_only_under_dig_dir_before_unlock,
                ),
                block_shell_before_unlock=_bool_value(
                    write_gate_raw.get("block_shell_before_unlock"),
                    defaults.write_gate.block_shell_before_unlock,
                ),
                block_variant_dir_before_unlock=_bool_value(
                    write_gate_raw.get("block_variant_dir_before_unlock"),
                    defaults.write_gate.block_variant_dir_before_unlock,
                ),
                block_results_dir_before_unlock=_bool_value(
                    write_gate_raw.get("block_results_dir_before_unlock"),
                    defaults.write_gate.block_results_dir_before_unlock,
                ),
            ),
        )


@dataclass
class QualityDiversityConfig:
    """Generation-aware QD policy independent of DIG execution.

    The initial generation can apply QD to the DIG candidate pools. Later
    generations use the established PI synthesis path, so QD does not require
    another DIG planner pass or another artifact type.
    """

    enabled: bool = True
    initial_generation_enabled: bool = True
    later_generations_enabled: bool = True
    cohort: DIGCohortQDConfig = field(default_factory=DIGCohortQDConfig)

    @classmethod
    def from_task_spec(
        cls,
        raw_task_spec: Any,
        *,
        dig_config: DIGLiteConfig,
    ) -> QualityDiversityConfig:
        if not isinstance(raw_task_spec, dict):
            return cls(
                enabled=False,
                initial_generation_enabled=False,
                later_generations_enabled=False,
                cohort=DIGCohortQDConfig(enabled=False),
            )

        qd_raw = raw_task_spec.get("quality_diversity")
        if isinstance(qd_raw, dict):
            parsed = DIGLiteConfig.from_raw({"cohort_qd": qd_raw})
            defaults = cls()
            return cls(
                enabled=_bool_value(qd_raw.get("enabled"), defaults.enabled),
                initial_generation_enabled=_bool_value(
                    qd_raw.get("initial_generation_enabled"),
                    defaults.initial_generation_enabled,
                ),
                later_generations_enabled=_bool_value(
                    qd_raw.get("later_generations_enabled"),
                    defaults.later_generations_enabled,
                ),
                cohort=parsed.cohort_qd,
            )

        # Backward compatibility: a task that configured DIG inherited its
        # nested cohort_qd policy. Tasks with no DIG/QD block retain the old
        # direct-peer behavior rather than silently opting into QD.
        dig_raw = raw_task_spec.get("dig_lite")
        if isinstance(dig_raw, dict):
            cohort_raw = dig_raw.get("cohort_qd")
            cohort_raw = cohort_raw if isinstance(cohort_raw, dict) else {}
            if not dig_config.enabled:
                return cls(
                    enabled=False,
                    initial_generation_enabled=False,
                    later_generations_enabled=False,
                    cohort=DIGCohortQDConfig(enabled=False),
                )
            return cls(
                # Legacy ``cohort_qd.enabled`` controlled only the cross-peer
                # allocator. DIG still performed its local quality/diversity
                # selection, so preserve that behavior while keeping the
                # allocator switch on ``cohort.enabled`` below.
                enabled=True,
                initial_generation_enabled=_bool_value(
                    cohort_raw.get("initial_generation_enabled"), True
                ),
                later_generations_enabled=_bool_value(
                    cohort_raw.get("later_generations_enabled"), True
                ),
                cohort=dig_config.cohort_qd,
            )

        return cls(
            enabled=False,
            initial_generation_enabled=False,
            later_generations_enabled=False,
            cohort=DIGCohortQDConfig(enabled=False),
        )

    def enabled_for_generation(self, generation_id: int) -> bool:
        if not self.enabled:
            return False
        if int(generation_id) == 0:
            return self.initial_generation_enabled
        return self.later_generations_enabled

    def pi_planning_policy(self, generation_id: int) -> dict[str, Any]:
        """Return a compact prompt policy for non-DIG PI candidate allocation."""

        if (
            int(generation_id) <= 0
            or not self.enabled_for_generation(generation_id)
            or not self.cohort.enabled
        ):
            return {}
        raw = asdict(self.cohort)
        return {
            "enabled": True,
            "candidate_source": "existing_pi_synthesis",
            "selection_mode": "prompt_guided_quality_diversity",
            "scoring_guidance": {
                "quality_weight": raw["quality_weight"],
                "novelty_weight": raw["novelty_weight"],
                "lane_fit_bonus": raw["lane_fit_bonus"],
                "risk_penalty_weight": raw["risk_penalty_weight"],
                "target_keyword_bonus": raw["target_keyword_bonus"],
            },
            "minimum_distinct": {
                "mechanism_families": raw["min_distinct_mechanism_families"],
                "intervention_surfaces": raw["min_distinct_intervention_surfaces"],
            },
            "soft_caps": {
                key: raw[key]
                for key in (
                    "max_same_diversity_cell_peers",
                    "max_same_mechanism_family_peers",
                    "max_same_mechanism_family_fraction",
                    "max_same_intervention_surface_peers",
                    "max_same_intervention_surface_fraction",
                    "max_same_intent_peers",
                    "max_same_intent_fraction",
                    "max_same_semantic_family_peers",
                    "max_same_semantic_family_fraction",
                    "max_same_parent_lineage_peers",
                    "max_same_parent_lineage_fraction",
                )
            },
            "target_keyword_groups": raw["target_keyword_groups"],
            "semantic_label_groups": raw["semantic_label_groups"],
            "parent_lineage_label_groups": raw["parent_lineage_label_groups"],
            "novelty_axis_label_groups": raw["novelty_axis_label_groups"],
            "label_synonyms": raw["label_synonyms"],
        }
