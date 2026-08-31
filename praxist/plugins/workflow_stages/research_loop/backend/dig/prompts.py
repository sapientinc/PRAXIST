"""Prompt builders for DIG-Lite planner calls."""

from __future__ import annotations

from typing import Any

import yaml

from .config import DIGLiteConfig

READ_ONLY_RULES = """You are in DIG-Lite read-only mode.

Hard prohibitions:
- Do not create or modify variant code.
- Do not create runs/<run_id>/variants/ directories.
- Do not run task code, tests, task jobs, benchmarks, evaluation scripts,
  or any empirical measurement command.
- Do not run any command that produces empirical results.
- Do not write results artifacts.
- Do not claim observed metrics. You may only write expected/predicted metric signatures.
- Do not output private chain-of-thought; provide concise structured rationale.

Output discipline:
- Return only one strict JSON object.
- Do not wrap the final JSON in Markdown fences.
- Do not include prose before or after the JSON.
- Use double-quoted JSON strings and arrays.
- Escape any embedded double quotes inside string values.
- Do not use Markdown backticks inside JSON string values.
- Do not add trailing commas.
"""


def _yaml_block(value: Any) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def _compact_context(ctx: dict[str, Any], config: DIGLiteConfig) -> dict[str, Any]:
    keys = (
        "peer_id",
        "gen_id",
        "logical_gen_id",
        "cohort_size",
        "workspace_dir",
        "run_dir",
        "results_dir",
        "variants_dir",
        "findings_dir",
        "frontier_summary",
        "validation_candidates",
        "validation_candidates_meta",
        "incubator_top_k",
        "validation_candidate_top_k",
        "diagnostic_control_top_k",
        "strong_parent_visibility_policy",
        "research_loop_control",
        "variant_hint",
        "research_agenda",
        "gems_context",
        "graph_session_context",
        "dig_selection_policy",
    )
    compact = {key: ctx.get(key) for key in keys if key in ctx}
    task_spec = ctx.get("task_spec")
    toolchain = getattr(task_spec, "toolchain", None) if task_spec is not None else None
    toolchain_payload = dict(getattr(toolchain, "__dict__", {}))
    if task_spec is not None:
        compact["task"] = {
            "task_id": getattr(task_spec, "task_id", ""),
            "task_name": getattr(task_spec, "task_name", ""),
            "research_direction": getattr(task_spec, "research_direction", ""),
            "primary_metric": getattr(getattr(task_spec, "evaluation", None), "primary_metric", ""),
            "metric_direction": getattr(getattr(task_spec, "evaluation", None), "direction", ""),
            "aux_metrics": getattr(getattr(task_spec, "evaluation", None), "aux_metrics", []),
            "anchor_metrics": getattr(getattr(task_spec, "evaluation", None), "anchor_metrics", []),
            "frontier_lanes": getattr(getattr(task_spec, "evaluation", None), "frontier_lanes", []),
            "diversity_dimensions": getattr(
                getattr(task_spec, "evaluation", None), "diversity_dimensions", []
            ),
            "must_explore_axes": getattr(
                getattr(task_spec, "evaluation", None), "must_explore_axes", []
            ),
            "toolchain": toolchain_payload,
        }
    compact["dig_config"] = {
        "candidate_count": config.candidate_count,
        "min_mechanism_families": config.min_mechanism_families,
        "min_intervention_surfaces": config.min_intervention_surfaces,
        "diversity_cell_fields": config.diversity.cell_fields,
        "innovation_policy": {
            "enabled": config.innovation.enabled,
            "forward_intents": config.innovation.forward_intents,
            "diagnostic_intents": config.innovation.diagnostic_intents,
            "diagnostic_mechanism_families": config.innovation.diagnostic_mechanism_families,
        },
    }
    if config.max_refinement_rounds is not None:
        compact["dig_config"]["max_refinement_rounds"] = config.max_refinement_rounds
    compact["allowed_file_rules"] = [
        "Variant-local code and config under the peer's assigned variant directory.",
        "Variant-local analysis scripts when the task allows them.",
        "DIG artifacts under the peer dig/ directory during DIG.",
    ]
    compact["allowed_file_rules"].extend(config.allowed_file_rules)
    compact["disallowed_file_rules"] = [
        "Do not modify evaluator files.",
        "Do not change data split files or split logic.",
        "Do not change metric calculation.",
        "Do not modify committed baselines unless explicitly assigned.",
        "Do not modify canonical baseline implementation files; copy or create variant-local implementation files instead.",
        "Do not modify unrelated task data.",
    ]
    compact["disallowed_file_rules"].extend(config.disallowed_file_rules)
    return compact


def build_baseline_map_prompt(ctx: dict[str, Any], config: DIGLiteConfig) -> str:
    """Build the read-only prompt for mapping baseline mechanisms."""

    return f"""# DIG-Lite PHASE 1: Baseline Mechanism Map

{READ_ONLY_RULES}

Map the task and baseline mechanisms before proposing a variant. Do not select
a final candidate yet.

Context:
```yaml
{_yaml_block(_compact_context(ctx, config))}
```

Output a single JSON object with this top-level shape:
{{
  "task_objective": {{
    "primary_metric": "",
    "secondary_metrics": [],
    "failure_constraints": [],
    "forbidden_shortcuts": []
  }},
  "baseline_core_path": [
    {{
      "file": "",
      "role": "",
      "key_functions_or_classes": [],
      "reason_it_matters": ""
    }}
  ],
  "task_execution_flow": {{
    "input_flow": [],
    "core_mechanism_flow": [],
    "update_or_optimization_flow": [],
    "evaluation_flow": []
  }},
  "intervention_surfaces": [
    {{
      "name": "",
      "files": [],
      "allowed": true,
      "expected_impact": "",
      "risk_level": "low"
    }}
  ],
  "forbidden_surfaces": [
    {{
      "name": "",
      "files": [],
      "allowed": false,
      "reason": ""
    }}
  ],
  "likely_bottlenecks": [
    {{
      "bottleneck": "",
      "evidence_from_context": "",
      "uncertainty": "low"
    }}
  ],
  "implementation_risks": [
    {{
      "risk": "",
      "affected_files": [],
      "mitigation": ""
    }}
  ]
}}

Requirements:
- list at least 3 allowed intervention surfaces;
- list forbidden surfaces;
- do not propose the final variant;
- do not mention observed empirical results from this DIG phase.
"""


def build_candidate_generation_prompt(
    ctx: dict[str, Any],
    baseline_map: dict[str, Any],
    config: DIGLiteConfig,
) -> str:
    """Build the read-only prompt for generating diverse candidate designs."""

    return f"""# DIG-Lite PHASE 2: Candidate Sketching

{READ_ONLY_RULES}

Generate {config.candidate_count} distinct mechanism-level candidate designs.
They must be task-appropriate mechanism designs, not mere parameter tweaks.
`implementation_sketch.files_to_modify` must name variant-local relative files
such as implementation files, config files, harness scripts, prompt/tool
adapters, controller/evaluator adapters, or analysis scripts as applicable. Do not
name canonical baseline, evaluator, data, result, finding, frontier, or task
configuration paths.

Coverage requirements:
- at least {config.min_mechanism_families} different mechanism families;
- at least {config.min_intervention_surfaces} different intervention surfaces;
- at least one diagnostic/falsifier candidate;
- at least one low-risk incremental candidate;
- at least one medium/high diagnostic-value candidate.

Selection intent:
- The candidate pool must include diagnostic/falsifier coverage, but coverage
  candidates should not dominate the final contract.
- If `dig_selection_policy.intent_slot` is `forward_innovation`, most candidates
  should be forward-moving mechanisms with intents such as exploit, explore,
  bridge, repair, or anti_mainline. In that slot, diagnostic/falsifier candidates
  are backup controls unless no viable forward candidate exists.
- If `dig_selection_policy.intent_slot` is `diagnostic`, make the diagnostic
  candidate strong and bounded, but still include forward alternatives.

Context:
```yaml
{_yaml_block(_compact_context(ctx, config))}
```

Baseline mechanism map:
```yaml
{_yaml_block(baseline_map)}
```

Output a single JSON object:
{{
  "candidates": [
    {{
      "candidate_id": "C01",
      "name": "",
      "mechanism_family": "task-appropriate family name, e.g. controller, estimator, planner, search, solver, safety, diagnostic_falsifier, other",
      "intervention_surface": "task-appropriate allowed surface, e.g. core_mechanism, algorithm, controller, planner, solver, prompt, tool, scoring, logging, config, other",
      "intent": "exploit",
      "semantic_family": "actual mechanism lineage in plain task-appropriate terms; use the same label for semantically similar variants even if mechanism/surface labels differ",
      "parent_lineage": "baseline, Gem/frontier parent, prior variant, independent, or another explicit parent lineage",
      "novelty_axis": "what is genuinely new relative to the parent, e.g. mechanism, interface, calibration, input curriculum, validation, safety, other",
      "hypothesis": "",
      "expected_gain_path": "",
      "implementation_sketch": {{
        "files_to_modify": [],
        "changes": []
      }},
      "diagnostic_prediction": {{
        "primary_metric": "",
        "secondary_or_safety_metric": "",
        "internal_signal": ""
      }},
      "risk": {{
        "implementation": "low",
        "metric_gaming": "low",
        "silent_bug": "low",
        "compute": "low"
      }},
      "ablation_hooks": [],
      "diversity_signature": {{
        "mechanism_family": "",
        "intervention_surface": "",
        "intent": ""
      }}
    }}
  ]
}}
"""


def build_candidate_review_prompt(
    ctx: dict[str, Any],
    baseline_map: dict[str, Any],
    candidate_pool: dict[str, Any],
    config: DIGLiteConfig,
) -> str:
    """Build the read-only prompt for compact candidate critique."""

    return f"""# DIG-Lite PHASE 3: Lightweight Critique

{READ_ONLY_RULES}

Review every candidate with compact structured critique. Do not select a final
candidate yet.

Scoring scale: 1-5.
- mechanism_plausibility, implementability, diagnostic_clarity, diversity_value:
  higher is better.
- shortcut_risk, silent_bug_risk, compute_risk: higher is worse.

Context:
```yaml
{_yaml_block(_compact_context(ctx, config))}
```

Baseline mechanism map:
```yaml
{_yaml_block(baseline_map)}
```

Candidate pool:
```yaml
{_yaml_block(candidate_pool)}
```

Output a single JSON object:
{{
  "reviews": [
    {{
      "candidate_id": "C01",
      "scores": {{
        "mechanism_plausibility": 1,
        "implementability": 1,
        "diagnostic_clarity": 1,
        "diversity_value": 1,
        "shortcut_risk": 1,
        "silent_bug_risk": 1,
        "compute_risk": 1
      }},
      "fatal_flaws": [],
      "repair_suggestion": "",
      "reasoned_summary": ""
    }}
  ]
}}
"""


def build_contract_prompt(
    ctx: dict[str, Any],
    baseline_map: dict[str, Any],
    candidate_pool: dict[str, Any],
    reviews: dict[str, Any],
    qd_selection: dict[str, Any],
    config: DIGLiteConfig,
) -> str:
    """Build the read-only prompt that locks the selected implementation contract."""

    return f"""# DIG-Lite PHASE 5-6: Quality-Diversity Pick and Contract Lock

{READ_ONLY_RULES}

Select a candidate using quality-diversity rules:
1. Remove candidates with fatal flaws.
2. Remove candidates that violate file rules.
3. Remove candidates incompatible with the peer lane.
4. Penalize near-duplicates of frontier, Gems, or sibling peer lanes.
5. Group by diversity cell: mechanism_family + intervention_surface + intent.
   The orchestrator also canonicalizes semantic_family, parent_lineage, and
   novelty_axis to prevent semantic collapse where many peers rename the same
   parent-family idea.
6. Prefer the best candidate within the peer's assigned lane.
7. Use adjacent-lane fallback only if needed.
8. Honor `dig_selection_policy.intent_slot`:
   - `forward_innovation`: select a forward-moving mechanism whenever any
     viable one survives. Do not select a diagnostic/falsifier merely because it
     has high diagnostic clarity or implementation simplicity.
   - `diagnostic`: select a bounded diagnostic/control candidate when one
     survives.

Do not simply select the global highest score if it collapses diversity.

Context:
```yaml
{_yaml_block(_compact_context(ctx, config))}
```

Baseline mechanism map:
```yaml
{_yaml_block(baseline_map)}
```

Candidate pool:
```yaml
{_yaml_block(candidate_pool)}
```

Candidate reviews:
```yaml
{_yaml_block(reviews)}
```

Deterministic QD pre-selection:
```yaml
{_yaml_block(qd_selection)}
```

Use the deterministic QD pre-selection's `selected_candidate_id`. Do not choose
a different candidate unless the output should fail validation and be retried.

Output a single JSON object matching selected_contract.yaml:
{{
  "selected_candidate_id": "",
  "variant_name": "",
  "diversity_cell": {{
    "mechanism_family": "",
    "intervention_surface": "",
    "intent": ""
  }},
  "semantic_family": "",
  "parent_lineage": "",
  "novelty_axis": "",
  "mechanism_hypothesis": "",
  "why_selected": "",
  "rejected_alternatives": [
    {{
      "candidate_id": "",
      "reason": ""
    }}
  ],
  "files_to_modify": [],
  "allowed_changes": [],
  "forbidden_changes": [
    "do not modify evaluator",
    "do not change data split",
    "do not change metric calculation"
  ],
  "implementation_plan": [
    {{
      "step": 1,
      "action": ""
    }}
  ],
  "expected_metric_signature": {{
    "primary": "",
    "secondary_or_safety": "",
    "diagnostic": ""
  }},
  "ablation_hooks": [],
  "fail_fast_checks": [],
  "contract_amendment_policy": {{
    "allowed_reasons": [
      "baseline assumption was wrong",
      "shape or API mismatch makes original implementation impossible",
      "contract would require touching a forbidden path"
    ],
    "required_artifact": "contract_amendment.yaml"
  }}
}}

`files_to_modify` must stay within the deterministic QD selected candidate's
variant-local implementation sketch. Use relative filenames for task-local
implementation, config, harness, adapter, or analysis files; do not list canonical baseline, evaluator, data, result,
finding, frontier, Gems, or task configuration paths.
"""


def implementation_contract_block(contract: dict[str, Any], contract_path: str) -> str:
    """Render a selected DIG contract as an implementation prompt block."""

    return f"""## DIG-Lite Selected Contract

Before this implementation session started, the orchestrator ran a read-only
Deep Innovation Gate and validated `selected_contract.yaml`.

Contract path: `{contract_path}`

You are now in implementation phase. Implement this contract. Do not silently
change the mechanism. Do not modify evaluator, data split, or metric
calculation. If implementation requires a meaningful deviation, write
`contract_amendment.yaml` under the same `dig/` directory and explain why the
mechanism remains equivalent.

```yaml
{_yaml_block(contract)}
```
"""
