"""v2026-05-04 PI / Synthesis agent.

Runs ONCE between every pair of generations. Reads the just-completed
generation's full state (findings, edges, frontier, leaderboard) and
writes a `research_agenda_gen{N+1}.yaml` that the next generation's
peers receive as explicit role contracts.

The PI agent does NOT run experiments, does NOT publish findings, does
NOT touch optimizer code. It only synthesizes and assigns. Its output
is a structured YAML; the active workflow and state contracts are described in
`docs/concepts/architecture.md` and `docs/concepts/runtime-model.md`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sqlite3
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from praxist.core.tool_servers import (
    LITERATURE_LOOKUP_MCP_TOOL_NAMES,
    LITERATURE_LOOKUP_SERVER_NAME,
)
from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    COMMITTED,
    DERIVED_VIEW,
    FAILED,
    PARTIAL_OUTPUT,
    artifact_semantics,
    attach_artifact_semantics,
    explicit_entry_generation_id,
    is_committed_runtime_fact_source,
)
from praxist.plugins.workflow_stages.research_loop.backend.effective_config import (
    EFFECTIVE_CONFIG_METADATA_KEYS,
    has_effective_config_metadata,
)
from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
    _is_committed_frontier_entry,
)
from praxist.plugins.workflow_stages.research_loop.backend.gems import (
    load_active_gems_for_prompt,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_metadata import (
    normalize_agenda_research_metadata,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
    VALIDATION_PARENT_IDENTITY_KEYS,
    VALIDATION_PARENT_USAGES,
    _is_placeholder,
    _validation_parent_identity_refs,
    _validation_parent_tokens,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
    _digest_validation_candidates,
    _validation_candidate_aliases_from_manifest,
)
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import (
    is_scoreless,
    load_scoreless_evidence,
)

logger = logging.getLogger(__name__)


AGENDAS_DIRNAME = "agendas"
AGENDA_FILE_PATTERN = "research_agenda_gen{}.yaml"


def _parse_agenda_file(path: Path) -> dict[str, Any] | None:
    """Robust agenda YAML loader.

    Handles common LLM output quirks (R2#22 / R2#27 fix):
    - UTF-8 BOM
    - Markdown ```yaml ... ``` fences (Claude often wraps when the
      schema example in the prompt itself uses a fence)
    - Trailing whitespace
    - Returns None for: missing file, parse error, non-dict result.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("agenda load: cannot read %s: %s", path, e)
        return None
    # Strip BOM
    if text and text[0] == "﻿":
        text = text[1:]
    text = text.strip()
    # Strip outer markdown fence if present
    if text.startswith("```"):
        # First line: ```yaml or ```
        lines = text.split("\n")
        if lines:
            lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop closing fence
        text = "\n".join(lines)
    if not text.strip():
        logger.warning("agenda load: empty content at %s", path)
        return None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        logger.error("agenda load: YAML parse failed for %s: %s", path, e)
        return None
    if not isinstance(data, dict):
        logger.error(
            "agenda load: top-level YAML at %s is %s, expected dict",
            path,
            type(data).__name__,
        )
        return None
    return data


def _agenda_is_committed_for_runtime(agenda: dict[str, Any]) -> bool:
    """Return whether an agenda may drive a peer generation.

    Old agendas did not carry artifact semantics, so they remain compatible.
    New agendas that explicitly declare partial/failed/superseded state are
    ignored by runtime loaders and resume inference.
    """

    semantics = agenda.get("artifact_semantics")
    if not isinstance(semantics, dict):
        return True
    status = str(semantics.get("status") or "").strip().lower()
    return status in {"", COMMITTED}


def _annotate_agenda_artifact(
    agenda: dict[str, Any],
    *,
    completed_gen_id: int,
    next_gen_id: int,
    actor: str,
    status: str = COMMITTED,
    notes: str | None = None,
) -> dict[str, Any]:
    """Mark a PI agenda as a derived plan, not an evidence fact owner."""

    return attach_artifact_semantics(
        agenda,
        role=DERIVED_VIEW,
        status=status,
        stage="pi_agenda",
        generation_id=next_gen_id,
        actor=actor,
        derived_from=[
            "frontier/frontier_manifest.json",
            "shared_store.db",
            "research_memory/*",
            "gems/gems_state.json",
        ],
        canonical_sources=[
            "frontier/frontier_manifest.json",
            "shared_store.db",
            "research_memory/*",
            "gems/gems_state.json",
            "shared_findings/*",
            "results/*",
        ],
        runtime_fact_source=False,
        notes=notes
        or (
            f"Agenda for gen {next_gen_id} derived after gen {completed_gen_id}. "
            "Peers may use it as a plan, but measured evidence must be read from "
            "canonical result/frontier/Gems state."
        ),
    )


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    os.replace(tmp, path)


def _write_rejected_agenda_with_raw_candidate(
    rejected_path: Path,
    *,
    candidate_path: Path,
    agenda: dict[str, Any],
    completed_gen_id: int,
    next_gen_id: int,
    validation_error: str,
) -> None:
    """Preserve raw failed PI output while tagging it as non-runtime partial output."""

    raw_text = ""
    with contextlib.suppress(OSError):
        raw_text = candidate_path.read_text(encoding="utf-8")
    payload = {
        "artifact_semantics": artifact_semantics(
            role=PARTIAL_OUTPUT,
            status=FAILED,
            stage="pi_agenda_rejected",
            generation_id=next_gen_id,
            actor="research_loop:single_pi",
            derived_from=[
                "frontier/frontier_manifest.json",
                "shared_store.db",
                "research_memory/*",
                "gems/gems_state.json",
            ],
            canonical_sources=[
                "frontier/frontier_manifest.json",
                "shared_store.db",
                "research_memory/*",
                "gems/gems_state.json",
                "shared_findings/*",
                "results/*",
            ],
            runtime_fact_source=False,
            notes=(
                f"Rejected single-PI agenda for gen {next_gen_id} after gen "
                f"{completed_gen_id}: {validation_error}"
            ),
        ),
        "validation_error": validation_error,
        "parsed_agenda": agenda,
        "raw_candidate_text": raw_text,
    }
    _write_yaml_atomic(rejected_path, payload)


def _validation_candidate_parent_ids(entries: list[dict[str, Any]] | None) -> set[str]:
    ids: set[str] = set()
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        for key in VALIDATION_PARENT_IDENTITY_KEYS:
            ids.update(_validation_parent_tokens(entry.get(key)))
        aliases = entry.get("identity_aliases")
        if isinstance(aliases, list):
            for value in aliases:
                ids.update(_validation_parent_tokens(value))
        metrics = entry.get("metrics")
        if isinstance(metrics, dict):
            for key in VALIDATION_PARENT_IDENTITY_KEYS:
                ids.update(_validation_parent_tokens(metrics.get(key)))
            aliases = metrics.get("identity_aliases")
            if isinstance(aliases, list):
                for value in aliases:
                    ids.update(_validation_parent_tokens(value))
    return ids


@dataclass
class PIAgentResult:
    """Legacy single-PI result shape containing agenda text and parsed contract data."""

    success: bool
    agenda_path: Path | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    next_gen_id: int = -1


class PIAgent:
    """Standalone synthesizer agent.

    Lifecycle:
      1. Orchestrator finishes gen N (synthesis trigger fired, peers drained).
      2. Orchestrator calls `await pi.run(completed_gen_id=N)`.
      3. PI reads:
            - findings + edges + challenges from SQLite (this gen + prior)
            - current frontier (from <run_dir>/frontier/)
            - Pareto leaderboard (computed)
            - prior agenda if exists
      4. PI synthesizes within an optional configured runtime limit.
      5. PI writes `<run_dir>/agendas/research_agenda_gen{N+1}.yaml`.
      6. Orchestrator validates the YAML; if invalid + `strict=False`
         falls back to prior agenda (or no agenda for gen 0→1 transition).
    """

    def __init__(
        self,
        run_dir: Path,
        workspace: Path,
        cohort_size: int,
        model: str,
        max_runtime_minutes: int | None = 15,
        strict: bool = False,
        store_db_filename: str = "shared_store.db",
        prompt_template_path: Path | None = None,
        mcp_servers: dict[str, Any] | None = None,
        local_mode: bool = True,
        # v2026-05-05: optional Multi-PI panel routing.
        use_multi_pi_panel: bool = False,
        multi_pi_config: Any | None = None,
        research_memory_config: Any | None = None,
        # 2026-05-07: passed through to run_panel() for PI + Chair
        # adaptive thinking + max effort. Threaded from task_spec.agent.
        premium_mode: bool = False,
        # Issue #75 batch 3: forwarded to run_panel so BasePI can resolve
        # ``task_role:*`` refs without a PRAXIST_TASK_PROJECT_PATH env read.
        task_project_path: Path | None = None,
        # Panel topology ref (from ``TaskSpec.panel_topology_ref``). When a
        # task project declares a custom topology under
        # ``praxist_plugins.panel.topology``, this carries it through
        # to ``run_panel`` so the executor instantiates the declared PIs
        # instead of the legacy three. Empty / None falls back to the
        # legacy default inside ``run_panel``.
        panel_topology_ref: str | None = None,
        # Issue #151: the plugin registry assembled by the orchestrator
        # (with task-project plugin roots included) is forwarded into
        # ``run_panel`` → ``panel_topology_for_ref`` so a task-level
        # ``praxist_plugins.panel.topology`` plugin actually resolves.
        # Without this, ``panel_topology_for_ref`` built its own
        # ``PluginRoots.defaults(workspace)`` (no ``task_path``) and the
        # custom topology was invisible, silently degrading the panel to
        # the legacy three-PI shape via
        # ``fallback_to_single_pi_on_panel_failure``.
        plugin_registry: Any | None = None,
        # Issue #83/#84: peer-role rotation from panel topology, threaded
        # from GenerationLoop._peer_role_rotation. When non-empty, the
        # validator uses these roles instead of the hardcoded five.
        peer_role_rotation: tuple[str, ...] = (),
        # #75 batch 9 (config discipline): explicit override for the
        # ``LOCAL_STORE_DIR`` env-fallback below. When set, the env read
        # is skipped entirely.
        local_store_dir: Path | str | None = None,
        # Optional generation-aware QD policy. It affects PI proposal
        # allocation and optional plan metadata, never measured evidence.
        quality_diversity_config: Any | None = None,
        # Task-configured axes used by enabled later-generation QD planning.
        # These become optional ``planned_dimensions`` on peer contracts;
        # they are planning metadata, never measured evidence.
        diversity_dimensions: list[dict[str, Any]] | None = None,
        reasoning_effort: str = "max",
        task_spec: Any = None,
    ):
        self.run_dir = Path(run_dir)
        self.workspace = Path(workspace)
        self.cohort_size = int(cohort_size)
        self.model = model
        self.max_runtime_minutes = (
            int(max_runtime_minutes) if max_runtime_minutes is not None else None
        )
        self.strict = bool(strict)
        self.use_multi_pi_panel = bool(use_multi_pi_panel)
        self.multi_pi_config = multi_pi_config
        self.research_memory_config = research_memory_config
        self.quality_diversity_config = quality_diversity_config
        self.diversity_dimensions = [
            dict(dimension)
            for dimension in (diversity_dimensions or [])
            if isinstance(dimension, dict) and str(dimension.get("name") or "").strip()
        ]
        self.premium_mode = bool(premium_mode)
        self.reasoning_effort = reasoning_effort
        self.task_spec = task_spec
        self.task_project_path: Path | None = (
            Path(task_project_path) if task_project_path is not None else None
        )
        self.panel_topology_ref: str | None = panel_topology_ref or None
        self._plugin_registry = plugin_registry  # #151: see __init__ docstring
        # Issue #83/#84: peer role rotation from panel topology. When a
        # task declares a custom topology with peer_role_rotation, this
        # overrides the hardcoded REQUIRED_PEER_ROLES in validate_agenda()
        # and the PI prompt template. Empty tuple = use legacy hardcoded set.
        self._peer_role_rotation: tuple[str, ...] = peer_role_rotation or ()
        # R2#11 fix: mirror SynthesisTrigger's env resolution.
        # #75 batch 9 (config discipline): ``local_store_dir`` is now
        # the only source. ``generation_loop`` threads it from
        # ``self.run_dir``; tests pass an explicit override. No env read.
        store_root = str(local_store_dir) if local_store_dir is not None else str(self.run_dir)
        self.db_path = Path(store_root) / store_db_filename
        self.local_mode = bool(local_mode)
        self.mcp_servers = mcp_servers or {}
        self._task_prior_metric_names_cache: set[str] | None = None

        # Default prompt template lives next to this module
        if prompt_template_path is None:
            prompt_template_path = Path(__file__).resolve().parent / "synthesis_prompt.jinja2"
        self.prompt_template_path = Path(prompt_template_path)

        self.agendas_dir = self.run_dir / AGENDAS_DIRNAME
        # R2#34 fix: defer mkdir to run() so import-time/test sites don't
        # create directories.

    # ------------------------------------------------------------------
    # State assembly
    # ------------------------------------------------------------------

    def _load_gen_findings(self, gen_id: int) -> list[dict[str, Any]]:
        """Read this generation's findings, using frozen evidence in scoreless mode."""
        if is_scoreless(getattr(self, "task_spec", None)):
            return [
                finding
                for finding in load_scoreless_evidence(self.run_dir, gen_id)
                if finding.get("generation_id") == gen_id
            ]
        if not self.db_path.exists():
            return []
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(
                    "SELECT id, finding_type, title, content, metrics, "
                    "variant_name, notes, peer_id, generation_id, timestamp, "
                    "extra FROM findings WHERE generation_id = ? "
                    "ORDER BY timestamp ASC",
                    (gen_id,),
                )
                rows = []
                for r in cur.fetchall():
                    d = dict(r)
                    # JSON-decode metrics + extra (best-effort)
                    for k in ("metrics", "extra"):
                        v = d.get(k)
                        if isinstance(v, str) and v:
                            with contextlib.suppress(Exception):
                                d[k] = json.loads(v)
                    metrics = d.get("metrics")
                    extra = d.get("extra")
                    if isinstance(extra, dict):
                        nested_extra = extra.get("extra")
                        if isinstance(nested_extra, dict):
                            merged_extra = dict(extra)
                            merged_extra.pop("extra", None)
                            merged_extra.update(nested_extra)
                            extra = merged_extra
                            d["extra"] = extra
                        if isinstance(metrics, dict):
                            merged_metrics = dict(extra)
                            merged_metrics.update(metrics)
                            d["metrics"] = merged_metrics
                        else:
                            d["metrics"] = extra
                    rows.append(d)
                return rows
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.warning("PIAgent: failed to load findings for gen %d: %s", gen_id, e)
            return []

    def _load_gen_edges(self, gen_id: int) -> list[dict[str, Any]]:
        """Edges where AT LEAST ONE endpoint is a finding from this gen."""
        scoreless = is_scoreless(getattr(self, "task_spec", None))
        retained_ids: list[str] = []
        current_ids: list[str] = []
        if scoreless:
            for finding in load_scoreless_evidence(self.run_dir, gen_id):
                finding_id = str(finding.get("id") or "")
                if finding_id:
                    retained_ids.append(finding_id)
                    if finding.get("generation_id") == gen_id:
                        current_ids.append(finding_id)
            if not current_ids:
                return []
        if not self.db_path.exists():
            return []
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            conn.row_factory = sqlite3.Row
            try:
                if scoreless:
                    retained_marks = ",".join("?" for _ in retained_ids)
                    current_marks = ",".join("?" for _ in current_ids)
                    # Edges remain advisory; both endpoints must already be
                    # visible in frozen evidence, never a late-only finding.
                    cur = conn.execute(
                        "SELECT e.edge_id, e.src_finding_id, e.dst_finding_id, "
                        "e.edge_type, e.confidence, e.created_by, e.rationale "
                        "FROM finding_edges e "
                        f"WHERE e.src_finding_id IN ({retained_marks}) "
                        f"AND e.dst_finding_id IN ({retained_marks}) "
                        f"AND (e.src_finding_id IN ({current_marks}) "
                        f"OR e.dst_finding_id IN ({current_marks})) "
                        "ORDER BY e.edge_id LIMIT 100",
                        (*retained_ids, *retained_ids, *current_ids, *current_ids),
                    )
                    edges = [dict(row) for row in cur.fetchall()]
                    for edge in edges:
                        if isinstance(edge.get("rationale"), str):
                            edge["rationale"] = edge["rationale"][:1200]
                    return edges
                cur = conn.execute(
                    "SELECT e.edge_id, e.src_finding_id, e.dst_finding_id, "
                    "e.edge_type, e.confidence, e.created_by, e.rationale "
                    "FROM finding_edges e "
                    "WHERE e.src_finding_id IN "
                    "  (SELECT id FROM findings WHERE generation_id = ?) "
                    "   OR e.dst_finding_id IN "
                    "  (SELECT id FROM findings WHERE generation_id = ?)",
                    (gen_id, gen_id),
                )
                return [dict(r) for r in cur.fetchall()]
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.warning("PIAgent: failed to load edges for gen %d: %s", gen_id, e)
            return []

    def _load_frontier_summary(self, completed_gen_id: int | None = None) -> list[dict[str, Any]]:
        """Read the current frontier manifest (cumulative across gens).

        FrontierStore writes the manifest with two keys: `cumulative_top`
        (a flat list of all promoted variants across all gens) and
        `generations` (a per-gen-id dict). Use cumulative_top first, then
        flatten generations as a fallback. (R1#2 fix.)
        """
        if is_scoreless(getattr(self, "task_spec", None)):
            return []
        manifest = self.run_dir / "frontier" / "frontier_manifest.json"
        if not manifest.exists():
            return []
        try:
            data = json.loads(manifest.read_text())
            if not is_committed_runtime_fact_source(data, legacy_ok=True):
                logger.warning(
                    "PIAgent: ignoring non-committed runtime frontier manifest: %s",
                    manifest,
                )
                return []
            trust_committed_membership = is_committed_runtime_fact_source(
                data,
                legacy_ok=False,
            )
            raw_entries = data.get("cumulative_top")
            entries_with_hints: list[tuple[dict[str, Any], int | None]] = []
            if isinstance(raw_entries, list) and raw_entries:
                entries_with_hints = [
                    (dict(entry), None) for entry in raw_entries if isinstance(entry, dict)
                ]
            else:
                # Fallback: flatten per-gen dict
                gens = data.get("generations") or {}
                if isinstance(gens, dict):
                    for g_key in sorted(
                        gens.keys(), key=lambda k: int(k) if str(k).lstrip("-").isdigit() else 0
                    ):
                        g_entries = gens.get(g_key) or []
                        if isinstance(g_entries, list):
                            members = g_entries
                        elif isinstance(g_entries, dict):
                            members = g_entries.get("members") or g_entries.get("entries") or []
                        else:
                            members = []
                        if isinstance(members, list):
                            for member in members:
                                if not isinstance(member, dict):
                                    continue
                                entry = dict(member)
                                generation_hint = (
                                    int(g_key) if str(g_key).lstrip("-").isdigit() else None
                                )
                                entries_with_hints.append((entry, generation_hint))
            # R5#2 fix: sanitize each entry's metrics so tojson cannot crash
            # on datetime / np scalar / NaN / Inf values.
            filtered_entries = []
            for e, generation_hint in entries_with_hints:
                if not trust_committed_membership and not _is_committed_frontier_entry(e):
                    continue
                if trust_committed_membership:
                    raw_gen = explicit_entry_generation_id(
                        e,
                        generation_hint=generation_hint,
                    )
                else:
                    raw_gen = e.get("generation_id")
                    if raw_gen is None and isinstance(e.get("metrics"), dict):
                        raw_gen = e["metrics"].get("generation_id")
                    if raw_gen is None:
                        raw_gen = generation_hint
                normalized_gen: int | None = None
                if raw_gen is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        normalized_gen = int(raw_gen)
                if completed_gen_id is not None:
                    if normalized_gen is None:
                        continue
                    if normalized_gen > int(completed_gen_id):
                        continue
                if normalized_gen is not None:
                    e["generation_id"] = normalized_gen
                if isinstance(e, dict) and isinstance(e.get("metrics"), dict):
                    e["metrics"] = self._sanitize_json_value(e["metrics"])
                filtered_entries.append(e)
            return filtered_entries
        except Exception as e:
            logger.warning("PIAgent: failed to load frontier manifest: %s", e)
            return []

    def _load_validation_candidates(
        self, completed_gen_id: int | None = None
    ) -> list[dict[str, Any]]:
        if is_scoreless(getattr(self, "task_spec", None)):
            return []
        return _digest_validation_candidates(
            self.run_dir,
            max_entries=16,
            current_gen_id=completed_gen_id,
        )

    def _load_validation_candidate_ids(self, completed_gen_id: int | None = None) -> set[str]:
        if is_scoreless(getattr(self, "task_spec", None)):
            return set()
        return _validation_candidate_aliases_from_manifest(
            self.run_dir,
            current_gen_id=completed_gen_id,
        ) | _validation_candidate_parent_ids(
            _digest_validation_candidates(
                self.run_dir,
                max_entries=10_000,
                current_gen_id=completed_gen_id,
            )
        )

    def _load_prior_agenda(self, completed_gen_id: int) -> dict[str, Any] | None:
        """Load the agenda that drove the just-completed gen, if any."""
        if completed_gen_id < 1:
            return None
        path = self.agendas_dir / AGENDA_FILE_PATTERN.format(completed_gen_id)
        if not path.exists():
            return None
        agenda = _parse_agenda_file(path)
        if agenda is None or not _agenda_is_committed_for_runtime(agenda):
            return None
        return agenda

    def _load_prior_agendas_summary(self, completed_gen_id: int) -> list[dict[str, Any]]:
        """R2#4 fix: PI's longitudinal memory.

        Returns one summarized entry per prior gen (gens 1..completed_gen_id − 1)
        with just the high-level facts. R5#6 fix: excludes the just-completed
        gen because it's already rendered in full as `prior_agenda` in the
        prompt — avoiding duplicate rendering at the gen N→N+1 boundary.

        R5#1 / R4-M1 / R5#2 fixes: every string field is coerced to str (None
        becomes ""), every dict access is guarded with isinstance, and lists
        run through sanitize_json_value so non-JSON-able elements are stringified.
        """
        summaries = []
        # R5#6 fix: range stops at completed_gen_id (exclusive) — the
        # just-completed gen N is rendered as full prior_agenda, no need
        # to also include it as a one-line summary entry.
        for prev_gen in range(1, completed_gen_id):
            path = self.agendas_dir / AGENDA_FILE_PATTERN.format(prev_gen)
            if not path.exists():
                continue
            ag = _parse_agenda_file(path)
            if ag is None or not _agenda_is_committed_for_runtime(ag):
                continue
            mo = ag.get("mainline_observation")
            mo = mo if isinstance(mo, dict) else {}
            hyps = ag.get("cross_peer_hypotheses") or []
            anti = ag.get("anti_mainline_contract")
            anti = anti if isinstance(anti, dict) else {}

            def _safe_str(x):
                return x if isinstance(x, str) else ("" if x is None else str(x))

            def _coerce_list(x):
                """R7-4 fix: ensure value is a list so jinja `| join` doesn't
                character-explode a string. PI sometimes emits scalar where
                list is expected."""
                if isinstance(x, list):
                    return x
                if isinstance(x, str) and x:
                    return [x]
                return []

            summaries.append(
                {
                    "generation": prev_gen,
                    "mainline_dominant": self._sanitize_json_value(
                        _coerce_list(mo.get("current_dominant_mechanisms"))
                    ),
                    "main_risk": _safe_str(mo.get("main_risk")),
                    "key_tradeoff": _safe_str(mo.get("key_tradeoff")),
                    "hypothesis_ids": [
                        h.get("id") for h in hyps if isinstance(h, dict) and h.get("id")
                    ],
                    "anti_mainline_forbidden": self._sanitize_json_value(
                        _coerce_list(anti.get("forbidden_mechanisms"))
                    ),
                }
            )
        return summaries

    # R4-M2 fix: cap each finding's metrics dict to a small allowlist +
    # absolute char limit. Without this, an 8-gen run inlining ~96
    # findings × full metrics dict (10-15 keys each, plus possible
    # diversity_overlap_* annotations) was inflating the synthesis
    # prompt by ~40-80 KB per call.
    _PRIOR_METRICS_ALLOWLIST = {
        *EFFECTIVE_CONFIG_METADATA_KEYS,
        # Standard primary anchors
        "score",
        "metric_value",
        "lane_metric_value",
        "primary_metric_value",
        "mean_score",
        "taskscore",
        "task_score",
        "test_score",
        "eval_score",
        "compute_overhead_ratio",
        "positive_cell_fraction",
        "bottleneck_target",
        "evidence_stage",
        "tradeoff_class",
        "primary_tradeoff",
        "next_step_intent",
        "parent_candidate",
        "parent_usage",
        "frontier_lane",
        "strategy_family",
        "clean_promotion_eligible",
        "is_gem_finding",
        # Tier metadata
        "tier",
        "tier_reached",
        "tier_status",
        "promotion_eligible",
    }
    _PRIOR_METRICS_PRIORITY = (
        "bottleneck_target",
        "evidence_stage",
        "tradeoff_class",
        "primary_tradeoff",
        "next_step_intent",
        "parent_candidate",
        "parent_usage",
        "score",
        "metric_value",
        "lane_metric_value",
        "primary_metric_value",
        "mean_score",
        "taskscore",
        "task_score",
        "test_score",
        "eval_score",
        "positive_cell_fraction",
        "tier",
        "tier_reached",
        "tier_status",
        "promotion_eligible",
        "clean_promotion_eligible",
        "frontier_lane",
        "strategy_family",
        "is_gem_finding",
        "compute_overhead_ratio",
        *EFFECTIVE_CONFIG_METADATA_KEYS,
    )
    _PRIOR_METRICS_MAX_CHARS = 700

    @staticmethod
    def _sanitize_json_value(v: Any) -> Any:
        """R5#2 fix: coerce non-JSON-serializable values + NaN/Inf to safe forms.

        Without this, `tojson` filter in synthesis_prompt.jinja2 crashes
        on datetime/np.float32/Decimal/etc., and NaN/Inf render as invalid
        JSON literals that confuse PI's parser.
        """
        import math

        if v is None or isinstance(v, (bool, int, str)):
            return v
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(v, dict):
            return {str(k): PIAgent._sanitize_json_value(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple, set)):
            return [PIAgent._sanitize_json_value(x) for x in v]
        # Fallback: stringify everything else (datetime, Decimal, np scalars, …)
        try:
            return str(v)
        except Exception:
            return None

    def _task_prior_metric_names(self) -> set[str]:
        if self._task_prior_metric_names_cache is not None:
            return set(self._task_prior_metric_names_cache)
        names: set[str] = set()
        spec_path = self.run_dir / "task_spec.yaml"
        if spec_path.exists():
            try:
                raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                raw = {}
            if isinstance(raw, dict):
                evaluation = (
                    raw.get("evaluation") if isinstance(raw.get("evaluation"), dict) else {}
                )
                primary = str(evaluation.get("primary_metric") or "").strip()
                if primary:
                    names.add(primary)
                for metric in evaluation.get("aux_metrics") or []:
                    text = str(metric).strip()
                    if text:
                        names.add(text)
                for axis in evaluation.get("anchor_metrics") or []:
                    if isinstance(axis, dict):
                        text = str(axis.get("name") or axis.get("metric") or "").strip()
                    elif isinstance(axis, (list, tuple)) and axis:
                        text = str(axis[0]).strip()
                    else:
                        text = str(axis).strip()
                    if text:
                        names.add(text)
                for lane in evaluation.get("frontier_lanes") or []:
                    if not isinstance(lane, dict):
                        continue
                    for field_name in (
                        "require_metrics",
                        "require_truthy_metrics",
                        "require_falsey_metrics",
                    ):
                        for metric in lane.get(field_name) or []:
                            text = str(metric).strip()
                            if text:
                                names.add(text)
                    for field_name in ("axes", "optional_axes"):
                        for axis in lane.get(field_name) or []:
                            if isinstance(axis, dict):
                                text = str(axis.get("name") or axis.get("metric") or "").strip()
                            elif isinstance(axis, (list, tuple)) and axis:
                                text = str(axis[0]).strip()
                            else:
                                text = str(axis).strip()
                            if text:
                                names.add(text)
                    for field_name in ("min_metrics", "max_metrics"):
                        bounds = lane.get(field_name) or {}
                        if isinstance(bounds, dict):
                            names.update(
                                str(metric).strip() for metric in bounds if str(metric).strip()
                            )
                gems = raw.get("gems") if isinstance(raw.get("gems"), dict) else {}
                for field_name in (
                    "primary_metric_keys",
                    "secondary_metric_keys",
                    "lower_tail_metric_keys",
                    "validation_metric_keys",
                    "cost_metric_keys",
                ):
                    for metric in gems.get(field_name) or []:
                        text = str(metric).strip()
                        if text:
                            names.add(text)
                for rule in gems.get("result_cell_metric_derivations") or []:
                    if not isinstance(rule, dict):
                        continue
                    text = str(rule.get("name") or rule.get("output") or "").strip()
                    if text:
                        names.add(text)
                aliases = gems.get("result_metric_aliases") or {}
                if isinstance(aliases, dict):
                    names.update(str(key).strip() for key in aliases if str(key).strip())
                    names.update(
                        str(value).strip() for value in aliases.values() if str(value).strip()
                    )
        self._task_prior_metric_names_cache = set(names)
        return names

    def _trim_prior_metrics(self, m: Any) -> dict[str, Any]:
        if not isinstance(m, dict):
            return {}
        task_metric_names = self._task_prior_metric_names()
        allowlist = set(self._PRIOR_METRICS_ALLOWLIST) | task_metric_names
        kept = {k: self._sanitize_json_value(v) for k, v in m.items() if k in allowlist}
        if has_effective_config_metadata(kept) and m.get("source_result_path") not in (None, ""):
            kept["source_result_path"] = self._sanitize_json_value(m["source_result_path"])
        # Hard char cap as defense-in-depth
        s = json.dumps(kept, default=str, separators=(",", ":"))
        if len(s) > self._PRIOR_METRICS_MAX_CHARS:
            priority = [
                *self._PRIOR_METRICS_PRIORITY,
                *sorted(
                    key for key in task_metric_names if key not in self._PRIOR_METRICS_PRIORITY
                ),
            ]
            ordered_keys = [key for key in priority if key in kept] + sorted(
                k for k in kept if k not in priority
            )
            trimmed: dict[str, Any] = {"_truncated": True, "_orig_chars": len(s)}
            for key in ordered_keys:
                candidate = dict(trimmed)
                candidate[key] = kept[key]
                rendered = json.dumps(candidate, default=str, separators=(",", ":"))
                if len(rendered) <= self._PRIOR_METRICS_MAX_CHARS:
                    trimmed = candidate
            kept = trimmed
        return kept

    def _load_prior_findings_summary(
        self,
        completed_gen_id: int,
        max_per_gen: int = 12,
    ) -> list[dict[str, Any]]:
        """R2#4 fix: short summaries of findings from gens 0..completed-1.

        Returns at most `max_per_gen` recent findings per prior gen,
        each as a one-line dict {gen, id, type, peer, variant, key_metric,
        title}. Lets PI reason about "we tried X at gen 1 and it failed"
        without inflating the prompt past ~150 lines for an 8-gen run.
        Scoreless mode instead retains bounded narrative context of every
        finding type from frozen generation manifests.
        """
        out: list[dict[str, Any]] = []
        if is_scoreless(getattr(self, "task_spec", None)):
            per_generation: dict[int, int] = {}
            for finding in load_scoreless_evidence(self.run_dir, completed_gen_id - 1):
                generation_id = finding["generation_id"]
                count = per_generation.get(generation_id, 0)
                if count >= max_per_gen:
                    continue
                per_generation[generation_id] = count + 1
                out.append(
                    {
                        **finding,
                        "gen": generation_id,
                        "type": finding.get("finding_type"),
                        "peer": finding.get("peer_id"),
                        "variant": finding.get("variant_name"),
                    }
                )
            return out
        if not self.db_path.exists():
            return out
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            conn.row_factory = sqlite3.Row
            try:
                for prev_gen in range(0, completed_gen_id):
                    cur = conn.execute(
                        "SELECT id, finding_type, peer_id, variant_name, "
                        "metrics, extra, title FROM findings "
                        "WHERE generation_id = ? "
                        "  AND finding_type IN ('result','insight') "
                        "ORDER BY timestamp DESC LIMIT ?",
                        (prev_gen, max_per_gen),
                    )
                    for r in cur.fetchall():
                        d = dict(r)
                        # Decode metrics best-effort, extract one key value
                        m = d.get("metrics")
                        if isinstance(m, str):
                            try:
                                m = json.loads(m)
                            except Exception:
                                m = None
                        extra = d.get("extra")
                        if isinstance(extra, str):
                            try:
                                extra = json.loads(extra)
                            except Exception:
                                extra = None
                        if isinstance(extra, dict):
                            nested_extra = extra.get("extra")
                            if isinstance(nested_extra, dict):
                                merged_extra = dict(extra)
                                merged_extra.pop("extra", None)
                                merged_extra.update(nested_extra)
                                extra = merged_extra
                            if isinstance(m, dict):
                                merged = dict(extra)
                                merged.update(m)
                                m = merged
                            else:
                                m = extra
                        out.append(
                            {
                                "gen": prev_gen,
                                "id": d.get("id"),
                                "type": d.get("finding_type"),
                                "peer": d.get("peer_id"),
                                "variant": d.get("variant_name"),
                                "title": (d.get("title") or "")[:120],
                                "metrics": self._trim_prior_metrics(m),
                            }
                        )
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.warning("PIAgent: prior findings summary load failed: %s", e)
        return out

    def _load_gems_context(self, completed_gen_id: int | None = None) -> dict[str, Any]:
        """Load durable Gems directly for single-PI fallback prompts."""

        if is_scoreless(getattr(self, "task_spec", None)):
            return {}
        filtered = load_active_gems_for_prompt(
            self.run_dir,
            max_entries=4,
            max_generation_id=completed_gen_id,
        )
        if not filtered:
            return {}
        # Match the durable Gems global visibility cap. Older resume states
        # may contain more than four entries, but prompts should expose only
        # the compact active Gems set.
        compact = []
        for entry in filtered.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            config_metadata = {
                key: entry.get(key)
                for key in EFFECTIVE_CONFIG_METADATA_KEYS
                if entry.get(key) not in (None, "")
            }
            compact_entry = {
                "gem_finding_id": entry.get("gem_finding_id", ""),
                "variant_name": entry.get("variant_name", ""),
                "frontier_lane": entry.get("frontier_lane", ""),
                "metric_name": entry.get("metric_name", ""),
                "metric_value": entry.get("metric_value"),
                "admission_metrics": entry.get("admission_metrics", {}) or {},
                "source_finding_id": entry.get("source_finding_id", ""),
                "source_generation_id": entry.get("source_generation_id"),
                "bottleneck_target": entry.get("bottleneck_target", ""),
                "evidence_stage": entry.get("evidence_stage", ""),
                "tradeoff_class": entry.get("tradeoff_class", ""),
                "primary_tradeoff": entry.get("primary_tradeoff", ""),
                "next_step_intent": entry.get("next_step_intent", ""),
                "parent_candidate": entry.get("parent_candidate", ""),
                "parent_usage": entry.get("parent_usage", ""),
                "gem_variant_ref": entry.get("gem_variant_ref", ""),
                "finding_path": entry.get("finding_path", ""),
                **config_metadata,
            }
            if config_metadata and entry.get("source_result_path") not in (None, ""):
                compact_entry["source_result_path"] = entry["source_result_path"]
            compact.append(compact_entry)
        return {
            "cycle_index": filtered.get("cycle_index", 0),
            "reset_count": filtered.get("reset_count", 0),
            "cycle_start_generation": filtered.get("cycle_start_generation", 0),
            "entries": compact,
            "bottleneck_reports": list(filtered.get("bottleneck_reports", []) or [])[-5:],
            "latest_soft_agenda_priors": filtered.get("latest_soft_agenda_priors", {}) or {},
        }

    # ------------------------------------------------------------------
    # Agenda validation
    # ------------------------------------------------------------------

    REQUIRED_TOP_LEVEL_KEYS = {
        "generation",
        "cross_peer_hypotheses",
        "peer_contracts",
    }
    REQUIRED_PEER_ROLES = {
        "exploit",
        "falsifier",
        "bridge",
        "anti_mainline",
    }

    @staticmethod
    def _normalize_role(raw: Any) -> str:
        """R2#28 fix: normalize role labels (case + hyphen/underscore).

        Accepts: 'anti_mainline', 'anti-mainline', 'Anti Mainline', etc.
        Returns: lowercase canonical with underscores.
        """
        if not isinstance(raw, str):
            return ""
        s = raw.strip().lower()
        # Map any of -, space, . into underscore
        for ch in (" ", "-", ".", "/"):
            s = s.replace(ch, "_")
        # Collapse repeats
        while "__" in s:
            s = s.replace("__", "_")
        return s.strip("_")

    def expected_peer_ids(self, next_gen_id: int) -> list[str]:
        """R2#20 fix: full canonical peer IDs that match runtime."""
        return [f"gen{next_gen_id}_peer{i}" for i in range(self.cohort_size)]

    def validate_agenda(
        self,
        agenda: dict[str, Any],
        next_gen_id: int,
        validation_candidate_ids: set[str] | None = None,
    ) -> str | None:
        """Return None if valid, else a human-readable error string."""
        if not isinstance(agenda, dict):
            return "agenda is not a dict"
        missing = self.REQUIRED_TOP_LEVEL_KEYS - set(agenda.keys())
        if missing:
            return f"missing top-level keys: {sorted(missing)}"
        # R2#21 fix: tolerate non-numeric `generation` (PI may write 'gen1').
        raw_gen = agenda.get("generation")
        try:
            agenda_gen = int(str(raw_gen))
        except (TypeError, ValueError):
            # Try to extract digits from a string like "gen1" or "Gen 1"
            if isinstance(raw_gen, str):
                import re as _re

                m = _re.search(r"-?\d+", raw_gen)
                if m:
                    try:
                        agenda_gen = int(m.group(0))
                    except ValueError:
                        return f"agenda.generation={raw_gen!r} cannot be parsed as int"
                else:
                    return f"agenda.generation={raw_gen!r} cannot be parsed as int"
            else:
                return f"agenda.generation={raw_gen!r} is not int-coercible"
        if agenda_gen != int(next_gen_id):
            return (
                f"agenda.generation={agenda_gen} does not match expected next_gen_id={next_gen_id}"
            )
        hyps = agenda.get("cross_peer_hypotheses") or []
        if not isinstance(hyps, list) or len(hyps) < 1:
            return "cross_peer_hypotheses must be a non-empty list"
        # R2#35 fix: drop non-dict / empty hypothesis entries before checking
        valid_hyps = [h for h in hyps if isinstance(h, dict) and h.get("id")]
        if not valid_hyps:
            return "cross_peer_hypotheses must contain at least one dict with an 'id' field"
        contracts = agenda.get("peer_contracts") or {}
        if not isinstance(contracts, dict):
            return "peer_contracts must be a dict (peer_id -> contract)"
        # R2#29 fix: enforce EXACT cohort_size (was '<', allowed extras)
        if len(contracts) != self.cohort_size:
            return (
                f"peer_contracts has {len(contracts)} entries; "
                f"exactly cohort_size={self.cohort_size} required"
            )
        # R2#20 fix: contracts MUST be keyed by full canonical peer_ids
        # (gen{N}_peer{i}), since prompt template lookups use that format.
        expected = set(self.expected_peer_ids(next_gen_id))
        missing_peers = expected - set(contracts.keys())
        if missing_peers:
            return (
                f"peer_contracts missing required peer_ids "
                f"{sorted(missing_peers)}; expected exactly "
                f"{sorted(expected)}"
            )
        # R2#36 fix: each contract must be a dict
        non_dict = [pid for pid, c in contracts.items() if not isinstance(c, dict)]
        if non_dict:
            return f"peer_contracts entries must be dicts; non-dict peers: {sorted(non_dict)}"
        # R2#28 fix: normalize roles (handles anti-mainline ↔ anti_mainline)
        roles = {self._normalize_role(c.get("role")) for c in contracts.values()}
        roles.discard("")
        # Issue #83/#84: if panel topology supplies peer_role_rotation,
        # use it instead of the hardcoded REQUIRED_PEER_ROLES.
        effective_roles = (
            {self._normalize_role(role) for role in self._peer_role_rotation}
            if self._peer_role_rotation
            else self.REQUIRED_PEER_ROLES
        )
        effective_roles.discard("")
        missing_roles = effective_roles - roles
        if missing_roles:
            return f"peer_contracts missing required roles: {sorted(missing_roles)}"

        def _has_placeholder(s: Any) -> bool:
            if not isinstance(s, str):
                return False
            return _is_placeholder(s)

        for h in valid_hyps:
            for k in ("claim", "minimal_test", "kill_condition", "promote_condition"):
                if _has_placeholder(h.get(k)):
                    return (
                        f"cross_peer_hypothesis {h.get('id')!r} field "
                        f"{k!r} contains literal placeholder text — PI "
                        f"copied the schema example verbatim instead of "
                        f"filling it in"
                    )
        for pid, c in contracts.items():
            for k in ("target_hypothesis", "success_signal"):
                if _has_placeholder(c.get(k)):
                    return f"peer_contract {pid!r} field {k!r} contains literal placeholder text"
        validation_ids: set[str] = set()
        for item in validation_candidate_ids or set():
            validation_ids.update(_validation_parent_tokens(item))

        def _check_validation_parent_value(parent: Any, usage: Any, label: str) -> str | None:
            parent_text = str(parent or "").strip()
            if not parent_text or not (_validation_parent_tokens(parent_text) & validation_ids):
                return None
            usage_text = str(usage or "").strip().lower()
            if usage_text in VALIDATION_PARENT_USAGES:
                return None
            return (
                f"{label}: parent_candidate {parent_text!r} is a validation candidate, "
                "not a durable frontier/Gems parent; use validation/repair/falsify/"
                "compare parent_usage or choose a mature parent"
            )

        def _check_validation_parent(container: dict[str, Any], label: str) -> str | None:
            err = _check_validation_parent_value(
                container.get("parent_candidate"),
                container.get("parent_usage"),
                label,
            )
            if err:
                return err
            for ref_label, ref_value in _validation_parent_identity_refs(container):
                err = _check_validation_parent_value(
                    ref_value,
                    container.get("parent_usage"),
                    f"{label}.{ref_label}",
                )
                if err:
                    return err
            return None

        if validation_ids:
            for key in (
                "mainline_observation",
                "bridge_hypothesis",
                "anti_mainline_contract",
                "falsification_contract",
                "success_metrics",
                "panel_summary",
            ):
                value = agenda.get(key)
                if isinstance(value, dict):
                    err = _check_validation_parent(value, key)
                    if err:
                        return err
            for key in (
                "DISSENT_TO_EXPERIMENT",
                "minority_high_upside",
                "claim_boundary_updates",
            ):
                value = agenda.get(key) or []
                if not isinstance(value, list):
                    continue
                for idx, item in enumerate(value):
                    if not isinstance(item, dict):
                        continue
                    err = _check_validation_parent(item, f"{key}[{item.get('id') or idx}]")
                    if err:
                        return err
            for h in valid_hyps:
                err = _check_validation_parent(h, f"cross_peer_hypothesis {h.get('id')}")
                if err:
                    return err
            consensus_actions = agenda.get("consensus_actions") or []
            if isinstance(consensus_actions, list):
                for idx, action in enumerate(consensus_actions):
                    if not isinstance(action, dict):
                        continue
                    err = _check_validation_parent(
                        action,
                        f"consensus_actions[{action.get('action_id') or idx}]",
                    )
                    if err:
                        return err
            for pid, c in contracts.items():
                err = _check_validation_parent(c, f"peer_contract {pid}")
                if err:
                    return err
        # R4-M1 fix: top-level fields PI may emit as wrong type (string
        # / list scalar). Downstream loaders (_load_prior_agendas_summary,
        # template render) crash on attribute access against non-dict
        # values. Reject upfront so the operator sees a clear error.
        for opt_dict_key in (
            "mainline_observation",
            "bridge_hypothesis",
            "anti_mainline_contract",
            "falsification_contract",
            "success_metrics",
        ):
            v = agenda.get(opt_dict_key)
            if v is not None and not isinstance(v, dict):
                return f"agenda.{opt_dict_key} must be a dict (or omitted), got {type(v).__name__}"
        # R4-M3 fix: in-place clean cross_peer_hypotheses to drop non-dict
        # / id-less entries so downstream prompt-template iteration
        # cannot encounter None/scalar items (template would AttributeError
        # on None.source_findings).
        agenda["cross_peer_hypotheses"] = valid_hyps
        return None

    # ------------------------------------------------------------------
    # Synthesis prompt + agent invocation
    # ------------------------------------------------------------------

    def _quality_diversity_policy(self, next_gen_id: int) -> dict[str, Any]:
        config = self.quality_diversity_config
        if config is None:
            return {}
        builder = getattr(config, "pi_planning_policy", None)
        if not callable(builder):
            return {}
        try:
            policy = builder(next_gen_id)
        except (TypeError, ValueError):
            return {}
        if not isinstance(policy, dict):
            return {}
        prompt_policy = dict(policy)
        if prompt_policy.get("enabled") and self.diversity_dimensions:
            prompt_policy["diversity_dimensions"] = [
                dict(dimension) for dimension in self.diversity_dimensions
            ]
        return prompt_policy

    def _build_synthesis_prompt(
        self,
        completed_gen_id: int,
        findings: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        frontier: list[dict[str, Any]],
        prior_agenda: dict[str, Any] | None,
        prior_agendas_summary: list[dict[str, Any]] | None,
        prior_findings_summary: list[dict[str, Any]] | None,
        agenda_output_path: Path,
        gems_context: dict[str, Any] | None = None,
        validation_candidates: list[dict[str, Any]] | None = None,
        quality_diversity_policy: dict[str, Any] | None = None,
    ) -> str:
        """Render the PI prompt by inlining state into the jinja template."""
        from jinja2 import Template

        tpl_text = self.prompt_template_path.read_text()
        tpl = Template(tpl_text)
        # R8-M5 fix: sanitize prior_agenda before tojson rendering.
        # YAML's safe_load can produce datetime/Decimal/etc. that crash
        # the | tojson filter; killing PI on gen N+1 silently.
        prior_agenda_safe = self._sanitize_json_value(prior_agenda) if prior_agenda else None
        # R8-M6 forward-compat: also sanitize raw findings + edges in case
        # a future template uses tojson on them.
        findings_safe = [self._sanitize_json_value(f) for f in findings]
        edges_safe = [self._sanitize_json_value(e) for e in edges]
        return tpl.render(
            graph_evidence_scope="advisory_edges_between_retained_findings"
            if is_scoreless(getattr(self, "task_spec", None))
            else "generation_edges",
            research_loop_mode="scoreless"
            if is_scoreless(getattr(self, "task_spec", None))
            else "metric",
            scoreless_evidence=load_scoreless_evidence(self.run_dir, completed_gen_id)
            if is_scoreless(getattr(self, "task_spec", None))
            else [],
            completed_gen_id=completed_gen_id,
            next_gen_id=completed_gen_id + 1,
            cohort_size=self.cohort_size,
            n_findings=len(findings_safe),
            findings=findings_safe,
            n_edges=len(edges_safe),
            edges=edges_safe,
            frontier=frontier,
            validation_candidates=validation_candidates or [],
            prior_agenda=prior_agenda_safe,
            literature_lookup_enabled=LITERATURE_LOOKUP_SERVER_NAME in self.mcp_servers,
            # R2#4 fix: longitudinal context for PI
            prior_agendas_summary=prior_agendas_summary or [],
            prior_findings_summary=prior_findings_summary or [],
            gems_context=gems_context or {},
            agenda_output_path=str(agenda_output_path),
            required_peer_roles=sorted(self._peer_role_rotation)
            if self._peer_role_rotation
            else sorted(self.REQUIRED_PEER_ROLES),
            quality_diversity_policy=quality_diversity_policy or {},
            effective_config_provenance_available=has_effective_config_metadata(
                (
                    findings_safe,
                    frontier,
                    validation_candidates or [],
                    gems_context or {},
                )
            ),
        )

    async def _invoke_synthesizer(
        self,
        prompt_text: str,
        output_path: Path,
        *,
        request_id: str,
    ) -> dict[str, Any] | None:
        """Run the PI agent via Claude SDK; return parsed YAML or None.

        R1#1 fix: BaseAgent exposes `execute(task=...)` not `send_message`.
        R2#22 fix: tolerate ```yaml ... ``` markdown fences and BOM in PI output.
        """
        # Lazy import to avoid circular dep + speed up module load
        from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent

        # Restricted toolset: PI doesn't run experiments.
        # R3-N9 fix: drop MCP tools when no MCP servers configured.
        # Calling an unregistered MCP tool produces opaque session errors
        # that PI hits on every attempt; better to remove them upfront
        # so non-orchestrator callers (tests, manual invocations) don't
        # silently fail.
        allowed_tools = ["Read", "Write", "Bash", "Glob", "Grep"]
        if self.mcp_servers:
            allowed_tools.extend(
                [
                    "mcp__evaluation-tools__get_leaderboard",
                    "mcp__evaluation-tools__read_tool_result",
                    "mcp__frontier-tools__get_frontier",
                    "mcp__finding-graph-query__get_finding_neighbors",
                    "mcp__finding-graph-query__get_finding_subgraph",
                    "mcp__finding-graph-query__get_unlinked_recent_findings",
                ]
            )
            if LITERATURE_LOOKUP_SERVER_NAME in self.mcp_servers:
                allowed_tools.extend(LITERATURE_LOOKUP_MCP_TOOL_NAMES)

        # R5#3 fix: pass a stop_check_fn that returns True if a process-
        # wide shutdown sentinel exists. Lets a Ctrl-C / SIGTERM cleanly
        # interrupt PI mid-synthesis instead of waiting the full 15-min
        # asyncio.wait_for timeout. Read from a per-run shutdown sentinel
        # the orchestrator may write on shutdown; falls through to False
        # in normal operation (PI runs between gens, no trigger active).
        shutdown_sentinel = self.run_dir / "ORCHESTRATOR_SHUTDOWN"

        def _pi_stop_check():
            try:
                return shutdown_sentinel.exists()
            except (OSError, ValueError):
                return False

        agent = BaseAgent(
            name="pi_synthesizer",
            execution_role="pi",
            allowed_tools=allowed_tools,
            workspace=self.workspace,
            mcp_servers=self.mcp_servers,
            model=self.model,
            stop_check_fn=_pi_stop_check,
            premium_mode=self.premium_mode,
            reasoning_effort=self.reasoning_effort,
            plugin_registry=self._plugin_registry,
            request_id=request_id,
            runtime_env_overrides={
                "PRAXIST_PEER_ID": request_id,
                "PEER_ID": request_id,
                "PRAXIST_RUN_DIR": str(self.run_dir),
                "AUTO_RESEARCH_RUN_DIR": str(self.run_dir),
            },
        )

        # Establish a fresh canonical boundary for this invocation. Panel
        # failures and interrupted prior attempts must not be mistaken for a
        # single-PI result.
        try:
            output_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.error(
                "PIAgent: cannot clear stale candidate %s before synthesis: %s",
                output_path,
                exc,
            )
            return None

        direct_shared_output = False
        try:
            result = await agent.execute(task=prompt_text)
            if not result.success:
                logger.error(
                    "PIAgent: synthesizer agent reported failure: %s",
                    result.error or "(no error detail)",
                )
                # Still attempt to load the YAML if PI wrote one
        except Exception as e:
            logger.error("PIAgent: SDK invocation raised: %s", e)
            return None
        finally:
            # The model never owns the shared agenda path. Clean it on every
            # completion path, including cancellation and runtime exceptions.
            try:
                output_path.lstat()
            except FileNotFoundError:
                pass
            except OSError as exc:
                direct_shared_output = True
                logger.error("PIAgent: cannot inspect shared candidate %s: %s", output_path, exc)
            else:
                direct_shared_output = True
                logger.warning(
                    "PIAgent: discarding direct shared candidate; expected invocation-owned output"
                )
                try:
                    output_path.unlink()
                except OSError as exc:
                    logger.error(
                        "PIAgent: cannot remove direct shared candidate %s: %s",
                        output_path,
                        exc,
                    )

        if direct_shared_output:
            return None

        returned_request_id = str(getattr(result, "request_id", "") or "").strip()
        if returned_request_id != request_id:
            logger.error(
                "PIAgent: synthesizer returned request id %r, expected %r; refusing output",
                returned_request_id,
                request_id,
            )
            return None

        workspace_candidate = self.run_dir / "peer_workspaces" / request_id / output_path.name
        try:
            candidate_stat = workspace_candidate.lstat()
        except OSError:
            candidate_stat = None
        if candidate_stat is None or not stat.S_ISREG(candidate_stat.st_mode):
            logger.error(
                "PIAgent: synthesizer did not write a regular candidate at %s or %s",
                output_path,
                workspace_candidate,
            )
            return None
        agenda = _parse_agenda_file(workspace_candidate)
        if agenda is None:
            return None
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(workspace_candidate, output_path)
        except OSError as exc:
            logger.error(
                "PIAgent: failed to adopt current invocation candidate %s: %s",
                workspace_candidate,
                exc,
            )
            return None
        try:
            adopted_stat = output_path.lstat()
        except OSError:
            return None
        if not stat.S_ISREG(adopted_stat.st_mode):
            with contextlib.suppress(OSError):
                output_path.unlink(missing_ok=True)
            logger.error("PIAgent: adopted candidate is not a regular file: %s", output_path)
            return None
        logger.info(
            "PIAgent: atomically adopted current invocation candidate from %s",
            workspace_candidate,
        )
        return agenda

    # ------------------------------------------------------------------
    # v2026-05-05: Multi-PI panel routing
    # ------------------------------------------------------------------

    async def _run_multi_pi_panel(
        self,
        completed_gen_id: int,
        out_path: Path,
    ) -> PIAgentResult:
        """Delegate synthesis to the Multi-PI panel runner.

        Translates panel result back into PIAgentResult shape.
        """
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import run_panel

        next_gen_id = completed_gen_id + 1
        cfg = self.multi_pi_config
        panel_mode = getattr(cfg, "panel_mode_default", "full")
        auto_escalate = getattr(cfg, "auto_escalate_to_high_stakes", True)
        scoreless = is_scoreless(getattr(self, "task_spec", None))
        pi_max = getattr(cfg, "pi_max_runtime_minutes", None if scoreless else 12)
        chair_max = getattr(cfg, "chair_max_runtime_minutes", None if scoreless else 8)
        n_rounds = int(getattr(cfg, "n_rounds", 2))
        r2_max = getattr(cfg, "round2_max_runtime_minutes", None if scoreless else 6)

        t0 = time.time()
        # Quick findings summary for the pack (counts per type)
        findings_summary: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            findings_summary = self._build_findings_summary_for_panel(completed_gen_id)

        # R2#6 fix: install ORCHESTRATOR_SHUTDOWN sentinel watcher and
        # plumb it through panel_runner -> ChairArbiter + BasePI. Without
        # this, a SIGTERM during chair / PI synthesis waits the full
        # ~9-13 min runtime cap before responding.
        shutdown_sentinel = self.run_dir / "ORCHESTRATOR_SHUTDOWN"

        def _panel_stop_check() -> bool:
            try:
                return shutdown_sentinel.exists()
            except OSError:
                return False

        run_panel_kwargs: dict[str, Any] = dict(
            run_dir=self.run_dir,
            workspace=self.workspace,
            model=self.model,
            completed_gen_id=completed_gen_id,
            panel_mode=panel_mode,
            cohort_size=self.cohort_size,
            findings_summary=findings_summary,
            mcp_servers=self.mcp_servers,
            pi_max_runtime_minutes=pi_max,
            chair_max_runtime_minutes=chair_max,
            stop_check_fn=_panel_stop_check,
            auto_escalate_to_high_stakes=auto_escalate,
            n_rounds=n_rounds,
            round2_max_runtime_minutes=r2_max,
            premium_mode=self.premium_mode,
            reasoning_effort=self.reasoning_effort,
            task_project_path=self.task_project_path,
            quality_diversity_policy=self._quality_diversity_policy(next_gen_id),
        )
        if self.panel_topology_ref:
            run_panel_kwargs["topology_ref"] = self.panel_topology_ref
        # #151: forward the registry only when present so legacy callers
        # (no task-level plugins) get the same default behaviour as before.
        if self._plugin_registry is not None:
            run_panel_kwargs["registry"] = self._plugin_registry
        result = await run_panel(**run_panel_kwargs)
        duration = time.time() - t0

        if not result.success:
            return PIAgentResult(
                success=False,
                agenda_path=None,
                duration_seconds=duration,
                error=result.error or "panel-level failure",
                next_gen_id=next_gen_id,
            )

        metadata_backfill = normalize_agenda_research_metadata(result.agenda)
        if metadata_backfill:
            logger.info(
                "PIAgent.multi_pi: backfilled %d research metadata agenda fields",
                len(metadata_backfill),
            )
        result.agenda = _annotate_agenda_artifact(
            result.agenda,
            completed_gen_id=completed_gen_id,
            next_gen_id=next_gen_id,
            actor="research_loop:multi_pi_chair",
        )

        # Persist final agenda at the canonical path so existing code
        # (load_agenda_for_gen) finds it without changes.
        try:
            _write_yaml_atomic(out_path, result.agenda)
        except Exception as e:
            logger.error("PIAgent.multi_pi: write agenda failed: %s", e)
            return PIAgentResult(
                success=False,
                agenda_path=None,
                duration_seconds=duration,
                error=f"write canonical agenda failed: {e}",
                next_gen_id=next_gen_id,
            )

        # Validate via existing v1 validator (for backward compat) — v2
        # validation already ran inside panel_runner.
        legacy_err = self.validate_agenda(result.agenda, next_gen_id)
        if legacy_err is not None:
            logger.warning(
                "PIAgent.multi_pi: agenda passed v2 validator but failed v1 "
                "back-compat check: %s. Keeping anyway (v2 takes precedence).",
                legacy_err,
            )

        n_hyp = len(result.agenda.get("cross_peer_hypotheses", []) or [])
        n_peers = len(result.agenda.get("peer_contracts", {}) or {})
        logger.info(
            "PIAgent.multi_pi: agenda for gen %d successfully synthesized in "
            "%.0fs (panel_mode=%s, %d hypotheses, %d peer contracts, written "
            "to %s)",
            next_gen_id,
            duration,
            result.panel_mode,
            n_hyp,
            n_peers,
            out_path,
        )
        return PIAgentResult(
            success=True,
            agenda_path=out_path,
            duration_seconds=duration,
            error=None,
            next_gen_id=next_gen_id,
        )

    def _build_findings_summary_for_panel(self, gen_id: int) -> dict[str, Any]:
        """Build a small summary of this-gen findings for the evidence pack."""
        if is_scoreless(getattr(self, "task_spec", None)):
            evidence = load_scoreless_evidence(self.run_dir, gen_id)
            current = [row for row in evidence if row.get("generation_id") == gen_id]
            by_type: dict[str, int] = {}
            for finding in current:
                finding_type = str(finding.get("finding_type") or "unknown")
                by_type[finding_type] = by_type.get(finding_type, 0) + 1
            return {
                "research_loop_mode": "scoreless",
                "surfaced_findings_count": len(current),
                "count_scope": "bounded_current_generation_evidence",
                "by_type": by_type,
                "scoreless_evidence": evidence,
                "advisory_graph_edges": self._load_gen_edges(gen_id),
            }
        if not self.db_path.exists():
            return {}
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            try:
                cur = conn.execute(
                    "SELECT finding_type, COUNT(*) FROM findings "
                    "WHERE generation_id = ? GROUP BY finding_type",
                    (gen_id,),
                )
                by_type = {row[0]: int(row[1]) for row in cur.fetchall()}
                cur = conn.execute(
                    "SELECT COUNT(*) FROM findings WHERE generation_id = ?",
                    (gen_id,),
                )
                total = int(cur.fetchone()[0])
            finally:
                conn.close()
            return {
                "total_since_last_synthesis": total,
                "by_type": by_type,
            }
        except sqlite3.Error as e:
            logger.warning("PIAgent: findings_summary failed: %s", e)
            return {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, completed_gen_id: int) -> PIAgentResult:
        """Synthesize agenda for gen (completed_gen_id + 1).

        v2026-05-05: when `use_multi_pi_panel` is True, delegates to the
        Multi-PI panel runner. Otherwise runs the v2026-05-04 single-PI
        synthesis. On panel failure with `fallback_to_single_pi_on_panel_failure`,
        falls back to single-PI.
        """
        next_gen_id = completed_gen_id + 1
        # R2#34 fix: lazy mkdir at runtime (not at __init__).
        self.agendas_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.agendas_dir / AGENDA_FILE_PATTERN.format(next_gen_id)
        candidate_path = out_path.with_suffix(out_path.suffix + ".candidate")

        rejected_path = out_path.with_suffix(out_path.suffix + ".rejected")

        def _clear_uncommitted_shared_outputs() -> bool:
            cleared = True
            for path in (out_path, candidate_path):
                try:
                    path.lstat()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    cleared = False
                    logger.error("PIAgent: cannot inspect uncommitted output %s: %s", path, exc)
                    continue
                try:
                    path.unlink()
                except OSError as exc:
                    try:
                        os.replace(path, rejected_path)
                    except OSError as quarantine_exc:
                        cleared = False
                        logger.error(
                            "PIAgent: cannot clear or quarantine uncommitted output %s: "
                            "%s; quarantine failed: %s",
                            path,
                            exc,
                            quarantine_exc,
                        )
                    else:
                        logger.warning(
                            "PIAgent: quarantined uncommitted output %s at %s after "
                            "unlink failed: %s",
                            path,
                            rejected_path,
                            exc,
                        )
            return cleared

        # R3-N7 fix: unlink any stale agenda from a prior aborted run BEFORE
        # invoking any synthesizer. Otherwise a timeout or no-fallback panel
        # failure where PI writes nothing would silently re-load + use the
        # previous run's file, masking the failure.
        if not _clear_uncommitted_shared_outputs():
            return PIAgentResult(
                success=False,
                agenda_path=None,
                duration_seconds=0.0,
                error="cannot establish a clean PI publication boundary",
                next_gen_id=next_gen_id,
            )

        # v2026-05-05: route to Multi-PI panel if configured.
        if self.use_multi_pi_panel and self.multi_pi_config is not None:
            fallback = getattr(
                self.multi_pi_config,
                "fallback_to_single_pi_on_panel_failure",
                True,
            )
            try:
                panel_result = await self._run_multi_pi_panel(completed_gen_id, out_path)
                if panel_result.success:
                    return panel_result
                cleanup_ok = _clear_uncommitted_shared_outputs()
                if not cleanup_ok:
                    return PIAgentResult(
                        success=False,
                        agenda_path=None,
                        duration_seconds=panel_result.duration_seconds,
                        error="cannot retire failed panel output",
                        next_gen_id=next_gen_id,
                    )
                if not fallback:
                    return panel_result
                logger.warning(
                    "PIAgent: multi-PI returned failure (%s); falling back to single-PI",
                    panel_result.error or "panel-level failure",
                )
            except Exception as e:
                logger.exception("PIAgent: multi-PI panel raised; checking fallback")
                cleanup_ok = _clear_uncommitted_shared_outputs()
                if not cleanup_ok:
                    return PIAgentResult(
                        success=False,
                        agenda_path=None,
                        duration_seconds=0.0,
                        error="cannot retire raised panel output",
                        next_gen_id=next_gen_id,
                    )
                if not fallback:
                    return PIAgentResult(
                        success=False,
                        agenda_path=None,
                        duration_seconds=0.0,
                        error=f"multi-pi panel failed (no fallback): {e}",
                        next_gen_id=next_gen_id,
                    )
                logger.warning(
                    "PIAgent: multi-PI failed (%s); falling back to single-PI",
                    e,
                )
                # fall through to single-PI path

        # A failed panel attempt never lends authority to the single-PI
        # fallback. The fallback starts from a clean shared publication boundary.
        if not _clear_uncommitted_shared_outputs():
            return PIAgentResult(
                success=False,
                agenda_path=None,
                duration_seconds=0.0,
                error="cannot establish a clean single-PI publication boundary",
                next_gen_id=next_gen_id,
            )

        t0 = time.time()
        logger.info(
            "PIAgent: starting synthesis (completed_gen=%d, target=gen %d), max_runtime=%s min",
            completed_gen_id,
            next_gen_id,
            self.max_runtime_minutes,
        )

        # Assemble state
        findings = self._load_gen_findings(completed_gen_id)
        edges = self._load_gen_edges(completed_gen_id)
        frontier = self._load_frontier_summary(completed_gen_id)
        validation_candidates = self._load_validation_candidates(completed_gen_id)
        validation_candidate_ids = self._load_validation_candidate_ids(completed_gen_id)
        prior_agenda = self._load_prior_agenda(completed_gen_id)
        # R2#4 fix: longitudinal memory across all prior gens
        prior_agendas_summary = self._load_prior_agendas_summary(completed_gen_id)
        prior_findings_summary = self._load_prior_findings_summary(completed_gen_id)
        gems_context = self._load_gems_context(completed_gen_id)

        if not findings:
            logger.warning(
                "PIAgent: no findings for gen %d — synthesizer will run "
                "with empty input. Likely a degenerate gen.",
                completed_gen_id,
            )

        from praxist.plugins.workflow_stages.research_loop.backend.agent import (
            create_agent_request_id,
        )

        synthesis_request_id = create_agent_request_id("pi_synthesizer")
        invocation_candidate_path = (
            self.run_dir / "peer_workspaces" / synthesis_request_id / candidate_path.name
        )
        try:
            invocation_candidate_path.parent.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            logger.error(
                "PIAgent: request workspace collision for %s; refusing synthesis",
                synthesis_request_id,
            )
            return PIAgentResult(
                success=False,
                agenda_path=None,
                duration_seconds=time.time() - t0,
                error="single-PI request workspace collision",
                next_gen_id=next_gen_id,
            )
        except OSError as exc:
            logger.error("PIAgent: cannot create request workspace: %s", exc)
            return PIAgentResult(
                success=False,
                agenda_path=None,
                duration_seconds=time.time() - t0,
                error=f"single-PI request workspace unavailable: {exc}",
                next_gen_id=next_gen_id,
            )

        prompt_text = self._build_synthesis_prompt(
            completed_gen_id=completed_gen_id,
            findings=findings,
            edges=edges,
            frontier=frontier,
            validation_candidates=validation_candidates,
            prior_agenda=prior_agenda,
            prior_agendas_summary=prior_agendas_summary,
            prior_findings_summary=prior_findings_summary,
            gems_context=gems_context,
            agenda_output_path=invocation_candidate_path,
            quality_diversity_policy=self._quality_diversity_policy(next_gen_id),
        )

        # Save the rendered prompt for debugging (R2#32 fix: explicit utf-8)
        prompt_log = self.agendas_dir / f"pi_prompt_for_gen{next_gen_id}.md"
        try:
            prompt_log.write_text(prompt_text, encoding="utf-8")
        except OSError as e:
            logger.debug("PIAgent: failed to log prompt to %s: %s", prompt_log, e)

        # Synthesize (with max_runtime as asyncio timeout)
        agenda = None
        try:
            agenda = await asyncio.wait_for(
                self._invoke_synthesizer(
                    prompt_text,
                    candidate_path,
                    request_id=synthesis_request_id,
                ),
                timeout=(
                    self.max_runtime_minutes * 60 if self.max_runtime_minutes is not None else None
                ),
            )
        except asyncio.CancelledError:
            _clear_uncommitted_shared_outputs()
            raise
        except TimeoutError:
            logger.error(
                "PIAgent: hit %s-min timeout; the unconfirmed invocation output "
                "will not be promoted.",
                self.max_runtime_minutes,
            )
            _clear_uncommitted_shared_outputs()

        direct_final_output = False
        try:
            out_path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            direct_final_output = True
            logger.error("PIAgent: cannot inspect direct final output %s: %s", out_path, exc)
        else:
            direct_final_output = True
            logger.warning(
                "PIAgent: discarding direct final agenda; only a validated candidate "
                "may be committed"
            )
        if direct_final_output:
            _clear_uncommitted_shared_outputs()
            agenda = None

        duration = time.time() - t0

        if agenda is None:
            _clear_uncommitted_shared_outputs()
            err = f"synthesizer produced no agenda after {duration:.0f}s"
            logger.error("PIAgent: %s", err)
            return PIAgentResult(
                success=False,
                agenda_path=None,
                duration_seconds=duration,
                error=err,
                next_gen_id=next_gen_id,
            )

        metadata_backfill = normalize_agenda_research_metadata(agenda)
        if metadata_backfill:
            logger.info(
                "PIAgent: backfilled %d research metadata agenda fields",
                len(metadata_backfill),
            )

        validation_error = self.validate_agenda(
            agenda,
            next_gen_id,
            validation_candidate_ids=validation_candidate_ids,
        )
        if validation_error is not None:
            # R9-M3 fix: rename rejected agenda so the next gen's peer-prompt
            # path (load_agenda_for_gen) cannot pick it up. Keeps the file
            # for inspection under a .rejected suffix.
            rejected_path = out_path.with_suffix(out_path.suffix + ".rejected")
            try:
                if isinstance(agenda, dict):
                    _write_rejected_agenda_with_raw_candidate(
                        rejected_path,
                        candidate_path=candidate_path,
                        agenda=agenda,
                        completed_gen_id=completed_gen_id,
                        next_gen_id=next_gen_id,
                        validation_error=validation_error,
                    )
                    candidate_path.unlink(missing_ok=True)
                    out_path.unlink(missing_ok=True)
                else:
                    candidate_path.replace(rejected_path)
                    out_path.unlink(missing_ok=True)
                logger.error(
                    "PIAgent: agenda validation failed: %s. Renamed bad "
                    "output to %s for inspection (next gen will fall back "
                    "to no-agenda mode).",
                    validation_error,
                    rejected_path,
                )
            except OSError as e:
                logger.error(
                    "PIAgent: agenda validation failed (%s) AND could not "
                    "rename bad output (%s). Best-effort unlink instead.",
                    validation_error,
                    e,
                )
                with contextlib.suppress(OSError):
                    out_path.unlink(missing_ok=True)
                with contextlib.suppress(OSError):
                    candidate_path.unlink(missing_ok=True)
            return PIAgentResult(
                success=False,
                agenda_path=rejected_path,
                duration_seconds=duration,
                error=f"validation: {validation_error}",
                next_gen_id=next_gen_id,
            )

        agenda = _annotate_agenda_artifact(
            agenda,
            completed_gen_id=completed_gen_id,
            next_gen_id=next_gen_id,
            actor="research_loop:single_pi",
        )
        try:
            candidate_path.unlink(missing_ok=True)
        except OSError as e:
            err = f"validated agenda candidate could not be retired before commit: {e}"
            logger.error("PIAgent: %s", err)
            with contextlib.suppress(OSError):
                out_path.unlink(missing_ok=True)
            return PIAgentResult(
                success=False,
                agenda_path=None,
                duration_seconds=duration,
                error=err,
                next_gen_id=next_gen_id,
            )
        try:
            _write_yaml_atomic(out_path, agenda)
        except OSError as e:
            err = f"validated agenda could not be persisted: {e}"
            logger.error("PIAgent: %s", err)
            return PIAgentResult(
                success=False,
                agenda_path=None,
                duration_seconds=duration,
                error=err,
                next_gen_id=next_gen_id,
            )

        logger.info(
            "PIAgent: agenda for gen %d successfully synthesized in %.0fs "
            "(%d hypotheses, %d peer contracts, written to %s)",
            next_gen_id,
            duration,
            len(agenda.get("cross_peer_hypotheses") or []),
            len(agenda.get("peer_contracts") or {}),
            out_path,
        )
        return PIAgentResult(
            success=True,
            agenda_path=out_path,
            duration_seconds=duration,
            next_gen_id=next_gen_id,
        )


def load_agenda_for_gen(run_dir: Path, gen_id: int, cohort_size: int = 5) -> dict[str, Any] | None:
    """Helper: load + structurally validate the agenda intended for `gen_id`.

    Returns None if no agenda exists (gen 0 or PI failed without strict mode).
    R7-3 fix: also runs a defensive type-check of the top-level fields so
    a stale unvalidated agenda (left from a failed PI run with strict=False)
    cannot crash the peer prompt template. Returns None for any structural
    problem; peers then fall through to the agenda-less prompt branch.
    """
    path = Path(run_dir) / AGENDAS_DIRNAME / AGENDA_FILE_PATTERN.format(gen_id)
    if not path.exists():
        return None
    if gen_id > 0:
        boundary_path = Path(run_dir) / f"gen_{gen_id - 1}" / "generation_boundary.json"
        if boundary_path.exists():
            try:
                boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "load_agenda_for_gen(%d): prior generation boundary is unreadable; "
                    "ignoring agenda: %s",
                    gen_id,
                    exc,
                )
                return None
            if str(boundary.get("pi_status") or "") != "succeeded":
                logger.warning(
                    "load_agenda_for_gen(%d): prior generation boundary records PI status %r; "
                    "ignoring agenda",
                    gen_id,
                    boundary.get("pi_status"),
                )
                return None
    agenda = _parse_agenda_file(path)
    if agenda is None:
        return None
    if not _agenda_is_committed_for_runtime(agenda):
        logger.warning(
            "load_agenda_for_gen(%d): agenda artifact status is not committed — ignoring agenda",
            gen_id,
        )
        return None
    # Lightweight defensive validation (NOT the full validator — we don't
    # have the right cohort_size context here, but we can guard against
    # the most common crash patterns).
    pc = agenda.get("peer_contracts")
    if pc is not None and not isinstance(pc, dict):
        logger.warning(
            "load_agenda_for_gen(%d): peer_contracts is %s, not dict — ignoring agenda",
            gen_id,
            type(pc).__name__,
        )
        return None
    mo = agenda.get("mainline_observation")
    if mo is not None and not isinstance(mo, dict):
        logger.warning(
            "load_agenda_for_gen(%d): mainline_observation is %s, not dict — ignoring agenda",
            gen_id,
            type(mo).__name__,
        )
        return None
    cph = agenda.get("cross_peer_hypotheses")
    if cph is not None and not isinstance(cph, list):
        logger.warning(
            "load_agenda_for_gen(%d): cross_peer_hypotheses is %s, not list — ignoring agenda",
            gen_id,
            type(cph).__name__,
        )
        return None
    normalize_agenda_research_metadata(agenda)
    return agenda
