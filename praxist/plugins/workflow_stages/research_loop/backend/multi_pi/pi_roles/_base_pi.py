"""Base PI role — common logic for memo generation.

Each subclass defines:
  role_name: str
  prompt_template_name: str
  private_kb_dir: relative path under multi_pi/private_kb/
  fixed_questions: list of strings to include in the memo prompt

The base class handles:
  - loading private KB entries (file glob; bm25-lite ranking by keyword overlap with shared_core)
  - rendering prompt with shared_core + private pack + private KB
  - calling Claude via BaseAgent.execute
  - parsing memo YAML output (with fence/BOM tolerance, like pi_agent.py)
  - validating private_knowledge_used disclosure
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from praxist.core.role_skills import RoleSkill, load_role_skill
from praxist.core.tool_servers import (
    LITERATURE_LOOKUP_MCP_TOOL_NAMES,
    LITERATURE_LOOKUP_SERVER_NAME,
)
from praxist.plugins.workflow_stages.research_loop.backend.effective_config import (
    has_effective_config_metadata,
)

logger = logging.getLogger(__name__)

# Match the pi_agent.py YAML loader behavior for consistency. R1#10 fix:
# more permissive fence (optional inline comment, multiple BOM variants).
_FENCE_RE = re.compile(
    r"```\s*(?:yaml|yml)?\s*(?:#[^\n]*)?\s*\n(.*?)\n\s*```",
    re.DOTALL | re.IGNORECASE,
)
_BOM_VARIANTS = ("\ufeff", "\ufffe")
# Above this line count, skip the O(N\u00b2) Strategy C in
# `_strip_trailing_prose` (mirrors chair_arbiter.py:_BOTH_ENDS_LINE_CAP).
_BOTH_ENDS_LINE_CAP = 100
_HEREDOC_RE = re.compile(
    r"<<-?\s*['\"]?(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]*)['\"]?[^\n]*\n"
    r"(?P<body>.*?)(?:\n[ \t]*(?P=tag)\s*(?:\n|$))",
    re.DOTALL,
)
_MEMO_PLACEHOLDER_RE = re.compile(
    r"<\s*(?:gen|nn|0\.\.1|evidence_id|exp_id|claim_id|id|why|kb id|float|"
    r"true\|false)\s*>"
    r"|<\s*(?:one sentence|one paragraph|excerpt|bounded version|scope:[^>\n]*|"
    r"low\|medium\|high|id from[^>\n]*|one paragraph:[^>\n]*|e\.g\.[^>\n]*)>"
    r"|PI\s+#\?",
    re.IGNORECASE,
)
_PEER_LABEL_RE = re.compile(r"^PI\s+#[A-Z]+$")


def _default_prompts_dir() -> Path:
    """Return the bundled ``multi_pi/prompts/`` directory.

    Used as the fallback search path when a panel-topology plugin does not
    supply ``PanelTopologySpec.prompts_dir``, and as the second entry in the
    Jinja ``FileSystemLoader`` search list when one is supplied.
    """
    from praxist.plugins.workflow_stages.research_loop.backend import multi_pi as mp_pkg

    return Path(mp_pkg.__file__).parent / "prompts"


def _strip_yaml_fence(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for bom in _BOM_VARIANTS:
        if text.startswith(bom):
            text = text[len(bom) :]
            break
    matches = _FENCE_RE.findall(text)
    if matches:
        return matches[-1]
    return text


def _strip_trailing_prose(text: str) -> str:
    """Extract a dict-yielding YAML slice from text that may have prose
    around it.

    Production Gen 0\u21921 / Gen 1\u21922 panels (run_2026-05-04_18-38-21) showed
    every PI emitting `"Now I have everything... Let me construct the memo.\\n\\nrole: builder\\n..."`
    \u2014 a preamble line + blank line + raw YAML body, no markdown fence. The
    original parser only stripped fences, so `yaml.safe_load` choked on
    the preamble and ALL 3 PIs were marked `_pi_unavailable`. This helper
    mirrors `chair_arbiter._strip_trailing_prose` so both paths recover
    from prose-around-YAML uniformly.

    Strategy A \u2014 pop trailing lines, O(N): "yaml ... Done."
    Strategy B \u2014 pop leading lines, O(N): "Preamble ... yaml" (production case)
    Strategy C \u2014 pop both ends, O(N\u00b2) gated by `_BOTH_ENDS_LINE_CAP`.

    Returns the original text if no slice parses to a dict.
    """
    import yaml

    if not isinstance(text, str) or not text.strip():
        return text

    def _is_dict(s: str) -> bool:
        try:
            return isinstance(yaml.safe_load(s), dict)
        except Exception:
            return False

    if _is_dict(text):
        return text
    lines = text.rstrip().split("\n")
    n = len(lines)
    for end in range(n - 1, 0, -1):
        candidate = "\n".join(lines[:end])
        if _is_dict(candidate):
            return candidate
    for start in range(1, n):
        candidate = "\n".join(lines[start:])
        if _is_dict(candidate):
            return candidate
    if n <= _BOTH_ENDS_LINE_CAP:
        for start in range(1, n):
            for end in range(n, start, -1):
                candidate = "\n".join(lines[start:end])
                if _is_dict(candidate):
                    return candidate
    return text


def _parse_memo_text(text: str) -> dict[str, Any]:
    """Tolerant YAML/JSON parse of PI memo output.

    Chain: _strip_yaml_fence \u2192 _strip_trailing_prose \u2192 yaml.safe_load. The
    prose-strip step recovers from "preamble + raw YAML" outputs which the
    LLM produces when it skips markdown fences (production-observed in
    18:38 run; 6/6 PI panels failed because this step was missing).
    """
    import yaml

    cleaned = _strip_trailing_prose(_strip_yaml_fence(text or ""))
    try:
        d = yaml.safe_load(cleaned)
    except Exception:
        try:
            d = json.loads(cleaned)
        except Exception:
            return {"_parse_error": True, "raw": text[:2000]}
    if not isinstance(d, dict):
        return {"_parse_error": True, "raw": text[:2000]}
    return d


def _has_required_schema(parsed: dict[str, Any], required_keys: tuple[str, ...]) -> bool:
    """Return True when parsed YAML is both parseable and schema-complete.

    A single prose line ending in ":" is valid YAML to `safe_load`, but it is
    not a usable PI memo. Treat that as parse failure before the panel sees it.
    """
    return (
        isinstance(parsed, dict)
        and not parsed.get("_parse_error")
        and not parsed.get("_failed")
        and all(k in parsed for k in required_keys)
    )


def _contains_memo_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_MEMO_PLACEHOLDER_RE.search(value))
    if isinstance(value, dict):
        return any(_contains_memo_placeholder(v) for v in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_memo_placeholder(v) for v in value)
    return False


def _memo_role_matches(parsed: dict[str, Any], expected_role: str) -> bool:
    role = parsed.get("role")
    return isinstance(role, str) and role.strip() == expected_role


def _has_round1_schema(parsed: dict[str, Any]) -> bool:
    """Return True for a usable independent PI memo.

    Parseable YAML with `top_claims: []` is not enough evidence for Chair
    synthesis. Require a role and at least one structured top claim so empty
    stubs do not masquerade as successful Round 1 memos.
    """
    if not _has_required_schema(parsed, ("role", "top_claims")):
        return False
    if not isinstance(parsed.get("role"), str) or not parsed["role"].strip():
        return False
    top_claims = parsed.get("top_claims")
    if not isinstance(top_claims, list) or not top_claims:
        return False
    first_claim = top_claims[0]
    if not isinstance(first_claim, dict):
        return False
    if not any(k in first_claim for k in ("id", "claim_id")):
        return False
    claim_text = first_claim.get("statement") or first_claim.get("claim")
    if not isinstance(claim_text, str) or not claim_text.strip():
        return False
    return not _contains_memo_placeholder(parsed)


def _has_round2_schema(parsed: dict[str, Any]) -> bool:
    """Return True for usable Round 2 cross-review YAML.

    Round 2 has two legitimate shapes: full peer-review answers, or the
    explicit no-peers skip shape emitted when this PI has no successful peers.
    Partial fragments such as `own_revisions` alone are parseable YAML, but
    should not be treated as a successful peer memo.
    """
    if not isinstance(parsed, dict) or parsed.get("_parse_error") or parsed.get("_failed"):
        return False
    if parsed.get("_no_peers") is True:
        if not all(k in parsed for k in ("role", "round", "note")):
            return False
        try:
            round_value = int(str(parsed.get("round")).strip())
        except Exception:
            return False
        note = parsed.get("note")
        return (
            isinstance(parsed.get("role"), str)
            and bool(parsed["role"].strip())
            and round_value == 2
            and isinstance(note, str)
            and bool(note.strip())
            and not _contains_memo_placeholder(parsed)
        )
    required = (
        "role",
        "round",
        "strongest_agreement",
        "strongest_objection",
        "missing_experiment",
        "private_kb_revealed_blind_spot",
        "claim_that_should_be_downgraded",
        "singleton_high_upside_idea_to_preserve",
    )
    if not all(k in parsed for k in required):
        return False
    if not isinstance(parsed.get("role"), str) or not parsed["role"].strip():
        return False
    try:
        round_value = int(str(parsed.get("round")).strip())
    except Exception:
        return False
    if round_value != 2:
        return False

    required_dicts = {
        "strongest_agreement": ("peer_label", "claim_id", "why"),
        "strongest_objection": (
            "peer_label",
            "claim_id",
            "objection",
            "proposed_kill_test",
        ),
        "missing_experiment": ("description", "why_critical"),
        "private_kb_revealed_blind_spot": ("triggered", "peer_label", "blind_spot"),
        "claim_that_should_be_downgraded": (
            "claim_id",
            "current_language",
            "recommended_language",
            "reason",
        ),
        "singleton_high_upside_idea_to_preserve": (
            "source",
            "peer_label",
            "idea_summary",
            "protected_budget_recommendation",
        ),
    }
    for key, child_keys in required_dicts.items():
        value = parsed.get(key)
        if not isinstance(value, dict) or not value:
            return False
        if key == "private_kb_revealed_blind_spot":
            if "triggered" not in value:
                return False
        elif key == "singleton_high_upside_idea_to_preserve":
            missing_required = [child_key for child_key in child_keys if child_key not in value]
            source = value.get("source")
            if source == "self":
                missing_required = [
                    child_key for child_key in missing_required if child_key != "peer_label"
                ]
            if missing_required:
                return False
        elif not all(child_key in value for child_key in child_keys):
            return False

        private_kb_triggered = None
        if key == "private_kb_revealed_blind_spot":
            private_kb_triggered = value.get("triggered")
            if not isinstance(private_kb_triggered, bool):
                return False
            if private_kb_triggered is False:
                # The prompt explicitly allows a negative answer to leave
                # peer/blind-spot fields null or omitted. Treat that as a
                # complete Round 2 answer instead of discarding the whole
                # peer-review memo.
                continue

        singleton_source = None
        if key == "singleton_high_upside_idea_to_preserve":
            singleton_source = value.get("source")
            if singleton_source not in {"self", "peer"}:
                return False
            if singleton_source == "peer" and "peer_label" not in value:
                return False

        for child_key in child_keys:
            if child_key not in value:
                if (
                    key == "singleton_high_upside_idea_to_preserve"
                    and singleton_source == "self"
                    and child_key == "peer_label"
                ):
                    continue
                return False
            child_value = value.get(child_key)
            nullable = (
                key == "singleton_high_upside_idea_to_preserve"
                and singleton_source == "self"
                and child_key == "peer_label"
            )
            if child_value is None:
                if nullable:
                    continue
                return False
            if child_value == "":
                if nullable:
                    continue
                return False
            if child_key == "triggered" and isinstance(child_value, bool):
                continue
            if child_value is False and child_key != "triggered":
                return False
            if isinstance(child_value, str) and not child_value.strip():
                return False
            if (
                child_key == "peer_label"
                and key in {"strongest_agreement", "strongest_objection"}
                and (
                    not isinstance(child_value, str)
                    or not _PEER_LABEL_RE.match(child_value.strip())
                )
            ):
                return False
    return not _contains_memo_placeholder(parsed)


def _schema_error_payload(
    text: str,
    parsed: dict[str, Any],
    required_keys: tuple[str, ...],
) -> dict[str, Any]:
    missing = [k for k in required_keys if k not in parsed]
    return {
        "_parse_error": True,
        "_schema_error": True,
        "error": f"memo schema incomplete; missing required keys: {missing}",
        "missing_required_keys": missing,
        "raw": (text or "")[:2000],
        "parsed_preview": parsed,
    }


def _iter_nested_strings(value: Any) -> list[str]:
    """Collect strings from a nested runtime output/tool-use structure."""
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(_iter_nested_strings(item))
    elif isinstance(value, list | tuple):
        for item in value:
            out.extend(_iter_nested_strings(item))
    return out


def _iter_yaml_candidates_from_text(text: str) -> list[str]:
    """Return possible YAML payloads from raw text or shell heredocs."""
    if not isinstance(text, str) or not text.strip():
        return []
    candidates: list[str] = []
    for match in _HEREDOC_RE.finditer(text):
        body = match.group("body").strip()
        if body:
            candidates.append(body)
    candidates.append(text)
    return candidates


def _recover_schema_from_tool_uses(
    output_dict: dict[str, Any],
    required_keys: tuple[str, ...],
    yaml_keywords: tuple[str, ...],
    schema_check: Callable[[dict[str, Any]], bool] | None = None,
    expected_role: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Recover YAML emitted through tool calls.

    Claude/OpenRouter runs can put a complete memo into a Bash heredoc command
    (`cat <<'YAML' ... YAML`) while direct text outputs contain only a preamble.
    AgentResult.tool_uses may expose that YAML only under `input.command`, not
    under `output` or `result`; inspect both directions.
    """
    for tool_use in output_dict.get("tool_uses", []) or []:
        if not isinstance(tool_use, dict):
            continue
        for raw in _iter_nested_strings(tool_use):
            if not any(kw in raw for kw in yaml_keywords):
                continue
            for candidate in _iter_yaml_candidates_from_text(raw):
                parsed = _parse_memo_text(candidate)
                if (
                    schema_check(parsed)
                    if schema_check is not None
                    else _has_required_schema(parsed, required_keys)
                ):
                    if expected_role is not None and not _memo_role_matches(parsed, expected_role):
                        continue
                    return parsed, candidate
    return None, None


@dataclass
class PIMemo:
    """Structured memo emitted by one PI role during panel synthesis."""

    role: str
    raw_text: str
    parsed: dict[str, Any]
    private_kb_used: list[dict[str, Any]]
    success: bool
    error: str | None = None


class BasePI:
    """Base class for a single PI role."""

    role_name: str = "base"
    role_ref: str = ""
    prompt_template_name: str = "base.jinja2"
    private_kb_dir_name: str = ""  # subdir under multi_pi/private_kb/

    def __init__(
        self,
        run_dir: Path,
        workspace: Path,
        model: str,
        max_runtime_minutes: int | None = 12,
        mcp_servers: dict[str, Any] | None = None,
        stop_check_fn: Callable[[], bool] | None = None,
        # 2026-05-07: passed to BaseAgent so PI Round 1 / Round 2 can
        # activate adaptive thinking + max effort when task_spec says so.
        premium_mode: bool = False,
        role_ref: str | None = None,
        # Optional plugin-supplied prompts directory. When set, Round 1 and
        # Round 2 template lookups search this directory first and fall back
        # to the framework's bundled ``multi_pi/prompts/`` for templates the
        # plugin omits. Wired in by ``role_bindings.instantiate_pi_roles`` from
        # ``PanelTopologySpec.prompts_dir``.
        prompts_dir: Path | None = None,
        # Issue #75 batch 3: when set, used by ``skill()`` so
        # ``role_skills.load_role_skill`` can resolve ``task_role:*`` refs
        # without falling back to ``PRAXIST_TASK_PROJECT_PATH``. Threaded from
        # the stage context through PIAgent → run_panel → instantiate_pi_roles.
        # ``None`` preserves the historical env-fallback path used by tests
        # and any caller that constructs a PI directly.
        task_project_path: Path | None = None,
        plugin_registry: Any | None = None,
        reasoning_effort: str = "max",
    ):
        self.run_dir = Path(run_dir)
        self.workspace = Path(workspace)
        self.model = model
        self.max_runtime_minutes = max_runtime_minutes
        self.mcp_servers = mcp_servers or {}
        self.stop_check_fn = stop_check_fn
        self.premium_mode = premium_mode
        self.reasoning_effort = reasoning_effort
        self.role_ref = role_ref or self.role_ref
        self.prompts_dir: Path | None = Path(prompts_dir) if prompts_dir is not None else None
        self.task_project_path: Path | None = (
            Path(task_project_path) if task_project_path is not None else None
        )
        self.plugin_registry = plugin_registry
        self._role_skill: RoleSkill | None | bool = None
        self._research_loop_mode = "metric"

    # ------------------------------------------------------------------ private KB

    def skill(self) -> RoleSkill | None:
        if self._role_skill is False:
            return None
        if isinstance(self._role_skill, RoleSkill):
            return self._role_skill
        if not self.role_ref:
            self._role_skill = False
            return None
        try:
            self._role_skill = load_role_skill(
                self.role_ref,
                workspace=self.workspace,
                task_project_path=self.task_project_path,
            )
            return self._role_skill
        except Exception as exc:  # noqa: BLE001 - compatibility path falls back to legacy KB.
            logger.warning(
                "PI %s: RoleSkill load failed for %s: %s", self.role_name, self.role_ref, exc
            )
            self._role_skill = False
            return None

    def _private_kb_path(self) -> Path:
        # multi_pi/private_kb/<role>/
        from praxist.plugins.workflow_stages.research_loop.backend import multi_pi as mp_pkg

        pkg_dir = Path(mp_pkg.__file__).parent
        return pkg_dir / "private_kb" / (self.private_kb_dir_name or self.role_name)

    def _private_kb_files(self) -> list[Path]:
        skill = self.skill()
        if skill is not None and skill.private_kb_paths:
            return [path for path in skill.private_kb_paths if path.is_file()]
        kb_dir = self._private_kb_path()
        if not kb_dir.exists():
            return []
        files = []
        files_scanned = 0
        for path in sorted(kb_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".md", ".yaml", ".yml"):
                continue
            files_scanned += 1
            if files_scanned > self._MAX_KB_FILES:
                logger.warning(
                    "PI %s: private KB dir %s contains > %d files; stopping scan. (R3#7 cap.)",
                    self.role_name,
                    kb_dir,
                    self._MAX_KB_FILES,
                )
                break
            files.append(path)
        return files

    # R3#7 fix: bound the number of files we'll consider from a KB directory.
    # If an operator misuses the path (e.g. dumps a backup archive there),
    # we should not OOM the orchestrator scanning thousands of files.
    _MAX_KB_FILES = 500

    def load_private_kb(self, top_k: int = 12, query_blob: str = "") -> list[dict[str, Any]]:
        """Load and rank private KB entries.

        Returns a list of {id, title, type, summary, source_relative_path}.
        Ranking uses simple keyword overlap with `query_blob` (e.g. shared_core
        text), no embedding model required.
        """
        entries = []
        for f in self._private_kb_files():
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            entry = {
                "id": f.stem,
                "title": f.stem.replace("_", " "),
                "type": f.suffix.lstrip(".").lower(),
                "summary": text[:500],
                "source_relative_path": str(f.relative_to(self.workspace.resolve()))
                if str(f).startswith(str(self.workspace.resolve()))
                else str(f),
                "_full_text": text,
            }
            entries.append(entry)
        if not query_blob:
            return entries[:top_k]
        # rank by token overlap (lowercase)
        q_tokens = set(re.findall(r"\w+", query_blob.lower()))
        if not q_tokens:
            return entries[:top_k]
        scored = []
        for e in entries:
            e_tokens = set(re.findall(r"\w+", e["_full_text"].lower()))
            score = len(q_tokens & e_tokens)
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    # ------------------------------------------------------------------ prompt rendering

    def render_prompt(
        self,
        shared_core: dict[str, Any],
        private_pack: list[dict[str, Any]],
        private_kb_entries: list[dict[str, Any]],
        target_decisions: list[str],
        fixed_questions: list[str] | None = None,
    ) -> str:
        """Render the Jinja prompt template."""
        from jinja2 import Environment, FileSystemLoader

        # When ``self.prompts_dir`` is provided by the panel topology plugin
        # (see ``PanelTopologySpec.prompts_dir``), templates are searched
        # there first; the framework's bundled ``multi_pi/prompts/`` directory
        # remains a fallback so plugins can override one template (e.g.
        # ``base.jinja2``) without copying the rest. When ``prompts_dir`` is
        # ``None``, the search path collapses to the bundled directory only.
        search_paths: list[str] = []
        if self.prompts_dir is not None:
            search_paths.append(str(self.prompts_dir))
        search_paths.append(str(_default_prompts_dir()))
        env = Environment(
            loader=FileSystemLoader(search_paths),
            autoescape=False,  # output is markdown / yaml, not HTML
            trim_blocks=True,
            lstrip_blocks=True,
        )
        try:
            tpl = env.get_template(self.prompt_template_name)
        except Exception as e:
            logger.error(
                "PI %s: template %s missing: %s; falling back to base",
                self.role_name,
                self.prompt_template_name,
                e,
            )
            tpl = env.get_template("base.jinja2")
        role_skill = self.skill()
        # Round 2 retains controller-supplied mode even if the model's memo omits it.
        self._research_loop_mode = (
            "scoreless" if shared_core.get("research_loop_mode") == "scoreless" else "metric"
        )
        return tpl.render(
            role_name=self.role_name,
            max_runtime_minutes=self.max_runtime_minutes,
            role_skill=role_skill.to_prompt_context() if role_skill is not None else {},
            shared_core=shared_core,
            private_pack=private_pack,
            private_kb_entries=private_kb_entries,
            target_decisions=target_decisions or [],
            fixed_questions=fixed_questions or [],
            literature_lookup_enabled=LITERATURE_LOOKUP_SERVER_NAME in self.mcp_servers,
            effective_config_provenance_available=has_effective_config_metadata(
                (shared_core, private_pack)
            ),
        )

    # ------------------------------------------------------------------ run

    async def run(
        self,
        shared_core: dict[str, Any],
        private_pack: list[dict[str, Any]],
        target_decisions: list[str],
    ) -> PIMemo:
        """Generate this PI's memo."""
        # Build query blob from shared_core for KB ranking
        query_blob = json.dumps(shared_core, default=str)[:8000]
        kb_entries = self.load_private_kb(top_k=10, query_blob=query_blob)
        prompt = self.render_prompt(
            shared_core=shared_core,
            private_pack=private_pack,
            private_kb_entries=kb_entries,
            target_decisions=target_decisions,
            fixed_questions=self.fixed_questions(),
        )
        # Call BaseAgent
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent

            # PI agents must reason from the prompt evidence pack plus
            # cutoff-aware memory MCP tools. Raw filesystem/shell tools would
            # let them read ledgers/findings/frontier files directly and
            # bypass generation cutoffs.
            allowed_tools: list[str] = []
            if self.mcp_servers:
                allowed_tools.extend(
                    [
                        "mcp__memory-tools__get_evidence_card",
                        "mcp__memory-tools__query_evidence_cards",
                        "mcp__memory-tools__query_coverage_matrix",
                        "mcp__memory-tools__list_active_claims",
                        "mcp__memory-tools__list_open_objections",
                        "mcp__memory-tools__get_ledger_entry",
                        "mcp__memory-tools__resolve_source_ref",
                    ]
                )
                if LITERATURE_LOOKUP_SERVER_NAME in self.mcp_servers:
                    allowed_tools.extend(LITERATURE_LOOKUP_MCP_TOOL_NAMES)
            # CRIT-fix: BaseAgent requires `name` as first positional arg.
            # Without it, Round 1 PIs fail at construction time, the panel
            # falls back to single-PI silently. Discovered in v2026-05-05+
            # first production panel attempt (commit 11fa826).
            agent = BaseAgent(
                name=f"pi_{self.role_name}_round1",
                execution_role="pi",
                allowed_tools=allowed_tools,
                workspace=self.workspace,
                mcp_servers=self.mcp_servers,
                model=self.model,
                stop_check_fn=self.stop_check_fn,
                premium_mode=self.premium_mode,
                reasoning_effort=self.reasoning_effort,
                plugin_registry=self.plugin_registry,
            )
            result = await asyncio.wait_for(
                agent.execute(task=prompt),
                timeout=(
                    self.max_runtime_minutes * 60 if self.max_runtime_minutes is not None else None
                ),
            )
            if not result.success:
                return PIMemo(
                    role=self.role_name,
                    raw_text="",
                    parsed={"_failed": True, "error": result.error or "unknown"},
                    private_kb_used=[],
                    success=False,
                    error=result.error or "agent reported failure",
                )
            # CRIT-fix R1#1/#5 (post-hotfix audit): AgentResult has no
            # `final_message` attr; result.output is Dict[str,Any] with
            # shape {text_outputs: [...], tool_uses: [...], ...}. Join
            # the text_outputs into a single string.
            output_dict = result.output if isinstance(result.output, dict) else {}
            _items = output_dict.get("text_outputs", []) or []
            text = "\n".join(str(x) for x in _items if x is not None)
            parsed = _parse_memo_text(text)

            # R1#11 fix (gen3_peer7 audit): PI agents (Claude SDK) may write
            # their memo YAML via Bash tool calls (heredoc/echo) instead of
            # direct text output. The text_outputs then contain only preamble
            # text ("Let me analyze..."). When text_outputs parsing fails
            # OR produces a dict lacking required memo schema keys, fall back
            # to searching tool_uses for YAML content from Bash or Write tool
            # outputs that contain memo schema keywords.
            _REQUIRED_MEMO_KEYS = ("role", "top_claims")
            _text_parse_failed = not (
                _has_round1_schema(parsed) and _memo_role_matches(parsed, self.role_name)
            )
            if _text_parse_failed:
                recovered, recovered_text = _recover_schema_from_tool_uses(
                    output_dict,
                    required_keys=_REQUIRED_MEMO_KEYS,
                    schema_check=_has_round1_schema,
                    expected_role=self.role_name,
                    yaml_keywords=(
                        "role:",
                        "top_claims:",
                        "proposed_experiments:",
                        "objections_or_warnings:",
                    ),
                )
                if recovered is not None and recovered_text is not None:
                    parsed = recovered
                    text = recovered_text
                    logger.info(
                        "PI %s: recovered Round 1 memo from tool_uses",
                        self.role_name,
                    )
                else:
                    parsed = _schema_error_payload(text, parsed, _REQUIRED_MEMO_KEYS)

            pkb_used = parsed.get("private_knowledge_used") if isinstance(parsed, dict) else []
            if not isinstance(pkb_used, list):
                pkb_used = []
            return PIMemo(
                role=self.role_name,
                raw_text=text,
                parsed=parsed,
                private_kb_used=pkb_used,
                success=not parsed.get("_parse_error"),
                error="parse_error" if parsed.get("_parse_error") else None,
            )
        except TimeoutError:
            logger.warning(
                "PI %s: timed out after %s min", self.role_name, self.max_runtime_minutes
            )
            return PIMemo(
                role=self.role_name,
                raw_text="",
                parsed={"_failed": True, "error": "timeout"},
                private_kb_used=[],
                success=False,
                error="timeout",
            )
        except Exception as e:
            logger.exception("PI %s: unexpected failure", self.role_name)
            return PIMemo(
                role=self.role_name,
                raw_text="",
                parsed={"_failed": True, "error": str(e)},
                private_kb_used=[],
                success=False,
                error=str(e),
            )

    def fixed_questions(self) -> list[str]:
        """Subclasses override to provide role-specific fixed questions."""
        skill = self.skill()
        return list(skill.fixed_questions) if skill is not None else []

    def _fixed_questions_or(self, fallback: list[str]) -> list[str]:
        skill = self.skill()
        questions = list(skill.fixed_questions) if skill is not None else []
        return questions if questions else fallback

    # ------------------------------------------------------------------ Round 2: cross-review

    async def run_cross_review(
        self,
        own_memo: dict[str, Any],
        anon_peers: dict[str, dict[str, Any]],
        round2_max_runtime_minutes: int | None = 6,
    ) -> PIMemo:
        """Round 2 LLM call: read anonymized peer memos, answer 6 fixed questions.

        Output schema is structured YAML (see round2_cross_review.jinja2).
        Does NOT rewrite the Round 1 memo; ONLY emits responses + (optionally)
        revisions to the PI's own claim confidence/boundary.
        """
        from jinja2 import Environment, FileSystemLoader

        # See render_prompt for the search-path contract. Round 2 honors the
        # same plugin-supplied prompts_dir override.
        search_paths: list[str] = []
        if self.prompts_dir is not None:
            search_paths.append(str(self.prompts_dir))
        search_paths.append(str(_default_prompts_dir()))
        env = Environment(
            loader=FileSystemLoader(search_paths),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        try:
            tpl = env.get_template("round2_cross_review.jinja2")
        except Exception as e:
            logger.error(
                "PI %s: round2 template missing: %s; skipping cross-review",
                self.role_name,
                e,
            )
            return PIMemo(
                role=self.role_name,
                raw_text="",
                parsed={"_round2_skipped": True, "reason": "template_missing"},
                private_kb_used=[],
                success=False,
                error="round2 template missing",
            )

        prompt = tpl.render(
            role_name=self.role_name,
            max_runtime_minutes=round2_max_runtime_minutes,
            research_loop_mode=self._research_loop_mode,
            own_memo=own_memo or {},
            anon_peers=anon_peers or {},
            peer_count=len(anon_peers or {}),
        )

        try:
            from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent

            # Round 2 has the same restricted toolset as Round 1: no raw
            # filesystem/shell access, only cutoff-aware memory MCP tools.
            allowed_tools: list[str] = []
            if self.mcp_servers:
                # Same memory-tools (read-only) — PI may still want to
                # cross-check evidence cited in peer memos.
                allowed_tools.extend(
                    [
                        "mcp__memory-tools__get_evidence_card",
                        "mcp__memory-tools__query_evidence_cards",
                        "mcp__memory-tools__list_active_claims",
                        "mcp__memory-tools__resolve_source_ref",
                    ]
                )
                if LITERATURE_LOOKUP_SERVER_NAME in self.mcp_servers:
                    allowed_tools.extend(LITERATURE_LOOKUP_MCP_TOOL_NAMES)
            # CRIT-fix: pass `name` (BaseAgent requires it).
            agent = BaseAgent(
                name=f"pi_{self.role_name}_round2",
                execution_role="review",
                allowed_tools=allowed_tools,
                workspace=self.workspace,
                mcp_servers=self.mcp_servers,
                model=self.model,
                stop_check_fn=self.stop_check_fn,
                premium_mode=self.premium_mode,
                reasoning_effort=self.reasoning_effort,
                plugin_registry=self.plugin_registry,
            )
            result = await asyncio.wait_for(
                agent.execute(task=prompt),
                timeout=(
                    round2_max_runtime_minutes * 60
                    if round2_max_runtime_minutes is not None
                    else None
                ),
            )
            if not result.success:
                return PIMemo(
                    role=self.role_name,
                    raw_text="",
                    parsed={"_round2_failed": True, "error": result.error or "unknown"},
                    private_kb_used=[],
                    success=False,
                    error=result.error or "round2 agent failure",
                )
            # CRIT-fix R1#1/#5 (post-hotfix audit): AgentResult has no
            # `final_message` attr; result.output is Dict[str,Any] with
            # shape {text_outputs: [...], tool_uses: [...], ...}. Join
            # the text_outputs into a single string.
            output_dict = result.output if isinstance(result.output, dict) else {}
            _items = output_dict.get("text_outputs", []) or []
            text = "\n".join(str(x) for x in _items if x is not None)
            parsed = _parse_memo_text(text)

            # R1#11 fix (gen3_peer7 audit): also search tool_uses for YAML
            # in Round 2, same pattern as Round 1 fix above.
            _R2_REQUIRED_KEYS = (
                "role",
                "round",
                "strongest_agreement",
                "strongest_objection",
                "missing_experiment",
                "private_kb_revealed_blind_spot",
                "claim_that_should_be_downgraded",
                "singleton_high_upside_idea_to_preserve",
            )
            _r2_parse_failed = not (
                _has_round2_schema(parsed) and _memo_role_matches(parsed, self.role_name)
            )
            if _r2_parse_failed:
                recovered, recovered_text = _recover_schema_from_tool_uses(
                    output_dict,
                    required_keys=_R2_REQUIRED_KEYS,
                    yaml_keywords=(
                        "role:",
                        "_no_peers:",
                        "note:",
                        "strongest_agreement:",
                        "strongest_objection:",
                        "missing_experiment:",
                        "private_kb_revealed_blind_spot:",
                        "claim_that_should_be_downgraded:",
                        "singleton_high_upside_idea_to_preserve:",
                    ),
                    schema_check=_has_round2_schema,
                    expected_role=self.role_name,
                )
                if (
                    recovered is not None
                    and recovered_text is not None
                    and _has_round2_schema(recovered)
                ):
                    parsed = recovered
                    text = recovered_text
                    logger.info(
                        "PI %s: recovered Round 2 memo from tool_uses",
                        self.role_name,
                    )
                else:
                    parsed = _schema_error_payload(text, parsed, _R2_REQUIRED_KEYS)

            if isinstance(parsed, dict) and not parsed.get("_parse_error"):
                parsed.setdefault("round", 2)
            return PIMemo(
                role=self.role_name,
                raw_text=text,
                parsed=parsed,
                private_kb_used=[],  # Round 2 does not re-disclose KB
                success=not parsed.get("_parse_error"),
                error="parse_error" if parsed.get("_parse_error") else None,
            )
        except TimeoutError:
            logger.warning(
                "PI %s: round2 timed out after %s min",
                self.role_name,
                round2_max_runtime_minutes,
            )
            return PIMemo(
                role=self.role_name,
                raw_text="",
                parsed={"_round2_failed": True, "error": "timeout"},
                private_kb_used=[],
                success=False,
                error="timeout",
            )
        except Exception as e:
            logger.exception("PI %s: round2 unexpected failure", self.role_name)
            return PIMemo(
                role=self.role_name,
                raw_text="",
                parsed={"_round2_failed": True, "error": str(e)},
                private_kb_used=[],
                success=False,
                error=str(e),
            )
