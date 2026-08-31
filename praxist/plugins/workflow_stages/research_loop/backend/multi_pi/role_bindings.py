"""Legacy PI class bindings for the generic Multi-PI topology."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from praxist.core.panel_topology import PanelRoleSpec
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import (
    BasePI,
    BuilderPI,
    ExternalValidityPI,
    PortfolioPI,
    SkepticPI,
)

ROLE_CLASS_BY_ID = {
    "builder": BuilderPI,
    "skeptic": SkepticPI,
    "portfolio": PortfolioPI,
    "external_validity": ExternalValidityPI,
}


def instantiate_pi_roles(
    role_ids: Sequence[str | PanelRoleSpec],
    *,
    run_dir: Path,
    workspace: Path,
    model: str,
    max_runtime_minutes: int | None,
    mcp_servers: dict[str, Any] | None,
    stop_check_fn: Callable[[], bool] | None,
    premium_mode: bool,
    prompts_dir: Path | None = None,
    task_project_path: Path | None = None,
    plugin_registry: Any | None = None,
    reasoning_effort: str = "max",
) -> list[BasePI]:
    """Instantiate legacy PI role classes from topology role references.

    ``prompts_dir`` is forwarded to each PI constructor so plugin-supplied
    Jinja templates take precedence over the bundled ``multi_pi/prompts/``
    directory. ``None`` (the default) preserves the legacy behavior.

    ``task_project_path`` (issue #75 batch 3) is threaded through so each
    PI can resolve ``task_role:*`` refs without falling back to
    ``PRAXIST_TASK_PROJECT_PATH`` from ``os.environ``.
    """
    pis: list[BasePI] = []
    for role in role_ids:
        role_id = role.legacy_role_id if isinstance(role, PanelRoleSpec) else str(role)
        role_ref = role.role_ref if isinstance(role, PanelRoleSpec) else None
        cls = ROLE_CLASS_BY_ID.get(role_id)
        if cls is None:
            continue
        pis.append(
            cls(
                run_dir=run_dir,
                workspace=workspace,
                model=model,
                max_runtime_minutes=max_runtime_minutes,
                mcp_servers=mcp_servers,
                stop_check_fn=stop_check_fn,
                premium_mode=premium_mode,
                reasoning_effort=reasoning_effort,
                role_ref=role_ref,
                prompts_dir=prompts_dir,
                task_project_path=task_project_path,
                plugin_registry=plugin_registry,
            )
        )
    return pis
