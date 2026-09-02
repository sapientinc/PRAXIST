"""Gate B protocol dataclasses shared by core, runtimes, and providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from praxist.core.credentials import CredentialRef

JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]

FailoverReason = Literal[
    "none",
    "auth_error",
    "quota_exhausted",
    "rate_limited",
    "timeout",
    "provider_unavailable",
    "runtime_error",
    "tool_unavailable",
    "invalid_request",
    "budget_denied",
    "budget_expired",
]


@dataclass(frozen=True)
class ModelProfile:
    """Declarative model capability profile selected by task, role, stage, or budget policy."""

    profile_id: str
    provider_ref: str
    model: str
    api_format: str
    capability_tags: list[str]
    cost_tier: str
    default_parameters: dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelCallSpec:
    """Provider-ready model call configuration derived from a ModelProfile and provider adapter."""

    profile_id: str
    provider_ref: str
    api_format: str
    model: str
    parameters: dict[str, JSONValue]
    credential_ref: CredentialRef | None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["credential_ref"] = self.credential_ref.to_dict() if self.credential_ref else None
        return out


@dataclass(frozen=True)
class ModelResult:
    """Normalized provider response metadata used for runtime accounting and failure classification."""

    success: bool
    provider_ref: str
    model: str
    text: str | None
    usage: dict[str, float]
    cost: dict[str, float]
    error: str | None
    failover_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolPermissionSet:
    """Tool allowance contract passed from workflow stages to agent runtime adapters."""

    mode: str = "allow_all"
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolServerRef:
    """Resolved tool-server endpoint visible to an agent or panel role."""

    ref: str
    server_name: str
    transport: str = "legacy_inprocess"
    tool_names: list[str] = field(default_factory=list)
    credential_ref: CredentialRef | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "server_name": self.server_name,
            "transport": self.transport,
            "tool_names": self.tool_names,
            "credential_ref": self.credential_ref.to_dict() if self.credential_ref else None,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ToolCallResult:
    """Normalized result of an in-process or MCP-shaped tool invocation."""

    server_name: str
    tool_name: str
    success: bool
    output: JSONValue
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    failover_reason: str | None = None
    raw_is_error: bool = False
    redaction_hits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnvPolicy:
    """Scoped environment injection policy for runtime, tool, and subprocess execution."""

    redaction_required: bool = True
    exposed_env_keys: list[str] = field(default_factory=list)
    scoped_credential_refs: list[CredentialRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "redaction_required": self.redaction_required,
            "exposed_env_keys": self.exposed_env_keys,
            "scoped_credential_refs": [ref.to_dict() for ref in self.scoped_credential_refs],
        }


@dataclass(frozen=True)
class CachePolicy:
    """Runtime/provider cache strategy recorded for prompt-layout replay checks."""

    mode: str
    frozen_prefix_hash: str | None
    cache_breakpoints: list[str] = field(default_factory=list)
    runtime_cache_strategy: str | None = None
    provider_cache_strategy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentRunRequest:
    """Serializable request passed from workflow stages to an AgentRuntime adapter."""

    request_id: str
    run_id: str
    stage_id: str
    role_ref: str | None
    agent_runtime_ref: str
    prompt_ref: dict[str, Any]
    system_prompt_ref: dict[str, Any] | None
    cwd: str
    model_profile_ref: str
    model_call: ModelCallSpec
    tool_permissions: ToolPermissionSet
    tool_servers: list[dict[str, Any]]
    env_policy: EnvPolicy
    credential_ref: CredentialRef | None
    credential_mode: str
    budget_grant_id: str | None
    artifact_scope: str
    timeout_seconds: int
    cache_policy: CachePolicy
    runtime_options: dict[str, JSONValue] = field(default_factory=dict)
    role_skill_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "role_ref": self.role_ref,
            "agent_runtime_ref": self.agent_runtime_ref,
            "prompt_ref": self.prompt_ref,
            "system_prompt_ref": self.system_prompt_ref,
            "cwd": self.cwd,
            "model_profile_ref": self.model_profile_ref,
            "model_call": self.model_call.to_dict(),
            "tool_permissions": self.tool_permissions.to_dict(),
            "tool_servers": self.tool_servers,
            "env_policy": self.env_policy.to_dict(),
            "credential_ref": self.credential_ref.to_dict() if self.credential_ref else None,
            "credential_mode": self.credential_mode,
            "budget_grant_id": self.budget_grant_id,
            "artifact_scope": self.artifact_scope,
            "timeout_seconds": self.timeout_seconds,
            "cache_policy": self.cache_policy.to_dict(),
            "runtime_options": self.runtime_options,
            "role_skill_sha256": self.role_skill_sha256,
        }


@dataclass(frozen=True)
class AgentEvent:
    """Normalized stream event emitted by an AgentRuntime for trajectory and replay."""

    event_id: str
    run_id: str
    agent_run_id: str | None
    stage_id: str | None
    type: str
    payload: dict[str, JSONValue]
    artifact_refs: list[dict[str, Any]]
    credential_refs: list[CredentialRef]
    timestamp_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "agent_run_id": self.agent_run_id,
            "stage_id": self.stage_id,
            "type": self.type,
            "payload": self.payload,
            "artifact_refs": self.artifact_refs,
            "credential_refs": [ref.to_dict() for ref in self.credential_refs],
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass(frozen=True)
class ToolCallRecord:
    """Compact record of one runtime-observed tool call."""

    tool_call_id: str
    server_name: str
    tool_name: str
    started_at_ms: int
    finished_at_ms: int | None
    success: bool
    artifact_refs: list[dict[str, Any]]
    failover_reason: str | None


@dataclass(frozen=True)
class AgentRunResult:
    """Normalized terminal result of one agent runtime execution."""

    success: bool
    events: list[AgentEvent]
    text_output_refs: list[dict[str, Any]]
    tool_uses: list[ToolCallRecord]
    error: str | None
    failover_reason: str | None
    credential_ref: CredentialRef | None
    usage: dict[str, float] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    terminal_status: str | None = None
    timed_out: bool = False
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "events": [event.to_dict() for event in self.events],
            "text_output_refs": self.text_output_refs,
            "tool_uses": [asdict(tool) for tool in self.tool_uses],
            "error": self.error,
            "failover_reason": self.failover_reason,
            "credential_ref": self.credential_ref.to_dict() if self.credential_ref else None,
            "usage": self.usage,
            "cost": self.cost,
            "terminal_status": self.terminal_status,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
        }


@dataclass(frozen=True)
class BudgetRequest:
    """Serializable request for compute, wall-clock, token, data, or tool budget."""

    request_id: str
    requester_id: str
    experiment_id: str
    model_profile_ref: str | None
    requested: dict[str, float]
    expected_value: dict[str, JSONValue]
    evidence_refs: list[str]
    cheaper_alternatives: list[str]
    abort_conditions: list[str]


@dataclass(frozen=True)
class BudgetGrant:
    """Approved budget envelope that execution guards can enforce and meter."""

    grant_id: str
    approved: dict[str, float]
    conditions: list[str]
    expires_at_generation: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetDecision:
    """Policy output describing grant, deny, downscope, defer, or require-review decisions."""

    decision: str
    reason_codes: list[str]
    grant: BudgetGrant | None
    model_profile_override: str | None = None
    review_target: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["grant"] = self.grant.to_dict() if self.grant else None
        return out


# --------------------------------------------------------------------------- #
# Runtime sandbox intent — runtime-neutral operator request, honored or
# explicitly rejected by each AgentRuntime plugin's declared capabilities.
# --------------------------------------------------------------------------- #


SANDBOX_FILESYSTEM_VALUES: tuple[str, ...] = ("read_only", "workspace_write", "full")
"""Allowed values for :attr:`RuntimeSandboxIntent.filesystem`."""

SANDBOX_NETWORK_VALUES: tuple[str, ...] = ("off", "on")
"""Allowed values for :attr:`RuntimeSandboxIntent.network`."""

SANDBOX_APPROVAL_VALUES: tuple[str, ...] = ("auto", "on_risk", "always_ask")
"""Allowed values for :attr:`RuntimeSandboxIntent.approval`."""

SANDBOX_ENFORCEMENT_APPROVAL_GATE = "approval_gate"
"""Runtime enforces sandbox intent through interactive approval prompts only."""

SANDBOX_ENFORCEMENT_OS_SANDBOX = "os_sandbox"
"""Runtime enforces sandbox intent through OS-level isolation (seatbelt, landlock, ...)."""


FilesystemIntent = Literal["read_only", "workspace_write", "full"]
NetworkIntent = Literal["off", "on"]
ApprovalIntent = Literal["auto", "on_risk", "always_ask"]


@dataclass(frozen=True)
class RuntimeSandboxIntent:
    """Runtime-neutral sandbox intent declared by the operator or task project.

    Captures what the operator wants the agent runtime to allow.  Each
    :class:`AgentRuntime` plugin declares in its manifest which values
    of each axis it can honor; resolution fails fast when the intent
    contains a value the chosen runtime cannot enforce.

    The vocabulary is intentionally small: three coarse axes that map
    cleanly to most CLI agents' approval and sandbox flags.  Finer-grained
    needs are escape-hatched via per-runtime raw flags rather than
    growing this enum.

    Attributes:
        filesystem: Filesystem write scope intent.  ``"read_only"`` means
            the runtime should reject writes outside its own state;
            ``"workspace_write"`` permits writes inside the active run
            directory; ``"full"`` lets the runtime touch anywhere the
            process user can write.
        network: Network egress intent.  ``"off"`` requests OS-level
            network isolation; ``"on"`` permits outbound network calls.
        approval: Action-approval policy intent.  ``"auto"`` lets the
            runtime act without prompting; ``"on_risk"`` prompts before
            mutating operations; ``"always_ask"`` prompts before every
            action.
    """

    filesystem: FilesystemIntent
    network: NetworkIntent
    approval: ApprovalIntent

    def __post_init__(self) -> None:
        if self.filesystem not in SANDBOX_FILESYSTEM_VALUES:
            raise ValueError(
                f"filesystem must be one of {SANDBOX_FILESYSTEM_VALUES}; got {self.filesystem!r}"
            )
        if self.network not in SANDBOX_NETWORK_VALUES:
            raise ValueError(
                f"network must be one of {SANDBOX_NETWORK_VALUES}; got {self.network!r}"
            )
        if self.approval not in SANDBOX_APPROVAL_VALUES:
            raise ValueError(
                f"approval must be one of {SANDBOX_APPROVAL_VALUES}; got {self.approval!r}"
            )

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable view for trajectory and replay."""
        return {
            "filesystem": self.filesystem,
            "network": self.network,
            "approval": self.approval,
        }
