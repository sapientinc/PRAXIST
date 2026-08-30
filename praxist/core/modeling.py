"""ModelProfile helpers and registry-backed ModelProvider dispatch."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import yaml

from praxist.core.credentials import CredentialRef
from praxist.core.protocol import ModelCallSpec, ModelProfile, ModelResult
from praxist.core.registry import (
    PluginLoader,
    PluginRef,
    PluginRegistry,
    PluginRoots,
    require_execution_plugin,
)


def validate_model_for_provider(
    provider_ref: str,
    model: str,
    registry: PluginRegistry | None = None,
) -> None:
    """Validate that a concrete model name is compatible with the selected provider adapter."""
    if not model:
        return
    contract = _provider_contract(provider_ref, registry)
    patterns = [str(item) for item in contract.get("compatible_model_patterns") or []]
    if not patterns:
        return
    for pattern in patterns:
        if fnmatch.fnmatchcase(model, pattern):
            return
    raise ValueError(f"model {model!r} is not compatible with {provider_ref}")


def normalize_model_for_provider(
    provider_ref: str,
    model: str,
    registry: PluginRegistry | None = None,
) -> str:
    """Reshape a model name into the format the provider's api_format expects.

    OpenRouter and OrcaRouter use ``vendor/model``; the other api_formats
    we ship (``openai_compatible``, ``anthropic_messages``, ``fake``)
    consume bare ``model`` names.  Operator configs and env vars often
    carry the ``vendor/`` prefix as a leftover from the era when
    openrouter was the only supported access path; this helper strips
    that prefix when the resolved provider is not openrouter or orcarouter
    so the same configuration works against either an aggregator or a
    direct provider endpoint.
    """
    if not model:
        return model
    contract = _provider_contract(provider_ref, registry)
    api_format = str(contract.get("api_format") or "openai_compatible")
    return _normalize_model_for_api_format(model, api_format)


def _normalize_model_for_api_format(model: str, api_format: str) -> str:
    """Strip a ``vendor/`` prefix when the api_format does not use one."""
    if not model or api_format in {"openrouter", "orcarouter"}:
        return model
    if "/" in model:
        return model.split("/", 1)[1]
    return model


class ModelProviderAdapter:
    """Small provider contract used by core before a full provider SDK call is made."""

    def __init__(self, provider_ref: str, api_format: str | None = None) -> None:
        self.provider_ref = provider_ref
        self.api_format = api_format or _provider_contract(provider_ref).get(
            "api_format", "openai_compatible"
        )

    def build_call(
        self,
        profile: ModelProfile,
        *,
        credential_ref: CredentialRef | None,
        runtime_options: dict[str, Any] | None = None,
    ) -> ModelCallSpec:
        parameters = dict(profile.default_parameters)
        parameters.update(runtime_options or {})
        return ModelCallSpec(
            profile_id=profile.profile_id,
            provider_ref=profile.provider_ref,
            api_format=profile.api_format,
            model=_normalize_model_for_api_format(profile.model, profile.api_format),
            parameters=parameters,
            credential_ref=credential_ref,
        )

    def normalize_result(self, raw_redacted: dict[str, Any]) -> ModelResult:
        if raw_redacted.get("error"):
            return ModelResult(
                success=False,
                provider_ref=self.provider_ref,
                model=str(raw_redacted.get("model", "")),
                text=None,
                usage={},
                error=str(raw_redacted.get("error")),
                failover_reason=self.classify_error(raw_redacted),
            )
        usage = raw_redacted.get("usage")
        return ModelResult(
            success=True,
            provider_ref=self.provider_ref,
            model=str(raw_redacted.get("model", "")),
            text=str(raw_redacted.get("text", "")),
            usage={str(key): float(value) for key, value in dict(usage or {}).items()},
            error=None,
            failover_reason="none",
        )

    def classify_error(self, raw_error_redacted: dict[str, Any]) -> str:
        status = raw_error_redacted.get("status")
        code = str(raw_error_redacted.get("code", "")).lower()
        message = str(
            raw_error_redacted.get("message", raw_error_redacted.get("error", ""))
        ).lower()
        if status in (401, 403) or "auth" in code or "unauthorized" in message:
            return "auth_error"
        if "quota" in code or "quota" in message or "insufficient_quota" in message:
            return "quota_exhausted"
        if status == 429 or "rate" in code or "rate limit" in message:
            return "rate_limited"
        if "timeout" in code or "timeout" in message:
            return "timeout"
        if status in (500, 502, 503, 504) or "unavailable" in code:
            return "provider_unavailable"
        if status == 400 or "invalid" in code:
            return "invalid_request"
        return "runtime_error"


def provider_for_ref(
    provider_ref: str, registry: PluginRegistry | None = None
) -> ModelProviderAdapter:
    """Return the model provider adapter for a registry plugin reference."""
    registry = _registry_with_provider(provider_ref, registry)
    require_execution_plugin(
        registry,
        provider_ref,
        kind="model_provider",
    )
    if registry is not None:
        parsed = provider_ref.split(":", 1)
        if len(parsed) == 2:
            plugin = registry.require(parsed[0], parsed[1])
            if hasattr(plugin, "build_call") and hasattr(plugin, "normalize_result"):
                return plugin
    return ModelProviderAdapter(provider_ref)


def default_model_profile(
    provider_ref: str,
    *,
    profile_id: str = "cheap_peer",
    cost_tier: str = "cheap",
    model: str | None = None,
    registry: PluginRegistry | None = None,
) -> ModelProfile:
    """Create the default ModelProfile selected for the current model provider and model name."""
    contract = _provider_contract(provider_ref, registry)
    api_format = str(contract.get("api_format") or "openai_compatible")
    profile_defaults = (
        contract.get("model_profiles") if isinstance(contract.get("model_profiles"), dict) else {}
    )
    resolved_model = model or str(
        profile_defaults.get(profile_id) or contract.get("default_model") or "unknown-model"
    )
    return ModelProfile(
        profile_id=profile_id,
        provider_ref=provider_ref,
        model=_normalize_model_for_api_format(resolved_model, api_format),
        api_format=api_format,
        capability_tags=_capability_tags(api_format),
        cost_tier=cost_tier,
        default_parameters={"temperature": 0},
    )


def model_profiles_snapshot(
    *,
    provider_ref: str,
    runtime_ref: str,
    credential_mode: str,
    cache_policy: Any,
    selected_model: str | None = None,
    registry: PluginRegistry | None = None,
) -> dict[str, Any]:
    """Build the run-local snapshot written to model_profiles.json for replay and cost attribution."""
    provider_contract = _provider_contract(provider_ref, registry)
    cheap = default_model_profile(
        provider_ref,
        profile_id="cheap_peer",
        cost_tier="cheap",
        model=selected_model,
        registry=registry,
    )
    strong = default_model_profile(
        provider_ref,
        profile_id="strong_reasoner",
        cost_tier="premium",
        model=selected_model,
        registry=registry,
    )
    return {
        "schema_version": "praxist.model_profiles.v1",
        "profiles": {
            cheap.profile_id: cheap.to_dict(),
            strong.profile_id: strong.to_dict(),
        },
        "selected_defaults": {
            "research_loop": "cheap_peer",
        },
        "provider_adapters": {
            provider_ref: provider_for_ref(provider_ref, registry=registry).api_format
        },
        "runtime_ref": runtime_ref,
        "routing_policy_ref": "core:model_routing_static",
        "credential_mode": credential_mode,
        "cache_policy": cache_policy.to_dict(),
        "provider_contract": {
            "provider_ref": provider_ref,
            "api_format": provider_contract.get("api_format"),
            "cache_strategy": provider_contract.get("cache_strategy"),
            "usage_reporting": provider_contract.get("usage_reporting"),
            "compatible_model_patterns": provider_contract.get("compatible_model_patterns") or [],
        },
    }


def _capability_tags(api_format: str) -> list[str]:
    if api_format == "fake":
        return ["offline_fixture", "deterministic"]
    if api_format == "anthropic_messages":
        return ["long_context", "prompt_cache", "tool_use"]
    if api_format in {"openrouter", "orcarouter"}:
        return ["routing", "openai_compatible", "tool_use"]
    return ["openai_compatible", "tool_use"]


def _load_single_provider_registry(ref: str) -> PluginRegistry:
    parsed = PluginRef.parse(ref)
    if parsed.kind != "model_provider":
        raise ValueError(f"Model provider ref must use kind model_provider: {ref}")
    loader = PluginLoader(PluginRoots.defaults())
    manifest = loader.resolve(
        [ref],
        run_id="model_provider_spec",
        root_task_ref=ref,
        enforce_bundled_execution=True,
    )
    return loader.load(manifest)


def _registry_with_provider(
    provider_ref: str,
    registry: PluginRegistry | None,
) -> PluginRegistry:
    """Return a registry containing the requested provider descriptor.

    Some runtime resume paths intentionally carry a stage-local registry that
    does not include model-provider descriptors.  Load only the requested
    bundled provider in that case; malformed or unknown providers still fail
    through the normal loader rather than being silently ignored.
    """

    if registry is None:
        return _load_single_provider_registry(provider_ref)
    parsed = PluginRef.parse(provider_ref)
    try:
        descriptor = getattr(registry, "descriptor", None)
        if callable(descriptor):
            descriptor(parsed.kind, parsed.name)
        else:
            registry.require(parsed.kind, parsed.name)
    except KeyError:
        return _load_single_provider_registry(provider_ref)
    return registry


def _provider_contract(provider_ref: str, registry: PluginRegistry | None = None) -> dict[str, Any]:
    registry = _registry_with_provider(provider_ref, registry)
    selected = require_execution_plugin(registry, provider_ref, kind="model_provider")
    if selected is None:
        return {}
    value = yaml.safe_load((Path(selected.path) / "plugin.yaml").read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        return {}
    provider = value.get("provider")
    return dict(provider) if isinstance(provider, dict) else {}


def provider_default_model(provider_ref: str, registry: PluginRegistry | None = None) -> str | None:
    """Return the plugin yaml's ``default_model`` for ``provider_ref``.

    Single source-of-truth lookup for "what model does this provider
    pick when the operator did not supply one?". Returns ``None``
    when the plugin cannot be loaded or has no ``default_model``
    declared; callers should fall back to a built-in literal
    (e.g. :data:`praxist.core.run_config.DEFAULT_AGENT_MODEL`).

    Centralising this here lets the yaml be the only place an
    operator has to edit to change a provider's default — no parallel
    hardcoded chain in workflow-stage startup code (see #144).
    """
    try:
        contract = _provider_contract(provider_ref, registry)
    except Exception:  # pragma: no cover - plugin loader edge cases
        return None
    value = contract.get("default_model")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
