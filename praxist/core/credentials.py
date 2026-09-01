"""CredentialRef skeleton and redacted env discovery."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class CredentialRef:
    """Stable, redacted reference to one provider credential without storing the raw secret."""

    scope: str
    provider: str
    target_ref: str | None
    key_id: str
    source: str
    status: str = "active"

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class CredentialSet:
    """Resolved credential inventory for one provider, including robust-mode fallback order."""

    mode: str
    credentials: list[CredentialRef]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "credentials": [credential.to_dict() for credential in self.credentials],
        }


class CredentialResolver:
    """Discover credentials without exposing raw secret values."""

    ENV_SPECS: tuple[tuple[str, str, str, str | None], ...] = (
        (
            "ANTHROPIC_API_KEY",
            "model_provider",
            "anthropic_messages",
            "model_provider:anthropic_messages",
        ),
        ("OPENROUTER_API_KEY", "model_provider", "openrouter", "model_provider:openrouter"),
        ("ORCAROUTER_API_KEY", "model_provider", "orcarouter", "model_provider:orcarouter"),
        (
            "OPENAI_API_KEY",
            "model_provider",
            "openai_compatible",
            "model_provider:openai_compatible",
        ),
        ("DEEPSEEK_API_KEY", "model_provider", "deepseek_alias", "model_provider:deepseek_alias"),
        ("CLOUDFLARE_API_KEY", "model_provider", "cloudflare", "model_provider:cloudflare"),
        (
            "SEMANTIC_SCHOLAR_API_KEY",
            "tool_server",
            "semantic_scholar",
            "tool_server:semantic_scholar_mcp",
        ),
    )

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self._env = env if env is not None else os.environ

    def discover(self, profile: str | None = None) -> CredentialSet:
        if profile == "fake_multi_key":
            credentials = [
                self._fake_ref("model_provider", "fake_provider", "fake_provider", "A"),
                self._fake_ref("model_provider", "fake_provider", "fake_provider", "B"),
            ]
            return CredentialSet(mode="robust", credentials=credentials)

        credentials: list[CredentialRef] = []
        for env_name, scope, provider, target_ref in self.ENV_SPECS:
            raw = self._env.get(env_name)
            if raw:
                credentials.append(
                    CredentialRef(
                        scope=scope,
                        provider=provider,
                        target_ref=target_ref,
                        key_id=self._key_id(env_name, provider, raw),
                        source="env",
                    )
                )
        mode = "robust" if len(credentials) > 1 else "single"
        return CredentialSet(mode=mode, credentials=credentials)

    def snapshot(self, credential_set: CredentialSet) -> dict[str, object]:
        scopes: dict[str, list[str]] = {}
        for credential in credential_set.credentials:
            scopes.setdefault(credential.scope, []).append(credential.key_id)
        return {
            "schema_version": "praxist.credentials.v1",
            "credential_profiles": [c.to_dict() for c in credential_set.credentials],
            "robust_mode_enabled": credential_set.mode == "robust",
            "scopes": scopes,
            "raw_secret_fields_present": False,
        }

    @staticmethod
    def _key_id(env_name: str, provider: str, raw: str) -> str:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        # Do not include the env var name: scanners intentionally flag
        # API_KEY/TOKEN-looking strings followed by a value separator.
        _ = env_name
        return f"{provider}:env:{digest}"

    @staticmethod
    def _fake_ref(scope: str, provider: str, target_ref: str, suffix: str) -> CredentialRef:
        return CredentialRef(
            scope=scope,
            provider=provider,
            target_ref=f"{scope}:{target_ref}",
            key_id=f"{provider}:fake:{suffix}",
            source="runtime_profile",
        )


class CredentialFailoverManager:
    """Minimal robust-mode selector with per-key failure state."""

    FAILOVER_REASONS = {
        "auth_error",
        "quota_exhausted",
        "rate_limited",
        "timeout",
        "provider_unavailable",
        "runtime_error",
    }

    def __init__(self, credential_set: CredentialSet) -> None:
        self.credential_set = credential_set
        self._status: dict[str, str] = {
            credential.key_id: credential.status for credential in credential_set.credentials
        }
        self._failures: list[dict[str, str]] = []

    def select(
        self,
        *,
        scope: str,
        provider: str | None = None,
        target_ref: str | None = None,
    ) -> CredentialRef | None:
        for credential in self.credential_set.credentials:
            if not self._matches(credential, scope=scope, provider=provider, target_ref=target_ref):
                continue
            if self._status.get(credential.key_id, credential.status) == "active":
                return credential
        return None

    def record_failure(self, credential: CredentialRef, reason: str) -> CredentialRef | None:
        status = "exhausted" if reason in {"auth_error", "quota_exhausted"} else "cooling_down"
        self._status[credential.key_id] = status
        self._failures.append({"key_id": credential.key_id, "reason": reason, "status": status})
        if self.credential_set.mode != "robust" or reason not in self.FAILOVER_REASONS:
            return None
        return self.select(
            scope=credential.scope, provider=credential.provider, target_ref=credential.target_ref
        )

    def active_ref(self, credential: CredentialRef) -> CredentialRef:
        return replace(credential, status=self._status.get(credential.key_id, credential.status))

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": "praxist.credential_failover.v1",
            "mode": self.credential_set.mode,
            "statuses": dict(self._status),
            "failures": list(self._failures),
        }

    @staticmethod
    def _matches(
        credential: CredentialRef,
        *,
        scope: str,
        provider: str | None,
        target_ref: str | None,
    ) -> bool:
        if credential.scope != scope:
            return False
        if provider is not None and credential.provider != provider:
            return False
        return not (target_ref is not None and credential.target_ref not in (None, target_ref))


def provider_name_from_ref(model_provider_ref: str) -> str:
    """Extract the provider name component from a model-provider plugin reference."""
    if not model_provider_ref.startswith("model_provider:"):
        raise ValueError(f"Expected model_provider ref, got {model_provider_ref!r}")
    return model_provider_ref.split(":", 1)[1]


def find_model_provider_credential(
    credential_set: CredentialSet,
    model_provider_ref: str,
) -> CredentialRef | None:
    """Find the best available credential set for a selected model provider."""
    provider = provider_name_from_ref(model_provider_ref)
    for credential in credential_set.credentials:
        if credential.scope != "model_provider":
            continue
        if credential.provider != provider:
            continue
        if credential.target_ref not in (None, model_provider_ref):
            continue
        if credential.status == "active":
            return credential
    return None


def require_model_provider_credential(
    credential_set: CredentialSet,
    model_provider_ref: str,
) -> CredentialRef | None:
    """Resolve a provider credential or fail fast with an operator-facing error."""
    credential = find_model_provider_credential(credential_set, model_provider_ref)
    if credential is not None:
        return credential
    if model_provider_ref == "model_provider:fake_provider":
        return None
    raise ValueError(f"{model_provider_ref} requires a matching active model provider credential")
