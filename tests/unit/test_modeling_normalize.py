"""Unit tests for provider-aware model-name normalization.

When the project was openrouter-only, operator configs and env vars
stored model names in the ``vendor/model`` form openrouter expects.
Now that codex_sdk + direct provider endpoints (DeepSeek, Moonshot, …)
are also wired through, the same configured name must reshape to the
provider's native format — otherwise codex sends ``deepseek/foo`` to
the DeepSeek Chat Completions API and the upstream rejects it.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from praxist.core import modeling
from praxist.core.credentials import CredentialRef


class NormalizeModelForApiFormatTest(unittest.TestCase):
    """The private helper that takes api_format directly."""

    def test_openrouter_keeps_vendor_prefix(self) -> None:
        self.assertEqual(
            modeling._normalize_model_for_api_format("anthropic/claude-opus-4.7", "openrouter"),
            "anthropic/claude-opus-4.7",
        )

    def test_openrouter_keeps_bare_name_unchanged(self) -> None:
        # Bare name into openrouter is the operator's responsibility;
        # we never invent a prefix.
        self.assertEqual(
            modeling._normalize_model_for_api_format("claude-opus-4.7", "openrouter"),
            "claude-opus-4.7",
        )

    def test_openai_compatible_strips_vendor_prefix(self) -> None:
        self.assertEqual(
            modeling._normalize_model_for_api_format(
                "deepseek/deepseek-v4-pro", "openai_compatible"
            ),
            "deepseek-v4-pro",
        )

    def test_anthropic_messages_strips_vendor_prefix(self) -> None:
        self.assertEqual(
            modeling._normalize_model_for_api_format(
                "anthropic/claude-opus-4.7", "anthropic_messages"
            ),
            "claude-opus-4.7",
        )

    def test_fake_strips_vendor_prefix(self) -> None:
        self.assertEqual(
            modeling._normalize_model_for_api_format("fake/fake-deterministic", "fake"),
            "fake-deterministic",
        )

    def test_already_bare_passes_through(self) -> None:
        self.assertEqual(
            modeling._normalize_model_for_api_format("deepseek-v4-pro", "openai_compatible"),
            "deepseek-v4-pro",
        )

    def test_empty_model_is_returned_unchanged(self) -> None:
        self.assertEqual(modeling._normalize_model_for_api_format("", "openai_compatible"), "")

    def test_only_first_slash_is_treated_as_prefix(self) -> None:
        # A model name containing multiple slashes only loses its first segment.
        self.assertEqual(
            modeling._normalize_model_for_api_format("deepseek/family/v4-pro", "openai_compatible"),
            "family/v4-pro",
        )


class NormalizeModelForProviderTest(unittest.TestCase):
    """Public entry point that resolves api_format from the provider contract."""

    def test_normalize_uses_provider_contract_api_format(self) -> None:
        with patch.object(
            modeling,
            "_provider_contract",
            return_value={"api_format": "openai_compatible"},
        ):
            self.assertEqual(
                modeling.normalize_model_for_provider(
                    "model_provider:deepseek_alias",
                    "deepseek/deepseek-v4-pro",
                ),
                "deepseek-v4-pro",
            )

    def test_normalize_preserves_prefix_when_provider_is_openrouter(self) -> None:
        with patch.object(
            modeling,
            "_provider_contract",
            return_value={"api_format": "openrouter"},
        ):
            self.assertEqual(
                modeling.normalize_model_for_provider(
                    "model_provider:openrouter",
                    "anthropic/claude-opus-4.7",
                ),
                "anthropic/claude-opus-4.7",
            )

    def test_normalize_defaults_to_openai_compatible_when_api_format_missing(self) -> None:
        with patch.object(modeling, "_provider_contract", return_value={}):
            self.assertEqual(
                modeling.normalize_model_for_provider(
                    "model_provider:unknown",
                    "vendor/bare",
                ),
                "bare",
            )

    def test_normalize_empty_model_returns_empty(self) -> None:
        # No provider lookup required for an empty model.
        with patch.object(modeling, "_provider_contract") as contract:
            self.assertEqual(
                modeling.normalize_model_for_provider("model_provider:x", ""),
                "",
            )
            contract.assert_not_called()

    def test_partial_registry_without_descriptor_remains_compatible(self) -> None:
        class LegacyRegistry:
            def require(self, kind: str, name: str) -> object:
                self.required = (kind, name)
                return object()

        registry = LegacyRegistry()
        with patch.object(modeling, "_load_single_provider_registry") as loader:
            selected = modeling._registry_with_provider(
                "model_provider:custom",
                registry,
            )

        self.assertIs(selected, registry)
        self.assertEqual(registry.required, ("model_provider", "custom"))
        loader.assert_not_called()


class DefaultModelProfileNormalizesTest(unittest.TestCase):
    """``default_model_profile`` applies normalization to its constructed ModelProfile.model."""

    def test_default_model_profile_strips_prefix_for_openai_compatible(self) -> None:
        with patch.object(
            modeling,
            "_provider_contract",
            return_value={"api_format": "openai_compatible"},
        ):
            profile = modeling.default_model_profile(
                "model_provider:deepseek_alias",
                model="deepseek/deepseek-v4-pro",
            )
        self.assertEqual(profile.model, "deepseek-v4-pro")
        self.assertEqual(profile.api_format, "openai_compatible")

    def test_default_model_profile_keeps_prefix_for_openrouter(self) -> None:
        with patch.object(
            modeling,
            "_provider_contract",
            return_value={"api_format": "openrouter"},
        ):
            profile = modeling.default_model_profile(
                "model_provider:openrouter",
                model="anthropic/claude-opus-4.7",
            )
        self.assertEqual(profile.model, "anthropic/claude-opus-4.7")

    def test_default_model_profile_normalizes_contract_default(self) -> None:
        # Even if a provider's manifest accidentally carries a vendor/
        # prefix in default_model, normalization corrects it.
        with patch.object(
            modeling,
            "_provider_contract",
            return_value={
                "api_format": "openai_compatible",
                "default_model": "deepseek/deepseek-chat",
            },
        ):
            profile = modeling.default_model_profile("model_provider:deepseek_alias")
        self.assertEqual(profile.model, "deepseek-chat")


class ModelProviderAdapterBuildCallNormalizesTest(unittest.TestCase):
    """``ModelProviderAdapter.build_call`` re-applies normalization defensively."""

    def test_build_call_strips_prefix_when_profile_has_one(self) -> None:
        adapter = modeling.ModelProviderAdapter(
            "model_provider:deepseek_alias",
            api_format="openai_compatible",
        )
        profile = modeling.ModelProfile(
            profile_id="cheap_peer",
            provider_ref="model_provider:deepseek_alias",
            # Simulate a stale ModelProfile that escaped normalization.
            model="deepseek/deepseek-v4-pro",
            api_format="openai_compatible",
            capability_tags=["openai_compatible", "tool_use"],
            cost_tier="cheap",
            default_parameters={"temperature": 0},
        )
        call = adapter.build_call(profile, credential_ref=None)
        self.assertEqual(call.model, "deepseek-v4-pro")

    def test_build_call_keeps_prefix_for_openrouter_profile(self) -> None:
        adapter = modeling.ModelProviderAdapter(
            "model_provider:openrouter",
            api_format="openrouter",
        )
        profile = modeling.ModelProfile(
            profile_id="strong_reasoner",
            provider_ref="model_provider:openrouter",
            model="anthropic/claude-opus-4.7",
            api_format="openrouter",
            capability_tags=["routing", "openai_compatible", "tool_use"],
            cost_tier="premium",
            default_parameters={"temperature": 0},
        )
        call = adapter.build_call(profile, credential_ref=None)
        self.assertEqual(call.model, "anthropic/claude-opus-4.7")


class CredentialRefImportSmokeTest(unittest.TestCase):
    """Sanity check that CredentialRef import path is wired (avoids dead-import lint)."""

    def test_credential_ref_optional_argument(self) -> None:
        adapter = modeling.ModelProviderAdapter(
            "model_provider:openai_compatible",
            api_format="openai_compatible",
        )
        profile = modeling.ModelProfile(
            profile_id="cheap_peer",
            provider_ref="model_provider:openai_compatible",
            model="gpt-5.2",
            api_format="openai_compatible",
            capability_tags=["openai_compatible", "tool_use"],
            cost_tier="cheap",
            default_parameters={"temperature": 0},
        )
        ref = CredentialRef(
            scope="model_provider",
            provider="openai",
            target_ref="model_provider:openai_compatible",
            key_id="openai:env:abc",
            source="test",
        )
        call = adapter.build_call(profile, credential_ref=ref)
        self.assertEqual(call.credential_ref, ref)


class ProviderDefaultModelTest(unittest.TestCase):
    """#144: ``provider_default_model`` returns yaml's ``default_model``."""

    def test_returns_default_model_from_plugin_yaml(self) -> None:
        # Production provider yamls: assert the yaml-driven lookup
        # returns the same value the yaml declares.
        cases = {
            "model_provider:anthropic_messages": "claude-opus-4-7",
            "model_provider:openrouter": "anthropic/claude-opus-4.7",
            "model_provider:openai_compatible": "gpt-5.2",
            "model_provider:deepseek_alias": "deepseek-v4-pro[1m]",
            "model_provider:groq_alias": "llama-3.3-70b-versatile",
            "model_provider:mistral_alias": "mistral-large-latest",
            "model_provider:xai_alias": "grok-2-latest",
        }
        for ref, expected in cases.items():
            with self.subTest(provider=ref):
                self.assertEqual(modeling.provider_default_model(ref), expected)

    def test_returns_none_for_unknown_provider(self) -> None:
        self.assertIsNone(modeling.provider_default_model("model_provider:does_not_exist_42"))

    def test_returns_none_when_contract_missing_default_model(self) -> None:
        with patch.object(modeling, "_provider_contract", return_value={}):
            self.assertIsNone(modeling.provider_default_model("model_provider:any"))

    def test_strips_surrounding_whitespace(self) -> None:
        with patch.object(
            modeling,
            "_provider_contract",
            return_value={"default_model": "  gpt-5.2  "},
        ):
            self.assertEqual(
                modeling.provider_default_model("model_provider:any"),
                "gpt-5.2",
            )

    def test_treats_non_string_default_as_missing(self) -> None:
        with patch.object(
            modeling,
            "_provider_contract",
            return_value={"default_model": 42},
        ):
            self.assertIsNone(modeling.provider_default_model("model_provider:any"))

    def test_partial_stage_registry_loads_only_requested_bundled_provider(self) -> None:
        from praxist.core.registry import PluginRegistry

        contract = modeling._provider_contract(
            "model_provider:openrouter",
            PluginRegistry({}, {}),
        )

        self.assertEqual(contract["api_format"], "openrouter")

    def test_partial_stage_registry_does_not_hide_unknown_provider(self) -> None:
        from praxist.core.registry import PluginRegistry

        with self.assertRaisesRegex(ValueError, "Required plugin not found"):
            modeling._provider_contract(
                "model_provider:does_not_exist_42",
                PluginRegistry({}, {}),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
