"""Provider environment helpers for the research_loop plugin."""

from __future__ import annotations

from collections.abc import Mapping

OPENROUTER_CLAUDE_SDK_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_OPENAI_COMPAT_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_CLAUDE_SDK_BASE_URL = "https://api.deepseek.com/anthropic"
DEEPSEEK_CLAUDE_DEFAULT_MODEL = "deepseek-v4-pro[1m]"
DEEPSEEK_CLAUDE_DEFAULT_HAIKU_MODEL = "deepseek-v4-flash"
DEEPSEEK_CLAUDE_DEFAULT_EFFORT = "max"


def normalize_openrouter_base_url(base_url: str) -> str:
    """Return the OpenRouter base URL expected by Claude SDK transports.

    The Claude SDK appends its own ``/v1/messages`` path. OpenRouter's
    OpenAI-compatible endpoint includes ``/api/v1``, but Claude SDK calls must
    use the parent ``/api`` endpoint or requests become ``/api/v1/v1/messages``.
    """

    cleaned = str(base_url).rstrip("/")
    if cleaned == OPENROUTER_OPENAI_COMPAT_BASE_URL:
        return OPENROUTER_CLAUDE_SDK_BASE_URL
    return cleaned


def freeze_provider_env(model_provider_ref: str, env: Mapping[str, str]) -> dict[str, str | None]:
    """Capture the provider environment surface passed into research-loop runtimes."""

    base = {
        "PRAXIST_MODEL_PROVIDER_REF": model_provider_ref,
        "ANTHROPIC_API_KEY": None,
        "ANTHROPIC_BASE_URL": None,
        "ANTHROPIC_AUTH_TOKEN": None,
        "ANTHROPIC_MODEL": None,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": None,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": None,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": None,
        "CLAUDE_CODE_SUBAGENT_MODEL": None,
        "CLAUDE_CODE_EFFORT_LEVEL": None,
        "OPENROUTER_API_KEY": None,
        "OPENAI_API_KEY": None,
        "DEEPSEEK_API_KEY": None,
        "GROQ_API_KEY": None,
        "MISTRAL_API_KEY": None,
        "XAI_API_KEY": None,
    }
    if model_provider_ref == "model_provider:openrouter":
        base_url = normalize_openrouter_base_url(
            env.get("ANTHROPIC_BASE_URL")
            or env.get("OPENROUTER_BASE_URL")
            or OPENROUTER_CLAUDE_SDK_BASE_URL
        )
        auth_token = env.get("ANTHROPIC_AUTH_TOKEN") or env.get("OPENROUTER_API_KEY")
        return {
            **base,
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "OPENROUTER_API_KEY": auth_token,
        }
    if model_provider_ref == "model_provider:anthropic_messages":
        return {
            **base,
            "ANTHROPIC_API_KEY": env.get("ANTHROPIC_API_KEY"),
        }
    if model_provider_ref == "model_provider:openai_compatible":
        return {**base, "OPENAI_API_KEY": env.get("OPENAI_API_KEY")}
    if model_provider_ref == "model_provider:groq_alias":
        return {**base, "GROQ_API_KEY": env.get("GROQ_API_KEY")}
    if model_provider_ref == "model_provider:mistral_alias":
        return {**base, "MISTRAL_API_KEY": env.get("MISTRAL_API_KEY")}
    if model_provider_ref == "model_provider:xai_alias":
        return {**base, "XAI_API_KEY": env.get("XAI_API_KEY")}
    if model_provider_ref == "model_provider:deepseek_alias":
        auth_token = env.get("DEEPSEEK_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")
        return {
            **base,
            "ANTHROPIC_BASE_URL": env.get("DEEPSEEK_ANTHROPIC_BASE_URL")
            or DEEPSEEK_CLAUDE_SDK_BASE_URL,
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "DEEPSEEK_API_KEY": env.get("DEEPSEEK_API_KEY") or auth_token,
            "ANTHROPIC_MODEL": env.get("ANTHROPIC_MODEL") or DEEPSEEK_CLAUDE_DEFAULT_MODEL,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": env.get("ANTHROPIC_DEFAULT_OPUS_MODEL")
            or DEEPSEEK_CLAUDE_DEFAULT_MODEL,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": env.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
            or DEEPSEEK_CLAUDE_DEFAULT_MODEL,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL")
            or DEEPSEEK_CLAUDE_DEFAULT_HAIKU_MODEL,
            "CLAUDE_CODE_SUBAGENT_MODEL": env.get("CLAUDE_CODE_SUBAGENT_MODEL")
            or DEEPSEEK_CLAUDE_DEFAULT_HAIKU_MODEL,
            "CLAUDE_CODE_EFFORT_LEVEL": env.get("CLAUDE_CODE_EFFORT_LEVEL")
            or DEEPSEEK_CLAUDE_DEFAULT_EFFORT,
        }
    if model_provider_ref == "model_provider:fake_provider":
        return base
    raise ValueError(f"Unsupported model provider env contract: {model_provider_ref}")
