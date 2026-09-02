"""Shared environment-variable and provider-registry helpers for the CLI.

``PRAXIST_AGENT_SYSTEM`` selects an installed peer runtime. Public values map
directly to production ``agent_runtime`` plugins; operator-facing agent CLIs
are configured independently from research runtimes.
"""

from __future__ import annotations

import os

AGENT_SYSTEM_VALUES: tuple[str, ...] = (
    "claude_sdk",
    "codex_sdk",
)
"""Allowed values for ``PRAXIST_AGENT_SYSTEM`` and the ``--agent-system`` flag."""

AGENT_SYSTEM_TO_RUNTIME_REF: dict[str, str] = {
    "claude_sdk": "agent_runtime:claude_sdk",
    "codex_sdk": "agent_runtime:codex_sdk",
}
"""Canonical runtime plugin for each supported agent system."""

CODEX_NATIVE_DEFAULT_MODEL = "gpt-5.6-luna"
"""Default model for the saved-ChatGPT-login Codex-native profile."""


PROVIDER_KEY_MAP: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "brave": "BRAVE_API_KEY",
}
"""Known provider/tool credentials accepted by Praxist operator configuration."""

PROVIDER_REF_FOR_SHORT_NAME: dict[str, str] = {
    "anthropic": "model_provider:anthropic_messages",
    "openai": "model_provider:openai_compatible",
    "openrouter": "model_provider:openrouter",
    "deepseek": "model_provider:deepseek_alias",
    "groq": "model_provider:groq_alias",
    "mistral": "model_provider:mistral_alias",
    "xai": "model_provider:xai_alias",
}
"""Canonical plugin ref for every built-in model provider."""


PROVIDER_BASE_URL: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "mistral": "https://api.mistral.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "xai": "https://api.x.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def getenv(name: str, default: str = "") -> str:
    """Read an environment variable with a string fallback."""
    return os.environ.get(name, default)


def default_provider_for_agent_system(agent_system: str) -> str:
    """Return the natural model provider for a given agent system.

    Codex SDK is OpenAI-native; Claude SDK defaults to Anthropic.
    """
    return "openai" if agent_system == "codex_sdk" else "anthropic"


def agent_system_for_runtime_ref(runtime_ref: str) -> str | None:
    """Return the canonical agent system for a built-in runtime ref."""

    normalized = runtime_ref.strip()
    for agent_system, candidate in AGENT_SYSTEM_TO_RUNTIME_REF.items():
        if normalized == candidate:
            return agent_system
    return None
