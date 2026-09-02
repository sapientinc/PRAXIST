# API Providers

`model_provider:*` API provider plugins describe API shape, model defaults,
credential requirements, cache capability, and route-specific compatibility.
Agent runtime plugins execute the agent loops.

## Built-In API Provider Shapes

- `model_provider:openrouter` for OpenRouter-routed model names.
- `model_provider:openai_compatible` for OpenAI-compatible endpoints.
- `model_provider:anthropic_messages` for native Anthropic Messages style.
- `model_provider:deepseek_alias` for DeepSeek-compatible aliases.
- `model_provider:groq_alias` for Groq OpenAI-compatible aliases.
- `model_provider:mistral_alias` for Mistral OpenAI-compatible aliases.
- `model_provider:xai_alias` for xAI OpenAI-compatible aliases.

API provider names represent API format and routing. A task or operator may
override the `ModelProfile` used by a stage.

## API Provider Manifest Expectations

An API provider manifest should declare:

- supported API format;
- default model, if any;
- endpoint base, if fixed;
- required credential refs;
- cache capability;
- usage reporting capability;
- compatibility with agent runtime plugins.

## Multi-Model Runs

Research-loop agents may use different model profiles when the task contract
and selected runtime support them. Peer exploration and planning roles should
resolve providers and model names through the same configuration boundary.

Do not hard-code a task-specific model inside a generic API provider plugin.

Run-wide reasoning effort belongs to the agent runtime policy. Configure it
under `agent.reasoning_effort` as documented in
[Agent Runtimes](agent-runtimes.md#reasoning-policy); adapters translate that
single policy to each API provider's supported wire contract. The default is
`max`; select `auto` explicitly to retain an API provider's native effort default.

## Provider Conformance

API provider tests should cover:

- credential resolution and redaction;
- API provider/agent runtime compatibility;
- cache capability mapping;
- missing or invalid key diagnostics;
- usage unknown behavior when the API provider does not return metering.
