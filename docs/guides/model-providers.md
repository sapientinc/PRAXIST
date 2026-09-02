# API Providers

`model_provider:*` API provider plugins describe API shape, model defaults,
credential requirements, cache capability, and route-specific compatibility.
Agent runtime plugins execute the agent loops.

## Built-In API Provider Shapes

- `model_provider:openrouter` for OpenRouter-routed model names.
- `model_provider:orcarouter` for OrcaRouter-routed model names.
- `model_provider:openai_compatible` for OpenAI-compatible endpoints.
- `model_provider:anthropic_messages` for native Anthropic Messages style.
- `model_provider:deepseek_alias` for DeepSeek-compatible aliases.
- `model_provider:cloudflare` for Cloudflare Workers AI.
- `model_provider:groq_alias` for Groq OpenAI-compatible aliases.
- `model_provider:mistral_alias` for Mistral OpenAI-compatible aliases.
- `model_provider:xai_alias` for xAI OpenAI-compatible aliases.

### Cloudflare Workers AI

Workers AI is OpenAI-compatible only; it publishes no Anthropic Messages
route, so it is compatible with `agent_runtime:codex_sdk` and not with
`agent_runtime:claude_sdk`. Its endpoint base is account-scoped rather than
fixed: `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1`,
interpolated from `CLOUDFLARE_ACCOUNT_ID`. Set `CLOUDFLARE_BASE_URL` to
override the whole base URL, and `CLOUDFLARE_API_KEY` for the bearer
credential.

The manifest declares the `cloudflare_workers_ai` API format rather than
`openai_compatible` for one reason: Workers AI model ids are full
`@cf/vendor/model` paths, and the generic OpenAI-compatible normalization
strips everything before the first `/`. The wire protocol is otherwise
plain OpenAI chat-completions.

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
