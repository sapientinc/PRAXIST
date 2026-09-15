# Credentials

Credentials are resolved by Python startup code and represented by redacted
credential references. Shell wrappers do not own credential behavior.

## API Provider Credentials

Set an API key for the selected API provider before starting a run. The
maintained direct DeepSeek V4 Pro route uses `model_provider:deepseek_alias`
with `agent_runtime:claude_sdk`:

```bash
export DEEPSEEK_API_KEY=...
praxist start --model-provider model_provider:deepseek_alias \
  --runtime agent_runtime:claude_sdk \
  --model deepseek-v4-pro
```

See [Open-Source Model APIs](open-source-model-apis.md) for route-selection
criteria. Credential handling is identical regardless of which API-backed
profile the operator selects.

When an operator explicitly selects `agent_runtime:codex_sdk`, the same
`DEEPSEEK_API_KEY` is scoped to a private run-local `codex-relay` because
DeepSeek exposes Chat Completions and the Codex app-server expects Responses.
OpenAI uses `OPENAI_API_KEY` directly without the relay. Neither path stores a
raw key in task files or human Codex CLI sessions.

### Codex-native mode authentication

Codex-native mode uses `agent_runtime:codex_sdk` with a saved ChatGPT login for
`model_provider:openai_compatible` when `OPENAI_API_KEY` is absent:

```bash
praxist setup --profile codex-native --install-skills codex
praxist start \
  --codex-native \
  --agent-system codex_sdk \
  --model-provider model_provider:openai_compatible \
  --task-path /path/to/task-project
```

The setup command verifies the Codex binary pinned inside the installed SDK,
not an unrelated executable found first on `PATH`. If that binary currently
uses API-key authentication, a local interactive terminal opens its login flow
with API provider key environment variables removed, then verifies that ChatGPT
authentication is active. A valid existing ChatGPT login is reused without a
new prompt. Noninteractive environments report the required local setup
command instead of partially configuring the profile.

This subscription check is exclusive to an explicitly selected Codex-native
profile or `--codex-native` operation. Ordinary `codex_sdk` API provider routes do
not fail readiness because of the operator's Codex login method.

`--codex-native` is authoritative after user and task configuration files are
loaded: inherited API provider/agent runtime/model defaults and API-key/custom-endpoint
variables cannot silently switch or misconfigure this run. An explicit CLI
`--model` remains authoritative. Outside this explicit mode, environment
configuration and credentials retain their normal precedence. Praxist records only a
redacted identity; for file-based login it includes a hash of the stable account
identifier, never a token. At runtime Praxist stages `auth.json`, when present,
inside a private disposable OS-temporary Codex home so the app-server can
refresh its own copy without writing the operator's `CODEX_HOME`. Keyring-backed
login retains the operator home as the stable operating-system keyring namespace;
database and log state remain run-local. A private file-backed home is removed
when its app-server closes and is never put in task files, run artifacts, logs,
or replay. The runtime verifies the app-server account is actually `chatgpt`
before starting a peer turn, blanks API-key endpoint overrides, and never falls
back to an API or relay when Codex-native mode was selected.

On resume, `--codex-native` may select saved-login authentication only when the
existing run already has the canonical `codex_sdk` agent runtime and native OpenAI
API provider. Resume never rewrites a historical run's agent runtime or API provider; start a
new run to change either canonical choice.

Other API providers remain supported when the operator explicitly chooses them:

```bash
export OPENROUTER_API_KEY=...
export ORCAROUTER_API_KEY=...
export ANTHROPIC_API_KEY=...
export CLOUDFLARE_API_KEY=...
export CLOUDFLARE_ACCOUNT_ID=...
export GROQ_API_KEY=...
export MISTRAL_API_KEY=...
export XAI_API_KEY=...
```

Cloudflare Workers AI needs both its API token and account ID; alternatively,
set `CLOUDFLARE_BASE_URL` to the complete account-scoped OpenAI-compatible URL.

Praxist reads `${XDG_CONFIG_HOME:-$HOME/.config}/praxist/env` by default. Use
`PRAXIST_CONFIG_FILE=/path/to/env` or command-local `--config-file
/path/to/env` for another configuration. The command-local flag takes precedence;
exported process credentials take precedence over file values.

`praxist configure-llm` manages built-in API provider configurations only. Custom
`model_provider` plugins remain supported through each plugin's documented task
or host environment contract; Praxist does not infer custom key-variable names.

Do not commit keys, paste keys into logs, or write keys into task files.

## Credential Failover Boundary

Single-key quickstart is supported. Built-in environment discovery currently
loads at most one credential per API provider. `CredentialFailoverManager` can
select a fallback when its caller supplies multiple credentials with matching
scope and API provider; an unset `target_ref` acts as a target wildcard. The
caller must also record a supported failure. Automatic runtime
failure-triggered fallback remains disabled, so this is not a user-selectable
runtime mode. Selection and failure state use redacted `CredentialRef` values.

## Tool Credentials

Tool-scoped keys are separate from API provider keys. The bundled
`tool_server:literature_lookup` is no-key-first: it uses public endpoints such
as arXiv, OpenAlex, PubMed metadata, and Crossref-style DOI metadata without
requiring task authors to configure another API provider key.

Future task-local or external plugins may add service-specific credentials for
higher rate limits or licensed sources. Those credentials are optional
enhancers, not generic Praxist requirements. Missing tool credentials must disable
or degrade only the affected lookup path and must not break tasks that do not
explicitly require that source.

## Redaction

Trajectory, logs, docs, generated sites, replay reports, and task templates must
not contain raw secrets. Tests under `tests/hardening` enforce this boundary.
