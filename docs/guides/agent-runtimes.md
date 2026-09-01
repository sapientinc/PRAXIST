# Agent Runtimes

Agent runtimes execute `AgentRunRequest`, emit normalized `AgentEvent` records,
and return `AgentRunResult`. The runtime controls the agent session; the API
provider (`model_provider:*`) separately describes API shape, endpoint,
model defaults, and credentials.

## Runtime Contract

Every production runtime adapter is responsible for:

- translating Praxist prompt, model, tool, cache, sandbox, and timeout intent;
- exposing selected MCP tool servers without leaking API provider response objects;
- normalizing assistant text, tool calls/results, errors, usage, and terminal
  state;
- preserving cancellation and timeout status;
- redacting credentials before events enter trajectory or logs;
- supporting concurrent long-running peer sessions within its declared
  capacity.

Exact capabilities vary by SDK. A runtime must report an unsupported contract
or unknown usage explicitly rather than pretending the capability exists.

## Bundled Runtimes

- `agent_runtime:claude_sdk` is the default and recommended production runtime
  for new task projects, tested with `claude-agent-sdk==0.2.136`.
- `agent_runtime:codex_sdk` is an explicitly selected production runtime built
  on the official `openai-codex==0.147.0` Python SDK.
- `agent_runtime:fake_runtime` is the deterministic offline runtime used by
  conformance tests.

Selecting `codex_sdk` does not change the default runtime for existing tasks.

## Claude SDK Liveness

Each `agent_runtime:claude_sdk` session consumes its SDK stream on an isolated
worker loop while the research loop retains timeout and cancellation authority.
The adapter tracks complete SDK messages, partial model-stream events,
foreground tool activity, and protected background work as distinct progress
signals. Partial events are observability-only and are not copied into the
canonical agent transcript.

A liveness warning is emitted only when the observed state is `model_waiting`
and every progress source has remained silent past the warning interval. A
long-running foreground tool or active background task is reported as a
low-frequency healthy-work state instead of an SDK stream stall. Tool names may
appear in that health record, but tool inputs, commands, task IDs, and other
sensitive payloads do not. These observations do not extend or replace the
configured runtime timeout, generation deadline, stop request, or cancellation
path.

## Codex SDK Architecture

`agent_runtime:codex_sdk` uses long-lived local Codex app-server clients rather
than launching a fresh CLI command for every peer turn. An agent runtime/API
provider/credential scope may share one client while each request receives an
independent ephemeral Codex thread. The adapter consumes typed app-server
notifications and maps them to Praxist events.

Selected Praxist tool servers are attached directly as stdio MCP servers. Tool
allow/deny metadata is translated into the app-server configuration; no shell
bridge is part of the runtime contract.

The runtime also provides:

- streaming assistant, tool, reasoning, plan, file-change, usage, and terminal
  notifications when the SDK emits them;
- turn interruption for Praxist stop requests and timeouts, followed by a
  bounded drain;
- replacement of an unhealthy app-server client without invalidating healthy
  concurrent turns prematurely;
- a private worker pool and bounded stream concurrency so large peer cohorts do
  not starve unrelated Praxist async work;
- runtime-scoped state under `<run_dir>/runtime_state/codex_sdk/`;
- native OpenAI authentication through either `OPENAI_API_KEY` or a saved
  ChatGPT login owned by the SDK-bundled Codex binary;
- a read-only account model-catalog probe so Codex-native mode launchers can
  reject unsupported explicit models before starting a peer cohort;
- Praxist sandbox-intent translation for read-only, workspace-write, and full
  access modes.

Codex turns keep strict caller `output_schema` values. When a known non-strict
object schema omits `additionalProperties: false`, Praxist leaves it out of the
first endpoint request instead of making a predictably rejected call. Existing
prompt parsing and task validation remain in force, and one runtime warning
records the compatibility fallback. Unrelated API provider errors and strict-schema
errors are not retried or hidden. Tasks should not shadow the bundled runtime
with model-specific schema adapters.

Usage is exact only when the app-server publishes token-usage notifications.
Otherwise the normal Praxist `usage_unknown` behavior applies. Prompt-cache
behavior remains agent runtime/API provider managed. Codex has a built-in shell surface,
so requests that require a provably shell-free runtime are rejected instead of
being represented as equivalent to Claude SDK behavior.

For Codex-native mode runs, Praxist automatically uses lossless
finding-event batching between independent threads. It keeps the complete task
contract and canonical artifacts, supplies a larger bounded set of unseen
finding references, and asks continuation sessions to reopen exact originals
only when needed. This reduces repeated thread bootstrap work without retaining
an indefinitely growing Codex conversation or relying on lossy compaction.
Stop, closing, resource-supply, timeout, and recovery semantics are unchanged.

## Capability Alignment And Differences

Both production adapters target the same Praxist request/result contract, but
they are not interchangeable implementations:

| Capability | `claude_sdk` | `codex_sdk` |
| --- | --- | --- |
| Selected Praxist MCP tools | Direct SDK MCP integration | Direct app-server stdio MCP integration |
| Streaming | Normalized from Claude SDK messages | Normalized from typed app-server notifications |
| Timeout and stop | Runtime-specific cancellation path | Turn interrupt plus bounded notification drain |
| Usage | Recorded when the SDK/API provider exposes it; otherwise unknown | Token-usage notifications when present; otherwise unknown |
| Sandbox intent | Claude SDK permission/sandbox integration | Codex read-only/workspace/full mapping; built-in shell remains part of the runtime |
| Long-run concurrency | Independent concurrent peer sessions | Independent threads over shared long-lived clients with bounded stream concurrency |
| Non-native API providers | Claude SDK/API provider compatibility path | Responses-to-Chat relay for supported Chat Completions API providers |

Do not claim complete behavioral equivalence. Tool naming, event granularity,
sandbox capabilities, cache behavior, API provider errors, and usage availability
remain SDK-specific even though Praxist normalizes their durable result shape.

## API Provider Routing

The Codex app-server consumes the Responses protocol. API provider routing is:

| API provider shape | Codex SDK path |
| --- | --- |
| OpenAI / `model_provider:openai_compatible` | Direct SDK/app-server connection |
| `model_provider:deepseek_alias` | Private run-scoped `codex-relay` to DeepSeek Chat Completions |
| `model_provider:openrouter` | Private run-scoped `codex-relay` to OpenRouter Chat Completions |
| `model_provider:orcarouter` | Private run-scoped `codex-relay` to OrcaRouter Chat Completions |
| `model_provider:cloudflare` | Private run-scoped `codex-relay` to Cloudflare Workers AI Chat Completions |

Praxist starts and stops the relay; operators must not launch a relay per peer.
The relay listens only on an ephemeral local port and receives only the selected
API provider credential. An API provider not declared compatible by the runtime plugin
must fail during resolution or startup rather than being silently rerouted.
For OpenRouter only, the relay adds a hashed run-scoped `session_id` for sticky
routing and cache locality. It does not enable response caching. DeepSeek relay
reasoning overrides are added only when the task selects an explicit policy;
`auto` preserves the existing route behavior.

Lossless context-efficiency controls are documented in
[Cost Optimization](cost-optimization.md). The automatic policy applies to
Codex-native mode and OpenRouter/OrcaRouter routes; direct DeepSeek is explicitly
excluded.

## Reasoning Policy

Task projects may set one reasoning policy across API providers for every peer,
Principal Investigator (PI), Chair, and Deep Innovation Gate (DIG) planner
call:

```yaml
agent:
  reasoning_effort: max  # auto | off | low | high | max
```

`max` is the default for new and existing task projects that omit the field.
It requests the strongest reasoning level supported by the selected route.
`auto` is an explicit opt-in to the API provider/agent runtime's native default. `off`
explicitly disables model reasoning when the route supports that control;
`low` and `high` request the corresponding effort. The legacy
`premium_mode: true` setting remains supported as `max` when
`reasoning_effort` is `auto`; an explicit non-`auto` policy takes precedence.

For `claude_sdk` with DeepSeek, Praxist maps this policy to DeepSeek's Anthropic
compatibility fields: `thinking.type` is `enabled` or `disabled`, and enabled
requests carry the selected effort. Other Claude-compatible API providers retain
their native adaptive-thinking mapping. For `codex_sdk` with DeepSeek, the
private run-scoped relay injects the same policy into Chat Completions requests
and retains the API provider's reasoning state across tool-call subrequests.
Praxist does not summarize or reconstruct that state. Native Codex models,
including Codex-native `gpt-5.6-luna`, receive the closest supported SDK effort
(`max` maps to `xhigh`). OpenRouter and OrcaRouter relay routes use their
unified `reasoning.effort` object.
Models whose API provider contract requires reasoning may reject `off`; Praxist
surfaces that API provider error and leaves the user's policy unchanged.

Reasoning controls change model behavior and may change latency, output-token
use, and cost. They do not change task evidence, promotion, timeout, or
generation-close contracts.

## Install And Select

Install the Codex runtime extra in the Praxist environment:

```bash
python -m pip install 'praxist[codex]'
```

The extra pins `openai-codex==0.147.0`, `claude-agent-sdk==0.2.136`, and
`codex-relay==0.5.5`, and includes the MCP dependencies used by bundled Praxist
tools. These versions are the tested runtime compatibility baseline; upgrade
them only together with Praxist runtime validation. The extra does not install a
separate task environment.

Select a runtime through a setup profile, task configuration, or explicit start
override. [Credentials](credentials.md) owns API-key and saved-login setup,
authentication precedence, private Codex homes, and verification. The
[Quickstart](../getting-started/quickstart.md) owns first-use profile selection;
the generated [CLI Reference](../reference/cli.md) owns exact command options.

## Direct Agent Skill Use

The human-facing Codex or Claude Code CLI can invoke the bundled Praxist skills
directly. This operator host is independent of the peer runtime. A saved Codex
ChatGPT login may authenticate the official Codex SDK runtime when native
OpenAI is explicitly selected, even when Claude Code hosts the operator
workflow. Peer execution still happens through Praxist-owned runtime clients,
not by attaching to the interactive operator session.

## Adapter Checklist

When adding or changing a runtime adapter:

1. accept the runtime-neutral request/context contract;
2. translate model, prompt, MCP, sandbox, cache, timeout, and tool options;
3. emit normalized typed events and a normalized terminal result;
4. record usage when available and unknown usage otherwise;
5. redact secrets and API provider response objects;
6. preserve cancellation, timeout, and concurrent-turn isolation;
7. add offline conformance plus focused API provider/MCP integration coverage;
8. document capability differences instead of claiming cross-SDK equivalence.
