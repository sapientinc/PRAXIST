# Quickstart

Praxist separates installation and configuration from research takeover. The
local-terminal lane keeps every setup prompt in the terminal. The agent-managed
lane lets Codex or Claude Code perform the same setup decisions in one
conversation. Both lanes write the same Praxist configuration and run the same
readiness checks; neither selects a project or launches research during
installation. They do not share an interaction controller or a second OOBE
state file.

Check [Installation](installation.md) for host requirements and
[Your First Task](first-task.md) for project readiness before takeover.

## Local-Terminal Setup

Run the Codex or Claude Code one-line command from
[Installation](installation.md#install-and-configure). It installs Praxist and
opens the local first-use wizard. To reopen only the wizard later, run
`praxist setup --interactive`.

An interactive terminal opens the first-use wizard automatically. Use Up/Down
and Enter to choose an item. Esc goes back or cancels without inventing a
choice. Setup covers these stages:

| Stage | Local interaction | Result |
|---|---|---|
| Install | Pip installs Praxist and its maintained runtime integrations into the selected Python environment. | `praxist` is available; no task-owned dependency is installed implicitly. |
| Legal terms | Review the Fair Source License, User Agreement, and data notice in a temporary scroll view, then explicitly agree or cancel. | Acceptance records the exact legal bundle version and digest; it does not enable optional data collection. |
| Privacy | When collection is available, separately choose whether to share pseudonymized product-usage status. Nothing is preselected. | Existing consent is preserved; cancellation leaves consent unset. |
| Runtime | Choose one setup profile combining an API provider, agent runtime, concrete model, and authentication mode. | A coherent profile is written to the user configuration. |
| Readiness | Register skills for the selected agent host, materialize writable examples, and run host diagnostics. | Installation finishes without selecting a project or starting a run. |

API keys are entered only in the local terminal. Praxist displays one `*` for
each character, supports paste and backspace, and never places the raw value in
the command line, shell history, agent conversation, or task project. Esc
cancels key entry.

## Agent-Managed Setup

Start Codex or Claude Code:

```bash
codex --yolo
# or
claude --dangerously-skip-permissions
```

Then ask:

```text
Install and configure Praxist. Follow the packaged OOBE runbook and stop after
readiness checks. Do not select a project or start research.
```

The agent reads `docs/agents/oobe-install.md` and uses its structured choices
for legal acceptance, privacy, and runtime decisions. It links to the scrollable
legal documents and must not infer acceptance or accept on the operator's
behalf. An API provider key is never requested in chat. When an API-backed setup
profile is selected, the agent pauses for the operator to enter the key through
Praxist's local masked prompt.

Pip installation is only the package boundary; it does not imply that the
operator accepted legal terms or selected a runtime. The agent must continue
the OOBE runbook rather than report that setup is complete. Its first state
query is:

```bash
praxist setup --agent-managed
```

Its JSON identifies the next required decision. The agent reruns it after each
decision and cannot claim setup decisions are complete until
`setup_decisions_complete` is `true`. Existing credentials, API provider defaults,
and doctor readiness never count as an operator profile selection.

This lane and the local-terminal lane share only configuration, validation, and
recovery contracts. The agent does not emulate terminal keystrokes, and the
local wizard does not reproduce the agent workflow.

## Start Research After Reading the Manual

Before handing over a project, read [Your First Task](first-task.md). It defines
what the project must already provide, what Praxist will add, and which choices
the takeover prompt must communicate. Installation success alone is not launch
authorization.

When the project is ready, start takeover as a separate command:

```bash
# Codex
praxist --takeover --task-path /absolute/path/to/research-project
# Claude Code
praxist --takeover --operator claude --task-path /absolute/path/to/research-project
```

## Choose a Setup Profile

The wizard presents coherent setup profiles that bundle an API provider, agent
runtime, concrete model, and authentication method.

### Codex-Native Mode: No API Key

Choose Codex-native mode to use the existing saved Codex login. This is the
shortest first-use path and does not write an API provider key. Only this explicit
profile verifies ChatGPT subscription authentication. If the SDK-pinned Codex
currently uses an API-key login, setup opens its local interactive login flow
and verifies the ChatGPT login before readiness checks continue. Other
profiles neither require nor inspect this login.

### API-Backed Profiles: Long Runs

For sustained, cost-sensitive research, prefer an
[open-source model API](../guides/open-source-model-apis.md) whose cache reuse,
quality, and throughput have been checked on a representative workload. The
selector includes maintained DeepSeek API, OpenRouter API, OrcaRouter API, and Anthropic API
profiles. Enter the selected provider's key at the local masked prompt when
requested.

Other supported API-backed setup profiles remain available in the same selector. To inspect
the exact current profile contract without changing configuration:

```bash
praxist setup --list-profiles
```

## Reopen a Step

The OOBE does not create a separate completion marker. It derives current state
from the legal-terms acceptance record, existing Praxist configuration,
the profile ID recorded by an explicit setup selection, separate product-usage
consent, diagnostics, and task artifacts. A repeated setup keeps the current
recognized profile selected and never launches takeover. An existing key for
the selected API-backed setup profile is preserved without a second prompt.
Reopen only the step you need:

```bash
praxist setup --interactive
praxist --takeover
praxist --takeover --operator claude
```

Review or verify the License and User Agreement independently with:

```bash
praxist user-agreement review
praxist user-agreement status --json
```

Use an explicit project path when the project is not the current directory:

```bash
praxist --takeover --task-path /absolute/path/to/research-project
```

Explicit `setup --profile` and provider automation flags remain available for
controlled non-interactive provisioning. They do not infer legal acceptance or
an operator profile choice.

## Bundled Complete Examples

Installation creates writable Python/JAX and Rust reference projects under
`~/PraxistExamples` and prints their absolute paths. Praxist never runs against
or writes into the read-only copies in its source or package directory.
Existing working copies are preserved during upgrades.

[Examples And Templates](../guides/examples-and-templates.md) owns the available
projects, copy commands, and guidance for choosing a starting point.

## Check Progress

Ask Codex or Claude Code in natural language:

```text
Report current research progress and list the strongest variant in every
completed generation with its task-defined performance metrics.
```

For direct shell operation:

```bash
praxist status --json
praxist --monitor --latest
```

`Ctrl-C` closes only the foreground monitor. It does not stop the research run.
Ask the agent to stop the current run, or use `praxist stop <run_id>` directly.

## Next

- [Installation](installation.md) defines package and platform boundaries.
- [Your First Task](first-task.md) explains project prerequisites and takeover.
- [Product Usage Controls](../operations/product-usage.md) explains consent commands.
- [`LICENSE.md`](https://github.com/sapientinc/praxist/blob/main/LICENSE.md) is the canonical software license.
- [Praxist User Agreement](../legal/user-agreement.md) defines the service terms accepted with it.
- [Direct CLI Operations](../guides/operators.md) is the shell lifecycle guide.
