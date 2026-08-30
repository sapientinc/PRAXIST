---
name: praxist-onboarding
description: Establish detailed context for Praxist before helping a user install, configure, operate, troubleshoot, or extend the system. Use when the user is new to Praxist, has installed or is about to install the package, asks what Praxist does, asks how Praxist works, asks about `praxist` commands, task projects, runs, peers, generations, frontier/incubator/Gems, PI panels, configuration, API keys, model providers, agent runtimes, plugin architecture, run artifacts, software boundaries, or asks an agent to inspect whether the local environment is ready for Praxist. Do not use for a specific research task's domain science unless Praxist system context is needed first.
---

# Praxist Onboarding

Use this skill to give the current agent enough Praxist system context to be useful on a first interaction. Keep the explanation system-level: describe purpose, operating model, configuration surfaces, software boundaries, and key module locations. Do not paste large code blocks or task-specific benchmark logic into the conversation.

## Onboarding-Only Invocation

This skill is primarily for context establishment. If the user invokes only:

```text
praxist-onboarding
```

or otherwise asks only to understand Praxist, do not start a run, initialize a task, edit files, change configuration, install dependencies, or perform any state-changing environment operation. Load the Praxist context, run the default read-only environment scan below, give a concise readiness note, and wait for the user's next instruction.

Use this shape for a bare invocation:

```text
Praxist onboarding context loaded. I understand the system as a task-agnostic autonomous-research control plane with task projects, peers, generations, frontier/incubator/Gems, PI/Chair planning, plugin boundaries, and packaged docs/templates/skills. I will run a read-only environment scan, report the result, and then wait for your next instruction.
```

The read-only environment scan below is part of the default onboarding action. Only take write or runtime action when the user gives an explicit goal such as installing a skill, changing configuration, creating a task project, running `praxist resolve`, launching a run, inspecting a run deeply, or editing code.

## Default Read-Only Environment Scan

When this skill is invoked, perform this read-only scan by default unless the user explicitly says not to scan. Do not install packages, modify shell config, start runs, write env files, edit task files, or change credentials during this scan.

Keep discovery bounded. If using `ls`, `find`, `fd`, `rg --files`, or similar filesystem discovery commands, limit the scan to explicitly selected roots and no more than two directory levels below each selected root. For default onboarding, the selected roots are the current directory and its immediate parent-level candidates only. Never recurse through `/`, `$HOME`, mount roots, shared-storage roots, full workspace roots, datasets, caches, or `experiments/` trees during this scan.

Check these surfaces:

1. **Praxist command and pip package**
   - `command -v praxist`
   - `praxist --help` when `praxist` exists
   - `python -m pip show praxist` when Python/pip are available
   - `python -c "import praxist; print(praxist.__file__)"` to locate the imported package

2. **Source checkouts near the current directory**
   - Inspect the current directory and one parent level for Praxist source trees.
   - A source tree is likely present when `pyproject.toml` declares `name = "praxist"` and a `praxist/` package directory exists.
   - If a source tree is git-managed, report branch and cleanliness with `git status --short --branch`.

3. **Agent skill installation**
   - Detect expected Praxist skills from the source checkout when present:
     `skills/*/SKILL.md`.
   - Otherwise check the current bundled set:
     `praxist-onboarding`, `praxist-runtime-install`,
     `praxist-task-initialization`, `praxist-scientific-research`,
     `praxist-interactive-task-init`, `praxist-takeover`,
     `praxist-takeover-codex`,
     `praxist-control`, `praxist-diagnostic`, and `terminal-line-plot`.
   - Check the current host's skill root for each expected skill: `${CODEX_SKILLS_DIR:-~/.agents/skills}` for Codex or `${CLAUDE_SKILLS_DIR:-~/.claude/skills}` for Claude Code.
   - If a skill is a symlink, report the symlink target.
   - If a skill is a copied directory, report that updates may require
     reinstalling or copying again.

4. **Environment variables and keys**
   - Report provider keys by presence only: `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `MOONSHOT_API_KEY`, `DASHSCOPE_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `XAI_API_KEY`.
   - Report non-secret Praxist configuration values when relevant: `PRAXIST_AGENT_SYSTEM`, `PRAXIST_LLM_PROVIDER`, `PRAXIST_STATE_DIR`, `PRAXIST_TASK_PROJECT_PATH`, `PRAXIST_MODEL`, `PRAXIST_MODEL_PROVIDER_REF`.
   - Never print raw values for variables containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, or `CREDENTIAL`.
   - Only when Codex-native mode is explicitly selected, report saved-auth
     readiness as `ChatGPT login present` or `not present` using `praxist
     doctor --codex-native`. For every other profile report `not checked` and
     do not probe subscription authentication. Never read auth files or display
     account identifiers/tokens.

Use a concise result table with: detected source checkout(s), pip package status, `praxist` CLI status, skill install status, configured provider keys, selected Praxist agent/provider env, and recommended next action. Prefer DeepSeek direct when available: `DEEPSEEK_API_KEY` + `model_provider:deepseek_alias` + `deepseek-v4-pro`.

## Source Order

Prefer live repository or installed-package facts over memory.

1. If a Praxist source checkout is available, read the smallest relevant subset of:
   - `README.md`
   - `docs/index.md`
   - `docs/getting-started/quickstart.md` for first-use mode selection
   - `docs/concepts/architecture.md`
   - `docs/concepts/runtime-model.md`
   - `docs/concepts/config_discipline.md`
   - `docs/guides/task-projects.md`
   - `docs/guides/operators.md`
   - `docs/guides/research-loop-variant-generation-flow.md`
   - `docs/guides/model-providers.md`, `docs/guides/credentials.md`, `docs/guides/budget-policies.md` when provider, key, or cost behavior matters
   - `docs/reference/cli.md` or `docs/reference/skills.md` only when an exact
     command or Skill activation contract is needed
2. If only the pip package is installed, inspect the installed CLI, package metadata, and packaged resources:
   - `praxist --help`
   - `praxist <subcommand> --help` for the specific command
   - `python -c "import praxist, inspect; print(praxist.__file__)"`
   - package metadata such as `python -m pip show praxist`
   - packaged docs under `praxist/resources/docs/`
   - packaged complete examples under `praxist/resources/examples/`
   - packaged task templates under `praxist/resources/templates/`
   - packaged skills under `praxist/resources/skills/`
3. To locate packaged resources from an installed wheel, run:

   ```bash
   python - <<'PY'
   from importlib.resources import files
   root = files("praxist")
   print(root / "resources" / "docs" / "index.md")
   print(root / "resources" / "examples")
   print(root / "resources" / "templates")
   print(root / "resources" / "skills")
   PY
   ```

4. Treat command suffixes under `praxist *` as evolving. Always verify with `praxist --help` before giving exact syntax unless the user supplied the command contract in the current thread.
5. Never infer task-specific behavior from any individual task project as generic Praxist behavior. Praxist must remain task-agnostic.

## Mental Model

Praxist is a generic autonomous-research control plane. It runs a simulated research lab over an external task project.

- A **task project** defines the actual research problem, baselines, datasets, evaluation protocol, task-local roles, and output policy.
- A **peer** is one autonomous agent session with tools. It reads the task context, writes candidate variants, runs evaluations, and emits structured evidence.
- A **cohort** is a group of peers running in parallel inside one generation.
- A **generation** is one research round. After enough evidence accumulates, the system closes the generation, ingests results, updates memory/frontier state, and plans the next round.
- A task that distinguishes mature/complete evidence from preliminary or
  diagnostic evidence should use a positive mature close quorum. Raw finding
  density may begin assessment, but it should not become normal completion;
  scheduler mature-supply targets prioritize work and do not create this gate.
- A task may enable the **central experiment scheduler**. Peers then submit
  stable semantic experiment IDs plus task-declared resource profiles and work
  classes; Praxist alone performs admission, final process launch, any
  backend-supported device assignment, infrastructure retry, and release. CPU
  pressure changes total experiment concurrency rather than reserving cores per
  job. Generation close freezes new/queued work while active process groups drain.
- **Findings** are structured evidence records. They explain what a result means, not just what files were produced.
- **Result artifacts** are machine-readable outputs under recursive
  `results/**/` paths. Compact summaries use supported names such as
  `summary.json`, `evaluation_summary.json`, `eval_summary.json`,
  `tiered_eval_summary.json`, or `custom_*_tiered_eval_summary.json`; the
  compatibility name `result_summary.json` is also accepted. They carry
  structured lane, maturity, ratio, protocol, and diagnostic metadata for
  materialization.
- **Frontier** contains durable high-value candidates selected for continued attention.
- **Incubator** keeps lower-admission but still durable, task-authorized,
  protocol-passed, non-suspect Pareto/new-high candidates so they are not
  discarded too early. Modes that the task marks non-parentable and
  protocol-suspect signals belong in validation candidates until the declared
  protocol authorizes them.
- **Gems** are compact long-horizon memory anchors, usually refreshed on reset cycles, that preserve especially valuable candidates or mechanisms.
- **PI/Chair panels** synthesize prior evidence into per-peer contracts for later generations.

The task owns maturity meaning, subject to the user's explicit protocol intent.
When a task enables `require_ratio_gate: true`, its evaluator emits finite
`effort_ratio` and `coverage_ratio` values. Staged protocols may additionally
declare task-owned `complete_stage_labels` / `preliminary_stage_labels`; despite
the compatibility field name, `complete_stage_labels` means the labels the task
declares mature and may include an explicitly authorized reduced mode. Labels
are optional vocabulary and never replace ratio evidence when the gate is
enabled. With a required gate, missing ratios remain unknown. Only mature
durable lanes set `parent_eligible: true`; lower-tier and diagnostic lanes set
it false even when they retain useful signals for revalidation.
New Gems configuration uses `selection_policy: mature_evidence_top_k` and a
task-owned `min_mature_eval_units` derived from the protocol authorized for Gems
and durable parent use. Staged tasks use `evidence_stage_min_units` to map their
own labels to cumulative evaluation-unit requirements.

Praxist does not centrally generate research code. Peers generate variant code by reading task instructions, baselines, agendas, and evidence, then editing task/run-local files through their agent runtime.

Optional scientific research support includes no-key
`tool_server:literature_lookup` for literature/database/open-access context and
a disabled-by-default local reviewer stage for artifact/provenance consistency
checks. These are context/audit helpers, not replacement evaluators and not
promotion mechanisms. Literature lookup is current-environment-only: sources can
inform better solutions under the task's existing local data, dependencies,
evaluator, hardware, and runtime, but must not cause agents to download new
datasets, install dependencies, provision simulators, or create new runtime
environments during a run.

Run artifacts are layered by ownership:

- Canonical current state: measured result/finding evidence,
  `frontier/frontier_manifest.json`, committed `gems/gems_state.json`, and
  `gen_N/generation_boundary.json`.
  The contiguous boundary markers, not optimistic status or frontier presence,
  define how many generations are committed. Markerless results/frontier state
  form a recoverable pending boundary. A predecessor run created before the
  marker contract may have a committed frontier-and-agenda prefix; resume may
  infer and atomically backfill that prefix once before applying the marker
  contract to all later generations.
- Validation signals: compact non-frontier leads such as task-defined
  preliminary, aligned, partial, failed-but-informative, lower-stage, or
  late-after-boundary result evidence.
  Late-after-boundary means a result summary was written after its generation
  boundary evidence cutoff; Praxist preserves it in the normal finding store for
  PI and later follow-up but does not treat it as clean promotion truth until
  revalidated. For expensive tasks, preliminary evidence
  is triage only; only a task-declared aligned stage should rank variants before
  complete evaluation, and that stage should keep the same or near-complete
  data/evaluation coverage while reducing training or optimization effort first.
  Praxist reads maturity from task-defined effort/coverage ratios and explicit
  stage-label mappings, not from globally meaningful tier names.
- Derived views: leaderboards and compact status/diagnostic tables generated
  from canonical state.
- Audit snapshots: rendered prompts, prompt layouts, PI evidence packs, PI
  agendas/memos, reviewer reports, logs, and behavior reports that preserve
  what agents saw or what an audit checked.
- Partial outputs: temporary, rejected, candidate, or interrupted phase files
  that resume/control should ignore or crop before continuation.

This distinction is central to current Praxist behavior. Derived/audit artifacts
remain visible for replay and analysis, but runtime readers should not treat
old snapshots as current truth.

Current template-style task descriptors run DIG only for absolute generation 0
by default: `dig_lite.enabled: true` with
`dig_lite.generation_scope: initial_only`. QD is independent:
`quality_diversity.initial_generation_enabled` controls QD over the gen0 DIG
pool, while `quality_diversity.later_generations_enabled` controls QD over the
existing PI synthesis path in later generations. Multi-PI uses the union of PI
memo proposals plus Chair allocation; single-PI performs the same soft guidance
inside its normal synthesis call. Periodic Gems reset stays
off by default through `gems.enabled: false`. Operators may independently
disable either QD phase, disable constructive peer-mix feedback, restore legacy
all-generation DIG explicitly, or enable Gems reset after task evidence
justifies the choice.

## What Praxist Owns

Praxist owns the generic control plane:

- CLI lifecycle and run registry.
- Task-project loading and validation.
- Plugin discovery and resolution.
- Prompt layout rendering and replay manifests.
- Agent runtime normalization.
- Model provider and credential references.
- Budget ledgers and cost policy hooks.
- Cohort and generation orchestration.
- Shared finding ingestion and finding graph maintenance.
- Frontier, incubator, Gems, and research memory maintenance.
- PI/Chair synthesis and next-generation agenda generation.
- Replay, artifact indexing, redaction, and run inspection.

Praxist does not own the user's actual research domain. Real research task logic belongs in an external task project, not in `praxist/core` or bundled generic plugins.

## Software Architecture

Use this architecture split when answering design or debugging questions:

```text
praxist.core
  Stable contracts: run config, protocol dataclasses, storage, replay,
  credentials, budget ledgers, task project resolution, plugin registry,
  prompt layout, runtime/provider interfaces.

praxist.plugins
  Generic replaceable components: agent runtimes, model providers,
  workflow stages, tool servers, graph maintainers, budget policies,
  panel topologies, roles, audit rules, evaluations.

external task project
  The actual research problem: task.yaml, description, roles, evaluations,
  harness, baseline, fixtures, datasets metadata, audit rules, experiments/.
```

Important current module areas:

- CLI entrypoint: `praxist/cli/__init__.py`
- Start/stop/status/init/resolve commands: `praxist/cli/*.py`
- Direct module runner: `praxist/run.py`
- Frozen run configuration: `praxist/core/run_config.py`
- Task project contract: `praxist/core/task_project.py`
- Plugin registry: `praxist/core/registry.py`
- Runtime contracts: `praxist/core/runtimes.py`
- Workflow contracts: `praxist/core/workflow.py`
- Research loop plugin: `praxist/plugins/workflow_stages/research_loop/`
- Model providers: `praxist/plugins/model_providers/`
- Agent runtimes: `praxist/plugins/agent_runtimes/`
- Tool plugins: `praxist/plugins/tools/`

Treat `praxist` as the only implementation package and installed console
script. Do not infer parallel package or CLI identities.

## Installation And CLI Context

Praxist is distributed as the `praxist` Python package and exposes:

```text
praxist = praxist.cli:main
```

For source checkouts, the common developer path is:

```bash
uv sync
uv run praxist --help
```

For installed environments, assume the user expects:

```bash
praxist --help
```

For first use in an operator-selected Python environment, install the complete
runtime extras before entering the existing setup state machine:

```bash
python3 -m pip install --index-url https://pypi.org/simple "praxist[agents,codex]"
praxist setup --agent-managed  # agent-managed lane
```

Installed wheels include the full Praxist documentation, complete examples,
task templates, and skill source tree as package resources:

```text
praxist/resources/docs/
praxist/resources/examples/
praxist/resources/templates/
praxist/resources/skills/
```

Use those packaged resources when no source checkout is available. Do not assume
the user has a repository clone after pip installation. Package resources are
read-only distribution sources: `praxist setup` materializes every bundled
example, while `praxist examples install <name>` materializes one selected
example under `${PRAXIST_EXAMPLES_HOME:-~/PraxistExamples}`. Never launch or
write run output inside a source/package-resource copy. Packaged templates
remain the right place to inspect official task-project shape before
scaffolding a new task.

Current operational CLI verbs include `start`, `stop`, `status`, `resume`, and
`resolve`; setup and support verbs include `doctor`, `docs`, `setup`,
`configure-llm`, `install-skills` / `uninstall-skills`, `examples`, and the first-project
`takeover` handoff. `uninstall` removes only a verified current-user Praxist
installation and refuses to run while local research is active; it never
removes task projects or run directories. For a first-time agent-managed installation, follow
`docs/agents/oobe-install.md` before ordinary onboarding. Verify current help
before relying on a command.

Treat successful pip installation as the package boundary, not OOBE
completion. Continue explicit User-Agreement acceptance, profile selection,
safe skill registration, and readiness. Never accept the
Agreement on the operator's behalf. Never pass `--force-unmanaged` without showing the operator every
same-name conflict and receiving an explicit backup-and-replace choice.
Immediately query `praxist setup --agent-managed`, follow its
`next_required_action`, and rerun it after each decision. Do not infer profile
selection from a saved login, environment key, provider default, or doctor
success. Once `setup_decisions_complete` is true, run readiness checks and stop.
Installation must not select a project or invoke takeover; those require a
separate request after the operator reads `docs/getting-started/first-task.md`.

Operator intent usually maps like this:

- "What is this / how do I start?" -> explain Praxist and run `praxist --help`.
- "Open the documentation" -> run `praxist docs`; on SSH or a headless host,
  present the printed authenticated Pages URL rather than starting a local server.
- "Create a new task" -> invoke `praxist-task-initialization`, use
  `praxist-interactive-task-init` when confirmation is required, or inspect
  task templates.
- "Show a complete example" -> run `praxist examples list`, use the printed
  writable project path, and distinguish it from replaceable task templates.
- "Check task wiring" -> use `praxist resolve`.
- "Launch a run" -> use `praxist start`; `praxist start` / `praxist resume` do
  not open a monitor. Their result and any agent
  handoff must clearly report `praxist --monitor --run-id <run_id>`.
- "Monitor runs" -> use `praxist status`, inspect the run directory, or open the
  independent foreground TUI with `praxist --monitor`, `praxist --monitor --run-id
  <run_id>`, or `praxist --monitor --latest`. Use `praxist --monitor --plain` only for
  non-interactive or append-friendly text output. `Ctrl-C` exits only the monitor
  interface and leaves the Praxist run active.
- "Stop runs" -> prefer `praxist stop <run-id>`; it manages only the run. Use
  `praxist stop --all` only when the user explicitly asks to stop every registered
  Praxist run in the environment; do not use it as the default stop path for a
  single research task.
- "Interactive guidance" -> use the installed Praxist skills for the current agent host when available.

## Task Templates And Complete Examples

The `templates/` tree is the canonical scaffold surface for task authoring. It
is especially important for later skills that convert a user's own research
directory into a formal Praxist task project. The separate `examples/` tree
contains complete task-specific reference projects and must not be treated as a
generic scaffold.

Locate it in one of two places:

```text
source checkout:  templates/
pip wheel:        praxist/resources/templates/
```

Use the task templates this way:

- `templates/tasks/template/` is the primary scaffold for a new task project. Future task-conversion skills should use its structure as the target shape.
- `templates/tasks/toy_math/` is the offline smoke fixture. Use it to understand the minimal no-network task contract and resolve-only behavior.
- `templates/tasks/sam_optimizer/` is a realistic ML-style task reference. Use it to understand a richer task layout, but do not copy SAM-specific research logic into generic Praxist docs, prompts, or plugins.

Use `examples/rocket_booster_recovery/` and
`examples/rocket_booster_recovery_rust/` only to inspect complete Python/JAX
and Rust integrations of project code, frozen assets, evaluator, task harness,
and evidence. Operate an installed writable copy reported by `praxist examples
list`; never infer cross-task defaults from either example's control-domain
metrics or hardware profile.

For task-conversion work, read these files first:

```text
templates/README.md
templates/tasks/README.md
templates/tasks/template/README.md
templates/tasks/template/task.yaml
templates/tasks/template/description.md
templates/tasks/template/prompt_task.jinja2
```

Then shape the user's research directory into the same task-project contract:

- `task.yaml` for identity, metrics, workflow refs, model/runtime defaults, task entrypoints, runtime environment, output roots, task-local refs, and assets.
- `description.md` for the agent-facing research brief, scope constraints, success criteria, and evaluation rules.
- `README.md` for operator-facing setup and run instructions.
- `roles/` for task-specific peer, PI, Chair, reviewer, or specialist contracts.
- `audit_rules/` for declarative task-specific scope, claim, and agenda criteria.
- `evaluations/` for the public evaluator command and metric interpretation.
- `assets/` for harness code, baselines, dataset metadata, literature, fixtures, and optional reference implementations.
- `.gitignore` to exclude `experiments/`, raw datasets, logs, caches, local secrets, and bulky generated artifacts.

Do not teach users to edit `praxist/plugins/**` to create a research task. A user task belongs in the user's task directory; Praxist generic code should remain task-agnostic.

## User-Level Run Flow

Before explaining peers or generation internals, orient new users around the external workflow:

```text
pip install / source checkout
  -> user opens a task project directory
  -> Praxist reads task.yaml and task-local assets
  -> user verifies wiring and credentials
  -> user starts a run
  -> Praxist writes an experiments/run_* directory
  -> user monitors, stops, resumes, or inspects that run
```

A task project is the user's research problem directory. It usually contains `task.yaml`, `description.md`, task-local roles, evaluations, assets, baselines, and an ignored `experiments/` output directory. Praxist is the runner and research control plane; the task directory supplies the problem and evaluation contract.

When helping a first-time user, prefer this order:

1. Ask for or locate the task project directory.
2. Run or suggest `praxist --help` to verify the installed command surface.
3. Inspect `task.yaml` and the task README/description before giving run-specific advice.
4. Use `praxist resolve` when available to check plugin refs, task-local refs, and required assets.
5. Confirm required model-provider credentials by presence only, never by printing values.
6. Start from the task directory or pass `--task-path` explicitly:

   ```bash
   praxist start --task-path /path/to/task-project
   ```

7. Monitor with `praxist status`, inspect the run directory under the task project's
   `experiments/`, or run `praxist --monitor --run-id <run-id>` in a separate
   foreground terminal. `Ctrl-C` exits the TUI without affecting the Praxist run.
8. Stop registered runs with `praxist stop <run-id>`; this manages only the run. Use
   `praxist stop --all` only when the user explicitly asks to stop every registered
   Praxist run in the environment.

## Research Loop Behavior

The default v1 workflow stage is `workflow_stage:research_loop`.

Typical flow:

```text
task project
  -> startup resolves task.yaml, plugins, credentials, runtime env
  -> prompt layout is rendered per peer
  -> cohort peers run through agent runtimes
  -> peers create variants and results under the run directory
  -> peers publish findings and artifacts
  -> generation boundary ingests and materializes evidence
  -> finding graph, frontier, incubator, Gems, and memory update
  -> PI/Chair panel writes next-generation agenda
  -> next generation starts with per-peer contracts
```

Gen 0 usually emphasizes broad exploration from the task prompt and baseline. Later generations receive PI/Chair agenda contracts, frontier/incubator/Gems summaries, negative evidence, and research memory. Explicit strategies such as `explore`, `mixed`, or `exploit` may exist for debugging or operator override, but `auto` is the normal PI-directed path.

The LLM/session boundary is event-driven. A peer session returns, then waits for meaningful events such as new findings, stop signals, workflow completion, or heartbeat expiry. This avoids idle token spend while preserving long-run survivability.

For Codex-native mode and OpenRouter routes, explain that Praxist
automatically coalesces finding-only event bursts before opening another peer
session. Stop, closing, and resource-supply events remain immediate; the full
task prompt, canonical findings, and archived tool outputs remain available.
This is lossless navigation and session-count control, not context compression.
Direct DeepSeek is explicitly excluded and keeps its established event cadence
and prompt behavior. Point users to `docs/guides/cost-optimization.md` for
`PRAXIST_CONTEXT_EFFICIENCY_MODE`, interval tuning, and usage interpretation.

## Configuration Rules

Use this priority model:

```text
CLI args > explicit env vars > override spec > task.yaml defaults
```

Credentials are special: raw secrets should be resolved through credential/provider logic and never copied into `task.yaml`, logs, docs, or prompts.

Key configuration principles:

- `PRAXIST_*` is the canonical Praxist env prefix. Some legacy bare env names may still be accepted as fallback.
- Core and plugin domain code should not read `os.environ` directly.
- Env reads belong at CLI entry boundaries and subprocess env construction boundaries.
- Downstream code should consume frozen `RunConfig` values or domain dataclasses derived from it.
- When exact usage is missing, Praxist records `usage_unknown`; do not silently convert unknown cost to zero.
- Do not print raw API keys. Report whether required credential variables are present, not their values.

Common provider/key context:

- Anthropic native: `ANTHROPIC_API_KEY`
- OpenRouter: `OPENROUTER_API_KEY`
- DeepSeek direct: `DEEPSEEK_API_KEY`
- OpenAI-compatible: `OPENAI_API_KEY`
- Agent system selector: `PRAXIST_AGENT_SYSTEM` or `--agent-system`
- Provider selector: `PRAXIST_LLM_PROVIDER` or `--model-provider`
- Config profile: command-local `--config-file`, then
  `PRAXIST_CONFIG_FILE`, then the user default
- Task path: `--task-path`, then `TASK_PATH`, then invocation directory
- State registry root: `PRAXIST_STATE_DIR`

Keep direct agent skill use distinct from peer execution:

- The Codex or Claude Code CLI in which this skill runs is a human operator
  interface that invokes bundled Praxist skills directly.
- `agent_runtime:codex_sdk` is an optional peer runtime implemented with the
  tested official `openai-codex==0.147.0` Python SDK and long-lived local
  app-server clients. It connects selected MCP servers directly and consumes
  typed notifications; it does not replay a human Codex CLI session. For native
  OpenAI only, it may use the saved ChatGPT authentication owned by the
  SDK-bundled Codex binary.

The default peer runtime remains `agent_runtime:claude_sdk`. If the user
explicitly selects `agent_runtime:codex_sdk`, verify the Praxist environment has
`openai-codex==0.147.0`, `claude-agent-sdk==0.2.136`,
`codex-relay==0.5.5`, and MCP support. For the default Claude runtime, verify
`claude-agent-sdk==0.2.136`. OpenAI connects directly. DeepSeek and OpenRouter
use a private run-scoped relay because those providers expose Chat Completions
while Codex expects Responses. Praxist owns that relay; task projects and users
should not launch one per peer.

When reasoning behavior matters, inspect `task.yaml:agent.reasoning_effort`.
`auto` preserves the selected provider/runtime behavior; explicit
`off`/`low`/`high`/`max` applies run-wide to peers, DIG, PIs, and Chair. Explain
that Praxist performs the runtime-specific mapping and that changing effort can
change latency, output usage, and cost. Do not copy provider wire fields into a
task project.

For Codex-native mode with native OpenAI, run `praxist setup --profile
codex-native --install-skills codex` in a local terminal, then select
`--codex-native` for lifecycle commands. Setup verifies or repairs only the
SDK-pinned Codex login. Do not require this subscription check for another
profile. The explicit mode fixes `agent_runtime:codex_sdk` plus
`model_provider:openai_compatible` and removes API-key/custom-endpoint and
inherited provider/model defaults after configuration loading. Outside that
mode, configured credentials retain normal precedence. Never claim
Codex-native authentication works with a relay provider.

Current preferred low-cost research configuration is DeepSeek direct API:

```bash
export DEEPSEEK_API_KEY="..."
praxist start \
  --task-path /path/to/task-project \
  --model-provider model_provider:deepseek_alias \
  --runtime agent_runtime:claude_sdk \
  --model deepseek-v4-pro
```

For agent-system short-name paths, the same provider can be selected explicitly:

```bash
export DEEPSEEK_API_KEY="..."
export PRAXIST_AGENT_SYSTEM=claude_sdk
export PRAXIST_LLM_PROVIDER=deepseek
praxist start --task-path /path/to/task-project --model deepseek-v4-pro
```

`model_provider:deepseek_alias` is an OpenAI-compatible DeepSeek provider. Its current default model is `deepseek-v4-pro`; `cheap_peer` maps to `deepseek-v4-flash`, and `strong_reasoner` maps to `deepseek-v4-pro`.

Emphasize the cost model: Praxist prompt layout keeps large frozen and semi-static prompt blocks stable across peers and generations, and the DeepSeek long-context route can achieve cache hit rates above 95% in well-structured long runs. Because cached tokens are much cheaper than uncached tokens, this is currently the most cost-effective configuration for large Praxist research runs when the task does not require a different provider.

Do not promise a 95%+ cache hit rate for tiny smoke tests, heavily changing prompts, or misconfigured runs. Present it as the expected advantage of the recommended DeepSeek setup when prompt layout stability is preserved.

For Codex-native mode/OpenRouter diagnostics, do not treat cache hit rate as
the only cost measure. Also inspect fresh session count, sessions per
peer-generation, total input, cached input, and uncached input. A high cache hit
rate can still accompany excessive logical input when agents repeatedly rebuild
the same context through shell reads.

Task projects can declare a `runtime_environment` block in `task.yaml`. Praxist validates and injects task-local runtime values such as `PRAXIST_TASK_PYTHON`, `PRAXIST_TASK_VENV`, `VIRTUAL_ENV`, `PRAXIST_TASK_SHELL_PREFIX`, and PATH additions so peer commands run in the task's environment.

Task projects should also declare generation-scoped DIG, independent QD,
constructive peer-mix, and Gems policy in `task.yaml` when they are intended for
real research. Use `templates/tasks/template/task.yaml` for the generic shape
and richer reference templates such as `templates/tasks/sam_optimizer` for a task-local
staged protocol. Keep labels and thresholds task-owned; one task's stage names
or evaluation-unit counts are not generic Praxist defaults.

## Task Project Boundary

A real task project normally contains:

- `task.yaml`
- `description.md`
- task-local `roles/`
- task-local `audit_rules/`
- task-local `evaluations/`
- `assets/` for harness code, baselines, fixtures, dataset metadata, and references
- ignored `experiments/` for run outputs

Task-local refs such as `task_role:*`, `task_audit:*`, and `task_evaluation:*` resolve inside the selected task project. Bundled generic refs such as `workflow_stage:*`, `model_provider:*`, and `agent_runtime:*` resolve through Praxist plugins.

Never add a private task's role prompts, benchmark logic, datasets, or domain-specific guardrails into Praxist core or bundled generic plugins. If two unrelated task projects cannot reuse a component, it probably belongs in the task project.

## Run Artifacts

Long-running run directories are a product surface. Important artifacts often include:

```text
run.json
startup_config.json
task_project_manifest.json
plugin_resolution.json
trajectory.jsonl
budget_ledger.jsonl
artifact_index.jsonl
credentials_redacted.json
run_summary.json
prompt_layouts/
findings/
frontier/
gems/
research_memory/
replay/
logs/
```

Compatibility or task-specific runs may also materialize `shared_store.db`, `shared_findings/`, `STOP_SIGNAL`, graph artifacts, `variants/`, `results/`, and `agendas/`.

When debugging a candidate that "did not affect the next generation", check:

1. Variant code exists under `variants/<variant_name>/`.
2. Evaluation artifacts exist under `results/<variant_name>/`.
3. A canonical finding exists or the result artifact was materialized.
4. Frontier/incubator/Gems state references it when appropriate.
5. The next agenda references it if PI/Chair decided to use it.

## Safe Operating Rules

- Prefer `praxist stop` over process-list killing when a run is registered.
- Do not write production run artifacts into the Praxist source checkout.
- Do not mutate a task project while answering conceptual questions unless the user requested edits.
- Do not push to remote repositories unless explicitly requested.
- Do not expose secrets. Redact keys and credential-bearing env.
- Do not over-tighten guards or filters unless the user asks for enforcement. Praxist favors "capture first, label uncertainty, continue when safe."
- Preserve weak but useful evidence with clear provenance labels instead of deleting it.
- Use current CLI help and docs for exact command names because the `praxist *` surface is evolving.

## Response Pattern For New Users

When a user opens Codex or Claude Code after installing Praxist, respond in this order:

1. If the user only invoked the skill or only asked to understand Praxist, establish context, run the default read-only environment scan, report the concise result table, and wait.
2. Identify whether they are in a source checkout, an installed pip environment, or a task project from the scan result.
3. Verify the CLI surface with `praxist --help` when commands matter.
4. Explain Praxist as a generic autonomous-research control plane, not the research task itself.
5. Ask for or locate the task project path before giving run-specific commands.
6. For setup, check provider credentials by presence only.
7. For task creation, invoke the task-initialization skills directly. For run
   operations, prefer `praxist resolve`, `praxist start`, `praxist status`,
   `praxist stop`, and `praxist resume` when available.
8. For architecture or development work, read the architecture docs and keep generic Praxist logic separate from task-specific logic.

Keep answers practical. Give the smallest next command or file path that moves the user's setup forward, and only expand into architecture details when they are needed for the user's decision.
