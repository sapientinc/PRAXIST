---
name: praxist-control
description: Start, stop, resume, inspect status, open the independent read-only foreground TUI, detect active runs, and safely repair Praxist run lifecycle boundaries from a supported agent interface. Use when the user asks to launch a Praxist task, stop or kill a run, continue/resume an interrupted run, restart from the latest safe generation, query current run progress, open or exit a live monitor, detect or list currently running Praxist tasks in the environment, inspect generation status, view incubator/frontier/leaderboard performance, check hardware load, handle interrupted PI panel or Gems reset boundaries, inspect whether a task directory is runnable, or control Praxist lifecycle commands with `praxist start`, `praxist stop`, `praxist status`, `praxist --monitor`, `praxist resume`, or `praxist resolve`.
---

# Praxist Control

Use this skill to operate an existing Praxist task project. It controls runs;
it does not create task projects, install dependencies, edit Praxist core, or change
provider credentials.

## Empty Request Guard

If the user invokes this skill without naming an operation, exit without taking
an action. Ask the user to specify one of: `start`, `stop`, `resume`, `status`,
`monitor`, or `detect-active-runs`. Do not infer a default lifecycle action from
an empty prompt.

## Command Surface

Verify the live CLI before acting:

```bash
praxist --help
praxist start --help
praxist stop --help
praxist status --help
praxist --monitor --help
praxist resume --help
praxist resolve --help
```

If `praxist` is not on `PATH` but the current directory is a Praxist source
checkout, use `uv run praxist ...`. If neither works, refuse the lifecycle action and
tell the user to install or activate Praxist first.

Prefer JSON for machine decisions:

```bash
praxist status --json
praxist start --task-path /path/to/task --daemonize --json
praxist stop <run_id> --grace 300 --json
praxist resume <run_id-or-run-dir> --daemonize --json
```

Use `--daemonize` for `start` and `resume` from agent or CI shells, or other
sandboxed shells so the run survives the launching process.

## Detect Active Runs Workflow

Use this read-only workflow when the user asks which Praxist tasks are currently
running, whether the environment is idle, whether another server/session has a
live run, or explicitly asks to "detect active run(s)".

1. Run:

   ```bash
   praxist status --json
   ```

   If `praxist` is unavailable but the current directory is a source checkout, use
   `uv run praxist status --json`.
2. Parse every row, not just rows under the current task path.
3. Classify rows:
   - **active**: state is `running`, `run_dir` is present when available, and
     the recorded PID is alive or the row source is a live ps-scan match.
   - **ps-only active**: Praxist-shaped process exists but no registry metadata is
     available.
   - **stale registry**: registry row points to a missing/dead PID, or state is
     `stale`; this is not an active run.
   - **unknown**: row cannot be verified from the available fields.
4. Cross-check active candidates with a lightweight process scan when possible:

   ```bash
   ps -eo pid,ppid,etime,cmd | rg 'praxist\.run run|praxist start|praxist resume'
   ```

   Do not use broad `pkill`, `kill`, or recursive filesystem scans. This
   workflow must not stop, resume, crop, or edit any run.
5. Report all active runs in a table with:
   - `run_id`;
   - PID;
   - elapsed time;
   - task path;
   - run directory;
   - generation, if known;
   - model and provider, if known;
   - state/source;
   - last updated time, if known.
6. Report stale rows separately in a compact list or count. Do not call stale
   rows "running".
7. If no active runs are found, say the environment has no detected live Praxist run
   and mention whether stale registry entries exist.

When multiple active runs are present, do not choose a target for stop/resume
unless the user explicitly asks for a follow-up operation and the target is
unambiguous.

## Status Workflow

Use this read-only workflow when the user asks for current progress, run state,
leaderboard/incubator contents, generation progress, cost-adjacent progress
signals, or hardware load.

Artifact source rule: treat measured result/finding summaries,
`frontier/frontier_manifest.json`, committed `gems/gems_state.json`, and
`gen_N/generation_boundary.json` as canonical current state. Treat
leaderboards, PI evidence packs, PI agendas, prompt layouts, rendered prompts,
diagnostics, and behavior reports as derived views or audit snapshots. Use
derived/audit artifacts to explain what agents saw, but do not let them
override canonical metrics, promotion state, Gems state, or resume boundaries.
Also report compact validation signals when present: task-defined preliminary,
aligned, partial, repair, failed-but-informative, ablation, diagnostic,
lower-stage, or late-after-boundary evidence. Classify them from the task's
declared mode permissions, not from words such as `partial`, `scout`, or
`complete`: an explicitly authorized reduced mode may be mature, while a mode
declared validation-only remains a follow-up lead. When the user has not chosen
otherwise, the recommended expensive-task default treats preliminary checks as
triage and uses a near-complete-coverage aligned mode for early prioritization.
Read the task's `preliminary_stage_labels`, `complete_stage_labels`, ratio
policy, and structured permission metadata instead of guessing maturity from a
stage name. For late-after-boundary signals, say that Praxist retained a
result summary written after its generation boundary and that it should be
reviewed or revalidated before use as a parent.

Search result summaries recursively under `results/**/` using `summary.json`,
`evaluation_summary.json`, `eval_summary.json`, `tiered_eval_summary.json`, or
`custom_*_tiered_eval_summary.json`; accept `result_summary.json` for
compatibility. Read materialized lane, maturity, effort/coverage, protocol, and
diagnostic metadata from structured fields. A lane with
`parent_eligible: false`, including lower-tier and diagnostic lanes, is a
follow-up signal source and not a durable implementation-parent source.
When reporting Gems, show `selection_policy`, `min_mature_eval_units`, and
`evidence_stage_min_units` when present. New tasks use
`mature_evidence_top_k`; stage thresholds are task-owned evaluation-unit
counts. Do not interpret historical compatibility fields as the recommended
task contract.

1. Resolve the target run:
   - Use an explicit `run_id` or run directory when supplied.
   - If the user says "current task", prefer active status rows whose `run_dir`
     is under the task `experiments/` directory.
   - If the user gives no target, select the latest active run for the current
     task. If multiple active runs match, refuse and show candidates.
2. Run:

   ```bash
   praxist status --json
   ```

   Prefer the row's `generation`, `state`, `pid`, `etime`, `run_id`,
   `run_dir`, `model`, `model_provider_ref`, `findings_total`, and
   `updated_at`.
3. Inspect small run artifacts, when present:
   - `task_spec.yaml` for `dig_lite.generation_scope`, independent initial/later
     `quality_diversity` switches, and
     `evaluation.constructive_peer_mix_enabled`. Under the default
     `initial_only` scope, report DIG as expected only for absolute gen0; do
     not treat missing DIG artifacts in later generations or after a Gems
     logical reset as a fault;
   - `orchestrator_status.json` for `current_generation`,
     `generations_completed`, `cohort_size`, `findings_total`,
     `frontier_candidates`, `gems_count`, `variants_total`,
     `variants_above_baseline`, `exit_condition`, `strategy`, and
     `updated_at`;
   - count contiguous `gen_N/generation_boundary.json` markers independently.
     Treat that count as authoritative. If status or summary claims more
     completed generations, report a pending/inconsistent boundary rather than
     accepting the optimistic count;
   - also read `last_stop_audit`, `last_peer_mix`, and
     `mature_quorum_required` when present so status reports can show whether
     the last generation closed on mature evidence and whether constructive
     solution-producing work was under target;
   - `gen_N/STOP_SIGNAL`, `gen_N/generation_results.json`, and
     `gen_N/generation_boundary.json` for the active generation boundary;
   - `frontier/frontier_manifest.json` for incubator/frontier lane contents;
   - `gems/gems_state.json` and Gems archives for current Gems;
   - `metrics_log.jsonl`, `shared_findings/`, or task-local result summaries
     only as needed to compute a concise performance table.
   - `run_summary.json` and canonical generation usage summaries for available
     runtime token counters. Report input, cached input, derived uncached input,
     output, session count, and cache-hit ratio without estimating monetary cost
     for subscription plans. Do not infer missing counters as zero.
   - `<task>/docs/praxist_reports/*.md` for existing human-readable derived reports
     generated by Praxist or diagnostics; list paths but do not let them override
     canonical frontier/Gems/result facts.
4. Report generation progress:
   - current generation number;
   - completed generation count;
   - whether the active generation has a stop signal, generation results, and
     boundary artifact;
   - recent status timestamp and whether artifacts are still updating.
5. Report incubator/leaderboard performance:
   - Prefer task-owned absolute primary metrics and metric names exactly as
     stored in artifacts. Do not rename relative metrics as absolute metrics.
   - For lane frontiers, group by lane, especially incubator-like lanes such as
     `incubator`, `candidate`, `confirmed`, task-specific performance lanes,
     diagnostic/control lanes, and Gems.
   - If no materialized leaderboard exists, compute a best-effort table from
     canonical findings or result summaries and label it as a filesystem or
     artifact-derived estimate.
   - Include task-owned evidence maturity such as stage and completed
     evaluation units and seed counts,
     `effort_ratio`, `coverage_ratio`, promotion eligibility, protocol
     warnings, and suspicious flags when available.
6. Report hardware load and scheduler state:
   - Use bounded platform-appropriate probes for uptime, CPU, memory, storage,
     process state, and the accelerator backend actually used by the task.
     Vendor tools are optional; their absence is not a warning by itself.
   - Do not run broad or expensive filesystem scans. If using `find`, limit
     depth to the task/run roots.
   - Attribute load to the selected run when possible by matching `run_dir`,
     `run_id`, or variant command arguments. Clearly separate unrelated system
     load.
   - When present, summarize `resource_scheduler/status.json`: queued/running/
     completed/failed/rejected counts, current host-wide concurrency, frozen
     generations, assessment state, work-class mix, assigned accelerator
     devices when the selected backend exposes them, idle
     supply waiters, mature/frontier lease priorities, any `release_pending`
     cleanup, `Q/M/D/A_target`, and
     aggregate supply stats. Distinguish capacity saturation from
     capacity that is available but lacks planned experiment supply.
     Include overall and per-priority supply conversion rates and the counts of
     declined, expired, revoked, stale-submission, and genuine reuse outcomes;
     lease counts are not a substitute for mature evidence.
   - Report `running_activity.by_resource_phase` separately from lifecycle
     `running`, plus `peer_capacity_blocked` and per-job blocked reasons. Do not
     equate a live wrapper with active GPU compute, and do not label
     `no_gpu_process_observed` or `unknown` as stalled without corroborating
     process, progress, log, and result evidence.
7. Draw at most two standard score curves at the end of status reports when
   enough per-generation mature evidence exists:
   - best mature primary/composite score by generation;
   - mean mature primary/composite score by generation, or the task's preferred
     absolute-performance metric when the user explicitly asks for it.

   Prefer `$terminal-line-plot` or its `scripts/plot_series.py` helper when
   available. Mark active or incomplete generations as provisional. Do not
   write image files, and do not draw more than two curves in routine status.
8. End with a concise table or bullet summary covering: run id, run dir, state,
   generation progress, active blockers or long-running evaluations,
   incubator/leaderboard highlights, runtime usage/cache summary when available,
   generated report paths, hardware load, and next check time.

If the user explicitly asks this control skill to generate a report, use
`praxist-diagnostic` or the `tool_server:run_report` manual report path.
Do not synthesize a long report inside routine status output.

Status is read-only. Do not stop, resume, crop, rerender, or edit files during a
status request unless the user explicitly asks for that action.

## Foreground Monitor Workflow

Monitoring is an independent foreground action. `praxist start` and `praxist resume` do
not open the monitor, and `praxist stop` does not manage it. After every successful
start or resume action, clearly report the run-specific
`praxist --monitor --run-id <run_id>` command. Do not start the TUI as a lifecycle
side effect.

The default monitor is a fullscreen read-only TUI with the lightweight,
full-color Praxist mark and a 5 FPS display cadence. Peer health rows use the
same animated status light as the header, colored from the canonical peer health
value (green, yellow, red, or gray when unknown). Its renderer reuses immutable
snapshots while the single read-only sampler checks run and hardware state no
more than once per second; do not replace this with five status scans or hardware
probes per second. Use `--plain` only for non-interactive terminals or
append-friendly text output. Monitoring must not edit task artifacts, research
artifacts, frontier/Gems state, prompts, source code, or the Praxist run process.
The long-running monitor uses the peer-owned bounded result summary for
responsiveness and must not recursively reconcile the full result tree in its
sampler. `praxist --monitor --once`, status, and diagnostic operations remain
available for an immediate complete view.

Resolve the target run the same way as status:

- explicit `run_id` or run directory first;
- otherwise latest active run for the current task;
- refuse and show candidates when multiple active runs match.

Useful commands:

```bash
praxist --monitor
praxist --monitor --run-id <run_id>
praxist --monitor --latest
praxist --monitor --plain
```

Operator behavior to explain:

- The default monitor view is a fullscreen TUI with five fixed dashboard panels
  for runs, selected run details, peers, compact recent-log context, and
  hardware/warning status, plus a dedicated lower live-log stream. Log updates
  should trigger full-frame refreshes and remain constrained to the log region
  instead of scrolling the dashboard panels away.
- The branded header and live indicator render at 5 FPS by default. This is a
  presentation rate, not permission to increase status, filesystem, process, or
  hardware sampling above the monitor's bounded read-only cadence.
- Use `praxist --monitor --plain` only when the user needs non-interactive or
  append-friendly output instead of fullscreen updates.
- `Ctrl-C` exits only the monitor interface. The Praxist run remains active.
- To monitor again after exiting, run a new foreground `praxist --monitor` command.

Treat monitor output as an operator view. It must not override canonical
metrics, promotion state, Gems state, or resume boundaries.

## Task Directory Gate

Before `start`, identify the task path:

1. Use the user-provided path if present.
2. Otherwise use a task path that is already explicit in the current Codex
   conversation context, such as a just-created task directory that passed
   validation.
3. If no specific runnable task directory is explicit in the current Codex
   context, do a bounded local candidate scan before starting. Search only the
   current directory, its direct children, and nearby task roots up to depth 3;
   exclude `.git`, `.venv`, `node_modules`, `experiments`, run directories,
   cache directories, raw data directories, and the Praxist source checkout itself.
   A candidate must contain `task.yaml` and then pass the checks below.
4. Ask the user to confirm the selected candidate before launching. If exactly
   one valid candidate is found, still show its absolute path and ask for a
   confirmation signal. If multiple candidates are found, show a compact table
   and ask the user to choose one. If none are found, refuse and say that
   task-initialization should be used first.

Refuse to start if any check fails:

- `task.yaml` is missing.
- `task.yaml` is not parseable YAML.
- `praxist_plugins.workflow.stage` is absent or not
  `workflow_stage:research_loop`.
- task-local roles, audit rules, or evaluations referenced by `task.yaml` are
  obviously missing.
- the selected path is the Praxist source checkout rather than a task project.
- `praxist resolve <task_path>` fails.

For a task just created or repaired in the current agent workflow with
`require_ratio_gate: true`, also require the task-initialization output-contract
preflight to have passed on a file produced by the real evaluator summary
writer: `praxist resolve <task_path> --result-summary <summary_path>`. Route a
failure back to task-harness repair; do not infer readiness from stage labels.
For an older independently created task where no safe bounded evaluator probe is
known, retain the legacy warn-and-continue behavior below instead of launching an
expensive experiment or inventing a new hard requirement.

Warn, but do not refuse solely for this reason, if a legacy real research task
lacks `description_file` but provides enough inline `research_direction` or
equivalent task metadata for Praxist to load it, if a referenced description file is
missing but the loader can fall back to inline task text, or if the task
lacks a lower-admission durable incubator/candidate library lane with
`admit_new_high: true`, or lacks task instructions/evaluator summaries that
emit exact `effort_ratio` and `coverage_ratio` maturity telemetry. When
`require_ratio_gate: false`, only explicitly configured task-owned stage labels
or completion flags may provide fallback maturity; Praxist does not assign global
meaning to common tier names. Recommend task initialization or diagnostic
repair after launch instead of blocking a valid legacy task.

Do not silently start a task whose enabled synthesis trigger has
`mature_quorum_fraction: 0.0` while its maturity policy distinguishes
close-grade results through ratio gates, complete/preliminary labels,
protocol-integrity requirements, or mature parent lanes. Explain that `0.0`
allows raw information-density findings to become the normal close condition.
For an unattended start, stop and route the task through task-initialization
repair. For an explicitly selected existing task, proceed only if the user
confirms in the current conversation that evidence-blind close is intentional
and the task records that rationale; a task with no separate maturity
distinction remains a valid flexible case.

Also warn, without blocking a resolved legacy task, when DIG is enabled without
an explicit generation scope or QD is still nested only under
`dig_lite.cohort_qd`. Newly initialized tasks should use absolute-gen0 DIG plus
the independent `quality_diversity` block; control must not rewrite legacy
configuration during start.

Do not "fix" a bad task project during control. Report the exact missing file,
field, or resolve error and stop. Use the task-initialization skill for task
creation or repair.

## Runtime Gate

Before `start`, verify runtime readiness without printing secrets:

- Reuse runtime facts already present in the conversation when available:
  selected model provider, model, agent runtime, conda/venv/container, task data
  paths, simulator state, and prior successful commands.
- If the conversation or environment selects a non-default Praxist config,
  preserve that exact profile across `doctor`, `resolve`, `start`, and later
  `resume` calls. Prefer one explicit `--config-file` value or
  `PRAXIST_CONFIG_FILE`; never validate with one profile and launch with another.
- Inspect `task.yaml.runtime_environment` for `venv`, `python`, `cwd`,
  `path_prepend`, and non-secret env requirements.
- Check provider key presence by name only. Common keys are
  `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `ORCAROUTER_API_KEY`, `ANTHROPIC_API_KEY`, and
  `OPENAI_API_KEY`.
- If the selected runtime is `agent_runtime:codex_sdk`, verify the Praxist
  environment can import `openai_codex` and `mcp`, and verify distribution
  versions `openai-codex==0.147.0` and `claude-agent-sdk==0.2.136`. For
  DeepSeek/OpenRouter also verify `codex-relay==0.5.5` is installed. For native
  OpenAI, accept either a present `OPENAI_API_KEY` or an exact
  `Logged in using ChatGPT` result from the SDK-bundled Codex binary. Do not
  print login output beyond that status and do not copy authentication files.
  Peer execution still uses SDK-owned local app-server clients and direct MCP
  tools.
- If the selected runtime is `agent_runtime:claude_sdk`, verify
  `claude-agent-sdk==0.2.136`; use `praxist-runtime-install` to repair a
  mismatch before launch instead of upgrading the SDK independently.
- When saved ChatGPT login is the selected authentication path, pass
  `--codex-native` to `doctor`, `resolve`, `start`, and later `resume`. Run
  `praxist doctor --codex-native --task-path <task>` as the readiness gate.
  This mode removes
  inherited provider/runtime/model defaults plus API-key/custom-endpoint values
  again after config loading, so a user or task env file cannot silently switch
  billing paths or reuse a provider-specific model. Preserve an explicit CLI
  model choice.
- Do not use `resume --codex-native` to change a historical run's canonical
  runtime or provider. It is valid only for a run already using `codex_sdk` and
  native OpenAI; otherwise start a new Codex-native run.
- Run the lightest task-owned smoke or help command only when the task documents
  one and it is safe.
- For newly initialized tasks, confirm the default scheduler profile matches the
  public evaluator, directed idle-supply feedback is configured, and any
  multi-device complete evaluator has a recorded natural-unit distribution test.

Refuse to start when required runtime information is missing and the user did
not provide it. Examples: missing venv/container, missing dataset/simulator
path, missing provider authentication, or no credible evaluator runtime.

## Start Workflow

1. Run the task directory gate.
2. Run the runtime gate.
3. Run resolve:

   ```bash
   praxist resolve /path/to/task
   ```

   Add `--runtime`, `--model-provider`, or `--model` when the user explicitly
   selected them. Add the selected `--config-file` when it is not the user
   default.

4. Launch:

   ```bash
   praxist start \
     --task-path /path/to/task \
     --daemonize \
     --json
   ```

   Preserve user-specified `--runtime`, `--model-provider`, `--model`,
   `--cohort`, `--generations`, `--strategy`, and `--config-file`.

5. Parse the JSON response. Report `run_id`, `pid`, `run_dir`, and log path.
   Prefer `extra.monitor_command` when present; for older run records, derive
   the same `praxist --monitor --run-id <run_id>` command from `run_id`.
6. Confirm with `praxist status --json`. If status is stale immediately, inspect the
   launcher log and summarize the failure.
7. Complete the launch handoff by clearly reporting the independent foreground
   TUI command without starting it as a side effect:

   ```bash
   praxist --monitor --run-id <run_id>
   ```

   Also state that `Ctrl-C` exits only the monitor interface and leaves the Praxist
   run active. Mention `praxist --monitor --plain` only when non-interactive or
   append-friendly text output is relevant.

## Stop Workflow

Resolve the target run first:

- If the user gives a `run_id`, use it.
- If the user gives a run directory, match it against `praxist status --json`.
- If the user says "current task", match active rows whose `run_dir` is under
  the task `experiments/` directory.
- If multiple active runs match, refuse and ask the user to choose.
- Do not use `praxist stop --all` unless the user explicitly asks to stop all runs
  or the matching set is proven to contain only the intended target.

Stop through Praxist first. This command manages only the selected run; no monitor
management is required:

```bash
praxist stop <run_id> --grace 300 --json
```

For registry-backed runs, this command closes new run admission before process
discovery and returns only after a bounded stable-empty rescan for late children.
It signals only identity-verified run-owned processes; do not replace this with a
broad path or command-name kill.

Then observe until clean:

1. Inspect `failed_run_ids`, `remaining_pids`, and warnings in the stop result.
   A failed run id means Praxist could not safely complete the stop, such as an
   admission-fence failure or a live legacy process group whose launcher
   identity is no longer verifiable. Poll `praxist status --json` for up to 5
   minutes when either list is non-empty or when the run has not disappeared
   from active status.
2. Treat `registry` and `ps-only` rows with the same target `run_id`, `pid`, or
   `run_dir` as residuals.
3. Only after Praxist stop plus observation fails, manually terminate targeted
   residual PIDs:

   ```bash
   kill -TERM <pid>
   sleep 10
   kill -KILL <pid>
   ```

Never kill unrelated Praxist processes. Never use a broad `pkill` pattern from this
skill.

## Resume Target Selection

If the user describes a run, search only plausible task experiment roots:

- the current task's `experiments/`;
- a user-provided experiments directory;
- run directories listed by `praxist status --json`.

For each candidate `run_*`, inspect small metadata files only:

- `run.json`
- `startup_config.json`
- `run_summary.json`
- `orchestrator_status.json`
- `resume_events.jsonl`
- recent launcher log tail

Choose the best match by task path, run id/name, timestamps, model/provider, and
status summary. If the description is ambiguous, refuse and show the top
candidates. If the user gives no description, choose the latest run directory by
mtime under the task `experiments/` directory.

Before resuming, ensure the selected run is not live:

```bash
praxist status --json
```

If it is verified live, stop it first or refuse; `--force` must never override a
verified live controller. Use `praxist resume --force` only when Praxist reports
that an old registry entry's process ownership is unknown and the operator has
explicitly confirmed that no controller is active.

## Resume Safety Check

Treat `praxist resume` as the primary continuation and recovery launcher for
boundaries that the resume plan recognizes. Before calling it, the agent should
inspect the run directory, back it up when manual cropping might be needed, and
avoid modifying artifacts that Praxist can recover internally.

When newer artifacts contain `artifact_semantics`, use it. Files with
`status: failed`, `partial`, or `superseded` are not clean handoff artifacts.
Derived or audit artifacts are not standalone resume boundary markers and must
not override canonical facts. A committed
`agendas/research_agenda_genN.yaml` may still be a valid next-generation plan
input when the preceding generation boundary is canonical and complete. Old
runs may lack this metadata; for those, keep the legacy file-shape checks and
do not manually rewrite historical snapshots merely to make them look new. The
runtime may atomically backfill a predecessor run's already-committed inferred
boundary prefix during resume so that the first newly written marker cannot
invalidate earlier completed generations.

The `completed_generation` policy is still the target boundary rule: continue
from the last generation whose cohort output and boundary artifacts are
consistent. Use the resume plan as a detector for incomplete PI or Gems
boundaries, then repair or crop those artifacts before launching resume.

When a Praxist source checkout is available, inspect the plan before
launching:

```bash
python - <<'PY' /path/to/run_dir
import json, sys
from pathlib import Path
from praxist.task_spec import load_task_spec
from praxist.plugins.workflow_stages.research_loop.backend.resume_state import inspect_resume_plan

run_dir = Path(sys.argv[1]).expanduser().resolve()
startup = json.loads((run_dir / "startup_config.json").read_text())
recorded_task_path = Path(startup["canonical_args"]["task_path"]).expanduser().resolve()
task_path = recorded_task_path
# If the checkout moved, replace this with the already selected current task
# path. Do not substitute a different task: `praxist resume` verifies the
# persisted task manifest and effective descriptor before it accepts resume.
spec = load_task_spec(str(task_path / "task.yaml"))
plan = inspect_resume_plan(
    run_dir,
    max_generations=spec.generation_policy.max_generations,
    pi_enabled=bool(getattr(spec.multi_pi, "enabled", False)),
)
print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
PY
```

If this snippet cannot run, inspect the same small artifacts manually. Do not
hand an obviously partial PI panel or Gems reset directly to `praxist resume`.

## Resume Breakpoint Classification

Before resuming, classify the interruption boundary from the resume plan and
small run artifacts. Prefer these sources:

- `inspect_resume_plan(...).to_dict()`, especially `start_generation`,
  `completed_generations`, `pending_boundary_generation`, and `warnings`;
- `gen_N/generation_results.json` and `gen_N/generation_boundary.json`;
- `agendas/research_agenda_genN.yaml` for the next-generation PI agenda;
- `gems/gems_state.json` reset events or `pending_reset`;
- `frontier/frontier_manifest.json` Gems metadata;
- `resume_events.jsonl`, `run_summary.json`, `orchestrator_status.json`, and
  recent launcher logs.

Prepare the run with Codex before calling `praxist resume`, but do not second-guess
Praxist recovery that the resume plan already supports. If `inspect_resume_plan`
reports a pending generation boundary, PI boundary, or Gems recovery that Praxist
can complete internally, preserve the active artifacts and hand the run to
`praxist resume`. Prefer documented Praxist repair or rerender commands when the live
CLI exposes them, but do not invent commands. Use backed-up file-level cropping
only when the resume plan cannot handle the boundary and the operator accepts
reverting to a named clean point.

A boundary is clean enough for handoff to `praxist resume` when either the resume
plan marks it as internally recoverable, or the active run path already has
these complete artifacts:

- no incomplete later `gen_N/` directory remains in the active run path;
- completed generations have valid `generation_results.json` and
  `generation_boundary.json`, or a committed Gems reset boundary recognized by
  the resume plan;
- if PI is enabled and the next generation should be planned, the expected
  `agendas/research_agenda_genN.yaml` exists or the boundary marker records a
  deliberate non-strict PI skip/failure;
- `gems/gems_state.json` has no active `pending_reset`;
- the frontier manifest is consistent with any committed Gems reset;
- stale stop or lock markers for the selected run have been removed.

Frontier entries plus `generation_results.json` without a boundary marker are
recoverable inputs, not proof of completion. Preserve them and let the supported
pending-boundary resume path finish the commit.

Handle common irregular breakpoints this way:

| Breakpoint | Evidence | Action |
|---|---|---|
| Unfinished generation after a complete PI agenda | Last `gen_N` lacks valid `generation_results.json`; prior generation boundary is complete; the next agenda exists | Back up the run. Move the unfinished `gen_N/` and matching transient artifacts aside. Preserve the completed PI agenda, remove stale stop/lock markers, then resume from that clean boundary. |
| Completed generation with interrupted PI panel | `generation_results.json` exists but `generation_boundary.json`, frontier ingestion, or next agenda is incomplete; `pending_boundary_generation` is set | Back up the run and preserve the cohort results. If the resume plan marks this as an internally recoverable pending boundary, call `praxist resume`. Otherwise run a documented Praxist PI/boundary rerender or repair command if available; crop only with operator approval. |
| Committed Gems reset with an incomplete next generation | Gems reset event is durable; frontier manifest reflects the reset; next `gen_N` is missing or incomplete | Back up the run. Preserve Gems state and reset archives. Move only the incomplete next generation aside, remove stale stop/lock markers, verify no `pending_reset`, then resume from the Gems-reset boundary. |
| Incomplete Gems reset transaction | `gems_state.json` has `pending_reset` or the resume warning cites a pending Gems reset transaction | Back up the run. If the resume plan marks Gems recovery as internally recoverable, call `praxist resume`. Otherwise run a documented Praxist Gems rerender/recovery command if available; crop only with operator approval. |

If the artifacts are contradictory, refuse destructive pruning and do not launch
resume. Ask the operator to choose between a backed-up manual crop to a named
clean boundary and a dedicated repair command if one exists.

## File-Level Cropping Rules

Do not pre-delete run artifacts by default. First classify the breakpoint and
make a backup. The normal objective is to let Praxist resume/recovery finish any
boundary it explicitly recognizes, and to crop only artifacts Praxist cannot safely
recover.

Physically prune files only for clearly partial final-generation artifacts, or
when no documented PI/Gems repair command exists and the operator accepts
reverting to the previous clean boundary. When pruning:

1. Back up the run directory first, for example:

   ```bash
   cp -a /path/to/run_dir /path/to/run_dir.resume_backup_YYYYMMDD_HHMMSS
   ```

2. Preserve all generations with complete `generation_results.json` plus
   `generation_boundary.json`, or a committed Gems reset boundary recognized by
   the resume plan.
3. Preserve completed PI agendas and committed Gems reset state even when the
   next generation is partial.
4. Move, do not delete, later incomplete `gen_N/` directories and matching
   transient generation artifacts into a run-local backup folder.
5. Remove transient generation stop markers only when they belong to the
   selected run: `STOP_SIGNAL`, `CLOSING_SIGNAL`, and stale `orchestrator.lock`.
   Do not manually remove `ORCHESTRATOR_SHUTDOWN`; `praxist resume` consumes it
   under the run lifecycle lock and restores it when startup fails.
6. Do not hand-edit `gems/gems_state.json` unless the malformed field is the
   exact reason resume cannot be prepared, the original file has been backed up,
   and the operator accepts the change.

## Resume Workflow

1. Select the run.
2. Check it is not live.
3. Inspect the resume plan when possible.
4. Classify the interruption boundary: normal completed-generation resume,
   pending PI boundary, committed Gems reset plus partial next generation, or
   pending Gems reset transaction.
5. Prepare the active run path:
   - for partial next-generation work, back it up and move it aside;
   - for interrupted PI panel or Gems generation that the resume plan marks
     internally recoverable, preserve artifacts and let `praxist resume` recover it;
   - otherwise use a documented repair or rerender command when available;
   - if no repair command exists, crop back to a named previous clean boundary
     only with operator approval, otherwise refuse.
6. Re-check the clean handoff criteria from the classification section.
7. Resume:

   ```bash
   praxist resume /path/to/run_dir --daemonize --json
   ```

   A registry `run_id` may be used instead of a path.

8. Preserve user-specified overrides such as `--task-path`, `--runtime`,
   `--model-provider`, `--model`, `--cohort`, and `--generations`.
9. Confirm with `praxist status --json`, then complete the resume handoff by clearly
   reporting `praxist --monitor --run-id <run_id>`. Do not start the monitor as a
   resume side effect.

## Final Report

For every control action, end with a concise table:

| Item | Value |
|---|---|
| Action | start / stop / resume / status / monitor / detect-active-runs |
| Task path | path or unavailable |
| Run id | id or unavailable |
| Run dir | path or unavailable |
| Command | redacted command |
| Resume boundary | completed / pending PI boundary / committed Gems reset / pending Gems reset / not applicable |
| Repair action | none / Codex crop / Praxist rerender command / refused |
| Status | launched / stopped / resumed / refused |
| Reason | refusal or warning when applicable |
| Next check | `praxist status`, `praxist --monitor --run-id <run_id>`, log path, or run dir |

Never print raw keys, tokens, credentials, or full secret-bearing environment
values.
