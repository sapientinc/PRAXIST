# Scoreless Research

Use scoreless research when a task should collect evidence and deliver a result
without optimizing a measured score. It uses the mandatory `research_loop`
stage, existing peer execution, generation boundaries, budgets, and replay.
Delivery completion does not establish the correctness of a research conclusion.

## Select the mode

Add this configuration to an [external task project](task-projects.md):

```yaml
research_loop:
  mode: scoreless

generation_policy:
  max_generations: null
  cohort_size: 4
  per_generation_hours: null
```

This example has no generation-count, generation-duration, or total wall-clock
ceiling. Cohort size selects how many peers work together, not how much research
may be completed. Use the review completion decision below to finish research,
or stop through the operator CLI. Explicit finite limits remain available for
tasks that want them. Start and resume through the supported CLI:

```bash
praxist start --task-path /path/to/task-project --daemonize --json
praxist resume <run-id>
```

Source-checkout operators use `uv run praxist ...`. No alternate research
launcher is needed.

In scoreless mode, omission or `null` leaves these limits unset:

| Configuration section | Optional limits |
| --- | --- |
| `generation_policy` | `max_generations`, `per_generation_hours` |
| `synthesis_trigger` | `max_interval_minutes` |
| `pi_agent` | `max_runtime_minutes` |
| `multi_pi` | `pi_max_runtime_minutes`, `chair_max_runtime_minutes`, `round2_max_runtime_minutes` |

Normal finding-based generation boundaries and planning still operate. Explicit
finite limits remain enforceable, and metric-mode defaults are unchanged.
Scoreless `synthesis_trigger.min_interval_minutes` defaults to zero, so it does
not impose a minimum wait before an otherwise eligible boundary.

When either scoreless generation count or per-generation duration is uncapped,
stage requests and grants use `null` for `tokens`, `wall_clock_seconds`, and
`gpu_hours`. Estimates carry `usage_estimate_status: unknown`; measured usage
remains finite, and missing measurements are recorded as `usage_unknown`, not
zero. These grants do not override an explicitly configured lifecycle deadline.
Bounded metric-mode planning is unchanged.

The default budget policy accepts nullable grants only with trusted controller
startup authorization; request metadata cannot authorize them. Custom
`BudgetPolicy` implementations used for uncapped runs must support
`decide(request, allow_uncapped=True)` and uncapped grants, or startup fails.
The bundled `default_basic` policy supports this contract.

Omit scored baselines, primary/auxiliary/anchor metrics, frontier lanes, maturity
policies, and Gems. Scoreless configuration rejects scored selection and a
positive mature-evidence quorum rather than inventing a metric or ignoring the
conflict. An evaluator command is not required. Numeric measurements and
calculations may still be evidence; they do not become a ranking objective.
Omitting `research_loop.mode` preserves the existing metric mode.

Scoreless cohorts skip the Deep Innovation Gate (DIG) and cohort quality-diversity
allocation, even when those features are enabled elsewhere in task configuration.
They launch research peers directly without evaluator-driven candidate selection.

All finding types are retained. Each committed generation writes complete
evidence to `gen_N/scoreless_evidence.json` with `evidence_status: not_scored`.
Later peer and planning prompts receive bounded summaries with references to
those full manifests. Scoreless retention does not promote a best-scoring
candidate, and reports identify evaluation as not configured.

## Prepare a runtime layout

Tasks that need a prepared output layout or ownership boundaries can declare a
trusted synchronous startup hook:

```yaml
runtime_environment:
  cwd: run_dir
  prepare_entrypoint: coordinator.py:prepare_runtime
  prepare_config: {}
```

The task function has signature
`prepare_runtime(*, task_path: Path, run_dir: Path, resume: bool, config: dict) -> None`.
It runs after fresh-run checks, task loading, and private resume selection,
before cwd validation and run-store creation. Configuration is passed as a
detached copy. The entrypoint must be a Python file inside the task root;
asynchronous functions and non-`None` results are rejected. Returned futures are
cancelled when possible, but the hook must not start background preparation.
A preparation failure stops startup.

Make this hook idempotent for resume. It may establish runtime directories and
permissions, but must not launch agents or an alternate research process.
Dependency installation remains a separate setup step. Preparation occurs before
a new research deadline is established and does not reset an existing deadline.

## Optional task lifecycle

A task can prepare inputs, review committed generations, and deliver a final
artifact through one asynchronous callback in its own source tree:

```yaml
research_loop:
  mode: scoreless
  lifecycle:
    entrypoint: coordinator.py:handle_lifecycle
    after_generation: true
    config:
      output_directory: deliveries
```

The entrypoint must identify an existing Python file inside the explicit task
root and an asynchronous function. Without a total wall-clock limit, omitted
`initial_seconds` and `finalization_seconds` mean no phase deadline. Explicit
`null` has the same meaning. `after_generation` defaults to false.
`config` is a task-owned JSON-compatible mapping captured at startup. Keep
credentials out of task configuration.

Praxist calls `async def handle_lifecycle(context)` in this order:

| `context.phase` | When it runs | Input findings |
| --- | --- | --- |
| `initial` | Before peer cohorts | Empty |
| `review` | After a committed generation boundary and its planning step, when enabled | Committed findings through that generation |
| `finalize` | After the last research boundary, before runtime teardown | All committed generation findings |

The context exposes:

| Attribute | Meaning |
| --- | --- |
| `task_path`, `run_dir` | Explicit task and run roots as `Path` objects |
| `phase` | `initial`, `review`, or `finalize` |
| `generation_id` | Nonnegative generation number for reviews; otherwise `None` |
| `deadline_at` | Original run deadline in Unix epoch seconds, or `None` when uncapped |
| `phase_deadline_at` | Deadline for this phase in Unix epoch seconds, or `None` when uncapped |
| `config` | A detached copy of lifecycle configuration |
| `findings` | A tuple of frozen finding dictionaries |

Call agents through the controller-provided method, rather than invoking an SDK
or subprocess research launcher from the task callback:

```python
agent_result = await context.run_agent(
    prompt,
    role="review",
    allowed_tools=None,
    timeout_seconds=None,
)
```

`role` accepts `research`, `pi`, `review`, or `final` and defaults to `research`.
`allowed_tools=None` inherits the controller's available tools; an explicit
list can narrow them, subject to task policy and runtime support. The optional
timeout is capped by the remaining phase time when a phase deadline exists.
With no requested timeout or phase deadline, the call runs without a lifecycle
timeout; cancellation still propagates. The return value is the normalized
`AgentRunResult`; the task must inspect it and validate its own output.

During `finalize`, every agent request has network access disabled and receives
no MCP servers, regardless of its requested role. Final delivery must use frozen
local evidence. Requests with the `review` role also use a read-only filesystem
policy; review output should be returned to the callback for recording.
These restrictions apply to dispatched agents; the trusted
Python callback remains responsible for respecting the same evidence boundary.

The callback returns a JSON-compatible mapping:

```json
{
  "status": "completed",
  "artifacts": ["deliveries/review-gen-0.json"],
  "summary": {"generation_id": 0, "research_complete": false}
}
```

Use `incomplete` when the task's delivery checks do not pass. Artifacts must be
existing regular files under the run root, named by relative paths without
traversal or symlinks. Praxist records their content hashes. Keep committed
artifacts immutable: write a separate path for every generation review instead
of overwriting one shared `review.json`. Publish each task artifact atomically.

A completed generation review can return `summary.research_complete: true` to
stop launching research cohorts and proceed to finalization. The value must be
a boolean: false or omission continues research. Only a committed review makes
this decision; an incomplete review or an initial/final summary does not.
The decision survives resume in the review checkpoint. The task owns the
substantive completion criteria; artifact validation alone is not a reason to
declare research complete.

Callback code is trusted controller code, executed with Praxist's Python
interpreter, not automatically with the task's declared virtual environment.
Keep its dependencies available there. It must cooperate with asynchronous
cancellation; a blocking Python function cannot be forcibly interrupted by an
async timeout. Partial artifacts are preserved on failure. An incomplete
initial or final delivery makes the run incomplete. An incomplete generation
review preserves prior deliveries and permits subsequent research, subject to
any explicitly configured limits.

## Deadlines and resume

The original start is persisted for scoreless runs, including runs without
callbacks. Uncapped runs persist `null` for their deadline, not an artificial
far-future timestamp or infinity. Initial, review, and finalization callbacks
remain available without a time limit, and resume does not introduce one.

Optional finite limits retain their original deadlines across resume. Set
`run_lifecycle.max_wall_clock_hours` for a total cap. Under that cap, omitted
initial/final phase allocations default to 1800 seconds each; their sum must
leave time for research. An explicit `null` phase allocation adds no phase cap
or finalization reserve, while the total cap still applies. Explicit finite
phase allocations must be positive and can also be used without a total cap.
Research ends before a configured total deadline to reserve any configured
finalization allocation. The initial deadline uses the original start;
finalization uses its first attempt's start. Neither phase can extend a total
deadline, and generation reviews share the research deadline.

An active task deadline requires the runtime's asynchronous `execute` contract.
Praxist rejects synchronous dispatch before it starts because cancelling a
Python worker-thread await cannot stop that thread's tools or file writes.
Async runtimes must cooperate with cancellation and drain their in-flight
sessions. Separately detached background jobs remain the responsibility of the
run's outer process supervisor. Synchronous fixture runtimes remain supported
when no task deadline is active.

Each first phase attempt freezes its inputs and input digest. Retries use those
same inputs and deadline. Completed phases are not executed again when their
artifact hashes still match; changed or missing artifacts are not silently
accepted. Callback exceptions and timeouts produce incomplete state.
Cancellation is recorded where possible and propagated.

Reviews have separate checkpoints, `review_gen_N`. Resume reconciles missing or
interrupted reviews for completed generations before launching a successor;
this also covers a crash between the generation commit and review creation.
Once finalization inputs are frozen, resume continues finalization and does not
restart research or unfinished generation reviews.

Final output materialization also verifies committed evidence and delivery
hashes. An integrity failure preserves the available artifacts but records a
failed run and returns a nonzero process exit code.

## Task-wide execution policy

`execution_policy` restricts dispatched agent requests and can select models by
execution role. This example uses placeholder model names; replace them with
models supported by the selected provider and runtime:

```yaml
execution_policy:
  model_by_role:
    research: {model: your-supported-model, reasoning_effort: high}
    pi: {model: your-supported-model, reasoning_effort: high}
    review: {model: your-supported-model, reasoning_effort: high}
    final: {model: your-supported-model, reasoning_effort: high}
  allowed_tools: [Read, Glob, Grep, Bash, Write, Edit]
  tool_execution_timeout_seconds: null
  sandbox_intent:
    filesystem: workspace_write
    network: "off"
    approval: auto
    readable_roots: [/workspace/task]
    writable_roots: [/workspace/output/work]
    denied_paths: [/run/praxist-control]
```

Only these four top-level policy fields are accepted. A nonempty
`model_by_role` must cover every dispatched role. Each entry requires `model`;
`reasoning_effort` accepts `auto`, `off`, `low`, `high`, `max`, or `xhigh`, subject
to runtime support, and defaults to `max`. Execution roles are separate from
task-local role-skill names.

Tool names are exact, not wildcard patterns. The effective allowlist is the
intersection of task and request permissions; task policy does not enable an
unavailable tool or register a server. Include required MCP tool names explicitly,
such as `mcp__SERVER__TOOL`. Network-off intent also removes built-in web tools.
The optional `tool_execution_timeout_seconds` accepts a positive whole number
or `null`. A finite value limits supported MCP tool execution and is capped by
any request budget. Omission or `null` adds no Praxist per-tool budget and omits
the native timeout override. It is not a universal per-shell-process timeout.
Agent requests are separately bounded by the active phase or research deadline
when one is configured.

Uncapped Praxist research does not remove SDK or provider transport limits. With
no override, the supported Codex runtime still applies a finite SDK-native MCP
call timeout; this is a per-call transport limit, not a whole-run ceiling.
Long-running tools must handle that boundary separately, for example through a
supported job-status polling protocol. Provider request limits, connection
timeouts, and rate limits can also interrupt individual operations.

Filesystem intent accepts `read_only`, `workspace_write`, or `full`; network
accepts `"on"` or `"off"`. Path lists use absolute POSIX paths without globs or
`..`. Providing any path-list field enables explicit path scope; omitted lists
are empty. Writable roots are also readable. Intersected scopes cannot widen a
call's permissions, and denied paths override descendant grants.

Strict task-wide sandbox/path and tool policies require
`agent_runtime:codex_sdk`; Claude SDK rejects these policies rather than silently
weakening them. There is no separate `strict_codex` configuration field. Explicit
paths select a native Codex profile that denies root access except minimal
runtime reads and the declared grants. Codex runs headlessly and requires
`approval: auto`.
Without explicit path scope, full filesystem access plus network-off is not
supported. Codex uses its shell for `Read`/`Glob`/`Grep`, so those names require
either shell permission or a read-only filesystem. Exact tool restrictions
map to native capability groups, not independent enforcement of each alias; see
[Agent Runtimes](agent-runtimes.md).

## Controller authority

For deployments where the controller has more privilege than peers, set
`PRAXIST_CONTROLLER_STATE_DIR` to a controller-owned private directory outside
peer access, such as `/run/praxist-control`. Praxist creates a per-run private
checkpoint directory with mode `0700` and a startup authority file with mode
`0600`. The private root must have trusted ancestors and no symlinks, and be
separate from task and run outputs. The task cannot override this operator
setting through `runtime_environment.env`. Private checkpoints are authoritative; the
public `lifecycle/state.json` is an observation copy and omits frozen phase
inputs. Missing or corrupt private startup authority does not fall back to a
public snapshot. Without the private directory, lifecycle state lives under the run
root and provides replay, not a security boundary against writable peers.

The deployment must keep the public run root and its ancestors controller-owned
and non-replaceable by peers, and grant peer writes only to designated work
subdirectories. With private controller state enabled, startup rejects an existing
run root owned by another user or writable by its group or others.
Controller-written artifact and state directories must also
remain protected. Keep task callback source and private control storage outside
those writable paths. Use separate OS identities or an equivalent container
boundary; a path in YAML does not make a same-identity peer unprivileged.
Runtime sandbox settings govern agent execution, not trusted callback code or
an independently running MCP server's filesystem/network authority. Configure
those host boundaries separately. Restrictive Codex requests isolate Python MCP
imports; the interpreter, installed dependencies, and editable source roots
must still be trusted and outside peer-writable paths. Verify the actual runtime
and OS deployment before relying on isolation.
