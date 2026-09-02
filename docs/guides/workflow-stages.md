# Workflow Stages

Workflow stages are executable steps in a Praxist run.

## Research Loop

`workflow_stage:research_loop` is mandatory. It owns the peer cohort, shared
findings, frontier, finding-graph guidance, Principal Investigator (PI) and
Chair synthesis, prompt layout, generation boundaries, and run artifacts.

The stage also owns the Deep Innovation Gate (DIG), a generation-scoped
pre-code design process, and the independently configurable
Quality-Diversity (QD) allocation path. See
[Deep Innovation Gate](deep-innovation-gate.md) and
[Quality-Diversity Allocation](qdig-cohort-allocator.md).

Each generation materializes its executable topology in
`gen_<N>/research_topology.json` before running the standard parallel peer
cohort. See [Research Topology Audit API](research-topology-and-module-api.md).

## Interface Placeholders

`workflow_stage:ideation_stub` and `workflow_stage:paper_writing_stub` are
registered interface placeholders, not product modules. They remain disabled
by default and do not provide ideation or paper-writing workflows.

Do not implement behavior for these placeholders as an isolated or casual
contribution. Before either stage becomes a supported workflow, a design
proposal should define its expected artifacts, lifecycle, budget behavior,
and test/conformance requirements.

Executable stage semantics belong in Python workflow plugins and should include
the corresponding conformance coverage.

## Local Reviewer

`workflow_stage:reviewer_stub` provides an optional local artifact and
provenance review when explicitly run in one of these modes:

```text
local
artifact
artifacts
run_artifact
claim_check
review
```

The reviewer reads `artifact_index.jsonl`, `trajectory.jsonl`, and
`run_summary.json`, verifies artifact hashes and references, and writes
`workflow/reviewer_report.json`. It does not rerun evaluators, assess scientific
quality, or affect frontier, incubator, Gems, or leaderboard state. It refuses
to append after `run.finalized`, preserving that event as the trajectory
terminus.

## Stage Contract

An executable stage must:

- validate its input contract;
- request budget before expensive work;
- emit lifecycle events;
- write replayable artifacts;
- preserve partial outputs where safe;
- report terminal status.

Stage semantics belong in Python workflow plugins, not shell wrappers or task
harnesses.
