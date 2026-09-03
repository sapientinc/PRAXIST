# Contributing

Praxist welcomes bug reports, improvement ideas, documentation fixes, and code
contributions. Contributors only need to open an issue or pull request; the
maintainer team takes responsibility for integration, full verification, and
real-task validation before a change reaches `main`.

## Submit an Issue or Pull Request

An issue should explain the observed behavior, the expected behavior, and any
reproduction evidence that can be shared safely.

A pull request should answer four questions:

1. What changed?
2. Why is the change needed?
3. How can the behavior be checked?
4. Is there anything maintainers should know about compatibility or risk?

Focused tests are appreciated when practical, but contributors are not expected
to run paid model calls, multi-generation research tasks, or the complete CI
matrix. Draft pull requests are welcome when early feedback would help.

## Work With the Architecture

Praxist separates framework code, generic plugins, task templates, complete
examples, and external task projects. Where practical, we ask contributors to
follow these existing boundaries so changes remain reusable across research
domains. The maintainers will help adapt a sound contribution when repository
conventions are unfamiliar.

Before making a substantial change, please consult:

- `AGENTS.md` for the machine-friendly repository contract;
- `docs/concepts/architecture.md` for the active architecture; and
- `docs/index.md` for the documentation source map.

Keep changes focused, avoid task-specific assumptions in shared code, and update
tests and documentation at the affected boundary.

## What Happens After Submission

1. **Review.** Maintainers examine each open issue and pull request for
   correctness, scope, compatibility, and overlap with other contributions.
2. **Integration.** Issue fixes are prepared on maintainer-owned validation
   branches. Closely related pull requests may be combined on one
   `validation/<topic>` branch so their interaction can be tested together.
3. **Automated verification.** The selected implementation receives focused
   tests, the full test suite, and local CI. The validation branch is frozen at
   an exact commit while evidence is collected.
4. **Real-task validation.** Changes that can affect runtime or research
   behavior are exercised on representative task projects. This commonly takes
   one to three days because the team also checks generation-level performance
   trends and research-loop integrity.
5. **Merge.** A validated branch is reconciled with the current `main`, checked
   again, merged, and then deleted. A branch that exposes a regression returns
   to implementation and validation instead.

Documentation-only and similarly isolated changes can use a faster path when
they cannot affect runtime behavior. The maintainer team runs at least one
review and validation cycle each week; validated changes are merged as soon as
they are ready.

## Pull Request Transfer and Credit

When a contribution moves to a maintainer-owned validation branch, the original
pull request may be closed after a maintainer links the destination branch and
commit. This means the contribution has entered validation, not that it was
rejected.

Original commits are retained when suitable. If maintainers need to combine or
adapt overlapping implementations, contributor credit is preserved with
recognized co-author attribution. The related issue remains available for
validation updates, findings, and the final outcome.

## Verification Responsibilities

Contributors should run the narrow checks they can reasonably run and state what
was not tested. Praxist automation and maintainers are responsible for the full
merge gate, platform checks, integration work, and any costly task runs.

For contributors who want to run the same repository checks locally:

```bash
uv sync --group dev --extra docs
uv run python -m unittest discover -s tests -q
uv run python scripts/run_test_coverage.py unit --fail-under 90 --fail-under-statements 95
uv run python scripts/run_test_coverage.py integration
uv run python -m compileall -q praxist tests templates examples scripts
uv run python scripts/build_docs_site.py
git diff --check
```

Coverage reports are written to the ignored `cover/unit/` and
`cover/integration/` directories.

## Changes That Need Extra Care

Changes to startup, process ownership, plugin resolution, task paths,
credentials, runtime invocation, prompt layout, peer scheduling, Finding Graph
guidance, budgets, replay verification, research retention mechanisms, or run
artifact schemas can affect unrelated task projects. These changes require
focused contract tests and maintainer-run task validation before merge.

Public and semi-public Python APIs use Google-style docstrings. Describe the
contract that callers can rely on, rather than the history of one fix. Comments
should explain only non-obvious invariants, recovery behavior, or failure
policy.
