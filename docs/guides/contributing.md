# Contributing

This repository is edited by both humans and agents. The durable rule is simple:
keep system code, generic plugins, task templates, complete examples, and real
task projects physically separate.

## Read First

- `AGENTS.md` is the machine-friendly repository contract.
- `docs/concepts/architecture.md` is the active architecture overview.
- `docs/index.md` is the source documentation entry.

## Good First Contributions

New contributors who are not yet familiar with the system's architecture are encouraged to start with well-scoped, low-risk changes. The following areas are suitable for a first contribution, particularly when no open issues are available.

### Recommended starter lanes

- **Graduate one small file from Pyrefly excludes**: Identify a file currently excluded from Pyrefly static checks, address the underlying issues, and re-enable checking.
- **Add focused tests around existing contracts**: Improve test coverage for a specific module or function without altering its behavior.
- **Improve generated-reference consistency checks**: Enhance the tooling that verifies cross-references in the documentation.
- **Fix documentation links or wording**: Correct broken links or improve clarity, while respecting the one-fact-one-owner policy.
- **Add plugin-local tests for existing generic plugins**: Write tests that verify plugin behavior in isolation.

### Areas to avoid for first contributions

The following areas involve core architectural surfaces and require deep system understanding. They are not recommended for first-time contributors:

- Startup semantics
- Credential selection
- Runtime invocation
- Resume boundaries
- Workflow-stage behavior

For a detailed explanation of these and other high-risk areas, refer to the [What Requires Extra Care](#what-requires-extra-care) section below.

## Default Change Flow

1. Identify whether the change belongs to core, a generic plugin, a task template,
   a complete example, an external task project, tests, docs, or operator scripts.
2. Make the smallest change that fits the existing boundary.
3. Add or update tests at the same boundary.
4. Rebuild docs when docstrings or docs changed.
5. Record high-risk architectural implementation context in the commit or
   pull-request description.

## Stable Docstrings

Public and semi-public Python APIs use Google-style docstrings. A docstring
should describe the contract that future callers can rely on, not the history of
one bug fix.

Use comments for non-obvious invariants, recovery behavior, and failure policy.
Do not add comments that only restate the next line of code.

## Default Verification

```bash
uv sync --group dev --extra docs
uv run python -m unittest discover -s tests -q
uv run python scripts/run_test_coverage.py unit --fail-under 90 --fail-under-statements 95
uv run python scripts/run_test_coverage.py integration
uv run python -m compileall -q praxist tests templates examples scripts
uv run python scripts/build_docs_site.py
git diff --check
```

Narrow changes can start with narrow tests, but the default handoff should still
include the full suite above.

The coverage command writes ignored local reports under `cover/unit/` and
`cover/integration/`. The `unit` profile is the offline non-integration test
layers and is held at 90% branch-aware total coverage plus 95% statement
coverage; the `integration` profile is `tests/integration` and remains
observational. It uses `coverage.py` from the dev dependency group and keeps the
test runner on `unittest`.

## What Requires Extra Care

Extra care is required when changing startup, plugin resolution, task path
resolution, credential selection, runtime invocation, prompt layout, event-driven
peer scheduling, finding graph guidance, budget policy, replay verification, or
run artifact schemas.

Those changes should include focused tests and, when they change architecture
contracts, a concise rationale in the pull-request description.
