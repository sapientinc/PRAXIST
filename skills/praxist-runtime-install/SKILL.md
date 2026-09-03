---
name: praxist-runtime-install
description: Install Praxist runtime dependencies and configure user-level Praxist provider credentials for a source checkout or pip-installed environment. Use when the user asks an agent to install or repair Praxist requirements, prepare a Praxist host, create the Praxist Python environment, install the `praxist` CLI, install Claude SDK or official Codex SDK runtime extras, install source-checkout test/dev dependencies, persist API keys or provider settings, verify imports, or diagnose missing dependencies. For source checkouts include the repository test/dev dependency group; for pip package installs keep the install runtime-only. Do not use for docs, task-specific training, dataset, benchmark, or experiment dependencies.
---

# Praxist Runtime Install

Use this skill to install dependencies needed to run Praxist itself. For local source checkouts, also install the repository's test/dev dependency group so the agent can validate Praxist code changes. Keep task-specific dependencies separate: each task project owns its own training/evaluation environment, datasets, CUDA/PyTorch stack, and benchmark packages.

This is separate from `scripts/install_codex_skills.sh`: that compatibility script installs or updates Codex or Claude Code skill symlinks. The bundled `scripts/install_runtime_deps.sh` in this skill installs Praxist runtime Python dependencies and, when explicitly supplied with provider/key options, writes user-level Praxist environment config.

## Scope

Install dependencies for the Praxist control plane:

- base package: `praxist`
- default runtime extras: `agents,codex`
- tested `claude-agent-sdk==0.2.136` plus MCP support for
  `agent_runtime:claude_sdk` when a package or repaired install did not already
  provide them
- tested `openai-codex==0.147.0`, `codex-relay==0.5.5`, and MCP support for
  `agent_runtime:codex_sdk`; this peer runtime uses the Python SDK and its local
  app-server. The SDK-bundled Codex binary can own saved ChatGPT authentication
  for native OpenAI, but Praxist does not reuse an interactive CLI process
- source-checkout validation dependencies: repository `dependency-groups.dev`
- user-level provider config: API keys, `PRAXIST_LLM_PROVIDER`, `PRAXIST_AGENT_SYSTEM`, and `PRAXIST_MODEL` only when the user supplies them
- optional remote-storage extra: `storage`, only when the user asks for S3/object-storage support
- built-in pseudonymous product-usage client and consent CLI; no separate SDK
  package or service dependency is required for ordinary Praxist operation
- bundled Praxist User Agreement, scrollable review command, and local
  version-bound acceptance record
- optional operator CLIs: report whether the human-facing `codex` command is
  present. The SDK-bundled binary is sufficient for Codex-native login, so do
  not install a global CLI unless the user explicitly asks

Use this install split:

- source checkout: install runtime extras plus Praxist repository test/dev dependencies
- pip package: install runtime extras only; do not install test/dev dependencies
- no task-local packages unless the user separately asks to set up a task project
- documentation dependencies come only from the source checkout's `docs`
  optional dependency group; do not invent a parallel requirements file

## Default Workflow

1. Identify the install surface.
   - Source checkout: a directory with `pyproject.toml` declaring `name = "praxist"` and an `praxist/` package.
   - Pip package: no source checkout is available, or the user explicitly wants a package install.
   - Existing venv: respect an explicitly active or user-specified venv when practical.

2. Verify Python.
   - Require Python `>=3.11`.
   - Prefer `python3.11`, then `python3`, then `python`.
   - Do not change system Python.

3. Install the published package and runtime extras directly with pip:

   ```bash
   python3 -m pip install --index-url https://pypi.org/simple "praxist[agents,codex]"
   ```

   For a source checkout under active development, use its bundled helper so
   repository test dependencies are included:

   ```bash
   bash skills/praxist-runtime-install/scripts/install_runtime_deps.sh --repo /path/to/Praxist
   ```

   Add remote-storage support only when requested:

   ```bash
   bash skills/praxist-runtime-install/scripts/install_runtime_deps.sh --repo /path/to/Praxist --with-storage
   ```

4. Verify the runtime install.
   - Run `<venv>/bin/praxist --help` or `uv run praxist --help`.
   - Run `<venv>/bin/python -c "import praxist; print(praxist.__file__)"`.
   - For `agent_runtime:codex_sdk`, import `openai_codex` and `mcp`, and confirm
     the venv contains `codex-relay` and the SDK-bundled Codex binary.
   - Only when the user explicitly selects Codex-native mode, run `praxist
     setup --profile codex-native --install-skills codex` in a local terminal.
     It reuses or repairs the SDK-bundled Codex ChatGPT login and verifies the
     result. Do not make this subscription check a requirement for another
     provider/runtime profile. Never print, manually inspect, or place token
     files in a task or run directory. The runtime may internally stage
     file-based auth in a private OS-temporary home that it deletes on close.
   - Optionally run `codex --version` only to verify the separate human
     operator interface used to invoke Praxist skills directly.
   - Report API key presence only; never print secret values.

   During a first agent-managed installation, run `praxist setup
   --agent-managed` immediately after package installation. Treat its
   `next_required_action` as authoritative and rerun it after every decision.
   A usable default, existing credential, or successful doctor report does not
   mean the operator selected a profile. When it returns
   `run_doctor_then_finish_setup`, run diagnostics and stop; project selection
   and takeover are separate post-manual actions.

5. Converge on the User Agreement before configuring a runtime.
   - Run `<venv>/bin/praxist user-agreement status --json` first.
   - If the current version is not accepted, present native choices to review,
     agree and continue, or cancel. Link the packaged `docs/legal/` Markdown
     files so the operator can scroll without filling the conversation.
   - Never infer agreement from the install request and never accept on the
     operator's behalf. Only after the explicit **Agree** choice, run
     `<venv>/bin/praxist user-agreement accept --agent-reply Agree` and verify
     that status reports `accepted: true`.
   - Keep this legally required acceptance separate from the optional
     product-usage choice below.

6. Converge on the product-usage consent flow after Agreement acceptance.
   - Run `<venv>/bin/praxist product-usage status --json` first.
   - If `collection_available` is false, report that this build collects
     nothing, leave consent unset, and continue installation normally.
   - Only if `collection_available` is true, link the packaged
     `docs/legal/product-usage-data-notice.md` or its hosted page for review.
     Avoid dumping the complete notice into the conversation unless no
     scrollable file or page surface is available.
   - Explain that the purpose is improving Praxist, the scope is aggregate Peer
     lifecycle status and timestamps, the implementation is built into
     `praxist/product_usage`, and research runs work normally with either
     choice. Treat the CLI notice as the consent source of truth instead of
     restating its complete field list.
   - Offer a native two-choice interaction with no preselected answer. If
     native choices are unavailable, run the local TTY consent selector. Do
     not ask the operator to type a yes/no reply or infer consent from prose.
   - Record only the supported token corresponding to that explicit choice with
     `<venv>/bin/praxist product-usage consent --agent-reply <reply>`.
   - If the interaction is non-interactive or the user does not reply, leave
     consent unset and make no collection request.

7. If the user selects an API provider, persist its profile to the user-level
   Praxist env file and pause for local masked key entry. Never request or
   transport a raw key through chat, command-line arguments, logs, or the
   response.

8. Stop after reporting the installation/configuration result and any remaining blockers. Do not start a Praxist run unless the user explicitly asks.

## Source Checkout Policy

For a source checkout, install from the checkout itself. The default script behavior is:

```bash
uv sync --group dev --extra agents
uv pip install --python .venv/bin/python \
  --index-url https://pypi.org/simple codex-relay==0.5.5
uv pip install --python .venv/bin/python \
  --index-url https://pypi.org/simple openai-codex==0.147.0
```

when `uv` is available, otherwise:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  --index-url https://pypi.org/simple codex-relay==0.5.5
.venv/bin/python -m pip install \
  --index-url https://pypi.org/simple openai-codex==0.147.0
.venv/bin/python -m pip install -e ".[agents,codex]"
.venv/bin/python -m pip install <dependency-groups.dev from pyproject.toml>
```

Use `--method pip` if the user wants to avoid `uv` creating or updating a lock file.
The public index is scoped to the two Codex-specific packages so an incomplete
default mirror cannot silently omit them. Praxist pins both agent SDKs to the
versions exercised by its runtime and conformance tests; do not upgrade either
SDK independently. Override `PRAXIST_RUNTIME_CODEX_INDEX_URL` only when the
approved mirror carries the exact pinned packages.

## Pip Package Policy

For package installs, create or reuse a venv and install:

```bash
python -m pip install --index-url https://pypi.org/simple "praxist[agents,codex]"
```

If `--with-storage` is requested, install:

```bash
python -m pip install --index-url https://pypi.org/simple "praxist[agents,codex,storage]"
```

The package ships docs, complete examples, task templates, and skills as package
resources; do not assume a source checkout exists after pip installation. After
pip installation, `praxist setup` materializes the writable bundled examples.
For package-only automation, run `praxist examples list`, install the selected
example with `praxist examples install <name>`, and report its absolute
destination. Never run against a read-only package-resource copy.

The `codex` extra provides the official Python SDK, its app-server support,
`codex-relay`, and MCP dependencies. OpenAI connects directly. DeepSeek and
OpenRouter require the relay because they expose Chat Completions while the
Codex app-server expects Responses. Praxist owns the private run-scoped relay;
do not configure a task-local relay or start one manually per peer.

## Credential Configuration

When the user provides a provider, model, agent system, or API key as part of this skill invocation, treat that as permission to write user-level Praxist config. Default file:

```text
${XDG_CONFIG_HOME:-$HOME/.config}/praxist/env
```

The bundled script writes shell-compatible `export ...=...` lines and sets the file mode to `0600`. Praxist still reads credentials through environment variables at startup; users can activate the file for the current shell with:

```bash
set -a; . "${XDG_CONFIG_HOME:-$HOME/.config}/praxist/env"; set +a
```

Provider names map to these env vars:

| Provider | Env var |
|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `orcarouter` | `ORCAROUTER_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `cloudflare` | `CLOUDFLARE_API_KEY` (also requires `CLOUDFLARE_ACCOUNT_ID`) |
| `moonshot` / `kimi` | `MOONSHOT_API_KEY` |
| `qwen` | `DASHSCOPE_API_KEY` |
| `google` | `GOOGLE_API_KEY` |
| `mistral` | `MISTRAL_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `xai` | `XAI_API_KEY` |
| `brave` | `BRAVE_API_KEY` |

Never ask the user to provide a raw key in chat. Pause and have the operator run
the local masked configuration prompt:

```bash
praxist setup --interactive
```

The prompt displays one `*` per character and keeps the secret out of argv,
shell history, agent messages, logs, and task files. Continue only after the
operator reports that local setup completed. If this is not an interactive
terminal, leave configuration incomplete rather than transporting the key
through the conversation.

If the key is already exported, avoid handling the raw secret:

```bash
bash skills/praxist-runtime-install/scripts/install_runtime_deps.sh \
  --skip-install \
  --provider deepseek \
  --api-key-env DEEPSEEK_API_KEY
```

Provider-only configuration is allowed:

```bash
bash skills/praxist-runtime-install/scripts/install_runtime_deps.sh \
  --skip-install \
  --provider openrouter
```

This writes `PRAXIST_LLM_PROVIDER=openrouter` and leaves API keys unchanged. Do not modify shell rc files unless the user explicitly asks for shell auto-loading; if that is needed, source this dedicated env file rather than writing raw secrets into `.bashrc` or `.zshrc`.

## Safety Rules

- Do not print raw values for environment variables containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, or `CREDENTIAL`.
- Do not install OS packages with `apt`, `yum`, `dnf`, `pacman`, `brew`, `npm -g`, `cargo install`, or similar global package managers without explicit user approval.
- Do not install CUDA, PyTorch, benchmark, dataset, or task-local packages as part of this skill.
- Do not write raw credentials into shell rc files, task configs, logs, docs, or git-tracked files. Use the dedicated user-level Praxist env file above.
- Do not recurse through large directories while locating a checkout. Check only explicit candidate paths and immediate parent-level candidates.

## Result Report

Finish with a concise table:

| Item | Status |
|---|---|
| Install mode | source checkout / pip package |
| Python | executable and version |
| Praxist CLI | path and `praxist --help` result |
| Runtime extras | installed extras and exact agent SDK versions |
| Test/dev deps | installed for source checkout / omitted for pip package |
| Provider config | config file path and redacted variable names written |
| Codex SDK runtime | `openai_codex` import and `codex-relay` path |
| Codex saved auth | ChatGPT login present / not requested / corrective action |
| Operator Codex CLI | present / missing / not requested |
| User Agreement | accepted current version / awaiting explicit operator choice |
| Product-usage consent | granted / denied / unset and awaiting explicit reply |
| Remaining blockers | exact next action |

If installation fails, include the failing command and the shortest corrective action. Do not continue into task setup or run launch.
