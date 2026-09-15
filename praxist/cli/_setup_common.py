"""Shared helpers for Praxist host setup and diagnostics CLI commands."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import shutil
import stat
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from praxist import __version__
from praxist.cli._env import (
    AGENT_SYSTEM_TO_RUNTIME_REF,
    AGENT_SYSTEM_VALUES,
    PROVIDER_KEY_MAP,
    PROVIDER_REF_FOR_SHORT_NAME,
    agent_system_for_runtime_ref,
)

SETUP_PROFILE_ENV_VAR = "PRAXIST_SETUP_PROFILE"
"""Profile ID explicitly selected through ``praxist setup``."""

PRAXIST_ENV_VARS = {
    "PRAXIST_AGENT_SYSTEM",
    "PRAXIST_LLM_PROVIDER",
    "PRAXIST_MODEL",
    SETUP_PROFILE_ENV_VAR,
}
"""Praxist-owned non-secret variables written by ``praxist configure-llm``."""

CLI_SELECTOR_ENV_VARS = {
    "MODEL",
    "MODEL_PROVIDER_REF",
    "PRAXIST_AGENT_RUNTIME_REF",
    "PRAXIST_MODEL_PROVIDER_REF",
    "RUNTIME_REF",
}
"""Canonical and legacy lifecycle selectors accepted from Praxist env files."""

LOADABLE_ENV_VARS = PRAXIST_ENV_VARS | CLI_SELECTOR_ENV_VARS | set(PROVIDER_KEY_MAP.values())
"""Praxist env-file variables that command entrypoints may import into process env."""


@dataclass(frozen=True)
class Check:
    """One ``praxist doctor`` readiness check."""

    name: str
    status: str
    detail: str = ""
    variable: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return a stable JSON-friendly representation."""
        out = {"name": self.name, "status": self.status, "detail": self.detail}
        if self.variable:
            out["variable"] = self.variable
        return out


def xdg_config_dir() -> Path:
    """Return the user-level Praxist config directory."""
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "praxist"


def xdg_state_dir() -> Path:
    """Return the user-level Praxist state directory."""
    return Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "praxist"


def xdg_data_dir() -> Path:
    """Return the user-level Praxist data directory."""
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "praxist"


def default_env_file() -> Path:
    """Return the default Praxist env config file path."""
    return xdg_config_dir() / "env"


def selected_config_file(raw: str | Path | None = None) -> Path:
    """Return the explicitly selected or default Praxist config file.

    Command-line values take precedence over ``PRAXIST_CONFIG_FILE``.  Keeping
    this selection in one helper prevents setup, diagnostics, and lifecycle
    commands from silently reading different configuration files.
    """

    candidate = str(raw or os.environ.get("PRAXIST_CONFIG_FILE", "")).strip()
    return Path(candidate).expanduser().resolve() if candidate else default_env_file()


def project_env_file() -> Path:
    """Return the Praxist env file in the current working directory."""
    return Path.cwd() / ".env"


def task_env_file(task_path: Path | None) -> Path | None:
    """Return the explicit task-local env file, if a task path was supplied."""

    if task_path is None:
        return None
    root = task_path.expanduser().resolve()
    return (root.parent if root.name == "task.yaml" else root) / ".env"


def provider_key_var(provider: str) -> str:
    """Map a provider or provider-ref string to its API key env var."""
    normalized = provider.strip().lower()
    normalized = normalized.removeprefix("model_provider:")
    normalized = {
        "anthropic_messages": "anthropic",
        "openai_compatible": "openai",
        "deepseek_alias": "deepseek",
        "groq_alias": "groq",
        "mistral_alias": "mistral",
        "xai_alias": "xai",
        "dashscope": "qwen",
    }.get(normalized, normalized)
    try:
        return PROVIDER_KEY_MAP[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown provider {provider!r}") from exc


def provider_short_name(provider: str) -> str:
    """Return the normalized provider name persisted as ``PRAXIST_LLM_PROVIDER``."""
    normalized = provider.strip().lower().removeprefix("model_provider:")
    return {
        "anthropic_messages": "anthropic",
        "openai_compatible": "openai",
        "deepseek_alias": "deepseek",
        "groq_alias": "groq",
        "mistral_alias": "mistral",
        "xai_alias": "xai",
        "dashscope": "qwen",
    }.get(normalized, normalized)


def provider_plugin_ref(provider: str) -> str:
    """Return a model-provider plugin ref without closing the plugin extension point."""

    normalized = provider.strip().lower().removeprefix("model_provider:")
    built_in = {
        "anthropic_messages": "anthropic",
        "openai_compatible": "openai",
        "deepseek_alias": "deepseek",
        "groq_alias": "groq",
        "mistral_alias": "mistral",
        "xai_alias": "xai",
    }.get(normalized, normalized)
    if built_in in PROVIDER_REF_FOR_SHORT_NAME:
        return PROVIDER_REF_FOR_SHORT_NAME[built_in]
    if not normalized or any(char.isspace() for char in normalized):
        raise ValueError(f"invalid provider name {provider!r}")
    return f"model_provider:{normalized}"


def read_env_file(path: Path) -> tuple[list[str], dict[str, str]]:
    """Read a shell-style Praxist env file preserving unparsed lines."""
    if not path.exists():
        return [], {}
    lines = path.read_text(encoding="utf-8").splitlines()
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("export "):
            body = stripped[len("export ") :]
        else:
            body = stripped
        if "=" not in body or body.startswith("#"):
            continue
        key, _, raw_value = body.partition("=")
        key = key.strip()
        if key:
            try:
                parsed = shlex.split(raw_value)
                values[key] = parsed[0] if parsed else ""
            except ValueError:
                values[key] = raw_value.strip("\"'")
    return lines, values


def load_env_file(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load Praxist-managed env-file values into ``os.environ``.

    Explicit process environment values win by default. Empty strings are
    treated as unset so tests, shell wrappers, and CI can clear a variable
    while still allowing the user-level Praxist config to supply it.
    """
    return load_env_files([path or default_env_file()], override=override)


def load_env_files(paths: list[Path], *, override: bool = False) -> dict[str, str]:
    """Load Praxist-managed env files in order, preserving explicit process env.

    Later files override earlier files, but values that were already present in
    the process environment before loading are kept unless ``override`` is true.
    """
    protected = set() if override else {key for key in LOADABLE_ENV_VARS if os.environ.get(key)}
    loaded: dict[str, str] = {}
    for path in paths:
        _, values = read_env_file(path)
        for key, value in values.items():
            if key not in LOADABLE_ENV_VARS:
                continue
            if key in protected:
                continue
            os.environ[key] = value
            loaded[key] = value
    return loaded


def load_cli_environment(
    task_path: Path | None = None,
    *,
    config_file: Path | None = None,
    override: bool = False,
) -> dict[str, str]:
    """Load the user config and one explicit task-local env file.

    Process environment values keep highest precedence. A task-local file is
    loaded only when the caller selected a task path; the current working
    directory is never guessed as a second secret source.
    """

    paths = [config_file or default_env_file()]
    task_file = task_env_file(task_path)
    if task_file is not None:
        paths.append(task_file)
    return load_env_files(paths, override=override)


def write_env_file(
    path: Path,
    updates: dict[str, str],
    *,
    remove_keys: set[str] | None = None,
) -> None:
    """Update Praxist-owned env assignments while preserving unrelated lines."""
    existing_lines, _ = read_env_file(path)
    update_keys = set(updates)
    remove_keys = set(remove_keys or ())
    written_keys: set[str] = set()
    out: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        body = stripped[len("export ") :] if stripped.startswith("export ") else stripped
        key = body.partition("=")[0].strip() if "=" in body else ""
        if key in remove_keys:
            continue
        if key in update_keys:
            out.append(format_export(key, updates[key]))
            written_keys.add(key)
        else:
            out.append(line)
    for key in sorted(update_keys - written_keys):
        out.append(format_export(key, updates[key]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with suppress(OSError):
        path.parent.chmod(path.parent.stat().st_mode | stat.S_IRWXU)
    payload = "\n".join(out).rstrip()
    if payload:
        payload += "\n"
    write_path = path.resolve(strict=False) if path.is_symlink() else path
    write_path.parent.mkdir(parents=True, exist_ok=True)
    with suppress(OSError):
        write_path.parent.chmod(write_path.parent.stat().st_mode | stat.S_IRWXU)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{write_path.name}.",
        suffix=".tmp",
        dir=write_path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, write_path)
    finally:
        with suppress(OSError):
            temporary.unlink()


def format_export(key: str, value: str) -> str:
    """Format one shell-compatible export line."""
    return f"export {key}={shlex.quote(value)}"


def bundled_skill_dirs() -> list[Path]:
    """Return bundled Praxist skill directories in source or wheel installs."""
    candidates: list[Path] = []
    package_root = resources.files("praxist")
    resource_root = package_root / "resources" / "skills"
    if resource_root.is_dir():
        for child in resource_root.iterdir():
            if child.is_dir() and (child / "SKILL.md").is_file():
                candidates.append(Path(str(child)))

    source_root = Path(__file__).resolve().parents[2] / "skills"
    if source_root.is_dir():
        for child in source_root.iterdir():
            if child.is_dir() and (child / "SKILL.md").is_file():
                candidates.append(child)

    seen: set[str] = set()
    unique: list[Path] = []
    for path in sorted(candidates, key=lambda item: item.name):
        if path.name in seen:
            continue
        seen.add(path.name)
        unique.append(path)
    return unique


def default_codex_skills_dir() -> Path:
    """Return the default Codex skills registration directory."""
    return Path(os.environ.get("CODEX_SKILLS_DIR", str(Path.home() / ".agents/skills")))


def default_claude_skills_dir() -> Path:
    """Return the default Claude Code skills registration directory."""
    return Path(os.environ.get("CLAUDE_SKILLS_DIR", str(Path.home() / ".claude/skills")))


def default_skills_dir(target: str) -> Path:
    """Return the standard registration directory for one supported skill host."""

    if target == "codex":
        return default_codex_skills_dir()
    if target == "claude":
        return default_claude_skills_dir()
    raise ValueError(f"unsupported skill host: {target}")


def copy_skill_tree(source: Path, dest: Path) -> None:
    """Copy one skill directory, replacing an existing Praxist-managed directory."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def write_skill_marker(
    dest: Path,
    *,
    source: str,
    skill_name: str | None = None,
) -> None:
    """Write the Praxist-managed skill marker file."""
    marker = {
        "managed_by": "praxist",
        "package": "praxist",
        "skill_name": skill_name or dest.name,
        "version": __version__,
        "source": source,
        "tree_digest": skill_tree_digest(dest),
    }
    (dest / ".praxist-skill.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def skill_tree_digest(root: Path) -> str:
    """Return a deterministic digest of a skill tree, excluding its marker."""

    digest = hashlib.sha256()
    marker_name = ".praxist-skill.json"
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative == marker_name:
            continue
        stat_result = path.lstat()
        if stat.S_ISLNK(stat_result.st_mode):
            kind = "symlink"
            payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
        elif stat.S_ISDIR(stat_result.st_mode):
            kind = "directory"
            payload = b""
        elif stat.S_ISREG(stat_result.st_mode):
            kind = "file"
            file_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            payload = file_digest.digest()
        else:
            kind = "other"
            payload = str(stat_result.st_mode).encode("ascii")
        digest.update(kind.encode("ascii"))
        digest.update(b"\0")
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def praxist_console_path() -> str:
    """Return the visible ``praxist`` console path, if any."""
    return shutil.which("praxist") or "(not on PATH)"


def version_checks() -> list[Check]:
    """Return package version checks for ``praxist doctor``."""
    return [
        Check("praxist_package", "ok", f"{__version__} {Path(__file__).parents[1]}"),
        Check(
            "praxist_console", "ok" if shutil.which("praxist") else "warn", praxist_console_path()
        ),
    ]


def python_check() -> Check:
    """Return current Python version check."""
    version = ".".join(str(part) for part in sys.version_info[:3])
    status = "ok" if sys.version_info >= (3, 11) else "missing"
    return Check("python", status, f"{version} {sys.executable}")


def platform_check() -> Check:
    """Report whether the host provides the supported runtime primitives."""

    supported = sys.platform.startswith("linux") or sys.platform == "darwin"
    detail = sys.platform
    if not supported:
        detail += "; research runs require Linux, macOS, or WSL"
    return Check("platform", "ok" if supported else "missing", detail)


def cli_checks(agent_system: str | None = None) -> list[Check]:
    """Return readiness checks for the selected production peer runtime."""

    specs_by_agent = {
        "codex_sdk": (
            ("openai_codex", "openai-codex"),
            ("mcp", "mcp"),
            ("claude_agent_sdk", "claude-agent-sdk"),
        ),
        "claude_sdk": (("claude_agent_sdk", "claude-agent-sdk"),),
    }
    selected = agent_system or os.environ.get("PRAXIST_AGENT_SYSTEM", "").strip()
    if selected:
        agents = (selected,) if selected in specs_by_agent else ()
    else:
        agents = tuple(value for value in AGENT_SYSTEM_VALUES if value in specs_by_agent)
    checks: list[Check] = []
    for name in agents:
        details: list[str] = []
        missing: list[str] = []
        for module, distribution in specs_by_agent[name]:
            available = importlib.util.find_spec(module) is not None
            try:
                version = importlib.metadata.version(distribution) if available else ""
            except importlib.metadata.PackageNotFoundError:
                version = ""
            if not available:
                missing.append(distribution)
            details.append(
                f"{distribution} {version}" if version else f"{distribution} is not installed"
            )
        checks.append(
            Check(
                name,
                "missing" if missing else "ok",
                "; ".join(details),
            )
        )
    return checks


def config_checks(
    agent_system: str | None = None,
    *,
    provider_override: str | None = None,
    model_override: str | None = None,
    saved_login_only: bool = False,
) -> list[Check]:
    """Return provider/model/key config checks without printing raw values."""
    provider = (
        provider_override
        if provider_override is not None
        else os.environ.get("PRAXIST_LLM_PROVIDER", "").strip()
    )
    selected_agent_system = (
        agent_system or os.environ.get("PRAXIST_AGENT_SYSTEM", "").strip() or "claude_sdk"
    )
    model = (
        model_override
        if model_override is not None
        else os.environ.get("PRAXIST_MODEL", "").strip()
    )
    checks = [
        Check(
            "PRAXIST_AGENT_SYSTEM",
            "ok" if selected_agent_system in AGENT_SYSTEM_VALUES else "missing",
            selected_agent_system,
        ),
        Check(
            "PRAXIST_LLM_PROVIDER",
            "ok" if provider else "warn",
            provider or "runtime default",
        ),
        Check("PRAXIST_MODEL", "ok" if model else "warn", model or "provider default"),
    ]
    if provider:
        try:
            var = provider_key_var(provider)
        except ValueError:
            checks.append(Check("provider_key", "warn", f"unknown provider {provider!r}"))
        else:
            if saved_login_only:
                checks.append(
                    Check(
                        "provider_key",
                        "ok",
                        "ignored in Codex-native mode; saved login required",
                    )
                )
                return checks
            checks.append(
                Check(
                    "provider_key",
                    (
                        "ok"
                        if os.environ.get(var)
                        or (
                            selected_agent_system == "codex_sdk"
                            and provider_short_name(provider) == "openai"
                        )
                        else "missing"
                    ),
                    "present" if os.environ.get(var) else "not set",
                    variable=var,
                )
            )
    return checks


def normalize_runtime_selection(
    *,
    agent_system: str | None,
    runtime_ref: str | None,
    default_agent_system: str,
) -> tuple[str, str]:
    """Resolve and cross-check an agent-system/runtime pair.

    Explicit CLI values outrank persisted environment values. Supplying only a
    built-in runtime therefore derives its matching agent system instead of
    being rejected by an older ``PRAXIST_AGENT_SYSTEM`` setting.
    """

    explicit_runtime = str(runtime_ref or "").strip()
    explicit_agent = str(agent_system or "").strip().lower()
    env_runtime = (
        os.environ.get("PRAXIST_AGENT_RUNTIME_REF", "").strip()
        or os.environ.get("RUNTIME_REF", "").strip()
    )
    env_agent = os.environ.get("PRAXIST_AGENT_SYSTEM", "").strip().lower()

    if explicit_runtime:
        selected_runtime = explicit_runtime
        runtime_agent = agent_system_for_runtime_ref(selected_runtime)
        selected_agent = explicit_agent or runtime_agent or env_agent or default_agent_system
    elif explicit_agent:
        selected_agent = explicit_agent
        selected_runtime = AGENT_SYSTEM_TO_RUNTIME_REF.get(selected_agent, "")
        runtime_agent = agent_system_for_runtime_ref(selected_runtime)
    else:
        selected_runtime = env_runtime
        runtime_agent = agent_system_for_runtime_ref(selected_runtime) if selected_runtime else None
        selected_agent = runtime_agent or env_agent

    if not selected_agent:
        selected_agent = runtime_agent or default_agent_system
    if selected_agent not in AGENT_SYSTEM_VALUES:
        raise ValueError(
            f"unknown PRAXIST_AGENT_SYSTEM={selected_agent!r}; expected one of {AGENT_SYSTEM_VALUES}"
        )
    if runtime_agent is not None and runtime_agent != selected_agent:
        raise ValueError(
            f"agent system {selected_agent!r} conflicts with runtime {selected_runtime!r}"
        )
    return selected_agent, selected_runtime or AGENT_SYSTEM_TO_RUNTIME_REF[selected_agent]
