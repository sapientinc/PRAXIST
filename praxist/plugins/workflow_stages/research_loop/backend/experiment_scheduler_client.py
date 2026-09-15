"""Client contract for the run-local central experiment scheduler."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import socket
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from praxist.task_spec import (
    declared_evaluation_entrypoint_chdir,
    declared_evaluation_entrypoint_token,
    shell_command_script_index,
)

ENV_SCHEDULER_ENDPOINT = "PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT"
ENV_SCHEDULER_CONFIG = "PRAXIST_EXPERIMENT_SCHEDULER_CONFIG"
RESOURCE_SUPPLY_DIR_NAME = "resource_supply"


def resource_supply_signal_path(gen_dir: Path, peer_id: str) -> Path:
    """Return the scheduler-owned directed supply signal for one peer."""

    safe_peer = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (peer_id or "unknown"))
    return Path(gen_dir) / RESOURCE_SUPPLY_DIR_NAME / f"{safe_peer}.json"


class SchedulerUnavailable(RuntimeError):
    """Raised when central mode is configured but its launcher is unavailable."""


class ExperimentRejected(RuntimeError):
    """Raised when a new experiment cannot be accepted."""


def is_sensitive_environment_name(name: str) -> bool:
    """Return whether an environment name commonly carries a credential."""

    normalized = name.upper()
    exact = {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ORCAROUTER_API_KEY",
        "DEEPSEEK_API_KEY",
        "API_KEY",
        "PASSWORD",
        "PGPASSWORD",
        "TOKEN",
        "SECRET",
        "DATABASE_URL",
    }
    suffixes = (
        "_KEY",
        "_API_KEY",
        "_ACCESS_KEY",
        "_SECRET_KEY",
        "_TOKEN",
        "_SECRET",
        "_PASSWORD",
        "_CREDENTIAL",
        "_COOKIE",
        "_AUTH",
        "_PRIVATE_KEY",
        "_CLIENT_SECRET",
        "_PAT",
        "_BEARER",
        "_AUTH_CONFIG",
        "_CONNECTION_STRING",
        "_DSN",
    )
    credential_url = normalized.endswith("_URL") and any(
        token in normalized for token in ("DATABASE", "POSTGRES", "MYSQL", "MONGO", "REDIS")
    )
    public_key = normalized == "PUBLIC_KEY" or normalized.endswith("_PUBLIC_KEY")
    return (
        normalized in exact or (normalized.endswith(suffixes) and not public_key) or credential_url
    )


def is_sensitive_environment_entry(name: str, value: str) -> bool:
    """Return whether an environment entry should be fingerprinted, not persisted."""

    if is_sensitive_environment_name(name):
        return True
    normalized = name.upper()
    if normalized in {
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "DOCKER_AUTH_CONFIG",
    }:
        return True
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered.startswith(("bearer ", "basic ", "sk-", "ghp_", "github_pat_")):
        return True
    if any(
        marker in lowered
        for marker in (
            "accountkey=",
            "sharedaccesssignature=",
            "password=",
            "passwd=",
            "pwd=",
            "secret=",
            "token=",
        )
    ):
        return True
    if "://" not in stripped:
        return False
    try:
        parsed = urlsplit(stripped)
    except ValueError:
        return False
    if parsed.username is not None or parsed.password is not None:
        return True
    sensitive_query_names = {
        "access_token",
        "api_key",
        "key",
        "password",
        "secret",
        "signature",
        "token",
    }
    return any(key.lower() in sensitive_query_names for key, _value in parse_qsl(parsed.query))


def recover_environment(event: dict[str, Any]) -> dict[str, str]:
    """Rebuild an exact non-secret environment from a durable scheduler event."""

    if "environment_values" in event:
        environment = {
            str(key): str(value)
            for key, value in (event.get("environment_values", {}) or {}).items()
        }
        for key, expected in (event.get("environment_sensitive_hashes", {}) or {}).items():
            current = os.environ.get(str(key))
            if current and hashlib.sha256(current.encode("utf-8")).hexdigest() == expected:
                environment[str(key)] = current
        return environment
    environment = dict(os.environ)
    for key in event.get("environment_unset", []) or []:
        environment.pop(str(key), None)
    environment.update(
        {str(key): str(value) for key, value in (event.get("environment_delta", {}) or {}).items()}
    )
    return environment


def rebase_recovered_task_context(
    command: list[str],
    environment: dict[str, str],
    cwd: object,
    *,
    current_environment: dict[str, str],
) -> tuple[list[str], dict[str, str], object]:
    """Relocate persisted task-owned paths when an unchanged task checkout moves."""

    old_root = str(environment.get("PRAXIST_TASK_PROJECT_PATH") or "").strip()
    new_root = str(current_environment.get("PRAXIST_TASK_PROJECT_PATH") or "").strip()
    if not old_root or not new_root:
        return list(command), dict(environment), cwd
    old_root = str(Path(old_root).expanduser().resolve())
    new_root = str(Path(new_root).expanduser().resolve())
    location_keys = _task_location_environment_keys(environment, current_environment)
    replacements: list[tuple[str, str]] = []
    for key in location_keys:
        previous = str(environment.get(key) or "")
        current = str(current_environment.get(key) or "")
        if previous and current and previous != current:
            replacements.append((previous, current))
    if old_root != new_root:
        replacements.append((old_root, new_root))
    replacements.sort(key=lambda item: len(item[0]), reverse=True)

    def relocate(value: object) -> str:
        relocated = str(value or "")
        for previous, current in replacements:
            path_token = re.compile(
                rf"(?<![A-Za-z0-9_./~-]){re.escape(previous)}(?![A-Za-z0-9_.~-])"
            )
            relocated = path_token.sub(
                lambda _match, replacement=current: replacement,
                relocated,
            )
        return relocated

    rebased_environment = {key: relocate(value) for key, value in environment.items()}
    for key in location_keys:
        current = current_environment.get(key)
        if current not in (None, ""):
            rebased_environment[key] = str(current)
    rebased_environment["PRAXIST_TASK_PROJECT_PATH"] = new_root
    return (
        [relocate(argument) for argument in command],
        rebased_environment,
        relocate(cwd) if cwd not in (None, "") else cwd,
    )


def sensitive_environment_matches(event: dict[str, Any]) -> bool:
    """Check that current credentials match fingerprints stored before restart."""

    for key, expected in (event.get("environment_sensitive_hashes", {}) or {}).items():
        current = os.environ.get(str(key), "")
        if hashlib.sha256(current.encode("utf-8")).hexdigest() != expected:
            return False
    return True


def semantic_experiment_key(run_id: str, generation_id: int, experiment_id: str) -> str:
    """Build the stable run-generation-science identity used for deduplication."""

    payload = f"{run_id}\0{generation_id}\0{experiment_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _task_runtime_environment_keys(environment: dict[str, str]) -> set[str]:
    return {
        key.strip()
        for key in environment.get("PRAXIST_TASK_RUNTIME_ENV_KEYS", "").split(",")
        if key.strip()
    }


def _task_location_environment_keys(
    environment: dict[str, str],
    current_environment: dict[str, str],
) -> tuple[str, ...]:
    task_runtime_keys = sorted(
        _task_runtime_environment_keys(environment)
        | _task_runtime_environment_keys(current_environment)
    )
    return tuple(
        dict.fromkeys(
            (
                "PRAXIST_TASK_PROJECT_PATH",
                "PRAXIST_WORKSPACE_ROOT",
                "PRAXIST_EVALUATION_ENTRYPOINT_PATH",
                "PRAXIST_DATASETS_DIR",
                "PRAXIST_DATA_ROOT",
                "PRAXIST_DATA_DIR",
                "PRAXIST_TASK_PYTHON",
                "PRAXIST_TASK_VENV",
                "PRAXIST_TASK_SHELL_PREFIX",
                "PRAXIST_TASK_WRITABLE_ROOTS",
                "PRAXIST_TASK_RUNTIME_ENV_KEYS",
                "VIRTUAL_ENV",
                "PATH",
                *task_runtime_keys,
            )
        )
    )


def task_runtime_context_changed(
    environment: dict[str, str],
    current_environment: dict[str, str],
) -> bool:
    """Return whether a persisted task launch location changed before execution."""

    for key in _task_location_environment_keys(environment, current_environment):
        previous = str(environment.get(key) or "").strip()
        current = str(current_environment.get(key) or "").strip()
        if previous and current and previous != current:
            return True
    return False


def _task_child_environment(environment: dict[str, str]) -> dict[str, str]:
    """Remove runner-owned Python import paths at the task-process boundary."""

    child = dict(environment)
    task_owned = _task_runtime_environment_keys(child)
    task_python = child.get("PRAXIST_TASK_PYTHON", "").strip()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        if key == "PYTHONHOME" and (key in task_owned or not task_python):
            continue
        preserved_paths: list[str] = []
        if key == "PYTHONPATH":
            guard_run_dir = child.get("PRAXIST_DELETE_GUARD_RUN_DIR", "").strip()
            guard_agent = child.get("PRAXIST_DELETE_GUARD_AGENT", "").strip()
            expected_guard_path: Path | None = None
            if guard_run_dir and guard_agent:
                expected_guard_path = (
                    Path(guard_run_dir).expanduser()
                    / ".runtime_guards"
                    / guard_agent
                    / "python_site"
                ).resolve()
            runner_root = str(child.get("PRAXIST_WORKSPACE_ROOT") or "").strip()
            resolved_runner_root = Path(runner_root).expanduser().resolve() if runner_root else None
            for raw_path in child.get("PYTHONPATH", "").split(os.pathsep):
                if not raw_path:
                    continue
                resolved_path = Path(raw_path).expanduser().resolve()
                if expected_guard_path is not None and resolved_path == expected_guard_path:
                    preserved_paths.append(str(expected_guard_path))
                elif key in task_owned:
                    # Explicit task configuration wins.  The inherited runner
                    # path is absent from task_owned, while a task may
                    # intentionally opt into the same source tree.
                    preserved_paths.append(raw_path)
                elif not task_python and (
                    resolved_runner_root is not None and resolved_path != resolved_runner_root
                ):
                    preserved_paths.append(raw_path)
        child.pop(key, None)
        if preserved_paths:
            child[key] = os.pathsep.join(dict.fromkeys(preserved_paths))
    if not task_python:
        return child
    task_bin = str(Path(task_python).expanduser().resolve().parent)
    path_parts = [part for part in (child.get("PATH") or os.defpath).split(os.pathsep) if part]
    if task_bin not in path_parts:
        child["PATH"] = os.pathsep.join([task_bin, *path_parts])
    return child


def _task_job_cwd(
    raw_cwd: object,
    environment: dict[str, str],
    *,
    require_exists: bool = True,
) -> str | None:
    """Resolve an experiment cwd, defaulting task-owned work to the task root."""

    explicit_cwd = raw_cwd not in (None, "") and bool(str(raw_cwd).strip())
    raw = str(
        raw_cwd if explicit_cwd else environment.get("PRAXIST_TASK_PROJECT_PATH") or ""
    ).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        task_root = str(environment.get("PRAXIST_TASK_PROJECT_PATH") or "").strip()
        base = Path.cwd() if explicit_cwd else Path(task_root) if task_root else Path.cwd()
        path = base / path
    path = path.resolve()
    if require_exists and not path.is_dir():
        raise ExperimentRejected(f"experiment cwd is not an existing directory: {path}")
    return str(path)


def _env_program_index(command: list[str]) -> int | None:
    index = 1
    options_with_value = {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}
    while index < len(command):
        value = command[index]
        if value == "--":
            return index + 1 if index + 1 < len(command) else None
        if value in options_with_value:
            index += 2
            continue
        if value.startswith("-") or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", value):
            index += 1
            continue
        return index
    return None


def _static_env_chdir(
    command: list[str],
    *,
    program_index: int,
) -> tuple[int, str, bool] | None:
    """Return the last static ``env --chdir`` operand before the program."""

    selected: tuple[int, str, bool] | None = None
    index = 1
    while index < program_index:
        value = command[index]
        if value == "--":
            break
        if value in {"-C", "--chdir"}:
            if index + 1 >= program_index:
                return None
            selected = (index + 1, command[index + 1], False)
            index += 2
            continue
        if value.startswith("--chdir="):
            selected = (index, value.split("=", 1)[1], True)
        if value in {"-u", "--unset", "-S", "--split-string"}:
            index += 2
            continue
        index += 1
    return selected


def _transparent_command_index(command: list[str]) -> int | None:
    """Return the wrapped program index for shell-transparent builtins."""

    if not command:
        return None
    program = Path(command[0]).name
    if program not in {"exec", "command"}:
        return None
    index = 1
    while index < len(command):
        value = command[index]
        if value == "--":
            return index + 1 if index + 1 < len(command) else None
        if program == "exec" and value == "-a":
            index += 2
            continue
        if program == "exec" and value in {"-c", "-l"}:
            index += 1
            continue
        if program == "command" and value == "-p":
            index += 1
            continue
        if value.startswith("-"):
            return None
        return index
    return None


def _normalize_task_python_command(command: list[str], environment: dict[str, str]) -> list[str]:
    task_python = environment.get("PRAXIST_TASK_PYTHON", "").strip()
    if not command or not task_python:
        return command
    normalized = list(command)
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", normalized[0]):
        normalized[0] = task_python
        return normalized
    if Path(normalized[0]).name in {"exec", "command"}:
        program_index = _transparent_command_index(normalized)
        if program_index is not None:
            normalized[program_index:] = _normalize_task_python_command(
                normalized[program_index:],
                environment,
            )
        return normalized
    if Path(normalized[0]).name == "env":
        program_index = _env_program_index(normalized)
        if program_index is not None:
            normalized[program_index:] = _normalize_task_python_command(
                normalized[program_index:],
                environment,
            )
        return normalized
    if Path(normalized[0]).name not in {"sh", "bash"}:
        return normalized
    script_index = shell_command_script_index(normalized)
    if script_index is None:
        return normalized
    task_owned = _task_runtime_environment_keys(environment)
    python_env_reset = ""
    script = normalized[script_index]
    if "PYTHONPATH" not in task_owned and "unset PYTHONPATH" not in script:
        python_env_reset += "unset PYTHONPATH; "
    if "PYTHONHOME" not in task_owned and "unset PYTHONHOME" not in script:
        python_env_reset += "unset PYTHONHOME; "
    if "PRAXIST_TASK_PYTHON=" in script:
        normalized[script_index] = python_env_reset + script
        return normalized
    quoted_python = shlex.quote(task_python)
    prelude = (
        python_env_reset + f"PRAXIST_TASK_PYTHON={quoted_python}; export PRAXIST_TASK_PYTHON; "
        'python() { command "$PRAXIST_TASK_PYTHON" "$@"; }; '
        'python3() { command "$PRAXIST_TASK_PYTHON" "$@"; }; '
    )
    normalized[script_index] = prelude + normalized[script_index]
    return normalized


def _command_path_index(command: list[str]) -> int | None:
    if not command:
        return None
    assignment_count = 0
    while assignment_count < len(command) and re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*=.*",
        command[assignment_count],
    ):
        assignment_count += 1
    if assignment_count:
        nested = _command_path_index(command[assignment_count:])
        return assignment_count + nested if nested is not None else None
    program_token = command[0]
    program = Path(program_token).name
    if program in {"exec", "command"}:
        index = _transparent_command_index(command)
        if index is None:
            return None
        nested = _command_path_index(command[index:])
        return index + nested if nested is not None else None
    if program == "env":
        index = 1
        while index < len(command):
            value = command[index]
            if value == "--":
                index += 1
                break
            if value in {"-C", "--chdir", "-S", "--split-string"} or value.startswith(
                ("--chdir=", "--split-string=")
            ):
                return None
            if value in {"-u", "--unset"}:
                index += 2
                continue
            if value.startswith("-") or ("=" in value and not value.startswith(("/", "."))):
                index += 1
                continue
            break
        nested = _command_path_index(command[index:])
        return index + nested if nested is not None else None
    python_suffix = program.removeprefix("python")
    python_parts = python_suffix.split(".")
    is_python = program_token in {"$PRAXIST_TASK_PYTHON", "${PRAXIST_TASK_PYTHON}"} or (
        program.startswith("python")
        and (
            not python_suffix
            or bool(python_parts)
            and all(part and part.isdigit() for part in python_parts)
        )
    )
    if is_python:
        index = 1
        while index < len(command):
            value = command[index]
            if value in {"-c", "-m"}:
                return None
            if value in {"-W", "-X", "--check-hash-based-pycs"}:
                index += 2
                continue
            if value == "--":
                return index + 1 if index + 1 < len(command) else None
            if value.startswith("-"):
                index += 1
                continue
            return index
        return None
    if program in {"sh", "bash"}:
        index = 1
        while index < len(command):
            value = command[index]
            if value in {"-c", "-lc"}:
                return None
            if value == "--":
                return index + 1 if index + 1 < len(command) else None
            if value.startswith("-"):
                index += 1
                continue
            return index
        return None
    if program in {".", "source"}:
        return 1 if len(command) > 1 else None
    return 0


def _replace_static_shell_words(
    script: str,
    original: list[str],
    replacement: list[str],
) -> str | None:
    """Replace parsed shell words without disturbing redirects or spacing."""

    rendered = script
    cursor = 0
    for source_token, replacement_token in zip(original, replacement, strict=True):
        token_forms = tuple(
            dict.fromkeys(
                (
                    source_token,
                    shlex.quote(source_token),
                    f'"{source_token}"',
                    f"'{source_token}'",
                )
            )
        )
        matches = [
            match
            for token_form in token_forms
            if (
                match := re.search(
                    rf"(?<![^\s;&|]){re.escape(token_form)}(?=[\s;&|]|$)",
                    rendered[cursor:],
                )
            )
            is not None
        ]
        if not matches:
            if source_token != replacement_token:
                return None
            continue
        match = min(matches, key=lambda item: item.start())
        start = cursor + match.start()
        end = cursor + match.end()
        if source_token != replacement_token:
            quoted = shlex.quote(replacement_token)
            rendered = rendered[:start] + quoted + rendered[end:]
            cursor = start + len(quoted)
        else:
            cursor = end
    return rendered


def _declared_evaluator_target(environment: dict[str, str]) -> Path | None:
    """Resolve the task-declared evaluator without consulting run-local copies."""

    task_root = str(environment.get("PRAXIST_TASK_PROJECT_PATH") or "").strip()
    resolved_path = str(environment.get("PRAXIST_EVALUATION_ENTRYPOINT_PATH") or "").strip()
    if resolved_path:
        return Path(resolved_path).expanduser().resolve()
    raw = str(environment.get("PRAXIST_EVALUATION_ENTRYPOINT") or "").strip()
    candidate_token = declared_evaluation_entrypoint_token(raw)
    if (
        not task_root
        or not candidate_token
        or any(marker in candidate_token for marker in ("$", "{", "}", "*", "?"))
    ):
        return None
    candidate = Path(candidate_token).expanduser()
    if not candidate.is_absolute():
        declared_chdir = declared_evaluation_entrypoint_chdir(raw)
        if declared_chdir:
            chdir_path = Path(declared_chdir).expanduser()
            candidate = (
                chdir_path / candidate
                if chdir_path.is_absolute()
                else Path(task_root).expanduser() / chdir_path / candidate
            )
        else:
            candidate = Path(task_root).expanduser() / candidate
    return candidate.resolve()


def _task_relative_path_is_declared_evaluator(
    candidate: Path,
    environment: dict[str, str],
) -> bool:
    declared = _declared_evaluator_target(environment)
    task_root = str(environment.get("PRAXIST_TASK_PROJECT_PATH") or "").strip()
    if declared is None or not task_root or candidate.is_absolute():
        return False
    declared_token = declared_evaluation_entrypoint_token(
        str(environment.get("PRAXIST_EVALUATION_ENTRYPOINT") or "")
    )
    declared_chdir = declared_evaluation_entrypoint_chdir(
        str(environment.get("PRAXIST_EVALUATION_ENTRYPOINT") or "")
    )
    return (
        bool(declared_token)
        and not declared_chdir
        and candidate.as_posix() == Path(declared_token).expanduser().as_posix()
    ) or (Path(task_root).expanduser().resolve() / candidate).resolve() == declared


def _shell_cd_chain_reaches_declared_evaluator(
    segments: list[tuple[list[str], str]],
    environment: dict[str, str],
) -> bool:
    """Return whether a static relative cd chain reaches the declared evaluator."""

    active = Path()
    declared = _declared_evaluator_target(environment)
    for segment, separator in segments:
        assignment_count = 0
        while assignment_count < len(segment) and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*=.*", segment[assignment_count]
        ):
            assignment_count += 1
        if assignment_count >= len(segment):
            continue
        if Path(segment[assignment_count]).name == "cd" and assignment_count + 1 < len(segment):
            previous = active
            raw = segment[assignment_count + 1]
            if raw.startswith("-") or any(
                marker in raw for marker in ("$", "{", "}", "*", "?", "~")
            ):
                return False
            candidate = Path(raw).expanduser()
            active = candidate if candidate.is_absolute() else active / candidate
            if separator in {"||", "|", "&"}:
                active = previous
            continue
        path_index = _command_path_index(segment)
        if path_index is None or path_index >= len(segment):
            continue
        candidate = Path(segment[path_index]).expanduser()
        if candidate.is_absolute():
            return declared is not None and candidate.resolve() == declared
        combined = active / candidate
        if combined.is_absolute():
            return declared is not None and combined.resolve() == declared
        if _task_relative_path_is_declared_evaluator(combined, environment):
            return True
    return False


def resolve_task_command_path(
    command: list[str],
    environment: dict[str, str],
    *,
    cwd: object = None,
) -> list[str]:
    """Resolve a static program from its explicit cwd, then the task root."""

    task_root = str(environment.get("PRAXIST_TASK_PROJECT_PATH") or "").strip()
    normalized = list(command)
    if normalized and Path(normalized[0]).name == "env":
        program_index = _env_program_index(normalized)
        chdir_spec = (
            _static_env_chdir(normalized, program_index=program_index)
            if program_index is not None
            else None
        )
        if program_index is not None and chdir_spec is not None:
            operand_index, raw_chdir, attached = chdir_spec
            if not raw_chdir or any(marker in raw_chdir for marker in ("$", "{", "}", "*", "?")):
                return normalized
            candidate = Path(raw_chdir).expanduser()
            search_roots: list[Path] = []
            if cwd not in (None, ""):
                explicit_cwd = Path(str(cwd)).expanduser()
                if not explicit_cwd.is_absolute():
                    explicit_cwd = (Path(task_root) if task_root else Path.cwd()) / explicit_cwd
                search_roots.append(explicit_cwd.resolve())
            if task_root:
                resolved_task_root = Path(task_root).expanduser().resolve()
                if resolved_task_root not in search_roots:
                    search_roots.append(resolved_task_root)
                nested_index = _command_path_index(normalized[program_index:])
                if nested_index is not None:
                    nested_token = normalized[program_index:][nested_index]
                    nested_candidate = Path(nested_token).expanduser()
                    if (
                        not nested_candidate.is_absolute()
                        and _task_relative_path_is_declared_evaluator(
                            candidate / nested_candidate,
                            environment,
                        )
                    ):
                        search_roots.remove(resolved_task_root)
                        search_roots.insert(0, resolved_task_root)
            if candidate.is_absolute():
                search_roots = [Path("/")]
            resolved_chdir = next(
                (
                    resolved
                    for root in search_roots
                    if (resolved := (root / candidate).resolve()).is_dir()
                ),
                None,
            )
            if resolved_chdir is None:
                return normalized
            normalized[operand_index] = (
                f"--chdir={resolved_chdir}" if attached else str(resolved_chdir)
            )
            normalized[program_index:] = resolve_task_command_path(
                normalized[program_index:],
                environment,
                cwd=resolved_chdir,
            )
            return normalized
    if normalized and Path(normalized[0]).name in {"sh", "bash"}:
        script_index = shell_command_script_index(normalized)
        if script_index is not None:
            script = normalized[script_index]
            static_redirect_probe = re.sub(r"\d*>\s*&\s*\d+", "", script)
            if any(marker in static_redirect_probe for marker in "`\n\r()"):
                return normalized
            try:
                lexer = shlex.shlex(script, posix=True, punctuation_chars=";&|")
                lexer.whitespace_split = True
                lexer.commenters = ""
                nested = list(lexer)
            except ValueError:
                return normalized
            for index, token in enumerate(nested):
                if token != "&":
                    continue
                descriptor_redirect = (
                    index > 0
                    and index + 1 < len(nested)
                    and re.fullmatch(r"\d*>", nested[index - 1]) is not None
                    and nested[index + 1].isdigit()
                )
                if not descriptor_redirect:
                    return normalized
            segments: list[tuple[list[str], str]] = []
            segment: list[str] = []
            for token in [*nested, ""]:
                if token in {"&&", "||", ";", "|", ""}:
                    segments.append((segment, token))
                    segment = []
                else:
                    segment.append(token)
            contains_cd = any(
                segment
                and next(
                    (
                        Path(value).name
                        for value in segment
                        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", value)
                    ),
                    "",
                )
                == "cd"
                for segment, _separator in segments
            )
            anchor_cd_chain = bool(task_root) and _shell_cd_chain_reaches_declared_evaluator(
                segments,
                environment,
            )
            active_shell_cwd: Path | None = None
            resolved_nested: list[str] = []
            for segment, separator in segments:
                resolved_segment = list(segment)
                if contains_cd:
                    assignment_count = 0
                    while assignment_count < len(resolved_segment) and re.fullmatch(
                        r"[A-Za-z_][A-Za-z0-9_]*=.*",
                        resolved_segment[assignment_count],
                    ):
                        assignment_count += 1
                    if (
                        assignment_count + 1 < len(resolved_segment)
                        and Path(resolved_segment[assignment_count]).name == "cd"
                        and not resolved_segment[assignment_count + 1].startswith("-")
                        and not any(
                            marker in resolved_segment[assignment_count + 1]
                            for marker in ("$", "{", "}", "*", "?", "~")
                        )
                    ):
                        previous_shell_cwd = active_shell_cwd
                        cd_path = Path(resolved_segment[assignment_count + 1])
                        roots: list[Path] = []
                        if cd_path.is_absolute():
                            roots.append(Path("/"))
                        elif active_shell_cwd is not None:
                            roots.append(active_shell_cwd)
                        elif anchor_cd_chain and task_root:
                            roots.append(Path(task_root).expanduser().resolve())
                        if active_shell_cwd is None and cwd not in (None, ""):
                            roots.append(Path(str(cwd)).expanduser().resolve())
                        if task_root:
                            resolved_task_root = Path(task_root).expanduser().resolve()
                            if resolved_task_root not in roots:
                                roots.append(resolved_task_root)
                        resolved_cd = next(
                            (
                                candidate
                                for root in roots
                                if (candidate := (root / cd_path).resolve()).is_dir()
                            ),
                            None,
                        )
                        if resolved_cd is not None:
                            resolved_segment[assignment_count + 1] = str(resolved_cd)
                            active_shell_cwd = resolved_cd
                        if separator in {"||", "|", "&"}:
                            active_shell_cwd = previous_shell_cwd
                    else:
                        segment_cwd: object = active_shell_cwd
                        if segment_cwd is None:
                            segment_cwd = (
                                Path(task_root).expanduser().resolve()
                                if anchor_cd_chain and task_root
                                else cwd
                            )
                        resolved_segment = resolve_task_command_path(
                            resolved_segment,
                            environment,
                            cwd=segment_cwd,
                        )
                else:
                    resolved_segment = resolve_task_command_path(
                        resolved_segment,
                        environment,
                        cwd=cwd,
                    )
                resolved_nested.extend(
                    _normalize_task_python_command(resolved_segment, environment)
                )
                if separator:
                    resolved_nested.append(separator)
            if resolved_nested != nested:
                rewritten = _replace_static_shell_words(script, nested, resolved_nested)
                if rewritten is not None:
                    normalized[script_index] = rewritten
            return normalized
    index = _command_path_index(command)
    if index is None or index >= len(command):
        return normalized
    token = command[index]
    if not token or any(marker in token for marker in ("$", "{", "}", "*", "?")):
        return normalized
    candidate = Path(token).expanduser()
    if candidate.is_absolute():
        return normalized
    search_roots: list[Path] = []
    if cwd not in (None, ""):
        explicit_cwd = Path(str(cwd)).expanduser()
        if not explicit_cwd.is_absolute():
            explicit_cwd = (Path(task_root) if task_root else Path.cwd()) / explicit_cwd
        search_roots.append(explicit_cwd.resolve())
    if task_root:
        resolved_task_root = Path(task_root).expanduser().resolve()
        if resolved_task_root not in search_roots:
            search_roots.append(resolved_task_root)
        if _task_relative_path_is_declared_evaluator(candidate, environment):
            declared_target = _declared_evaluator_target(environment)
            if declared_target is not None and declared_target.is_file():
                normalized[index] = str(declared_target)
                return normalized
            search_roots.remove(resolved_task_root)
            search_roots.insert(0, resolved_task_root)
    for search_root in search_roots:
        resolved = (search_root / candidate).resolve()
        path_is_program = index == 0
        if Path(command[0]).name == "env":
            path_is_program = index == _env_program_index(command)
        else:
            assignment_count = 0
            while assignment_count < len(command) and re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*=.*",
                command[assignment_count],
            ):
                assignment_count += 1
            path_is_program = path_is_program or index == assignment_count
        if resolved.is_file() and (not path_is_program or os.access(resolved, os.X_OK)):
            normalized[index] = str(resolved)
            break
    return normalized


def prepare_task_subprocess(
    command: list[str],
    environment: dict[str, str],
    *,
    cwd: object = None,
    require_cwd_exists: bool = True,
) -> tuple[list[str], dict[str, str], str | None]:
    """Apply the shared task interpreter, import-path, command, and cwd boundary."""

    child_environment = _task_child_environment(environment)
    resolved_cwd = _task_job_cwd(
        cwd,
        child_environment,
        require_exists=require_cwd_exists,
    )
    normalized_command = _normalize_task_python_command(
        resolve_task_command_path(list(command), child_environment, cwd=resolved_cwd),
        child_environment,
    )
    return normalized_command, child_environment, resolved_cwd


def _rpc(endpoint: str, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
    if not endpoint:
        raise SchedulerUnavailable("central scheduler endpoint is not configured")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        try:
            client.connect(endpoint)
            client.sendall((json.dumps(request) + "\n").encode("utf-8"))
            chunks = bytearray()
            while not chunks.endswith(b"\n"):
                block = client.recv(65536)
                if not block:
                    break
                chunks.extend(block)
        except (OSError, TimeoutError) as exc:
            raise SchedulerUnavailable(
                f"central scheduler unavailable at {endpoint}: {exc}"
            ) from exc
    response = json.loads(bytes(chunks).decode("utf-8"))
    if not response.get("ok"):
        raise ExperimentRejected(str(response.get("error", "scheduler request failed")))
    return response


def scheduler_endpoint_for_run(run_dir: Path | None) -> str:
    """Resolve and verify the deterministic endpoint owned by one run."""

    if run_dir is not None:
        resolved_run_dir = Path(run_dir).expanduser().resolve(strict=False)
        uid = getattr(os, "getuid", lambda: 0)()
        digest = hashlib.sha256(str(resolved_run_dir).encode()).hexdigest()[:16]
        expected = str(Path(f"/tmp/praxist-scheduler-{uid}") / f"{digest}.sock")
        endpoint_path = Path(run_dir) / "resource_scheduler" / "endpoint.json"
        if endpoint_path.exists():
            try:
                payload = json.loads(endpoint_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise SchedulerUnavailable(
                    f"central scheduler endpoint metadata is unreadable: {endpoint_path}"
                ) from exc
            if not isinstance(payload, dict):
                raise SchedulerUnavailable(
                    f"central scheduler endpoint metadata is invalid: {endpoint_path}"
                )
            endpoint = str(payload.get("endpoint") or "").strip()
            if endpoint != expected:
                raise SchedulerUnavailable(
                    f"central scheduler endpoint metadata does not match its run: {endpoint_path}"
                )
            return expected
        inherited = os.environ.get(ENV_SCHEDULER_ENDPOINT, "").strip()
        if inherited and inherited != expected:
            raise SchedulerUnavailable(
                "inherited central scheduler endpoint does not match the requested run"
            )
        return inherited
    return os.environ.get(ENV_SCHEDULER_ENDPOINT, "").strip()


def scheduler_attempt_is_active(run_dir: Path, attempt_id: str, pgid: int) -> bool:
    """Ask the run-owned scheduler whether this process group owns an attempt."""

    try:
        endpoint = scheduler_endpoint_for_run(run_dir)
        if not endpoint:
            return False
        response = _rpc(
            endpoint,
            {"action": "validate_attempt", "attempt_id": attempt_id, "pgid": pgid},
            timeout=3,
        )
    except (ExperimentRejected, SchedulerUnavailable, OSError, ValueError):
        return False
    return response.get("active") is True


def scheduler_active_process_groups(run_dir: Path) -> dict[int, tuple[int, str]]:
    """Return scheduler-owned groups bound to a live launcher identity."""

    try:
        endpoint = scheduler_endpoint_for_run(run_dir)
        if not endpoint:
            return {}
        response = _rpc(endpoint, {"action": "active_process_groups"}, timeout=3)
    except (ExperimentRejected, SchedulerUnavailable, OSError, ValueError):
        return {}
    groups: dict[int, tuple[int, str]] = {}
    for raw in response.get("groups", []):
        if not isinstance(raw, dict):
            continue
        try:
            pgid = int(raw.get("pgid", 0))
            pid = int(raw.get("pid", 0))
        except (TypeError, ValueError):
            continue
        identity = raw.get("pid_start_time")
        if isinstance(identity, int):
            token = f"proc:{identity}"
        elif isinstance(identity, str) and identity.startswith(("proc:", "ps:")):
            token = identity
        else:
            continue
        if pgid > 1 and pid > 1 and token:
            groups[pgid] = (pid, token)
    return groups


def submit_and_wait(
    command: list[str],
    *,
    peer_id: str,
    experiment_id: str,
    profile: str = "",
    work_class: str = "ordinary",
    eta_seconds: int = 0,
    run_dir: Path | None = None,
    cwd: Path | None = None,
    wait_timeout_seconds: float | None = None,
    scheduler_endpoint: str | None = None,
    retry_terminal: bool = False,
) -> int:
    """Submit one semantic experiment and wait for its terminal return code."""

    endpoint = (
        scheduler_endpoint.strip()
        if isinstance(scheduler_endpoint, str)
        else scheduler_endpoint_for_run(run_dir)
    )
    generation_id = _generation_from_peer(peer_id)
    environment = dict(os.environ)
    supply_lease_id = environment.get("PRAXIST_RESOURCE_SUPPLY_LEASE_ID", "")
    if supply_lease_id:
        try:
            active_supply = get_supply_lease(peer_id, generation_id, supply_lease_id)
        except (ExperimentRejected, SchedulerUnavailable):
            active_supply = None
        if active_supply == {}:
            environment.pop("PRAXIST_RESOURCE_SUPPLY_LEASE_ID", None)
            supply_lease_id = ""
    resolved_command, environment, resolved_cwd = prepare_task_subprocess(
        command,
        environment,
        cwd=cwd if cwd is not None else Path.cwd(),
    )
    submit = _rpc(
        endpoint,
        {
            "action": "submit",
            "peer_id": peer_id,
            "generation_id": generation_id,
            "experiment_id": experiment_id,
            "profile": profile,
            "work_class": work_class,
            "eta_seconds": eta_seconds,
            "cwd": resolved_cwd or "",
            "environment": environment,
            "supply_lease_id": supply_lease_id,
            "retry_terminal": retry_terminal,
            "command": resolved_command,
            "run_dir": str(run_dir) if run_dir else "",
        },
        timeout=10,
    )
    submitted_job = submit["job"]
    submitted_state = str(submitted_job.get("state", ""))
    if submit.get("retry_requires_explicit_request") is True and not retry_terminal:
        raise ExperimentRejected(
            f"experiment {experiment_id!r} already has terminal state {submitted_state!r}; "
            "correct the request and resubmit with retry_terminal=True "
            "(--retry-terminal in the launch CLI)"
        )
    job_id = str(submitted_job["job_id"])
    response = _rpc(
        endpoint,
        {"action": "wait", "job_id": job_id, "timeout_seconds": wait_timeout_seconds},
        timeout=None if wait_timeout_seconds is None else wait_timeout_seconds + 5,
    )
    if response.get("timeout"):
        if response["job"]["state"] == "queued":
            cancelled = _rpc(
                endpoint,
                {"action": "cancel_queued", "job_id": job_id},
                timeout=5,
            )
            if cancelled.get("cancelled"):
                raise TimeoutError(f"experiment {experiment_id!r} admission timed out")
        response = _rpc(
            endpoint,
            {"action": "wait", "job_id": job_id, "timeout_seconds": None},
            timeout=None,
        )
    job = response["job"]
    if job["state"] == "rejected":
        raise ExperimentRejected(str(job.get("error", "experiment rejected")))
    return int(job.get("exit_code") if job.get("exit_code") is not None else 2)


def register_idle_supply(peer_id: str, generation_id: int) -> dict[str, Any]:
    """Register one productive idle peer for a directed supply wakeup."""

    endpoint = os.environ.get(ENV_SCHEDULER_ENDPOINT, "")
    if not endpoint:
        return {}
    response = _rpc(
        endpoint,
        {"action": "register_idle_supply", "peer_id": peer_id, "generation_id": generation_id},
        timeout=5,
    )
    return dict(response.get("supply", {}) or {})


def unregister_idle_supply(peer_id: str, generation_id: int) -> None:
    """Withdraw one peer from supply feedback while it is not productively idle."""

    endpoint = os.environ.get(ENV_SCHEDULER_ENDPOINT, "")
    if endpoint:
        _rpc(
            endpoint,
            {
                "action": "unregister_idle_supply",
                "peer_id": peer_id,
                "generation_id": generation_id,
            },
            timeout=5,
        )


def get_supply_lease(peer_id: str, generation_id: int, lease_id: str) -> dict[str, Any]:
    """Fetch the scheduler-owned canonical payload for one wake locator."""

    endpoint = os.environ.get(ENV_SCHEDULER_ENDPOINT, "")
    if not endpoint or not lease_id:
        return {}
    response = _rpc(
        endpoint,
        {
            "action": "get_supply_lease",
            "peer_id": peer_id,
            "generation_id": generation_id,
            "lease_id": lease_id,
        },
        timeout=2,
    )
    return dict(response.get("supply", {}) or {})


def release_supply_lease(
    lease_id: str,
    peer_id: str,
    *,
    declined: bool = False,
    reason: str = "",
) -> None:
    """Release an unused directed supply lease; consumed leases are idempotent."""

    endpoint = os.environ.get(ENV_SCHEDULER_ENDPOINT, "")
    if endpoint and lease_id:
        _rpc(
            endpoint,
            {
                "action": "release_supply_lease",
                "lease_id": lease_id,
                "peer_id": peer_id,
                "declined": bool(declined),
                "reason": reason,
            },
            timeout=5,
        )


def generation_advice(peer_id: str, generation_id: int) -> dict[str, Any]:
    """Return scheduler-owned first-wave advice for one peer."""

    endpoint = os.environ.get(ENV_SCHEDULER_ENDPOINT, "")
    if not endpoint:
        return {}
    response = _rpc(
        endpoint,
        {"action": "generation_advice", "peer_id": peer_id, "generation_id": generation_id},
        timeout=2,
    )
    return dict(response.get("advice", {}) or {})


def begin_assessment(generation_id: int, reason: str = "assessment") -> bool:
    """Fence ordinary queued/new work while mature top-ups remain admissible."""

    endpoint = os.environ.get(ENV_SCHEDULER_ENDPOINT, "")
    if not endpoint:
        return False
    _rpc(
        endpoint,
        {"action": "begin_assessment", "generation_id": generation_id, "reason": reason},
        timeout=5,
    )
    return True


def freeze_generation(generation_id: int, reason: str = "") -> None:
    """Stop new admission for one generation while active work drains."""

    endpoint = os.environ.get(ENV_SCHEDULER_ENDPOINT, "")
    if endpoint:
        _rpc(
            endpoint,
            {"action": "freeze", "generation_id": generation_id, "reason": reason},
            timeout=5,
        )


def freeze_all_for_run(run_dir: Path, reason: str = "external_stop") -> bool:
    """Best-effort stop fence used by ``praxist stop`` before process discovery."""

    try:
        endpoint = scheduler_endpoint_for_run(run_dir)
        if not endpoint:
            return False
        _rpc(endpoint, {"action": "freeze_all", "reason": reason}, timeout=3)
        return True
    except (OSError, ValueError, json.JSONDecodeError, SchedulerUnavailable, ExperimentRejected):
        return False


def _generation_from_peer(peer_id: str) -> int:
    from .protected_pids import _generation_id_from_peer_id

    parsed = _generation_id_from_peer_id(peer_id)
    if parsed is not None:
        return parsed
    try:
        return int(os.environ.get("GENERATION_ID", "0"))
    except ValueError:
        return 0
