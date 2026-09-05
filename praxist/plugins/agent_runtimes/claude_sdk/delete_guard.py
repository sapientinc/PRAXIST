"""Runtime guard for Claude Code Bash/Python tool calls.

The research loop gives peers Bash access so they can train, evaluate, and
inspect variants.  That access must not let one peer remove the run directory,
shared evidence, or another peer's outputs.  This module provides two layers:

1. a Claude SDK ``PreToolUse`` validator for Bash commands before execution;
2. a generated Python ``sitecustomize`` guard for real filesystem/process
   side effects that escape the pre-tool-use path.
"""

from __future__ import annotations

import atexit
import contextlib
import glob
import hashlib
import json
import os
import re
import shlex
import stat
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from praxist.core.runtime_guard_policy import (
    GUARD_WARNING_ENV_KEY,
    IMMUTABLE_GUARD_ENV_KEYS,
    OPERATOR_ONLY_ENV_KEYS,
    PROTECTED_ROOT_ENV_KEYS,
    PYTHON_GUARD_ENV_KEYS,
    RESOURCE_STATE_DIR_NAMES,
    SHELL_GUARD_ENV_KEYS,
    TRUSTED_PROJECT_ENV_KEYS,
    TRUSTED_PROJECT_EXTRA_ROOTS_ENV,
    TRUSTED_RESOURCE_GUARD_MODULE_SUFFIXES,
    TRUSTED_RESOURCE_GUARD_MODULES,
    split_path_list,
)

_SHELL_SEPARATORS = {
    ";",
    "&&",
    "||",
    "|",
    "(",
    ")",
    "\n",
    "then",
    "do",
    "else",
    "elif",
    "fi",
    "done",
}
_RM_BASENAMES = {"rm"}
_DIRECT_DELETE_BASENAMES = {"rm", "rmdir", "unlink"}
_SHELL_BASENAMES = {"bash", "sh", "dash", "zsh", "ksh"}
_SCRIPT_RUNTIME_BASENAMES = {"perl", "ruby", "node", "nodejs", "php"}
_BROAD_KILL_BASENAMES = {"kill", "pkill", "killall"}
_IDENTITY_GUARD_ENV_KEYS = {
    "PRAXIST_DELETE_GUARD_AGENT",
    "PRAXIST_PEER_WORKSPACE",
    "PEER_ID",
    "PRAXIST_PEER_ID",
}
_LOADER_INJECTION_ENV_KEYS = {
    "LD_PRELOAD",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
}
_WRITE_TOOL_NAMES = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
_READ_ONLY_SYSTEM_BASENAMES = {
    "basename",
    "cat",
    "cut",
    "date",
    "df",
    "dirname",
    "du",
    "echo",
    "env",
    "file",
    "free",
    "find",
    "grep",
    "head",
    "hostname",
    "id",
    "ls",
    "lscpu",
    "nvidia-smi",
    "pgrep",
    "pidof",
    "printf",
    "ps",
    "pwd",
    "readlink",
    "realpath",
    "rg",
    "seq",
    "sleep",
    "stat",
    "tail",
    "test",
    "timeout",
    "tr",
    "true",
    "false",
    "uname",
    "uptime",
    "wc",
    "whoami",
    "which",
    "xargs",
}
_CLASSIFIED_MUTATING_BASENAMES = {
    "awk",
    "cc",
    "chmod",
    "chgrp",
    "chown",
    "clang",
    "clang++",
    "cp",
    "curl",
    "dd",
    "fallocate",
    "g++",
    "gawk",
    "gcc",
    "git",
    "install",
    "ld",
    "ln",
    "make",
    "mawk",
    "mkdir",
    "mv",
    "ninja",
    "patch",
    "rsync",
    "sed",
    "shred",
    "sort",
    "split",
    "tar",
    "tee",
    "touch",
    "truncate",
    "unzip",
    "wget",
    "zip",
}
_PYTHON_DELETE_PATTERNS = (
    "rmtree",
    "shutil.rmtree",
    "from shutil import rmtree",
    "os.remove",
    "os.unlink",
    "os.truncate",
    "os.ftruncate",
    "os.open",
    "_io.open",
    "io.open",
    "posix.open",
    "os.chmod",
    "posix.chmod",
    "os.link",
    "os.symlink",
    "shutil.copyfile",
    "shutil.copy(",
    "shutil.copy2",
    "shutil.copytree",
    ".unlink(",
    ".rmdir(",
    ".chmod(",
    ".write_text(",
    ".write_bytes(",
)
_PYTHON_SHELL_ESCAPE_PATTERNS = (
    "__import__(",
    "getattr(",
    "os.system",
    "subprocess",
    "popen(",
)
_PYTHON_SHELL_DELETE_TERMS = (
    "rm -",
    "rm ",
    "/bin/rm",
    "rmdir",
    "unlink",
    "remove",
    "rmtree",
)
_ENV_OPTIONS_WITH_VALUE = {
    "-u",
    "--unset",
    "-C",
    "--chdir",
    "-S",
    "--split-string",
}
_COMMAND_OPTIONS = {"-p", "-v", "-V"}
_NICE_OPTIONS_WITH_VALUE = {"-n", "--adjustment"}
_GUARD_ENV_VARS = set(SHELL_GUARD_ENV_KEYS)
_DANGEROUS_BUILD_TARGETS = {
    "clean",
    "distclean",
    "clobber",
    "mrproper",
    "delete",
    "remove",
    "purge",
    "wipe",
    "reset",
}
_MAKEFILE_NAMES = ("GNUmakefile", "makefile", "Makefile")
_DANGEROUS_BUILD_RECIPE_MARKERS = (
    "$(shell",
    "$(",
    "${",
    "$$",
    ">",
    ">>",
    "`",
    "/bin/rm",
    " rm ",
    "\trm ",
    "rm -",
    "rmdir",
    "unlink",
    "shred",
    "touch ",
    "\ttouch ",
    "cp ",
    "\tcp ",
    "install ",
    "\tinstall ",
    "mv ",
    "\tmv ",
    "truncate ",
    "\ttruncate ",
    "tee ",
    "\ttee ",
    "dd ",
    "\tdd ",
    "rsync",
    "--delete",
    "python -c",
    "python3 -c",
    "bash -c",
    "sh -c",
    "env -i",
    "shared_findings",
    "frontier",
    "gems",
    "shared_store.db",
)


@dataclass(frozen=True)
class DeleteGuardDecision:
    """Decision returned by the delete guard for one attempted tool call."""

    allowed: bool
    message: str = ""
    severity: str = "allow"
    rule_id: str = ""
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        severity = self.severity
        if not self.allowed and severity == "allow":
            severity = "deny"
        if self.allowed and severity == "deny":
            severity = "warning"
        if severity not in {"allow", "warning", "deny"}:
            raise ValueError(f"unknown delete guard severity: {severity!r}")
        object.__setattr__(self, "severity", severity)

    @property
    def warning(self) -> bool:
        return self.allowed and self.severity == "warning"


def _allow() -> DeleteGuardDecision:
    return DeleteGuardDecision(True)


def _warn(message: str, *, rule_id: str) -> DeleteGuardDecision:
    return DeleteGuardDecision(
        True,
        message,
        severity="warning",
        rule_id=rule_id,
        warnings=(message,),
    )


def _deny(message: str, *, rule_id: str) -> DeleteGuardDecision:
    return DeleteGuardDecision(False, message, severity="deny", rule_id=rule_id)


def _combine_warnings(warnings: list[DeleteGuardDecision]) -> DeleteGuardDecision:
    if not warnings:
        return _allow()
    messages: list[str] = []
    rule_ids: list[str] = []
    for warning in warnings:
        if warning.message:
            messages.append(warning.message)
        if warning.rule_id:
            rule_ids.append(warning.rule_id)
    message = " | ".join(messages) if messages else "Praxist runtime guard warning."
    return DeleteGuardDecision(
        True,
        message,
        severity="warning",
        rule_id=",".join(dict.fromkeys(rule_ids)),
        warnings=tuple(messages),
    )


def _append_guard_warning(
    *,
    env: dict[str, str],
    rule_id: str,
    message: str,
    command: str | None = None,
    tool_name: str | None = None,
) -> None:
    """Best-effort structured warning sink shared by hook and runtime guard."""

    raw_path = env.get(GUARD_WARNING_ENV_KEY)
    if not raw_path:
        return
    try:
        path = Path(raw_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "praxist.runtime_guard.warning.v1",
            "timestamp": time.time(),
            "severity": "warning",
            "rule_id": rule_id,
            "message": message,
            "effect": "allowed",
        }
        if command is not None:
            payload["command"] = command
        if tool_name is not None:
            payload["tool"] = tool_name
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        return


def _ensure_guard_scoped_dir(path: Path, *, run_dir: Path, parent_rel: tuple[str, ...]) -> Path:
    """Create a guard-owned directory without following run-local symlinks."""

    expected_parent = run_dir.joinpath(*parent_rel)
    expected_parent.mkdir(parents=True, exist_ok=True)
    try:
        relative = path.relative_to(run_dir)
    except ValueError as exc:
        raise RuntimeError(f"delete guard path escapes run directory: {path}") from exc

    cursor = run_dir
    for part in relative.parts:
        cursor = cursor / part
        if cursor.exists() or cursor.is_symlink():
            try:
                mode = cursor.lstat().st_mode
            except OSError as exc:
                raise RuntimeError(f"delete guard could not inspect path: {cursor}") from exc
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"delete guard refuses symlink in guard path: {cursor}")
            if cursor == path and not stat.S_ISDIR(mode):
                raise RuntimeError(f"delete guard expected directory path: {cursor}")

    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    parent_resolved = expected_parent.resolve()
    if resolved != parent_resolved and parent_resolved not in resolved.parents:
        raise RuntimeError(f"delete guard path resolved outside expected subtree: {path}")
    return resolved


_MAX_RUNTIME_TMP_BYTES = 64
_REGISTERED_RUNTIME_TMP_LINKS: set[tuple[Path, Path]] = set()


def _cleanup_runtime_tmp_link(link: Path, target: Path) -> None:
    try:
        st = link.lstat()
        if stat.S_ISLNK(st.st_mode) and link.resolve(strict=False) == target.resolve(strict=False):
            link.unlink()
    except (OSError, RuntimeError):
        pass


def _register_runtime_tmp_link(link: Path, target: Path) -> None:
    key = (link, target)
    if key in _REGISTERED_RUNTIME_TMP_LINKS:
        return
    _REGISTERED_RUNTIME_TMP_LINKS.add(key)
    atexit.register(_cleanup_runtime_tmp_link, link, target)


def _ensure_short_runtime_tmp(
    *,
    run_dir: Path,
    safe_agent: str,
    fallback: Path,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Return a short link to run-owned temp storage for Unix-domain sockets."""

    uid = os.geteuid() if hasattr(os, "geteuid") else 0
    identity = f"{run_dir}\0{safe_agent}\0{uid}".encode()
    leaf = f"praxist-tmp-{hashlib.sha256(identity).hexdigest()[:20]}"
    effective_environment = os.environ if environment is None else environment
    bases: list[str] = []
    with contextlib.suppress(OSError, RuntimeError):
        bases.append(tempfile.gettempdir())
    bases.extend(
        raw
        for raw in (
            effective_environment.get("XDG_RUNTIME_DIR"),
            f"/run/user/{uid}",
            "/dev/shm",
            "/tmp",
            "/var/tmp",
            effective_environment.get("HOME"),
        )
        if raw
    )
    seen: set[Path] = set()

    for raw_base in bases:
        try:
            base = Path(raw_base).expanduser().resolve()
            if base in seen:
                continue
            seen.add(base)
            candidate = base / leaf
            if len(os.fsencode(candidate)) > _MAX_RUNTIME_TMP_BYTES:
                continue
            if not base.is_dir() or not os.access(base, os.W_OK | os.X_OK):
                continue
            with contextlib.suppress(FileExistsError):
                candidate.symlink_to(fallback, target_is_directory=True)
            st = candidate.lstat()
            if not stat.S_ISLNK(st.st_mode):
                continue
            if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
                continue
            if candidate.resolve(strict=False) != fallback.resolve(strict=False):
                continue
            _register_runtime_tmp_link(candidate, fallback)
            return candidate
        except (OSError, RuntimeError):
            continue
    return fallback


def prepare_delete_guard_env(
    env: dict[str, str], *, workspace: Path, agent_name: str
) -> dict[str, str]:
    """Return env with per-agent delete guard settings injected.

    The allowed destructive-delete area is a peer/agent-local scratch directory
    under ``<run_dir>/peer_workspaces/<agent>``.  Peers can still write variants,
    results, and findings, but destructive recursive cleanup is confined to
    their own scratch area.
    """

    scoped = dict(env)
    if scoped.get("PRAXIST_DISABLE_DELETE_GUARD", "").strip() == "1":
        return scoped
    for key in OPERATOR_ONLY_ENV_KEYS:
        scoped.pop(key, None)

    run_dir_raw = scoped.get("PRAXIST_RUN_DIR") or scoped.get("AUTO_RESEARCH_RUN_DIR") or ""
    run_dir = Path(run_dir_raw).expanduser() if run_dir_raw else Path(workspace).expanduser()
    run_dir = run_dir.resolve()
    raw_agent = (
        agent_name
        if agent_name in _PI_AGENDA_WRITER_AGENTS
        else scoped.get("PEER_ID") or agent_name
    )
    safe_agent = _safe_name(raw_agent or agent_name or "agent")
    scoped["PRAXIST_DELETE_GUARD_AGENT"] = safe_agent

    peer_workspace = _ensure_guard_scoped_dir(
        run_dir / "peer_workspaces" / safe_agent,
        run_dir=run_dir,
        parent_rel=("peer_workspaces",),
    )
    peer_tmp = _ensure_guard_scoped_dir(
        peer_workspace / "tmp",
        run_dir=run_dir,
        parent_rel=("peer_workspaces", safe_agent),
    )
    guard_dir = _ensure_guard_scoped_dir(
        run_dir / ".runtime_guards" / safe_agent,
        run_dir=run_dir,
        parent_rel=(".runtime_guards",),
    )
    python_site_dir = _ensure_guard_scoped_dir(
        guard_dir / "python_site",
        run_dir=run_dir,
        parent_rel=(".runtime_guards", safe_agent),
    )
    runtime_tmp = _ensure_short_runtime_tmp(
        run_dir=run_dir,
        safe_agent=safe_agent,
        fallback=peer_tmp,
        environment=scoped,
    )

    safe_roots = _join_paths([peer_workspace, peer_tmp, runtime_tmp])
    scoped["PRAXIST_PEER_WORKSPACE"] = str(peer_workspace)
    scoped["PRAXIST_SAFE_DELETE_ROOTS"] = safe_roots
    scoped["PRAXIST_DELETE_GUARD_AGENT"] = safe_agent
    scoped["PRAXIST_DELETE_GUARD_RUN_DIR"] = str(run_dir)
    scoped[GUARD_WARNING_ENV_KEY] = str(guard_dir / "guard_warnings.jsonl")
    scoped["TMPDIR"] = str(runtime_tmp)
    _add_current_praxist_trusted_root(scoped)

    (python_site_dir / "sitecustomize.py").write_text(_sitecustomize_text(), encoding="utf-8")

    scoped["PYTHONPATH"] = (
        f"{python_site_dir}{os.pathsep}{scoped.get('PYTHONPATH') or os.environ.get('PYTHONPATH', '')}"
    )
    return scoped


def _add_current_praxist_trusted_root(scoped: dict[str, str]) -> None:
    """Trust Praxist core code for resource guard writes in guarded peer Python.

    Peers should not need a peer-facing ``PRAXIST_REPO_DIR`` contract for Praxist core
    modules to update run-local resource state through their public APIs.
    """

    current_root = Path(__file__).resolve().parents[4]
    roots = [*split_path_list(scoped.get(TRUSTED_PROJECT_EXTRA_ROOTS_ENV)), current_root]
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve(strict=False)
        if resolved in seen:
            continue
        unique.append(resolved)
        seen.add(resolved)
    scoped[TRUSTED_PROJECT_EXTRA_ROOTS_ENV] = os.pathsep.join(str(root) for root in unique)


def validate_tool_use(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    env: dict[str, str],
    cwd: Path,
) -> DeleteGuardDecision:
    """Validate one Claude Code tool use before execution."""

    if tool_name == "Bash":
        command = str(tool_input.get("command") or "")
        return validate_bash_command(command, env=env, cwd=cwd)
    if tool_name in _WRITE_TOOL_NAMES:
        return _validate_write_tool_use(tool_name, tool_input, env=env, cwd=cwd)
    return DeleteGuardDecision(True)


def validate_bash_command(command: str, *, env: dict[str, str], cwd: Path) -> DeleteGuardDecision:
    """Return whether a Bash command is safe with respect to destructive deletes."""

    return _validate_bash_command(command, env=env, cwd=cwd, depth=0)


def _validate_bash_command(
    command: str, *, env: dict[str, str], cwd: Path, depth: int
) -> DeleteGuardDecision:
    """Implementation with recursion support for nested shell ``-c`` commands."""

    if not command.strip():
        return _allow()
    if depth > 3:
        return _deny(
            "Praxist delete guard blocked deeply nested shell command. Use a script "
            "inside $PRAXIST_PEER_WORKSPACE for cleanup.",
            rule_id="deeply_nested_shell",
        )
    allowed_roots = _allowed_roots(env)
    if not allowed_roots:
        return _allow()

    warnings: list[DeleteGuardDecision] = []
    tokens, tokenization_warning = _shell_tokens_with_warning(command)
    if tokenization_warning:
        warnings.append(
            _warn(
                tokenization_warning,
                rule_id="shell_tokenization_warning",
            )
        )

    def observe(decision: DeleteGuardDecision) -> DeleteGuardDecision | None:
        if not decision.allowed:
            return decision
        if decision.warning:
            warnings.append(decision)
        return None

    guard_mutation_decision = _validate_guard_mutation_patterns(tokens)
    if blocked := observe(guard_mutation_decision):
        return blocked

    process_kill_decision = _validate_broad_process_kill(tokens)
    if blocked := observe(process_kill_decision):
        return blocked

    expanded_command_decision = _validate_expanded_command_words(tokens)
    if blocked := observe(expanded_command_decision):
        return blocked

    guard_strip_decision = _validate_guard_stripping_launches(tokens, env=env)
    if blocked := observe(guard_strip_decision):
        return blocked

    split_decision = _validate_env_split_strings(tokens, env=env, cwd=cwd, depth=depth)
    if blocked := observe(split_decision):
        return blocked

    nested_decision = _validate_nested_shells(tokens, env=env, cwd=cwd, depth=depth)
    if blocked := observe(nested_decision):
        return blocked

    stdin_decision = _validate_interpreter_stdin(command, tokens)
    if blocked := observe(stdin_decision):
        return blocked

    script_decision = _validate_shell_script_operands(tokens, env=env, cwd=cwd, depth=depth)
    if blocked := observe(script_decision):
        return blocked

    python_script_decision = _validate_python_script_operands(tokens, env=env, cwd=cwd)
    if blocked := observe(python_script_decision):
        return blocked

    executable_script_decision = _validate_executable_script_words(
        tokens, env=env, cwd=cwd, depth=depth
    )
    if blocked := observe(executable_script_decision):
        return blocked

    for rm_args in _iter_delete_invocations_anywhere(tokens):
        decision = _validate_rm_args(rm_args, env=env, cwd=cwd, allowed_roots=allowed_roots)
        if blocked := observe(decision):
            return blocked

    overwrite_decision = _validate_overwrite_patterns(
        command, tokens=tokens, env=env, cwd=cwd, allowed_roots=allowed_roots
    )
    if blocked := observe(overwrite_decision):
        return blocked

    source_destructive_decision = _validate_source_destructive_commands(
        tokens, env=env, cwd=cwd, allowed_roots=allowed_roots
    )
    if blocked := observe(source_destructive_decision):
        return blocked

    find_decision = _validate_find_delete(tokens, env=env, cwd=cwd, allowed_roots=allowed_roots)
    if blocked := observe(find_decision):
        return blocked

    python_decision = _validate_python_delete_patterns(
        command, env=env, allowed_roots=allowed_roots
    )
    if blocked := observe(python_decision):
        return blocked

    unclassified_decision = _validate_unclassified_protected_arguments(tokens, env=env, cwd=cwd)
    if blocked := observe(unclassified_decision):
        return blocked

    decision = _combine_warnings(warnings)
    if decision.warning:
        _append_guard_warning(
            env=env,
            rule_id=decision.rule_id,
            message=decision.message,
            command=command,
            tool_name="Bash",
        )
    return decision


def validate_rm_argv(argv: list[str], *, env: dict[str, str], cwd: Path) -> DeleteGuardDecision:
    """Validate argv for the shell-level ``rm`` wrapper."""

    allowed_roots = _allowed_roots(env)
    if not allowed_roots:
        return DeleteGuardDecision(True)
    return _validate_rm_args(argv, env=env, cwd=cwd, allowed_roots=allowed_roots)


def _validate_write_tool_use(
    tool_name: str, tool_input: dict[str, Any], *, env: dict[str, str], cwd: Path
) -> DeleteGuardDecision:
    path_keys = ("file_path", "path", "notebook_path")
    for key in path_keys:
        raw = tool_input.get(key)
        if raw:
            decision = _validate_run_write_path(
                str(raw),
                env=env,
                cwd=cwd,
                op=tool_name,
                allow_mutable_peer_memory=True,
            )
            if not decision.allowed:
                return decision
    return DeleteGuardDecision(True)


def _validate_run_write_path(
    raw: str,
    *,
    env: dict[str, str],
    cwd: Path,
    op: str,
    allow_mutable_peer_memory: bool = False,
) -> DeleteGuardDecision:
    resolved_targets = _resolve_target_candidates(raw, env=env, cwd=cwd)
    if not resolved_targets:
        if op == "mkdir":
            return _warn(
                f"Praxist delete guard allowed ambiguous mkdir target with warning: {raw!r}.",
                rule_id="ambiguous_mkdir_target",
            )
        return DeleteGuardDecision(
            False,
            f"Praxist delete guard blocked ambiguous {op} target: {raw!r}.",
        )
    apparent_memory_target = (
        _apparent_nonglob_target(raw, env=env, cwd=cwd) if allow_mutable_peer_memory else None
    )
    apparent_is_peer_memory = False
    if apparent_memory_target is not None:
        apparent_is_peer_memory = _is_apparent_peer_mutable_memory_run_path(
            apparent_memory_target, env
        )
        if apparent_is_peer_memory and not _is_safe_peer_mutable_memory_direct_write_path(
            apparent_memory_target, env
        ):
            return DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked {op} target through unsafe peer memory file: "
                f"{apparent_memory_target}.",
            )
    elif not allow_mutable_peer_memory:
        apparent_target = _apparent_nonglob_target(raw, env=env, cwd=cwd)
        if apparent_target is not None and _is_apparent_peer_mutable_memory_run_path(
            apparent_target, env
        ):
            return DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked {op} target against peer memory state. "
                "Use the Write tool or Python direct write path for mutable memory files.",
            )

    allowed_roots = _allowed_roots(env)
    protected_roots = _protected_roots(env, cwd=cwd)
    run_dir = _run_dir(env)
    for resolved in resolved_targets:
        if resolved == Path("/dev/null"):
            continue
        if _is_within_any(resolved, allowed_roots):
            if not _is_safe_existing_write_target(resolved):
                return DeleteGuardDecision(
                    False,
                    f"Praxist delete guard blocked {op} target through unsafe file in peer workspace: {resolved}.",
                )
            continue
        if _is_peer_owned_run_write_path(resolved, env):
            if not _is_safe_peer_owned_run_write_target(resolved, env):
                return DeleteGuardDecision(
                    False,
                    f"Praxist delete guard blocked {op} target through unsafe peer-owned file: {resolved}.",
                )
            continue
        if _is_system_agenda_write_path(resolved, env):
            if not _is_safe_existing_write_target(resolved):
                return DeleteGuardDecision(
                    False,
                    f"Praxist delete guard blocked {op} target through unsafe agenda file: {resolved}.",
                )
            continue
        if apparent_is_peer_memory:
            continue
        if run_dir is not None and _is_within_any(resolved, [run_dir]):
            return DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked {op} target inside shared run state but "
                f"outside this peer's owned experiment paths: {resolved}. Use "
                "variants/<peer_id>_*, results/<peer_id>_* or results/gen_<N>/<peer_id>/*, "
                "shared_findings/<peer_id>_*, or $PRAXIST_PEER_WORKSPACE.",
            )
        if _is_within_any(resolved, protected_roots) and resolved.exists():
            return DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked {op} against existing protected project file: {resolved}.",
            )
        if _is_within_any(resolved, protected_roots):
            return DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked {op} target inside protected project state: {resolved}.",
            )
    return DeleteGuardDecision(True)


def _validate_run_write_paths(
    paths: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    op: str,
    allow_mutable_peer_memory: bool = False,
) -> DeleteGuardDecision:
    warnings: list[DeleteGuardDecision] = []
    for raw in paths:
        decision = _validate_run_write_path(
            raw,
            env=env,
            cwd=cwd,
            op=op,
            allow_mutable_peer_memory=allow_mutable_peer_memory,
        )
        if not decision.allowed:
            return decision
        if decision.warning:
            warnings.append(decision)
    return _combine_warnings(warnings)


def _validate_implicit_cwd_write(*, env: dict[str, str], cwd: Path, op: str) -> DeleteGuardDecision:
    resolved = Path(cwd).expanduser().resolve(strict=False)
    allowed_roots = _allowed_roots(env)
    protected_roots = _protected_roots(env, cwd=cwd)
    if _is_within_any(resolved, allowed_roots) or _is_peer_owned_run_write_path(resolved, env):
        return DeleteGuardDecision(True)
    if _is_within_any(resolved, protected_roots):
        return DeleteGuardDecision(
            False,
            f"Praxist delete guard blocked {op} with implicit output in protected cwd: {resolved}. "
            "Use an explicit output path inside $PRAXIST_PEER_WORKSPACE or an owned result directory.",
        )
    return DeleteGuardDecision(True)


def _validate_link_sources(
    paths: list[str], *, env: dict[str, str], cwd: Path
) -> DeleteGuardDecision:
    protected_roots = _protected_roots(env, cwd=cwd)
    for raw in paths:
        resolved_targets = _resolve_target_candidates(raw, env=env, cwd=cwd)
        if not resolved_targets:
            return DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked ambiguous hardlink source: {raw!r}.",
            )
        for resolved in resolved_targets:
            if (
                _is_within_any(resolved, protected_roots)
                and not _is_within_any(resolved, _allowed_roots(env))
                and not _is_peer_owned_run_delete_path(resolved, env)
            ):
                return DeleteGuardDecision(
                    False,
                    f"Praxist delete guard blocked hardlink source from protected state: {resolved}.",
                )
    return DeleteGuardDecision(True)


def _validate_rm_args(
    args: list[str], *, env: dict[str, str], cwd: Path, allowed_roots: list[Path]
) -> DeleteGuardDecision:
    targets = _rm_targets(args)
    if not targets:
        return DeleteGuardDecision(
            False,
            "Praxist delete guard blocked rm without explicit validated targets. "
            "Use $PRAXIST_PEER_WORKSPACE for destructive cleanup.",
        )
    for target in targets:
        resolved_targets = _resolve_target_candidates(target, env=env, cwd=cwd)
        if not resolved_targets:
            return DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked ambiguous rm target: {target!r}. "
                "Use $PRAXIST_PEER_WORKSPACE for destructive cleanup.",
            )
        for resolved in resolved_targets:
            if not _is_within_any(resolved, allowed_roots) and not _is_peer_owned_run_delete_path(
                resolved, env
            ):
                return DeleteGuardDecision(
                    False,
                    "Praxist delete guard blocked rm target outside this agent's scratch "
                    f"workspace/owned experiment paths: {resolved}. "
                    f"Allowed roots: {_join_paths(allowed_roots)}",
                )
    return DeleteGuardDecision(True)


def _validate_find_delete(
    tokens: list[str], *, env: dict[str, str], cwd: Path, allowed_roots: list[Path]
) -> DeleteGuardDecision:
    command_start = True
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _SHELL_SEPARATORS:
            command_start = True
            i += 1
            continue
        if not command_start:
            i += 1
            continue
        j = _skip_command_prefixes(tokens, i)
        if j >= len(tokens) or _basename(tokens[j]) != "find":
            command_start = False
            i += 1
            continue
        segment = _command_segment(tokens, j + 1)
        if _find_execs_destructive_command(segment):
            return DeleteGuardDecision(
                False,
                "Praxist delete guard blocked destructive `find -exec`. Use "
                "`find $PRAXIST_PEER_WORKSPACE ... -delete` for scratch cleanup.",
            )
        if "-delete" not in segment:
            command_start = False
            i += 1
            continue
        roots = _find_start_paths(segment)
        if not roots:
            roots = ["."]
        for root in roots:
            resolved = _resolve_target(root, env=env, cwd=cwd)
            if resolved is None or (
                not _is_within_any(resolved, allowed_roots)
                and not _is_peer_owned_run_delete_path(resolved, env)
            ):
                return DeleteGuardDecision(
                    False,
                    "Praxist delete guard blocked `find -delete` outside this agent's "
                    f"scratch workspace/owned experiment paths: {root!r}.",
                )
        command_start = False
        i += 1
    return DeleteGuardDecision(True)


def _validate_source_destructive_commands(
    tokens: list[str], *, env: dict[str, str], cwd: Path, allowed_roots: list[Path]
) -> DeleteGuardDecision:
    command_start = True
    effective_cwd = cwd
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _SHELL_SEPARATORS:
            command_start = True
            i += 1
            continue
        if not command_start:
            i += 1
            continue
        j = _skip_command_prefixes(tokens, i)
        if j >= len(tokens):
            break
        base = _basename(tokens[j])
        segment = _command_segment(tokens, j + 1)
        if base == "cd":
            target = _cd_target(segment)
            if target is None:
                return DeleteGuardDecision(
                    False,
                    "Praxist delete guard blocked `cd` before destructive command validation "
                    "because the target directory is ambiguous.",
                )
            resolved = _resolve_target(target, env=env, cwd=effective_cwd)
            if resolved is None:
                return DeleteGuardDecision(
                    False,
                    f"Praxist delete guard blocked ambiguous `cd` target: {target!r}.",
                )
            effective_cwd = resolved
        elif base == "mv":
            sources, destinations = _mv_sources_destinations(segment)
            decision = _validate_path_args(
                sources,
                env=env,
                cwd=effective_cwd,
                allowed_roots=allowed_roots,
                op="mv source",
            )
            if not decision.allowed:
                return decision
            decision = _validate_run_write_paths(
                destinations,
                env=env,
                cwd=effective_cwd,
                op="mv destination",
            )
            if not decision.allowed:
                return decision
        elif base == "tar" and "--remove-files" in segment:
            return DeleteGuardDecision(
                False,
                "Praxist delete guard blocked `tar --remove-files`; use $PRAXIST_PEER_WORKSPACE "
                "for source-destructive archive cleanup.",
            )
        elif base == "tar" and any(token.startswith("--checkpoint-action") for token in segment):
            return DeleteGuardDecision(
                False,
                "Praxist delete guard blocked `tar --checkpoint-action` because it can execute "
                "hidden shell commands.",
            )
        elif base == "tar" and _tar_segment_extracts(segment):
            return DeleteGuardDecision(
                False,
                "Praxist delete guard blocked `tar` extraction in peer Bash because archive "
                "members can escape declared output paths. Use Python-safe extraction "
                "inside $PRAXIST_PEER_WORKSPACE if extraction is truly needed.",
            )
        elif base == "zip" and _zip_segment_removes_sources(segment):
            return DeleteGuardDecision(
                False,
                "Praxist delete guard blocked `zip -m`; use $PRAXIST_PEER_WORKSPACE for "
                "source-destructive archive cleanup.",
            )
        elif base == "shred" and _shred_segment_removes_sources(segment):
            decision = _validate_path_args(
                _source_destructive_path_args(segment),
                env=env,
                cwd=effective_cwd,
                allowed_roots=allowed_roots,
                op="shred -u",
            )
            if not decision.allowed:
                return decision
        elif base == "rsync" and _rsync_segment_deletes(segment):
            return DeleteGuardDecision(
                False,
                "Praxist delete guard blocked source/destination destructive rsync options. "
                "Use non-destructive rsync or keep cleanup inside $PRAXIST_PEER_WORKSPACE.",
            )
        elif base == "git" and _git_segment_is_destructive(segment):
            return DeleteGuardDecision(
                False,
                "Praxist delete guard blocked destructive git command during peer execution. "
                "Do not use git clean/reset/rm/forced checkout from peers.",
            )
        elif base == "git":
            git_write_paths = _git_write_paths(segment)
            if git_write_paths:
                decision = _validate_run_write_paths(
                    git_write_paths,
                    env=env,
                    cwd=effective_cwd,
                    op="git write target",
                )
                if not decision.allowed:
                    return decision
        elif base in {"make", "gmake", "ninja"}:
            decision = _validate_build_command_prefixes(tokens[i:j], env=env, cwd=effective_cwd)
            if not decision.allowed:
                return decision
            decision = _validate_make_ninja_segment(
                [base, *segment], env=env, cwd=effective_cwd, allowed_roots=allowed_roots
            )
            if not decision.allowed:
                return decision
        command_start = False
        i += 1
    return DeleteGuardDecision(True)


def _cd_target(segment: list[str]) -> str | None:
    if not segment:
        return None
    i = 0
    while i < len(segment):
        token = segment[i]
        if token == "--":
            i += 1
            break
        if token in {"-L", "-P", "-e"}:
            i += 1
            continue
        if token.startswith("-"):
            return None
        return token
    return segment[i] if i < len(segment) else None


def _validate_overwrite_patterns(
    command: str,
    *,
    tokens: list[str],
    env: dict[str, str],
    cwd: Path,
    allowed_roots: list[Path],
) -> DeleteGuardDecision:
    redirection_decision = _validate_redirections(
        command, env=env, cwd=cwd, allowed_roots=allowed_roots
    )
    if not redirection_decision.allowed:
        return redirection_decision
    command_start = True
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _SHELL_SEPARATORS:
            command_start = True
            i += 1
            continue
        if not command_start:
            i += 1
            continue
        j = _skip_command_prefixes(tokens, i)
        if j >= len(tokens):
            break
        base = _basename(tokens[j])
        segment = _command_segment(tokens, j + 1)
        if base == "truncate":
            decision = _validate_run_write_paths(
                _source_destructive_path_args(segment),
                env=env,
                cwd=cwd,
                op="truncate",
            )
            if not decision.allowed:
                return decision
        elif base == "dd":
            for arg in segment:
                if arg.startswith("of="):
                    decision = _validate_run_write_paths(
                        [arg.split("=", 1)[1]],
                        env=env,
                        cwd=cwd,
                        op="dd of=",
                    )
                    if not decision.allowed:
                        return decision
        elif base in {"cp", "install"}:
            if base == "cp" and _cp_segment_hardlinks(segment):
                source_decision = _validate_link_sources(
                    _cp_install_sources(segment), env=env, cwd=cwd
                )
                if not source_decision.allowed:
                    return source_decision
            destinations = _cp_install_destinations(segment)
            decision = _validate_run_write_paths(destinations, env=env, cwd=cwd, op=base)
            if not decision.allowed:
                return decision
        elif base == "tee":
            decision = _validate_run_write_paths(
                _source_destructive_path_args(segment),
                env=env,
                cwd=cwd,
                op="tee",
            )
            if not decision.allowed:
                return decision
        elif base == "sed" and any(token == "-i" or token.startswith("-i") for token in segment):
            decision = _validate_run_write_paths(
                _sed_inplace_files(segment),
                env=env,
                cwd=cwd,
                op="sed -i",
            )
            if not decision.allowed:
                return decision
        elif base == "sort":
            output = _option_value(segment, "-o", "--output")
            if output:
                decision = _validate_run_write_paths(
                    [output],
                    env=env,
                    cwd=cwd,
                    op="sort -o",
                )
                if not decision.allowed:
                    return decision
            temp_dir = _option_value(segment, "-T", "--temporary-directory")
            if temp_dir:
                decision = _validate_run_write_paths(
                    [temp_dir],
                    env=env,
                    cwd=cwd,
                    op="sort temporary directory",
                )
                if not decision.allowed:
                    return decision
        elif base == "tar" and _tar_segment_extracts(segment):
            return DeleteGuardDecision(
                False,
                "Praxist delete guard blocked tar extraction in peer Bash because archive "
                "members can write outside declared directories.",
            )
        elif base == "tar":
            outputs = _tar_archive_outputs(segment)
            if outputs:
                decision = _validate_run_write_paths(outputs, env=env, cwd=cwd, op="tar archive")
                if not decision.allowed:
                    return decision
            elif _tar_creates_archive(segment):
                decision = _validate_implicit_cwd_write(env=env, cwd=cwd, op="tar archive")
                if not decision.allowed:
                    return decision
        elif base == "zip":
            outputs = _zip_archive_outputs(segment)
            if outputs:
                decision = _validate_run_write_paths(outputs, env=env, cwd=cwd, op="zip archive")
                if not decision.allowed:
                    return decision
            else:
                decision = _validate_implicit_cwd_write(env=env, cwd=cwd, op="zip archive")
                if not decision.allowed:
                    return decision
        elif base == "touch":
            decision = _validate_run_write_paths(
                _source_destructive_path_args(segment),
                env=env,
                cwd=cwd,
                op="touch",
            )
            if not decision.allowed:
                return decision
        elif base == "mkdir":
            decision = _validate_run_write_paths(
                _source_destructive_path_args(segment),
                env=env,
                cwd=cwd,
                op="mkdir",
            )
            if not decision.allowed or decision.warning:
                return decision
        elif base == "fallocate":
            decision = _validate_run_write_paths(
                _source_destructive_path_args(segment),
                env=env,
                cwd=cwd,
                op="fallocate",
            )
            if not decision.allowed:
                return decision
        elif base in {"gcc", "g++", "cc", "clang", "clang++", "ld"}:
            output = _option_value(segment, "-o", "--output")
            if output:
                decision = _validate_run_write_paths(
                    [output],
                    env=env,
                    cwd=cwd,
                    op=f"{base} -o",
                )
                if not decision.allowed:
                    return decision
        elif base in {"curl", "wget"}:
            output = _option_value(segment, "-o" if base == "curl" else "-O", "--output-document")
            if base == "curl" and not output:
                output = _option_value(segment, "-o", "--output")
            if output:
                decision = _validate_run_write_paths(
                    [output],
                    env=env,
                    cwd=cwd,
                    op=f"{base} output",
                )
                if not decision.allowed:
                    return decision
            output_dirs = _downloader_output_dirs(segment, base=base)
            if output_dirs:
                decision = _validate_run_write_paths(
                    output_dirs,
                    env=env,
                    cwd=cwd,
                    op=f"{base} output directory",
                )
                if not decision.allowed:
                    return decision
            elif _downloader_writes_to_cwd(segment, base=base):
                decision = _validate_implicit_cwd_write(env=env, cwd=cwd, op=f"{base} download")
                if not decision.allowed:
                    return decision
        elif base in {"unzip", "patch", "split"}:
            if base == "unzip":
                output_dir = _option_value(segment, "-d", "-d")
                if output_dir:
                    decision = _validate_run_write_paths(
                        [output_dir],
                        env=env,
                        cwd=cwd,
                        op="unzip output directory",
                    )
                    if not decision.allowed:
                        return decision
                else:
                    decision = _validate_implicit_cwd_write(env=env, cwd=cwd, op="unzip")
                    if not decision.allowed:
                        return decision
            elif base == "patch":
                decision = _validate_implicit_cwd_write(env=env, cwd=cwd, op="patch")
                if not decision.allowed:
                    return decision
            elif base == "split":
                output_prefix = _split_output_prefix(segment)
                if output_prefix:
                    decision = _validate_run_write_paths(
                        [output_prefix],
                        env=env,
                        cwd=cwd,
                        op="split output prefix",
                    )
                    if not decision.allowed:
                        return decision
                else:
                    decision = _validate_implicit_cwd_write(env=env, cwd=cwd, op="split")
                    if not decision.allowed:
                        return decision
        elif base in {"ln", "awk", "gawk", "mawk"}:
            if base == "ln":
                source_decision = _validate_link_sources(_ln_sources(segment), env=env, cwd=cwd)
                if not source_decision.allowed:
                    return source_decision
                destinations = _ln_destinations(segment)
            else:
                destinations = _awk_redirection_targets(segment)
                if (
                    not destinations
                    and ">" in " ".join(segment)
                    and _mentions_protected_target(" ".join(segment), env)
                ):
                    return DeleteGuardDecision(
                        False,
                        "Praxist delete guard blocked awk redirection toward protected state.",
                    )
            decision = _validate_run_write_paths(destinations, env=env, cwd=cwd, op=base)
            if not decision.allowed:
                return decision
        elif base in {"chmod", "chown", "chgrp"}:
            decision = _validate_run_write_paths(
                _metadata_target_args(segment),
                env=env,
                cwd=cwd,
                op=base,
            )
            if not decision.allowed:
                return decision
        elif base == "rsync":
            destination = _rsync_destination(segment)
            if destination:
                decision = _validate_run_write_paths(
                    [destination], env=env, cwd=cwd, op="rsync destination"
                )
                if not decision.allowed:
                    return decision
            secondary_dirs = _rsync_secondary_write_dirs(segment)
            if secondary_dirs:
                decision = _validate_run_write_paths(
                    secondary_dirs,
                    env=env,
                    cwd=cwd,
                    op="rsync secondary write directory",
                )
                if not decision.allowed:
                    return decision
        command_start = False
        i += 1
    return DeleteGuardDecision(True)


def _validate_redirections(
    command: str, *, env: dict[str, str], cwd: Path, allowed_roots: list[Path]
) -> DeleteGuardDecision:
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError as exc:
        return _warn(
            "Praxist runtime guard allowed shell redirection check with warning: "
            f"command was not fully parseable ({exc}); destructive raw patterns "
            "remain guarded.",
            rule_id="shell_redirection_tokenization_warning",
        )
    redirection_tokens = {">", ">>", ">|", "&>", "&>>"}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        is_fd_redirect = (
            token.isdigit() and i + 1 < len(tokens) and tokens[i + 1] in redirection_tokens
        )
        if token in redirection_tokens or is_fd_redirect:
            target_index = i + 2 if is_fd_redirect else i + 1
            if target_index >= len(tokens):
                return DeleteGuardDecision(
                    False, "Praxist delete guard blocked redirection without target."
                )
            target = tokens[target_index]
            if target in {"&1", "&2", "-"}:
                i += 1
                continue
            decision = _validate_run_write_paths(
                [target],
                env=env,
                cwd=cwd,
                op="shell redirection",
                allow_mutable_peer_memory=True,
            )
            if not decision.allowed or decision.warning:
                return decision
            i = target_index + 1
            continue
        i += 1
    return DeleteGuardDecision(True)


def _validate_path_args(
    args: list[str], *, env: dict[str, str], cwd: Path, allowed_roots: list[Path], op: str
) -> DeleteGuardDecision:
    if not args:
        return DeleteGuardDecision(
            False,
            f"Praxist delete guard blocked {op} without explicit validated paths.",
        )
    for target in args:
        resolved_targets = _resolve_target_candidates(target, env=env, cwd=cwd)
        if not resolved_targets:
            return DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked ambiguous {op} path: {target!r}.",
            )
        for resolved in resolved_targets:
            if not _is_within_any(resolved, allowed_roots) and not _is_peer_owned_run_delete_path(
                resolved, env
            ):
                return DeleteGuardDecision(
                    False,
                    f"Praxist delete guard blocked {op} path outside this agent's scratch "
                    f"workspace/owned experiment paths: {resolved}.",
                )
    return DeleteGuardDecision(True)


def _validate_build_command_prefixes(
    prefix_tokens: list[str], *, env: dict[str, str], cwd: Path
) -> DeleteGuardDecision:
    for token in prefix_tokens:
        if "=" not in token or token.startswith("-"):
            continue
        key, value = token.split("=", 1)
        if not key.isidentifier():
            continue
        expanded = _expand_target(value, env)
        if _looks_pathlike_or_protected(expanded) or _mentions_protected_target(expanded, env):
            return DeleteGuardDecision(
                False,
                "Praxist delete guard blocked make/ninja launched with pathlike or protected "
                "environment assignment.",
            )
        if key in _GUARD_ENV_VARS:
            return DeleteGuardDecision(
                False,
                "Praxist delete guard blocked make/ninja launched with delete-guard env override.",
            )
    return DeleteGuardDecision(True)


def _validate_make_ninja_segment(
    segment: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    allowed_roots: list[Path],
) -> DeleteGuardDecision:
    base = Path(segment[0]).name if segment else ""
    argv = [str(arg) for arg in segment[1:]]
    lowered = [arg.lower() for arg in argv if arg and not arg.startswith("-")]
    if any(target in _DANGEROUS_BUILD_TARGETS for target in lowered):
        return DeleteGuardDecision(
            False,
            "Praxist delete guard blocked make/ninja clean or destructive target.",
        )
    build_paths: list[str] = []
    explicit_build_files: list[str] = []
    has_explicit_build_dir = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in {"-C", "--directory"} and i + 1 < len(argv):
            build_paths.append(argv[i + 1])
            has_explicit_build_dir = True
            i += 2
            continue
        if arg.startswith("--directory="):
            build_paths.append(arg.split("=", 1)[1])
            has_explicit_build_dir = True
        if arg.startswith("-C") and len(arg) > 2:
            build_paths.append(arg[2:])
            has_explicit_build_dir = True
        if base in {"make", "gmake"}:
            if arg in {"-f", "--file", "--makefile"} and i + 1 < len(argv):
                explicit_build_files.append(argv[i + 1])
                i += 2
                continue
            if arg.startswith(("-f", "--file=", "--makefile=")):
                value = arg[2:] if arg.startswith("-f") and len(arg) > 2 else arg.split("=", 1)[-1]
                explicit_build_files.append(value)
        elif base == "ninja":
            if arg == "-f" and i + 1 < len(argv):
                explicit_build_files.append(argv[i + 1])
                i += 2
                continue
            if arg.startswith("-f") and len(arg) > 2:
                explicit_build_files.append(arg[2:])
        i += 1
    resolved_build_paths = [cwd.resolve(strict=False)]
    if has_explicit_build_dir:
        for raw in build_paths:
            next_paths: dict[str, Path] = {}
            for base_path in resolved_build_paths:
                candidates = _resolve_target_candidates(raw, env=env, cwd=base_path)
                for candidate in candidates or []:
                    resolved = candidate.resolve(strict=False)
                    next_paths[str(resolved)] = resolved
            if len(next_paths) != 1:
                return DeleteGuardDecision(
                    False,
                    f"Praxist delete guard blocked ambiguous make/ninja build path: {raw!r}.",
                )
            resolved_build_paths = list(next_paths.values())
    for resolved in resolved_build_paths:
        if not (
            _is_within_any(resolved, allowed_roots)
            or _is_peer_owned_run_write_path(resolved, env)
            or _is_trusted_project_script(resolved, env, cwd=cwd)
        ):
            return DeleteGuardDecision(
                False,
                "Praxist delete guard blocked make/ninja outside this peer's workspace "
                f"or trusted task project paths: {resolved}.",
            )
    build_files = _candidate_build_files(
        base=base,
        explicit_build_files=explicit_build_files,
        build_paths=resolved_build_paths,
        env=env,
        cwd=cwd,
    )
    if not build_files:
        return DeleteGuardDecision(
            False,
            "Praxist delete guard blocked make/ninja without an inspectable build file.",
        )
    for path in build_files:
        resolved = path.resolve(strict=False)
        if not (
            _is_within_any(resolved, allowed_roots)
            or _is_peer_owned_run_write_path(resolved, env)
            or _is_trusted_project_script(resolved, env, cwd=path.parent)
        ):
            return DeleteGuardDecision(
                False,
                "Praxist delete guard blocked make/ninja build file outside this "
                f"peer's workspace or trusted task project paths: {resolved}.",
            )
    for path in build_files:
        decision = _validate_build_file_content(path, env=env)
        if not decision.allowed:
            return decision
    return DeleteGuardDecision(True)


def _candidate_build_files(
    *,
    base: str,
    explicit_build_files: list[str],
    build_paths: list[Path],
    env: dict[str, str],
    cwd: Path,
) -> list[Path]:
    files: list[Path] = []
    for raw in explicit_build_files:
        files.extend(
            _explicit_build_file_candidates(
                raw,
                build_paths=build_paths,
                env=env,
                cwd=cwd,
            )
        )
    if explicit_build_files:
        unique: dict[str, Path] = {}
        for path in files:
            unique[str(path.resolve(strict=False))] = path
        return list(unique.values())
    if base in {"make", "gmake"}:
        for build_path in build_paths:
            for name in _MAKEFILE_NAMES:
                path = build_path / name
                if path.exists():
                    files.append(path)
    elif base == "ninja":
        for build_path in build_paths:
            path = build_path / "build.ninja"
            if path.exists():
                files.append(path)
    unique: dict[str, Path] = {}
    for path in files:
        unique[str(path.resolve(strict=False))] = path
    return list(unique.values())


def _explicit_build_file_candidates(
    raw: str,
    *,
    build_paths: list[Path],
    env: dict[str, str],
    cwd: Path,
) -> list[Path]:
    expanded = _expand_target(raw, env)
    if "$" in expanded:
        return []
    try:
        explicit_path = Path(expanded).expanduser()
    except (TypeError, ValueError):
        return []
    bases = [cwd] if explicit_path.is_absolute() else (build_paths or [cwd])
    candidates: list[Path] = []
    for base in bases:
        resolved = _resolve_target_candidates(raw, env=env, cwd=base)
        if resolved:
            candidates.extend(resolved)
    return candidates


_MAX_BUILD_INCLUDE_DEPTH = 8


def _build_include_targets(line: str, *, is_ninja: bool) -> tuple[str, list[str], bool] | None:
    """Return build include directive metadata for a single non-recipe line."""

    stripped = line.strip()
    if not stripped:
        return None
    if is_ninja:
        match = re.match(r"^(include|subninja)\s+(.+)$", stripped, flags=re.IGNORECASE)
        if not match:
            return None
        directive = match.group(1).lower()
        target = match.group(2).strip()
        return directive, [target], False

    match = re.match(r"^(-?include|sinclude)\s+(.+)$", stripped, flags=re.IGNORECASE)
    if not match:
        return None
    directive = match.group(1).lower()
    optional = directive in {"-include", "sinclude"}
    raw_targets = match.group(2).strip()
    if not raw_targets:
        return directive, [], optional
    try:
        targets = shlex.split(raw_targets, comments=False, posix=True)
    except ValueError:
        return directive, [], optional
    return directive, targets, optional


def _validate_build_include_target(
    raw: str,
    *,
    source_path: Path,
    env: dict[str, str],
) -> tuple[DeleteGuardDecision, Path | None]:
    if not raw or any(token in raw for token in ("$", "`", "*", "?", "[", '"', "'")):
        return (
            DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked dynamic make/ninja include in {source_path}: {raw!r}.",
            ),
            None,
        )
    candidates = _resolve_target_candidates(raw, env=env, cwd=source_path.parent)
    if not candidates or len(candidates) != 1:
        return (
            DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked ambiguous make/ninja include in {source_path}: {raw!r}.",
            ),
            None,
        )
    resolved = candidates[0]
    allowed_roots = _allowed_roots(env)
    if not (
        _is_within_any(resolved, allowed_roots)
        or _is_peer_owned_run_write_path(resolved, env)
        or _is_trusted_project_script(resolved, env, cwd=source_path.parent)
    ):
        return (
            DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked make/ninja include outside trusted paths: {resolved}.",
            ),
            None,
        )
    return DeleteGuardDecision(True), resolved


def _validate_build_file_content(
    path: Path,
    *,
    env: dict[str, str],
    depth: int = 0,
    seen: set[str] | None = None,
) -> DeleteGuardDecision:
    resolved_path = path.resolve(strict=False)
    key = str(resolved_path)
    seen = set() if seen is None else seen
    if key in seen:
        return DeleteGuardDecision(True)
    if depth > _MAX_BUILD_INCLUDE_DEPTH:
        return DeleteGuardDecision(
            False,
            f"Praxist delete guard blocked make/ninja include recursion deeper than {_MAX_BUILD_INCLUDE_DEPTH}: {path}.",
        )
    seen.add(key)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return DeleteGuardDecision(
            False,
            f"Praxist delete guard blocked make/ninja because build file is unreadable: {path}.",
        )
    lower = f"\n{text.lower()}\n"
    if _mentions_protected_target(text, env) or any(
        marker in lower for marker in _DANGEROUS_BUILD_RECIPE_MARKERS
    ):
        return DeleteGuardDecision(
            False,
            f"Praxist delete guard blocked make/ninja recipe with protected or destructive operations: {path}.",
        )
    is_ninja = path.name == "build.ninja" or path.suffix == ".ninja"
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^SHELL\s*[:+?]?=", stripped):
            return DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked makefile shell override: {path}.",
            )
        if not line.startswith("\t"):
            include = _build_include_targets(line, is_ninja=is_ninja)
            if include is not None:
                _directive, targets, optional = include
                for target in targets:
                    decision, include_path = _validate_build_include_target(
                        target,
                        source_path=path,
                        env=env,
                    )
                    if not decision.allowed:
                        return decision
                    if include_path is None or not include_path.exists():
                        if optional:
                            continue
                        return DeleteGuardDecision(
                            False,
                            f"Praxist delete guard blocked missing make/ninja include from {path}: {target!r}.",
                        )
                    include_decision = _validate_build_file_content(
                        include_path,
                        env=env,
                        depth=depth + 1,
                        seen=seen,
                    )
                    if not include_decision.allowed:
                        return include_decision
        recipe = ""
        if line.startswith("\t"):
            recipe = line.lstrip()
            while recipe.startswith(("@", "+", "-")):
                recipe = recipe[1:].lstrip()
        elif stripped.lower().startswith("command") and "=" in stripped:
            recipe = stripped.split("=", 1)[1].strip()
        if recipe:
            decision = _validate_bash_command(recipe, env=env, cwd=path.parent, depth=1)
            if not decision.allowed:
                return DeleteGuardDecision(
                    False,
                    f"Praxist delete guard blocked make/ninja unsafe recipe in {path}: {decision.message}",
                )
    return DeleteGuardDecision(True)


def _source_destructive_path_args(segment: list[str]) -> list[str]:
    args: list[str] = []
    i = 0
    while i < len(segment):
        token = segment[i]
        if token == "--":
            args.extend(segment[i + 1 :])
            break
        if token in {"-t", "--target-directory"} and i + 1 < len(segment):
            args.append(segment[i + 1])
            i += 2
            continue
        if token.startswith("-t") and not token.startswith("--") and len(token) > 2:
            args.append(token[2:])
            i += 1
            continue
        if token.startswith("--target-directory="):
            args.append(token.split("=", 1)[1])
            i += 1
            continue
        if token.startswith("-"):
            i += 1
            continue
        args.append(token)
        i += 1
    return args


def _mv_sources_destinations(segment: list[str]) -> tuple[list[str], list[str]]:
    paths = _source_destructive_path_args(segment)
    target_dir: str | None = None
    i = 0
    while i < len(segment):
        token = segment[i]
        if token in {"-t", "--target-directory"} and i + 1 < len(segment):
            target_dir = segment[i + 1]
            i += 2
            continue
        if token.startswith("-t") and not token.startswith("--") and len(token) > 2:
            target_dir = token[2:]
            i += 1
            continue
        if token.startswith("--target-directory="):
            target_dir = token.split("=", 1)[1]
            i += 1
            continue
        i += 1
    if target_dir:
        sources = [path for path in paths if path != target_dir]
        return sources, [target_dir]
    if len(paths) < 2:
        return paths, []
    return paths[:-1], [paths[-1]]


def _zip_segment_removes_sources(segment: list[str]) -> bool:
    return any(token == "-m" or (token.startswith("-") and "m" in token[1:]) for token in segment)


def _shred_segment_removes_sources(segment: list[str]) -> bool:
    return any(
        token == "-u" or token == "--remove" or token.startswith("--remove=") for token in segment
    )


def _rsync_segment_deletes(segment: list[str]) -> bool:
    return any(
        token in {"--delete", "--remove-source-files"} or token.startswith("--delete-")
        for token in segment
    )


def _git_segment_is_destructive(segment: list[str]) -> bool:
    if not segment:
        return False
    subcommand = segment[0]
    if subcommand in {"clean", "rm"}:
        return True
    if subcommand == "reset" and "--hard" in segment:
        return True
    if subcommand in {"checkout", "restore"}:
        return any(token in {"-f", "--force"} for token in segment)
    return False


def _git_segment_is_read_only(segment: list[str]) -> bool:
    parsed = _git_subcommand_and_args(segment)
    if parsed is None:
        return False
    subcommand, rest = parsed
    if any(token in {"-o", "--output"} or token.startswith("--output=") for token in rest):
        return False
    return subcommand in {
        "status",
        "rev-parse",
        "diff",
        "log",
        "show",
        "ls-files",
        "describe",
        "merge-base",
    }


def _git_subcommand_and_args(segment: list[str]) -> tuple[str, list[str]] | None:
    i = 0
    options_with_values = {
        "-C",
        "-c",
        "--config-env",
        "--exec-path",
        "--git-dir",
        "--namespace",
        "--work-tree",
    }
    while i < len(segment):
        token = segment[i]
        if token == "--":
            i += 1
            break
        if token in options_with_values:
            i += 2
            continue
        if any(
            token.startswith(f"{option}=")
            for option in options_with_values
            if option.startswith("--")
        ):
            i += 1
            continue
        if token.startswith("-"):
            i += 1
            continue
        return token, segment[i + 1 :]
    if i < len(segment):
        return segment[i], segment[i + 1 :]
    return None


def _git_write_paths(segment: list[str]) -> list[str]:
    paths: list[str] = []
    args = list(segment)
    while args and args[0] == "-C" and len(args) >= 2:
        paths.append(args[1])
        args = args[2:]
    if not args:
        return paths
    subcommand = args[0]
    rest = args[1:]
    if subcommand == "init":
        positional = [arg for arg in rest if not arg.startswith("-")]
        if positional:
            paths.append(positional[-1])
    elif subcommand == "clone":
        positional = [arg for arg in rest if not arg.startswith("-")]
        if len(positional) >= 2:
            paths.append(positional[-1])
    elif subcommand in {"archive", "diff"}:
        output = _option_value(rest, "-o", "--output")
        if output:
            paths.append(output)
    elif subcommand == "worktree" and rest and rest[0] == "add":
        positional = [arg for arg in rest[1:] if not arg.startswith("-")]
        if positional:
            paths.append(positional[0])
    return paths


def _last_path_arg(segment: list[str]) -> str | None:
    paths = _source_destructive_path_args(segment)
    return paths[-1] if paths else None


def _cp_install_destinations(segment: list[str]) -> list[str]:
    paths = _source_destructive_path_args(segment)
    target_dir: str | None = None
    i = 0
    while i < len(segment):
        token = segment[i]
        if token in {"-t", "--target-directory"} and i + 1 < len(segment):
            target_dir = segment[i + 1]
            i += 2
            continue
        if token.startswith("-") and not token.startswith("--") and "t" in token[1:]:
            suffix = token[token.rfind("t") + 1 :]
            if suffix:
                target_dir = suffix
                i += 1
                continue
            if i + 1 < len(segment):
                target_dir = segment[i + 1]
                i += 2
                continue
        if token.startswith("--target-directory="):
            target_dir = token.split("=", 1)[1]
        i += 1
    if target_dir:
        return [target_dir]
    return [paths[-1]] if paths else []


def _cp_install_sources(segment: list[str]) -> list[str]:
    paths = _source_destructive_path_args(segment)
    target_dir = any(
        token in {"-t", "--target-directory"}
        or token.startswith("--target-directory=")
        or (token.startswith("-") and not token.startswith("--") and "t" in token[1:])
        for token in segment
    )
    if target_dir:
        return paths[:-1] if len(paths) > 1 else paths
    return paths[:-1] if len(paths) >= 2 else []


def _cp_segment_hardlinks(segment: list[str]) -> bool:
    return any(
        token in {"-l", "--link"}
        or (token.startswith("-") and not token.startswith("--") and "l" in token[1:])
        for token in segment
    )


def _ln_destinations(segment: list[str]) -> list[str]:
    paths = _source_destructive_path_args(segment)
    if len(paths) >= 2:
        return [paths[-1]]
    if len(paths) == 1:
        return [Path(paths[0]).name]
    return []


def _ln_sources(segment: list[str]) -> list[str]:
    if any(token == "-s" or (token.startswith("-") and "s" in token[1:]) for token in segment):
        return []
    paths = _source_destructive_path_args(segment)
    return paths[:-1] if len(paths) >= 2 else []


def _awk_redirection_targets(segment: list[str]) -> list[str]:
    targets: list[str] = []
    for i, token in enumerate(segment):
        if token in {">", ">>"} and i + 1 < len(segment):
            targets.append(segment[i + 1])
        elif token.startswith(">") and len(token) > 1:
            targets.append(token[1:])
    return targets


def _metadata_target_args(segment: list[str]) -> list[str]:
    args: list[str] = []
    skip_next = False
    for token in segment:
        if skip_next:
            skip_next = False
            continue
        if token == "--":
            continue
        if token in {"--reference", "-c"}:
            skip_next = True
            continue
        if token.startswith("--reference="):
            continue
        if token.startswith("-"):
            continue
        # First non-option is mode/owner/group; later args are targets.
        if args or token.startswith("/") or token.startswith("$") or "/" in token:
            args.append(token)
        else:
            args.append("__MODE_PLACEHOLDER__")
    return [arg for arg in args if arg != "__MODE_PLACEHOLDER__"]


def _rsync_destination(segment: list[str]) -> str | None:
    paths = _source_destructive_path_args(segment)
    return paths[-1] if len(paths) >= 2 else None


def _rsync_secondary_write_dirs(segment: list[str]) -> list[str]:
    dirs: list[str] = []
    i = 0
    while i < len(segment):
        token = segment[i]
        if token in {"--backup-dir", "--partial-dir"} and i + 1 < len(segment):
            dirs.append(segment[i + 1])
            i += 2
            continue
        if token.startswith("--backup-dir=") or token.startswith("--partial-dir="):
            dirs.append(token.split("=", 1)[1])
        i += 1
    return dirs


def _option_value(segment: list[str], short: str, long: str) -> str | None:
    i = 0
    while i < len(segment):
        token = segment[i]
        if token in {short, long} and i + 1 < len(segment):
            return segment[i + 1]
        if token.startswith(f"{long}="):
            return token.split("=", 1)[1]
        if short and token.startswith(short) and len(token) > len(short):
            return token[len(short) :]
        i += 1
    return None


def _downloader_output_dirs(segment: list[str], *, base: str) -> list[str]:
    dirs: list[str] = []
    i = 0
    while i < len(segment):
        token = segment[i]
        if base == "curl":
            if token == "--output-dir" and i + 1 < len(segment):
                dirs.append(segment[i + 1])
                i += 2
                continue
            if token.startswith("--output-dir="):
                dirs.append(token.split("=", 1)[1])
        elif base == "wget":
            if token in {"-P", "--directory-prefix"} and i + 1 < len(segment):
                dirs.append(segment[i + 1])
                i += 2
                continue
            if token.startswith("-P") and len(token) > 2:
                dirs.append(token[2:])
            if token.startswith("--directory-prefix="):
                dirs.append(token.split("=", 1)[1])
        i += 1
    return dirs


def _downloader_writes_to_cwd(segment: list[str], *, base: str) -> bool:
    if base == "curl":
        return any(
            token == "-O"
            or (token.startswith("-") and not token.startswith("--") and "O" in token[1:])
            for token in segment
        )
    if base == "wget":
        return not any(
            token in {"-O", "--output-document", "-P", "--directory-prefix"}
            or token.startswith("--output-document=")
            or token.startswith("-O")
            or token.startswith("-P")
            or token.startswith("--directory-prefix=")
            for token in segment
        )
    return False


def _tar_archive_outputs(segment: list[str]) -> list[str]:
    outputs: list[str] = []
    i = 0
    while i < len(segment):
        token = segment[i]
        if token in {"-f", "--file"} and i + 1 < len(segment):
            outputs.append(segment[i + 1])
            i += 2
            continue
        if token.startswith("--file="):
            outputs.append(token.split("=", 1)[1])
        elif token.startswith("-") and not token.startswith("--") and "f" in token[1:]:
            suffix = token[token.rfind("f") + 1 :]
            if suffix:
                outputs.append(suffix)
            elif i + 1 < len(segment):
                outputs.append(segment[i + 1])
                i += 2
                continue
        i += 1
    return outputs


def _tar_creates_archive(segment: list[str]) -> bool:
    return any(
        token in {"-c", "--create"}
        or (token.startswith("-") and not token.startswith("--") and "c" in token[1:])
        for token in segment
    )


def _zip_archive_outputs(segment: list[str]) -> list[str]:
    outputs: list[str] = []
    for token in segment:
        if token == "--":
            continue
        if token.startswith("-"):
            continue
        outputs.append(token)
        break
    return outputs


def _split_output_prefix(segment: list[str]) -> str | None:
    positional = [token for token in segment if token != "--" and not token.startswith("-")]
    if len(positional) >= 2:
        return positional[-1]
    return None


def _tar_segment_extracts(segment: list[str]) -> bool:
    return any(
        token in {"-x", "--extract", "--get"}
        or (token.startswith("-") and not token.startswith("--") and "x" in token[1:])
        for token in segment
    )


def _sed_inplace_files(segment: list[str]) -> list[str]:
    files: list[str] = []
    skip_next = False
    script_seen = False
    for token in segment:
        if skip_next:
            skip_next = False
            continue
        if token == "--":
            script_seen = True
            continue
        if token in {"-e", "-f"}:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        if not script_seen:
            script_seen = True
            continue
        files.append(token)
    return files


def _find_start_paths(segment: list[str]) -> list[str]:
    roots: list[str] = []
    i = 0
    while i < len(segment):
        token = segment[i]
        if token == "--":
            i += 1
            continue
        if _is_find_expression_token(token):
            break
        if token.startswith("-"):
            break
        roots.append(token)
        i += 1
    return roots


def _is_find_expression_token(token: str) -> bool:
    if token in {"!", "(", ")"}:
        return True
    if token.startswith("-"):
        return True
    return token in {"-o", "-a", ","}


def _validate_nested_shells(
    tokens: list[str], *, env: dict[str, str], cwd: Path, depth: int
) -> DeleteGuardDecision:
    i = 0
    while i < len(tokens):
        j = _skip_command_prefixes(tokens, i)
        if j < len(tokens) and _basename(tokens[j]) in _SHELL_BASENAMES:
            segment = _command_segment(tokens, j + 1)
            nested = _shell_c_argument(segment)
            if nested is not None:
                decision = _validate_bash_command(nested, env=env, cwd=cwd, depth=depth + 1)
                if not decision.allowed:
                    return decision
        i += 1
    return DeleteGuardDecision(True)


def _validate_env_split_strings(
    tokens: list[str], *, env: dict[str, str], cwd: Path, depth: int
) -> DeleteGuardDecision:
    for i, token in enumerate(tokens):
        if _command_token_basename(token) != "env":
            continue
        j = i + 1
        while j < len(tokens):
            arg = tokens[j]
            if arg in ("-S", "--split-string"):
                payload = tokens[j + 1] if j + 1 < len(tokens) else ""
                return _validate_bash_command(payload, env=env, cwd=cwd, depth=depth + 1)
            if arg.startswith("--split-string="):
                return _validate_bash_command(
                    arg.split("=", 1)[1], env=env, cwd=cwd, depth=depth + 1
                )
            if arg in _SHELL_SEPARATORS:
                break
            j += 1
    return DeleteGuardDecision(True)


def _validate_broad_process_kill(tokens: list[str]) -> DeleteGuardDecision:
    command_start = True
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _SHELL_SEPARATORS:
            command_start = True
            i += 1
            continue
        if not command_start:
            i += 1
            continue
        j = _skip_command_prefixes(tokens, i)
        if j >= len(tokens):
            break
        base = _command_token_basename(tokens[j])
        segment = _command_segment(tokens, j + 1)
        if base in _BROAD_KILL_BASENAMES:
            return _deny(
                "Praxist runtime guard blocked broad process signalling. Peers must not "
                "kill shared run processes or other peers.",
                rule_id="broad_process_kill",
            )
        if base == "xargs" and any(
            _command_token_basename(arg) in _BROAD_KILL_BASENAMES for arg in segment
        ):
            return _deny(
                "Praxist runtime guard blocked xargs-driven broad process signalling.",
                rule_id="xargs_process_kill",
            )
        command_start = False
        i += 1
    return DeleteGuardDecision(True)


def _validate_interpreter_stdin(command: str, tokens: list[str]) -> DeleteGuardDecision:
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        redir_tokens = list(lexer)
    except ValueError:
        redir_tokens = tokens

    command_start = True
    previous_was_pipe = False
    i = 0
    while i < len(redir_tokens):
        token = redir_tokens[i]
        if token in _SHELL_SEPARATORS or token in {"<", "<<", "<<<", ">", ">>", ">|", "&>", "&>>"}:
            previous_was_pipe = token == "|"
            command_start = token in _SHELL_SEPARATORS
            i += 1
            continue
        if not command_start:
            i += 1
            continue
        j = _skip_command_prefixes(redir_tokens, i)
        if j >= len(redir_tokens):
            break
        base = _command_token_basename(redir_tokens[j])
        segment = _command_segment_with_redirs(redir_tokens, j + 1)
        input_redirect = any(tok in {"<", "<<", "<<<"} for tok in segment)
        if base in _SHELL_BASENAMES:
            script = _shell_script_operand(segment)
            if (
                previous_was_pipe
                or input_redirect
                or (_shell_c_argument(segment) is None and script is None)
            ):
                return _warn(
                    "Praxist runtime guard allowed shell interpreter stdin with warning: "
                    "the payload is less inspectable than an explicit script or `bash -c`.",
                    rule_id="shell_stdin_uninspected",
                )
        if base.startswith("python"):
            if previous_was_pipe or input_redirect:
                return _warn(
                    "Praxist runtime guard allowed Python stdin with warning: the payload is "
                    "less inspectable than `python -c` or an explicit script path.",
                    rule_id="python_stdin_uninspected",
                )
            if (
                _shell_c_argument(segment) is None
                and not _python_uses_module_mode(segment)
                and _python_script_operand(segment) is None
            ):
                return _warn(
                    "Praxist runtime guard allowed Python without an explicit script or -c "
                    "payload; filesystem/process side effects remain guarded at runtime.",
                    rule_id="python_no_explicit_payload",
                )
        previous_was_pipe = False
        command_start = False
        i += 1
    return DeleteGuardDecision(True)


def _command_segment_with_redirs(tokens: list[str], start: int) -> list[str]:
    segment: list[str] = []
    for token in tokens[start:]:
        if token in _SHELL_SEPARATORS:
            break
        segment.append(token)
    return segment


def _shell_c_argument(args: list[str]) -> str | None:
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            i += 1
            continue
        if arg == "-c":
            if i + 1 < len(args) and args[i + 1] == "--" and i + 2 < len(args):
                return args[i + 2]
            return args[i + 1] if i + 1 < len(args) else ""
        if arg.startswith("-") and "c" in arg[1:]:
            after_c = arg.split("c", 1)[1]
            if after_c:
                return after_c
            if i + 1 < len(args) and args[i + 1] == "--" and i + 2 < len(args):
                return args[i + 2]
            return args[i + 1] if i + 1 < len(args) else ""
        i += 1
    return None


def _validate_expanded_command_words(tokens: list[str]) -> DeleteGuardDecision:
    command_start = True
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _SHELL_SEPARATORS:
            command_start = True
            i += 1
            continue
        if not command_start:
            i += 1
            continue
        j = _skip_command_prefixes(tokens, i)
        if j < len(tokens) and _has_command_word_expansion(tokens[j]):
            return DeleteGuardDecision(
                False,
                "Praxist delete guard blocked shell-expanded executable name. Use a literal "
                "command name so destructive cleanup can be validated.",
            )
        command_start = False
        i += 1
    return DeleteGuardDecision(True)


def _has_command_word_expansion(token: str) -> bool:
    return any(marker in token for marker in ("$", "`"))


def _validate_guard_mutation_patterns(tokens: list[str]) -> DeleteGuardDecision:
    if _operator_bypass_launches_workload(tokens):
        return _deny(
            "Praxist runtime guard blocked BYPASS_GPU_GOVERNOR on a workload launch. "
            "Resource-governed workloads must use the configured governor path.",
            rule_id="operator_bypass_workload",
        )
    if _guard_identity_launches_workload(tokens):
        return _deny(
            "Praxist runtime guard blocked peer identity/workspace environment override on "
            "a workload launch. Peer identity controls owned run paths.",
            rule_id="guard_identity_workload",
        )
    if _loader_injection_launches_workload(tokens):
        return _deny(
            "Praxist runtime guard blocked dynamic-loader environment override on a "
            "workload launch. Loader injection can bypass runtime guard behavior.",
            rule_id="loader_injection_workload",
        )
    for i, token in enumerate(tokens):
        base = _command_token_basename(token)
        if base == "trap":
            segment = _command_segment(tokens, i + 1)
            if "DEBUG" in segment:
                return _warn(
                    "Praxist runtime guard allowed Bash DEBUG trap mutation with warning; "
                    "guard enforcement no longer depends on a hidden shell trap.",
                    rule_id="bash_debug_trap_mutation",
                )
        if base == "unset":
            segment = _command_segment(tokens, i + 1)
            if any(arg in _GUARD_ENV_VARS for arg in segment):
                return _warn(
                    "Praxist runtime guard allowed unsetting guard-related environment "
                    "variables with warning; protected state and destructive effects "
                    "remain guarded.",
                    rule_id="guard_env_unset",
                )
        if base in {"export", "declare", "typeset", "local"}:
            segment = _command_segment(tokens, i + 1)
            if any(_is_guard_assignment(arg) for arg in segment):
                return _warn(
                    "Praxist runtime guard allowed guard-related environment assignment "
                    "with warning; protected state and destructive effects remain guarded.",
                    rule_id="guard_env_assignment",
                )
    for token in tokens:
        if _is_guard_assignment(token):
            return _warn(
                "Praxist runtime guard allowed guard-related environment assignment with "
                "warning; protected state and destructive effects remain guarded.",
                rule_id="guard_env_assignment",
            )
    return DeleteGuardDecision(True)


def _validate_executable_script_words(
    tokens: list[str], *, env: dict[str, str], cwd: Path, depth: int
) -> DeleteGuardDecision:
    command_start = True
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _SHELL_SEPARATORS:
            command_start = True
            i += 1
            continue
        if not command_start:
            i += 1
            continue
        j = _skip_command_prefixes(tokens, i)
        if j < len(tokens):
            token = tokens[j]
            base = _command_token_basename(token)
            if (
                base not in _SHELL_BASENAMES
                and not base.startswith("python")
                and base not in _DIRECT_DELETE_BASENAMES
                and base not in {"make", "gmake", "ninja"}
            ):
                path = _resolve_executable_word(token, tokens[:j], env=env, cwd=cwd)
                if path is not None and path.is_file() and os.access(path, os.X_OK):
                    allowed_roots = _allowed_roots(env)
                    trusted_or_read_only_git = (
                        _is_trusted_system_executable(path)
                        or _is_trusted_project_script(path, env, cwd=cwd)
                        or (
                            base == "git"
                            and _git_segment_is_read_only(_command_segment(tokens, j + 1))
                        )
                    )
                    if trusted_or_read_only_git:
                        pass
                    elif _is_within_any(path, allowed_roots) or _is_peer_owned_run_write_path(
                        path, env
                    ):
                        decision = _validate_script_file(str(path), env=env, cwd=cwd, depth=depth)
                        if not decision.allowed:
                            return decision
                    else:
                        return DeleteGuardDecision(
                            False,
                            "Praxist delete guard blocked executable script outside trusted "
                            f"system paths and this peer's workspace: {path}.",
                        )
        command_start = False
        i += 1
    return DeleteGuardDecision(True)


def _validate_unclassified_protected_arguments(
    tokens: list[str], *, env: dict[str, str], cwd: Path
) -> DeleteGuardDecision:
    command_start = True
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _SHELL_SEPARATORS:
            command_start = True
            i += 1
            continue
        if not command_start:
            i += 1
            continue
        j = _skip_command_prefixes(tokens, i)
        if j >= len(tokens):
            break
        cmd = tokens[j]
        base = _command_token_basename(cmd)
        if (
            base.startswith("python")
            or base in _SHELL_BASENAMES
            or base in _DIRECT_DELETE_BASENAMES
            or base in _SCRIPT_RUNTIME_BASENAMES
            or base in _READ_ONLY_SYSTEM_BASENAMES
            or base in _CLASSIFIED_MUTATING_BASENAMES
        ):
            command_start = False
            i += 1
            continue
        executable = _resolve_executable_word(cmd, tokens[:j], env=env, cwd=cwd)
        if executable is not None and _is_trusted_system_executable(executable):
            segment = _command_segment(tokens, j + 1)
            for arg in _pathlike_segment_args(segment):
                decision = _validate_unclassified_path_arg(arg, base=base, env=env, cwd=cwd)
                if not decision.allowed:
                    return decision
        command_start = False
        i += 1
    return DeleteGuardDecision(True)


def _pathlike_segment_args(segment: list[str]) -> list[str]:
    args: list[str] = []
    skip_next = False
    for token in segment:
        if skip_next:
            skip_next = False
            continue
        if token == "--":
            continue
        if token in {"-c", "-e", "-f"}:
            skip_next = True
            continue
        if token.startswith("-") and "=" not in token:
            continue
        value = token.split("=", 1)[1] if token.startswith("--") and "=" in token else token
        if _looks_pathlike_or_protected(value):
            args.append(value)
    return args


def _looks_pathlike_or_protected(value: str) -> bool:
    if not value:
        return False
    if value.startswith(("/", ".", "~", "$")):
        return True
    if "/" in value:
        return True
    return any(
        word in value
        for word in (
            "shared_findings",
            "frontier",
            "gems",
            "gen_",
            "results",
            "variants",
            "peer_workspaces",
            "shared_store.db",
        )
    )


def _validate_unclassified_path_arg(
    raw: str, *, base: str, env: dict[str, str], cwd: Path
) -> DeleteGuardDecision:
    resolved_targets = _resolve_target_candidates(raw, env=env, cwd=cwd)
    if not resolved_targets:
        return DeleteGuardDecision(
            False,
            f"Praxist delete guard blocked unclassified `{base}` command with ambiguous path: {raw!r}.",
        )
    protected_roots = _protected_roots(env, cwd=cwd)
    allowed_roots = _allowed_roots(env)
    for resolved in resolved_targets:
        if (
            _is_within_any(resolved, allowed_roots)
            or _is_peer_owned_run_write_path(resolved, env)
            or (
                _is_system_agenda_write_path(resolved, env)
                and _is_safe_existing_write_target(resolved)
            )
        ):
            continue
        if _is_within_any(resolved, protected_roots):
            return DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked unclassified `{base}` command touching protected path: {resolved}.",
            )
    return DeleteGuardDecision(True)


def _resolve_executable_word(
    token: str, prefix_tokens: list[str], *, env: dict[str, str], cwd: Path
) -> Path | None:
    if "/" in token:
        return _resolve_target(token, env=env, cwd=cwd)
    path_value = env.get("PATH", os.environ.get("PATH", ""))
    for prefix in prefix_tokens:
        if prefix.startswith("PATH="):
            path_value = _expand_target(prefix.split("=", 1)[1], env)
    for raw_dir in path_value.split(os.pathsep):
        if not raw_dir:
            continue
        directory = _absolute_path(_expand_target(raw_dir, env), cwd).resolve(strict=False)
        candidate = (directory / token).resolve(strict=False)
        if candidate.exists():
            return candidate
    return None


def _validate_guard_stripping_launches(
    tokens: list[str], *, env: dict[str, str]
) -> DeleteGuardDecision:
    for i, token in enumerate(tokens):
        if token in _SHELL_SEPARATORS:
            continue
        base = _command_token_basename(token)
        if base == "env":
            segment = _command_segment(tokens, i + 1)
            if _segment_contains_guarded_launcher(segment) and _env_segment_strips_guard(segment):
                return _warn(
                    "Praxist runtime guard allowed a launcher with a stripped guard-related "
                    "environment with warning; destructive side effects remain guarded.",
                    rule_id="stripped_guard_env_launch",
                )
        if base.startswith("python"):
            segment = _command_segment(tokens, i + 1)
            if _python_segment_disables_sitecustomize(segment):
                return _warn(
                    "Praxist runtime guard allowed Python launched with guard-disabling flags "
                    "with warning; destructive shell command patterns are still inspected "
                    "before execution.",
                    rule_id="python_guard_disabling_flags",
                )
        if base in _SCRIPT_RUNTIME_BASENAMES:
            segment = _command_segment(tokens, i + 1)
            if _script_runtime_payload_is_destructive(base, segment, env=env):
                return _deny(
                    "Praxist runtime guard blocked non-Python script runtime payload that "
                    "targets protected run/project state.",
                    rule_id="script_runtime_destructive_payload",
                )
            return _warn(
                "Praxist runtime guard allowed non-Python script runtime execution with "
                "warning; protected filesystem/process effects remain guarded by the "
                "outer command validator.",
                rule_id="non_python_script_runtime",
            )
    return DeleteGuardDecision(True)


def _script_runtime_payload_is_destructive(
    base: str, segment: list[str], *, env: dict[str, str]
) -> bool:
    payload = " ".join([base, *segment])
    lowered = payload.lower()
    destructive = any(
        term in lowered
        for term in (
            "unlink",
            "rmdir",
            "remove",
            "rmtree",
            "rm ",
            "/bin/rm",
            "open(",
            "write",
            "truncate",
        )
    )
    return destructive and _mentions_protected_target(payload, env)


def _segment_contains_guarded_launcher(segment: list[str]) -> bool:
    for token in segment:
        base = _command_token_basename(token)
        if base.startswith("python") or base in _SHELL_BASENAMES:
            return True
        if base in _DIRECT_DELETE_BASENAMES or base in {"find", "shred", "rsync"}:
            return True
        if base in {"make", "gmake", "ninja"}:
            return True
    return False


def _env_segment_strips_guard(segment: list[str]) -> bool:
    i = 0
    while i < len(segment):
        token = segment[i]
        if token in {"-i", "--ignore-environment", "-"}:
            return True
        if token in {"-u", "--unset"} and i + 1 < len(segment):
            if segment[i + 1] in _GUARD_ENV_VARS:
                return True
            i += 2
            continue
        if token.startswith("--unset=") and token.split("=", 1)[1] in _GUARD_ENV_VARS:
            return True
        if _is_guard_assignment(token):
            return True
        i += 1
    return False


def _python_segment_disables_sitecustomize(segment: list[str]) -> bool:
    for token in segment:
        if token == "--":
            continue
        if token in {"-S", "-I", "-E"}:
            return True
        if token.startswith("-") and any(flag in token[1:] for flag in ("S", "I", "E")):
            return True
    return False


def _validate_shell_script_operands(
    tokens: list[str], *, env: dict[str, str], cwd: Path, depth: int
) -> DeleteGuardDecision:
    i = 0
    while i < len(tokens):
        base = _command_token_basename(tokens[i])
        if base in _SHELL_BASENAMES:
            segment = _command_segment(tokens, i + 1)
            if _shell_c_argument(segment) is None:
                script = _shell_script_operand(segment)
                if script is not None:
                    warning: DeleteGuardDecision | None = None
                    if base != "bash":
                        warning = _warn(
                            "Praxist runtime guard allowed non-Bash shell script execution "
                            "with warning; script content is inspected before launch.",
                            rule_id="non_bash_shell_script",
                        )
                    decision = _validate_script_file(script, env=env, cwd=cwd, depth=depth)
                    if not decision.allowed:
                        return decision
                    if warning is not None:
                        return warning
        if base in {"source", "."}:
            segment = _command_segment(tokens, i + 1)
            script = segment[0] if segment else None
            if script:
                decision = _validate_script_file(script, env=env, cwd=cwd, depth=depth)
                if not decision.allowed:
                    return decision
        i += 1
    return DeleteGuardDecision(True)


def _validate_python_script_operands(
    tokens: list[str], *, env: dict[str, str], cwd: Path
) -> DeleteGuardDecision:
    i = 0
    while i < len(tokens):
        base = _command_token_basename(tokens[i])
        if base.startswith("python"):
            segment = _command_segment(tokens, i + 1)
            if _python_uses_module_mode(segment):
                decision = _validate_python_module_mode(segment, env=env, cwd=cwd)
                if not decision.allowed or decision.warning:
                    return decision
                i += 1
                continue
            script = _python_script_operand(segment)
            if script is not None:
                decision = _validate_python_script_file(script, env=env, cwd=cwd)
                if not decision.allowed:
                    return decision
        i += 1
    return DeleteGuardDecision(True)


def _python_uses_module_mode(segment: list[str]) -> bool:
    return any(token == "-m" for token in segment)


def _validate_python_module_mode(
    segment: list[str], *, env: dict[str, str], cwd: Path
) -> DeleteGuardDecision:
    for i, token in enumerate(segment):
        if token == "-m" and i + 1 < len(segment):
            module = segment[i + 1]
            if not _is_safe_python_module_name(module):
                return DeleteGuardDecision(
                    False,
                    "Praxist delete guard blocked unsafe `python -m` module name.",
                )
            if module in TRUSTED_RESOURCE_GUARD_MODULES:
                return DeleteGuardDecision(True)
            candidates = _python_module_file_candidates(module, env=env, cwd=cwd)
            if not candidates:
                return _warn(
                    "Praxist delete guard allowed unresolved safe `python -m` module "
                    "with warning; module file could not be resolved for inspection.",
                    rule_id="unresolved_python_m",
                )
            for candidate in candidates:
                decision = _validate_python_script_file(str(candidate), env=env, cwd=cwd)
                if not decision.allowed:
                    return decision
            return DeleteGuardDecision(True)
    return DeleteGuardDecision(
        False,
        "Praxist delete guard blocked malformed `python -m` invocation.",
    )


def _python_module_file_candidates(module: str, *, env: dict[str, str], cwd: Path) -> list[Path]:
    rel = Path(*module.split("."))
    roots: list[Path] = [cwd]
    roots.extend(_allowed_roots(env))
    roots.extend(_trusted_project_roots(env, cwd=cwd))
    pythonpath = env.get("PYTHONPATH") or ""
    for raw in pythonpath.split(os.pathsep):
        if raw.strip():
            roots.append(_absolute_path(_expand_target(raw, env), cwd).resolve(strict=False))
    seen_roots: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        root = root.resolve(strict=False)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        module_file = (root / rel).with_suffix(".py").resolve(strict=False)
        package_file = (root / rel / "__main__.py").resolve(strict=False)
        for path in (module_file, package_file):
            if path.exists() and path.is_file():
                candidates.append(path)
    unique: dict[str, Path] = {}
    for path in candidates:
        unique[str(path.resolve(strict=False))] = path
    return list(unique.values())


def _is_safe_python_module_name(module: str) -> bool:
    parts = module.split(".")
    if not parts:
        return False
    for part in parts:
        if not part or not (part[0].isalpha() or part[0] == "_"):
            return False
        if not all(ch.isalnum() or ch == "_" for ch in part):
            return False
    return True


def _python_script_operand(segment: list[str]) -> str | None:
    i = 0
    while i < len(segment):
        token = segment[i]
        if token == "--":
            i += 1
            break
        if token in {"-c", "-m"} or token == "-":
            return None
        if token in {"-W", "-X"}:
            i += 2
            continue
        if token.startswith("-W") or token.startswith("-X"):
            i += 1
            continue
        if token.startswith("-"):
            i += 1
            continue
        return token
    return segment[i] if i < len(segment) else None


def _shell_script_operand(segment: list[str]) -> str | None:
    i = 0
    while i < len(segment):
        token = segment[i]
        if token == "--":
            i += 1
            break
        if token.startswith("-"):
            i += 1
            continue
        return token
    return segment[i] if i < len(segment) else None


def _validate_python_script_file(
    script: str, *, env: dict[str, str], cwd: Path
) -> DeleteGuardDecision:
    path = _resolve_target(script, env=env, cwd=cwd)
    if path is None:
        return DeleteGuardDecision(
            False,
            f"Praxist delete guard blocked ambiguous Python script path: {script!r}.",
        )
    if not path.exists() or not path.is_file():
        return DeleteGuardDecision(True)
    run_dir = _run_dir(env)
    if _is_trusted_project_script(path, env, cwd=cwd) and not (
        run_dir is not None and _is_within_any(path.resolve(strict=False), [run_dir])
    ):
        return DeleteGuardDecision(True)
    try:
        head = path.read_bytes()[:4096]
        if b"\x00" in head or head.startswith(b"\x7fELF"):
            return DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked binary executable from peer workspace: {path}",
            )
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return DeleteGuardDecision(
            False,
            f"Praxist delete guard could not inspect Python script {path}: {exc}",
        )
    if len(content) > 200_000:
        return DeleteGuardDecision(
            False,
            f"Praxist delete guard blocked oversized Python script without inspection: {path}",
        )
    return _validate_python_delete_patterns(content, env=env, allowed_roots=_allowed_roots(env))


def _validate_script_file(
    script: str, *, env: dict[str, str], cwd: Path, depth: int
) -> DeleteGuardDecision:
    path = _resolve_target(script, env=env, cwd=cwd)
    if path is None:
        return DeleteGuardDecision(
            False,
            f"Praxist delete guard blocked ambiguous shell script path: {script!r}.",
        )
    try:
        path.relative_to(_resolve_target("$PRAXIST_PEER_WORKSPACE", env=env, cwd=cwd) or path)
        inside_peer_workspace = True
    except ValueError:
        inside_peer_workspace = False
    if not path.exists():
        return DeleteGuardDecision(bool(inside_peer_workspace))
    if path.is_dir():
        return DeleteGuardDecision(True)
    try:
        head = path.read_bytes()[:4096]
        if _looks_binary_payload(head):
            return DeleteGuardDecision(
                False,
                f"Praxist delete guard blocked binary executable from peer workspace: {path}",
            )
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return DeleteGuardDecision(
            False,
            f"Praxist delete guard could not inspect shell script {path}: {exc}",
        )
    if len(content) > 200_000:
        return DeleteGuardDecision(
            False,
            f"Praxist delete guard blocked oversized shell script without inspection: {path}",
        )
    return _validate_bash_command(content, env=env, cwd=path.parent, depth=depth + 1)


def _looks_binary_payload(head: bytes) -> bool:
    return (
        b"\x00" in head
        or head.startswith(b"\x7fELF")
        or head.startswith(b"\xcf\xfa\xed\xfe")
        or head.startswith(b"\xfe\xed\xfa\xcf")
        or head.startswith(b"\xca\xfe\xba\xbe")
    )


def _is_guard_assignment(token: str) -> bool:
    if "=" not in token or token.startswith("-"):
        return False
    name = token.split("=", 1)[0]
    return name in _GUARD_ENV_VARS


def _is_operator_only_assignment(token: str) -> bool:
    if "=" not in token or token.startswith("-"):
        return False
    return token.split("=", 1)[0] in OPERATOR_ONLY_ENV_KEYS


def _is_identity_guard_assignment(token: str) -> bool:
    if "=" not in token or token.startswith("-"):
        return False
    return token.split("=", 1)[0] in _IDENTITY_GUARD_ENV_KEYS


def _is_loader_injection_assignment(token: str) -> bool:
    if "=" not in token or token.startswith("-"):
        return False
    return token.split("=", 1)[0] in _LOADER_INJECTION_ENV_KEYS


def _assignment_launches_workload(
    tokens: list[str], predicate: Any, *, allowed_commands: set[str]
) -> bool:
    assignment_seen = False
    persistent_assignment_seen = False
    command_start = True
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _SHELL_SEPARATORS:
            assignment_seen = persistent_assignment_seen
            command_start = True
            i += 1
            continue
        if predicate(token):
            assignment_seen = True
            i += 1
            continue
        base = _command_token_basename(token)
        if base in {"export", "declare", "typeset", "local"}:
            segment = _command_segment(tokens, i + 1)
            if any(predicate(arg) for arg in segment):
                assignment_seen = True
                persistent_assignment_seen = True
            i += 1
            continue
        if base == "env":
            i += 1
            continue
        if base == "unset":
            assignment_seen = False
            persistent_assignment_seen = False
            i += 1
            continue
        if not command_start:
            i += 1
            continue
        if assignment_seen:
            return base not in allowed_commands
        command_start = False
        i += 1
    return False


def _operator_bypass_launches_workload(tokens: list[str]) -> bool:
    return _assignment_launches_workload(
        tokens,
        _is_operator_only_assignment,
        allowed_commands={"echo", "env", "false", "printf", "pwd", "test", "true", "which"},
    )


def _guard_identity_launches_workload(tokens: list[str]) -> bool:
    return _assignment_launches_workload(
        tokens,
        _is_identity_guard_assignment,
        allowed_commands={"echo", "env", "false", "printf", "pwd", "test", "true", "which"},
    )


def _loader_injection_launches_workload(tokens: list[str]) -> bool:
    return _assignment_launches_workload(
        tokens,
        _is_loader_injection_assignment,
        allowed_commands={"echo", "env", "false", "printf", "pwd", "test", "true", "which"},
    )


def _find_execs_destructive_command(segment: list[str]) -> bool:
    if "-exec" not in segment and "-execdir" not in segment:
        return False
    for token in segment:
        base = _basename(token)
        if (
            base in _DIRECT_DELETE_BASENAMES
            or base in _CLASSIFIED_MUTATING_BASENAMES
            or base in {"shred", "rsync"}
        ):
            return True
        if base == "env":
            return True
        if base in _SHELL_BASENAMES:
            return True
        if base.startswith("python"):
            return True
        if base in _SCRIPT_RUNTIME_BASENAMES:
            return True
    return False


def _iter_delete_invocations_anywhere(tokens: list[str]) -> list[list[str]]:
    invocations = _iter_delete_invocations(tokens)
    seen = {tuple(args) for args in invocations}
    for i, token in enumerate(tokens):
        if _command_token_basename(token) in _DIRECT_DELETE_BASENAMES:
            segment = _command_segment(tokens, i + 1)
            key = tuple(segment)
            if key not in seen:
                invocations.append(segment)
                seen.add(key)
    return invocations


def _validate_python_delete_patterns(
    command: str, *, env: dict[str, str], allowed_roots: list[Path]
) -> DeleteGuardDecision:
    lowered = command.lower()
    protected_target = _mentions_protected_target(command, env)
    direct_delete = any(pattern in lowered for pattern in _PYTHON_DELETE_PATTERNS)
    shell_escape = any(pattern in lowered for pattern in _PYTHON_SHELL_ESCAPE_PATTERNS)
    shell_delete = any(pattern in lowered for pattern in _PYTHON_SHELL_DELETE_TERMS)
    hard_native_escape = any(
        pattern in lowered
        for pattern in (
            "ctypes",
            "cffi",
            "_cffi_backend",
            "_ctypes",
            "cdll",
            "pydll",
            "libdl",
            "syscall",
            "sitecustomize",
            "_roots",
            "_protected_roots",
            "_orig",
            "_orig_",
            "_orig_by_wrapper",
            "_origs",
            "_orig_by",
            "object.__getattribute__",
            "__getattribute__",
            "os.putenv",
            "os.unsetenv",
            "os.kill",
            "os.killpg",
            "signal.sigkill",
            "signal.sigterm",
        )
    )
    guard_state_mutation = any(
        pattern in lowered
        for pattern in (
            "_roots",
            "_protected_roots",
        )
    )
    process_signal_escape = any(
        pattern in lowered
        for pattern in (
            "os.kill",
            "os.killpg",
            "signal.sigkill",
            "signal.sigterm",
        )
    )
    soft_native_escape = any(
        pattern in lowered
        for pattern in (
            "__dict__",
            "__import__(",
            "importlib",
            "getattr(",
            "setattr(",
            "delattr(",
            "os.environ",
        )
    )
    hard_introspection_escape = any(
        pattern in lowered
        for pattern in (
            "sys.modules",
            ".modules",
            "modules[",
            "modules.",
            "__globals__",
            "__defaults__",
            "__kwdefaults__",
            "__closure__",
            "__code__",
            "func_globals",
            "exec(",
            "eval(",
            "sitecustomize",
        )
    )
    soft_introspection_escape = any(
        pattern in lowered
        for pattern in (
            "globals()",
            "locals()",
            "vars(",
            "dir(",
        )
    )
    escape_context = protected_target or direct_delete or shell_delete or shell_escape
    if guard_state_mutation:
        return DeleteGuardDecision(
            False,
            "Praxist delete guard blocked Python mutation of delete-guard runtime state.",
        )
    if process_signal_escape:
        return DeleteGuardDecision(
            False,
            "Praxist delete guard blocked Python process signalling code in a Bash command.",
        )
    if hard_native_escape and escape_context:
        return DeleteGuardDecision(
            False,
            "Praxist delete guard blocked Python native/private runtime escape code.",
        )
    if soft_native_escape and escape_context:
        return DeleteGuardDecision(
            False,
            "Praxist delete guard blocked Python native/private runtime escape code.",
        )
    if hard_introspection_escape and escape_context:
        return DeleteGuardDecision(
            False,
            "Praxist delete guard blocked Python runtime introspection that could bypass "
            "peer-local file safety.",
        )
    if soft_introspection_escape and escape_context:
        return DeleteGuardDecision(
            False,
            "Praxist delete guard blocked Python runtime introspection that could bypass "
            "peer-local file safety.",
        )
    warnings: list[DeleteGuardDecision] = []
    if hard_native_escape:
        warnings.append(
            _warn(
                "Praxist delete guard allowed Python native/private runtime reference with "
                "warning; protected filesystem and process effects remain guarded.",
                rule_id="python_native_private_reference",
            )
        )
    if hard_introspection_escape:
        warnings.append(
            _warn(
                "Praxist delete guard allowed Python runtime introspection reference with "
                "warning; protected filesystem and process effects remain guarded.",
                rule_id="python_runtime_introspection_reference",
            )
        )
    if not direct_delete and not (shell_escape and (protected_target or shell_delete)):
        return _combine_warnings(warnings)
    return DeleteGuardDecision(
        False,
        "Praxist delete guard blocked Python deletion code in a Bash command. "
        "Use shell `rm` only inside $PRAXIST_PEER_WORKSPACE for destructive cleanup.",
    )


def _shell_tokens(command: str) -> list[str]:
    tokens, _warning = _shell_tokens_with_warning(command)
    return tokens


def _shell_tokens_with_warning(command: str) -> tuple[list[str], str]:
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer), ""
    except ValueError as exc:
        primary_error = str(exc)
    try:
        return shlex.split(command, posix=True), (
            f"Praxist runtime guard recovered from shell tokenization warning: {primary_error}."
        )
    except ValueError as exc:
        tokens = _rough_shell_tokens(command)
        return tokens, (
            "Praxist runtime guard allowed shell command with warning: command was "
            f"not fully parseable ({exc}); destructive raw patterns remain guarded."
        )


def _rough_shell_tokens(command: str) -> list[str]:
    return re.findall(r"[;&|()<>]+|[^\s;&|()<>]+", command.replace("\n", " ; "))


def _iter_delete_invocations(tokens: list[str]) -> list[list[str]]:
    invocations: list[list[str]] = []
    command_start = True
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _SHELL_SEPARATORS:
            command_start = True
            i += 1
            continue
        if not command_start:
            i += 1
            continue
        j = _skip_command_prefixes(tokens, i)
        if j < len(tokens) and _basename(tokens[j]) in _DIRECT_DELETE_BASENAMES:
            invocations.append(_command_segment(tokens, j + 1))
        command_start = False
        i += 1
    return invocations


def _skip_command_prefixes(tokens: list[str], i: int) -> int:
    while i < len(tokens):
        token = tokens[i]
        if "=" in token and not token.startswith("-") and token.split("=", 1)[0].isidentifier():
            i += 1
            continue
        base = _basename(token)
        if base == "command":
            i = _skip_command_options(tokens, i + 1)
            continue
        if base == "builtin":
            i += 1
            continue
        if base == "exec":
            i += 1
            continue
        if base == "env":
            i = _skip_env_prefix(tokens, i + 1)
            continue
        if base == "sudo":
            i = _skip_simple_options(tokens, i + 1)
            continue
        if base == "time":
            i = _skip_simple_options(tokens, i + 1)
            continue
        if base == "nice":
            i = _skip_nice_prefix(tokens, i + 1)
            continue
        if base == "nohup":
            i += 1
            continue
        break
    return i


def _skip_command_options(tokens: list[str], i: int) -> int:
    while i < len(tokens):
        if tokens[i] == "--":
            return i + 1
        if tokens[i] not in _COMMAND_OPTIONS:
            return i
        i += 1
    return i


def _skip_env_prefix(tokens: list[str], i: int) -> int:
    while i < len(tokens):
        token = tokens[i]
        if token == "--":
            return i + 1
        if token in _ENV_OPTIONS_WITH_VALUE:
            i += 2
            continue
        if token.startswith("--unset=") or token.startswith("--chdir="):
            i += 1
            continue
        if token.startswith("-"):
            i += 1
            continue
        if "=" in token and token.split("=", 1)[0].isidentifier():
            i += 1
            continue
        return i
    return i


def _skip_simple_options(tokens: list[str], i: int) -> int:
    while i < len(tokens):
        token = tokens[i]
        if token == "--":
            return i + 1
        if token.startswith("-"):
            i += 1
            continue
        return i
    return i


def _skip_nice_prefix(tokens: list[str], i: int) -> int:
    while i < len(tokens):
        token = tokens[i]
        if token == "--":
            return i + 1
        if token in _NICE_OPTIONS_WITH_VALUE:
            i += 2
            continue
        if token.startswith("--adjustment="):
            i += 1
            continue
        if token.startswith("-"):
            i += 1
            continue
        return i
    return i


def _command_segment(tokens: list[str], start: int) -> list[str]:
    segment: list[str] = []
    for token in tokens[start:]:
        if token in _SHELL_SEPARATORS:
            break
        segment.append(token)
    return segment


def _rm_targets(args: list[str]) -> list[str]:
    targets: list[str] = []
    parsing_options = True
    for arg in args:
        if parsing_options and arg == "--":
            parsing_options = False
            continue
        if parsing_options and arg.startswith("-"):
            continue
        targets.append(arg)
    return targets


def _resolve_target(target: str, *, env: dict[str, str], cwd: Path) -> Path | None:
    candidates = _resolve_target_candidates(target, env=env, cwd=cwd)
    if candidates is None:
        return None
    if len(candidates) != 1:
        return None
    return candidates[0]


def _resolve_target_candidates(target: str, *, env: dict[str, str], cwd: Path) -> list[Path] | None:
    if not target:
        return None
    expanded = _expand_target(target, env)
    if "$" in expanded:
        return None
    if _has_glob_meta(expanded):
        matches = glob.glob(expanded, recursive=True)
        if matches:
            return [_absolute_path(match, cwd).resolve(strict=False) for match in matches]
        glob_prefix = re.split(r"[*?\\[]", expanded, maxsplit=1)[0]
        if not glob_prefix or ".." in Path(expanded).parts:
            return None
        return [_absolute_path(glob_prefix, cwd).resolve(strict=False)]
    return [_absolute_path(expanded, cwd).resolve(strict=False)]


def _expand_target(target: str, env: dict[str, str]) -> str:
    """Expand shell-style env references without mutating process environment."""

    def repl(match: re.Match[str]) -> str:
        braced = match.group(1)
        plain = match.group(2)
        key = braced if braced is not None else plain
        if key is None:
            return match.group(0)
        return env.get(key, os.environ.get(key, match.group(0)))

    return re.sub(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)", repl, target)


def _allowed_roots(env: dict[str, str]) -> list[Path]:
    roots: list[Path] = []
    for raw in (env.get("PRAXIST_SAFE_DELETE_ROOTS") or "").split(os.pathsep):
        if raw.strip():
            roots.append(Path(raw).expanduser().resolve())
    return roots


def _run_dir(env: dict[str, str]) -> Path | None:
    raw = env.get("PRAXIST_DELETE_GUARD_RUN_DIR") or env.get("PRAXIST_RUN_DIR")
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _protected_roots(env: dict[str, str], *, cwd: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    for key in PROTECTED_ROOT_ENV_KEYS:
        raw = env.get(key)
        if raw:
            root = Path(raw).expanduser().resolve()
            roots.append(root)
    roots.extend(_trusted_project_roots(env, cwd=cwd))
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root not in seen:
            unique.append(root)
            seen.add(root)
    return unique


def _peer_id(env: dict[str, str]) -> str:
    raw = env.get("PRAXIST_DELETE_GUARD_AGENT") or env.get("PEER_ID") or ""
    if not raw and env.get("PRAXIST_PEER_WORKSPACE"):
        raw = Path(env["PRAXIST_PEER_WORKSPACE"]).name
    return _safe_name(raw or "agent")


def _run_relative_parts(path: Path, env: dict[str, str]) -> tuple[str, ...] | None:
    run_dir = _run_dir(env)
    if run_dir is None:
        return None
    try:
        return path.resolve(strict=False).relative_to(run_dir).parts
    except ValueError:
        return None


def _apparent_nonglob_target(target: str, *, env: dict[str, str], cwd: Path) -> Path | None:
    if not target:
        return None
    expanded = _expand_target(target, env)
    if "$" in expanded or _has_glob_meta(expanded):
        return None
    return _absolute_path(expanded, cwd)


def _run_relative_parts_apparent(path: Path, env: dict[str, str]) -> tuple[str, ...] | None:
    run_dir = _run_dir(env)
    if run_dir is None:
        return None
    normalized = Path(os.path.normpath(str(path.expanduser())))
    for candidate in (normalized, normalized.resolve(strict=False)):
        try:
            return candidate.relative_to(run_dir).parts
        except ValueError:
            continue
    return None


_PEER_MUTABLE_MEMORY_FILES = {
    "peer_state.yaml",
    "experiment_ledger.jsonl",
    "session_handoff.md",
    "session_auto_handoff.md",
}

_PEER_MUTABLE_MEMORY_DIRECT_WRITE_OPS = {
    "open/write",
    "io.open/write",
    "_io.open/write",
    "os.open/write",
    "posix.open/write",
    "Path.open/write",
    "Path.write_text",
    "Path.write_bytes",
    "truncate",
    "ftruncate",
    "posix.truncate",
    "posix.ftruncate",
}
_PI_AGENDA_WRITER_AGENTS = {"pi_synthesizer", "chair_arbiter"}


def _is_peer_owned_run_write_path(path: Path, env: dict[str, str]) -> bool:
    parts = _run_relative_parts(path, env)
    if not parts:
        return False
    peer = _peer_id(env)
    if parts[0] in {"variants", "results"} and len(parts) >= 2:
        if len(parts) >= 3 and parts[0] == "results" and re.fullmatch(r"gen_\d+", parts[1]):
            parts = parts[1:]
        return parts[1].startswith(f"{peer}_") or parts[1] == peer
    if parts[0] == "shared_findings" and len(parts) == 2:
        return parts[1].startswith(f"{peer}_")
    if (
        len(parts) == 5
        and re.fullmatch(r"gen_\d+", parts[0])
        and parts[1] == "peers"
        and parts[2] == peer
        and parts[3] == "dig"
        and parts[4] in {"contract_amendment.yaml", "contract_amendment.yml"}
    ):
        return True
    return len(parts) == 1 and parts[0] == f"notebook_{peer}.json"


def _is_pi_panel_agenda_writer(env: dict[str, str]) -> bool:
    return _peer_id(env) in _PI_AGENDA_WRITER_AGENTS


def _is_system_agenda_write_path(path: Path, env: dict[str, str]) -> bool:
    if not _is_pi_panel_agenda_writer(env):
        return False
    parts = _run_relative_parts(path, env)
    if not parts or len(parts) != 2 or parts[0] != "agendas":
        return False
    return re.fullmatch(r"research_agenda_gen\d+\.ya?ml(?:\.candidate)?", parts[1]) is not None


def _is_peer_mutable_memory_run_path(path: Path, env: dict[str, str]) -> bool:
    parts = _run_relative_parts(path, env)
    return _is_peer_mutable_memory_run_parts(parts, env)


def _is_apparent_peer_mutable_memory_run_path(path: Path, env: dict[str, str]) -> bool:
    parts = _run_relative_parts_apparent(path, env)
    return _is_peer_mutable_memory_run_parts(parts, env)


def _is_peer_mutable_memory_run_parts(parts: tuple[str, ...] | None, env: dict[str, str]) -> bool:
    if not parts:
        return False
    peer = _peer_id(env)
    return bool(
        len(parts) == 5
        and re.fullmatch(r"gen_\d+", parts[0])
        and parts[1] == "peers"
        and parts[2] == peer
        and parts[3] == "memory"
        and parts[4] in _PEER_MUTABLE_MEMORY_FILES
    )


def _is_safe_peer_mutable_memory_direct_write_path(path: Path, env: dict[str, str]) -> bool:
    if not _is_peer_mutable_memory_run_path(path, env):
        return False
    try:
        st = path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return False
    if stat.S_ISDIR(st.st_mode):
        return True
    if st.st_nlink > 1:
        return False
    return stat.S_ISREG(st.st_mode)


def _is_safe_peer_owned_run_write_target(path: Path, env: dict[str, str]) -> bool:
    if not _is_peer_owned_run_write_path(path, env):
        return False
    apparent = Path(os.path.normpath(str(path.expanduser())))
    try:
        st = apparent.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return False
    if stat.S_ISDIR(st.st_mode):
        return True
    if st.st_nlink > 1:
        return False
    return stat.S_ISREG(st.st_mode)


def _is_safe_existing_write_target(path: Path) -> bool:
    try:
        st = path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return False
    if stat.S_ISDIR(st.st_mode):
        return True
    if st.st_nlink > 1:
        return False
    return stat.S_ISREG(st.st_mode)


def _is_peer_owned_run_delete_path(path: Path, env: dict[str, str]) -> bool:
    parts = _run_relative_parts(path, env)
    if not parts or len(parts) < 2 or parts[0] not in {"variants", "results"}:
        return False
    if len(parts) >= 3 and parts[0] == "results" and re.fullmatch(r"gen_\d+", parts[1]):
        parts = parts[1:]
    peer = _peer_id(env)
    return parts[1].startswith(f"{peer}_") or parts[1] == peer


def _trusted_project_roots(env: dict[str, str], *, cwd: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    for key in TRUSTED_PROJECT_ENV_KEYS:
        raw = env.get(key)
        if raw:
            roots.append(Path(raw).expanduser().resolve(strict=False))
    for extra in split_path_list(env.get(TRUSTED_PROJECT_EXTRA_ROOTS_ENV)):
        roots.append(extra.resolve(strict=False))
    run_dir = _run_dir(env)
    if run_dir is not None and run_dir.parent.name == "experiments":
        roots.append(run_dir.parent.parent.resolve(strict=False))
    if cwd is not None:
        roots.extend(_discover_task_roots(cwd))
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root not in seen:
            unique.append(root)
            seen.add(root)
    return unique


def _discover_task_roots(cwd: Path) -> list[Path]:
    roots: list[Path] = []
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "experiments").is_dir() and (
            (candidate / "description.md").exists()
            or (candidate / "assets").is_dir()
            or (candidate / "evaluations").is_dir()
        ):
            roots.append(candidate.resolve(strict=False))
            break
        if candidate.name == "experiments":
            roots.append(candidate.parent.resolve(strict=False))
            break
    return roots


def _is_trusted_project_script(path: Path, env: dict[str, str], *, cwd: Path | None = None) -> bool:
    roots = _trusted_project_roots(env, cwd=cwd)
    return _is_within_any(path.resolve(strict=False), roots)


def _is_within_any(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _is_trusted_system_executable(path: Path) -> bool:
    trusted = [
        Path("/bin"),
        Path("/usr/bin"),
        Path("/usr/sbin"),
        Path("/sbin"),
        Path("/opt/homebrew/bin"),
        Path("/opt/homebrew/sbin"),
        Path("/opt/homebrew/Cellar"),
        Path("/opt/homebrew/opt"),
        Path("/usr/local/bin"),
        Path("/usr/local/sbin"),
        Path("/usr/local/Cellar"),
        Path("/usr/local/opt"),
    ]
    return _is_within_any(
        path.resolve(strict=False), [root.resolve(strict=False) for root in trusted]
    )


def _basename(token: str) -> str:
    return Path(token).name


def _command_token_basename(token: str) -> str:
    cleaned = token.strip("`$(){}\"'")
    return Path(cleaned).name


def _absolute_path(raw: str, cwd: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(cwd) / path
    return path


def _has_glob_meta(value: str) -> bool:
    return any(ch in value for ch in ("*", "?", "["))


def _mentions_protected_target(command: str, env: dict[str, str]) -> bool:
    expanded = _expand_target(command, env)
    run_dir = env.get("PRAXIST_DELETE_GUARD_RUN_DIR") or env.get("PRAXIST_RUN_DIR") or ""
    if run_dir and run_dir in expanded:
        return True
    if "$PRAXIST_RUN_DIR" in command or "${PRAXIST_RUN_DIR}" in command:
        return True
    protected_words = (
        "shared_findings",
        "frontier",
        "gems",
        "gem",
        "gen_",
        "results",
        "variants",
        "peer_workspaces",
        "generation_results",
        "run_summary",
        "shared_store.db",
    )
    return any(word in command for word in protected_words)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned or "agent"


def _join_paths(paths: list[Path]) -> str:
    return os.pathsep.join(str(path) for path in paths)


def _bash_env_text() -> str:
    return """# Generated by Praxist Claude SDK runtime guard.
# Bash runtime hooks are intentionally disabled. PreToolUse validation is the
# primary shell decision point; generated Python sitecustomize protects real
# filesystem/process side effects that escape that path.
export PRAXIST_DELETE_GUARD_ACTIVE=1
"""


def _sitecustomize_text() -> str:
    body = r'''"""Praxist peer-local Python delete guard; generated at runtime."""
from __future__ import annotations

import os
import builtins
import json
import pathlib
import posix
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import io
import _io
import sqlite3
import ctypes
import _ctypes
import importlib.machinery
import importlib.util
import time
import fcntl
from urllib.parse import parse_qs, unquote, urlparse

_ROOTS = [
    pathlib.Path(p).expanduser().resolve()
    for p in os.environ.get("PRAXIST_SAFE_DELETE_ROOTS", "").split(os.pathsep)
    if p
]
_ROOTS = tuple(_ROOTS)
_RUN_DIR = os.environ.get("PRAXIST_DELETE_GUARD_RUN_DIR") or os.environ.get("PRAXIST_RUN_DIR") or ""
_RUN_ROOT = pathlib.Path(_RUN_DIR).expanduser().resolve() if _RUN_DIR else None
_TRUSTED_PROJECT_ROOTS = []
for _key in __PRAXIST_TRUSTED_PROJECT_ENV_KEYS__:
    _raw = os.environ.get(_key) or ""
    if _raw:
        _root = pathlib.Path(_raw).expanduser().resolve()
        if _root not in _TRUSTED_PROJECT_ROOTS:
            _TRUSTED_PROJECT_ROOTS.append(_root)
for _raw in (os.environ.get(__PRAXIST_TRUSTED_PROJECT_EXTRA_ROOTS_ENV__) or "").split(os.pathsep):
    if _raw.strip():
        _root = pathlib.Path(_raw).expanduser().resolve()
        if _root not in _TRUSTED_PROJECT_ROOTS:
            _TRUSTED_PROJECT_ROOTS.append(_root)
if _RUN_ROOT is not None and _RUN_ROOT.parent.name == "experiments":
    _task_root = _RUN_ROOT.parent.parent.resolve()
    if _task_root not in _TRUSTED_PROJECT_ROOTS:
        _TRUSTED_PROJECT_ROOTS.append(_task_root)
_TRUSTED_PROJECT_ROOTS = tuple(_TRUSTED_PROJECT_ROOTS)
_PROTECTED_ROOTS = []
for _key in __PRAXIST_PROTECTED_ROOT_ENV_KEYS__:
    _raw = os.environ.get(_key) or ""
    if _raw:
        _root = pathlib.Path(_raw).expanduser().resolve()
        if _root not in _PROTECTED_ROOTS:
            _PROTECTED_ROOTS.append(_root)
if _RUN_ROOT is not None and _RUN_ROOT.parent.name == "experiments":
    _task_root = _RUN_ROOT.parent.parent.resolve()
    if _task_root not in _PROTECTED_ROOTS:
        _PROTECTED_ROOTS.append(_task_root)
_PROTECTED_ROOTS = tuple(_PROTECTED_ROOTS)
_RESOURCE_GUARD_STATE_NAMES = __PRAXIST_RESOURCE_STATE_DIR_NAMES__
_RESOURCE_GUARD_MODULE_SUFFIXES = __PRAXIST_TRUSTED_RESOURCE_GUARD_MODULE_SUFFIXES__
try:
    _GUARD_FILE_PEER_ID = pathlib.Path(__file__).resolve().parents[1].name
except Exception:
    _GUARD_FILE_PEER_ID = ""
_PEER_ID = _GUARD_FILE_PEER_ID or os.environ.get("PRAXIST_DELETE_GUARD_AGENT") or os.environ.get("PEER_ID") or ""
if not _PEER_ID and os.environ.get("PRAXIST_PEER_WORKSPACE"):
    _PEER_ID = pathlib.Path(os.environ["PRAXIST_PEER_WORKSPACE"]).name
_PEER_ID = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in (_PEER_ID or "agent")).strip("._-") or "agent"
_GUARD_WARNING_PATH = os.environ.get(__PRAXIST_GUARD_WARNING_ENV_KEY__) or ""
_RUNTIME_TMP_APPARENT = (
    pathlib.Path(os.environ["TMPDIR"]).expanduser().absolute()
    if os.environ.get("TMPDIR")
    else None
)
_RUNTIME_TEMP_DELETE_DIRS = tuple(
    dict.fromkeys(
        path
        for raw in ("/dev/shm", os.environ.get("TMPDIR", ""), "/tmp", "/var/tmp")
        if raw
        for path in (
            pathlib.Path(raw).expanduser().absolute(),
            pathlib.Path(raw).expanduser().resolve(),
        )
    )
)


def _module_code_objects(module):
    codes = set()

    def add(value):
        if isinstance(value, (classmethod, staticmethod)):
            value = value.__func__
        code = getattr(value, "__code__", None)
        if code is not None:
            codes.add(code)

    for value in vars(module).values():
        add(value)
        if isinstance(value, type):
            for member in vars(value).values():
                add(member)
    return frozenset(codes)


_TEMPFILE_CODE_OBJECTS = _module_code_objects(tempfile)
_RUNTIME_TEMP_DELETE_PREFIXES = (
    "pym-",
    "pymp-",
    "psm_",
    "sem.",
    "torch_",
)
_PEER_MUTABLE_MEMORY_FILES = {
    "peer_state.yaml",
    "experiment_ledger.jsonl",
    "session_handoff.md",
    "session_auto_handoff.md",
}
_PI_AGENDA_WRITER_AGENTS = {"pi_synthesizer", "chair_arbiter"}
_PEER_MUTABLE_MEMORY_DIRECT_WRITE_OPS = {
    "open/write",
    "io.open/write",
    "_io.open/write",
    "os.open/write",
    "posix.open/write",
    "Path.open/write",
    "Path.write_text",
    "Path.write_bytes",
    "truncate",
    "ftruncate",
    "posix.truncate",
    "posix.ftruncate",
}
_DANGEROUS_BUILD_TARGETS = {
    "clean",
    "distclean",
    "clobber",
    "mrproper",
    "delete",
    "remove",
    "purge",
    "wipe",
    "reset",
}
_MAKEFILE_NAMES = ("GNUmakefile", "makefile", "Makefile")
_DANGEROUS_BUILD_RECIPE_MARKERS = (
    "$(shell",
    "$(",
    "${",
    "$$",
    ">",
    ">>",
    "`",
    "/bin/rm",
    " rm ",
    "\trm ",
    "rm -",
    "rmdir",
    "unlink",
    "shred",
    "touch ",
    "\ttouch ",
    "cp ",
    "\tcp ",
    "install ",
    "\tinstall ",
    "mv ",
    "\tmv ",
    "truncate ",
    "\ttruncate ",
    "tee ",
    "\ttee ",
    "dd ",
    "\tdd ",
    "rsync",
    "--delete",
    "python -c",
    "python3 -c",
    "bash -c",
    "sh -c",
    "env -i",
    "shared_findings",
    "frontier",
    "gems",
    "shared_store.db",
)


def _resolve_path(path, cwd=None):
    candidate = pathlib.Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = pathlib.Path(cwd or os.getcwd()) / candidate
    return candidate.resolve()


def _under_allowed(path, cwd=None) -> bool:
    if not _ROOTS:
        return True
    resolved = _resolve_path(path, cwd=cwd)
    return any(resolved == root or root in resolved.parents for root in _ROOTS)


def _under_any(path, roots, cwd=None) -> bool:
    resolved = _resolve_path(path, cwd=cwd)
    return any(resolved == root or root in resolved.parents for root in roots)


def _under_trusted_project(path, cwd=None) -> bool:
    resolved = _resolve_path(path, cwd=cwd)
    return any(resolved == root or root in resolved.parents for root in _TRUSTED_PROJECT_ROOTS)


def _is_resource_guard_state_path(path, cwd=None) -> bool:
    if _RUN_ROOT is None:
        return False
    try:
        parts = _resolve_path(path, cwd=cwd).relative_to(_RUN_ROOT).parts
    except ValueError:
        return False
    return bool(parts and parts[0] in _RESOURCE_GUARD_STATE_NAMES)


def _is_run_control_stop_signal_path(path, cwd=None) -> bool:
    if _RUN_ROOT is None:
        return False
    try:
        parts = _resolve_path(path, cwd=cwd).relative_to(_RUN_ROOT).parts
    except ValueError:
        return False
    if parts == ("run_control",):
        return True
    if len(parts) != 2 or parts[0] != "run_control":
        return False
    name = parts[1]
    return name == "stop.json" or (name.startswith("stop.json.") and name.endswith(".tmp"))


def _is_resource_guard_caller() -> bool:
    try:
        frame = sys._getframe(1)
    except Exception:
        return False
    while frame is not None:
        filename = str(frame.f_code.co_filename).replace("\\", "/")
        if any(filename.endswith(suffix) for suffix in _RESOURCE_GUARD_MODULE_SUFFIXES):
            try:
                if _under_trusted_project(pathlib.Path(filename)):
                    return True
            except Exception:
                return False
        frame = frame.f_back
    return False


def _is_resource_guard_write_path(path, cwd=None) -> bool:
    return (
        _is_resource_guard_state_path(path, cwd=cwd)
        or _is_run_control_stop_signal_path(path, cwd=cwd)
    ) and _is_resource_guard_caller()


def _is_guard_warning_log_path(path, cwd=None) -> bool:
    if not _GUARD_WARNING_PATH:
        return False
    try:
        return _resolve_path(path, cwd=cwd) == _resolve_path(_GUARD_WARNING_PATH)
    except Exception:
        return False


def _is_guard_warning_caller() -> bool:
    try:
        frame = sys._getframe(1)
    except Exception:
        return False
    while frame is not None:
        if frame.f_code.co_name == "_warn_runtime":
            try:
                return pathlib.Path(frame.f_code.co_filename).resolve() == pathlib.Path(__file__).resolve()
            except Exception:
                return False
        frame = frame.f_back
    return False


def _is_guard_warning_log_write_path(path, cwd=None) -> bool:
    return _is_guard_warning_log_path(path, cwd=cwd) and _is_guard_warning_caller()


def _run_parts(path, cwd=None):
    if _RUN_ROOT is None:
        return None
    try:
        return _resolve_path(path, cwd=cwd).relative_to(_RUN_ROOT).parts
    except ValueError:
        return None


def _apparent_path(path, cwd=None):
    candidate = pathlib.Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = pathlib.Path(cwd or os.getcwd()) / candidate
    return pathlib.Path(os.path.normpath(str(candidate)))


def _apparent_run_parts(path, cwd=None):
    if _RUN_ROOT is None:
        return None
    try:
        return _apparent_path(path, cwd=cwd).relative_to(_RUN_ROOT).parts
    except ValueError:
        return None


def _is_peer_owned_write_path(path, cwd=None) -> bool:
    parts = _run_parts(path, cwd=cwd)
    if not parts:
        return False
    if parts[0] in {"variants", "results"} and len(parts) >= 2:
        if len(parts) >= 3 and parts[0] == "results" and re.fullmatch(r"gen_\d+", parts[1]):
            parts = parts[1:]
        return parts[1] == _PEER_ID or parts[1].startswith(_PEER_ID + "_")
    if parts[0] == "shared_findings" and len(parts) == 2:
        return parts[1].startswith(_PEER_ID + "_")
    if (
        len(parts) == 5
        and re.fullmatch(r"gen_\d+", parts[0])
        and parts[1] == "peers"
        and parts[2] == _PEER_ID
        and parts[3] == "dig"
        and parts[4] in {"contract_amendment.yaml", "contract_amendment.yml"}
    ):
        return True
    return len(parts) == 1 and parts[0] == f"notebook_{_PEER_ID}.json"


def _is_system_agenda_write_path(path, cwd=None) -> bool:
    if _PEER_ID not in _PI_AGENDA_WRITER_AGENTS:
        return False
    parts = _run_parts(path, cwd=cwd)
    if not parts or len(parts) != 2 or parts[0] != "agendas":
        return False
    return re.fullmatch(r"research_agenda_gen\d+\.ya?ml(?:\.candidate)?", parts[1]) is not None


def _is_safe_system_agenda_write_path(path, cwd=None) -> bool:
    if not _is_system_agenda_write_path(path, cwd=cwd):
        return False
    apparent = _apparent_path(path, cwd=cwd)
    try:
        st = apparent.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return False
    if stat.S_ISDIR(st.st_mode):
        return False
    if st.st_nlink > 1:
        return False
    return stat.S_ISREG(st.st_mode)


def _is_peer_mutable_memory_write_path(path, cwd=None) -> bool:
    if isinstance(path, int):
        return False
    parts = _run_parts(path, cwd=cwd)
    return _is_peer_mutable_memory_parts(parts)


def _is_apparent_peer_mutable_memory_write_path(path, cwd=None) -> bool:
    if isinstance(path, int):
        return False
    parts = _apparent_run_parts(path, cwd=cwd)
    return _is_peer_mutable_memory_parts(parts)


def _is_peer_mutable_memory_parts(parts) -> bool:
    return bool(
        parts
        and len(parts) == 5
        and re.fullmatch(r"gen_\d+", parts[0])
        and parts[1] == "peers"
        and parts[2] == _PEER_ID
        and parts[3] == "memory"
        and parts[4] in _PEER_MUTABLE_MEMORY_FILES
    )


def _is_safe_peer_mutable_memory_write_path(path, cwd=None) -> bool:
    if not _is_peer_mutable_memory_write_path(path, cwd=cwd):
        return False
    apparent = _apparent_path(path, cwd=cwd)
    try:
        st = apparent.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return False
    if st.st_nlink > 1:
        return False
    return stat.S_ISREG(st.st_mode)


def _is_safe_peer_owned_write_path(path, cwd=None) -> bool:
    if not _is_peer_owned_write_path(path, cwd=cwd):
        return False
    apparent = _apparent_path(path, cwd=cwd)
    try:
        st = apparent.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return False
    if stat.S_ISDIR(st.st_mode):
        return True
    if st.st_nlink > 1:
        return False
    return stat.S_ISREG(st.st_mode)


def _is_safe_allowed_write_path(path, cwd=None) -> bool:
    if not _under_allowed(path, cwd=cwd):
        return False
    apparent = _apparent_path(path, cwd=cwd)
    try:
        st = apparent.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return bool(
            _RUNTIME_TMP_APPARENT is not None
            and apparent == _RUNTIME_TMP_APPARENT
            and apparent.resolve().is_dir()
        )
    if stat.S_ISDIR(st.st_mode):
        return True
    if st.st_nlink > 1:
        return False
    return stat.S_ISREG(st.st_mode)


def _is_peer_owned_delete_path(path, cwd=None) -> bool:
    parts = _run_parts(path, cwd=cwd)
    if not parts or len(parts) < 2 or parts[0] not in {"variants", "results"}:
        return False
    if len(parts) >= 3 and parts[0] == "results" and re.fullmatch(r"gen_\d+", parts[1]):
        parts = parts[1:]
    return parts[1] == _PEER_ID or parts[1].startswith(_PEER_ID + "_")


def _is_runtime_temp_delete_path(path, cwd=None) -> bool:
    """Allow Python runtime IPC/temp cleanup without weakening project guards.

    Multiprocessing/resource_tracker may unlink POSIX shared-memory scratch
    files in /dev/shm with names like ``pym-*`` or ``psm_*``. Those files are
    interpreter-owned transient resources, not research artifacts. The allowlist
    is intentionally prefix-based and limited to runtime temp directories.
    """

    try:
        apparent = _apparent_path(path, cwd=cwd)
        apparent_parent = apparent.parent
        resolved_parent = apparent_parent.resolve()
    except Exception:
        return False
    for root in _RUNTIME_TEMP_DELETE_DIRS:
        try:
            relative = apparent_parent.relative_to(root)
            resolved_root = root.resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if any(
            part.startswith(prefix)
            for part in (relative.parts + (apparent.name,))
            for prefix in _RUNTIME_TEMP_DELETE_PREFIXES
        ) and (
            resolved_parent == resolved_root
            or resolved_root in resolved_parent.parents
            or _under_allowed(resolved_parent)
        ):
            return True
    return False


def _is_stdlib_tempfile_cleanup(path, cwd=None) -> bool:
    """Allow stdlib tempfile cleanup in temp roots, never in protected state."""

    if not _TEMPFILE_CODE_OBJECTS:
        return False
    try:
        apparent = _apparent_path(path, cwd=cwd)
        apparent_parent = apparent.parent
        resolved_parent = apparent_parent.resolve()
        st = apparent.lstat()
    except (FileNotFoundError, OSError):
        return False
    if not any(
        (apparent_parent == root or root in apparent_parent.parents)
        and (
            resolved_parent == root.resolve()
            or root.resolve() in resolved_parent.parents
            or _under_allowed(resolved_parent)
        )
        for root in _RUNTIME_TEMP_DELETE_DIRS
    ):
        return False
    if _under_any(resolved_parent, _PROTECTED_ROOTS) and not _under_allowed(resolved_parent):
        return False
    if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
        return False
    try:
        frame = sys._getframe(1)
    except Exception:
        return False
    while frame is not None:
        if frame.f_code in _TEMPFILE_CODE_OBJECTS:
            return True
        frame = frame.f_back
    return False


def _check(path, op: str, cwd=None) -> None:
    if _is_runtime_temp_delete_path(path, cwd=cwd) or _is_stdlib_tempfile_cleanup(path, cwd=cwd):
        return
    if _is_resource_guard_write_path(path, cwd=cwd):
        return
    if not _under_allowed(path, cwd=cwd) and not _is_peer_owned_delete_path(path, cwd=cwd):
        raise PermissionError(
            f"Praxist delete guard blocked Python {op} outside $PRAXIST_PEER_WORKSPACE/owned experiment paths: {path}"
        )


def _check_peer_memory_safe_if_apparent(path, op: str, cwd=None) -> None:
    if isinstance(path, int):
        return
    if _is_apparent_peer_mutable_memory_write_path(
        path, cwd=cwd
    ) and not _is_safe_peer_mutable_memory_write_path(path, cwd=cwd):
        raise PermissionError(
            f"Praxist delete guard blocked Python {op} through unsafe peer memory file: {path}"
        )


def _protected_existing_path(path, cwd=None):
    if isinstance(path, int):
        path = _fd_path(path)
        if path is None:
            return None
    resolved = _resolve_path(path, cwd=cwd)
    if _under_allowed(resolved):
        return None
    if _is_peer_owned_write_path(resolved):
        return None
    if _is_safe_system_agenda_write_path(resolved):
        return None
    if _is_resource_guard_write_path(resolved):
        return None
    if _is_guard_warning_log_write_path(resolved):
        return None
    if _under_any(resolved, _PROTECTED_ROOTS):
        return resolved
    return None


def _check_run_write(path, op: str, cwd=None) -> None:
    _check_peer_memory_safe_if_apparent(path, op, cwd=cwd)
    if not isinstance(path, int):
        if op in _PEER_MUTABLE_MEMORY_DIRECT_WRITE_OPS and _is_safe_peer_mutable_memory_write_path(
            path, cwd=cwd
        ):
            return
        if _under_allowed(path, cwd=cwd) and not _is_safe_allowed_write_path(path, cwd=cwd):
            raise PermissionError(
                f"Praxist delete guard blocked Python {op} through unsafe peer workspace file: {path}"
            )
        if _is_peer_owned_write_path(path, cwd=cwd) and not _is_safe_peer_owned_write_path(
            path, cwd=cwd
        ):
            raise PermissionError(
                f"Praxist delete guard blocked Python {op} through unsafe peer-owned file: {path}"
            )
    protected = _protected_existing_path(path, cwd=cwd)
    if protected is None:
        return
    raise PermissionError(
        f"Praxist delete guard blocked Python {op} against existing protected file outside "
        f"$PRAXIST_PEER_WORKSPACE: {protected}"
    )


def _check_protected_create(path, op: str, cwd=None) -> None:
    _check_peer_memory_safe_if_apparent(path, op, cwd=cwd)
    if op in _PEER_MUTABLE_MEMORY_DIRECT_WRITE_OPS and _is_safe_peer_mutable_memory_write_path(
        path, cwd=cwd
    ):
        return
    resolved = _resolve_path(path, cwd=cwd)
    if _under_allowed(resolved):
        if not _is_safe_allowed_write_path(resolved):
            raise PermissionError(
                f"Praxist delete guard blocked Python {op} through unsafe peer workspace file: {resolved}"
            )
        return
    if _is_peer_owned_write_path(resolved):
        if not _is_safe_peer_owned_write_path(resolved):
            raise PermissionError(
                f"Praxist delete guard blocked Python {op} through unsafe peer-owned file: {resolved}"
            )
        return
    if _is_system_agenda_write_path(resolved):
        if not _is_safe_system_agenda_write_path(resolved):
            raise PermissionError(
                f"Praxist delete guard blocked Python {op} through unsafe agenda file: {resolved}"
            )
        return
    if _is_resource_guard_write_path(resolved):
        return
    if _is_guard_warning_log_write_path(resolved):
        return
    if _under_any(resolved, _PROTECTED_ROOTS):
        raise PermissionError(
            f"Praxist delete guard blocked Python {op} target inside protected state "
            f"outside $PRAXIST_PEER_WORKSPACE: {resolved}"
        )


def _check_link_source(path, op: str, cwd=None) -> None:
    resolved = _resolve_path(path, cwd=cwd)
    if _under_allowed(resolved) or _is_peer_owned_delete_path(resolved):
        return
    if _is_resource_guard_write_path(resolved):
        return
    if _under_any(resolved, _PROTECTED_ROOTS):
        raise PermissionError(
            f"Praxist delete guard blocked Python {op} from protected state outside "
            f"$PRAXIST_PEER_WORKSPACE: {resolved}"
        )


def _check_replace_destination(path, op: str, cwd=None) -> None:
    _check_peer_memory_safe_if_apparent(path, op, cwd=cwd)
    if _is_apparent_peer_mutable_memory_write_path(path, cwd=cwd):
        raise PermissionError(
            f"Praxist delete guard blocked Python {op} indirect replace into peer memory state: {path}"
        )
    resolved = _resolve_path(path, cwd=cwd)
    if _under_allowed(resolved) or _is_peer_owned_write_path(resolved):
        return
    if _is_system_agenda_write_path(resolved):
        if not _is_safe_system_agenda_write_path(resolved):
            raise PermissionError(
                f"Praxist delete guard blocked Python {op} through unsafe agenda file: {resolved}"
            )
        return
    if _is_resource_guard_write_path(resolved):
        return
    if _under_any(resolved, _PROTECTED_ROOTS):
        raise PermissionError(
            f"Praxist delete guard blocked Python {op} outside owned paths: {resolved}"
        )


def _check_output_arg(value, op: str) -> None:
    if value is None:
        return
    candidate = value
    if hasattr(value, "name") and not isinstance(value, (str, bytes, os.PathLike)):
        try:
            candidate = value.name
        except Exception:
            return
    if isinstance(candidate, bytes):
        candidate = os.fsdecode(candidate)
    if isinstance(candidate, (str, os.PathLike)):
        _check_protected_create(candidate, op)


def _fd_path(fd):
    try:
        return pathlib.Path(os.readlink(f"/proc/self/fd/{int(fd)}"))
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            raw = fcntl.fcntl(int(fd), 50, b"\0" * 1024)
            path = raw.split(b"\0", 1)[0].decode()
            return pathlib.Path(path) if path else None
        except Exception:
            return None
    return None


def _dir_fd_cwd(fd):
    if fd is None:
        return None
    return _fd_path(fd)


def _flags_write(flags) -> bool:
    raw = int(flags or 0)
    if raw < 0:
        return False
    access_mode = raw & os.O_ACCMODE
    write_flags = os.O_WRONLY | os.O_RDWR
    create_flags = 0
    for name in ("O_CREAT", "O_TRUNC", "O_APPEND"):
        create_flags |= getattr(os, name, 0)
    return bool(access_mode & write_flags or raw & create_flags)


def _audit(event, args):
    try:
        if event == "open" and args:
            target = args[0]
            if not isinstance(target, (str, bytes, os.PathLike)):
                return None
            mode = str(args[1] or "") if len(args) > 1 else ""
            flags = args[2] if len(args) > 2 else 0
            writes = any(flag in mode for flag in ("w", "a", "x", "+")) or _flags_write(flags)
            if writes:
                _check_protected_create(target, "audit open/write")
        elif event in {"os.remove", "os.unlink", "os.rmdir"} and args:
            _check(
                args[0],
                f"audit {event}",
                cwd=_dir_fd_cwd(args[1] if len(args) > 1 else None),
            )
        elif event in {"os.rename", "os.replace"} and len(args) >= 2:
            _check(
                args[0],
                f"audit {event} source",
                cwd=_dir_fd_cwd(args[2] if len(args) > 2 else None),
            )
            _check_replace_destination(
                args[1],
                f"audit {event} destination",
                cwd=_dir_fd_cwd(args[3] if len(args) > 3 else None),
            )
    except PermissionError:
        raise
    except Exception:
        raise PermissionError(f"Praxist delete guard blocked unverified audit event: {event}")
    return None


sys.addaudithook(_audit)


_orig_rmtree = shutil.rmtree
_orig_move = shutil.move
_orig_open = builtins.open
_orig_io_open = io.open
_orig__io_open = _io.open
_orig_os_open = os.open
_orig_putenv = os.putenv
_orig_unsetenv = getattr(os, "unsetenv", None)
_orig_environ_setitem = type(os.environ).__setitem__
_orig_environ_delitem = type(os.environ).__delitem__
_orig_mkdir = os.mkdir
_orig_makedirs = os.makedirs
_orig_mknod = getattr(os, "mknod", None)
_orig_mkfifo = getattr(os, "mkfifo", None)
_orig_remove = os.remove
_orig_unlink = os.unlink
_orig_rmdir = os.rmdir
_orig_rename = os.rename
_orig_replace = os.replace
_orig_truncate = os.truncate
_orig_ftruncate = os.ftruncate
_orig_chmod = os.chmod
_orig_fchmod = getattr(os, "fchmod", None)
_orig_link = os.link
_orig_symlink = os.symlink
_orig_chown = getattr(os, "chown", None)
_orig_lchown = getattr(os, "lchown", None)
_orig_fchown = getattr(os, "fchown", None)
_orig_posix_open = getattr(posix, "open", None)
_orig_posix_mkdir = getattr(posix, "mkdir", None)
_orig_posix_mknod = getattr(posix, "mknod", None)
_orig_posix_mkfifo = getattr(posix, "mkfifo", None)
_orig_posix_unlink = posix.unlink
_orig_posix_rmdir = posix.rmdir
_orig_posix_chmod = getattr(posix, "chmod", None)
_orig_path_unlink = pathlib.Path.unlink
_orig_path_rmdir = pathlib.Path.rmdir
_orig_path_rename = pathlib.Path.rename
_orig_path_replace = pathlib.Path.replace
_orig_path_open = pathlib.Path.open
_orig_path_write_text = pathlib.Path.write_text
_orig_path_write_bytes = pathlib.Path.write_bytes
_orig_path_chmod = pathlib.Path.chmod
_orig_path_mkdir = pathlib.Path.mkdir
_orig_copyfile = shutil.copyfile
_orig_copy = shutil.copy
_orig_copy2 = shutil.copy2
_orig_copytree = shutil.copytree
_orig_sqlite_connect = sqlite3.connect
_orig_import = builtins.__import__
_orig_ctypes_cdll = ctypes.CDLL
_orig_ctypes_pydll = ctypes.PyDLL
_orig_ctypes_pydll_loadlibrary = ctypes.pydll.LoadLibrary
_orig__ctypes_dlopen = getattr(_ctypes, "dlopen", None)
_orig_popen = subprocess.Popen
_orig_os_system = os.system
_orig_spawnv = getattr(os, "spawnv", None)
_orig_spawnve = getattr(os, "spawnve", None)
_orig_spawnvp = getattr(os, "spawnvp", None)
_orig_spawnvpe = getattr(os, "spawnvpe", None)
_orig_execv = getattr(os, "execv", None)
_orig_execve = getattr(os, "execve", None)
_orig_execvp = getattr(os, "execvp", None)
_orig_execvpe = getattr(os, "execvpe", None)
_orig_posix_spawn = getattr(os, "posix_spawn", None)
_orig_posix_spawnp = getattr(os, "posix_spawnp", None)
_ORIG_BY_WRAPPER = {}


class _OriginalCall:
    __slots__ = ("_orig", "_code")

    def __init__(self, orig, code):
        object.__setattr__(self, "_orig", orig)
        object.__setattr__(self, "_code", code)

    def __call__(self, *args, **kwargs):
        try:
            caller_code = sys._getframe(1).f_code
        except Exception as exc:
            raise PermissionError("Praxist delete guard blocked unverified original call") from exc
        if caller_code is not object.__getattribute__(self, "_code"):
            raise PermissionError("Praxist delete guard blocked direct original-function access")
        return object.__getattribute__(self, "_orig")(*args, **kwargs)


def _bind_orig(wrapper, orig):
    _ORIG_BY_WRAPPER[id(wrapper)] = _OriginalCall(orig, wrapper.__code__)
    return wrapper


def _orig_for(wrapper):
    orig = _ORIG_BY_WRAPPER.get(id(wrapper))
    if orig is None:
        raise PermissionError("Praxist delete guard internal original-function registry unavailable")
    return orig


def _warning_open(*args, **kwargs):
    return _orig_for(_warning_open)(*args, **kwargs)


def _warning_makedirs(*args, **kwargs):
    return _orig_for(_warning_makedirs)(*args, **kwargs)


def _warn_runtime(rule_id, message, command=None):
    if not _GUARD_WARNING_PATH:
        return
    try:
        path = pathlib.Path(_GUARD_WARNING_PATH)
        _warning_makedirs(str(path.parent), exist_ok=True)
        payload = {
            "schema_version": "praxist.runtime_guard.warning.v1",
            "timestamp": time.time(),
            "severity": "warning",
            "rule_id": str(rule_id),
            "message": str(message),
            "effect": "allowed",
        }
        if command is not None:
            payload["command"] = str(command)
        with _warning_open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        return


def _guarded_rmtree(path, *args, **kwargs):
    _check(path, "rmtree")
    return _orig_for(_guarded_rmtree)(path, *args, **kwargs)


def _mode_writes(mode) -> bool:
    raw = str(mode or "r")
    return any(flag in raw for flag in ("w", "a", "x", "+"))


def _guarded_open(file, mode="r", *args, **kwargs):
    if _mode_writes(mode):
        _check_run_write(file, "open/write")
    return _orig_for(_guarded_open)(file, mode, *args, **kwargs)


def _guarded_io_open(file, mode="r", *args, **kwargs):
    if _mode_writes(mode):
        _check_run_write(file, "io.open/write")
    return _orig_for(_guarded_io_open)(file, mode, *args, **kwargs)


def _guarded__io_open(file, mode="r", *args, **kwargs):
    if _mode_writes(mode):
        _check_run_write(file, "_io.open/write")
    return _orig_for(_guarded__io_open)(file, mode, *args, **kwargs)


def _guarded_os_open(path, flags, *args, **kwargs):
    if _flags_write(flags):
        _check_run_write(path, "os.open/write", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_os_open)(path, flags, *args, **kwargs)


def _guarded_putenv(key, value, *args, **kwargs):
    if str(key) in _GUARD_ENV_VARS:
        _warn_runtime(
            "guard_env_mutation",
            f"Praxist runtime guard allowed mutation of guard-related environment variable {key!r}.",
        )
    return _orig_for(_guarded_putenv)(key, value, *args, **kwargs)


def _guarded_unsetenv(key, *args, **kwargs):
    if str(key) in _GUARD_ENV_VARS:
        _warn_runtime(
            "guard_env_unset",
            f"Praxist runtime guard allowed removal of guard-related environment variable {key!r}.",
        )
    return _orig_for(_guarded_unsetenv)(key, *args, **kwargs)


def _guarded_environ_setitem(self, key, value):
    if str(key) in _GUARD_ENV_VARS:
        _warn_runtime(
            "guard_env_mutation",
            f"Praxist runtime guard allowed mutation of guard-related environment variable {key!r}.",
        )
    return _orig_for(_guarded_environ_setitem)(self, key, value)


def _guarded_environ_delitem(self, key):
    if str(key) in _GUARD_ENV_VARS:
        _warn_runtime(
            "guard_env_unset",
            f"Praxist runtime guard allowed removal of guard-related environment variable {key!r}.",
        )
    return _orig_for(_guarded_environ_delitem)(self, key)


def _guarded_posix_open(path, flags, *args, **kwargs):
    if _flags_write(flags):
        _check_run_write(path, "posix.open/write", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_posix_open)(path, flags, *args, **kwargs)


def _guarded_mkdir(path, mode=0o777, *args, **kwargs):
    _check_protected_create(path, "mkdir", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_mkdir)(path, mode, *args, **kwargs)


def _guarded_makedirs(name, mode=0o777, exist_ok=False):
    if exist_ok:
        try:
            existing = pathlib.Path(name)
            if existing.exists() and existing.is_dir():
                return _orig_for(_guarded_makedirs)(name, mode, exist_ok=exist_ok)
        except Exception:
            pass
    _check_protected_create(name, "makedirs")
    return _orig_for(_guarded_makedirs)(name, mode, exist_ok=exist_ok)


def _guarded_mknod(path, *args, **kwargs):
    _check_protected_create(path, "mknod", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_mknod)(path, *args, **kwargs)


def _guarded_mkfifo(path, *args, **kwargs):
    _check_protected_create(path, "mkfifo", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_mkfifo)(path, *args, **kwargs)


def _guarded_posix_mkdir(path, mode=0o777, *args, **kwargs):
    _check_protected_create(path, "posix.mkdir", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_posix_mkdir)(path, mode, *args, **kwargs)


def _guarded_posix_mknod(path, *args, **kwargs):
    _check_protected_create(path, "posix.mknod", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_posix_mknod)(path, *args, **kwargs)


def _guarded_posix_mkfifo(path, *args, **kwargs):
    _check_protected_create(path, "posix.mkfifo", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_posix_mkfifo)(path, *args, **kwargs)


def _guarded_move(src, dst, *args, **kwargs):
    _check(src, "shutil.move source")
    _check_replace_destination(dst, "shutil.move destination")
    return _orig_for(_guarded_move)(src, dst, *args, **kwargs)


def _guarded_remove(path, *args, **kwargs):
    _check(path, "remove", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_remove)(path, *args, **kwargs)


def _guarded_unlink(path, *args, **kwargs):
    _check(path, "unlink", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_unlink)(path, *args, **kwargs)


def _guarded_rmdir(path, *args, **kwargs):
    _check(path, "rmdir", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_rmdir)(path, *args, **kwargs)


def _guarded_rename(src, dst, *args, **kwargs):
    _check(src, "rename source", cwd=_dir_fd_cwd(kwargs.get("src_dir_fd")))
    _check_replace_destination(dst, "rename destination", cwd=_dir_fd_cwd(kwargs.get("dst_dir_fd")))
    return _orig_for(_guarded_rename)(src, dst, *args, **kwargs)


def _guarded_replace(src, dst, *args, **kwargs):
    _check(src, "replace source", cwd=_dir_fd_cwd(kwargs.get("src_dir_fd")))
    _check_replace_destination(dst, "replace destination", cwd=_dir_fd_cwd(kwargs.get("dst_dir_fd")))
    return _orig_for(_guarded_replace)(src, dst, *args, **kwargs)


def _guarded_truncate(path, length, *args, **kwargs):
    _check_run_write(path, "truncate")
    return _orig_for(_guarded_truncate)(path, length, *args, **kwargs)


def _guarded_ftruncate(fd, length, *args, **kwargs):
    target = _fd_path(fd)
    if target is not None:
        _check_run_write(target, "ftruncate")
    return _orig_for(_guarded_ftruncate)(fd, length, *args, **kwargs)


def _guarded_chmod(path, mode, *args, **kwargs):
    _check_run_write(path, "chmod", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_chmod)(path, mode, *args, **kwargs)


def _guarded_fchmod(fd, mode, *args, **kwargs):
    target = _fd_path(fd)
    if target is not None:
        _check_run_write(target, "fchmod")
    return _orig_for(_guarded_fchmod)(fd, mode, *args, **kwargs)


def _guarded_posix_chmod(path, mode, *args, **kwargs):
    _check_run_write(path, "posix.chmod", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_posix_chmod)(path, mode, *args, **kwargs)


def _guarded_chown(path, uid, gid, *args, **kwargs):
    _check_run_write(path, "chown", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_chown)(path, uid, gid, *args, **kwargs)


def _guarded_lchown(path, uid, gid, *args, **kwargs):
    _check_run_write(path, "lchown")
    return _orig_for(_guarded_lchown)(path, uid, gid, *args, **kwargs)


def _guarded_fchown(fd, uid, gid, *args, **kwargs):
    target = _fd_path(fd)
    if target is not None:
        _check_run_write(target, "fchown")
    return _orig_for(_guarded_fchown)(fd, uid, gid, *args, **kwargs)


def _guarded_link(src, dst, *args, **kwargs):
    _check_link_source(src, "link source", cwd=_dir_fd_cwd(kwargs.get("src_dir_fd")))
    _check_protected_create(dst, "link destination", cwd=_dir_fd_cwd(kwargs.get("dst_dir_fd")))
    return _orig_for(_guarded_link)(src, dst, *args, **kwargs)


def _guarded_symlink(src, dst, *args, **kwargs):
    _check_protected_create(dst, "symlink destination", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_symlink)(src, dst, *args, **kwargs)


def _guarded_posix_unlink(path, *args, **kwargs):
    _check(path, "posix.unlink", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_posix_unlink)(path, *args, **kwargs)


def _guarded_posix_rmdir(path, *args, **kwargs):
    _check(path, "posix.rmdir", cwd=_dir_fd_cwd(kwargs.get("dir_fd")))
    return _orig_for(_guarded_posix_rmdir)(path, *args, **kwargs)


def _guarded_path_unlink(self, *args, **kwargs):
    _check(self, "Path.unlink")
    return _orig_for(_guarded_path_unlink)(self, *args, **kwargs)


def _guarded_path_rmdir(self, *args, **kwargs):
    _check(self, "Path.rmdir")
    return _orig_for(_guarded_path_rmdir)(self, *args, **kwargs)


def _guarded_path_rename(self, target, *args, **kwargs):
    _check(self, "Path.rename source")
    _check_replace_destination(target, "Path.rename destination")
    return _orig_for(_guarded_path_rename)(self, target, *args, **kwargs)


def _guarded_path_replace(self, target, *args, **kwargs):
    _check(self, "Path.replace source")
    _check_replace_destination(target, "Path.replace destination")
    return _orig_for(_guarded_path_replace)(self, target, *args, **kwargs)


def _guarded_path_open(self, mode="r", *args, **kwargs):
    if _mode_writes(mode):
        _check_run_write(self, "Path.open/write")
    return _orig_for(_guarded_path_open)(self, mode, *args, **kwargs)


def _guarded_path_write_text(self, *args, **kwargs):
    _check_run_write(self, "Path.write_text")
    return _orig_for(_guarded_path_write_text)(self, *args, **kwargs)


def _guarded_path_write_bytes(self, *args, **kwargs):
    _check_run_write(self, "Path.write_bytes")
    return _orig_for(_guarded_path_write_bytes)(self, *args, **kwargs)


def _guarded_path_chmod(self, mode, *args, **kwargs):
    _check_run_write(self, "Path.chmod")
    return _orig_for(_guarded_path_chmod)(self, mode, *args, **kwargs)


def _guarded_path_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    if exist_ok:
        try:
            if self.exists() and self.is_dir():
                return _orig_for(_guarded_path_mkdir)(self, mode=mode, parents=parents, exist_ok=exist_ok)
        except Exception:
            pass
    _check_protected_create(self, "Path.mkdir")
    return _orig_for(_guarded_path_mkdir)(self, mode=mode, parents=parents, exist_ok=exist_ok)


def _guarded_copyfile(src, dst, *args, **kwargs):
    _check_run_write(dst, "shutil.copyfile destination")
    return _orig_for(_guarded_copyfile)(src, dst, *args, **kwargs)


def _guarded_copy(src, dst, *args, **kwargs):
    _check_run_write(dst, "shutil.copy destination")
    return _orig_for(_guarded_copy)(src, dst, *args, **kwargs)


def _guarded_copy2(src, dst, *args, **kwargs):
    _check_run_write(dst, "shutil.copy2 destination")
    return _orig_for(_guarded_copy2)(src, dst, *args, **kwargs)


def _guarded_copytree(src, dst, *args, **kwargs):
    _check_protected_create(dst, "shutil.copytree destination")
    return _orig_for(_guarded_copytree)(src, dst, *args, **kwargs)


def _guarded_sqlite_connect(database, *args, **kwargs):
    if isinstance(database, (str, bytes, os.PathLike)):
        db_text = os.fsdecode(database)
        uri = bool(kwargs.get("uri", False))
        db_path = db_text
        readonly = False
        if uri and db_text.startswith("file:"):
            parsed = urlparse(db_text)
            query = parse_qs(parsed.query, keep_blank_values=True)
            readonly = query.get("mode", [""])[0] == "ro"
            db_path = unquote(parsed.path)
        if not readonly:
            _check_run_write(db_path.split("?", 1)[0], "sqlite3.connect")
    return _orig_for(_guarded_sqlite_connect)(database, *args, **kwargs)


_PATCHED_NATIVE_WRITERS = set()


def _patch_callable_output(owner, attr, *, index=0, keywords=(), op=None):
    func = getattr(owner, attr, None)
    key = (id(owner), attr)
    if key in _PATCHED_NATIVE_WRITERS or not callable(func):
        return

    def _wrapped(*args, **kwargs):
        if len(args) > index:
            _check_output_arg(args[index], op or attr)
        for keyword in keywords:
            if keyword in kwargs:
                _check_output_arg(kwargs.get(keyword), op or attr)
        return func(*args, **kwargs)

    try:
        setattr(_wrapped, "_praxist_delete_guard_wrapped", True)
        setattr(owner, attr, _wrapped)
        _PATCHED_NATIVE_WRITERS.add(key)
    except Exception:
        return


def _patch_native_writers(module):
    try:
        name = getattr(module, "__name__", "")
    except Exception:
        return
    if name == "numpy":
        _patch_callable_output(module, "save", index=0, op="numpy.save")
        _patch_callable_output(module, "savez", index=0, op="numpy.savez")
        _patch_callable_output(module, "savez_compressed", index=0, op="numpy.savez_compressed")
        ndarray = getattr(module, "ndarray", None)
        tofile = getattr(ndarray, "tofile", None) if ndarray is not None else None
        key = (id(ndarray), "tofile")
        if callable(tofile) and key not in _PATCHED_NATIVE_WRITERS:
            def _guarded_ndarray_tofile(self, fid, *args, **kwargs):
                _check_output_arg(fid, "numpy.ndarray.tofile")
                return tofile(self, fid, *args, **kwargs)
            try:
                setattr(ndarray, "tofile", _guarded_ndarray_tofile)
                _PATCHED_NATIVE_WRITERS.add(key)
            except Exception:
                pass
        fmt = getattr(getattr(module, "lib", None), "format", None)
        if fmt is not None:
            _patch_callable_output(fmt, "open_memmap", index=0, op="numpy.open_memmap")
    elif name == "pandas":
        dataframe = getattr(module, "DataFrame", None)
        series = getattr(module, "Series", None)
        if dataframe is not None:
            _patch_callable_output(dataframe, "to_parquet", index=1, keywords=("path",), op="pandas.DataFrame.to_parquet")
            _patch_callable_output(dataframe, "to_csv", index=1, keywords=("path_or_buf",), op="pandas.DataFrame.to_csv")
            _patch_callable_output(dataframe, "to_pickle", index=1, keywords=("path",), op="pandas.DataFrame.to_pickle")
        if series is not None:
            _patch_callable_output(series, "to_csv", index=1, keywords=("path_or_buf",), op="pandas.Series.to_csv")
        _patch_callable_output(module, "to_pickle", index=1, keywords=("filepath_or_buffer",), op="pandas.to_pickle")
    elif name == "joblib":
        _patch_callable_output(module, "dump", index=1, op="joblib.dump")
    elif name == "torch":
        _patch_callable_output(module, "save", index=1, op="torch.save")
    elif name == "pyarrow":
        _patch_callable_output(module, "OSFile", index=0, op="pyarrow.OSFile")
        _patch_callable_output(module, "memory_map", index=0, op="pyarrow.memory_map")
    elif name == "pyarrow.parquet":
        _patch_callable_output(module, "write_table", index=1, keywords=("where",), op="pyarrow.parquet.write_table")
        _patch_callable_output(module, "write_to_dataset", index=1, keywords=("root_path",), op="pyarrow.parquet.write_to_dataset")
        _patch_callable_output(module, "ParquetWriter", index=0, keywords=("where",), op="pyarrow.parquet.ParquetWriter")


def _patch_imported_native_writers(name):
    module_names = {str(name).split(".", 1)[0], str(name)}
    module_names.update({"numpy", "pandas", "joblib", "torch", "pyarrow", "pyarrow.parquet"})
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is not None:
            _patch_native_writers(module)


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root_name = str(name).split(".", 1)[0]
    if root_name in {"cffi", "_cffi_backend"}:
        raise ImportError("Praxist delete guard blocked cffi native loader access")
    module = _orig_for(_guarded_import)(name, globals, locals, fromlist, level)
    _patch_imported_native_writers(name)
    return module


class _BlockPeerExtensionImporter:
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        origin = getattr(spec, "origin", None) if spec is not None else None
        if origin and str(origin).endswith((".so", ".pyd", ".dylib")):
            resolved = _resolve_path(origin)
            if _under_allowed(resolved) or _is_peer_owned_write_path(resolved):
                raise ImportError(f"Praxist delete guard blocked peer-local extension import: {origin}")
        return None


def _guarded_ctypes_cdll(name, *args, **kwargs):
    raw = "" if name is None else str(name)
    lowered = raw.lower()
    if (
        name is None
        or "libc.so" in lowered
        or "libdl.so" in lowered
        or lowered.endswith("/libc.dylib")
        or lowered.endswith("/libdl.dylib")
    ):
        raise PermissionError("Praxist delete guard blocked ctypes access to process/libc symbols")
    if raw and not _is_trusted_library_path(raw):
        raise PermissionError("Praxist delete guard blocked ctypes loading of untrusted/peer-local library")
    return _orig_for(_guarded_ctypes_cdll)(name, *args, **kwargs)


def _guarded_ctypes_pydll(name, *args, **kwargs):
    raw = "" if name is None else str(name)
    lowered = raw.lower()
    if (
        name is None
        or "libc.so" in lowered
        or "libdl.so" in lowered
        or lowered.endswith("/libc.dylib")
        or lowered.endswith("/libdl.dylib")
    ):
        raise PermissionError("Praxist delete guard blocked ctypes PyDLL access to process/libc symbols")
    if raw and not _is_trusted_library_path(raw):
        raise PermissionError("Praxist delete guard blocked ctypes PyDLL loading of untrusted/peer-local library")
    return _orig_for(_guarded_ctypes_pydll)(name, *args, **kwargs)


def _guarded_ctypes_pydll_loadlibrary(name, *args, **kwargs):
    raw = "" if name is None else str(name)
    lowered = raw.lower()
    if (
        name is None
        or "libc.so" in lowered
        or "libdl.so" in lowered
        or lowered.endswith("/libc.dylib")
        or lowered.endswith("/libdl.dylib")
    ):
        raise PermissionError("Praxist delete guard blocked ctypes.pydll LoadLibrary access to process/libc symbols")
    if raw and not _is_trusted_library_path(raw):
        raise PermissionError("Praxist delete guard blocked ctypes.pydll loading of untrusted/peer-local library")
    return _orig_for(_guarded_ctypes_pydll_loadlibrary)(name, *args, **kwargs)


def _guarded__ctypes_dlopen(name, *args, **kwargs):
    raw = "" if name is None else str(name)
    lowered = raw.lower()
    if (
        name is None
        or "libc.so" in lowered
        or "libdl.so" in lowered
        or lowered.endswith("/libc.dylib")
        or lowered.endswith("/libdl.dylib")
    ):
        raise PermissionError("Praxist delete guard blocked _ctypes access to process/libc symbols")
    if raw and not _is_trusted_library_path(raw):
        raise PermissionError("Praxist delete guard blocked _ctypes loading of untrusted/peer-local library")
    return _orig_for(_guarded__ctypes_dlopen)(name, *args, **kwargs)


_GUARD_ENV_VARS = set(__PRAXIST_PYTHON_GUARD_ENV_KEYS__)
_SHELLS = {"sh", "dash", "bash", "zsh", "ksh"}
_READ_ONLY_SYSTEM_BASENAMES = {
    "basename", "cat", "cut", "date", "df", "dirname", "du", "echo", "env",
    "file", "find", "free", "grep", "head", "hostname", "id", "ls", "lscpu",
    "nvidia-smi", "pgrep", "pidof", "printf", "ps", "pwd", "readlink",
    "realpath", "rg", "seq", "sleep", "stat", "tail", "test", "timeout",
    "tr", "true", "false", "uname", "uptime", "wc", "whoami", "which",
    "xargs",
}
_CLASSIFIED_MUTATING_BASENAMES = {
    "awk", "cc", "chmod", "chgrp", "chown", "clang", "clang++", "cp", "curl",
    "dd", "fallocate", "g++", "gawk", "gcc", "git", "install", "ld", "ln",
    "make", "mawk", "mkdir", "mv", "ninja", "rsync", "sed", "shred", "sort",
    "tar", "tee", "touch", "truncate", "wget", "zip",
}
_BROAD_KILL_BASENAMES = {"kill", "pkill", "killall"}
_TRUSTED_SYSTEM_DIRS = [
    pathlib.Path(p)
    for p in (
        "/bin",
        "/usr/bin",
        "/usr/sbin",
        "/sbin",
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/opt/homebrew/Cellar",
        "/opt/homebrew/opt",
        "/usr/local/bin",
        "/usr/local/sbin",
        "/usr/local/Cellar",
        "/usr/local/opt",
    )
]
_TRUSTED_PYTHON_LIBRARY_ROOTS = []
for _raw_root in (
    getattr(sys, "prefix", ""),
    getattr(sys, "base_prefix", ""),
    getattr(sys, "exec_prefix", ""),
    getattr(sys, "base_exec_prefix", ""),
):
    if _raw_root:
        _root = pathlib.Path(_raw_root).expanduser().resolve()
        if _root not in _TRUSTED_PYTHON_LIBRARY_ROOTS:
            _TRUSTED_PYTHON_LIBRARY_ROOTS.append(_root)
for _raw_path in sys.path:
    if "site-packages" in str(_raw_path) or "dist-packages" in str(_raw_path):
        _root = pathlib.Path(_raw_path).expanduser().resolve()
        if _root not in _TRUSTED_PYTHON_LIBRARY_ROOTS:
            _TRUSTED_PYTHON_LIBRARY_ROOTS.append(_root)
_TRUSTED_PYTHON_LIBRARY_ROOTS = tuple(_TRUSTED_PYTHON_LIBRARY_ROOTS)
_IMMUTABLE_GUARD_ENV = {
    key: os.environ.get(key)
    for key in __PRAXIST_IMMUTABLE_GUARD_ENV_KEYS__
}


def _is_trusted_system_executable(path) -> bool:
    try:
        resolved = _resolve_path(path)
    except Exception:
        return False
    return any(resolved == root or root in resolved.parents for root in _TRUSTED_SYSTEM_DIRS)


def _is_trusted_library_path(raw) -> bool:
    text = str(raw or "")
    if not text or "/" not in text:
        return True
    try:
        resolved = _resolve_path(text)
    except Exception:
        return False
    if (
        _under_allowed(resolved)
        or _is_peer_owned_write_path(resolved)
        or _is_safe_system_agenda_write_path(resolved)
    ):
        return False
    if _is_trusted_system_executable(resolved):
        return True
    if _under_any(resolved, _TRUSTED_PYTHON_LIBRARY_ROOTS):
        return True
    return resolved.exists() and _under_any(resolved, _PROTECTED_ROOTS)


def _argv_executable_path(args, cwd=None):
    if not isinstance(args, (list, tuple)) or not args:
        return None
    exe = str(args[0])
    if not exe:
        return None
    if "/" in exe:
        try:
            return _resolve_path(exe, cwd=cwd)
        except Exception:
            return None
    for raw_dir in os.environ.get("PATH", "").split(os.pathsep):
        if not raw_dir:
            continue
        candidate = pathlib.Path(raw_dir) / exe
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if resolved.exists():
            return resolved
    return None


def _is_binary_executable(path) -> bool:
    try:
        with _orig_for(_guarded_open)(path, "rb") as handle:
            head = handle.read(4096)
    except Exception:
        return False
    return (
        b"\x00" in head
        or head.startswith(b"\x7fELF")
        or head.startswith(b"\xcf\xfa\xed\xfe")
        or head.startswith(b"\xfe\xed\xfa\xcf")
        or head.startswith(b"\xca\xfe\xba\xbe")
    )


def _check_peer_binary_execution(args, cwd=None) -> None:
    path = _argv_executable_path(args, cwd=cwd)
    if path is None:
        return
    if (_under_allowed(path) or _is_peer_owned_write_path(path)) and _is_binary_executable(path):
        raise PermissionError(f"Praxist delete guard blocked peer-local binary executable: {path}")


def _env_payload(args):
    if not isinstance(args, (list, tuple)) or not args:
        return None, False
    if pathlib.Path(str(args[0])).name != "env":
        return None, False
    stripped = False
    i = 1
    while i < len(args):
        arg = str(args[i])
        if arg in {"-i", "--ignore-environment"}:
            stripped = True
            i += 1
            continue
        if arg == "-u":
            if i + 1 < len(args) and str(args[i + 1]) in _GUARD_ENV_VARS:
                stripped = True
            i += 2
            continue
        if arg.startswith("-u") and len(arg) > 2:
            if arg[2:] in _GUARD_ENV_VARS:
                stripped = True
            i += 1
            continue
        if arg.startswith("--unset="):
            if arg.split("=", 1)[1] in _GUARD_ENV_VARS:
                stripped = True
            i += 1
            continue
        if arg == "-S" or arg.startswith("-S"):
            stripped = True
            i += 1
            continue
        if "=" in arg and not arg.startswith("-"):
            key = arg.split("=", 1)[0]
            if key in _GUARD_ENV_VARS or key in _IMMUTABLE_GUARD_ENV:
                stripped = True
            i += 1
            continue
        break
    return list(args[i:]), stripped


def _effective_argv(args):
    payload, _ = _env_payload(args)
    return payload if payload else args


def _argv_strips_guard(args) -> bool:
    payload, stripped = _env_payload(args)
    return bool(stripped and payload)


def _python_argv_disables_guard(args) -> bool:
    args = _effective_argv(args)
    if not isinstance(args, (list, tuple)) or not args:
        return False
    exe = pathlib.Path(str(args[0])).name
    if not exe.startswith("python"):
        return False
    for raw in args[1:]:
        arg = str(raw)
        if arg == "--":
            return False
        if not arg.startswith("-"):
            return False
        if arg in {"-c", "-m", "-"}:
            return False
        if arg in {"-S", "-I", "-E"}:
            return True
        if arg.startswith("-") and any(flag in arg[1:] for flag in "SIE"):
            return True
        if arg in {"-W", "-X"}:
            continue
    return False


def _pathlike_value(value) -> bool:
    value = str(value)
    if not value:
        return False
    if value.startswith(("/", ".", "~", "$")) or "/" in value:
        return True
    return any(word in value for word in ("shared_findings", "frontier", "gems", "gen_", "results", "variants", "peer_workspaces", "shared_store.db"))


def _path_arg_touches_protected(raw, cwd=None) -> bool:
    value = str(raw)
    if not _pathlike_value(value):
        return False
    try:
        resolved = _resolve_path(value, cwd=cwd)
    except Exception:
        return True
    if (
        _under_allowed(resolved)
        or _is_peer_owned_write_path(resolved)
        or _is_safe_system_agenda_write_path(resolved)
    ):
        return False
    return _under_any(resolved, _PROTECTED_ROOTS)


def _command_path_args(argv):
    args = []
    i = 0
    while i < len(argv):
        token = str(argv[i])
        if token == "--":
            args.extend(str(arg) for arg in argv[i + 1 :])
            break
        if token in {"-t", "--target-directory"} and i + 1 < len(argv):
            args.append(str(argv[i + 1]))
            i += 2
            continue
        if token.startswith("-t") and not token.startswith("--") and len(token) > 2:
            args.append(token[2:])
            i += 1
            continue
        if token.startswith("--target-directory="):
            args.append(token.split("=", 1)[1])
            i += 1
            continue
        if token.startswith("-"):
            i += 1
            continue
        args.append(token)
        i += 1
    return args


def _mv_sources_destinations(args):
    args = _effective_argv(args)
    if not isinstance(args, (list, tuple)) or len(args) < 2:
        return [], []
    argv = [str(arg) for arg in args[1:]]
    paths = _command_path_args(argv)
    target_dir = None
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in {"-t", "--target-directory"} and i + 1 < len(argv):
            target_dir = argv[i + 1]
            i += 2
            continue
        if token.startswith("-t") and not token.startswith("--") and len(token) > 2:
            target_dir = token[2:]
            i += 1
            continue
        if token.startswith("--target-directory="):
            target_dir = token.split("=", 1)[1]
            i += 1
            continue
        i += 1
    if target_dir:
        return [path for path in paths if path != target_dir], [target_dir]
    if len(paths) < 2:
        return paths, []
    return paths[:-1], [paths[-1]]


def _find_start_paths(argv):
    roots = []
    for raw in argv:
        token = str(raw)
        if token in {"--"}:
            continue
        if token.startswith("-") or token in {"(", ")", "!", ","}:
            break
        roots.append(token)
    return roots or ["."]


def _find_exec_destructive(argv) -> bool:
    destructive = {
        "rm", "rmdir", "unlink", "mv", "cp", "install", "touch", "mkdir",
        "chmod", "chown", "chgrp", "truncate", "dd", "tee", "sed", "shred",
        "rsync", "tar", "zip", "python", "python3", "sh", "bash", "dash",
        "env",
    }
    for i, raw in enumerate(argv):
        token = str(raw)
        if token not in {"-exec", "-execdir"}:
            continue
        if i + 1 >= len(argv):
            return True
        base = pathlib.Path(str(argv[i + 1])).name
        if base in destructive:
            return True
        for arg in argv[i + 2:]:
            value = str(arg)
            if value in {";", "+"}:
                break
            if _path_arg_touches_protected(value):
                return True
    return False


def _check_find_command(args, cwd=None) -> None:
    argv = [str(arg) for arg in args[1:]]
    if "-delete" not in argv and "-exec" not in argv and "-execdir" not in argv:
        return
    roots = _find_start_paths(argv)
    unsafe_roots = [
        root for root in roots
        if _path_arg_touches_protected(root, cwd=cwd) or not (
            _under_allowed(_resolve_path(root, cwd=cwd)) or _is_peer_owned_delete_path(_resolve_path(root, cwd=cwd))
        )
    ]
    if unsafe_roots:
        raise PermissionError("Praxist delete guard blocked subprocess find destructive action outside peer-owned paths")
    if _find_exec_destructive(argv):
        raise PermissionError("Praxist delete guard blocked subprocess destructive find -exec")


_MAX_BUILD_INCLUDE_DEPTH = 8


def _build_include_targets(line, *, is_ninja):
    stripped = str(line).strip()
    if not stripped:
        return None
    parts = stripped.split(None, 1)
    directive = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    if is_ninja:
        if directive in {"include", "subninja"}:
            return directive, [rest], False
        return None
    if directive in {"-include", "sinclude", "include"}:
        optional = directive in {"-include", "sinclude"}
        return directive, rest.split() if rest else [], optional
    return None


def _build_include_allowed_path(path, cwd=None):
    resolved = _resolve_path(path, cwd=cwd)
    return (
        resolved
        if _under_allowed(resolved) or _is_peer_owned_write_path(resolved, cwd=cwd) or _under_trusted_project(resolved)
        else None
    )


def _check_build_include_target(raw, *, source_path):
    raw = str(raw).strip()
    if not raw or any(token in raw for token in ("$", "`", "*", "?", "[", "\"", "'")):
        raise PermissionError("Praxist delete guard blocked dynamic make/ninja include")
    resolved = _build_include_allowed_path(raw, cwd=source_path.parent)
    if resolved is None:
        raise PermissionError("Praxist delete guard blocked make/ninja include outside trusted paths")
    return resolved


def _check_build_recipe_command(recipe, *, cwd=None):
    try:
        argv = shlex.split(str(recipe), comments=False, posix=True)
    except ValueError:
        raise PermissionError("Praxist delete guard blocked unparsable make/ninja recipe")
    if not argv:
        return
    if _argv_strips_guard(argv):
        raise PermissionError("Praxist delete guard blocked make/ninja recipe with stripped guard env")
    effective_args = _effective_argv(argv)
    if _python_argv_disables_guard(effective_args):
        raise PermissionError("Praxist delete guard blocked make/ninja recipe Python guard-disabling flags")
    _check_peer_binary_execution(effective_args, cwd=cwd)
    if isinstance(effective_args, (list, tuple)) and effective_args:
        base = pathlib.Path(str(effective_args[0])).name
        if base in _SHELLS:
            payload = _shell_c_payload(effective_args)
            if payload is not None and _shell_payload_dangerous(payload):
                raise PermissionError("Praxist delete guard blocked make/ninja unsafe shell payload recipe")
            if payload is None:
                script = _shell_script_operand(effective_args)
                if script is None:
                    raise PermissionError("Praxist delete guard blocked make/ninja shell recipe without inspectable script")
                if base != "bash":
                    raise PermissionError("Praxist delete guard blocked make/ninja non-Bash shell script recipe")
                if _shell_script_content_dangerous(script, cwd=cwd):
                    raise PermissionError("Praxist delete guard blocked make/ninja unsafe shell script recipe")
    for target in _subprocess_delete_target(effective_args, cwd=cwd):
        if target in {"<implicit>", "<shell-command>"}:
            raise PermissionError("Praxist delete guard blocked make/ninja destructive recipe command")
        resolved = _resolve_path(target, cwd=cwd)
        if not (_under_allowed(resolved) or _is_peer_owned_delete_path(resolved)):
            raise PermissionError("Praxist delete guard blocked make/ninja recipe deleting protected path")
    _check_mutating_system_command(effective_args, cwd=cwd)
    _check_unclassified_system_command(effective_args, cwd=cwd)


def _check_build_file_content(path, *, depth=0, seen=None):
    path = pathlib.Path(path).expanduser().resolve(strict=False)
    seen = set() if seen is None else seen
    key = str(path)
    if key in seen:
        return
    if depth > _MAX_BUILD_INCLUDE_DEPTH:
        raise PermissionError("Praxist delete guard blocked make/ninja include recursion")
    seen.add(key)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        raise PermissionError("Praxist delete guard blocked subprocess make/ninja unreadable build file")
    lower = "\n" + text.lower() + "\n"
    if any(marker in lower for marker in _DANGEROUS_BUILD_RECIPE_MARKERS):
        raise PermissionError("Praxist delete guard blocked subprocess make/ninja destructive recipe")
    is_ninja = path.name == "build.ninja" or path.suffix == ".ninja"
    for line in text.splitlines():
        raw_stripped = line.strip()
        stripped = raw_stripped.lower()
        if stripped.startswith("shell") and "=" in stripped:
            raise PermissionError("Praxist delete guard blocked subprocess makefile shell override")
        if not line.startswith("\t"):
            include = _build_include_targets(line, is_ninja=is_ninja)
            if include is not None:
                _directive, targets, optional = include
                for target in targets:
                    include_path = _check_build_include_target(target, source_path=path)
                    if not include_path.exists():
                        if optional:
                            continue
                        raise PermissionError("Praxist delete guard blocked missing make/ninja include")
                    _check_build_file_content(include_path, depth=depth + 1, seen=seen)
        recipe = ""
        if line.startswith("\t"):
            recipe = line.lstrip()
            while recipe.startswith(("@", "+", "-")):
                recipe = recipe[1:].lstrip()
        elif stripped.startswith("command") and "=" in stripped:
            recipe = raw_stripped.split("=", 1)[1].strip()
        if recipe:
            _check_build_recipe_command(recipe, cwd=path.parent)


def _explicit_build_file_candidates(raw, *, build_paths, cwd=None):
    expanded = os.path.expandvars(str(raw))
    if "$" in expanded:
        return []
    try:
        explicit_path = pathlib.Path(expanded).expanduser()
    except Exception:
        return []
    bases = [cwd or os.getcwd()] if explicit_path.is_absolute() else (build_paths or [pathlib.Path(cwd or os.getcwd()).resolve()])
    candidates = []
    for base in bases:
        candidates.append(_resolve_path(expanded, cwd=base))
    return candidates


def _check_make_ninja_command(args, cwd=None) -> None:
    args = _effective_argv(args)
    if not isinstance(args, (list, tuple)) or not args:
        return
    base = pathlib.Path(str(args[0])).name
    argv = [str(arg) for arg in args[1:]]
    lowered = [arg.lower() for arg in argv if arg and not arg.startswith("-")]
    if any(target in _DANGEROUS_BUILD_TARGETS for target in lowered):
        raise PermissionError("Praxist delete guard blocked subprocess make/ninja clean target")
    build_paths = []
    explicit_build_files = []
    has_explicit_build_dir = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in {"-C", "--directory"} and i + 1 < len(argv):
            build_paths.append(argv[i + 1])
            has_explicit_build_dir = True
            i += 2
            continue
        if arg.startswith("--directory="):
            build_paths.append(arg.split("=", 1)[1])
            has_explicit_build_dir = True
        if arg.startswith("-C") and len(arg) > 2:
            build_paths.append(arg[2:])
            has_explicit_build_dir = True
        if base in {"make", "gmake"}:
            if arg in {"-f", "--file", "--makefile"} and i + 1 < len(argv):
                explicit_build_files.append(argv[i + 1])
                i += 2
                continue
            if arg.startswith(("-f", "--file=", "--makefile=")):
                value = arg[2:] if arg.startswith("-f") and len(arg) > 2 else arg.split("=", 1)[-1]
                explicit_build_files.append(value)
        elif base == "ninja":
            if arg == "-f" and i + 1 < len(argv):
                explicit_build_files.append(argv[i + 1])
                i += 2
                continue
            if arg.startswith("-f") and len(arg) > 2:
                explicit_build_files.append(arg[2:])
        i += 1
    resolved_build_paths = [pathlib.Path(cwd or os.getcwd()).resolve()]
    if has_explicit_build_dir:
        for raw in build_paths:
            next_paths = {}
            for base_path in resolved_build_paths:
                resolved = _resolve_path(raw, cwd=base_path)
                next_paths[str(resolved)] = resolved
            if len(next_paths) != 1:
                raise PermissionError("Praxist delete guard blocked subprocess ambiguous make/ninja build path")
            resolved_build_paths = list(next_paths.values())
    for resolved in resolved_build_paths:
        if not (_under_allowed(resolved) or _is_peer_owned_write_path(resolved, cwd=cwd) or _under_trusted_project(resolved)):
            raise PermissionError("Praxist delete guard blocked subprocess make/ninja outside peer workspace")
    build_files = []
    for raw in explicit_build_files:
        build_files.extend(_explicit_build_file_candidates(raw, build_paths=resolved_build_paths, cwd=cwd))
    if not explicit_build_files:
        if base in {"make", "gmake"}:
            for build_path in resolved_build_paths:
                for name in _MAKEFILE_NAMES:
                    path = build_path / name
                    if path.exists():
                        build_files.append(path)
        elif base == "ninja":
            for build_path in resolved_build_paths:
                path = build_path / "build.ninja"
                if path.exists():
                    build_files.append(path)
    unique = {}
    for path in build_files:
        unique[str(path.resolve(strict=False))] = path
    if not unique:
        raise PermissionError("Praxist delete guard blocked subprocess make/ninja without inspectable build file")
    for path in unique.values():
        if not (_under_allowed(path) or _is_peer_owned_write_path(path, cwd=cwd) or _under_trusted_project(path)):
            raise PermissionError("Praxist delete guard blocked subprocess make/ninja build file outside peer workspace")
        _check_build_file_content(path)


def _check_mutating_system_command(args, cwd=None) -> None:
    args = _effective_argv(args)
    if not isinstance(args, (list, tuple)) or not args:
        return
    base = pathlib.Path(str(args[0])).name
    argv = [str(arg) for arg in args[1:]]
    if base in _BROAD_KILL_BASENAMES:
        raise PermissionError("Praxist delete guard blocked subprocess broad process signalling")
    if base == "xargs" and any(pathlib.Path(str(arg)).name in _BROAD_KILL_BASENAMES for arg in argv):
        raise PermissionError("Praxist delete guard blocked subprocess xargs-driven process signalling")
    if base == "find":
        _check_find_command(args, cwd=cwd)
    elif base in {"touch", "mkdir", "fallocate"}:
        for arg in argv:
            if not arg.startswith("-") and _path_arg_touches_protected(arg, cwd=cwd):
                raise PermissionError(f"Praxist delete guard blocked subprocess {base} touching protected path")
    elif base == "mv":
        sources, destinations = _mv_sources_destinations(args)
        for arg in sources:
            if not (_under_allowed(_resolve_path(arg, cwd=cwd)) or _is_peer_owned_delete_path(_resolve_path(arg, cwd=cwd))):
                raise PermissionError("Praxist delete guard blocked subprocess mv source outside peer-owned paths")
        for arg in destinations:
            _check_replace_destination(arg, "subprocess mv destination", cwd=cwd)
    elif base in {"cp", "install"}:
        paths = [arg for arg in argv if _pathlike_value(arg) and not arg.startswith("-")]
        if any(_path_arg_touches_protected(arg, cwd=cwd) for arg in paths[-1:]):
            raise PermissionError(f"Praxist delete guard blocked subprocess {base} protected destination")
    elif base in {"chmod", "chown", "chgrp", "truncate", "sort", "tee", "sed", "ln", "rsync", "dd"}:
        for arg in argv:
            value = arg.split("=", 1)[1] if "=" in arg and arg.startswith(("of=", "--output", "--target-directory")) else arg
            if _path_arg_touches_protected(value, cwd=cwd):
                raise PermissionError(f"Praxist delete guard blocked subprocess {base} touching protected path")
    elif base in {"make", "gmake", "ninja"}:
        _check_make_ninja_command(args, cwd=cwd)
    elif base == "tar" and any("--checkpoint-action" in arg or "--remove-files" in arg for arg in argv):
        raise PermissionError("Praxist delete guard blocked subprocess tar checkpoint/remove action")
    elif base in {"gcc", "g++", "cc", "clang", "clang++", "ld", "curl", "wget"}:
        for arg in argv:
            if _path_arg_touches_protected(arg, cwd=cwd):
                raise PermissionError(f"Praxist delete guard blocked subprocess {base} touching protected path")


def _check_unclassified_system_command(args, cwd=None) -> None:
    args = _effective_argv(args)
    if not isinstance(args, (list, tuple)) or not args:
        return
    path = _argv_executable_path(args, cwd=cwd)
    if path is None or not _is_trusted_system_executable(path):
        return
    base = pathlib.Path(str(args[0])).name
    if (
        base.startswith("python")
        or base in _SHELLS
        or base in _READ_ONLY_SYSTEM_BASENAMES
        or base in _CLASSIFIED_MUTATING_BASENAMES
        or base in {"rm", "rmdir", "unlink"}
    ):
        return
    if any(_path_arg_touches_protected(arg, cwd=cwd) for arg in args[1:]):
        raise PermissionError(
            f"Praxist delete guard blocked unclassified system command touching protected path: {base}"
        )


def _shell_c_payload(args):
    if not isinstance(args, (list, tuple)):
        return None
    i = 1
    while i < len(args):
        arg = str(args[i])
        if arg == "--":
            i += 1
            continue
        if arg == "-c":
            if i + 1 < len(args) and str(args[i + 1]) == "--" and i + 2 < len(args):
                return str(args[i + 2])
            return str(args[i + 1]) if i + 1 < len(args) else ""
        if arg.startswith("-") and "c" in arg[1:]:
            after_c = arg.split("c", 1)[1]
            if after_c:
                return after_c
            if i + 1 < len(args) and str(args[i + 1]) == "--" and i + 2 < len(args):
                return str(args[i + 2])
            return str(args[i + 1]) if i + 1 < len(args) else ""
        i += 1
    return None


def _env_strips_guard(env):
    if env is None:
        return False
    if not hasattr(env, "get"):
        return True
    for key in ("PRAXIST_SAFE_DELETE_ROOTS", "PRAXIST_PEER_WORKSPACE", "PRAXIST_DELETE_GUARD_RUN_DIR", "PYTHONPATH"):
        if not env.get(key):
            return True
    for key, expected in _IMMUTABLE_GUARD_ENV.items():
        if key in {"PYTHONPATH", "PATH"}:
            continue
        if expected is None:
            if env.get(key):
                return True
            continue
        if env.get(key) != expected:
            return True
    for key in _GUARD_ENV_VARS:
        if key in env and not env.get(key):
            return True
    expected_pythonpath = _IMMUTABLE_GUARD_ENV.get("PYTHONPATH") or ""
    if expected_pythonpath:
        expected_guard_parts = [
            part for part in expected_pythonpath.split(os.pathsep) if ".runtime_guards" in part
        ]
        current_parts = [part for part in str(env.get("PYTHONPATH", "")).split(os.pathsep) if part]
        for guard_part in expected_guard_parts:
            if guard_part not in current_parts:
                return True
            guard_index = current_parts.index(guard_part)
            for prior in current_parts[:guard_index]:
                try:
                    prior_path = _resolve_path(prior)
                except Exception:
                    return True
                if _under_allowed(prior_path) or _is_peer_owned_write_path(prior_path):
                    return True
    return False


def _launcher_is_guarded_python(args):
    args = _effective_argv(args)
    if not isinstance(args, (list, tuple)) or not args:
        return False
    exe = pathlib.Path(str(args[0])).name
    if not exe.startswith("python"):
        return False
    joined = " ".join(str(arg).lower() for arg in args[1:])
    tamper_terms = (
        "sys.modules",
        "sitecustomize",
        "_orig",
        "_orig_by_wrapper",
        "object.__getattribute__",
        "__globals__",
        "__closure__",
    )
    if any(term in joined for term in tamper_terms):
        return True
    if "bypass_gpu_governor" in joined:
        return True
    if ("ctypes" in joined or "_ctypes" in joined or "cffi" in joined) and (
        "libc" in joined or "libdl" in joined or "cdll(none" in joined
    ):
        return True
    if _script_mentions_protected_target(joined):
        protected_write_terms = (
            "os.remove",
            "os.unlink",
            "os.rmdir",
            "shutil.rmtree",
            "os.rename",
            "os.replace",
            "os.truncate",
            "os.open",
            "open(",
            ".write(",
            "write(",
            "posix.open",
            "chmod",
            "chown",
            "link",
            "symlink",
            "subprocess",
            "os.system",
            "spawn",
            "execv",
        )
        return any(term in joined for term in protected_write_terms)
    return False


def _python_script_from_argv(args):
    args = _effective_argv(args)
    i = 1
    while i < len(args):
        arg = str(args[i])
        if arg == "--":
            i += 1
            break
        if arg in {"-c", "-m"} or arg == "-":
            return None
        if arg in {"-W", "-X"}:
            i += 2
            continue
        if arg.startswith("-W") or arg.startswith("-X"):
            i += 1
            continue
        if arg.startswith("-"):
            i += 1
            continue
        return arg
    return str(args[i]) if i < len(args) else None


def _python_script_content_dangerous(script):
    try:
        path = _resolve_path(script)
    except Exception:
        return True
    if not path.exists() or not path.is_file():
        return False
    try:
        content = path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return True
    tamper_terms = (
        "sys.modules",
        ".modules",
        "__globals__",
        "__defaults__",
        "__kwdefaults__",
        "__closure__",
        "__code__",
        "sitecustomize",
        "_orig",
        "_orig_by_wrapper",
        "object.__getattribute__",
        "__getattribute__",
    )
    if any(term in content for term in tamper_terms):
        return True
    if ("_ctypes" in content or "ctypes" in content or "cffi" in content) and (
        "libc" in content or "libdl" in content or "cdll(none" in content
    ):
        return True
    return False


def _payload_uses_operator_bypass_workload(lower) -> bool:
    return "bypass_gpu_governor" in lower and (
        "python" in lower
        or "train" in lower
        or "cuda" in lower
        or "torch" in lower
        or ".py" in lower
    )


def _payload_has_broad_process_signal(lower) -> bool:
    if re.search(r"(^|[;&|`$()\s])(pkill|killall)(\s|$)", lower):
        return True
    if re.search(r"(^|[;&|`$()\s])xargs\s+.*\bkill\b", lower):
        return True
    return False


def _payload_has_high_risk_delete(lower) -> bool:
    return (
        ("find " in lower and (" -delete" in lower or " -exec" in lower or " -execdir" in lower))
        or "rsync --delete" in lower
        or "--remove-source-files" in lower
        or "--remove-files" in lower
        or "--checkpoint-action" in lower
        or "zip -m" in lower
        or "shred -u" in lower
        or "rm -rf /" in lower
        or "rm -fr /" in lower
    )


def _payload_has_protected_write_or_delete(lower) -> bool:
    protected_markers = (
        ">",
        ">>",
        " rm ",
        "r${",
        "/bin/rm",
        "/bin/r$",
        "rmdir ",
        "unlink ",
        "mv ",
        "cp ",
        "install ",
        "touch ",
        "mkdir ",
        "chmod ",
        "chown ",
        "truncate ",
        "tee ",
        "ln ",
        "dd ",
        "sed -i",
        "open(",
        ".write(",
        "write(",
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "shutil.rmtree",
        "os.rename",
        "os.replace",
        "os.truncate",
        "os.open",
        "subprocess",
        "os.system",
        "spawn",
        "execv",
    )
    return any(marker in lower for marker in protected_markers) or _payload_has_high_risk_delete(lower)


def _shell_payload_dangerous(payload) -> bool:
    lower = str(payload).lower()
    if _payload_uses_operator_bypass_workload(lower):
        return True
    if _payload_has_broad_process_signal(lower):
        return True
    if _payload_has_high_risk_delete(lower):
        return True
    return _script_mentions_protected_target(lower) and _payload_has_protected_write_or_delete(lower)


def _shell_script_operand(args):
    args = _effective_argv(args)
    if not isinstance(args, (list, tuple)) or not args:
        return None
    exe = pathlib.Path(str(args[0])).name
    if exe not in _SHELLS:
        return None
    if _shell_c_payload(args) is not None:
        return None
    i = 1
    while i < len(args):
        arg = str(args[i])
        if arg == "--":
            i += 1
            break
        if arg in {"-o", "--rcfile", "--init-file"}:
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        return arg
    return str(args[i]) if i < len(args) else None


def _script_mentions_protected_target(content) -> bool:
    lower = str(content).lower()
    roots = [root for root in (_RUN_ROOT, *_PROTECTED_ROOTS) if root is not None]
    for root in roots:
        try:
            if str(root).lower() in lower:
                return True
        except Exception:
            continue
    protected_words = (
        "shared_findings",
        "frontier",
        "gems",
        "gem",
        "gen_",
        "results",
        "variants",
        "peer_workspaces",
        "generation_results",
        "run_summary",
        "shared_store.db",
    )
    return any(word in lower for word in protected_words)


def _shell_script_content_dangerous(script, cwd=None) -> bool:
    try:
        path = _resolve_path(script, cwd=cwd)
    except Exception:
        return True
    if not path.exists():
        return False
    if path.is_dir():
        return False
    try:
        head = path.read_bytes()[:4096]
        if (
            b"\x00" in head
            or head.startswith(b"\x7fELF")
            or head.startswith(b"\xcf\xfa\xed\xfe")
            or head.startswith(b"\xfe\xed\xfa\xcf")
            or head.startswith(b"\xca\xfe\xba\xbe")
        ):
            return True
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    if len(content) > 200_000:
        return True
    if not _script_mentions_protected_target(content):
        return False
    lower = content.lower()
    dangerous_markers = (
        ">",
        ">>",
        "$(",
        "`",
        " rm ",
        "r${",
        "/bin/rm",
        "/bin/r$",
        "\trm ",
        "rmdir ",
        "unlink ",
        "mv ",
        "cp ",
        "install ",
        "touch ",
        "mkdir ",
        "tee ",
        "truncate ",
        "sed -i",
        "open(",
        ".write(",
        "write(",
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "shutil.rmtree",
        "os.rename",
        "os.replace",
        "subprocess",
        "os.system",
        "--remove-files",
        "--checkpoint-action",
        "rsync --delete",
        "--remove-source-files",
    )
    return any(marker in lower for marker in dangerous_markers)


def _subprocess_delete_target(args, cwd=None):
    if isinstance(args, (list, tuple)) and args:
        args = _effective_argv(args)
        if not args:
            return ["<shell-command>"]
        exe = pathlib.Path(str(args[0])).name
        if exe == "mv":
            targets, _ = _mv_sources_destinations(args)
            return targets or ["<implicit>"]
        if exe in {"rm", "rmdir", "unlink"}:
            targets = [
                str(arg)
                for arg in args[1:]
                if str(arg) != "--" and not str(arg).startswith("-")
            ]
            return targets or ["<implicit>"]
        if exe == "tar" and "--remove-files" in [str(arg) for arg in args[1:]]:
            return ["<shell-command>"]
        if exe == "zip" and any(str(arg) == "-m" or (str(arg).startswith("-") and "m" in str(arg)[1:]) for arg in args[1:]):
            return ["<shell-command>"]
        if exe == "shred" and any(str(arg) == "-u" or str(arg) == "--remove" or str(arg).startswith("--remove=") for arg in args[1:]):
            targets = [
                str(arg)
                for arg in args[1:]
                if str(arg) != "--" and not str(arg).startswith("-")
            ]
            return targets or ["<implicit>"]
        if exe == "rsync" and any(str(arg) in {"--delete", "--remove-source-files"} or str(arg).startswith("--delete-") for arg in args[1:]):
            return ["<shell-command>"]
        if _launcher_is_guarded_python(args):
            return ["<shell-command>"]
        if exe in _SHELLS:
            payload = _shell_c_payload(args)
            if payload is not None and _shell_payload_dangerous(payload):
                return ["<shell-command>"]
            script = _shell_script_operand(args)
            if script is not None and _shell_script_content_dangerous(script, cwd=cwd):
                return ["<shell-command>"]
    if isinstance(args, str):
        if _shell_payload_dangerous(args):
            return ["<shell-command>"]
    return []


def _check_popen_args(*popenargs, **kwargs):
    args = popenargs[0] if popenargs else kwargs.get("args")
    effective_env = kwargs.get("env") if kwargs.get("env") is not None else os.environ
    if _env_strips_guard(effective_env):
        _warn_runtime(
            "subprocess_stripped_guard_env",
            "Praxist runtime guard allowed subprocess with stripped guard-related environment.",
            command=args,
        )
    child_cwd = kwargs.get("cwd") or os.getcwd()
    if _argv_strips_guard(args):
        _warn_runtime(
            "subprocess_env_command_strips_guard",
            "Praxist runtime guard allowed env command that strips guard-related variables.",
            command=args,
        )
    effective_args = _effective_argv(args)
    if _python_argv_disables_guard(effective_args):
        _warn_runtime(
            "subprocess_python_guard_disabling_flags",
            "Praxist runtime guard allowed Python guard-disabling flags for a subprocess.",
            command=effective_args,
        )
    _check_peer_binary_execution(effective_args, cwd=child_cwd)
    _check_mutating_system_command(effective_args, cwd=child_cwd)
    _check_unclassified_system_command(effective_args, cwd=child_cwd)
    for target in _subprocess_delete_target(effective_args, cwd=child_cwd):
        if target in {"<implicit>", "<shell-command>"}:
            raise PermissionError("Praxist delete guard blocked subprocess delete command")
        _check(target, "subprocess delete", cwd=child_cwd)


def _guarded_popen(*popenargs, **kwargs):
    _check_popen_args(*popenargs, **kwargs)
    return _orig_for(_guarded_popen)(*popenargs, **kwargs)


class _GuardedPopen(_orig_popen):
    def __init__(self, *popenargs, **kwargs):
        _check_popen_args(*popenargs, **kwargs)
        super().__init__(*popenargs, **kwargs)


def _guarded_os_system(command):
    if _shell_payload_dangerous(command):
        raise PermissionError("Praxist delete guard blocked os.system delete command")
    return _orig_for(_guarded_os_system)(command)


def _guarded_spawn_common(file, args, env=None):
    effective_env = env if env is not None else os.environ
    if _env_strips_guard(effective_env):
        _warn_runtime(
            "spawn_stripped_guard_env",
            "Praxist runtime guard allowed spawn with stripped guard-related environment.",
            command=args or [file],
        )
    raw_args = args or [file]
    if _argv_strips_guard(raw_args):
        _warn_runtime(
            "spawn_env_command_strips_guard",
            "Praxist runtime guard allowed env command that strips guard-related variables.",
            command=raw_args,
        )
    effective_args = _effective_argv(raw_args)
    if _python_argv_disables_guard(effective_args):
        _warn_runtime(
            "spawn_python_guard_disabling_flags",
            "Praxist runtime guard allowed Python guard-disabling flags for spawn/exec.",
            command=effective_args,
        )
    _check_peer_binary_execution(effective_args)
    _check_mutating_system_command(effective_args)
    _check_unclassified_system_command(effective_args)
    targets = _subprocess_delete_target(effective_args)
    for target in targets:
        if target in {"<implicit>", "<shell-command>"}:
            raise PermissionError("Praxist delete guard blocked spawn/exec destructive command")
        _check(target, "spawn/exec delete")


def _guarded_spawnv(mode, file, args):
    _guarded_spawn_common(file, args)
    return _orig_for(_guarded_spawnv)(mode, file, args)


def _guarded_spawnve(mode, file, args, env):
    _guarded_spawn_common(file, args, env=env)
    return _orig_for(_guarded_spawnve)(mode, file, args, env)


def _guarded_spawnvp(mode, file, args):
    _guarded_spawn_common(file, args)
    return _orig_for(_guarded_spawnvp)(mode, file, args)


def _guarded_spawnvpe(mode, file, args, env):
    _guarded_spawn_common(file, args, env=env)
    return _orig_for(_guarded_spawnvpe)(mode, file, args, env)


def _guarded_execv(file, args):
    _guarded_spawn_common(file, args)
    return _orig_for(_guarded_execv)(file, args)


def _guarded_execve(file, args, env):
    _guarded_spawn_common(file, args, env=env)
    return _orig_for(_guarded_execve)(file, args, env)


def _guarded_execvp(file, args):
    _guarded_spawn_common(file, args)
    return _orig_for(_guarded_execvp)(file, args)


def _guarded_execvpe(file, args, env):
    _guarded_spawn_common(file, args, env=env)
    return _orig_for(_guarded_execvpe)(file, args, env)


def _guarded_posix_spawn(path, argv, env, *args, **kwargs):
    _guarded_spawn_common(path, argv, env=env)
    return _orig_for(_guarded_posix_spawn)(path, argv, env, *args, **kwargs)


def _guarded_posix_spawnp(path, argv, env, *args, **kwargs):
    _guarded_spawn_common(path, argv, env=env)
    return _orig_for(_guarded_posix_spawnp)(path, argv, env, *args, **kwargs)


_bind_orig(_guarded_rmtree, _orig_rmtree)
_bind_orig(_warning_open, _orig_open)
_bind_orig(_warning_makedirs, _orig_makedirs)
_bind_orig(_guarded_move, _orig_move)
_bind_orig(_guarded_open, _orig_open)
_bind_orig(_guarded_io_open, _orig_io_open)
_bind_orig(_guarded__io_open, _orig__io_open)
_bind_orig(_guarded_os_open, _orig_os_open)
_bind_orig(_guarded_putenv, _orig_putenv)
if _orig_unsetenv is not None:
    _bind_orig(_guarded_unsetenv, _orig_unsetenv)
_bind_orig(_guarded_environ_setitem, _orig_environ_setitem)
_bind_orig(_guarded_environ_delitem, _orig_environ_delitem)
_bind_orig(_guarded_mkdir, _orig_mkdir)
_bind_orig(_guarded_makedirs, _orig_makedirs)
if _orig_mknod is not None:
    _bind_orig(_guarded_mknod, _orig_mknod)
if _orig_mkfifo is not None:
    _bind_orig(_guarded_mkfifo, _orig_mkfifo)
_bind_orig(_guarded_remove, _orig_remove)
_bind_orig(_guarded_unlink, _orig_unlink)
_bind_orig(_guarded_rmdir, _orig_rmdir)
_bind_orig(_guarded_rename, _orig_rename)
_bind_orig(_guarded_replace, _orig_replace)
_bind_orig(_guarded_truncate, _orig_truncate)
_bind_orig(_guarded_ftruncate, _orig_ftruncate)
_bind_orig(_guarded_chmod, _orig_chmod)
if _orig_fchmod is not None:
    _bind_orig(_guarded_fchmod, _orig_fchmod)
_bind_orig(_guarded_link, _orig_link)
_bind_orig(_guarded_symlink, _orig_symlink)
if _orig_chown is not None:
    _bind_orig(_guarded_chown, _orig_chown)
if _orig_lchown is not None:
    _bind_orig(_guarded_lchown, _orig_lchown)
if _orig_fchown is not None:
    _bind_orig(_guarded_fchown, _orig_fchown)
if _orig_posix_open is not None:
    _bind_orig(_guarded_posix_open, _orig_posix_open)
if _orig_posix_mkdir is not None:
    _bind_orig(_guarded_posix_mkdir, _orig_posix_mkdir)
if _orig_posix_mknod is not None:
    _bind_orig(_guarded_posix_mknod, _orig_posix_mknod)
if _orig_posix_mkfifo is not None:
    _bind_orig(_guarded_posix_mkfifo, _orig_posix_mkfifo)
_bind_orig(_guarded_posix_unlink, _orig_posix_unlink)
_bind_orig(_guarded_posix_rmdir, _orig_posix_rmdir)
if _orig_posix_chmod is not None:
    _bind_orig(_guarded_posix_chmod, _orig_posix_chmod)
_bind_orig(_guarded_path_unlink, _orig_path_unlink)
_bind_orig(_guarded_path_rmdir, _orig_path_rmdir)
_bind_orig(_guarded_path_rename, _orig_path_rename)
_bind_orig(_guarded_path_replace, _orig_path_replace)
_bind_orig(_guarded_path_open, _orig_path_open)
_bind_orig(_guarded_path_write_text, _orig_path_write_text)
_bind_orig(_guarded_path_write_bytes, _orig_path_write_bytes)
_bind_orig(_guarded_path_chmod, _orig_path_chmod)
_bind_orig(_guarded_path_mkdir, _orig_path_mkdir)
_bind_orig(_guarded_copyfile, _orig_copyfile)
_bind_orig(_guarded_copy, _orig_copy)
_bind_orig(_guarded_copy2, _orig_copy2)
_bind_orig(_guarded_copytree, _orig_copytree)
_bind_orig(_guarded_sqlite_connect, _orig_sqlite_connect)
_bind_orig(_guarded_import, _orig_import)
_bind_orig(_guarded_ctypes_cdll, _orig_ctypes_cdll)
_bind_orig(_guarded_ctypes_pydll, _orig_ctypes_pydll)
_bind_orig(_guarded_ctypes_pydll_loadlibrary, _orig_ctypes_pydll_loadlibrary)
if _orig__ctypes_dlopen is not None:
    _bind_orig(_guarded__ctypes_dlopen, _orig__ctypes_dlopen)
_bind_orig(_guarded_popen, _orig_popen)
_bind_orig(_guarded_os_system, _orig_os_system)
if _orig_spawnv is not None:
    _bind_orig(_guarded_spawnv, _orig_spawnv)
if _orig_spawnve is not None:
    _bind_orig(_guarded_spawnve, _orig_spawnve)
if _orig_spawnvp is not None:
    _bind_orig(_guarded_spawnvp, _orig_spawnvp)
if _orig_spawnvpe is not None:
    _bind_orig(_guarded_spawnvpe, _orig_spawnvpe)
if _orig_execv is not None:
    _bind_orig(_guarded_execv, _orig_execv)
if _orig_execve is not None:
    _bind_orig(_guarded_execve, _orig_execve)
if _orig_execvp is not None:
    _bind_orig(_guarded_execvp, _orig_execvp)
if _orig_execvpe is not None:
    _bind_orig(_guarded_execvpe, _orig_execvpe)
if _orig_posix_spawn is not None:
    _bind_orig(_guarded_posix_spawn, _orig_posix_spawn)
if _orig_posix_spawnp is not None:
    _bind_orig(_guarded_posix_spawnp, _orig_posix_spawnp)
shutil.rmtree = _guarded_rmtree
shutil.move = _guarded_move
builtins.open = _guarded_open
io.open = _guarded_io_open
_io.open = _guarded__io_open
os.open = _guarded_os_open
os.putenv = _guarded_putenv
if _orig_unsetenv is not None:
    os.unsetenv = _guarded_unsetenv
type(os.environ).__setitem__ = _guarded_environ_setitem
type(os.environ).__delitem__ = _guarded_environ_delitem
os.mkdir = _guarded_mkdir
os.makedirs = _guarded_makedirs
if _orig_mknod is not None:
    os.mknod = _guarded_mknod
if _orig_mkfifo is not None:
    os.mkfifo = _guarded_mkfifo
os.remove = _guarded_remove
os.unlink = _guarded_unlink
os.rmdir = _guarded_rmdir
os.rename = _guarded_rename
os.replace = _guarded_replace
os.truncate = _guarded_truncate
os.ftruncate = _guarded_ftruncate
os.chmod = _guarded_chmod
if _orig_fchmod is not None:
    os.fchmod = _guarded_fchmod
os.link = _guarded_link
os.symlink = _guarded_symlink
if _orig_chown is not None:
    os.chown = _guarded_chown
if _orig_lchown is not None:
    os.lchown = _guarded_lchown
if _orig_fchown is not None:
    os.fchown = _guarded_fchown
if _orig_posix_open is not None:
    posix.open = _guarded_posix_open
if _orig_posix_mkdir is not None:
    posix.mkdir = _guarded_posix_mkdir
if _orig_posix_mknod is not None:
    posix.mknod = _guarded_posix_mknod
if _orig_posix_mkfifo is not None:
    posix.mkfifo = _guarded_posix_mkfifo
posix.unlink = _guarded_posix_unlink
posix.rmdir = _guarded_posix_rmdir
if _orig_posix_chmod is not None:
    posix.chmod = _guarded_posix_chmod
pathlib.Path.unlink = _guarded_path_unlink
pathlib.Path.rmdir = _guarded_path_rmdir
pathlib.Path.rename = _guarded_path_rename
pathlib.Path.replace = _guarded_path_replace
pathlib.Path.open = _guarded_path_open
pathlib.Path.write_text = _guarded_path_write_text
pathlib.Path.write_bytes = _guarded_path_write_bytes
pathlib.Path.chmod = _guarded_path_chmod
pathlib.Path.mkdir = _guarded_path_mkdir
shutil.copyfile = _guarded_copyfile
shutil.copy = _guarded_copy
shutil.copy2 = _guarded_copy2
shutil.copytree = _guarded_copytree
sqlite3.connect = _guarded_sqlite_connect
builtins.__import__ = _guarded_import
ctypes.CDLL = _guarded_ctypes_cdll
ctypes.cdll.LoadLibrary = _guarded_ctypes_cdll
ctypes.PyDLL = _guarded_ctypes_pydll
ctypes.pydll.LoadLibrary = _guarded_ctypes_pydll_loadlibrary
if _orig__ctypes_dlopen is not None:
    _ctypes.dlopen = _guarded__ctypes_dlopen
sys.meta_path.insert(0, _BlockPeerExtensionImporter())
subprocess.Popen = _GuardedPopen
os.system = _guarded_os_system
if _orig_spawnv is not None:
    os.spawnv = _guarded_spawnv
if _orig_spawnve is not None:
    os.spawnve = _guarded_spawnve
if _orig_spawnvp is not None:
    os.spawnvp = _guarded_spawnvp
if _orig_spawnvpe is not None:
    os.spawnvpe = _guarded_spawnvpe
if _orig_execv is not None:
    os.execv = _guarded_execv
if _orig_execve is not None:
    os.execve = _guarded_execve
if _orig_execvp is not None:
    os.execvp = _guarded_execvp
if _orig_execvpe is not None:
    os.execvpe = _guarded_execvpe
if _orig_posix_spawn is not None:
    os.posix_spawn = _guarded_posix_spawn
if _orig_posix_spawnp is not None:
    os.posix_spawnp = _guarded_posix_spawnp
for _name in list(globals()):
    if _name.startswith("_orig_") and _name != "_orig_for":
        globals()[_name] = None

# Preserve an existing interpreter/site customization, but only after Praxist's
# filesystem and subprocess guards are active. Resolve candidates so an alias of
# this generated directory cannot recursively load the guard itself.
try:
    _this_file = pathlib.Path(__file__).resolve()
    for _path_entry in list(sys.path):
        if not _path_entry:
            continue
        _candidate = pathlib.Path(_path_entry) / "sitecustomize.py"
        if _candidate.is_file() and _candidate.resolve() != _this_file:
            _spec = importlib.util.spec_from_file_location("_next_sitecustomize", _candidate)
            if _spec and _spec.loader:
                _module = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_module)
            break
except Exception:
    pass
'''
    return (
        body.replace("__PRAXIST_PROTECTED_ROOT_ENV_KEYS__", repr(PROTECTED_ROOT_ENV_KEYS))
        .replace("__PRAXIST_TRUSTED_PROJECT_ENV_KEYS__", repr(TRUSTED_PROJECT_ENV_KEYS))
        .replace(
            "__PRAXIST_TRUSTED_PROJECT_EXTRA_ROOTS_ENV__", repr(TRUSTED_PROJECT_EXTRA_ROOTS_ENV)
        )
        .replace("__PRAXIST_RESOURCE_STATE_DIR_NAMES__", repr(RESOURCE_STATE_DIR_NAMES))
        .replace(
            "__PRAXIST_TRUSTED_RESOURCE_GUARD_MODULE_SUFFIXES__",
            repr(TRUSTED_RESOURCE_GUARD_MODULE_SUFFIXES),
        )
        .replace("__PRAXIST_PYTHON_GUARD_ENV_KEYS__", repr(PYTHON_GUARD_ENV_KEYS))
        .replace("__PRAXIST_IMMUTABLE_GUARD_ENV_KEYS__", repr(IMMUTABLE_GUARD_ENV_KEYS))
        .replace("__PRAXIST_GUARD_WARNING_ENV_KEY__", repr(GUARD_WARNING_ENV_KEY))
    )


def _main(argv: list[str]) -> int:
    if not argv or argv[0] not in {"validate-rm", "validate-bash", "validate-shell-argv"}:
        print(
            "usage: delete_guard.py validate-rm [rm-args...] | validate-bash COMMAND | "
            "validate-shell-argv SHELL [ARGS...]",
            file=sys.stderr,
        )
        return 2
    env = dict(os.environ)
    if argv[0] == "validate-rm":
        decision = validate_rm_argv(argv[1:], env=env, cwd=Path.cwd())
    elif argv[0] == "validate-bash":
        decision = validate_bash_command(" ".join(argv[1:]), env=env, cwd=Path.cwd())
    else:
        if len(argv) < 2:
            print("validate-shell-argv requires a shell name", file=sys.stderr)
            return 2
        command = " ".join(shlex.quote(part) for part in argv[1:])
        decision = validate_bash_command(command, env=env, cwd=Path.cwd())
    if decision.allowed:
        if decision.warning and decision.message:
            print(decision.message, file=sys.stderr)
        return 0
    print(decision.message, file=sys.stderr)
    return 126


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
