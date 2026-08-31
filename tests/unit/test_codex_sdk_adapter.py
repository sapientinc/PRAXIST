"""Official Codex Python SDK adapter lifecycle tests."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from praxist.core.credentials import CredentialRef
from praxist.core.protocol import (
    AgentRunRequest,
    CachePolicy,
    EnvPolicy,
    ModelCallSpec,
    ToolPermissionSet,
)
from praxist.core.runtimes import AgentRuntimeExecutionContext
from praxist.plugins.agent_runtimes.codex_sdk import adapter
from praxist.plugins.agent_runtimes.codex_sdk.adapter import CodexSdkRuntime, create_runtime


@dataclass(frozen=True)
class _Payload:
    data: dict[str, Any]

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class _Notification:
    method: str
    payload: _Payload


def _notification(method: str, payload: dict[str, Any]) -> _Notification:
    return _Notification(method=method, payload=_Payload(payload))


def _successful_notifications(text: str) -> list[_Notification]:
    return [
        _notification(
            "item/completed",
            {"item": {"id": "message-1", "type": "agentMessage", "text": text}},
        ),
        _notification(
            "thread/tokenUsage/updated",
            {
                "tokenUsage": {
                    "total": {
                        "inputTokens": 10,
                        "cachedInputTokens": 4,
                        "outputTokens": 3,
                        "reasoningOutputTokens": 1,
                        "totalTokens": 13,
                    }
                }
            },
        ),
        _notification(
            "turn/completed",
            {"turn": {"id": "turn-1", "status": "completed", "items": []}},
        ),
    ]


async def _wait_until(predicate: Any, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true before timeout")
        await asyncio.sleep(0.01)


def _request(
    run_dir: str,
    *,
    request_id: str = "request-1",
    prompt: str = "Inspect the task.",
    provider_ref: str = "model_provider:openai",
    runtime_ref: str = "agent_runtime:codex_sdk",
    runtime_options: dict[str, Any] | None = None,
    timeout_seconds: int = 60,
    tool_permissions: ToolPermissionSet | None = None,
    tool_servers: list[dict[str, Any]] | None = None,
    exposed_env_keys: list[str] | None = None,
    model: str | None = None,
    credential_ref: CredentialRef | None = None,
) -> AgentRunRequest:
    options = {"run_dir": run_dir}
    options.update(runtime_options or {})
    return AgentRunRequest(
        request_id=request_id,
        run_id="run-1",
        stage_id="stage-1",
        role_ref=None,
        agent_runtime_ref=runtime_ref,
        prompt_ref={"text": prompt},
        system_prompt_ref=None,
        cwd=run_dir,
        model_profile_ref="model_profile:default",
        model_call=ModelCallSpec(
            profile_id="default",
            provider_ref=provider_ref,
            api_format="responses",
            model=model or ("deepseek-v4-pro" if "deepseek" in provider_ref else "gpt-5"),
            parameters={},
            credential_ref=credential_ref,
        ),
        tool_permissions=tool_permissions or ToolPermissionSet(),
        tool_servers=list(tool_servers or []),
        env_policy=EnvPolicy(exposed_env_keys=list(exposed_env_keys or [])),
        credential_ref=credential_ref,
        credential_mode="env",
        budget_grant_id=None,
        artifact_scope="run",
        timeout_seconds=timeout_seconds,
        cache_policy=CachePolicy(mode="deterministic_no_cache", frozen_prefix_hash=None),
        runtime_options=options,
    )


class _CodexConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _ApprovalMode:
    deny_all = "approval:deny_all"
    auto_review = "approval:auto_review"


class _Sandbox:
    read_only = "sandbox:read_only"
    workspace_write = "sandbox:workspace_write"
    full_access = "sandbox:full_access"


class _ReasoningEffort:
    none = "effort:none"
    low = "effort:low"
    medium = "effort:medium"
    high = "effort:high"
    xhigh = "effort:xhigh"


class _WaitIterator:
    def __init__(
        self,
        event: threading.Event,
        notification: _Notification | None,
    ) -> None:
        self._event = event
        self._notification = notification
        self._done = False

    def __iter__(self) -> _WaitIterator:
        return self

    def __next__(self) -> _Notification:
        if self._done:
            raise StopIteration
        self._event.wait(timeout=10)
        self._done = True
        if self._notification is None:
            raise StopIteration
        return self._notification


class _Turn:
    def __init__(self, client: _Client, prompt: str, mode: str) -> None:
        self.client = client
        self.prompt = prompt
        self.mode = mode
        self.interrupted = threading.Event()
        self.interrupt_calls = 0

    def stream(self) -> Any:
        if self.mode in {"schema_error", "schema_error_after_message"}:
            message = (
                "Invalid schema for response_format: 'additionalProperties' "
                "is required to be supplied and to be false"
            )
            notifications = (
                [
                    _notification(
                        "item/completed",
                        {
                            "item": {
                                "id": "message-before-schema-error",
                                "type": "agentMessage",
                                "text": "partial output",
                            }
                        },
                    )
                ]
                if self.mode == "schema_error_after_message"
                else []
            )
            notifications.extend(
                (
                    _notification(
                        "error",
                        {
                            "error": {
                                "code": "invalid_json_schema",
                                "message": message,
                            },
                            "willRetry": False,
                        },
                    ),
                    _notification(
                        "turn/completed",
                        {
                            "turn": {
                                "id": "turn-schema-error",
                                "status": "failed",
                                "error": {"message": message},
                                "items": [],
                            }
                        },
                    ),
                )
            )
            return iter(notifications)
        if self.mode == "interruptible":
            return _WaitIterator(
                self.interrupted,
                _notification(
                    "turn/completed",
                    {"turn": {"id": "turn-stop", "status": "completed", "items": []}},
                ),
            )
        if self.mode == "stuck":
            return _WaitIterator(self.client.closed_event, None)
        return iter(_successful_notifications(f"reply:{self.prompt}"))

    def interrupt(self) -> None:
        self.interrupt_calls += 1
        if self.client.harness.block_phase == "interrupt":
            self.client.harness.phase_started.set()
            self.client.closed_event.wait(timeout=10)
            return
        if self.mode != "stuck":
            self.interrupted.set()


class _Thread:
    def __init__(self, client: _Client, kwargs: dict[str, Any]) -> None:
        self.client = client
        self.kwargs = kwargs

    def turn(self, prompt: str, **kwargs: Any) -> _Turn:
        self.client.turn_calls.append((prompt, kwargs))
        if self.client.harness.turn_errors:
            raise self.client.harness.turn_errors.pop(0)
        block_prompt = self.client.harness.block_turn_prompt
        if self.client.harness.block_phase == "turn" and (
            block_prompt is None or block_prompt == prompt
        ):
            self.client.harness.phase_started.set()
            if block_prompt is None:
                self.client.closed_event.wait(timeout=10)
            else:
                self.client.harness.phase_release.wait(timeout=10)
        mode = (
            self.client.harness.turn_mode_sequence.pop(0)
            if self.client.harness.turn_mode_sequence
            else self.client.harness.turn_modes.get(prompt, self.client.harness.turn_mode)
        )
        turn = _Turn(self.client, prompt, mode)
        self.client.turns.append(turn)
        return turn


class _Client:
    def __init__(self, harness: _SdkHarness, config: _CodexConfig) -> None:
        self.harness = harness
        self.config = config
        self.thread_calls: list[dict[str, Any]] = []
        self.turn_calls: list[tuple[str, dict[str, Any]]] = []
        self.turns: list[_Turn] = []
        self.close_calls = 0
        self.account_calls: list[bool] = []
        self.closed_event = threading.Event()

    def account(self, *, refresh_token: bool = False) -> Any:
        self.account_calls.append(refresh_token)
        account = SimpleNamespace(type=self.harness.account_type)
        return SimpleNamespace(account=SimpleNamespace(root=account))

    def thread_start(self, **kwargs: Any) -> _Thread:
        self.thread_calls.append(kwargs)
        if self.harness.block_phase == "thread_start":
            self.harness.phase_started.set()
            self.closed_event.wait(timeout=10)
        return _Thread(self, kwargs)

    def close(self) -> None:
        self.close_calls += 1
        self.closed_event.set()


class _SdkHarness:
    def __init__(
        self,
        *,
        turn_mode: str = "normal",
        client_error: Exception | None = None,
        config_error: Exception | None = None,
        block_phase: str | None = None,
        block_turn_prompt: str | None = None,
        turn_modes: dict[str, str] | None = None,
        turn_mode_sequence: list[str] | None = None,
        turn_errors: list[Exception] | None = None,
        account_type: str = "chatgpt",
    ) -> None:
        self.turn_mode = turn_mode
        self.client_error = client_error
        self.config_error = config_error
        self.block_phase = block_phase
        self.block_turn_prompt = block_turn_prompt
        self.turn_modes = dict(turn_modes or {})
        self.turn_mode_sequence = list(turn_mode_sequence or [])
        self.turn_errors = list(turn_errors or [])
        self.account_type = account_type
        self.phase_started = threading.Event()
        self.phase_release = threading.Event()
        self.configs: list[_CodexConfig] = []
        self.clients: list[_Client] = []

    def sdk(self) -> dict[str, Any]:
        harness = self

        def config(**kwargs: Any) -> _CodexConfig:
            if harness.config_error is not None:
                raise harness.config_error
            value = _CodexConfig(**kwargs)
            harness.configs.append(value)
            return value

        def codex(config_value: _CodexConfig) -> _Client:
            if harness.client_error is not None:
                raise harness.client_error
            if harness.block_phase == "constructor":
                harness.phase_started.set()
                harness.phase_release.wait(timeout=10)
            client = _Client(harness, config_value)
            harness.clients.append(client)
            return client

        return {
            "ApprovalMode": _ApprovalMode,
            "Codex": codex,
            "CodexConfig": config,
            "ReasoningEffort": _ReasoningEffort,
            "Sandbox": _Sandbox,
        }


class _Relay:
    provider = "deepseek"
    port = 43101
    base_url = "http://127.0.0.1:43101/v1"

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class RuntimeFactoryAndValidationTest(unittest.IsolatedAsyncioTestCase):
    def test_reasoning_policy_maps_at_the_provider_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            deepseek = lambda policy: _request(  # noqa: E731 - compact policy matrix.
                tmp,
                provider_ref="model_provider:deepseek_alias",
                runtime_options={"reasoning_effort": policy},
            )
            self.assertEqual(adapter._relay_reasoning_options(deepseek("auto")), (None, ()))
            self.assertEqual(
                adapter._relay_reasoning_options(deepseek("off")),
                ({"thinking": {"type": "disabled"}}, ("reasoning_effort",)),
            )
            self.assertEqual(
                adapter._relay_reasoning_options(deepseek("max")),
                (
                    {"thinking": {"type": "enabled"}, "reasoning_effort": "max"},
                    (),
                ),
            )
            native = _request(tmp, runtime_options={"reasoning_effort": "high"})
            self.assertEqual(
                adapter._reasoning_effort(native, _ReasoningEffort, {"OPENAI_API_KEY": "x"}),
                _ReasoningEffort.high,
            )
            openrouter = lambda policy: _request(  # noqa: E731 - compact policy matrix.
                tmp,
                provider_ref="model_provider:openrouter",
                runtime_options={"reasoning_effort": policy},
            )
            self.assertEqual(
                adapter._relay_reasoning_options(openrouter("off")),
                ({"reasoning": {"effort": "none"}}, ()),
            )
            self.assertEqual(
                adapter._relay_reasoning_options(openrouter("high")),
                ({"reasoning": {"effort": "high"}}, ()),
            )

    def test_deepseek_client_identity_includes_reasoning_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = _request(tmp, provider_ref="model_provider:deepseek_alias")
            off = replace(base, runtime_options={**base.runtime_options, "reasoning_effort": "off"})
            maximum = replace(
                base,
                runtime_options={**base.runtime_options, "reasoning_effort": "max"},
            )
            env = {"DEEPSEEK_API_KEY": "test-key"}
            self.assertNotEqual(adapter._client_key(off, env), adapter._client_key(maximum, env))

    async def test_factory_and_offline_conformance_entrypoint(self) -> None:
        runtime = create_runtime()
        self.addAsyncCleanup(runtime.aclose)
        with tempfile.TemporaryDirectory() as tmp:
            result = runtime.execute_sync(_request(tmp))

        self.assertIsInstance(runtime, CodexSdkRuntime)
        self.assertTrue(result.success)
        self.assertEqual(result.terminal_status, "completed")
        self.assertEqual(result.events[-1].type, "final_result")

    async def test_runtime_mismatch_and_shell_free_requirement_fail_before_sdk(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        context = AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "test-key"})
        with tempfile.TemporaryDirectory() as tmp, patch.object(adapter, "_load_sdk") as load:
            mismatch = await runtime.execute(
                _request(tmp, runtime_ref="agent_runtime:other"),
                context,
            )
            no_shell = await runtime.execute(
                _request(tmp, runtime_options={"require_no_shell_runtime": True}),
                context,
            )

        self.assertFalse(mismatch.success)
        self.assertIn("runtime mismatch", mismatch.error or "")
        self.assertFalse(no_shell.success)
        self.assertIn("shell-free", no_shell.error or "")
        load.assert_not_called()

    async def test_read_only_requirement_and_invalid_sandbox_fail_before_sdk(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        context = AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "test-key"})
        with tempfile.TemporaryDirectory() as tmp, patch.object(adapter, "_load_sdk") as load:
            writable = await runtime.execute(
                _request(tmp, runtime_options={"require_read_only_runtime": True}),
                context,
            )
            invalid = await runtime.execute(
                _request(
                    tmp,
                    runtime_options={
                        "sandbox_intent": {
                            "filesystem": "invalid",
                            "network": "on",
                            "approval": "auto",
                        }
                    },
                ),
                context,
            )

        self.assertFalse(writable.success)
        self.assertIn("read-only runtime request", writable.error or "")
        self.assertFalse(invalid.success)
        self.assertIn("filesystem must be one of", invalid.error or "")
        load.assert_not_called()

    async def test_missing_provider_key_is_normalized_without_starting_client(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = await runtime.execute(
                _request(tmp, provider_ref="model_provider:deepseek_alias"),
                AgentRuntimeExecutionContext(env={}),
            )

        self.assertFalse(result.success)
        self.assertIn("DEEPSEEK_API_KEY", result.error or "")
        self.assertEqual(harness.clients, [])


class SdkImportTest(unittest.TestCase):
    def test_missing_sdk_has_actionable_optional_dependency_error(self) -> None:
        with (
            patch.dict(sys.modules, {"openai_codex": None}),
            self.assertRaisesRegex(RuntimeError, r"install praxist\[codex\]"),
        ):
            adapter._load_sdk()


class RuntimeExecutionTest(unittest.IsolatedAsyncioTestCase):
    async def test_task_python_shell_environment_drops_runner_import_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner_imports = root / "runner-imports"
            runner_imports.mkdir()
            (runner_imports / "runner_only.py").write_text(
                "VALUE = 'runner-visible'\n",
                encoding="utf-8",
            )
            probe = root / "probe.py"
            probe.write_text(
                "import importlib.util\nassert importlib.util.find_spec('runner_only') is None\n",
                encoding="utf-8",
            )
            task_python = root / "task-python"
            task_python.write_text(
                f'#!/bin/sh\nexec {sys.executable} "$@"\n',
                encoding="utf-8",
            )
            task_python.chmod(0o755)
            request = _request(
                tmp,
                exposed_env_keys=[
                    "PRAXIST_RUNNER_PYTHON",
                    "PRAXIST_TASK_PYTHON",
                    "PRAXIST_TASK_RUNTIME_ENV_KEYS",
                    "PYTHONPATH",
                    "PYTHONHOME",
                ],
            )
            shell_env = adapter._subprocess_env(
                request,
                {
                    "PRAXIST_RUNNER_PYTHON": sys.executable,
                    "PRAXIST_TASK_PYTHON": str(task_python),
                    "PYTHONPATH": str(runner_imports),
                    "PYTHONHOME": "/runner/python-home",
                },
            )
            completed = subprocess.run(
                [shell_env["PRAXIST_TASK_PYTHON"], str(probe)],
                env={**shell_env, "PATH": os.environ.get("PATH", os.defpath)},
                check=False,
            )
            runner_env = dict(shell_env)
            runner_env["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(runner_imports), shell_env.get("PYTHONPATH")))
            )
            runner_completed = subprocess.run(
                [
                    shell_env["PRAXIST_RUNNER_PYTHON"],
                    "-c",
                    "import runner_only; assert runner_only.VALUE == 'runner-visible'",
                ],
                env={**runner_env, "PATH": os.environ.get("PATH", os.defpath)},
                check=False,
            )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(runner_completed.returncode, 0)
        self.assertNotIn("PYTHONPATH", shell_env)
        self.assertNotIn("PYTHONHOME", shell_env)

        task_owned = adapter._subprocess_env(
            request,
            {
                "PRAXIST_TASK_PYTHON": sys.executable,
                "PRAXIST_TASK_RUNTIME_ENV_KEYS": "PYTHONPATH,PYTHONHOME",
                "PYTHONPATH": "/task/imports",
                "PYTHONHOME": "/task/python-home",
            },
        )
        self.assertEqual(task_owned["PYTHONPATH"], "/task/imports")
        self.assertEqual(task_owned["PYTHONHOME"], "/task/python-home")

    async def test_openai_direct_maps_thread_turn_mcp_sandbox_and_callbacks(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        observed: list[Any] = []
        permissions = ToolPermissionSet(
            mode="allow_list",
            allowed_tools=["mcp__evaluation_tools__evaluate"],
            denied_tools=["inspect"],
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            patch.object(adapter, "start_relay") as relay,
        ):
            request = _request(
                tmp,
                prompt="canonical user prompt",
                runtime_options={
                    "system_prompt": "system instructions",
                    "premium_mode": True,
                    "output_schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    "sandbox_intent": {
                        "filesystem": "workspace_write",
                        "network": "off",
                        "approval": "auto",
                    },
                },
                tool_permissions=permissions,
                tool_servers=[
                    {
                        "server_name": "evaluation-tools",
                        "factory": "task.evaluator:create_server",
                        "tool_names": ["evaluate", "inspect"],
                    }
                ],
                exposed_env_keys=["TASK_MODE", "SECRET_TOKEN"],
            )
            result = await runtime.execute(
                request,
                AgentRuntimeExecutionContext(
                    env={
                        "OPENAI_API_KEY": "openai-test-key",
                        "PRAXIST_RUN_DIR": tmp,
                        "TASK_MODE": "test",
                        "SECRET_TOKEN": "do-not-pass-to-shell",
                    },
                    message_callback=observed.append,
                ),
            )

        self.assertTrue(result.success, result.error)
        relay.assert_not_called()
        self.assertEqual(len(harness.clients), 1)
        client = harness.clients[0]
        thread_call = client.thread_calls[0]
        self.assertEqual(thread_call["approval_mode"], _ApprovalMode.deny_all)
        self.assertEqual(thread_call["base_instructions"], "system instructions")
        self.assertEqual(thread_call["model"], "gpt-5")
        self.assertEqual(thread_call["model_provider"], "openai")
        self.assertEqual(thread_call["sandbox"], _Sandbox.workspace_write)
        self.assertTrue(thread_call["ephemeral"])
        self.assertEqual(thread_call["service_name"], "praxist")
        config = thread_call["config"]
        self.assertFalse(config["sandbox_workspace_write"]["network_access"])
        self.assertIn("evaluation-tools", config["mcp_servers"])
        self.assertEqual(
            config["mcp_servers"]["evaluation-tools"]["enabled_tools"],
            ["evaluate"],
        )
        self.assertEqual(
            config["mcp_servers"]["evaluation-tools"]["default_tools_approval_mode"],
            "approve",
        )
        shell_env = config["shell_environment_policy"]["set"]
        self.assertEqual(shell_env["TASK_MODE"], "test")
        self.assertNotIn("SECRET_TOKEN", shell_env)
        self.assertEqual(config["shell_environment_policy"]["inherit"], "none")
        self.assertFalse(config["features"]["shell_tool"])
        self.assertFalse(config["features"]["multi_agent"])
        self.assertFalse(config["include_apply_patch_tool"])
        self.assertEqual(config["web_search"], "disabled")
        prompt, turn_options = client.turn_calls[0]
        self.assertEqual(prompt, "canonical user prompt")
        self.assertEqual(turn_options["effort"], _ReasoningEffort.xhigh)
        self.assertEqual(turn_options["output_schema"]["required"], ["answer"])
        self.assertEqual(
            [event.type for event in observed],
            [
                "agent_run_started",
                "assistant_text",
                "usage",
                "final_result",
            ],
        )

    async def test_non_strict_schema_is_omitted_before_first_endpoint_request(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        observed: list[Any] = []
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
        ):
            result = await runtime.execute(
                _request(tmp, runtime_options={"output_schema": schema}),
                AgentRuntimeExecutionContext(
                    env={"OPENAI_API_KEY": "test-key"},
                    message_callback=observed.append,
                ),
            )

        self.assertTrue(result.success, result.error)
        calls = harness.clients[0].turn_calls
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0][1]["output_schema"])
        warnings = [event for event in result.events if event.type == "runtime_warning"]
        self.assertEqual(len(warnings), 1)
        self.assertFalse(warnings[0].payload["will_retry"])
        self.assertEqual(sum(event.type == "runtime_warning" for event in observed), 1)

    async def test_non_strict_schema_preflight_keeps_clean_final_result(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        observed: list[Any] = []
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
        ):
            result = await runtime.execute(
                _request(tmp, runtime_options={"output_schema": schema}),
                AgentRuntimeExecutionContext(
                    env={"OPENAI_API_KEY": "test-key"},
                    message_callback=observed.append,
                ),
            )

        self.assertTrue(result.success, result.error)
        self.assertEqual(len(harness.clients[0].turn_calls), 1)
        self.assertIsNone(harness.clients[0].turn_calls[0][1]["output_schema"])
        self.assertEqual(sum(event.type == "agent_run_started" for event in result.events), 1)
        self.assertEqual(sum(event.type == "runtime_warning" for event in result.events), 1)
        self.assertFalse(any(event.type == "runtime_error" for event in result.events))
        self.assertEqual(sum(event.type == "runtime_warning" for event in observed), 1)

    async def test_schema_preflight_does_not_mask_unrelated_or_strict_schema_errors(self) -> None:
        cases = (
            (
                {"type": "object"},
                RuntimeError("provider connection failed"),
            ),
            (
                {"type": "object", "additionalProperties": False},
                RuntimeError("invalid_json_schema: additionalProperties rejected"),
            ),
        )
        for schema, error in cases:
            with self.subTest(schema=schema, error=str(error)):
                runtime = CodexSdkRuntime()
                harness = _SdkHarness(turn_errors=[error])
                with (
                    tempfile.TemporaryDirectory() as tmp,
                    patch.object(
                        adapter,
                        "_load_sdk",
                        side_effect=lambda harness=harness: harness.sdk(),
                    ),
                ):
                    result = await runtime.execute(
                        _request(tmp, runtime_options={"output_schema": schema}),
                        AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "test-key"}),
                    )
                await runtime.aclose()

                self.assertFalse(result.success)
                self.assertEqual(len(harness.clients[0].turn_calls), 1)

    async def test_streamed_schema_rejection_is_not_replayed_after_visible_output(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(turn_mode_sequence=["schema_error_after_message"])
        schema = {"type": "object"}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
        ):
            result = await runtime.execute(
                _request(tmp, runtime_options={"output_schema": schema}),
                AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "test-key"}),
            )

        self.assertFalse(result.success)
        self.assertEqual(len(harness.clients[0].turn_calls), 1)
        self.assertTrue(any(event.type == "assistant_text" for event in result.events))

    async def test_chatgpt_subscription_uses_saved_auth_with_run_local_state(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import (
            SUBSCRIPTION_ENV_KEYS,
            chatgpt_credential_key_id,
        )

        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            codex_home = root / "operator-codex"
            codex_home.mkdir()
            auth_file = codex_home / "auth.json"
            config_file = codex_home / "config.toml"
            auth_file.write_text(
                '{"auth_mode":"chatgpt","tokens":{"account_id":"account-a"}}',
                encoding="utf-8",
            )
            config_file.write_text('model_provider="operator-provider"\n', encoding="utf-8")
            before = {
                path.name: path.read_bytes() for path in codex_home.iterdir() if path.is_file()
            }
            credential = CredentialRef(
                scope="model_provider",
                provider="openai_compatible",
                target_ref="model_provider:openai_compatible",
                key_id=chatgpt_credential_key_id(codex_home),
                source="runtime_session",
            )
            with (
                patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
                patch.object(adapter, "start_relay") as relay,
                patch.object(adapter, "resolve_codex_binary", return_value="/sdk/codex"),
                patch.object(adapter, "verify_chatgpt_login") as verify_login,
                patch.dict(
                    "os.environ",
                    {
                        "CODEX_HOME": str(codex_home),
                        "OPENAI_API_KEY": "host-api-key-must-not-win",
                        "CODEX_ACCESS_TOKEN": "host-token-must-not-win",
                        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/keyring/bus",
                        "XDG_RUNTIME_DIR": "/keyring/runtime",
                    },
                    clear=False,
                ),
            ):
                result = await runtime.execute(
                    _request(
                        str(run_dir),
                        provider_ref="model_provider:openai_compatible",
                        model="gpt-5.6-luna",
                        credential_ref=credential,
                    ),
                    AgentRuntimeExecutionContext(env={}),
                )

            self.assertTrue(result.success, result.error)
            relay.assert_not_called()
            verify_login.assert_called_once()
            client = harness.clients[0]
            self.assertEqual(client.account_calls, [False])
            self.assertEqual(client.thread_calls[0]["model"], "gpt-5.6-luna")
            self.assertEqual(client.turn_calls[0][1]["effort"], _ReasoningEffort.xhigh)
            config = harness.configs[0].kwargs
            staged_home = Path(config["env"]["CODEX_HOME"])
            self.assertNotEqual(staged_home, codex_home)
            self.assertEqual(config["codex_bin"], "/sdk/codex")
            self.assertEqual((staged_home / "auth.json").read_bytes(), auth_file.read_bytes())
            self.assertEqual(
                config["env"]["DBUS_SESSION_BUS_ADDRESS"],
                "unix:path=/keyring/bus",
            )
            self.assertEqual(config["env"]["XDG_RUNTIME_DIR"], "/keyring/runtime")
            for key in SUBSCRIPTION_ENV_KEYS:
                self.assertEqual(config["env"][key], "")
            overrides = config["config_overrides"]
            for expected in (
                'model_provider="openai"',
                "mcp_servers={}",
                "notify=[]",
                "skills.bundled=[]",
                "features.plugins=false",
                'cli_auth_credentials_store="file"',
            ):
                self.assertIn(expected, overrides)
            sqlite_override = next(value for value in overrides if value.startswith("sqlite_home="))
            log_override = next(value for value in overrides if value.startswith("log_dir="))
            self.assertIn(str(run_dir / "runtime_state" / "codex_sdk"), sqlite_override)
            self.assertIn(str(run_dir / "runtime_state" / "codex_sdk"), log_override)
            after = {
                path.name: path.read_bytes() for path in codex_home.iterdir() if path.is_file()
            }
            self.assertEqual(after, before)
            await runtime.aclose()
            self.assertFalse(staged_home.exists())

    async def test_chatgpt_subscription_preserves_keyring_home_identity(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import chatgpt_credential_key_id

        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            codex_home = root / "operator-codex"
            codex_home.mkdir()
            marker = codex_home / "config.toml"
            marker.write_text('model = "operator-default"\n', encoding="utf-8")
            credential = CredentialRef(
                scope="model_provider",
                provider="openai_compatible",
                target_ref="model_provider:openai_compatible",
                key_id=chatgpt_credential_key_id(codex_home),
                source="runtime_session",
            )
            with (
                patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
                patch.object(adapter, "start_relay") as relay,
                patch.object(adapter, "resolve_codex_binary", return_value="/sdk/codex"),
                patch.object(adapter, "verify_chatgpt_login"),
                patch.dict("os.environ", {"CODEX_HOME": str(codex_home)}, clear=False),
            ):
                result = await runtime.execute(
                    _request(
                        str(run_dir),
                        provider_ref="model_provider:openai_compatible",
                        credential_ref=credential,
                    ),
                    AgentRuntimeExecutionContext(env={}),
                )

            self.assertTrue(result.success, result.error)
            relay.assert_not_called()
            client = harness.clients[0]
            self.assertEqual(client.account_calls, [False])
            config = harness.configs[0].kwargs
            self.assertEqual(Path(config["env"]["CODEX_HOME"]), codex_home)
            self.assertIn(
                'cli_auth_credentials_store="keyring"',
                config["config_overrides"],
            )
            await runtime.aclose()
            self.assertEqual(marker.read_text(encoding="utf-8"), 'model = "operator-default"\n')

    async def test_chatgpt_subscription_rejects_non_chatgpt_app_server_account(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import chatgpt_credential_key_id

        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(account_type="apiKey")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            codex_home = root / "operator-codex"
            codex_home.mkdir()
            credential = CredentialRef(
                scope="model_provider",
                provider="openai_compatible",
                target_ref="model_provider:openai_compatible",
                key_id=chatgpt_credential_key_id(codex_home),
                source="runtime_session",
            )
            with (
                patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
                patch.object(adapter, "start_relay") as relay,
                patch.object(adapter, "resolve_codex_binary", return_value="/sdk/codex"),
                patch.object(adapter, "verify_chatgpt_login"),
                patch.dict("os.environ", {"CODEX_HOME": str(codex_home)}, clear=False),
            ):
                result = await runtime.execute(
                    _request(
                        str(run_dir),
                        provider_ref="model_provider:openai_compatible",
                        credential_ref=credential,
                    ),
                    AgentRuntimeExecutionContext(env={}),
                )

        self.assertFalse(result.success)
        self.assertIn("API-key fallback is disabled", result.error or "")
        relay.assert_not_called()
        self.assertEqual(harness.clients[0].close_calls, 1)
        self.assertEqual(runtime._clients, {})

    async def test_chatgpt_subscription_rejects_auth_home_changed_after_startup(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import chatgpt_credential_key_id

        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_home = root / "original"
            current_home = root / "current"
            original_home.mkdir()
            current_home.mkdir()
            credential = CredentialRef(
                scope="model_provider",
                provider="openai_compatible",
                target_ref="model_provider:openai_compatible",
                key_id=chatgpt_credential_key_id(original_home),
                source="runtime_session",
            )
            with (
                patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
                patch.object(adapter, "resolve_codex_binary", return_value="/sdk/codex"),
                patch.object(adapter, "verify_chatgpt_login"),
                patch.dict("os.environ", {"CODEX_HOME": str(current_home)}, clear=False),
            ):
                result = await runtime.execute(
                    _request(
                        str(root / "run"),
                        provider_ref="model_provider:openai_compatible",
                        credential_ref=credential,
                    ),
                    AgentRuntimeExecutionContext(env={}),
                )

        self.assertFalse(result.success)
        self.assertIn("changed after startup", result.error or "")
        self.assertEqual(harness.clients, [])

    async def test_request_api_key_takes_precedence_over_subscription_reference(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import chatgpt_credential_key_id

        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "operator"
            home.mkdir()
            credential = CredentialRef(
                scope="model_provider",
                provider="openai_compatible",
                target_ref="model_provider:openai_compatible",
                key_id=chatgpt_credential_key_id(home),
                source="runtime_session",
            )
            with (
                patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
                patch.object(adapter, "verify_chatgpt_login") as verify_login,
                patch.object(adapter, "stage_chatgpt_home") as stage_auth,
            ):
                result = await runtime.execute(
                    _request(
                        tmp,
                        provider_ref="model_provider:openai_compatible",
                        credential_ref=credential,
                    ),
                    AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "request-api-key"}),
                )

        self.assertTrue(result.success, result.error)
        verify_login.assert_not_called()
        stage_auth.assert_not_called()
        config = harness.configs[0].kwargs
        self.assertIsNone(config["codex_bin"])
        self.assertEqual(config["env"]["OPENAI_API_KEY"], "request-api-key")
        self.assertFalse(
            any(
                value.startswith("cli_auth_credentials_store=")
                for value in config["config_overrides"]
            )
        )
        self.assertEqual(
            harness.clients[0].turn_calls[0][1]["effort"],
            _ReasoningEffort.xhigh,
        )

    async def test_client_creation_is_single_flight_and_concurrent_turns_are_independent(
        self,
    ) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
        ):
            context = AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "test-key"})
            first, second = await asyncio.gather(
                runtime.execute(
                    _request(tmp, request_id="request-a", prompt="prompt-a"),
                    context,
                ),
                runtime.execute(
                    _request(tmp, request_id="request-b", prompt="prompt-b"),
                    context,
                ),
            )
            third = await runtime.execute(
                _request(tmp, request_id="request-c", prompt="prompt-c"),
                context,
            )

        self.assertEqual(len(harness.clients), 1)
        self.assertEqual(len(harness.clients[0].thread_calls), 3)
        self.assertEqual(
            {prompt for prompt, _options in harness.clients[0].turn_calls},
            {"prompt-a", "prompt-b", "prompt-c"},
        )
        outputs = {result.events[1].payload["text"] for result in (first, second, third)}
        self.assertEqual(outputs, {"reply:prompt-a", "reply:prompt-b", "reply:prompt-c"})

    async def test_deepseek_uses_private_relay_configuration_and_closes_both(self) -> None:
        runtime = CodexSdkRuntime()
        harness = _SdkHarness()
        relay = _Relay()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            patch.object(adapter, "start_relay", return_value=relay) as start,
        ):
            result = await runtime.execute(
                _request(
                    tmp,
                    provider_ref="model_provider:deepseek_alias",
                    model="deepseek-v4-pro[1m]",
                ),
                AgentRuntimeExecutionContext(env={"DEEPSEEK_API_KEY": "deepseek-test-key"}),
            )
            await runtime.aclose()

        self.assertTrue(result.success, result.error)
        start.assert_called_once()
        self.assertEqual(start.call_args.kwargs["provider"], "deepseek")
        self.assertEqual(start.call_args.kwargs["api_key"], "deepseek-test-key")
        config = harness.configs[0].kwargs
        overrides = config["config_overrides"]
        self.assertIn('model_provider="praxist_relay"', overrides)
        self.assertTrue(any(relay.base_url in value for value in overrides))
        self.assertEqual(config["env"]["DEEPSEEK_API_KEY"], "deepseek-test-key")
        self.assertEqual(config["env"]["OPENAI_API_KEY"], "deepseek-test-key")
        self.assertTrue(str(config["env"]["CODEX_HOME"]).endswith("/home"))
        self.assertEqual(harness.clients[0].thread_calls[0]["model"], "deepseek-v4-pro")
        self.assertTrue(result.events[-1].payload["relay_used"])
        self.assertEqual(harness.clients[0].close_calls, 1)
        self.assertEqual(relay.close_calls, 1)

    async def test_openrouter_relay_uses_stable_non_secret_run_session_id(self) -> None:
        runtime = CodexSdkRuntime()
        harness = _SdkHarness()
        relay = _Relay()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            patch.object(adapter, "start_relay", return_value=relay) as start,
        ):
            request = _request(
                tmp,
                provider_ref="model_provider:openrouter",
                model="provider/model",
            )
            result = await runtime.execute(
                request,
                AgentRuntimeExecutionContext(env={"OPENROUTER_API_KEY": "secret-key"}),
            )
            await runtime.aclose()

        self.assertTrue(result.success, result.error)
        session_id = start.call_args.kwargs["upstream_session_id"]
        self.assertRegex(session_id, r"^praxist-[0-9a-f]{32}$")
        self.assertEqual(session_id, adapter._openrouter_session_id(request))
        self.assertNotIn("secret", session_id)

        other_profile = replace(
            request,
            model_call=replace(request.model_call, model="provider/other-model"),
        )
        self.assertEqual(session_id, adapter._openrouter_session_id(other_profile))

    async def test_host_credentials_are_blanked_from_app_server_environment(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            patch.dict(
                "os.environ",
                {
                    "OPENROUTER_API_KEY": "host-secret",
                    "UNRELATED_TOKEN": "host-token",
                    "PATH": "/usr/bin",
                },
                clear=True,
            ),
        ):
            result = await runtime.execute(
                _request(tmp),
                AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "request-key"}),
            )

        self.assertTrue(result.success, result.error)
        client_env = harness.configs[0].kwargs["env"]
        self.assertEqual(client_env["OPENAI_API_KEY"], "request-key")
        self.assertEqual(client_env["OPENROUTER_API_KEY"], "")
        self.assertEqual(client_env["UNRELATED_TOKEN"], "")
        self.assertNotIn("PATH", client_env)

    async def test_explicit_task_credential_reaches_task_children_not_app_server(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
        ):
            request = _request(
                tmp,
                exposed_env_keys=[
                    "HF_TOKEN",
                    "OPENROUTER_API_KEY",
                    "PRAXIST_TASK_RUNTIME_ENV_KEYS",
                ],
            )
            result = await runtime.execute(
                request,
                AgentRuntimeExecutionContext(
                    env={
                        "OPENAI_API_KEY": "request-key",
                        "OPENROUTER_API_KEY": "unrelated-model-key",
                        "HF_TOKEN": "task-token",
                        "PRAXIST_TASK_RUNTIME_ENV_KEYS": "HF_TOKEN",
                    }
                ),
            )

        self.assertTrue(result.success, result.error)
        thread_config = harness.clients[0].thread_calls[0]["config"]
        shell_env = thread_config["shell_environment_policy"]["set"]
        self.assertEqual(shell_env["HF_TOKEN"], "task-token")
        self.assertNotIn("OPENROUTER_API_KEY", shell_env)
        client_env = harness.configs[0].kwargs["env"]
        self.assertNotEqual(client_env.get("HF_TOKEN"), "task-token")
        self.assertFalse(client_env.get("OPENROUTER_API_KEY"))

    async def test_client_constructor_failure_closes_relay_and_leaves_no_cache(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(client_error=RuntimeError("app server failed"))
        relay = _Relay()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            patch.object(adapter, "start_relay", return_value=relay),
        ):
            result = await runtime.execute(
                _request(tmp, provider_ref="model_provider:deepseek_alias"),
                AgentRuntimeExecutionContext(env={"DEEPSEEK_API_KEY": "deepseek-test-key"}),
            )

        self.assertFalse(result.success)
        self.assertIn("app server failed", result.error or "")
        self.assertEqual(relay.close_calls, 1)
        self.assertEqual(runtime._clients, {})

    async def test_config_failure_closes_relay_before_client_ownership_transfer(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(config_error=RuntimeError("config failed"))
        relay = _Relay()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            patch.object(adapter, "start_relay", return_value=relay),
        ):
            result = await runtime.execute(
                _request(tmp, provider_ref="model_provider:deepseek_alias"),
                AgentRuntimeExecutionContext(env={"DEEPSEEK_API_KEY": "deepseek-test-key"}),
            )

        self.assertFalse(result.success)
        self.assertIn("config failed", result.error or "")
        self.assertEqual(relay.close_calls, 1)
        self.assertEqual(runtime._clients, {})

    async def test_subscription_config_failure_removes_staged_auth_home(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import chatgpt_credential_key_id

        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(config_error=RuntimeError("config failed"))
        staged_paths: list[Path] = []
        real_stage = adapter.stage_chatgpt_home

        def capture_stage(home: Path) -> Any:
            staged = real_stage(home)
            staged_paths.append(staged.path)
            return staged

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator_home = root / "operator"
            operator_home.mkdir()
            (operator_home / "auth.json").write_text(
                '{"auth_mode":"chatgpt","tokens":{"account_id":"account-a"}}',
                encoding="utf-8",
            )
            credential = CredentialRef(
                scope="model_provider",
                provider="openai_compatible",
                target_ref="model_provider:openai_compatible",
                key_id=chatgpt_credential_key_id(operator_home),
                source="runtime_session",
            )
            with (
                patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
                patch.object(adapter, "resolve_codex_binary", return_value="/sdk/codex"),
                patch.object(adapter, "verify_chatgpt_login"),
                patch.object(adapter, "stage_chatgpt_home", side_effect=capture_stage),
                patch.dict("os.environ", {"CODEX_HOME": str(operator_home)}, clear=False),
            ):
                result = await runtime.execute(
                    _request(
                        tmp,
                        provider_ref="model_provider:openai_compatible",
                        credential_ref=credential,
                    ),
                    AgentRuntimeExecutionContext(env={}),
                )

        self.assertFalse(result.success)
        self.assertIn("config failed", result.error or "")
        self.assertEqual(len(staged_paths), 1)
        self.assertFalse(staged_paths[0].exists())
        self.assertEqual(runtime._clients, {})

    async def test_callback_failure_does_not_abort_execution(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()

        def broken_callback(_event: Any) -> None:
            raise RuntimeError("observer failed")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            self.assertLogs(adapter.logger, level="ERROR"),
        ):
            result = await runtime.execute(
                _request(tmp),
                AgentRuntimeExecutionContext(
                    env={"OPENAI_API_KEY": "test-key"},
                    message_callback=broken_callback,
                ),
            )

        self.assertTrue(result.success, result.error)

    async def test_stop_interrupts_and_drains_turn_without_poisoning_client(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(turn_mode="interruptible")

        def stop_after_start() -> bool:
            return bool(harness.clients and harness.clients[0].turns)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
        ):
            result = await runtime.execute(
                _request(tmp),
                AgentRuntimeExecutionContext(
                    env={"OPENAI_API_KEY": "test-key"},
                    stop_requested=stop_after_start,
                ),
            )

        self.assertTrue(result.success)
        self.assertTrue(result.cancelled)
        self.assertFalse(result.timed_out)
        self.assertEqual(harness.clients[0].turns[0].interrupt_calls, 1)
        self.assertEqual(harness.clients[0].close_calls, 0)
        self.assertEqual(len(runtime._clients), 1)

    async def test_timeout_interrupts_turn_and_marks_timeout(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(turn_mode="interruptible")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
        ):
            result = await runtime.execute(
                _request(tmp, timeout_seconds=1),
                AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "test-key"}),
            )

        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.cancelled)
        self.assertEqual(result.failover_reason, "timeout")
        self.assertEqual(harness.clients[0].turns[0].interrupt_calls, 1)

    async def test_stuck_stream_is_marked_unhealthy_and_client_close_unblocks_reader(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(turn_mode="stuck")

        def stop_after_start() -> bool:
            return bool(harness.clients and harness.clients[0].turns)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            patch.object(adapter, "_INTERRUPT_DRAIN_SECONDS", 0.01),
        ):
            result = await runtime.execute(
                _request(tmp),
                AgentRuntimeExecutionContext(
                    env={"OPENAI_API_KEY": "test-key"},
                    stop_requested=stop_after_start,
                ),
            )
            await asyncio.sleep(0.02)

        self.assertTrue(result.cancelled)
        self.assertEqual(harness.clients[0].turns[0].interrupt_calls, 1)
        self.assertEqual(harness.clients[0].close_calls, 1)
        self.assertEqual(runtime._clients, {})

    async def test_stop_bounds_blocked_client_constructor_and_closes_late_client(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(block_phase="constructor")
        stop = threading.Event()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            patch.object(adapter, "_STREAM_POLL_SECONDS", 0.01),
        ):
            execution = asyncio.create_task(
                runtime.execute(
                    _request(tmp),
                    AgentRuntimeExecutionContext(
                        env={"OPENAI_API_KEY": "test-key"},
                        stop_requested=stop.is_set,
                    ),
                )
            )
            self.assertTrue(await asyncio.to_thread(harness.phase_started.wait, 0.5))
            stop.set()
            result = await asyncio.wait_for(execution, timeout=0.5)
            self.assertTrue(result.cancelled)
            self.assertEqual(harness.clients, [])

            harness.phase_release.set()
            await _wait_until(lambda: bool(harness.clients and harness.clients[0].close_calls))

        self.assertEqual(harness.clients[0].close_calls, 1)
        self.assertEqual(runtime._clients, {})

    async def test_aclose_during_subscription_constructor_cleans_client_and_auth_home(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import chatgpt_credential_key_id

        runtime = CodexSdkRuntime()
        harness = _SdkHarness(block_phase="constructor")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator_home = root / "operator"
            operator_home.mkdir()
            (operator_home / "auth.json").write_text(
                '{"auth_mode":"chatgpt","tokens":{"account_id":"account-a"}}',
                encoding="utf-8",
            )
            credential = CredentialRef(
                scope="model_provider",
                provider="openai_compatible",
                target_ref="model_provider:openai_compatible",
                key_id=chatgpt_credential_key_id(operator_home),
                source="runtime_session",
            )
            with (
                patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
                patch.object(adapter, "resolve_codex_binary", return_value="/sdk/codex"),
                patch.object(adapter, "verify_chatgpt_login"),
                patch.dict("os.environ", {"CODEX_HOME": str(operator_home)}, clear=False),
            ):
                execution = asyncio.create_task(
                    runtime.execute(
                        _request(
                            tmp,
                            provider_ref="model_provider:openai_compatible",
                            credential_ref=credential,
                        ),
                        AgentRuntimeExecutionContext(env={}),
                    )
                )
                self.assertTrue(await asyncio.to_thread(harness.phase_started.wait, 0.5))
                staged_home = Path(harness.configs[0].kwargs["env"]["CODEX_HOME"])
                closing = asyncio.create_task(runtime.aclose())
                await asyncio.sleep(0)
                harness.phase_release.set()
                result = await asyncio.wait_for(execution, timeout=1.0)
                await asyncio.wait_for(closing, timeout=1.0)

        self.assertFalse(result.success)
        self.assertIn("shutting down", result.error or "")
        self.assertEqual(harness.clients[0].close_calls, 1)
        self.assertFalse(staged_home.exists())
        self.assertEqual(runtime._clients, {})

    async def test_stop_bounds_blocked_thread_start_and_closes_client(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(block_phase="thread_start")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            patch.object(adapter, "_STREAM_POLL_SECONDS", 0.01),
            patch.object(adapter, "_RESOURCE_CLOSE_SECONDS", 0.5),
        ):
            result = await asyncio.wait_for(
                runtime.execute(
                    _request(tmp),
                    AgentRuntimeExecutionContext(
                        env={"OPENAI_API_KEY": "test-key"},
                        stop_requested=harness.phase_started.is_set,
                    ),
                ),
                timeout=1.0,
            )

        self.assertTrue(result.cancelled)
        self.assertEqual(harness.clients[0].close_calls, 1)
        self.assertEqual(runtime._clients, {})

    async def test_request_deadline_bounds_blocked_turn_creation(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(block_phase="turn")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            patch.object(adapter, "_RESOURCE_CLOSE_SECONDS", 0.5),
        ):
            result = await asyncio.wait_for(
                runtime.execute(
                    _request(tmp, timeout_seconds=1),
                    AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "test-key"}),
                ),
                timeout=1.75,
            )

        self.assertTrue(result.timed_out)
        self.assertEqual(harness.clients[0].close_calls, 1)
        self.assertEqual(runtime._clients, {})

    async def test_stopped_late_turn_is_interrupted_without_closing_shared_client_early(
        self,
    ) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(
            block_phase="turn",
            block_turn_prompt="blocked",
            turn_modes={"holder": "interruptible"},
        )
        stop = threading.Event()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            patch.object(adapter, "_STREAM_POLL_SECONDS", 0.01),
        ):
            context = AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "test-key"})
            holder = asyncio.create_task(
                runtime.execute(_request(tmp, request_id="holder", prompt="holder"), context)
            )
            await _wait_until(lambda: bool(harness.clients and harness.clients[0].turns))

            blocked = asyncio.create_task(
                runtime.execute(
                    _request(tmp, request_id="blocked", prompt="blocked"),
                    AgentRuntimeExecutionContext(
                        env={"OPENAI_API_KEY": "test-key"},
                        stop_requested=stop.is_set,
                    ),
                )
            )
            self.assertTrue(await asyncio.to_thread(harness.phase_started.wait, 0.5))
            stop.set()
            blocked_result = await asyncio.wait_for(blocked, timeout=0.5)

            self.assertTrue(blocked_result.cancelled)
            self.assertEqual(harness.clients[0].close_calls, 0)
            harness.phase_release.set()
            await _wait_until(
                lambda: (
                    len(harness.clients[0].turns) == 2
                    and harness.clients[0].turns[1].interrupt_calls == 1
                )
            )

            harness.clients[0].turns[0].interrupted.set()
            holder_result = await asyncio.wait_for(holder, timeout=0.5)
            await _wait_until(lambda: harness.clients[0].close_calls == 1)

        self.assertTrue(holder_result.success)
        self.assertEqual(runtime._clients, {})

    async def test_blocked_interrupt_is_bounded_and_client_close_unblocks_workers(self) -> None:
        runtime = CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness(turn_mode="stuck", block_phase="interrupt")

        def stop_after_turn_start() -> bool:
            return bool(harness.clients and harness.clients[0].turns)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
            patch.object(adapter, "_STREAM_POLL_SECONDS", 0.01),
            patch.object(adapter, "_INTERRUPT_REQUEST_SECONDS", 0.02),
            patch.object(adapter, "_INTERRUPT_DRAIN_SECONDS", 0.01),
            patch.object(adapter, "_RESOURCE_CLOSE_SECONDS", 0.5),
        ):
            result = await asyncio.wait_for(
                runtime.execute(
                    _request(tmp),
                    AgentRuntimeExecutionContext(
                        env={"OPENAI_API_KEY": "test-key"},
                        stop_requested=stop_after_turn_start,
                    ),
                ),
                timeout=1.0,
            )
            await _wait_until(harness.clients[0].closed_event.is_set)

        self.assertTrue(result.cancelled)
        self.assertTrue(harness.phase_started.is_set())
        self.assertEqual(harness.clients[0].turns[0].interrupt_calls, 1)
        self.assertEqual(harness.clients[0].close_calls, 1)
        self.assertEqual(runtime._clients, {})

    async def test_aclose_closes_reused_client_once_and_clears_runtime_state(self) -> None:
        runtime = CodexSdkRuntime()
        harness = _SdkHarness()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", side_effect=lambda: harness.sdk()),
        ):
            context = AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "test-key"})
            await runtime.execute(_request(tmp, request_id="request-a"), context)
            await runtime.execute(_request(tmp, request_id="request-b"), context)
            await runtime.aclose()
            await runtime.aclose()

        self.assertEqual(len(harness.clients), 1)
        self.assertEqual(harness.clients[0].close_calls, 1)
        self.assertEqual(runtime._clients, {})
        self.assertIsNone(runtime._executor)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
