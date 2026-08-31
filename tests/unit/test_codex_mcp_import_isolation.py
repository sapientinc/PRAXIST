"""Trusted MCP startup must not import modules from analyst-controlled paths."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.runtimes import AgentRuntimeExecutionContext
from praxist.plugins.agent_runtimes.codex_sdk import adapter
from praxist.plugins.agent_runtimes.codex_sdk._mcp import mcp_configuration
from tests.unit.test_codex_sdk_adapter import _request, _SdkHarness


def _shadow_module(directory: Path, name: str) -> None:
    (directory / f"{name}.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        f"Path(os.environ['SHADOW_MARKER']).write_text({name!r})\n"
    )


class McpImportIsolationTests(unittest.TestCase):
    def test_isolated_launcher_ignores_working_directory_and_pythonpath_shadows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            injected_imports = root / "injected-imports"
            workspace.mkdir()
            injected_imports.mkdir()
            marker = root / "shadow-imported"
            _shadow_module(workspace, "praxist")
            _shadow_module(injected_imports, "sitecustomize")
            env = {
                "PATH": os.defpath,
                "PYTHONPATH": str(injected_imports),
                "SHADOW_MARKER": str(marker),
            }
            # The harmless shadow really executes without isolated startup.
            vulnerable = mcp_configuration([{"server_name": "memory-tools"}]).config
            command = vulnerable["mcp_servers"]["memory-tools"]
            subprocess.run(
                [command["command"], *command["args"], "--help"],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertTrue(marker.exists())
            marker.unlink()

            isolated = mcp_configuration(
                [{"server_name": "memory-tools"}], isolated_python=True
            ).config
            command = isolated["mcp_servers"]["memory-tools"]
            result = subprocess.run(
                [command["command"], *command["args"], "--help"],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Praxist stdio MCP launcher", result.stdout)
            self.assertFalse(marker.exists())

    def test_isolated_launcher_ignores_pythonhome_without_changing_work_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = mcp_configuration(
                [{"server_name": "task-tools", "factory": "installed_task.tools:create_server"}],
                python_executable=sys.executable,
                isolated_python=True,
                env={"PRAXIST_RUN_DIR": tmp},
            ).config
            command = config["mcp_servers"]["task-tools"]
            result = subprocess.run(
                [command["command"], *command["args"], "--help"],
                cwd=tmp,
                env={"PATH": os.defpath, "PYTHONHOME": str(Path(tmp) / "not-a-runtime")},
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(command["env"]["PRAXIST_RUN_DIR"], tmp)
            self.assertEqual(command["args"][-1], "installed_task.tools:create_server")


class McpImportIsolationDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_restrictions_select_isolated_startup_for_every_factory(self):
        for marker in ("require_task_sandbox_policy", "require_task_tool_policy", "path_profile"):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as tmp:
                runtime = adapter.CodexSdkRuntime()
                self.addAsyncCleanup(runtime.aclose)
                harness = _SdkHarness()
                with patch.object(adapter, "_load_sdk", side_effect=harness.sdk):
                    runtime_options = (
                        {
                            "sandbox_intent": {
                                "filesystem": "workspace_write",
                                "network": "off",
                                "approval": "auto",
                                "readable_roots": [tmp],
                            }
                        }
                        if marker == "path_profile"
                        else {marker: True}
                    )
                    request = _request(
                        tmp,
                        runtime_options=runtime_options,
                        tool_servers=[
                            {"server_name": "memory-tools"},
                            {
                                "server_name": "task-tools",
                                "factory": "installed_task.tools:create_server",
                            },
                        ],
                    )
                    result = await runtime.execute(
                        request, AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "test-key"})
                    )
                self.assertTrue(result.success, result.error)
                servers = harness.clients[0].thread_calls[0]["config"]["mcp_servers"]
                for command in servers.values():
                    self.assertEqual(command["args"][:2], ["-I", "-m"])


if __name__ == "__main__":
    unittest.main()
