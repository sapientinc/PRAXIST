from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import yaml

from praxist.plugins.workflow_stages.research_loop import startup
from praxist.plugins.workflow_stages.research_loop.runtime_preparation import prepare_task_runtime
from tests.helpers.paths import REPO_ROOT


class RuntimePreparationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.task = self.root / "task"
        self.task.mkdir()
        self.run_dir = self.root / "output" / "run"
        self.descriptor = {
            "runtime_environment": {
                "prepare_entrypoint": "coordinator.py:prepare_runtime",
                "prepare_config": {"label": "prepared"},
            }
        }

    def source(self, text):
        (self.task / "coordinator.py").write_text(text, encoding="utf-8")

    def prepare(self, *, resume=False):
        prepare_task_runtime(
            task_descriptor=self.descriptor,
            task_path=self.task,
            run_dir=self.run_dir,
            resume=resume,
        )

    def test_sync_hook_receives_detached_config_and_explicit_paths(self):
        self.source("""
import json
def prepare_runtime(*, task_path, run_dir, resume, config):
    assert task_path.is_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prepared.json").write_text(json.dumps({"label": config["label"], "resume": resume}))
    config["label"] = "mutated"
""")
        self.prepare(resume=True)
        self.assertEqual(
            json.loads((self.run_dir / "prepared.json").read_text()),
            {"label": "prepared", "resume": True},
        )
        self.assertEqual(
            self.descriptor["runtime_environment"]["prepare_config"], {"label": "prepared"}
        )

    def test_absent_hook_does_not_create_a_run_directory(self):
        self.descriptor = {"runtime_environment": {"cwd": "run_dir"}}
        self.prepare()
        self.assertFalse(self.run_dir.exists())

    def test_null_runtime_environment_does_not_load_task_code(self):
        self.source("raise AssertionError('task preparation must not run')\n")
        self.descriptor = {"runtime_environment": None}

        self.prepare()

        self.assertFalse(self.run_dir.exists())

    def test_malformed_runtime_environment_is_rejected_before_task_execution(self):
        self.source("raise AssertionError('task preparation must not run')\n")
        for runtime in ([], "coordinator.py:prepare_runtime", True):
            with self.subTest(runtime=runtime):
                self.descriptor = {"runtime_environment": runtime}
                with self.assertRaisesRegex(ValueError, "runtime_environment must be a mapping"):
                    self.prepare()
        self.assertFalse(self.run_dir.exists())

    def test_invalid_or_orphaned_prepare_configuration_is_not_silently_ignored(self):
        for config in ([], None, {"unused": True}):
            with self.subTest(config=config):
                self.descriptor = {"runtime_environment": {"prepare_config": config}}
                with self.assertRaises(ValueError):
                    self.prepare()
        self.assertFalse(self.run_dir.exists())

    def test_invalid_entrypoints_and_config_fail_before_loading_task_code(self):
        marker = self.root / "executed"
        (self.root / "outside.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        )
        for entrypoint in (
            "../outside.py:prepare",
            "/tmp/outside.py:prepare",
            "coordinator:prepare",
            "missing.py:prepare",
            "x.py:handler.attr",
            "x.py:a:b",
            "",
            None,
        ):
            with self.subTest(entrypoint=entrypoint):
                self.descriptor["runtime_environment"]["prepare_entrypoint"] = entrypoint
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(marker.exists())
        self.source("def prepare_runtime(**kwargs): pass\n")
        self.descriptor["runtime_environment"]["prepare_entrypoint"] = (
            "coordinator.py:prepare_runtime"
        )
        self.descriptor["runtime_environment"]["prepare_config"] = []
        with self.assertRaises(ValueError):
            self.prepare()

    def test_non_json_config_is_rejected_before_import_or_callback_execution(self):
        imported = self.root / "imported"
        self.source(
            f"from pathlib import Path\nPath({str(imported)!r}).touch()\n"
            "def prepare_runtime(**kwargs):\n"
            "    kwargs['run_dir'].mkdir(parents=True)\n"
        )
        circular = {}
        circular["self"] = circular
        configurations = {
            "object": {"value": object()},
            "set": {"value": {1, 2}},
            "nan": {"value": float("nan")},
            "infinity": {"value": float("inf")},
            "circular": circular,
        }
        old_path = list(sys.path)
        for label, config in configurations.items():
            with self.subTest(config=label):
                self.descriptor["runtime_environment"]["prepare_config"] = config
                with self.assertRaisesRegex(ValueError, "prepare_config must be JSON-compatible"):
                    self.prepare()
                self.assertFalse(imported.exists())
                self.assertFalse(self.run_dir.exists())
                self.assertEqual(sys.path, old_path)

    def test_entrypoint_symlink_cannot_escape_task_root(self):
        outside = self.root / "outside.py"
        outside.write_text("def prepare_runtime(**kwargs): raise AssertionError('executed')\n")
        (self.task / "coordinator.py").symlink_to(outside)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_async_hook_is_rejected_without_running_body(self):
        self.source("""
async def prepare_runtime(**kwargs):
    kwargs["run_dir"].mkdir(parents=True)
""")
        with self.assertRaisesRegex(TypeError, "synchronous"):
            self.prepare()
        self.assertFalse(self.run_dir.exists())

    def test_sync_hook_returning_awaitable_is_rejected_and_closed(self):
        self.source("""
async def deferred():
    raise AssertionError("must never be awaited")
def prepare_runtime(**kwargs):
    return deferred()
""")
        with self.assertRaisesRegex(TypeError, "awaitable"):
            self.prepare()

    def test_returned_futures_are_rejected_and_cancelled(self):
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        futures = [concurrent.futures.Future(), loop.create_future()]
        self.source(
            "from preparation_future_fixture import pending\n"
            "def prepare_runtime(**kwargs): return pending\n"
        )
        for pending in futures:
            with self.subTest(future_type=type(pending).__name__):
                fixture = ModuleType("preparation_future_fixture")
                fixture.pending = pending
                with (
                    patch.dict(sys.modules, {fixture.__name__: fixture}),
                    self.assertRaises(TypeError),
                ):
                    self.prepare()
                self.assertTrue(pending.cancelled())

    def test_returned_generator_is_closed_before_rejecting_deferred_work(self):
        closed = []

        def deferred():
            try:
                yield "pending"
            finally:
                closed.append(True)

        pending = deferred()
        self.addCleanup(pending.close)
        self.assertEqual(next(pending), "pending")
        fixture = ModuleType("preparation_generator_fixture")
        fixture.pending = pending
        self.source(
            "from preparation_generator_fixture import pending\n"
            "def prepare_runtime(**kwargs): return pending\n"
        )

        with (
            patch.dict(sys.modules, {fixture.__name__: fixture}),
            self.assertRaisesRegex(TypeError, "must not return deferred work"),
        ):
            self.prepare()

        self.assertEqual(closed, [True])
        with self.assertRaises(StopIteration):
            next(pending)
        self.assertFalse(self.run_dir.exists())

    def test_non_none_return_is_rejected(self):
        self.source("def prepare_runtime(**kwargs): return {'status': 'prepared'}\n")
        with self.assertRaisesRegex(TypeError, "None"):
            self.prepare()

    def test_local_imports_override_cached_modules_and_restore_import_state(self):
        cached = ModuleType("prepare_helper")
        cached.VALUE = "wrong"
        (self.task / "prepare_helper.py").write_text('VALUE = "local"\n')
        self.source("""
from prepare_helper import VALUE
def prepare_runtime(*, task_path, run_dir, resume, config):
    run_dir.mkdir(parents=True)
    (run_dir / "value").write_text(VALUE)
""")
        old_path = list(sys.path)
        with patch.dict(sys.modules, {"prepare_helper": cached}):
            self.prepare()
            self.assertIs(sys.modules["prepare_helper"], cached)
        self.assertEqual((self.run_dir / "value").read_text(), "local")
        self.assertEqual(sys.path, old_path)
        self.assertFalse(any(name.startswith("_praxist_runtime_prepare_") for name in sys.modules))

    def test_import_failure_restores_state_and_propagates(self):
        self.source('raise RuntimeError("preparation failed")\n')
        old_path = list(sys.path)
        with self.assertRaisesRegex(RuntimeError, "preparation failed"):
            self.prepare()
        self.assertEqual(sys.path, old_path)
        self.assertFalse(any(name.startswith("_praxist_runtime_prepare_") for name in sys.modules))

    def test_nested_callback_import_failure_restores_local_import_state(self):
        hooks = self.task / "hooks"
        hooks.mkdir()
        self.descriptor["runtime_environment"]["prepare_entrypoint"] = (
            "hooks/coordinator.py:prepare"
        )
        (self.task / "prepare_helper.py").write_text('VALUE = "root shadow"\n')
        (hooks / "prepare_helper.py").write_text('VALUE = "nested sibling"\n')
        (self.task / "prepare_root_helper.py").write_text('VALUE = "task root"\n')
        (hooks / "prepare_transient.py").write_text('VALUE = "loaded"\n')
        (hooks / "coordinator.py").write_text("""
import json
def prepare(*, task_path, run_dir, resume, config):
    from prepare_helper import VALUE
    from prepare_root_helper import VALUE as ROOT_VALUE
    run_dir.mkdir(parents=True)
    (run_dir / "imports.json").write_text(json.dumps([VALUE, ROOT_VALUE]))
    from prepare_transient import MISSING_VALUE
""")
        cached_helper = ModuleType("prepare_helper")
        cached_helper.VALUE = "cached sibling"
        cached_root = ModuleType("prepare_root_helper")
        cached_root.VALUE = "cached root"
        old_path = list(sys.path)

        with patch.dict(
            sys.modules,
            {"prepare_helper": cached_helper, "prepare_root_helper": cached_root},
        ):
            with self.assertRaisesRegex(ImportError, "MISSING_VALUE"):
                self.prepare()
            self.assertIs(sys.modules["prepare_helper"], cached_helper)
            self.assertIs(sys.modules["prepare_root_helper"], cached_root)
            self.assertNotIn("prepare_transient", sys.modules)

        self.assertEqual(
            json.loads((self.run_dir / "imports.json").read_text()),
            ["nested sibling", "task root"],
        )
        self.assertEqual(sys.path, old_path)
        self.assertFalse(any(name.startswith("_praxist_runtime_prepare_") for name in sys.modules))


class RuntimePreparationStartupTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.task = self.root / "task"
        shutil.copytree(REPO_ROOT / "templates/tasks/toy_math", self.task)
        self.run_dir = self.root / "output" / "run"
        self.events = self.run_dir / "preparation-events.json"
        self.control = self.root / "control"
        self.control.mkdir(mode=0o700)
        (self.task / "coordinator.py").write_text(
            """
import json
def prepare_runtime(*, task_path, run_dir, resume, config):
    if not resume:
        assert not (run_dir / "run.json").exists()
        assert not (run_dir / "trajectory.jsonl").exists()
    (run_dir / "peer-work").mkdir(parents=True, exist_ok=True)
    path = run_dir / "preparation-events.json"
    events = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps([*events, resume]))
""",
            encoding="utf-8",
        )
        descriptor = yaml.safe_load((self.task / "task.yaml").read_text())
        descriptor["runtime_environment"] = {
            "cwd": str(self.run_dir / "peer-work"),
            "prepare_entrypoint": "coordinator.py:prepare_runtime",
            "prepare_config": {},
        }
        (self.task / "task.yaml").write_text(yaml.safe_dump(descriptor))
        env = patch.dict(
            os.environ,
            {
                "PRAXIST_CONTROLLER_STATE_DIR": str(self.control),
                "PRAXIST_STATE_DIR": str(self.root / "registry"),
                "PRAXIST_BUNDLED_PLUGIN_ROOTS": str(REPO_ROOT / "tests/fixtures/plugins"),
            },
        )
        env.start()
        self.addCleanup(env.stop)

    def prepare(self, **overrides):
        options = {
            "task_project_path": self.task,
            "workspace": self.root,
            "run_dir": self.run_dir,
            "runtime_ref": "agent_runtime:fake_runtime",
            "model_provider_ref": "model_provider:fake_provider",
            "budget_policy_ref": "budget_policy:fake_tiered",
            "model": "fake-deterministic",
            "local_mode": True,
            "frontier_strategy": "auto",
            "credential_profile": "fake_multi_key",
        }
        return startup.prepare_research_loop_plugin_run(**{**options, **overrides})

    def test_hook_runs_before_cwd_and_store_setup_on_start_and_resume(self):
        prepared = self.prepare()
        self.assertEqual(prepared.task_execution_cwd, self.run_dir / "peer-work")
        self.prepare(resume=True)
        self.assertEqual(json.loads(self.events.read_text()), [False, True])

    def test_fresh_run_rejection_occurs_before_preparation(self):
        self.prepare()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(json.loads(self.events.read_text()), [False])

    def test_private_resume_selection_rejection_occurs_before_preparation(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "private startup authority"):
            self.prepare(resume=True, runtime_ref="agent_runtime:untrusted")
        self.assertEqual(json.loads(self.events.read_text()), [False])

    def test_complete_resume_identity_rejection_occurs_before_preparation(self):
        self.prepare()
        for overrides in ({"model": "different-model"}, {"frontier_strategy": "different"}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.prepare(resume=True, **overrides)
                self.assertEqual(json.loads(self.events.read_text()), [False])
        with (
            patch.dict(os.environ, {"PRAXIST_COHORT_SIZE": "999"}),
            self.assertRaises(ValueError),
        ):
            self.prepare(resume=True)
        self.assertEqual(json.loads(self.events.read_text()), [False])

    def test_preparation_failure_stops_before_runtime_store_creation(self):
        (self.task / "coordinator.py").write_text(
            "def prepare_runtime(**kwargs):\n    raise OSError('layout unavailable')\n"
        )
        with self.assertRaisesRegex(OSError, "layout unavailable"):
            self.prepare()
        self.assertFalse(self.run_dir.exists())


if __name__ == "__main__":
    unittest.main()
