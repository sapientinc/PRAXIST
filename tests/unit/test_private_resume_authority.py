"""Controller-private startup authority cannot be replaced by peer run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.cli import registry, resume
from praxist.core import controller_state
from praxist.core.controller_state import read_private_startup_config, write_private_startup_config
from praxist.plugins.workflow_stages.research_loop import startup
from praxist.plugins.workflow_stages.research_loop.backend import resume_state
from tests.helpers.paths import REPO_ROOT
from tests.unit.test_cli_resume import _entry_kwargs


class PrivateResumeAuthorityTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.control = self.root / "control"
        self.control.mkdir(mode=0o700)
        self.task = self.root / "task"
        shutil.copytree(REPO_ROOT / "templates/tasks/toy_math", self.task)
        self.run_dir = self.root / "output" / "run_private_authority"
        # Controller metadata lives under a controller-owned, peer-read-only
        # root; only explicitly provisioned child directories may be writable.
        self.run_dir.mkdir(parents=True, mode=0o755)
        self.run_dir.chmod(0o755)
        self.private = (
            self.control
            / hashlib.sha256(str(self.run_dir.resolve()).encode()).hexdigest()[:24]
            / "startup_config.json"
        )
        environment = patch.dict(
            os.environ,
            {
                "PRAXIST_CONTROLLER_STATE_DIR": str(self.control),
                "PRAXIST_STATE_DIR": str(self.root / "registry"),
                "PRAXIST_BUNDLED_PLUGIN_ROOTS": str(REPO_ROOT / "tests/fixtures/plugins"),
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def _prepare(self, **overrides):
        kwargs = {
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
        return startup.prepare_research_loop_plugin_run(**{**kwargs, **overrides})

    def _tamper_public_startup(self) -> None:
        (self.run_dir / "startup_config.json").write_text(
            json.dumps(
                {
                    "canonical_args": {
                        "task_path": str(self.root / "attacker_task"),
                        "runtime": "agent_runtime:attacker",
                        "model_provider": "model_provider:attacker",
                        "model": "attacker-model",
                        "codex_native": True,
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_actual_startup_creates_private_authority_under_per_run_hash(self) -> None:
        previous_umask = os.umask(0o002)
        try:
            prepared = self._prepare()
        finally:
            os.umask(previous_umask)
        self.assertTrue(self.private.is_file())
        authority = json.loads(self.private.read_text(encoding="utf-8"))
        self.assertEqual(authority["canonical_args"], prepared.startup_config["canonical_args"])
        self.assertEqual(authority["resume_identity"], prepared.startup_config["resume_identity"])
        self.assertEqual(stat.S_IMODE(self.private.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.private.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.control.stat().st_mode), 0o700)
        self.assertEqual(self.private.stat().st_uid, os.geteuid())

    def test_resume_recovers_interrupted_private_publication(self) -> None:
        original_unlink = os.unlink

        def interrupted_unlink(path, *args, **kwargs):
            if str(path).startswith(".startup-"):
                raise SystemExit("interrupted after publication")
            return original_unlink(path, *args, **kwargs)

        with (
            patch.object(controller_state.os, "unlink", side_effect=interrupted_unlink),
            self.assertRaisesRegex(SystemExit, "interrupted after publication"),
        ):
            self._prepare()
        self.assertEqual(self.private.stat().st_nlink, 2)
        authority = read_private_startup_config(self.run_dir)
        self.assertEqual(authority["canonical_args"]["task_path"], str(self.task))
        self.assertEqual(self.private.stat().st_nlink, 1)
        self.assertEqual(list(self.private.parent.iterdir()), [self.private])

    def test_private_authority_rejects_unexplained_extra_hardlinks(self) -> None:
        self._prepare()
        os.link(self.private, self.root / "unexpected-copy")
        with self.assertRaisesRegex(ValueError, "private startup authority"):
            read_private_startup_config(self.run_dir)

    def test_peer_writable_run_root_rejected_before_loading_plugins(self) -> None:
        self.run_dir.chmod(0o777)
        with (
            patch.object(startup.PluginLoader, "load") as load,
            self.assertRaisesRegex(ValueError, "controller run root"),
        ):
            self._prepare()
        load.assert_not_called()

    def test_private_resume_rejects_run_root_that_became_peer_writable(self) -> None:
        self._prepare()
        self.run_dir.chmod(0o777)
        with self.assertRaisesRegex(ValueError, "controller run root"):
            read_private_startup_config(self.run_dir)

    def test_resume_launch_ignores_tampered_public_startup_and_credentials(self) -> None:
        self._prepare()
        self._tamper_public_startup()
        (self.run_dir / "credentials_redacted.json").write_text(
            json.dumps(
                {
                    "credential_profiles": [
                        {
                            "provider": "openai_compatible",
                            "source": "runtime_session",
                            "key_id": "x:chatgpt:x",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        entry = registry.RegistryEntry(
            **_entry_kwargs(run_dir=str(self.run_dir), task_path=str(self.task))
        )
        with patch.object(resume.start, "launch_run", return_value=entry) as launch:
            resumed = resume.resume_run(target=str(self.run_dir))
        self.assertIs(resumed, entry)
        self.assertEqual(launch.call_args.kwargs["task_path"], str(self.task))
        self.assertEqual(launch.call_args.kwargs["runtime_ref"], "agent_runtime:fake_runtime")
        self.assertEqual(
            launch.call_args.kwargs["model_provider_ref"], "model_provider:fake_provider"
        )
        self.assertEqual(launch.call_args.kwargs["model"], "fake-deterministic")
        self.assertFalse(launch.call_args.kwargs["codex_native"])

    def test_registry_projection_cannot_replace_private_launch_identity(self) -> None:
        self._prepare()
        entry = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(self.run_dir),
                task_path="/attacker/task",
                runtime_ref="agent_runtime:attacker",
                model_provider_ref="model_provider:attacker",
                model="attacker-model",
            )
        )
        target = resume._target_from_registry(entry)
        self.assertEqual(target.task_path, str(self.task))
        self.assertEqual(target.runtime_ref, "agent_runtime:fake_runtime")
        self.assertEqual(target.model, "fake-deterministic")

    def test_public_startup_snapshot_is_not_required_as_resume_authority(self) -> None:
        self._prepare()
        (self.run_dir / "startup_config.json").unlink()
        target = resume.resolve_resume_target(str(self.run_dir))
        self.assertEqual(target.task_path, str(self.task))
        self.assertEqual(target.runtime_ref, "agent_runtime:fake_runtime")
        self._prepare(resume=True)

    def test_startup_failure_after_private_commit_preserves_resume_authority(self) -> None:
        with (
            patch.object(startup, "write_json", side_effect=OSError("public disk unavailable")),
            self.assertRaisesRegex(OSError, "public disk unavailable"),
        ):
            self._prepare()
        self.assertEqual(
            read_private_startup_config(self.run_dir)["canonical_args"]["task_path"], str(self.task)
        )

    def test_existing_private_startup_authority_cannot_be_replaced(self) -> None:
        prepared = self._prepare()
        original = self.private.read_bytes()
        changed = {**prepared.startup_config, "command": "replacement"}
        with self.assertRaisesRegex(ValueError, "private.*startup|startup.*authority"):
            write_private_startup_config(self.run_dir, changed)
        self.assertEqual(self.private.read_bytes(), original)

    def test_missing_private_authority_fails_closed_even_with_valid_public_snapshot(self) -> None:
        self._prepare()
        if self.private.exists():
            self.private.unlink()
        with self.assertRaisesRegex(resume.ResumeError, "private.*startup|startup.*authority"):
            resume.resolve_resume_target(str(self.run_dir))
        with (
            patch.object(startup.PluginLoader, "load") as load,
            self.assertRaisesRegex(ValueError, "private.*startup|startup.*authority"),
        ):
            self._prepare(resume=True)
        load.assert_not_called()

    def test_corrupt_private_authority_does_not_fall_back_to_public_snapshot(self) -> None:
        self._prepare()
        self.private.parent.mkdir(parents=True, exist_ok=True)
        self.private.write_text("not json", encoding="utf-8")
        self.private.chmod(0o600)
        with self.assertRaisesRegex(resume.ResumeError, "private.*startup|startup.*authority"):
            resume.resolve_resume_target(str(self.run_dir))

    def test_backend_resume_preserves_initial_authority_despite_public_snapshot_tampering(
        self,
    ) -> None:
        self._prepare()
        original = self.private.read_bytes()
        self._tamper_public_startup()
        resumed = self._prepare(resume=True)
        self.assertEqual(self.private.read_bytes(), original)
        self.assertTrue(resumed.startup_config["resume"]["enabled"])

    def test_public_schema_tampering_cannot_downgrade_resume_boundary_contract(self) -> None:
        self._prepare()
        for name in ("startup_config.json", "run.json"):
            (self.run_dir / name).write_text("{}", encoding="utf-8")
        self.assertEqual(resume_state._boundary_marker_contract_start(self.run_dir), 0)

    def test_backend_rejects_task_or_runtime_switch_before_plugin_loading(self) -> None:
        self._prepare()
        alternate_task = self.root / "alternate_task"
        shutil.copytree(self.task, alternate_task)
        for override in (
            {"task_project_path": alternate_task},
            {"runtime_ref": "agent_runtime:attacker"},
            {"model_provider_ref": "model_provider:attacker"},
        ):
            with (
                self.subTest(override=override),
                patch.object(startup.PluginLoader, "load") as load,
                self.assertRaisesRegex(ValueError, "private.*startup|startup.*authority"),
            ):
                self._prepare(resume=True, **override)
            load.assert_not_called()

    def test_task_environment_cannot_redirect_controller_authority(self) -> None:
        with self.assertRaisesRegex(ValueError, "PRAXIST_CONTROLLER_STATE_DIR"):
            startup._task_runtime_env(
                task_project_path=self.task,
                workspace=self.root,
                task_id="test",
                env={},
                task_descriptor={
                    "runtime_environment": {"env": {"PRAXIST_CONTROLLER_STATE_DIR": "/attacker"}}
                },
            )

    def test_private_snapshot_symlink_is_rejected_without_reading_public_target(self) -> None:
        self._prepare()
        self.private.unlink(missing_ok=True)
        self.private.parent.mkdir(parents=True, exist_ok=True)
        self.private.symlink_to(self.run_dir / "startup_config.json")
        with self.assertRaisesRegex(resume.ResumeError, "private.*startup|startup.*authority"):
            resume.resolve_resume_target(str(self.run_dir))

    def test_group_readable_private_authority_directory_is_rejected(self) -> None:
        self.control.chmod(0o750)
        with self.assertRaisesRegex(ValueError, "private.*controller|controller.*private"):
            self._prepare()

    def test_authority_snapshot_from_another_run_is_rejected(self) -> None:
        self._prepare()
        original = json.loads(self.private.read_text(encoding="utf-8"))
        original["canonical_args"]["run_dir"] = str(self.root / "another_run")
        self.private.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaisesRegex(resume.ResumeError, "another run"):
            resume.resolve_resume_target(str(self.run_dir))

    def test_private_authority_file_cannot_be_group_writable(self) -> None:
        self._prepare()
        self.private.chmod(0o660)
        with self.assertRaisesRegex(resume.ResumeError, "private.*startup|startup.*authority"):
            resume.resolve_resume_target(str(self.run_dir))

    def test_private_authority_cannot_be_under_a_peer_writable_ancestor(self) -> None:
        writable_parent = self.root / "shared"
        writable_parent.mkdir(mode=0o770)
        writable_parent.chmod(0o770)
        os.environ["PRAXIST_CONTROLLER_STATE_DIR"] = str(writable_parent / "control")
        with self.assertRaisesRegex(ValueError, "private.*controller|controller.*private"):
            self._prepare()

    def test_private_authority_root_cannot_be_a_symlink(self) -> None:
        link = self.root / "control_link"
        link.symlink_to(self.control, target_is_directory=True)
        os.environ["PRAXIST_CONTROLLER_STATE_DIR"] = str(link)
        with self.assertRaisesRegex(ValueError, "private.*controller|controller.*private"):
            self._prepare()

    def test_configured_authority_root_must_be_absolute_and_separate_from_outputs(self) -> None:
        self._prepare()
        for location in ("", "relative/control", str(self.run_dir / "control"), str(self.root)):
            with (
                self.subTest(location=location),
                patch.dict(os.environ, {"PRAXIST_CONTROLLER_STATE_DIR": location}),
                self.assertRaisesRegex(resume.ResumeError, "absolute directory|separate"),
            ):
                resume.resolve_resume_target(str(self.run_dir))

    def test_private_reader_rejects_symlinked_public_run_root(self) -> None:
        self._prepare()
        alias = self.root / "run_alias"
        alias.symlink_to(self.run_dir, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "controller run root must be a safe directory"):
            read_private_startup_config(alias)

    def test_incomplete_private_snapshot_is_rejected_before_loading_plugins(self) -> None:
        self._prepare()
        original = json.loads(self.private.read_text(encoding="utf-8"))
        missing_runtime = {
            **original,
            "canonical_args": {**original["canonical_args"], "runtime": ""},
        }
        for payload in (
            [],
            {"schema_version": "praxist.startup.v1", "canonical_args": []},
            missing_runtime,
            {**original, "resume_identity": None},
        ):
            self.private.write_text(json.dumps(payload), encoding="utf-8")
            with (
                self.subTest(payload=payload),
                patch.object(startup.PluginLoader, "load") as load,
                self.assertRaisesRegex(ValueError, "private startup authority"),
            ):
                self._prepare(resume=True)
            load.assert_not_called()

    def test_changed_task_contents_are_rejected_before_loading_plugins(self) -> None:
        self._prepare()
        descriptor = self.task / "task.yaml"
        descriptor.write_text(
            descriptor.read_text(encoding="utf-8") + "\n# Task fixture changed after startup.\n",
            encoding="utf-8",
        )
        with (
            patch.object(startup.PluginLoader, "load") as load,
            self.assertRaisesRegex(ValueError, "task_project_manifest_sha256"),
        ):
            self._prepare(resume=True)
        load.assert_not_called()

    def test_changed_execution_mode_is_rejected_before_loading_plugins(self) -> None:
        self._prepare()
        with (
            patch.object(startup.PluginLoader, "load") as load,
            self.assertRaisesRegex(ValueError, "local_mode"),
        ):
            self._prepare(resume=True, local_mode=False)
        load.assert_not_called()

    def test_private_authority_cannot_be_published_inside_task_project(self) -> None:
        prepared = self._prepare()
        forbidden = self.task / "controller_state"
        with (
            patch.dict(os.environ, {"PRAXIST_CONTROLLER_STATE_DIR": str(forbidden)}),
            self.assertRaisesRegex(ValueError, "separate from the task project"),
        ):
            write_private_startup_config(self.run_dir, prepared.startup_config)
        self.assertFalse(forbidden.exists())

    def test_dangling_entry_prevents_startup_in_an_occupied_run_directory(self) -> None:
        (self.run_dir / "unexpected_entry").symlink_to(self.root / "missing_target")
        with (
            patch.object(startup.PluginLoader, "load") as load,
            self.assertRaisesRegex(ValueError, "not empty"),
        ):
            self._prepare()
        load.assert_not_called()
        self.assertFalse(self.private.exists())

    def test_unset_controller_mode_keeps_legacy_public_snapshot_behavior(self) -> None:
        os.environ.pop("PRAXIST_CONTROLLER_STATE_DIR", None)
        self._prepare()
        self.assertFalse(self.private.exists())
        self._tamper_public_startup()
        target = resume.resolve_resume_target(str(self.run_dir))
        self.assertEqual(target.runtime_ref, "agent_runtime:attacker")


if __name__ == "__main__":
    unittest.main()
