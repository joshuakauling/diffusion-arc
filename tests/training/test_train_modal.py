"""Test Modal account guards and bounded remote orchestration without cloud resources."""

from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.cloud import train_modal as launch
from diffusion_arc.training.config import load_training_config


class ModalTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.args = SimpleNamespace(config=launch.DEFAULT_CONFIG, run_name="epochs5-6",
                                    profile="new-account", expected_workspace="new-workspace",
                                    environment="main", volume="test-v2", secret="training")
        self.plan = launch.plan_run(self.args)

    def test_recipe_changes_only_initialization_seed_provider_and_paths(self):
        previous = json.loads(launch.DEFAULT_CONFIG.with_name("arc_agi_1_epoch3_high_noise.json").read_text())
        current = json.loads(launch.DEFAULT_CONFIG.read_text())
        for key in ("execution", "loop", "canvas", "flow", "loss", "optimizer", "scheduler",
                    "validation", "checkpointing", "final_test"):
            self.assertEqual(current[key], previous[key], key)
        self.assertEqual([x["seed"] for x in self.plan["epochs"]], [171, 172])
        self.assertEqual([x["final_step"] for x in self.plan["epochs"]], [4530, 5436])
        self.assertEqual([x["optimizer_reset"] for x in self.plan["epochs"]], [True, False])
        self.assertEqual(self.plan["transformer_source"]["subfolder"], "m7y2mfn2/step-003624/transformer")
        self.assertEqual(self.plan["transformer_source"]["revision"], "a0080ba91538ab001fe958d2632a2f6a0cf71fb5")

    def test_dry_run_needs_no_modal_installation_or_authentication(self):
        program = """
import importlib.abc, sys
class RejectModal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'modal' or fullname.startswith('modal.'):
            raise AssertionError('Dry-run imported Modal')
sys.meta_path.insert(0, RejectModal())
from scripts.cloud.train_modal import main
main(['run', '--profile', 'unconfigured', '--expected-workspace', 'unconfigured',
      '--environment', 'main', '--dry-run'])
"""
        result = subprocess.run([sys.executable, "-B", "-c", program], cwd=launch.ROOT,
                                text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)["epochs"][1]["epoch"], 6)

    def test_account_override_fails_before_import_and_mismatch_before_objects(self):
        with patch.dict(os.environ, {"MODAL_TOKEN_ID": "override"}), patch.object(launch.importlib, "import_module") as imports:
            with self.assertRaisesRegex(ValueError, "override"):
                launch.connect_account("new", "new")
            imports.assert_not_called()
        fake = Mock()
        fake.Workspace.from_context.return_value.name = "old-account"
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules) as modules, \
                patch.object(launch.importlib, "import_module", return_value=fake):
            modules.pop("modal", None)
            with self.assertRaisesRegex(ValueError, "account mismatch"):
                launch.execute(self.plan)
            self.assertEqual(os.environ["MODAL_PROFILE"], self.args.profile)
        fake.Secret.from_name.assert_not_called()
        fake.App.lookup.assert_not_called()
        fake.Sandbox.create.assert_not_called()

    def test_training_requires_successful_matching_preparation(self):
        job = self.workspace / "modal-jobs" / self.plan["run_name"]
        other = copy.deepcopy(self.plan)
        other["transformer_source"]["revision"] = "different"
        launch.shared.write_json(job / "prepare.json", {"status": "succeeded", "plan": other})
        with patch.object(launch, "run_command") as child, patch.object(launch.subprocess, "run") as sync:
            with self.assertRaisesRegex(ValueError, "exactly this launch plan"):
                launch.worker(self.plan, "train", workspace=self.workspace)
        child.assert_not_called()
        sync.assert_called_once()
        self.assertEqual(json.loads((job / "train.json").read_text())["status"], "failed")

    def test_cpu_preparation_and_full_state_second_epoch(self):
        commands = []
        with patch.dict(os.environ, dict.fromkeys(launch.SECRETS, "test")), \
                patch.object(launch, "run_command", side_effect=lambda command, **kw: commands.append(command)), \
                patch.object(launch.subprocess, "run") as sync:
            self.assertEqual(launch.worker(self.plan, "prepare", workspace=self.workspace), 0)
            self.assertEqual(launch.worker(self.plan, "train", workspace=self.workspace), 0)
        self.assertEqual(len(commands), 3)
        self.assertIn("--cpu-preparation", commands[0])
        self.assertNotIn("--prepare-only", commands[1])
        self.assertIn("--use-prepared", commands[1])
        self.assertTrue(commands[2][1].endswith("continue_runpod.py"))
        self.assertEqual(commands[2][commands[2].index("--from-run") + 1], "epochs5-6-epoch5")
        self.assertEqual(commands[2][commands[2].index("--augmentation-seed") + 1], "172")
        self.assertEqual(sync.call_count, 2)

    def test_upload_or_training_failure_prevents_second_epoch_and_preserves_status(self):
        job = self.workspace / "modal-jobs" / self.plan["run_name"]
        launch.shared.write_json(job / "prepare.json", {"status": "succeeded", "plan": self.plan})
        with patch.dict(os.environ, dict.fromkeys(launch.SECRETS, "test")), \
                patch.object(launch, "run_command", side_effect=RuntimeError("upload failed")) as child, \
                patch.object(launch.subprocess, "run") as sync:
            with self.assertRaisesRegex(RuntimeError, "upload failed"):
                launch.worker(self.plan, "train", workspace=self.workspace)
        child.assert_called_once()
        sync.assert_called_once()
        self.assertEqual(json.loads((job / "train.json").read_text())["status"], "failed")

    def test_volume_commit_failure_is_not_reported_as_success(self):
        with patch.dict(os.environ, dict.fromkeys(launch.SECRETS, "test")), \
                patch.object(launch, "run_command"), \
                patch.object(launch.subprocess, "run", side_effect=TimeoutError("sync timeout")):
            with self.assertRaisesRegex(TimeoutError, "sync timeout"):
                launch.worker(self.plan, "prepare", workspace=self.workspace)
        status = json.loads((self.workspace / "modal-jobs/epochs5-6/prepare.json").read_text())
        self.assertEqual(status["status"], "failed")

    def test_real_child_timeout_and_failure_are_bounded(self):
        with (self.workspace / "child.log").open("w") as log:
            with self.assertRaises(subprocess.TimeoutExpired):
                launch.run_command([sys.executable, "-c", "import time; time.sleep(10)"], log=log, timeout=0.1)
            with self.assertRaises(subprocess.CalledProcessError):
                launch.run_command([sys.executable, "-c", "print('failed child'); raise SystemExit(7)"],
                                   log=log, timeout=5)
        self.assertIn("failed child", (self.workspace / "child.log").read_text())

    def test_follow_explicitly_terminates_on_finished_entrypoint(self):
        sandbox = Mock(stdout=iter(["done\n"]), stderr=iter([]))
        sandbox.poll.side_effect = [None, 0]
        with patch.object(launch.time, "sleep"), redirect_stdout(io.StringIO()):
            self.assertEqual(launch.follow(sandbox), 0)
        sandbox.terminate.assert_called_once_with(wait=True)

    def test_modal_disk_capacity_is_not_read_from_statfs(self):
        config = load_training_config(launch.DEFAULT_CONFIG)
        with patch.object(launch.shared.shutil, "disk_usage") as disk, redirect_stdout(io.StringIO()):
            launch.shared.preflight_disk(config, self.workspace, 0)
        disk.assert_not_called()

    def test_gpu_is_created_only_after_cpu_success_with_explicit_account_resources(self):
        source = io.BytesIO()
        with tarfile.open(fileobj=source, mode="w:gz") as archive:
            info = tarfile.TarInfo("pyproject.toml")
            info.size = 0
            archive.addfile(info, io.BytesIO())
        manifest = {"files": {self.plan["config"]: {}}}
        for preparation_code in (0, 1):
            with self.subTest(preparation_code=preparation_code):
                fake = Mock()
                fake.enable_output.side_effect = nullcontext
                prepared, trained = Mock(object_id="sb-cpu"), Mock(object_id="sb-gpu")
                fake.Sandbox.create.side_effect = [prepared, trained]
                image = Mock()
                for name in ("pip_install", "env", "workdir", "add_local_dir"):
                    getattr(image, name).return_value = image
                with patch.object(launch, "ROOT", self.workspace), \
                        patch.object(launch, "connect_account", return_value=fake), \
                        patch("scripts.cloud._training_image.training_image", return_value=image), \
                        patch("scripts.cloud.launch_runpod.source_snapshot", return_value=(source.getvalue(), manifest)), \
                        patch.object(launch, "follow", side_effect=[preparation_code, 0]):
                    if preparation_code:
                        with self.assertRaisesRegex(RuntimeError, "prepare failed"):
                            launch.execute(self.plan)
                    else:
                        launch.execute(self.plan)
                calls = fake.Sandbox.create.call_args_list
                self.assertEqual(len(calls), 1 if preparation_code else 2)
                self.assertNotIn("gpu", calls[0].kwargs)
                self.assertEqual(calls[0].kwargs["timeout"], launch.PREPARE_SECONDS)
                if not preparation_code:
                    self.assertEqual(calls[1].kwargs["gpu"], "H200")
                    self.assertEqual(calls[1].kwargs["timeout"], launch.TRAIN_SECONDS)
                    self.assertEqual(calls[0].kwargs["name"], calls[1].kwargs["name"])
                    self.assertIn("worker", calls[1].args)
                    trained.detach.assert_called_once()
                prepared.detach.assert_called_once()
                fake.Volume.from_name.assert_called_once_with(
                    self.plan["volume"], environment_name="main", create_if_missing=True, version=2)
                fake.Secret.from_name.assert_called_once_with(
                    self.plan["secret"], environment_name="main", required_keys=list(launch.SECRETS))

    def test_modal_local_lock_does_not_touch_volume_and_rejects_same_container_writer(self):
        with patch("scripts.cloud.train_runpod.gettempdir", return_value=str(self.workspace)):
            volume = self.workspace / "volume"
            volume.mkdir()
            with launch.shared.workspace_lock(volume, local_only=True):
                with self.assertRaisesRegex(RuntimeError, "Another launcher"):
                    with launch.shared.workspace_lock(volume, local_only=True):
                        self.fail("Concurrent writer was accepted")
            self.assertFalse((volume / ".diffusion-arc.lock").exists())


if __name__ == "__main__":
    unittest.main()
