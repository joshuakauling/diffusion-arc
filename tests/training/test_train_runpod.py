"""Exercise Pod startup failures and upload boundaries without network calls or a GPU."""

from contextlib import ExitStack, contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from requests import Response

from scripts.cloud import train_runpod as launcher
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.run.checkpointing import checkpoint_identity


class RunPodTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.embedded = self.workspace / "image-models"
        embedded_patch = patch.object(launcher, "EMBEDDED_MODELS", self.embedded)
        embedded_patch.start()
        self.addCleanup(embedded_patch.stop)
        self.output = self.workspace / "runs/first"
        self.args = SimpleNamespace(
            config=launcher.REPOSITORY_ROOT / "configs/training/arc_agi_1_epoch.json",
            workspace=self.workspace, resume=None, wandb_project="research", wandb_entity="tester",
        )
        self.config = launcher.resolve_config(self.args, self.output)
        self.api = Mock()
        self.api.whoami.return_value = {"name": "tester"}
        self.api.create_commit.return_value = SimpleNamespace(oid="commit-id")

    @staticmethod
    def hub_error(status):
        response = Response()
        response.status_code = status
        response.url = "https://huggingface.co/api/repos/create"
        error_type = RepositoryNotFoundError if status in (401, 404) else HfHubHTTPError
        return error_type(f"HTTP {status}", response=response)

    @contextmanager
    def hub_write_check(self, status=200):
        self.api.endpoint = "https://huggingface.co"
        self.api.token = "fixture-token"
        response = Response()
        response.status_code = status
        response.url = "https://huggingface.co/api/models/tester/weights/auth-check"
        with patch("huggingface_hub.utils.get_session") as session:
            session.return_value.get.return_value = response
            yield session.return_value.get

    def test_existing_public_and_private_repositories_keep_visibility_and_check_write_access(self):
        for private in (False, True):
            with self.subTest(private=private):
                self.api.repo_info.return_value = SimpleNamespace(private=private)
                with self.hub_write_check() as get:
                    launcher.prepare_output_repository(self.api, "tester/weights", "model")
                self.api.create_repo.assert_not_called()
                self.api.update_repo_settings.assert_not_called()
                self.assertEqual(get.call_args.kwargs["params"], {"write": "true"})

    def test_repository_creation_is_limited_to_missing_destinations(self):
        self.api.repo_info.side_effect = [self.hub_error(404), SimpleNamespace(private=True)]
        with self.hub_write_check():
            launcher.prepare_output_repository(self.api, "tester/weights", "model")
        self.api.create_repo.assert_called_once_with(
            "tester/weights", repo_type="model", private=True, exist_ok=True)

    def test_ambiguous_server_error_rechecks_existence_before_repeating_creation(self):
        self.api.repo_info.side_effect = [self.hub_error(404), SimpleNamespace(private=True)]
        self.api.create_repo.side_effect = self.hub_error(500)
        with self.hub_write_check(), patch.object(launcher.time, "sleep") as sleep:
            launcher.prepare_output_repository(self.api, "tester/weights", "model")
        self.api.create_repo.assert_called_once()
        sleep.assert_called_once_with(2)

    def test_persistent_hub_failure_stops_after_three_attempts(self):
        self.api.repo_info.side_effect = self.hub_error(503)
        with self.hub_write_check(), patch.object(launcher.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts.*HTTP 503"):
                launcher.prepare_output_repository(self.api, "tester/weights", "model")
        self.assertEqual(self.api.repo_info.call_count, 3)
        self.assertEqual([call.args for call in sleep.call_args_list], [(2,), (4,)])
        self.api.create_repo.assert_not_called()

    def test_repository_access_denial_is_not_retried(self):
        for result in (self.hub_error(401), self.hub_error(403)):
            with self.subTest(result=result):
                self.api.reset_mock(side_effect=True)
                self.api.repo_info.side_effect = [result]
                with self.hub_write_check(), patch.object(launcher.time, "sleep") as sleep:
                    with self.assertRaises((HfHubHTTPError, ValueError)):
                        launcher.prepare_output_repository(self.api, "tester/weights", "model")
                sleep.assert_not_called()
                self.api.create_repo.assert_not_called()

    def test_read_access_without_write_access_still_fails(self):
        for private in (False, True):
            with self.subTest(private=private):
                self.api.repo_info.return_value = SimpleNamespace(private=private)
                with self.hub_write_check(status=403), patch.object(launcher.time, "sleep") as sleep:
                    with self.assertRaises(HfHubHTTPError):
                        launcher.prepare_output_repository(self.api, "tester/weights", "model")
                sleep.assert_not_called()
                self.api.create_repo.assert_not_called()

    def checkpoint(self, step=300, run_id="wandb-one"):
        path = self.output / "checkpoints" / run_id / f"step-{step:06d}"
        files = {"transformer/config.json": b"{}", "transformer/weights.safetensors": b"weights",
                 "transformer/weights.safetensors.index.json": b"{}", "training_state.pt": b"optimizer",
                 "transformer/private-notes.txt": b"not for upload"}
        for name, data in files.items():
            file = path / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(data)
        targets = min(step * 48, 43470)
        launcher.write_json(path / "checkpoint.json", {
            "schema_version": 1, "files": {name: len(data) for name, data in files.items()},
            "recipe": self.config.recipe_dict(),
            "identity": checkpoint_identity(self.config, data_fingerprint="data", validation_manifest={}),
            "progress": {"optimizer_step": step, "targets_seen": targets,
                         "sources_seen": targets // 3, "workload_totals": {}},
            "wandb_run": {"id": run_id, "project": "research", "entity": "tester"},
        })
        return path

    def test_missing_secrets_fail_before_sdk_workspace_or_download_work(self):
        with patch.dict(os.environ, {"HF_TOKEN": "secret", "WANDB_API_KEY": "  ",
                                    "KAGGLE_API_TOKEN": "{{ RUNPOD_SECRET_missing }}"}, clear=True), \
                patch("huggingface_hub.HfApi") as api, \
                patch.object(launcher, "workspace_lock") as lock, \
                patch.object(launcher, "download_models") as download:
            with self.assertRaisesRegex(ValueError, "WANDB_API_KEY, KAGGLE_API_TOKEN") as failure:
                launcher.main(["run", "--run-name", "first"])
            self.assertNotIn("secret", str(failure.exception).removeprefix("Missing environment secrets"))
        api.assert_not_called()
        lock.assert_not_called()
        download.assert_not_called()

    def test_resolved_config_preserves_science_and_uses_workspace_paths(self):
        original = load_training_config(self.args.config).recipe_dict()
        actual = self.config.recipe_dict()
        for field in ("optimizer", "scheduler", "flow", "loss", "loop", "execution"):
            self.assertEqual(actual[field], original[field])
        self.assertEqual(self.config.checkpoint_steps, (300, 600, 906))
        self.assertEqual(actual["models"]["nvarc_checkpoint"], str(self.workspace / "models/nvarc-qwen3-4b-grids15-sft139"))
        self.assertEqual(actual["logging"]["project"], "research")
        self.assertFalse(self.output.exists())

    def embedded_bundle(self):
        files = {
            "nvarc": {"config.json": b"{}", "tokenizer.json": b"{}", "tokenizer_config.json": b"{}",
                      "model.safetensors.index.json": b'{"weight_map": {"weight": "weights.safetensors"}}',
                      "weights.safetensors": b"tiny-model"},
            "flux": {"transformer/config.json": b"{}", "transformer/model.safetensors": b"flux",
                     "vae/config.json": b"{}", "vae/model.safetensors": b"vae"},
        }
        manifest = {"schema_version": 1,
                    "nvarc": {"handle": launcher.NVARC_HANDLE, "path": "nvarc-qwen3-4b-grids15-sft139"},
                    "flux": {"repo_id": launcher.FLUX_REPO, "revision": launcher.FLUX_REVISION,
                             "path": "flux2-klein-base-4b"}}
        for component, content in files.items():
            entry = manifest[component]
            entry["files"] = {}
            for name, data in content.items():
                path = self.embedded / entry["path"] / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                entry["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        launcher.write_json(self.embedded / "manifest.json", manifest)
        weights_patch = patch.object(launcher, "NVARC_WEIGHT_SIZES", {"weights.safetensors": 10})
        weights_patch.start()
        self.addCleanup(weights_patch.stop)
        return manifest

    def test_embedded_config_preserves_recipe_and_skips_remote_model_work(self):
        self.embedded_bundle()
        keys = self.workspace / "keys.json"
        keys.write_text("[]")
        model_download = Mock(side_effect=AssertionError("embedded model must not be downloaded"))
        dataset_download = Mock(return_value=str(keys))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {
                "HF_TOKEN": "hf", "WANDB_API_KEY": "wandb", "KAGGLE_API_TOKEN": "kaggle",
            }))
            stack.enter_context(patch.dict("sys.modules", {
                "wandb": SimpleNamespace(Api=Mock(return_value=SimpleNamespace(viewer={"id": "test"}))),
                "kagglehub": SimpleNamespace(model_download=model_download, dataset_download=dataset_download),
            }))
            stack.enter_context(patch("huggingface_hub.HfApi", return_value=self.api))
            stack.enter_context(patch.object(launcher, "DATA_SHA256", {
                **launcher.DATA_SHA256, "arc_agi_1_training_keys.json": launcher.sha256_file(keys),
            }))
            private = stack.enter_context(patch.object(launcher, "prepare_output_repository"))
            data = stack.enter_context(patch.object(launcher, "prepare_data"))
            download = stack.enter_context(patch.object(launcher, "download_models"))
            train = stack.enter_context(patch.object(launcher.subprocess, "run"))
            for name in ("prepare_runtime", "preflight_gpu_and_disk"):
                stack.enter_context(patch.object(launcher, name))
            self.assertEqual(launcher.main([
                "run", "--workspace", str(self.workspace), "--run-name", "first", "--prepare-only",
            ]), 0)
        config = load_training_config(self.output / "training.json")
        self.assertEqual(config.models.nvarc_checkpoint, self.embedded / "nvarc-qwen3-4b-grids15-sft139")
        self.assertEqual(config.models.flux_checkpoint, self.embedded / "flux2-klein-base-4b")
        for field in ("optimizer", "scheduler", "flow", "loss", "loop", "execution"):
            self.assertEqual(config.recipe_dict()[field], self.config.recipe_dict()[field])
        model_download.assert_not_called()
        download.assert_not_called()
        self.api.auth_check.assert_not_called()
        train.assert_not_called()
        private.assert_called_once_with(self.api, "tester/diffusion-arc-weights", "model")
        dataset_download.assert_called_once_with(launcher.DATA_HANDLE, path="arc_agi_1_training_keys.json")
        data.assert_called_once()
        receipt = json.loads((self.output / "startup.json").read_text())
        self.assertEqual(receipt["models"], {
            "provider": "image", "manifest_sha256": launcher.sha256_file(self.embedded / "manifest.json"),
        })

    def test_embedded_manifest_identity_and_file_integrity_fail_closed(self):
        manifest = self.embedded_bundle()
        manifest_path = self.embedded / "manifest.json"
        original = manifest_path.read_bytes()
        for component, field in (("nvarc", "handle"), ("flux", "repo_id"), ("flux", "revision"), ("flux", "path")):
            with self.subTest(component=component, field=field):
                modified = json.loads(original)
                modified[component][field] = "unexpected"
                launcher.write_json(manifest_path, modified)
                with self.assertRaisesRegex(ValueError, "source differs"):
                    launcher.embedded_model_paths()
        manifest_path.write_bytes(original)
        root = self.embedded / manifest["flux"]["path"]
        weight = root / "vae/model.safetensors"
        weight.write_bytes(b"VAe")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            launcher.embedded_model_paths()
        weight.write_bytes(b"truncated")
        with self.assertRaisesRegex(ValueError, "file is incomplete"):
            launcher.embedded_model_paths()
        weight.unlink()
        with self.assertRaisesRegex(ValueError, "file inventory differs"):
            launcher.embedded_model_paths()
        weight.symlink_to(self.workspace / "outside")
        with self.assertRaisesRegex(ValueError, "non-regular"):
            launcher.embedded_model_paths()

    def test_prepared_embedded_models_reuse_receipt_without_weight_reads_or_inventory_scan(self):
        self.embedded_bundle()
        with patch.object(Path, "rglob", side_effect=AssertionError("inventory scan")), \
                patch.object(launcher, "verify_file", side_effect=AssertionError("weight rehash")):
            paths = launcher.embedded_model_paths(verify_payloads=False)
        self.assertEqual(set(paths), {"nvarc_checkpoint", "flux_checkpoint"})

    def test_embedded_manifest_requires_local_complete_model_components(self):
        manifest = self.embedded_bundle()
        root = self.embedded / manifest["flux"]["path"]
        manifest_path = self.embedded / "manifest.json"
        manifest["flux"]["files"]["../outside"] = {"bytes": 1, "sha256": "0" * 64}
        launcher.write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "file inventory differs"):
            launcher.embedded_model_paths()
        del manifest["flux"]["files"]["../outside"]
        del manifest["flux"]["files"]["vae/config.json"]
        (root / "vae/config.json").unlink()
        launcher.write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "missing vae config or weights"):
            launcher.embedded_model_paths()

    def test_missing_embedded_manifest_retains_legacy_download_paths(self):
        self.assertIsNone(launcher.embedded_model_paths())
        resolved = launcher.resolve_config(self.args, self.output)
        self.assertEqual(resolved.models, self.config.models)

    def test_runtime_adds_kaggle_client_with_all_installed_versions_constrained(self):
        (self.workspace / "pyproject.toml").write_text(
            '[project]\ndependencies = ["torch==2.13.0", "diffusers==0.39.0"]\n')
        installed = {"torch": "2.13.0+cu130", "diffusers": "0.39.0", "requests": "2.32.5"}

        def version(name):
            if name not in installed:
                raise PackageNotFoundError(name)
            return installed[name]

        def check_install(command, **kwargs):
            if "install" in command:
                constraints = Path(command[command.index("--constraint") + 1]).read_text()
                self.assertEqual(set(constraints.splitlines()), {
                    "torch==2.13.0+cu130", "diffusers==0.39.0", "requests==2.32.5",
                })
                self.assertEqual(command[-1], "kagglehub==1.0.2")

        packages = [SimpleNamespace(metadata={"Name": name}, version=value)
                    for name, value in installed.items()]
        with patch.object(launcher, "REPOSITORY_ROOT", self.workspace), \
                patch("importlib.metadata.version", side_effect=version), \
                patch("importlib.metadata.distributions", return_value=packages), \
                patch.object(launcher.subprocess, "run", side_effect=check_install) as run:
            launcher.prepare_runtime()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0][-2:], ["pip", "check"])

    def test_runtime_rejects_incompatible_packages_without_reinstalling_them(self):
        (self.workspace / "pyproject.toml").write_text('[project]\ndependencies = ["torch==2.13.0"]\n')
        for versions, message in ((["2.12.0"], "requires torch==2.13.0"),
                                  ([PackageNotFoundError("torch")], "missing torch==2.13.0"),
                                  (["2.13.0+cu130", "0.9.0"], "Expected kagglehub==1.0.2")):
            with self.subTest(message=message), \
                    patch.object(launcher, "REPOSITORY_ROOT", self.workspace), \
                    patch("importlib.metadata.version", side_effect=versions), \
                    patch.object(launcher.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, message):
                    launcher.prepare_runtime()
                run.assert_not_called()

    def test_prepared_runtime_does_not_run_pip(self):
        (self.workspace / "pyproject.toml").write_text('[project]\ndependencies = ["torch==2.13.0"]\n')
        with patch.object(launcher, "REPOSITORY_ROOT", self.workspace), \
                patch("importlib.metadata.version", side_effect=["2.13.0+cu130", "1.0.2"]), \
                patch.object(launcher.subprocess, "run") as run:
            launcher.prepare_runtime()
            run.assert_not_called()

    def test_resume_requires_latest_checkpoint_and_keeps_model_paths(self):
        first = self.checkpoint(300)
        latest = self.checkpoint(600)
        with self.assertRaisesRegex(ValueError, "latest complete"):
            launcher.validate_start(self.config, self.output, first)
        self.args.resume = latest
        resumed = launcher.resolve_config(self.args, self.output)
        self.assertEqual(launcher.validate_start(resumed, self.output, latest), 600)
        self.assertEqual(resumed.models, self.config.models)
        with self.assertRaisesRegex(ValueError, "already attempted"):
            launcher.validate_start(self.config, self.output, None)
        final = self.checkpoint(906)
        with self.assertRaisesRegex(ValueError, "Epoch is complete"):
            launcher.validate_start(self.config, self.output, final)

    def test_failed_training_without_checkpoint_requires_new_run_name(self):
        launcher.write_json(self.output / "training-started.json", {"resume_from": None})
        with self.assertRaisesRegex(ValueError, "already attempted"):
            launcher.validate_start(self.config, self.output, None)

    def test_pruned_states_are_skipped_for_resume_but_weights_remain_uploadable(self):
        from diffusion_arc.training.run.checkpointing import load_checkpoint_metadata, remove_training_state

        paths = [self.checkpoint(step) for step in (300, 600, 906)]
        for path in paths[:2]:
            remove_training_state(load_checkpoint_metadata(path, self.config))
        self.assertEqual([item.path for item in launcher.run_checkpoints(self.config)], [paths[-1]])
        self.assertEqual([item.path for item in launcher.run_checkpoints(self.config, include_weights_only=True)], paths)
        launcher.upload_weights(self.config, api=self.api, output_repo="tester/weights")
        self.assertEqual(self.api.create_commit.call_count, 3)
        self.assertTrue((paths[-1] / "training_state.pt").exists())
        (paths[-1] / "training_state.pt").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete checkpoint inventory"):
            launcher.run_checkpoints(self.config)

    def test_model_alias_is_stable_across_preparation_training_and_resume(self):
        cache = self.workspace / ".cache/kagglehub/model-version-1"
        cache.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
            (cache / name).write_text("{}")
        (cache / "weights.safetensors").write_bytes(b"tiny-model")
        with patch.dict("sys.modules", {"kagglehub": SimpleNamespace(model_download=Mock(return_value=str(cache)))}), \
                patch.dict(os.environ, {"HF_TOKEN": "hf"}), \
                patch.object(launcher, "NVARC_WEIGHT_SIZES", {"weights.safetensors": 10}), \
                patch("huggingface_hub.snapshot_download") as download:
            launcher.download_models(self.config, workspace=self.workspace)
            self.assertIn("transformer/*.safetensors", download.call_args.kwargs["allow_patterns"])
            launcher.download_models(self.config, workspace=self.workspace, include_transformer=False)
            self.assertEqual(download.call_args.kwargs["allow_patterns"], ["vae/*.json", "vae/*.safetensors"])
            self.config = launcher.resolve_config(self.args, self.output)
            launcher.write_json(self.config.path, self.config.recipe_dict())
            training_config = load_training_config(self.config.path)
            self.assertEqual(training_config.models.nvarc_checkpoint, cache)
            self.assertEqual(training_config.recipe_dict(), self.config.recipe_dict())
            # The repeated download still checks the fixed alias, even though config holds its target.
            launcher.download_models(self.config, workspace=self.workspace)
            path = self.checkpoint()
            self.args.resume = path
            resumed = launcher.resolve_config(self.args, self.output)
            self.assertEqual(launcher.validate_start(resumed, self.output, path), 300)
            launcher.upload_weights(training_config, api=self.api, output_repo="tester/weights")

    def test_multiple_runs_and_truncated_checkpoints_are_rejected(self):
        path = self.checkpoint()
        (path / "training_state.pt").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "file size differs"):
            launcher.run_checkpoints(self.config)
        self.checkpoint(run_id="other-run")
        with self.assertRaisesRegex(ValueError, "multiple W&B"):
            launcher.run_checkpoints(self.config)

    def test_upload_only_sends_transformer_files_and_writes_receipt_after_success(self):
        checkpoint = self.checkpoint()
        launcher.upload_weights(self.config, api=self.api, output_repo="tester/weights")
        operations = self.api.create_commit.call_args.kwargs["operations"]
        self.assertEqual({operation.path_in_repo for operation in operations}, {
            "wandb-one/step-000300/transformer/config.json",
            "wandb-one/step-000300/transformer/weights.safetensors",
            "wandb-one/step-000300/transformer/weights.safetensors.index.json",
        })
        receipt = self.output / "uploads/step-000300.json"
        self.assertEqual(json.loads(receipt.read_text())["commit"], "commit-id")
        receipt.unlink()
        self.api.create_commit.side_effect = RuntimeError("offline")
        with self.assertRaisesRegex(RuntimeError, "offline"):
            launcher.upload_weights(self.config, api=self.api, output_repo="tester/weights")
        self.assertFalse(receipt.exists())
        self.assertTrue((checkpoint / "training_state.pt").exists())

    def test_upload_command_works_without_wandb_kaggle_or_gpu(self):
        self.checkpoint(906)
        launcher.write_json(self.config.path, self.config.recipe_dict())
        launcher.write_json(self.output / "startup.json", {"output_repo": "tester/original-destination"})
        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=True), \
                patch("huggingface_hub.HfApi", return_value=self.api), \
                patch.object(launcher, "prepare_output_repository"), \
                patch.object(launcher, "download_models") as download, \
                patch.object(launcher, "preflight_gpu_and_disk") as gpu, \
                patch.object(launcher.subprocess, "run") as train:
            result = launcher.main(["upload", "--workspace", str(self.workspace), "--run-name", "first"])
        self.assertEqual(result, 0)
        self.assertEqual(self.api.create_commit.call_count, 1)
        self.assertEqual(self.api.create_commit.call_args.kwargs["repo_id"], "tester/original-destination")
        download.assert_not_called()
        gpu.assert_not_called()
        train.assert_not_called()

    def test_disk_failure_prevents_gpu_execution(self):
        with patch.object(launcher.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * 1024**3)), \
                patch.object(launcher.subprocess, "run") as gpu:
            with self.assertRaisesRegex(ValueError, "190 GiB free"):
                launcher.preflight_gpu_and_disk(self.config, self.workspace, 0)
        gpu.assert_not_called()

    def ddp_config(self, count=2):
        payload = self.config.recipe_dict()
        payload["execution"].update(distributed_strategy="ddp", flux_packing_algorithm="balanced_release_v1")
        payload["flow"]["rng_scheme"] = "global_step_target_v1"
        payload["deployment"]["gpu_count"] = count
        path = self.workspace / "ddp.json"
        launcher.write_json(path, payload)
        return load_training_config(path)

    def test_training_command_preserves_stage_arguments_and_launches_configured_rank_count(self):
        commands = ([sys.executable, "scripts/train.py", "--config", "recipe.json"],
                    [sys.executable, "-c", "from scripts.cloud.continue_runpod import train; "
                     "import sys; train(*sys.argv[1:])", "recipe.json", "parent run"],
                    [sys.executable, "-c", "from scripts.cloud.finetune_runpod import train; "
                     "import sys; train(*sys.argv[1:])", "recipe.json", "startup.json"])
        for command in commands:
            self.assertEqual(launcher.training_command(self.config, command), command)
            for count in (1, 2):
                launched = launcher.training_command(self.ddp_config(count), command)
                self.assertEqual(launched[:3], [sys.executable, "-m", "torch.distributed.run"])
                self.assertIn(f"--nproc-per-node={count}", launched)
                self.assertIn("--max-restarts=0", launched)
                self.assertEqual(launched[launched.index("--no-python") + 1:], command)

    def test_ddp_preflight_checks_each_gpu_and_rejects_missing_devices(self):
        config = self.ddp_config()
        cuda = Mock()
        cuda.is_available.return_value = True
        cuda.device_count.return_value = 2
        cuda.get_device_name.return_value = "NVIDIA H200"
        cuda.get_device_capability.return_value = (9, 0)
        from contextlib import nullcontext
        cuda.device.side_effect = lambda index: nullcontext()
        backend = Mock()
        with patch.object(launcher, "preflight_disk"), \
                patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda),
                                        "diffusion_arc.model.flux.packing": SimpleNamespace(Flash3Backend=backend)}), \
                patch.object(launcher.subprocess, "run", side_effect=lambda command, **kw: exec(command[2], {})):
            launcher.preflight_gpu_and_disk(config, self.workspace, 0)
            self.assertEqual([call.args for call in backend.call_args_list], [("cuda:0",), ("cuda:1",)])
            backend.reset_mock()
            cuda.device_count.return_value = 1
            with self.assertRaisesRegex(AssertionError, "exactly 2"):
                launcher.preflight_gpu_and_disk(config, self.workspace, 0)
            backend.assert_not_called()

    def test_data_preparation_reuses_manifest_without_rechecking_task_contents(self):
        from tests.arc_data.test_nvarc_benchmark import _fixture
        from scripts.data.build_nvarc_arc_agi_1_training_only import canonical_jsonl_record

        source = self.workspace / "remote-data"
        records = _fixture(source, task_count=58)
        tree = source / "nvarc-all-classified"
        with tarfile.open(source / "nvarc-all-classified.tar.gz") as archive:
            archive.extractall(tree, filter="data")
        index = tree / "metadata/views/nvarc_train_tasks.jsonl"
        index.parent.mkdir(parents=True)
        shutil.copyfile(source / "nvarc_train_arc_agi_1_training_only_tasks.jsonl", index)
        hashes = {
            "nvarc_train_tasks.jsonl": launcher.sha256_file(index),
            "arc_agi_1_training_keys.json": launcher.sha256_file(source / "arc_agi_1_training_keys.json"),
            "nvarc_train_arc_agi_1_training_only_tasks.jsonl": hashlib.sha256(
                b"".join(canonical_jsonl_record(record) for record in records)).hexdigest(),
        }
        payload = self.config.recipe_dict()
        payload["data"]["expected_task_count"] = 58
        payload["scheduler"].update(warmup_steps=3, first_update_warmup_fraction=1 / 3)
        launcher.write_json(self.config.path, payload)
        config = load_training_config(self.config.path)
        # Validation preparation has separate tests; exercise only training data here.
        config.recipe["validation"]["bundle"].mkdir(parents=True)

        def download(handle, *, output_dir):
            destination = Path(output_dir)
            if not destination.exists():
                shutil.copytree(source, destination)
            self.assertEqual(handle, "joshuakauling/nvarc-data/versions/1")
            return str(destination)

        with patch.object(launcher, "DATA_SHA256", hashes), \
                patch("scripts.data.build_nvarc_arc_agi_1_training_only.EXPECTED_SOURCE_TASKS", 58), \
                patch.dict("sys.modules", {"kagglehub": SimpleNamespace(dataset_download=Mock(side_effect=download))}), \
                patch("huggingface_hub.hf_hub_download") as hf_download, \
                patch("diffusion_arc.training.run.validation.validate_validation_bundle"):
            arguments = dict(workspace=self.workspace)
            launcher.prepare_data(config, **arguments)
            hf_download.assert_not_called()
            manifest = json.loads((config.data.split_directory / "manifest.json").read_text())
            self.assertEqual(manifest["source_count"], 58)
            self.assertEqual(manifest["target_count"], 174)
            self.assertEqual(manifest["augmentation"]["geometry"], "uniform_D4")
            raw = config.data.split_directory.parent.parent / "raw"
            with patch.object(launcher, "verify_file", wraps=launcher.verify_file) as verify:
                launcher.prepare_data(config, **arguments)
            self.assertTrue(verify.called)  # Small metadata files still identify the source pool.
            self.assertFalse(any(call.args[0].is_relative_to(raw) for call in verify.call_args_list))
            # Source files are trusted after preparation and parsed when consumed.
            from diffusion_arc.training.data import ArcTrainingDataset
            dataset = ArcTrainingDataset(config.data.split_directory, config.data.view_name, tokenizer=None)
            task = dataset.source_directory / f"{dataset.keys[0].task_id}.json"
            task.write_text("{}")
            launcher.prepare_data(config, **arguments)
            with self.assertRaisesRegex(TypeError, "'train' must be a list"):
                dataset[0]

    def test_prepare_only_then_run_invokes_existing_training_cli_and_upload(self):
        kaggle_config = self.workspace / "config.json"
        kaggle_config.write_text("{}")
        keys = self.workspace / "arc_agi_1_training_keys.json"
        keys.write_text('{"fixture": true}')
        dataset_download = Mock(return_value=str(keys))
        events = []
        with ExitStack() as stack:
            stack.enter_context(patch.object(launcher, "DATA_SHA256", {
                **launcher.DATA_SHA256, keys.name: launcher.sha256_file(keys),
            }))
            stack.enter_context(patch.dict(os.environ, {
                "HF_TOKEN": "hf", "WANDB_API_KEY": "wandb", "KAGGLE_API_TOKEN": "kaggle",
            }))
            stack.enter_context(patch.dict("sys.modules", {
                "wandb": SimpleNamespace(Api=Mock(return_value=SimpleNamespace(viewer={"id": "test"}))),
                "kagglehub": SimpleNamespace(model_download=Mock(return_value=str(kaggle_config)),
                                             dataset_download=dataset_download),
            }))
            stack.enter_context(patch("huggingface_hub.HfApi", return_value=self.api))
            self.api.repo_info.return_value = SimpleNamespace(sha="pinned-revision")
            for name in ("prepare_output_repository", "prepare_runtime", "preflight_gpu_and_disk", "prepare_data", "download_models", "upload_weights"):
                stack.enter_context(patch.object(launcher, name, side_effect=lambda *a, _name=name, **k: events.append(_name)))
            train = stack.enter_context(patch.object(launcher.subprocess, "run"))
            arguments = ["run", "--workspace", str(self.workspace), "--run-name", "first"]
            dataset_download.side_effect = PermissionError("private dataset access denied")
            with self.assertRaisesRegex(PermissionError, "dataset access denied"):
                launcher.main([*arguments, "--prepare-only"])
            dataset_download.side_effect = None
            keys.write_text("wrong dataset content")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                launcher.main([*arguments, "--prepare-only"])
            self.assertNotIn("download_models", events)
            self.assertNotIn("prepare_data", events)
            train.assert_not_called()
            self.assertFalse((self.output / "startup.json").exists())
            keys.write_text('{"fixture": true}')
            events.clear()
            self.assertEqual(launcher.main([*arguments, "--prepare-only"]), 0)
            train.assert_not_called()
            self.assertNotIn("upload_weights", events)
            self.assertFalse((self.output / "training-started.json").exists())
            self.assertEqual(launcher.main(arguments), 0)
            self.assertTrue((self.output / "training-started.json").exists())
            self.assertEqual(train.call_args.args[0][-2:], ["--config", str(self.config.path)])
            self.assertEqual(os.environ["PYTHONPATH"].split(os.pathsep)[0], str(launcher.REPOSITORY_ROOT / "src"))
            self.assertEqual(events[-1], "upload_weights")
            dataset_download.assert_called_with(launcher.DATA_HANDLE, path="arc_agi_1_training_keys.json")
            self.api.auth_check.assert_called_with(launcher.FLUX_REPO)
            self.assertTrue(all(call.args == (launcher.FLUX_REPO,) for call in self.api.auth_check.call_args_list))
            receipt = json.loads((self.output / "startup.json").read_text())
            self.assertEqual(receipt["data"]["handle"], launcher.DATA_HANDLE)


if __name__ == "__main__":
    unittest.main()
