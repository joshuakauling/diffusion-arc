"""Verify pinned weights-only startup and local-terminal dispatch without remote work."""

from contextlib import ExitStack
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.cloud import finetune_runpod as launch
from scripts.cloud import launch_runpod
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.run.checkpointing import checkpoint_identity


class FinetuneRunPodTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workspace = Path(temp.name).resolve()
        self.config = load_training_config(launch.DEFAULT_CONFIG)
        self.parent_config = load_training_config(launch.DEFAULT_CONFIG.with_name("arc_agi_1_epoch.json"))
        summary = {f"{phase}/{name}_total": 12 for phase in ("train", "val", "sample")
                   for name in self.config.recipe["logging"]["workload_counters"]["names"]}
        self.run = SimpleNamespace(id="m7y2mfn2", project="diffusion-arc", entity="owner",
                                   state="finished", lastHistoryStep=1812, summary=summary,
                                   config=self.parent_config.recipe_dict())
        self.wandb_api = Mock(default_entity="owner", viewer={"id": "viewer"})
        self.wandb_api.run.return_value = self.run
        self.api = Mock()
        self.transformer = self.workspace / "download/transformer"
        self.source = {**self.config.recipe_dict()["models"]["transformer_source"], "files": {}}

    def mocks(self, stack, *, train_failure=False):
        stack.enter_context(patch.dict(os.environ, {"HF_TOKEN": "hf", "WANDB_API_KEY": "wb", "KAGGLE_API_TOKEN": "kg"}))
        stack.enter_context(patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=Mock(return_value=self.wandb_api))}))
        stack.enter_context(patch("huggingface_hub.HfApi", return_value=self.api))
        result = {name: stack.enter_context(patch.object(launch.pod, name)) for name in (
            "prepare_runtime", "prepare_output_repository", "preflight_gpu_and_disk", "preflight_disk", "preflight_data_access", "prepare_data",
            "download_models", "upload_weights")}
        result["embedded"] = stack.enter_context(patch.object(launch.pod, "embedded_model_paths", return_value={
            "nvarc_checkpoint": self.workspace / "image/nvarc", "flux_checkpoint": self.workspace / "image/flux"}))
        result["transformer"] = stack.enter_context(patch.object(launch, "download_transformer", return_value=(self.transformer, self.source)))
        result["train"] = stack.enter_context(patch.object(launch.subprocess, "run"))
        if train_failure:
            result["train"].side_effect = launch.subprocess.CalledProcessError(1, ["training"])
        return result

    def arguments(self, *extra):
        return ["--workspace", str(self.workspace), "--run-name", "epoch3", *extra]

    def test_preparation_pins_export_uses_fresh_view_and_preserves_frozen_models(self):
        with ExitStack() as stack:
            mocks = self.mocks(stack)
            self.assertEqual(launch.main(self.arguments("--prepare-only")), 0)
        config = load_training_config(self.workspace / "runs/epoch3/training.json")
        self.assertEqual(config.recipe["models"]["transformer_checkpoint"], self.transformer)
        self.assertEqual(config.models.flux_checkpoint, self.workspace / "image/flux")
        self.assertEqual(config.data.split_directory, self.workspace / "runs/epoch3/data/splits/train")
        self.assertEqual(config.seed, 169)
        self.assertEqual(config.recipe["validation"]["sampling"], self.parent_config.recipe["validation"]["sampling"])
        mocks["transformer"].assert_called_once_with(self.config.recipe_dict()["models"]["transformer_source"],
                                                    workspace=self.workspace, max_gib=24)
        mocks["download_models"].assert_not_called()
        mocks["train"].assert_not_called()
        receipt = json.loads((self.workspace / "runs/epoch3/startup.json").read_text())
        self.assertEqual(receipt["parent_run"]["wandb_run"]["id"], "m7y2mfn2")
        self.assertEqual(receipt["output_repo"], "flipchip167/diffusion-arc-weights")

    def test_successful_child_uploads_weights_and_failed_child_does_not(self):
        for failed in (False, True):
            with self.subTest(failed=failed), ExitStack() as stack:
                mocks = self.mocks(stack, train_failure=failed)
                args = ["--workspace", str(self.workspace), "--run-name", f"run-{failed}"]
                if failed:
                    with self.assertRaises(launch.subprocess.CalledProcessError):
                        launch.main(args)
                    mocks["upload_weights"].assert_not_called()
                else:
                    self.assertEqual(launch.main(args), 0)
                    mocks["upload_weights"].assert_called_once()
                command = mocks["train"].call_args.args[0]
                self.assertIn("from scripts.cloud.finetune_runpod import train", command[2])
                self.assertTrue((self.workspace / f"runs/run-{failed}/training-started.json").is_file())

    def test_weights_only_start_can_move_from_legacy_single_parent_to_ddp(self):
        payload = self.config.recipe_dict()
        payload["execution"].update(distributed_strategy="ddp", flux_packing_algorithm="balanced_release_v1")
        payload["flow"]["rng_scheme"] = "global_step_target_v1"
        payload["deployment"]["gpu_count"] = 2
        path = self.workspace / "ddp.json"
        path.write_text(json.dumps(payload))
        # Real parent W&B runs predate the explicitly versioned defaults.
        for key in ("flux_packing_algorithm", "flux_max_spread_tokens"):
            self.run.config["execution"].pop(key, None)
        self.run.config["flow"].pop("rng_scheme", None)
        with ExitStack() as stack:
            mocks = self.mocks(stack)
            self.assertEqual(launch.main(self.arguments("--config", str(path))), 0)
        command = mocks["train"].call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-m", "torch.distributed.run"])
        self.assertIn("--nproc-per-node=2", command)
        self.assertIn("--max-restarts=0", command)
        self.assertIn("from scripts.cloud.finetune_runpod import train", command[-3])
        mocks["upload_weights"].assert_called_once()
        receipt = json.loads((self.workspace / "runs/epoch3/startup.json").read_text())
        self.assertEqual(receipt["execution"], {"strategy": "ddp", "gpu_count": 2})
        self.run.config["execution"]["nvarc_attention_backend"] = "different"
        with self.assertRaisesRegex(ValueError, "execution differs"):
            launch.check_parent_run(self.wandb_api, load_training_config(path))

    def test_resume_uses_new_stage_checkpoint_without_downloading_or_resetting_again(self):
        with ExitStack() as stack:
            mocks = self.mocks(stack)
            launch.main(self.arguments("--prepare-only"))
            output = self.workspace / "runs/epoch3"
            config = load_training_config(output / "training.json")
            checkpoint = output / "checkpoints/m7y2mfn2/step-002112"
            files = {"transformer/config.json": b"{}", "transformer/model.safetensors": b"weights",
                     "training_state.pt": b"new epoch optimizer state"}
            for name, content in files.items():
                path = checkpoint / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            launch.pod.write_json(checkpoint / "checkpoint.json", {
                "schema_version": 1, "files": {name: len(value) for name, value in files.items()},
                "recipe": config.recipe_dict(),
                "identity": checkpoint_identity(config, data_fingerprint="data", validation_manifest={}),
                "progress": {"optimizer_step": 2112, "epoch": 3, "optimizer_start_step": 1812,
                             "targets_seen": 101340, "sources_seen": 33780, "workload_totals": {}},
                "wandb_run": {"id": "m7y2mfn2", "project": "diffusion-arc", "entity": "owner"},
            })
            self.run.lastHistoryStep = 2120
            mocks["transformer"].reset_mock()
            self.assertEqual(launch.main(self.arguments("--resume", str(checkpoint))), 0)
            mocks["transformer"].assert_not_called()
            self.assertEqual(mocks["preflight_gpu_and_disk"].call_args.args[2], 300)
            resolved = load_training_config(output / "training.json")
            self.assertEqual(resolved.recipe["models"]["resume_from"], checkpoint)

    def test_private_data_denial_fails_before_large_downloads(self):
        with ExitStack() as stack:
            mocks = self.mocks(stack)
            mocks["preflight_data_access"].side_effect = ValueError("private data denied")
            with self.assertRaisesRegex(ValueError, "private data denied"):
                launch.main(self.arguments())
            mocks["download_models"].assert_not_called()
            mocks["transformer"].assert_not_called()

    def test_cpu_preparation_skips_gpu_checks_but_cannot_train(self):
        with ExitStack() as stack:
            mocks = self.mocks(stack)
            launch.main(self.arguments("--prepare-only", "--cpu-preparation"))
            mocks["preflight_gpu_and_disk"].assert_not_called()
            mocks["preflight_disk"].assert_called()
            mocks["transformer"].assert_called_once()
            mocks["train"].assert_not_called()
        with self.assertRaises(SystemExit):
            launch.main(self.arguments("--cpu-preparation"))

    def test_prepared_restart_and_gpu_handoff_skip_all_asset_preparation(self):
        with ExitStack() as stack:
            mocks = self.mocks(stack)
            launch.main(self.arguments("--prepare-only", "--cpu-preparation"))
            # This also represents a successful receipt from the previous launcher:
            # it needs no new marker or source-file audit to be reusable.
            for options in (("--prepare-only", "--cpu-preparation"), ("--use-prepared",)):
                for name in ("transformer", "download_models", "prepare_data", "preflight_data_access", "train"):
                    mocks[name].reset_mock()
                self.assertEqual(launch.main(self.arguments(*options)), 0)
                for name in ("transformer", "download_models", "prepare_data", "preflight_data_access"):
                    mocks[name].assert_not_called()
                if "--use-prepared" in options:
                    mocks["train"].assert_called_once()
                else:
                    mocks["train"].assert_not_called()

    def test_gpu_requires_receipt_and_prepared_recipe_cannot_change(self):
        with ExitStack() as stack:
            mocks = self.mocks(stack)
            with self.assertRaisesRegex(ValueError, "startup receipt"):
                launch.main(self.arguments("--use-prepared"))
            mocks["transformer"].assert_not_called()
            launch.main(self.arguments("--prepare-only"))
            path = self.workspace / "changed.json"
            recipe = self.config.recipe_dict()
            recipe["seed"] += 1
            launch.pod.write_json(path, recipe)
            mocks["transformer"].reset_mock()
            with self.assertRaisesRegex(ValueError, "Prepared run differs"):
                launch.main(self.arguments("--config", str(path), "--use-prepared"))
            mocks["transformer"].assert_not_called()
            mocks["train"].assert_not_called()

    def test_image_without_models_downloads_only_frozen_base_components(self):
        with ExitStack() as stack:
            mocks = self.mocks(stack)
            mocks["embedded"].return_value = None
            launch.main(self.arguments("--prepare-only"))
            self.assertFalse(mocks["download_models"].call_args.kwargs["include_transformer"])
            self.api.auth_check.assert_called_once_with(launch.pod.FLUX_REPO)

    def test_stale_active_or_incompatible_parent_fails_before_download(self):
        for change in (lambda: setattr(self.run, "lastHistoryStep", 1900),
                       lambda: setattr(self.run, "state", "running"),
                       lambda: self.run.config["validation"]["sampling"].update(steps=20)):
            self.setUp()
            change()
            with ExitStack() as stack:
                mocks = self.mocks(stack)
                with self.assertRaises(ValueError):
                    launch.main(self.arguments("--prepare-only"))
                mocks["transformer"].assert_not_called()
                mocks["train"].assert_not_called()

    def test_missing_secrets_fail_before_connections_or_downloads(self):
        with patch.dict(os.environ, {}, clear=True), patch("huggingface_hub.HfApi") as api:
            with self.assertRaisesRegex(ValueError, "HF_TOKEN, WANDB_API_KEY, KAGGLE_API_TOKEN"):
                launch.main(self.arguments())
        api.assert_not_called()

    def test_local_terminal_dry_run_selects_new_recipe_and_uploads_all_dependencies(self):
        with patch("sys.stdout", new_callable=io.StringIO) as output, patch.object(launch_runpod, "remote_python") as remote:
            result = launch.main(["--ssh", "ssh example@ssh.runpod.io -i ~/.ssh/id_rsa",
                                  "--run-name", "epoch3", "--dry-run"])
        self.assertEqual(result, 0)
        remote.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["arguments"][0], "finetune")
        self.assertIn("configs/training/arc_agi_1_epoch3_high_noise.json", plan["arguments"])
        self.assertEqual(plan["flow"]["endpoint_targets_per_full_batch"], 24)
        self.assertEqual(plan["initialization"]["transformer_source"]["revision"], "1126da2fc1a43c210d102f75d385f17279003460")


if __name__ == "__main__":
    unittest.main()
