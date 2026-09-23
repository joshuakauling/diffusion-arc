"""CPU-only checks of continuation setup, fresh views and local/remote dispatch."""

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.data import ArcTrainingDataset
from diffusion_arc.training.run.checkpointing import checkpoint_identity
from scripts.cloud import continue_runpod as continuation, launch_runpod, train_runpod
from scripts.data.prepare_training_data import prepare_training_data
from tests.arc_data.test_nvarc_benchmark import DERIVED_GROUP, _fixture


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.source = self.workspace / "source"
        _fixture(self.source, task_count=26)
        self.parent_run = self.workspace / "runs/first"
        self.next_run = self.workspace / "runs/second"
        payload = json.loads((continuation.ROOT / "configs/training/arc_agi_1_epoch.json").read_text())
        payload["data"].update(
            expected_task_count=26, source_archive=str(self.source / "nvarc-all-classified.tar.gz"),
            task_index=str(self.source / f"{DERIVED_GROUP}_tasks.jsonl"),
            official_training_keys=str(self.source / "arc_agi_1_training_keys.json"),
            prepared_split_directory=str(self.workspace / "first-data/splits/train"))
        payload["loop"].update(max_optimizer_steps=2, expected_target_count=78, expected_final_batch_target_count=30)
        payload["scheduler"].update(warmup_steps=2, first_update_warmup_fraction=.5)
        payload["deployment"]["output_directory"] = str(self.parent_run)
        payload["logging"].update(project="test-project", entity="test-owner")
        self.parent_path = self.parent_run / "training.json"
        train_runpod.write_json(self.parent_path, payload)
        self.parent = load_training_config(self.parent_path, expected_type="training_run")
        self.original_manifest = prepare_training_data(self.parent, config_path=self.parent_path)
        self.checkpoint_path = self.save_checkpoint(self.parent, epoch=1, step=2)
        train_runpod.write_json(self.parent_run / "startup.json", {"output_repo": "owner/weights"})
        self.wandb_run = SimpleNamespace(state="finished", lastHistoryStep=2)
        self.wandb_api = SimpleNamespace(viewer=True, run=Mock(return_value=self.wandb_run))

    def save_checkpoint(self, config, *, epoch, step):
        directory = config.recipe["deployment"]["output_directory"] / "checkpoints/same-run" / f"step-{step:06d}"
        (directory / "transformer").mkdir(parents=True)
        (directory / "transformer/config.json").write_text("{}")
        (directory / "transformer/model.safetensors").write_bytes(b"weights")
        (directory / "training_state.pt").write_bytes(b"optimizer state")
        targets = (epoch - 1) * 78 + min((step - (epoch - 1) * 2) * 48, 78)
        train_runpod.write_json(directory / "checkpoint.json", {
            "schema_version": 1,
            "recipe": config.recipe_dict(),
            "files": {str(path.relative_to(directory)): path.stat().st_size for path in directory.rglob("*") if path.is_file()},
            "identity": checkpoint_identity(config, data_fingerprint=(
                ArcTrainingDataset(config.data.split_directory, config.data.view_name, tokenizer=None).fingerprint
                if config.data.split_directory.exists() else "fixture"), validation_manifest={}),
            "progress": {"epoch": epoch, "optimizer_step": step, "sources_seen": targets // 3,
                         "targets_seen": targets, "workload_totals": {}},
            "wandb_run": {"id": "same-run", "project": "test-project", "entity": "test-owner"},
        })
        return directory

    def next_config(self, *, seed=None, resume=None):
        parent, checkpoint = continuation.load_parent(self.parent_run)
        return continuation.continuation_config(parent, checkpoint, self.next_run, seed=seed, resume=resume)

    def test_derives_new_view_without_changing_optimizer_schedule_models_or_validation(self):
        config = self.next_config()
        self.assertEqual(config.seed, self.parent.seed + 1)
        self.assertEqual(config.data.split_directory, self.next_run / "data/splits/train")
        for field in ("models", "optimizer", "scheduler", "loop"):
            self.assertEqual(getattr(config, field), getattr(self.parent, field))
        self.assertEqual(config.recipe["validation"], self.parent.recipe["validation"])
        self.assertEqual(config.recipe["logging"]["entity"], "test-owner")
        self.assertFalse(self.next_run.exists())
        for seed in (self.parent.seed, -1, 2**64):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.next_config(seed=seed)

    def test_preparation_changes_keys_preserves_source_tasks_and_reuses_prepared_view(self):
        original = (self.parent.data.split_directory / "manifest.json").read_bytes()
        config = self.next_config()
        train_runpod.write_json(config.path, config.recipe_dict())
        continuation.prepare_next_view(self.parent, config)
        manifest_path = config.data.split_directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.assertNotEqual(manifest["keys_sha256"], self.original_manifest["keys_sha256"])
        self.assertEqual(manifest["inputs_sha256"], self.original_manifest["inputs_sha256"])
        before = manifest_path.read_bytes()
        continuation.prepare_next_view(self.parent, config)
        self.assertEqual(before, manifest_path.read_bytes())
        self.assertEqual(original, (self.parent.data.split_directory / "manifest.json").read_bytes())
        config.recipe["data"]["task_index"].write_text("changed")
        with self.assertRaisesRegex(ValueError, "source pool changed"):
            continuation.prepare_next_view(self.parent, config)

    def test_resume_uses_saved_seed_and_local_step_but_retains_global_checkpoint_count(self):
        config = self.next_config(seed=999)
        train_runpod.write_json(config.path, config.recipe_dict())
        path = self.save_checkpoint(config, epoch=2, step=3)
        resumed = self.next_config(resume=path)
        self.assertEqual(resumed.seed, 999)
        self.assertEqual(train_runpod.validate_start(resumed, self.next_run, path), 1)
        with self.assertRaisesRegex(ValueError, "Existing continuation config differs"):
            self.next_config(seed=1000)
        final = self.save_checkpoint(config, epoch=2, step=4)
        with self.assertRaisesRegex(ValueError, "Epoch is complete"):
            train_runpod.validate_start(resumed, self.next_run, final)

    def test_parent_must_be_complete_and_include_optimizer_file(self):
        metadata_path = self.checkpoint_path / "checkpoint.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["progress"].update(optimizer_step=1, targets_seen=48, sources_seen=16)
        train_runpod.write_json(metadata_path, metadata)
        with self.assertRaisesRegex(ValueError, "Finish the previous epoch"):
            continuation.load_parent(self.parent_run)
        (self.checkpoint_path / "training_state.pt").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete checkpoint inventory"):
            continuation.load_parent(self.parent_run)

    def test_pod_entrypoint_prepares_or_trains_and_preserves_wandb_and_upload_destination(self):
        for prepare_only in (True, False):
            with self.subTest(prepare_only=prepare_only), ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {"HF_TOKEN": "fixture", "WANDB_API_KEY": "fixture"}))
                stack.enter_context(patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=Mock(return_value=self.wandb_api))}))
                stack.enter_context(patch("huggingface_hub.HfApi"))
                for name in ("prepare_runtime", "prepare_output_repository", "preflight_gpu_and_disk"):
                    stack.enter_context(patch.object(train_runpod, name))
                child = stack.enter_context(patch.object(continuation.subprocess, "run"))
                upload = stack.enter_context(patch.object(train_runpod, "upload_weights"))
                args = ["--from-run", "first", "--run-name", "second", "--workspace", str(self.workspace)]
                self.assertEqual(continuation.main(args + (["--prepare-only"] if prepare_only else [])), 0)
                receipt = json.loads((self.next_run / "startup.json").read_text())
                self.assertEqual(receipt["parent"]["checkpoint"], str(self.checkpoint_path))
                self.assertEqual(receipt["output_repo"], "owner/weights")
                if prepare_only:
                    child.assert_not_called()
                    upload.assert_not_called()
                else:
                    child.assert_called_once()
                    self.assertIn(str(self.parent_run), child.call_args.args[0])
                    self.assertEqual(upload.call_args.kwargs["output_repo"], "owner/weights")

    def test_disk_failure_happens_before_data_preparation_or_training(self):
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"HF_TOKEN": "fixture", "WANDB_API_KEY": "fixture"}))
            stack.enter_context(patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=Mock(return_value=self.wandb_api))}))
            stack.enter_context(patch("huggingface_hub.HfApi"))
            stack.enter_context(patch.object(train_runpod, "prepare_runtime"))
            stack.enter_context(patch.object(train_runpod, "prepare_output_repository"))
            stack.enter_context(patch.object(train_runpod, "preflight_gpu_and_disk", side_effect=ValueError("disk space")))
            prepare = stack.enter_context(patch.object(continuation, "prepare_next_view"))
            child = stack.enter_context(patch.object(continuation.subprocess, "run"))
            with self.assertRaisesRegex(ValueError, "disk space"):
                continuation.main(["--from-run", "first", "--run-name", "second", "--workspace", str(self.workspace)])
            prepare.assert_not_called()
            child.assert_not_called()
            self.assertFalse(self.next_run.exists())

    def test_ddp_epoch_continuation_keeps_parent_state_and_launches_all_ranks(self):
        payload = self.parent.recipe_dict()
        payload["execution"].update(distributed_strategy="ddp", flux_packing_algorithm="balanced_release_v1")
        payload["flow"]["rng_scheme"] = "global_step_target_v1"
        payload["deployment"]["gpu_count"] = 2
        train_runpod.write_json(self.parent_path, payload)
        parent = load_training_config(self.parent_path)
        metadata_path = self.checkpoint_path / "checkpoint.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(schema_version=2, execution={"strategy": "ddp", "world_size": 2},
                        recipe=parent.recipe_dict(), identity=checkpoint_identity(
                            parent, data_fingerprint=metadata["identity"]["data_sha256"], validation_manifest={}))
        train_runpod.write_json(metadata_path, metadata)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"HF_TOKEN": "fixture", "WANDB_API_KEY": "fixture"}))
            stack.enter_context(patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=Mock(return_value=self.wandb_api))}))
            stack.enter_context(patch("huggingface_hub.HfApi"))
            for name in ("prepare_runtime", "prepare_output_repository", "preflight_gpu_and_disk"):
                stack.enter_context(patch.object(train_runpod, name))
            child = stack.enter_context(patch.object(continuation.subprocess, "run"))
            upload = stack.enter_context(patch.object(train_runpod, "upload_weights"))
            self.assertEqual(continuation.main([
                "--from-run", "first", "--run-name", "second", "--workspace", str(self.workspace)]), 0)
        command = child.call_args.args[0]
        self.assertIn("torch.distributed.run", command)
        self.assertIn("--nproc-per-node=2", command)
        self.assertEqual(command[-1], str(self.parent_run))
        resolved = load_training_config(self.next_run / "training.json")
        self.assertEqual(resolved.recipe["execution"], parent.recipe["execution"])
        upload.assert_called_once()

    def test_active_or_outdated_wandb_parent_is_rejected_without_resuming_or_relabeling_it(self):
        _, checkpoint = continuation.load_parent(self.parent_run)
        with patch.dict("sys.modules", {"wandb": SimpleNamespace(init=Mock())}) as modules:
            continuation.check_wandb_run(self.wandb_api, checkpoint, last_allowed_step=2)
            self.wandb_api.run.assert_called_with("test-owner/test-project/same-run")
            self.wandb_run.state = "running"
            with self.assertRaisesRegex(ValueError, "still active"):
                continuation.check_wandb_run(self.wandb_api, checkpoint, last_allowed_step=2)
            self.wandb_run.state = "finished"
            self.wandb_run.lastHistoryStep = 4
            with self.assertRaisesRegex(ValueError, "later steps"):
                continuation.check_wandb_run(self.wandb_api, checkpoint, last_allowed_step=2)
            modules["wandb"].init.assert_not_called()
            self.assertEqual(self.wandb_run.state, "finished")
            # Replaying this epoch after a partial checkpoint remains allowed.
            continuation.check_wandb_run(self.wandb_api, checkpoint, last_allowed_step=4)

    def test_local_entrypoint_dry_run_uploads_source_and_selects_continuation_arguments(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(continuation.main([
                "--ssh", "ssh example.invalid", "--from-run", "first", "--run-name", "second",
                "--augmentation-seed", "0", "--dry-run"]), 0)
        arguments = json.loads(output.getvalue())["arguments"]
        self.assertEqual(arguments[0], "continue")
        self.assertEqual(arguments[arguments.index("--augmentation-seed") + 1], "0")
        self.assertIn("--from-run", arguments)

    def test_worker_dispatches_new_entrypoint_and_propagates_failure(self):
        source = self.workspace / "code"
        script = source / "scripts/cloud/continue_runpod.py"
        script.parent.mkdir(parents=True)
        script.write_text("raise SystemExit(7)\n")
        job = self.workspace / "job"
        train_runpod.write_json(job / "request.json", {
            "source_directory": str(source), "arguments": ["continue", "--from-run", "first", "--run-name", "second"]})
        self.assertEqual(launch_runpod.run_remote_job(str(job)), 7)
        self.assertEqual(json.loads((job / "status.json").read_text())["state"], "failed")


if __name__ == "__main__":
    unittest.main()
