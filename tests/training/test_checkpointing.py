"""CPU checkpoint roundtrips with real HF safetensors and optimizer continuation."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
import numpy as np
import torch

from diffusion_arc.training.run import checkpointing
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.step.lr_schedule import LinearWarmupConstantLR
from diffusion_arc.training.runner import RunProgress


ROOT = Path(__file__).resolve().parents[2]


class TinyTransformer(ModelMixin, ConfigMixin):
    """Exercise Diffusers' actual model serialization without downloading weights."""

    @register_to_config
    def __init__(self, input_width=3, output_width=2):
        super().__init__()
        self.projection = torch.nn.Linear(input_width, output_width)

    def forward(self, values):
        return self.projection(values)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        payload = json.loads((ROOT / "configs/training/arc_agi_1_epoch.json").read_text())
        payload["deployment"]["output_directory"] = str(self.directory / "output")
        payload["data"]["expected_task_count"] = 58
        payload["loop"].update(max_optimizer_steps=4, expected_target_count=174,
                               expected_final_batch_target_count=30)
        payload["scheduler"].update(warmup_steps=3, first_update_warmup_fraction=1 / 3)
        payload["optimizer"]["learning_rate"] = .001
        path = self.directory / "recipe.json"
        path.write_text(json.dumps(payload))
        self.config = load_training_config(path, expected_type="training_run")
        self.identity = checkpointing.checkpoint_identity(
            self.config, data_fingerprint="a" * 64,
            validation_manifest={"validation_task_ids": ["held-out-task"]},
        )
        self.wandb_run = {"id": "existing-run", "project": "research", "entity": "local-test"}
        self.device = torch.device("cpu")

    def assertNestedEqual(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                self.assertNestedEqual(left[name], right[name])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right, strict=True):
                self.assertNestedEqual(first, second)
        else:
            self.assertEqual(left, right)

    def _components(self, model=None):
        random.seed(15)
        np.random.seed(25)
        torch.manual_seed(35)
        model = TinyTransformer() if model is None else model
        settings = self.config.optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings.learning_rate,
            betas=(settings.beta1, settings.beta2), eps=settings.epsilon,
            weight_decay=settings.weight_decay,
        )
        scheduler = LinearWarmupConstantLR(optimizer, warmup_steps=3)
        generator = torch.Generator(device="cpu").manual_seed(45)
        return model, optimizer, scheduler, generator

    def _advance(self, components, count=1):
        model, optimizer, scheduler, generator = components
        records = []
        for _ in range(count):
            python_draw, numpy_draw = random.random(), float(np.random.random())
            inputs = torch.randn(4, 3)
            flow_noise = torch.randn(4, 2, generator=generator)
            target = flow_noise + python_draw + numpy_draw
            loss = (model(inputs) - target).square().mean()
            loss.backward()
            records.append({
                "python": python_draw, "numpy": numpy_draw, "inputs": inputs,
                "flow_noise": flow_noise, "loss": loss.detach().clone(),
                "gradients": [parameter.grad.clone() for parameter in model.parameters()],
                "learning_rate": optimizer.param_groups[0]["lr"],
            })
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        return records

    def _progress(self, step):
        targets = min(step * 48, 174)
        return RunProgress(step, targets // 3, targets, {
            "train/flux_forward_calls": 6 * step, "sample/flux_forward_calls": 240,
        })

    def _save(self, components, *, config=None):
        model, optimizer, scheduler, generator = components
        metrics = checkpointing.save_checkpoint(
            self._progress(scheduler.last_epoch), config=self.config if config is None else config,
            transformer=model, optimizer=optimizer, scheduler=scheduler,
            flow_generator=generator, device=self.device, identity=self.identity,
            wandb_run=self.wandb_run,
        )
        return Path(metrics["checkpoint/path"]), metrics

    def _restore(self, path, components):
        model, optimizer, scheduler, generator = components
        checkpoint = checkpointing.read_checkpoint(path, self.config)
        checkpointing.restore_training_state(
            checkpoint, transformer=model, optimizer=optimizer, scheduler=scheduler,
            flow_generator=generator, device=self.device,
        )

    def _rewrite_metadata(self, path, change):
        file = path / "checkpoint.json"
        metadata = json.loads(file.read_text())
        change(metadata)
        file.write_text(json.dumps(metadata))

    def test_hf_checkpoint_and_resume_match_uninterrupted_rng_gradients_rates_and_adamw_state(self):
        baseline = self._components()
        expected_records = self._advance(baseline, 4)
        interrupted = self._components()
        first_records = self._advance(interrupted, 2)
        before_weights = deepcopy(interrupted[0].state_dict())
        before_optimizer = deepcopy(interrupted[1].state_dict())
        path, metrics = self._save(interrupted)

        # Saving itself leaves the original run's future updates unchanged.
        continuing_records = self._advance(interrupted, 2)
        self.assertNestedEqual(first_records + continuing_records, expected_records)
        self.assertNestedEqual(interrupted[0].state_dict(), baseline[0].state_dict())
        self.assertNestedEqual(interrupted[1].state_dict(), baseline[1].state_dict())

        loaded_model = TinyTransformer.from_pretrained(path / "transformer", local_files_only=True)
        self.assertNestedEqual(loaded_model.state_dict(), before_weights)
        resumed = self._components(loaded_model)
        # Model loading and frozen-cache preparation can consume process RNGs.
        self._advance(resumed)
        loaded_model.load_state_dict(before_weights)
        self._restore(path, resumed)
        self.assertNestedEqual(resumed[1].state_dict(), before_optimizer)
        self.assertTrue(all(parameter.grad is None for parameter in loaded_model.parameters()))
        self.assertNestedEqual(self._advance(resumed, 2), expected_records[2:])
        self.assertNestedEqual(loaded_model.state_dict(), baseline[0].state_dict())
        self.assertNestedEqual(resumed[1].state_dict(), baseline[1].state_dict())
        self.assertNestedEqual(resumed[2].state_dict(), baseline[2].state_dict())
        self.assertEqual([record["learning_rate"] for record in expected_records],
                         [.001 / 3, .001 * 2 / 3, .001, .001])

        files = {str(file.relative_to(path)) for file in path.rglob("*") if file.is_file()}
        self.assertEqual(files, {"checkpoint.json", "training_state.pt", "transformer/config.json",
                                 "transformer/diffusion_pytorch_model.safetensors"})
        metadata = json.loads((path / "checkpoint.json").read_text())
        self.assertEqual(metadata["wandb_run"], self.wandb_run)
        self.assertEqual(metadata["progress"]["optimizer_step"], 2)
        self.assertIn("nvarc_checkpoint", metadata["recipe"]["models"])
        self.assertEqual(metrics["checkpoint/size_bytes"],
                         sum(file.stat().st_size for file in path.rglob("*") if file.is_file()))
        self.assertGreaterEqual(metrics["checkpoint/seconds"], 0)

    def test_identity_binds_training_data_and_validation_but_allows_deployment_and_resume_changes(self):
        recipe = self.config.recipe_dict()
        recipe["deployment"]["output_directory"] = "/other-volume"
        recipe["logging"]["project"] = "another-project"
        recipe["checkpointing"]["keep_last"] = 1
        recipe["models"]["resume_from"] = "/a-checkpoint"
        recipe["name"] = "continued"
        continuation = replace(self.config, recipe=recipe)
        self.assertEqual(self.identity, checkpointing.checkpoint_identity(
            continuation, data_fingerprint="a" * 64,
            validation_manifest={"validation_task_ids": ["held-out-task"]},
        ))
        self.assertNotEqual(self.identity, checkpointing.checkpoint_identity(
            self.config, data_fingerprint="b" * 64,
            validation_manifest={"validation_task_ids": ["held-out-task"]},
        ))
        self.assertNotEqual(self.identity, checkpointing.checkpoint_identity(
            self.config, data_fingerprint="a" * 64,
            validation_manifest={"validation_task_ids": ["different-task"]},
        ))
        components = self._components()
        self._advance(components)
        path, _ = self._save(components)
        recipe["loss"]["region_weights"]["grid"] = .7
        with self.assertRaisesRegex(ValueError, "recipe differs"):
            checkpointing.read_checkpoint(path, continuation)

    def test_read_rejects_missing_truncated_or_unlisted_files_before_loading_tensors(self):
        components = self._components()
        self._advance(components)
        saved, _ = self._save(components)
        for fault in ("missing", "truncated", "unlisted", "unsupported_schema"):
            with self.subTest(fault=fault):
                path = self.directory / fault
                shutil.copytree(saved, path)
                state = path / "training_state.pt"
                if fault == "missing":
                    state.unlink()
                elif fault == "truncated":
                    state.write_bytes(state.read_bytes()[:40])
                elif fault == "unlisted":
                    (path / "unlisted.bin").write_bytes(b"untracked")
                else:
                    self._rewrite_metadata(path, lambda item: item.update(schema_version=100))
                with patch.object(checkpointing.torch, "load") as load, self.assertRaises(ValueError):
                    checkpointing.read_checkpoint(path, self.config)
                load.assert_not_called()

    def test_read_rejects_non_boundary_progress_and_negative_workload(self):
        components = self._components()
        self._advance(components)
        saved, _ = self._save(components)
        changes = ({"optimizer_step": 0}, {"optimizer_step": 5}, {"optimizer_step": 1.0},
                   {"targets_seen": 47}, {"sources_seen": 15},
                   {"workload_totals": {"train/flux_forward_calls": -1}})
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                path = self.directory / f"cursor-{index}"
                shutil.copytree(saved, path)
                self._rewrite_metadata(path, lambda item: item["progress"].update(change))
                with self.assertRaises(ValueError):
                    checkpointing.read_checkpoint(path, self.config)

    def test_restore_rejects_optimizer_parameter_order_scheduler_or_missing_moments(self):
        components = self._components()
        self._advance(components, 2)
        saved, _ = self._save(components)
        for fault in ("parameter_order", "scheduler_step", "scheduler_warmup", "moment_shape", "missing_moment"):
            with self.subTest(fault=fault):
                path = self.directory / fault
                shutil.copytree(saved, path)
                state_file = path / "training_state.pt"
                state = torch.load(state_file, weights_only=True)
                if fault == "parameter_order":
                    state["parameter_names"].reverse()
                elif fault == "scheduler_step":
                    state["scheduler"]["last_epoch"] = 1
                elif fault == "scheduler_warmup":
                    state["scheduler"]["warmup_steps"] = 99
                else:
                    first = next(iter(state["optimizer"]["state"].values()))
                    if fault == "moment_shape":
                        first["exp_avg"] = torch.zeros(1)
                    else:
                        del first["exp_avg_sq"]
                torch.save(state, state_file)
                self._rewrite_metadata(path, lambda item: item["files"].update(
                    {"training_state.pt": state_file.stat().st_size}))
                restored = self._components(TinyTransformer.from_pretrained(
                    path / "transformer", local_files_only=True))
                with self.assertRaises(ValueError):
                    self._restore(path, restored)

    def test_failed_write_is_atomic_cleans_temporary_and_preserves_rng_and_previous_checkpoint(self):
        components = self._components()
        self._advance(components, 2)
        old, _ = self._save(components)
        old_contents = {str(file.relative_to(old)): file.read_bytes()
                        for file in old.rglob("*") if file.is_file()}
        self._advance(components)
        rng = checkpointing._capture_rng(components[3], self.device)
        weights = deepcopy(components[0].state_dict())

        def fail_write(state, file):
            Path(file).write_bytes(b"partial optimizer state")
            random.random()
            np.random.random()
            torch.randn(4)
            torch.randn(4, generator=components[3])
            raise OSError("simulated full disk")

        with patch.object(checkpointing.torch, "save", side_effect=fail_write):
            with self.assertRaisesRegex(OSError, "full disk"):
                self._save(components)
        self.assertEqual(set(old.parent.iterdir()), {old})
        self.assertEqual(old_contents, {str(file.relative_to(old)): file.read_bytes()
                                       for file in old.rglob("*") if file.is_file()})
        self.assertNestedEqual(checkpointing._capture_rng(components[3], self.device), rng)
        self.assertNestedEqual(components[0].state_dict(), weights)
        with self.assertRaises(FileExistsError):
            checkpointing.save_checkpoint(
                self._progress(2), config=self.config, transformer=components[0], optimizer=components[1],
                scheduler=components[2], flow_generator=components[3], device=self.device,
                identity=self.identity, wandb_run=self.wandb_run,
            )

    def test_retention_keeps_all_by_default_and_only_removes_own_completed_checkpoints(self):
        components = self._components()
        self._advance(components)
        first, _ = self._save(components)
        self._advance(components)
        second, _ = self._save(components)
        self.assertTrue(first.is_dir())
        self.assertTrue(second.is_dir())

        foreign = first.parent / "step-000000"
        shutil.copytree(first, foreign)
        self._rewrite_metadata(foreign, lambda item: item["wandb_run"].update(id="foreign-run"))
        unfinished = first.parent / "step-000005"
        unfinished.mkdir()
        (unfinished / "partial.bin").write_bytes(b"unfinished")
        recipe = {**self.config.recipe, "checkpointing": {**self.config.recipe["checkpointing"], "keep_last": 1}}
        retained_config = replace(self.config, recipe=recipe)
        self._advance(components)
        third, _ = self._save(components, config=retained_config)
        self.assertEqual(set(first.parent.iterdir()), {foreign, unfinished, third})
        self.assertEqual((unfinished / "partial.bin").read_bytes(), b"unfinished")
        self.assertEqual(json.loads((foreign / "checkpoint.json").read_text())["wandb_run"]["id"], "foreign-run")

    def test_save_requires_cleared_gradients_and_completed_scheduler(self):
        components = self._components()
        self._advance(components)
        next(components[0].parameters()).grad = torch.ones_like(next(components[0].parameters()))
        with self.assertRaisesRegex(ValueError, "cleared gradients"):
            self._save(components)
        components[1].zero_grad(set_to_none=True)
        with self.assertRaisesRegex(ValueError, "scheduler"):
            checkpointing.save_checkpoint(
                self._progress(2), config=self.config, transformer=components[0], optimizer=components[1],
                scheduler=components[2], flow_generator=components[3], device=self.device,
                identity=self.identity, wandb_run=self.wandb_run,
            )


if __name__ == "__main__":
    unittest.main()
