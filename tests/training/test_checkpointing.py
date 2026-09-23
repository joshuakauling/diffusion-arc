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


class CheckpointRuntime:
    """Replay a two-rank rendezvous without replacing the real state serialization."""

    strategy = "ddp"
    world_size = 2
    device = torch.device("cpu")

    def __init__(self, rank, gathered, publication):
        self.rank = rank
        self.is_primary = rank == 0
        self.gathered = gathered
        self.publication = publication

    def gather(self, value):
        values = deepcopy(self.gathered)
        values[self.rank] = value
        return values

    def primary(self, callback):
        if self.is_primary:
            self.publication["result"] = callback()
        return self.publication["result"]


class CheckpointTests(unittest.TestCase):
    def test_new_optimizer_at_global_epoch_three_restores_warmup_and_exact_next_update(self):
        payload = self.config.recipe_dict()
        source = json.loads((ROOT / "configs/training/arc_agi_1_epoch3_high_noise.json").read_text())
        payload["models"].update({key: source["models"][key] for key in (
            "initialization", "completed_epochs", "transformer_source")})
        payload["models"]["transformer_checkpoint"] = str(self.directory / "initial")
        payload["logging"].update(parent_run_id="existing-run", resume_parent_run=True)
        self.config.path.write_text(json.dumps(payload))
        self.config = load_training_config(self.config.path)
        self.identity = checkpointing.checkpoint_identity(self.config, data_fingerprint="a" * 64,
                                                          validation_manifest={})
        components = self._components()
        self._advance(components, 2)
        progress = RunProgress(optimizer_step=10, sources_seen=148, targets_seen=444,
                               epoch=3, optimizer_start_step=8)
        model, optimizer, scheduler, generator = components
        metrics = checkpointing.save_checkpoint(
            progress, config=self.config, transformer=model, optimizer=optimizer, scheduler=scheduler,
            flow_generator=generator, device=self.device, identity=self.identity, wandb_run=self.wandb_run)
        path = Path(metrics["checkpoint/path"])
        expected = self._advance(components)
        restored = self._components(TinyTransformer.from_pretrained(path / "transformer", local_files_only=True))
        self._restore(path, restored)
        self.assertEqual(restored[2].last_epoch, 2)
        actual = self._advance(restored)
        self.assertNestedEqual(expected, actual)
        self.assertNestedEqual(components[0].state_dict(), restored[0].state_dict())
        self._rewrite_metadata(path, lambda m: m["progress"].update(optimizer_start_step=0))
        with self.assertRaisesRegex(ValueError, "optimizer start"):
            checkpointing.read_checkpoint(path, self.config)

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

    def _save(self, components, *, config=None, **kwargs):
        model, optimizer, scheduler, generator = components
        metrics = checkpointing.save_checkpoint(
            self._progress(scheduler.last_epoch), config=self.config if config is None else config,
            transformer=model, optimizer=optimizer, scheduler=scheduler,
            flow_generator=generator, device=self.device, identity=self.identity,
            wandb_run=self.wandb_run, **kwargs,
        )
        return Path(metrics["checkpoint/path"]), metrics

    def _restore(self, path, components, **kwargs):
        model, optimizer, scheduler, generator = components
        checkpoint = checkpointing.read_checkpoint(path, self.config)
        checkpointing.restore_training_state(
            checkpoint, transformer=model, optimizer=optimizer, scheduler=scheduler,
            flow_generator=generator, device=self.device, **kwargs,
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

    def test_second_epoch_roundtrip_preserves_adamw_rng_and_constant_lr_without_new_warmup(self):
        components = self._components()
        self._advance(components, 4)
        parent, _ = self._save(components)
        checkpoint = checkpointing.read_checkpoint(parent, self.config)
        loaded = self._components(TinyTransformer.from_pretrained(parent / "transformer", local_files_only=True))
        self._restore(parent, loaded)
        self.assertEqual(loaded[1].param_groups[0]["lr"], .001)
        # First step of epoch two is global step five, despite epoch one's partial final batch.
        self._advance(loaded)
        progress = RunProgress(5, 74, 222, {}, epoch=2)
        metrics = checkpointing.save_checkpoint(
            progress, config=self.config, transformer=loaded[0], optimizer=loaded[1], scheduler=loaded[2],
            flow_generator=loaded[3], device=self.device, identity=self.identity, wandb_run=self.wandb_run)
        path = Path(metrics["checkpoint/path"])
        saved = checkpointing.read_checkpoint(path, self.config)
        self.assertEqual(saved.metadata["progress"]["targets_seen"], 174 + 48)
        self.assertEqual(saved.metadata["progress"]["epoch"], 2)
        expected = self._advance(loaded, 3)
        resumed = self._components(TinyTransformer.from_pretrained(path / "transformer", local_files_only=True))
        self._restore(path, resumed)
        self.assertNestedEqual(self._advance(resumed, 3), expected)
        self.assertNestedEqual(resumed[1].state_dict(), loaded[1].state_dict())
        self.assertEqual(resumed[2].last_epoch, 8)
        # Old checkpoints without an epoch field still load as epoch one.
        self._rewrite_metadata(parent, lambda item: item["progress"].pop("epoch"))
        checkpointing.read_checkpoint(parent, self.config)

    def test_second_epoch_cursor_rejects_full_batch_arithmetic_across_partial_epoch_boundary(self):
        components = self._components()
        self._advance(components)
        path, _ = self._save(components)
        for update in ({"epoch": 0}, {"epoch": True}, {"epoch": 2},
                       {"epoch": 2, "optimizer_step": 5, "targets_seen": 240, "sources_seen": 80}):
            with self.subTest(update=update):
                self._rewrite_metadata(path, lambda item: item.update(progress={
                    "optimizer_step": 1, "targets_seen": 48, "sources_seen": 16,
                    "workload_totals": {}, **update}))
                with self.assertRaises(ValueError):
                    checkpointing.read_checkpoint(path, self.config)

    def test_removing_training_state_keeps_loadable_weights_and_requires_explicit_weights_only_reads(self):
        components = self._components()
        self._advance(components)
        path, _ = self._save(components)
        checkpoint = checkpointing.read_checkpoint(path, self.config)
        before_weights = {file.name: file.read_bytes() for file in (path / "transformer").iterdir()}
        expected_bytes = (path / "training_state.pt").stat().st_size
        self.assertEqual(checkpointing.remove_training_state(checkpoint), expected_bytes)
        self.assertFalse((path / "training_state.pt").exists())
        self.assertEqual(before_weights, {file.name: file.read_bytes() for file in (path / "transformer").iterdir()})
        retained = checkpointing.read_checkpoint(path, self.config, require_training_state=False)
        self.assertTrue(retained.metadata["weights_only"])
        self.assertNotIn("training_state.pt", retained.metadata["files"])
        self.assertEqual(checkpointing.remove_training_state(retained), 0)
        with self.assertRaisesRegex(ValueError, "training state was removed"):
            checkpointing.read_checkpoint(path, self.config)
        restored_model = TinyTransformer.from_pretrained(path / "transformer", local_files_only=True)
        self.assertNestedEqual(restored_model.state_dict(), components[0].state_dict())

    def test_failed_pruning_metadata_publication_restores_full_checkpoint(self):
        components = self._components()
        self._advance(components)
        path, _ = self._save(components)
        checkpoint = checkpointing.read_checkpoint(path, self.config)
        before = {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()}
        with patch.object(Path, "replace", side_effect=OSError("publication failed")):
            with self.assertRaisesRegex(OSError, "publication failed"):
                checkpointing.remove_training_state(checkpoint)
        self.assertEqual(before, {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()})
        self.assertEqual(list(path.parent.iterdir()), [path])
        checkpointing.read_checkpoint(path, self.config)

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
        with self.assertRaises(FileExistsError):
            self._save(components)
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

    def _use_ddp_config(self):
        recipe = self.config.recipe_dict()
        recipe["deployment"]["output_directory"] = self.config.recipe["deployment"]["output_directory"]
        recipe["deployment"]["gpu_count"] = 2
        recipe["execution"].update(distributed_strategy="ddp", flux_packing_algorithm="balanced_release_v1")
        recipe["flow"]["rng_scheme"] = "global_step_target_v1"
        self.config = replace(self.config, recipe=recipe)
        self.identity = checkpointing.checkpoint_identity(
            self.config, data_fingerprint="a" * 64,
            validation_manifest={"validation_task_ids": ["held-out-task"]})

    def test_legacy_schema_expands_only_equivalent_rng_and_packing_defaults(self):
        components = self._components()
        self._advance(components)
        path, _ = self._save(components)

        def old_schema(metadata):
            metadata["schema_version"] = 1
            metadata.pop("execution")
            for recipe in (metadata["recipe"], metadata["identity"]["recipe"]):
                recipe["flow"].pop("rng_scheme", None)
                recipe["execution"].pop("flux_packing_algorithm", None)
                recipe["execution"].pop("flux_max_spread_tokens", None)

        self._rewrite_metadata(path, old_schema)
        checkpoint = checkpointing.read_checkpoint(path, self.config)
        self.assertTrue(checkpointing.checkpoint_identities_match(checkpoint.metadata["identity"], self.identity))
        for section, field, value in (("flow", "rng_scheme", "global_step_target_v1"),
                                      ("execution", "flux_packing_algorithm", "balanced_release_v1"),
                                      ("execution", "flux_max_spread_tokens", 250)):
            with self.subTest(field=field):
                recipe = self.config.recipe_dict()
                recipe[section][field] = value
                changed = replace(self.config, recipe=recipe)
                with self.assertRaisesRegex(ValueError, "recipe differs"):
                    checkpointing.read_checkpoint(path, changed)

    def test_ddp_roundtrip_keeps_each_rank_rng_and_existing_gradient_buffers(self):
        self._use_ddp_config()
        components = self._components()
        self._advance(components, 2)
        rng0 = checkpointing._capture_rng(components[3], self.device)
        expected = [self._advance(components)]
        rng1 = checkpointing._capture_rng(components[3], self.device)
        expected.append(self._advance(components))

        # Replicated weights/optimizer are saved once; rank-specific streams differ.
        components = self._components()
        self._advance(components, 2)
        gathered = [{"error": None, "rng": rng} for rng in (rng0, rng1)]
        publication = {}
        buffers = []
        for parameter in components[0].parameters():
            parameter.grad = torch.zeros_like(parameter)
            buffers.append(parameter.grad)
        runtime0 = CheckpointRuntime(0, gathered, publication)
        path, metrics = self._save(components, runtime=runtime0, completed_update=True)
        self.assertNestedEqual(checkpointing._capture_rng(components[3], self.device), rng0)
        checkpointing._restore_rng(rng1, components[3], self.device)
        runtime1 = CheckpointRuntime(1, gathered, publication)
        with patch.object(checkpointing.torch, "save") as save:
            rank1_path, rank1_metrics = self._save(components, runtime=runtime1, completed_update=True)
        save.assert_not_called()
        self.assertEqual(rank1_path, path)
        self.assertEqual(rank1_metrics, metrics)
        self.assertNestedEqual(checkpointing._capture_rng(components[3], self.device), rng1)
        payload = torch.load(path / "training_state.pt", weights_only=True)
        self.assertNotIn("rng", payload)
        self.assertNestedEqual(payload["rng_by_rank"], [rng0, rng1])
        self.assertEqual(json.loads((path / "checkpoint.json").read_text())["execution"],
                         {"strategy": "ddp", "world_size": 2})

        for rank, runtime in enumerate((runtime0, runtime1)):
            with self.subTest(rank=rank):
                loaded = self._components(TinyTransformer.from_pretrained(path / "transformer", local_files_only=True))
                original_buffers = []
                for parameter in loaded[0].parameters():
                    parameter.grad = torch.ones_like(parameter)
                    original_buffers.append(parameter.grad)
                self._restore(path, loaded, runtime=runtime)
                self.assertNestedEqual(loaded[1].state_dict(), payload["optimizer"])
                for parameter, buffer in zip(loaded[0].parameters(), original_buffers, strict=True):
                    self.assertIs(parameter.grad, buffer)
                    self.assertEqual(torch.count_nonzero(buffer).item(), 0)
                draws = self._advance(loaded)[0]
                if rank == 0:
                    self.assertNestedEqual(draws, expected[0][0])
                # Rank one's expected weights differ here; compare its four independent RNG streams.
                for key in ("python", "numpy", "inputs", "flow_noise"):
                    self.assertNestedEqual(draws[key], expected[rank][0][key])

    def test_ddp_rejects_unconfirmed_update_and_other_rank_boundary_failure(self):
        self._use_ddp_config()
        components = self._components()
        self._advance(components)
        rng = checkpointing._capture_rng(components[3], self.device)
        gathered = [{"error": None, "rng": rng}, {"error": None, "rng": rng}]
        runtime = CheckpointRuntime(0, gathered, {})
        with self.assertRaisesRegex(ValueError, "explicitly completed"):
            self._save(components, runtime=runtime)
        gathered[1]["error"] = "checkpoint scheduler is not at the completed optimizer step"
        with self.assertRaisesRegex(ValueError, "rank 1.*scheduler"):
            self._save(components, runtime=runtime, completed_update=True)
        self.assertFalse((self.directory / "output/checkpoints").exists())

    def test_ddp_rejects_changed_world_size_strategy_and_incomplete_rank_rng(self):
        self._use_ddp_config()
        components = self._components()
        self._advance(components)
        rng = checkpointing._capture_rng(components[3], self.device)
        gathered = [{"error": None, "rng": rng}, {"error": None, "rng": rng}]
        runtime = CheckpointRuntime(0, gathered, {})
        path, _ = self._save(components, runtime=runtime, completed_update=True)
        for strategy, world_size in (("single_process", 1), ("ddp", 1), ("ddp", 4)):
            with self.subTest(strategy=strategy, world_size=world_size):
                recipe = self.config.recipe_dict()
                recipe["deployment"]["gpu_count"] = world_size
                recipe["execution"]["distributed_strategy"] = strategy
                with self.assertRaisesRegex(ValueError, "strategy or world size"):
                    checkpointing.read_checkpoint(path, replace(self.config, recipe=recipe))
        with self.assertRaisesRegex(ValueError, "strategy or world size"):
            self._restore(path, components)
        state_file = path / "training_state.pt"
        state = torch.load(state_file, weights_only=True)
        for faulty_rng in ([rng], [rng, {}]):
            state["rng_by_rank"] = faulty_rng
            torch.save(state, state_file)
            self._rewrite_metadata(path, lambda metadata: metadata["files"].update(
                {"training_state.pt": state_file.stat().st_size}))
            with self.assertRaisesRegex(ValueError, "RNG state.*incomplete"):
                self._restore(path, components, runtime=runtime)


if __name__ == "__main__":
    unittest.main()
