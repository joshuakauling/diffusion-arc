"""Full native/DDP lifecycle with real training, HF checkpoints and continuation."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusion_arc.training import runner
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.distributed import DistributedRuntime
from diffusion_arc.training.distributed_data import plan_distributed_batches
from diffusion_arc.training.run.checkpointing import read_checkpoint
from diffusion_arc.training.trainer import Trainer
from tests.model.test_flux_packing import SegmentedReference
from tests.training.test_ddp_runner import Sources
from tests.training.test_ddp_trainer import FrozenText, FrozenVae, _model, _render, GRAD_TOLERANCE, PARAM_TOLERANCE


ROOT = Path(__file__).resolve().parents[2]


def _config(directory, *, strategy="single_process", epoch=3, resume_from=None, output="complete"):
    base = load_training_config(ROOT / "configs/training/arc_agi_1_epoch3_high_noise.json")
    payload = base.recipe_dict()
    payload["seed"] = 169 + epoch - 3
    payload["deployment"].update(gpu_count=2 if strategy == "ddp" else 1,
                                  output_directory=str(directory / strategy / output))
    payload["models"].update(transformer_checkpoint=str(directory / "selected-transformer"),
                              resume_from=str(resume_from) if resume_from else None)
    payload["execution"].update(distributed_strategy=strategy, flux_packing_algorithm="balanced_release_v1",
                                 flux_max_joint_tokens=9300)
    payload["flow"].update(rng_scheme="global_step_target_v1", endpoint_targets_per_full_batch=6,
                           endpoint_targets_in_final_batch=4, high_noise_targets_per_full_batch=3,
                           high_noise_targets_in_final_batch=3)
    payload["data"].update(expected_task_count=7, num_workers=0, prefetch_factor=None,
                           prepared_split_directory=str(directory / f"epoch-{epoch}"))
    payload["loop"].update(optimizer_batch_size=12, max_optimizer_steps=2,
                           expected_target_count=21, expected_final_batch_target_count=9)
    payload["optimizer"].update(learning_rate=1e-4, max_grad_norm=0.1)
    payload["scheduler"].update(warmup_steps=2, first_update_warmup_fraction=0.5)
    payload["validation"].update(expected_task_count=1, expected_query_count=1)
    payload["validation"]["sampling"].update(expected_task_count=1, expected_query_count=1)
    for cadence in (payload["validation"]["loss"], payload["validation"]["sampling"], payload["checkpointing"]):
        cadence.update(every_optimizer_steps=1, replace_last_periodic_with_final=False)
    payload["logging"].update(resume_parent_run=False)
    # Every rank validates the same recipe without concurrently writing a file.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=directory) as handle:
        json.dump(payload, handle)
        handle.flush()
        return load_training_config(handle.name, expected_type="training_run")


def _execute(config, runtime, *, previous_epoch=None, stop_after=None):
    """Replace external services while preserving the complete runner lifecycle."""
    snapshots, records, events, loaded = [], [], [], []
    dataset = Sources(source_count=7)
    dataset.fingerprint = f"fixture-source-view-{config.seed}"
    target = SimpleNamespace(task_id="validation-task", query_index=0, expected_grid=((1,),))
    manifest = dict(validation_task_ids=[target.task_id], validation_output_count=1,
                    sampling_task_ids=[target.task_id], sampling_output_count=1)
    models = None

    def primary_event(name):
        assert runtime.is_primary, f"{name} executed on rank {runtime.rank}"
        events.append(name)

    class Logger:
        run_identity = {"id": "unified-local", "project": "tests", "entity": "local"}

        def __init__(self, *args, **kwargs):
            primary_event("logger")
            self.run = SimpleNamespace(config=SimpleNamespace(update=lambda *a, **kw: None))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def begin_step(self):
            primary_event("begin")

        def collect_hardware_metrics(self, **kwargs):
            primary_event("hardware")
            return {}

        def build_prediction_table(self, *args, **kwargs):
            primary_event("table")
            return "fixture-predictions"

        def log(self, metrics):
            primary_event("log")
            records.append(deepcopy(metrics))
            if metrics.get("checkpoint/path") and metrics["progress/optimizer_step"] == stop_after:
                raise RuntimeError("intentional interruption after completed checkpoint")

    def load_models(config, *, device, transformer_checkpoint):
        nonlocal models
        loaded.append(Path(transformer_checkpoint))
        model = type(_model()).from_pretrained(transformer_checkpoint, local_files_only=True)
        save = model.save_pretrained

        def save_pretrained(*args, **kwargs):
            primary_event("save")
            return save(*args, **kwargs)

        model.save_pretrained = save_pretrained
        models = SimpleNamespace(tokenizer=object(), flux_transformer=model,
            nvarc_encoder=FrozenText(), vae_encoder=FrozenVae(), vae_decoder=object(),
            flux_compute_dtype=torch.float32, flux_packing_backend=SegmentedReference())
        return models

    def prepare(*args, **kwargs):
        primary_event("prepare")
        return (target,)

    def evaluate(*args, **kwargs):
        primary_event("validate")
        # Resume must restore the saved RNG after validation-cache preparation.
        torch.rand(1)
        return dict(target_count=1, boundary_target_count=1, uniform_inner_loss=1.0, endpoint_loss=2.0,
            per_timestep=[dict(timestep=t, weighted_loss=1.0, canvas_loss=1.0, grid_loss=1.0,
                               boundary_loss=1.0) for t in kwargs["timesteps"]],
            nvarc_tokens=0, flux_text_tokens=12, flux_image_tokens=5400,
            flux_joint_tokens=5412, flux_forward_calls=6)

    def generate(*args, **kwargs):
        primary_event("sample")
        return dict(task_count=1, prediction_count=2, shape_correct_count=2, pass_at_1=1.0,
            pass_at_2=1.0, solved_at_1=1, solved_at_2=1, shape_accuracy=1.0, cell_accuracy=1.0,
            invalid_grid_rate=0.0, nvarc_tokens=0, flux_text_tokens=4, flux_image_tokens=1800,
            flux_joint_tokens=1804, flux_forward_calls=2)

    class RecordingTrainer(Trainer):
        def step(self, sequences):
            result = super().step(sequences)
            snapshots.append(dict(parameters=deepcopy(models.flux_transformer.state_dict()),
                optimizer=deepcopy(self.optimizer.state_dict()), scheduler=deepcopy(self.scheduler.state_dict()),
                times=tuple(target.timestep for target in result.targets), loss=result.loss,
                target_count=result.target_count, high_noise_quota=self.flow_sampler.high_noise_count,
                rng=torch.random.get_rng_state().clone()))
            return result

    with ExitStack() as stack:
        replacements = dict(load_models=load_models, WandbLogger=Logger, Trainer=RecordingTrainer,
            validate_validation_bundle=lambda *a, **kw: manifest,
            tokenize_validation_targets=lambda *a, **kw: ((SimpleNamespace(input_ids=(1, 2)),), (((1,),),)),
            prepare_validation_cache=prepare, plan_validation_packs=lambda *a, **kw: ((0,),),
            load_sampling_input_grids=lambda *a, **kw: {(target.task_id, 0): ((1,),)},
            evaluate_packed_loss=evaluate, generate_packed_samples=generate)
        for name, value in replacements.items():
            stack.enter_context(patch.object(runner, name, value))
        stack.enter_context(patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=dataset))
        stack.enter_context(patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render))
        try:
            progress = runner.run_training(config, runtime=runtime, previous_epoch=previous_epoch)
        except RuntimeError as error:
            if stop_after is None or "intentional interruption" not in str(error):
                raise
            progress = None
        else:
            assert stop_after is None, "interruption did not fire"

    assert all(not parameter.requires_grad and parameter.grad is None
               for frozen in (models.nvarc_encoder, models.vae_encoder) for parameter in frozen.parameters())
    if not runtime.is_primary:
        assert records == events == []
    return SimpleNamespace(progress=progress, snapshots=snapshots, records=records, events=events,
                           loaded=loaded, visited=dataset.visited)


def _checkpoint(config, step):
    return config.recipe["deployment"]["output_directory"] / "checkpoints" / "unified-local" / f"step-{step:06d}"


def _compare_updates(actual, expected, *, exact=False):
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed["parameters"], reference["parameters"],
                                   **(dict(atol=0, rtol=0) if exact else PARAM_TOLERANCE))
        torch.testing.assert_close(observed["optimizer"], reference["optimizer"],
                                   **(dict(atol=0, rtol=0) if exact else GRAD_TOLERANCE))
        torch.testing.assert_close(observed["scheduler"], reference["scheduler"], atol=0, rtol=0)
        if exact:
            torch.testing.assert_close(observed["rng"], reference["rng"], atol=0, rtol=0)
            assert observed["times"] == reference["times"]


def _assert_services(run, *, steps, initial):
    assert run.events.count("logger") == run.events.count("prepare") == 1
    for event in ("begin", "hardware", "save"):
        assert run.events.count(event) == steps
    for event in ("validate", "sample", "table", "log"):
        assert run.events.count(event) == steps + int(initial)


def _worker(rank, directory, native, next_native):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=(directory / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    runtime = DistributedRuntime(rank, rank, 2, torch.device("cpu"), dist.group.WORLD)
    try:
        config = _config(directory, strategy="ddp")
        full = _execute(config, runtime)
        _compare_updates(full.snapshots, native)
        assert (full.progress.optimizer_step, full.progress.sources_seen, full.progress.targets_seen) == (6, 21, 63)
        assert full.loaded == [directory / "selected-transformer"]
        plans = plan_distributed_batches((3,) * 7, optimizer_batch_size=12, rank=rank, world_size=2)
        assert full.visited == [index for plan in plans for index in plan.source_indices]
        for snapshot, reference, plan in zip(full.snapshots, native, plans, strict=True):
            assert snapshot["times"] == tuple(reference["times"][i] for i in plan.target_indices)
            assert snapshot["high_noise_quota"] == reference["high_noise_quota"] == 3
        if rank == 0:
            _assert_services(full, steps=2, initial=True)
            training = [row for row in full.records if "train/loss" in row]
            assert [row["train/targets"] for row in training] == [12, 9]
            assert [row["train/targets_t1"] for row in training] == [6, 4]
            for row, reference in zip(training, native, strict=True):
                torch.testing.assert_close(row["train/loss"], reference["loss"], atol=3e-6, rtol=3e-5)
                assert row["train/targets_high_noise"] >= 3
        state = torch.load(_checkpoint(config, 5) / "training_state.pt", weights_only=True)
        assert len(state["rng_by_rank"]) == 2 and "rng" not in state

        interrupted_config = _config(directory, strategy="ddp", output="interrupted")
        interrupted = _execute(interrupted_config, runtime, stop_after=5)
        assert interrupted.progress is None and len(interrupted.snapshots) == 1
        _compare_updates(interrupted.snapshots, full.snapshots[:1], exact=True)
        resume_path = _checkpoint(interrupted_config, 5)
        resume_config = _config(directory, strategy="ddp", output="interrupted", resume_from=resume_path)
        resumed = _execute(resume_config, runtime)
        _compare_updates(resumed.snapshots, full.snapshots[1:], exact=True)
        # Training progress is exact; rebuilding the validation cache on resume
        # performs two additional frozen-encoder tokens, which telemetry records.
        assert resumed.progress.optimizer_step == full.progress.optimizer_step
        assert resumed.progress.sources_seen == full.progress.sources_seen
        assert resumed.progress.targets_seen == full.progress.targets_seen
        expected_work = dict(full.progress.workload_totals)
        expected_work["val/nvarc_tokens"] += 2
        assert resumed.progress.workload_totals == expected_work
        assert (resumed.progress.epoch, resumed.progress.optimizer_start_step) == (
            full.progress.epoch, full.progress.optimizer_start_step)
        assert resumed.loaded == [resume_path / "transformer"]
        assert resumed.visited == list(plans[1].source_indices)
        if rank == 0:
            _assert_services(resumed, steps=1, initial=False)

        previous = read_checkpoint(_checkpoint(config, 6), config)
        next_config = _config(directory, strategy="ddp", epoch=4)
        continued = _execute(next_config, runtime, previous_epoch=previous)
        _compare_updates(continued.snapshots, next_native)
        assert continued.loaded == [previous.path / "transformer"]
        assert (continued.progress.epoch, continued.progress.optimizer_step,
                continued.progress.sources_seen, continued.progress.targets_seen) == (4, 8, 28, 84)
        if rank == 0:
            _assert_services(continued, steps=2, initial=False)
    finally:
        dist.destroy_process_group()


class UnifiedRunnerTests(unittest.TestCase):
    def test_native_and_ddp_share_weights_recipe_services_resume_and_next_epoch(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory(prefix="unified-training-") as directory, torch.random.fork_rng(devices=[]):
            directory = Path(directory).resolve()
            torch.manual_seed(167)
            _model().save_pretrained(directory / "selected-transformer", safe_serialization=True)
            runtime = DistributedRuntime(0, 0, 1, torch.device("cpu"), None, "single_process")
            config = _config(directory)
            native = _execute(config, runtime)
            assert not dist.is_initialized()
            _assert_services(native, steps=2, initial=True)
            assert native.loaded == [directory / "selected-transformer"]
            assert native.visited == list(range(7))
            checkpoint = read_checkpoint(_checkpoint(config, 6), config)
            next_config = _config(directory, epoch=4)
            next_native = _execute(next_config, runtime, previous_epoch=checkpoint)
            _assert_services(next_native, steps=2, initial=False)

            processes = mp.spawn(_worker, args=(str(directory), native.snapshots, next_native.snapshots),
                                 nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 120
                while not processes.join(timeout=1):
                    if time.monotonic() > deadline:
                        raise TimeoutError("unified runner check exceeded 120 seconds")
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes.processes:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
