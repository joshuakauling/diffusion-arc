"""Compare the actual packed training pipeline with plain and two-rank execution."""

from contextlib import nullcontext
from dataclasses import asdict
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion_arc.model.flux.packing import PackedFluxForward
from .ddp_runner import _training_settings
from .distributed import DistributedRuntime, distributed_runtime
from .distributed_data import build_distributed_dataloader
from .run.numerics import (NumericalCriteria, compare_records, inspect_gradients, json_digest,
                           state_fingerprint, tensor_fingerprint, write_json)
from .setup import build_optimizer, load_models, prepare_device, seed_process
from .step.distributed_flow import DistributedFlowSampler, FLOW_SEED_SCHEME
from .step.loss import build_velocity_loss
from .step.lr_schedule import build_lr_scheduler
from .trainer import Trainer


def _gather(value, runtime):
    if runtime.process_group is None:
        return [value]
    result = [None] * runtime.world_size
    dist.all_gather_object(result, value, group=runtime.process_group)
    return result


def _frozen_checks(models, chunk_elements):
    modules = dict(nvarc=models.nvarc_encoder, vae=models.vae_encoder)
    return ({name: state_fingerprint(module, chunk_elements) for name, module in modules.items()},
            all(not p.requires_grad and p.grad is None for m in modules.values() for p in m.parameters()))


def _read_steps(directory):
    return [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]


def run_numerical_phase(experiment, criteria: NumericalCriteria, *, phase: str,
                        output_directory: Path, scratch_directory: Path, source_identity: dict):
    """Run exactly three actual updates; expensive capture stays in this diagnostic."""
    if phase not in ("reference", "ddp"):
        raise ValueError("training phase must be reference or ddp")
    if dist.is_initialized():
        raise RuntimeError("numerical runner must own its distributed lifecycle")
    config = experiment.training
    if config.recipe["models"]["resume_from"] is not None or criteria.steps > config.loop.max_optimizer_steps:
        raise ValueError("numerical comparison requires fresh base weights and three available updates")
    output_directory = Path(output_directory)
    directory = output_directory / phase
    reference_directory = output_directory / "reference"
    gradient_directory = Path(scratch_directory) / "reference-gradients"
    reference_steps = _read_steps(reference_directory) if phase == "ddp" else None
    reference_manifest = (json.loads((reference_directory / "manifest.json").read_text())
                          if phase == "ddp" else None)
    if phase == "ddp" and len(reference_steps) != criteria.steps:
        raise ValueError("reference phase must finish before distributed execution")
    if phase == "reference":
        torch.set_num_threads(8)
        context = nullcontext(DistributedRuntime(0, 0, 1, prepare_device(config), None))
    else:
        # Rank zero also reads the full CPU gradient reference. Keep this
        # diagnostic's collective timeout separate from normal training.
        context = distributed_runtime(config, timeout_seconds=600)
    with context as runtime:
        if phase == "ddp" and runtime.world_size != 2:
            raise ValueError("numerical comparison requires exactly two DDP ranks")
        if runtime.rank == 0:
            directory.mkdir(parents=True, exist_ok=False)
        print(json.dumps(dict(phase=phase, rank=runtime.rank, stage="load_models")), flush=True)
        seed_process(config.seed)
        models = load_models(config, device=runtime.device)
        loader, plans = build_distributed_dataloader(
            config, tokenizer=models.tokenizer, rank=runtime.rank, world_size=runtime.world_size,
            expected_source_count=experiment.expected_source_count,
        )
        if len(plans) < criteria.steps:
            raise ValueError("not enough global batches for the numerical comparison")
        frozen_initial, frozen_ok = _frozen_checks(models, criteria.chunk_elements)
        if not frozen_ok:
            raise ValueError("frozen modules have trainable parameters or gradients")
        initial = dict(flux=state_fingerprint(models.flux_transformer, criteria.chunk_elements),
                       frozen=frozen_initial)
        contract = dict(criteria=asdict(criteria), training=_training_settings(config),
                        recipe=config.recipe_dict(), data_fingerprint=loader.dataset.fingerprint,
                        flow_seed_scheme=FLOW_SEED_SCHEME, initial=initial,
                        source_sha256=source_identity["source_sha256"],
                        packages={name: version(name) for name in
                                  ("torch", "transformers", "diffusers", "accelerate", "flash-attn-3")},
                        python=platform.python_version(), cuda=torch.version.cuda)
        identities = _gather(contract, runtime)
        if any(value != contract for value in identities):
            raise ValueError("rank inputs, initial state or software differ")
        if reference_manifest is not None and reference_manifest["contract"] != contract:
            raise ValueError("reference and DDP initial state, recipe, criteria, data or software differ")
        hardware = dict(rank=runtime.rank, device=str(runtime.device), threads=torch.get_num_threads(),
                        gpu=torch.cuda.get_device_name(runtime.device) if runtime.device.type == "cuda" else "cpu",
                        allocator=os.environ.get("PYTORCH_ALLOC_CONF"),
                        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        devices = _gather(hardware, runtime)
        if runtime.rank == 0:
            write_json(directory / "manifest.json", dict(phase=phase, world_size=runtime.world_size,
                       ddp=phase == "ddp", contract=contract, devices=devices, source_identity=source_identity,
                       note="Instrumented numerical diagnostic; timings are not benchmark measurements."))

        forward = PackedFluxForward(models.flux_transformer, compute_dtype=models.flux_compute_dtype,
                                    backend=models.flux_packing_backend)
        if phase == "ddp":
            forward = DDP(forward, device_ids=[runtime.device.index] if runtime.device.type == "cuda" else None,
                          process_group=runtime.process_group, forward_sync_buffers=False,
                          gradient_as_bucket_view=True)
        optimizer = build_optimizer(config, flux_transformer=models.flux_transformer)
        step, gradients = 0, None

        def trace(stage, details):
            nonlocal gradients
            if stage == "clip_start" and step == 1:
                print(json.dumps(dict(phase=phase, rank=runtime.rank, stage="capture_gradients")), flush=True)
                gradients = inspect_gradients(
                    models.flux_transformer, criteria.chunk_elements,
                    save_to=gradient_directory if phase == "reference" else None,
                    compare_to=gradient_directory if phase == "ddp" and runtime.rank == 0 else None,
                )

        trainer = Trainer(
            nvarc_encoder=models.nvarc_encoder, vae_encoder=models.vae_encoder,
            flux_transformer=forward, optimizer=optimizer,
            scheduler=build_lr_scheduler(config, optimizer=optimizer),
            flow_sampler=lambda _: (), loss_function=build_velocity_loss(config),
            flux_compute_dtype=models.flux_compute_dtype, flux_packing_backend=models.flux_packing_backend,
            nvarc_max_tokens=config.nvarc_max_tokens, vae_microbatch_size=config.loop.vae_microbatch_size,
            flux_max_joint_tokens=config.flux_packing.max_tokens,
            flux_max_spread_tokens=config.flux_packing.max_spread_tokens,
            min_context_pairs_before_target=config.loop.min_context_pairs_before_target,
            max_grad_norm=config.optimizer.max_grad_norm, trace=trace,
        )
        samples = iter(loader)
        rows = []
        for step, plan in enumerate(plans[:criteria.steps], 1):
            print(json.dumps(dict(phase=phase, rank=runtime.rank, stage="step", step=step)), flush=True)
            sequences = tuple(next(samples) for _ in plan.source_indices)
            inputs = [dict(index=index, key=str(sample.key),
                           sha256=json_digest([sample.input_ids, sample.prefix_token_ends, sample.target_grids]))
                      for index, sample in zip(plan.source_indices, sequences, strict=True)]
            expected_draws = None
            if reference_steps is not None:
                expected = reference_steps[step - 1]["ranks"][0]
                expected_inputs = {x["index"]: x for x in expected["inputs"]}
                if inputs != [expected_inputs[index] for index in plan.source_indices]:
                    raise ValueError("actual source contents differ from the reference")
                expected_draws = {x["target"]: x for x in expected["draws"]}
            draws = []

            def observe_draw(index, time, noise):
                draw = dict(target=index, time=float(time.item()),
                            noise_sha256=tensor_fingerprint(noise, criteria.chunk_elements))
                if expected_draws is not None and draw != expected_draws[index]:
                    raise ValueError("actual random draw differs from the reference")
                draws.append(draw)

            trainer.flow_sampler = DistributedFlowSampler(
                config.seed, plan.step, plan.global_target_count, plan.target_indices,
                config.recipe["flow"]["endpoint_targets_per_full_batch"], observe_draw=observe_draw,
            )
            result = trainer.step(sequences)
            frozen, frozen_ok = _frozen_checks(models, criteria.chunk_elements)
            local = dict(rank=runtime.rank, inputs=inputs, draws=draws, loss=result.loss,
                         targets=result.target_count, sources=result.source_count,
                         learning_rates=list(result.learning_rates), gradient_norm=result.gradient_norm,
                         gradient_clipped=result.gradient_clipped,
                         parameters=state_fingerprint(models.flux_transformer, criteria.chunk_elements),
                         gradients=gradients if step == 1 else None,
                         frozen_unchanged=frozen == frozen_initial, frozen_no_grad=frozen_ok,
                         packs=asdict(result.diagnostics),
                         target_metrics=[asdict(target) for target in result.targets])
            ranks = _gather(local, runtime)
            targets = sum(r["targets"] for r in ranks)
            if targets != plan.global_target_count:
                raise ValueError("actual global target count differs")
            loss = math.fsum(r["loss"] * r["targets"] for r in ranks) / targets
            row = dict(step=step, loss=loss, targets=targets, ranks=ranks)
            rows.append(row)
            if runtime.rank == 0:
                with (directory / "steps.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(dict(phase=phase, step=step, loss=loss,
                                      gradient_norm=ranks[0]["gradient_norm"])), flush=True)
        if runtime.rank == 0:
            write_json(directory / "summary.json", dict(event="complete", optimizer_steps=len(rows), phase=phase))


def compare_numerical_phases(output_directory: Path, criteria: NumericalCriteria):
    output_directory = Path(output_directory)
    manifests = [json.loads((output_directory / phase / "manifest.json").read_text())
                 for phase in ("reference", "ddp")]
    if manifests[0]["contract"] != manifests[1]["contract"]:
        raise ValueError("numerical comparison contracts differ")
    if manifests[0]["contract"]["criteria"] != asdict(criteria):
        raise ValueError("criteria changed after execution")
    if manifests[0]["ddp"] or manifests[0]["world_size"] != 1 or manifests[1]["world_size"] != 2:
        raise ValueError("expected a plain reference and two DDP ranks")
    summary = compare_records(_read_steps(output_directory / "reference"),
                              _read_steps(output_directory / "ddp"), criteria)
    summary["criteria"] = asdict(criteria)
    summary["limitations"] = ["Three initial updates only; no long-run convergence claim.",
                              "Full reference-vs-DDP gradients compared only at step one, before clipping.",
                              "Subsequent loss comparisons include accumulated model-state differences.",
                              "Per-block metrics are diagnostic; the preregistered gradient gate is global L2."]
    write_json(output_directory / "summary.json", summary)
    return summary
