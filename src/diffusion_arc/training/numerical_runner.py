"""Compare three packed training updates with single-device, DDP and FSDP2 execution."""

from contextlib import nullcontext
from dataclasses import asdict, replace
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
from .run.experiment import training_settings as _training_settings
from .config import parse_training_config
from .distributed import TrainingRuntime, distributed_runtime
from .distributed_data import build_distributed_dataloader
from .distribution_strategy.fsdp import shard_flux
from .run.numerics import (NumericalCriteria, compare_records, inspect_gradients, json_digest,
                           state_fingerprint, tensor_fingerprint, write_json)
from .setup import build_optimizer, load_models, prepare_device, seed_process
from .step.indexed_flow import build_indexed_flow_sampler, FLOW_SEED_SCHEME
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
    if phase not in ("reference", "ddp", "fsdp2"):
        raise ValueError("training phase must be reference, ddp or fsdp2")
    if dist.is_initialized():
        raise RuntimeError("numerical runner must own its distributed lifecycle")
    config = experiment.training
    if config.rank_batch_assignment == "joint_packing_v1":
        raise ValueError("numerical comparison does not support background joint packing")
    if (config.recipe["models"]["resume_from"] is not None
            or config.recipe["models"]["initialization"] != "base_checkpoint"
            or criteria.steps > config.loop.max_optimizer_steps):
        raise ValueError("numerical comparison requires fresh base weights and three available updates")
    input_recipe = config.recipe_dict()
    if config.recipe["flow"]["rng_scheme"] != FLOW_SEED_SCHEME:
        raise ValueError("numerical comparison requires explicit global_step_target_v1 flow RNG")
    output_directory = Path(output_directory)
    directory = output_directory / phase
    reference_directory = output_directory / "reference"
    gradient_directory = Path(scratch_directory) / "reference-gradients"
    ddp_gradient_directory = Path(scratch_directory) / "ddp-gradients"
    reference_steps = _read_steps(reference_directory) if phase != "reference" else None
    reference_manifest = (json.loads((reference_directory / "manifest.json").read_text())
                          if phase != "reference" else None)
    if reference_steps is not None and len(reference_steps) != criteria.steps:
        raise ValueError("reference phase must finish before distributed execution")
    # Keep the benchmark data override while selecting the real loading/placement path.
    phase_recipe = config.recipe_dict()
    phase_recipe["execution"]["distributed_strategy"] = "single_process" if phase == "reference" else phase
    if phase != "fsdp2":
        phase_recipe["execution"].pop("fsdp_reshard_after_forward", None)
        phase_recipe["execution"].pop("fsdp_blocks_per_group", None)
    phase_recipe["deployment"]["gpu_count"] = 1 if phase == "reference" else 2
    config = replace(parse_training_config(phase_recipe, config_path=config.path), data=config.data)
    if phase == "reference":
        torch.set_num_threads(8)
        context = nullcontext(TrainingRuntime(0, 0, 1, prepare_device(config), None))
    else:
        # Rank zero also reads the full CPU gradient reference. Keep this
        # diagnostic's collective timeout separate from normal training.
        context = distributed_runtime(config, timeout_seconds=600)
    with context as runtime:
        if phase != "reference" and runtime.world_size != 2:
            raise ValueError("numerical comparison requires exactly two distributed ranks")
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
                        recipe=input_recipe, input_recipe=input_recipe,
                        data_fingerprint=loader.dataset.fingerprint,
                        rank_batch_assignment=getattr(loader.dataset, "rank_assignment_metadata", None),
                        flow_seed_scheme=FLOW_SEED_SCHEME, initial=initial,
                        source_sha256=source_identity["source_sha256"],
                        packages={name: version(name) for name in
                                  ("torch", "transformers", "diffusers", "accelerate", "flash-attn-3")},
                        python=platform.python_version(), cuda=torch.version.cuda)
        identities = _gather(contract, runtime)
        if any(value != contract for value in identities):
            raise ValueError("rank inputs, initial state or software differ")
        if reference_manifest is not None and reference_manifest["contract"] != contract:
            raise ValueError("reference and distributed initial state, recipe, criteria, data or software differ")
        hardware = dict(rank=runtime.rank, device=str(runtime.device), threads=torch.get_num_threads(),
                        gpu=torch.cuda.get_device_name(runtime.device) if runtime.device.type == "cuda" else "cpu",
                        allocator=os.environ.get("PYTORCH_ALLOC_CONF"),
                        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        devices = _gather(hardware, runtime)
        if runtime.rank == 0:
            write_json(directory / "manifest.json", dict(phase=phase, world_size=runtime.world_size,
                       ddp=phase == "ddp", strategy=config.recipe["execution"]["distributed_strategy"],
                       effective_recipe=config.recipe_dict(), contract=contract, devices=devices, source_identity=source_identity,
                       note="Instrumented numerical diagnostic; timings are not benchmark measurements."))

        flux_model = PackedFluxForward(models.flux_transformer, compute_dtype=models.flux_compute_dtype,
                                    backend=models.flux_packing_backend)
        if phase == "ddp":
            flux_model = DDP(flux_model, device_ids=[runtime.device.index] if runtime.device.type == "cuda" else None,
                          process_group=runtime.process_group, forward_sync_buffers=False,
                          gradient_as_bucket_view=True)
        elif phase == "fsdp2":
            flux_model = shard_flux(flux_model, runtime=runtime,
                                    reshard_after_forward=config.fsdp_reshard_after_forward,
                                    blocks_per_group=config.fsdp_blocks_per_group)
        optimizer = build_optimizer(config, flux_transformer=flux_model)
        step, gradients = 0, None

        trainer = Trainer(
            nvarc_encoder=models.nvarc_encoder, vae_encoder=models.vae_encoder,
            flux_transformer=flux_model, optimizer=optimizer,
            scheduler=build_lr_scheduler(config, optimizer=optimizer),
            flow_sampler=lambda _: (), loss_function=build_velocity_loss(config),
            flux_compute_dtype=models.flux_compute_dtype, flux_packing_backend=models.flux_packing_backend,
            nvarc_max_tokens=config.nvarc_max_tokens, vae_microbatch_size=config.loop.vae_microbatch_size,
            flux_max_joint_tokens=config.flux_packing.max_tokens,
            flux_max_spread_tokens=config.flux_packing.max_spread_tokens,
            flux_packing_algorithm=config.flux_packing.algorithm,
            min_context_pairs_before_target=config.loop.min_context_pairs_before_target,
            max_grad_norm=config.optimizer.max_grad_norm,
        )
        # Capture unclipped gradients only in this numerical diagnostic.
        clip_gradients = trainer._clip_gradients

        def inspect_then_clip():
            nonlocal gradients
            if step == 1:
                print(json.dumps(dict(phase=phase, rank=runtime.rank, stage="capture_gradients")), flush=True)
                gradients = inspect_gradients(
                    models.flux_transformer, criteria.chunk_elements,
                    save_to=(gradient_directory if phase == "reference" else
                             ddp_gradient_directory if phase == "ddp" and runtime.rank == 0
                             and input_recipe["execution"]["distributed_strategy"] == "fsdp2" else None),
                    compare_to=gradient_directory if phase != "reference" and runtime.rank == 0 else None,
                    compare_to_ddp=ddp_gradient_directory if phase == "fsdp2" and runtime.rank == 0 else None,
                )
            return clip_gradients()

        trainer._clip_gradients = inspect_then_clip
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

            trainer.flow_sampler = replace(build_indexed_flow_sampler(
                config, step=plan.epoch_batch_index, global_target_count=plan.global_target_count,
                target_indices=plan.target_indices), observe_draw=observe_draw)
            result = trainer.step(sequences)
            frozen, frozen_ok = _frozen_checks(models, criteria.chunk_elements)
            local = dict(rank=runtime.rank, inputs=inputs, draws=draws, loss=result.loss,
                         targets=result.target_count, sources=result.source_count,
                         learning_rates=list(result.learning_rates), gradient_norm=result.gradient_norm,
                         gradient_clipped=result.gradient_clipped,
                         parameters=state_fingerprint(models.flux_transformer, criteria.chunk_elements),
                         gradients=gradients if step == 1 else None,
                         padding_forward_calls=result.padding_forward_calls,
                         frozen_unchanged=frozen == frozen_initial, frozen_no_grad=frozen_ok,
                         # Keep the persisted diagnostics schema independent of Python field names.
                         packs=dict(preparation=asdict(result.diagnostics.preparation),
                                    flux_batches=tuple(asdict(microbatch_plan)
                                                       for microbatch_plan in result.diagnostics.flux_microbatch_plans),
                                    target_prefix_tokens=result.diagnostics.target_prefix_tokens),
                         target_metrics=[asdict(target) for target in result.target_metrics])
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


def compare_numerical_phases(output_directory: Path, criteria: NumericalCriteria, *, include_fsdp2=False):
    output_directory = Path(output_directory)
    phases = ("reference", "ddp", "fsdp2") if include_fsdp2 else ("reference", "ddp")
    manifests = [json.loads((output_directory / phase / "manifest.json").read_text()) for phase in phases]
    if any(manifest["contract"] != manifests[0]["contract"] for manifest in manifests[1:]):
        raise ValueError("numerical comparison contracts differ")
    if manifests[0]["contract"]["criteria"] != asdict(criteria):
        raise ValueError("criteria changed after execution")
    if manifests[0]["ddp"] or manifests[0]["world_size"] != 1 or manifests[1]["world_size"] != 2:
        raise ValueError("expected a plain reference and two DDP ranks")
    if include_fsdp2 and (manifests[1]["strategy"] != "ddp" or manifests[2]["strategy"] != "fsdp2"
                         or manifests[2]["world_size"] != 2):
        raise ValueError("expected two FSDP2 ranks after the DDP reference")
    records = {phase: _read_steps(output_directory / phase) for phase in phases}
    summary = compare_records(records["reference"], records["ddp"], criteria)
    if include_fsdp2:
        comparisons = dict(single_gpu_vs_ddp=summary,
                           single_gpu_vs_fsdp2=compare_records(records["reference"], records["fsdp2"],
                                                              criteria, strategy="fsdp2"),
                           ddp_vs_fsdp2=compare_records(records["ddp"], records["fsdp2"], criteria,
                                                       strategy="fsdp2", reference_strategy="ddp"))
        failures = [f"{name}/{failure}" for name, result in comparisons.items() for failure in result["failures"]]
        summary = dict(event="complete", status="failed" if failures else "passed",
                       optimizer_steps=criteria.steps, failures=failures, comparisons=comparisons)
    summary["criteria"] = asdict(criteria)
    summary["limitations"] = ["Three initial updates only; no long-run convergence claim.",
                              "Full gradients compared only at step one, before clipping.",
                              "Subsequent loss comparisons include accumulated model-state differences.",
                              "Per-block metrics are diagnostic; the preregistered gradient gate is global L2.",
                              "Parameter updates and AdamW moments are not directly compared between strategies."]
    write_json(output_directory / "summary.json", summary)
    return summary
