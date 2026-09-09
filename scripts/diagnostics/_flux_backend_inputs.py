"""Fixed native-Qwen and ARC inputs for checkpoint/backend sensitivity diagnostics."""

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path

import torch

from diffusion_arc.arc_data import (
    apply_training_augmentation, parse_augmentation_key, render_training_prompt,
    tokenize_nvarc_training_sample, validate_augmentation_for_task,
)
from diffusion_arc.arc_data._grid import validate_arc_grid
from diffusion_arc.arc_data.canvas import render_grid_canvas
from diffusion_arc.arc_data.palette import TARGET_PALETTE_ID
from diffusion_arc.arc_data.types import ArcPair, ArcTrainingTask
from diffusion_arc.model.flux_conditioning import pad_flux_text_prefixes
from diffusion_arc.model.flux_velocity import build_flux_image_position_ids


@dataclass(frozen=True)
class PreparedBackendInput:
    """One shared, detached CPU FP32 exposure, prepared before either checkpoint."""

    name: str
    text: torch.Tensor
    text_ids: torch.Tensor
    images: torch.Tensor
    image_ids: torch.Tensor
    timesteps: torch.Tensor
    target_velocity: torch.Tensor | None
    metadata: dict


def json_sha256(value):
    """Match the canonical JSON identities in the historical overfitting report."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _render_view(view):
    source = view["source_content"]
    pairs = []
    for row in (*source["train"], *source["test"]):
        grids = tuple(tuple(tuple(line) for line in row[key]) for key in ("input", "output"))
        for grid in grids:
            validate_arc_grid(grid)
        pairs.append(ArcPair(input_grid=grids[0], output_grid=grids[1]))
    task = ArcTrainingTask(task_id=view["task_id"], pairs=tuple(pairs))
    key = parse_augmentation_key(view["key"])
    validate_augmentation_for_task(key, task)
    if key.task_id != task.task_id or key.augmentation.pair_indices[-1] != len(task.pairs) - 1:
        raise ValueError("diagnostic must retain the final labelled query as its only target")
    augmented = apply_training_augmentation(task, key.augmentation)
    payload = [{"input": pair.input_grid, "output": pair.output_grid} for pair in augmented.pairs]
    identities = {"source_content_sha256": json_sha256(source),
                  "full_view_sha256": json_sha256(payload),
                  "query_grid_sha256": json_sha256(augmented.pairs[-1].input_grid),
                  "target_grid_sha256": json_sha256(augmented.pairs[-1].output_grid)}
    if any(view[field] != digest for field, digest in identities.items()):
        raise ValueError("ARC diagnostic source/grid identity differs from its fixture")
    return render_training_prompt(augmented)


def load_fixture(path):
    """Validate tiny, embedded training-source grids and historical pool membership."""
    fixture = json.loads(Path(path).read_text())
    if fixture["schema_version"] != 1:
        raise ValueError("unsupported backend-input fixture schema")
    expected_preprocessing = {
        "palette": TARGET_PALETTE_ID, "canvas_size": 30, "cell_size": 16,
        "image_tokens": 900, "image_channels": 128, "text_width": 7680,
        "hidden_state_layers": [9, 18, 27], "nvarc_dtype": "bfloat16", "vae_dtype": "float32",
        "retained_input_dtype": "float32", "nvarc_physical_batch_size": 1,
        "historical_nvarc_physical_batch_sizes": [512, 129],
    }
    if fixture["preprocessing"] != expected_preprocessing:
        raise ValueError("fixture changes the historical ARC preparation contract")
    source = fixture["source_split"]
    train_ids, validation_ids = set(source["train_task_ids"]), set(source["validation_task_ids"])
    if source["dataset"] != "arc-agi-2" or source["split"] != "training" or train_ids & validation_ids:
        raise ValueError("fixture violates the ARC training task boundary")
    overfit = fixture["overfit"]
    if overfit["step"] != 4096 or overfit["run_id"] != "20260827T011822050154Z":
        raise ValueError("unexpected overfit checkpoint lineage")
    pools = {}
    for name, count in (("train_keys", 512), ("holdout_keys", 128)):
        values = overfit[name]
        if len(values) != count or len(set(values)) != count or json_sha256(values) != overfit[name + "_sha256"]:
            raise ValueError("historical overfit pool membership differs")
        pools[name] = set(values)
    if pools["train_keys"] & pools["holdout_keys"]:
        raise ValueError("seen and unseen augmentation pools overlap")
    if [(v["name"], v["category"]) for v in fixture["arc_views"]] != [
        ("arc_seen", "seen"), ("arc_unseen", "unseen_same_task"), ("arc_other", "unrelated_train_task")
    ]:
        raise ValueError("fixture must contain the three predetermined ARC controls")
    for view in fixture["arc_views"]:
        if view["task_id"] not in train_ids or view["task_id"] in validation_ids:
            raise ValueError("diagnostic task is outside the explicit training whitelist")
        category, key = view["category"], view["key"]
        exposure_count = view["historical_cumulative_exposure_audit"]["total"]
        if bool(exposure_count) != view["trained_in_overfit"]:
            raise ValueError("seen status differs from the completed training exposure audit")
        if category == "seen" and not view["historical_continuation_exposure_steps"]:
            raise ValueError("seen view has no recorded completed optimizer steps")
        if category == "seen" and (key not in pools["train_keys"] or not view["trained_in_overfit"]):
            raise ValueError("seen view is absent from the checkpoint training pool")
        if category == "unseen_same_task" and (key not in pools["holdout_keys"] or view["trained_in_overfit"]):
            raise ValueError("unseen view is absent from the checkpoint holdout pool")
        if category == "unrelated_train_task" and (view["task_id"] == "0d3d703e" or view["trained_in_overfit"]):
            raise ValueError("unrelated control belongs to the overfit task")
        _render_view(view)
    if not isinstance(fixture["native_prompt"], str) or not fixture["native_prompt"].strip():
        raise ValueError("native control requires an ordinary image prompt")
    return fixture


def shared_noise(seed):
    """Use one CPU-generated Gaussian realization for every conditioning group."""
    if type(seed) is not int or seed < 0:
        raise ValueError("noise seed must be a nonnegative integer")
    return torch.randn((1, 900, 128), generator=torch.Generator().manual_seed(seed), dtype=torch.float32)


def prepare_arc_inputs(fixture, *, tokenizer, nvarc_runner, vae_encoder, seed=167, timesteps=(1.0, 0.5)):
    """Prepare each true target-free NVARC prefix and VAE target exactly once."""
    if not timesteps or any(type(t) not in (int, float) or not 0 < t <= 1 for t in timesteps):
        raise ValueError("ARC timesteps must lie in (0, 1]")
    noise, result = shared_noise(seed), []
    for view in fixture["arc_views"]:
        rendered = _render_view(view)
        sample = tokenize_nvarc_training_sample(rendered, tokenizer, max_tokens=8400)
        prefix = sample.input_ids[:sample.prefix_token_ends[-1]]
        if view["prefix_ids_sha256"] is not None and json_sha256(prefix) != view["prefix_ids_sha256"]:
            raise ValueError("NVARC token IDs differ from the successful overfit preparation")
        if view["expected_prefix_tokens"] is not None and len(prefix) != view["expected_prefix_tokens"]:
            raise ValueError("NVARC prefix length differs from the successful overfit preparation")
        with torch.no_grad():
            encoded = nvarc_runner(torch.tensor(prefix, dtype=torch.int64),
                                   torch.tensor([0, len(prefix)], dtype=torch.int64))
            hidden = encoded.sequence_hidden_states[0].detach().to(device="cpu", dtype=torch.float32).clone()
            clean = vae_encoder(render_grid_canvas(sample.target_grids[-1]).unsqueeze(0))
            clean = clean.detach().to(device="cpu", dtype=torch.float32).clone()
        del encoded
        if hidden.shape != (len(prefix), 7680) or clean.shape != (1, 900, 128):
            raise ValueError("prepared ARC tensors violate the native FLUX input dimensions")
        text = pad_flux_text_prefixes((hidden,))
        metadata = {"category": view["category"], "augmentation_key": view["key"],
                    "source_split": "arc-agi-2/training", "trained_in_overfit": view["trained_in_overfit"],
                    "prefix_ids_sha256": json_sha256(prefix), "prefix_tokens": len(prefix),
                    "target_grid": sample.target_grids[-1], "target_grid_sha256": view["target_grid_sha256"],
                    "text_sha256": tensor_sha256(text.encoder_hidden_states),
                    "clean_latent_sha256": tensor_sha256(clean), "noise_sha256": tensor_sha256(noise),
                    "noise_seed": seed, "nvarc_physical_batch_size": 1,
                    "historical_nvarc_physical_batch_sizes": [512, 129],
                    "nvarc_reencoding": "same frozen checkpoint and exact token IDs; re-encoded as singleton",
                    "historical_cumulative_exposures": view["historical_cumulative_exposure_audit"]["total"],
                    "palette": TARGET_PALETTE_ID, "clean_latent_policy": "FP32 VAE posterior mode"}
        for timestep in timesteps:
            images = (1 - timestep) * clean + timestep * noise
            result.append(PreparedBackendInput(
                name=f"{view['name']}_t{timestep:g}", text=text.encoder_hidden_states,
                text_ids=text.text_position_ids, images=images,
                image_ids=build_flux_image_position_ids(1, torch.device("cpu")),
                timesteps=torch.tensor([timestep], dtype=torch.float32), target_velocity=noise - clean,
                metadata={**metadata, "timestep": timestep, "images_sha256": tensor_sha256(images)},
            ))
    return result


def prepare_native_input(fixture, *, text_encoder, tokenizer, seed=167):
    """Use the installed FLUX pipeline's exact 512-position native Qwen helper."""
    from diffusers import Flux2KleinPipeline

    captured = {}
    def capture(_module, _args, kwargs):
        captured.update({key: tensor_sha256(kwargs[key]) for key in ("input_ids", "attention_mask")})
        captured["unpadded_tokens"] = int(kwargs["attention_mask"].sum().item())

    handle = text_encoder.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            text = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                text_encoder, tokenizer, fixture["native_prompt"], max_sequence_length=512,
                hidden_states_layers=(9, 18, 27),
            ).detach().to(device="cpu", dtype=torch.float32).clone()
    finally:
        handle.remove()
    if text.shape != (1, 512, 7680):
        raise ValueError("native Qwen helper must retain all 512 FLUX conditioning positions")
    images = shared_noise(seed)
    return PreparedBackendInput(
        name="native_prompt_t1", text=text, text_ids=Flux2KleinPipeline._prepare_text_ids(text),
        images=images, image_ids=build_flux_image_position_ids(1, torch.device("cpu")),
        timesteps=torch.ones(1, dtype=torch.float32), target_velocity=None,
        metadata={"category": "native_prompt", "prompt": fixture["native_prompt"],
                  "text_encoder_source": "flux2-klein-base-4b/text_encoder", "noise_seed": seed,
                  "noise_sha256": tensor_sha256(images), "images_sha256": tensor_sha256(images),
                  "text_sha256": tensor_sha256(text), "native_qwen_call": captured,
                  "text_tokens": 512, "flux_retains_all_padding_positions": True,
                  "hidden_state_layers": [9, 18, 27], "timestep": 1.0,
                  "target_policy": "pure noise at t=1; no paired photograph or flow-loss target"},
    )


def prepare_arc_from_checkpoints(fixture, *, nvarc_checkpoint, base_checkpoint, device, seed=167,
                                 timesteps=(1.0, 0.5)):
    """Load only frozen components offline, then release them before FLUX loads."""
    from diffusers import AutoencoderKLFlux2
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from diffusion_arc.model.flux_vae import FluxVaeEncoder
    from diffusion_arc.model.nvarc_encoder import NvarcBatchRunner
    from diffusion_arc.training.single_gpu_setup import _validate_nvarc_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(nvarc_checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        nvarc_checkpoint, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa",
    )
    _validate_nvarc_checkpoint(model, tokenizer, expected_device=device, expected_dtype=torch.bfloat16)
    runner = NvarcBatchRunner(model)
    vae = AutoencoderKLFlux2.from_pretrained(
        base_checkpoint, subfolder="vae", local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float32, device_map={"": device},
    )
    encoder = FluxVaeEncoder(vae)
    result = prepare_arc_inputs(fixture, tokenizer=tokenizer, nvarc_runner=runner, vae_encoder=encoder,
                                seed=seed, timesteps=timesteps)
    del model, runner, vae, encoder, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def prepare_native_from_checkpoint(fixture, *, base_checkpoint, device, seed=167):
    """Load the original Qwen encoder and tokenizer from existing local files."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(base_checkpoint) / "tokenizer", local_files_only=True,
                                             trust_remote_code=False)
    encoder = AutoModelForCausalLM.from_pretrained(
        Path(base_checkpoint) / "text_encoder", local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa",
    ).eval().requires_grad_(False)
    result = prepare_native_input(fixture, text_encoder=encoder, tokenizer=tokenizer, seed=seed)
    del encoder, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
