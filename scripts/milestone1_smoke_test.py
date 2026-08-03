#!/usr/bin/env python3
"""Run the real frozen-NVARC -> trainable-FLUX -> frozen-VAE interface smoke test.

This runner is intentionally local-only. Attach the model checkpoints and ARC
JSON files as Kaggle inputs, then point every CLI argument at those paths.
It performs one deterministic flow-loss forward/backward pass and writes a
machine-readable evidence report; it does not download models or data.

Terminal example (run from the repository root and replace the placeholder paths):

    python scripts/milestone1_smoke_test.py \
      --nvarc-path /kaggle/input/.../nvarc \
      --flux-path /kaggle/input/.../flux-2-klein-base-4b.safetensors \
      --vae-path /kaggle/input/.../ae.safetensors \
      --task-json /kaggle/input/.../arc-agi_training_challenges.json \
      --solution-json /kaggle/input/.../arc-agi_training_solutions.json \
      --task-id TASK_ID \
      --nvarc-device cuda:0 \
      --flux-device cuda:1 \
      --vae-device cuda:0 \
      --report-path /kaggle/working/milestone1-smoke.json

For the identity-verified rerun, add:

    --expected-nvarc-manifest-sha256 NVARC_MANIFEST_SHA256
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from diffusion_arc.nvarc_conditioning import NVARCConditioner  # noqa: E402
from diffusion_arc.loading import (  # noqa: E402
    OFFICIAL_FLUX2_KLEIN_BASE_4B_SHA256,
    OFFICIAL_FLUX2_VAE_SHA256,
    load_bfl_autoencoder_local,
    load_bfl_flux2_klein_4b_local,
    load_nvarc_local,
    sha256_file,
)
from diffusion_arc.model import (  # noqa: E402
    ARC_IMAGE_TOKENS,
    ARC_LATENT_CHANNELS,
    DiffusionARCModel,
)
from diffusion_arc.serialization import serialize_arc_task  # noqa: E402


BFL_SOURCE_COMMIT = "50fe5162777813d869182b139e83b10743caef15"
DEFAULT_NVARC_SOURCE = (
    "kaggle:sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Milestone-1 forward/backward smoke test for the complete "
            "Diffusion-ARC PyTorch model."
        )
    )
    parser.add_argument("--nvarc-path", type=Path, required=True)
    parser.add_argument("--nvarc-source", default=DEFAULT_NVARC_SOURCE)
    parser.add_argument("--flux-path", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--solution-json", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--nvarc-device", default="cuda:0")
    parser.add_argument("--flux-device", default="cuda:1")
    parser.add_argument("--vae-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flow-time", type=float, default=0.5)
    parser.add_argument(
        "--generation-steps",
        type=int,
        default=1,
        help=(
            "Real conditioned Euler steps to execute after backward. The default "
            "one-step run validates the inference/VAE/decode interfaces, not quality."
        ),
    )
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--expected-nvarc-manifest-sha256")
    parser.add_argument(
        "--expected-flux-sha256",
        default=OFFICIAL_FLUX2_KLEIN_BASE_4B_SHA256,
    )
    parser.add_argument(
        "--expected-vae-sha256",
        default=OFFICIAL_FLUX2_VAE_SHA256,
    )
    parser.add_argument(
        "--skip-checkpoint-hashes",
        action="store_true",
        help=(
            "Skip identity hashing unless an expected hash was supplied. "
            "Hashing is enabled by default for reproducible evidence."
        ),
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected a local JSON file: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_arc_example(
    task_path: Path,
    solution_path: Path | None,
    *,
    task_id: str | None,
    test_index: int,
) -> tuple[dict[str, Any], Any, str]:
    """Load one challenge and target without ever injecting the target into it."""

    payload = _read_json(task_path)
    if isinstance(payload, dict) and {"train", "test"} <= payload.keys():
        task = payload
        selected_id = task_id or task_path.stem
    else:
        if not isinstance(payload, dict):
            raise TypeError("task JSON must contain one ARC task or a task-id mapping.")
        if task_id is None:
            raise ValueError("--task-id is required for a multi-task challenge JSON.")
        if task_id not in payload:
            raise KeyError(f"Task id {task_id!r} was not found in {task_path}.")
        task = payload[task_id]
        selected_id = task_id

    if not isinstance(test_index, int) or isinstance(test_index, bool) or test_index < 0:
        raise ValueError("--test-index must be a non-negative integer.")
    tests = task.get("test") if isinstance(task, dict) else None
    if not isinstance(tests, list) or test_index >= len(tests):
        raise IndexError(
            f"test index {test_index} is outside task {selected_id!r}."
        )

    if solution_path is not None:
        solutions = _read_json(solution_path)
        if isinstance(solutions, dict) and selected_id in solutions:
            targets = solutions[selected_id]
        elif isinstance(solutions, list):
            targets = solutions
        else:
            raise KeyError(
                f"Solutions for task {selected_id!r} were not found in {solution_path}."
            )
        if test_index >= len(targets):
            raise IndexError(
                f"test index {test_index} has no solution for task {selected_id!r}."
            )
        target = targets[test_index]
    else:
        test_example = tests[test_index]
        if not isinstance(test_example, dict) or "output" not in test_example:
            raise ValueError(
                "The selected test example has no output; provide --solution-json."
            )
        target = test_example["output"]

    # The serializer explicitly ignores test outputs, including any present in
    # training-format task files. This assertion exercises that path here.
    prompt = serialize_arc_task(task, test_index=test_index)
    if not prompt.endswith("<|im_start|>assistant\n"):
        raise AssertionError("Serialized NVARC prompt has an unexpected suffix.")
    return task, target, selected_id


def _tokenize_exact(tokenizer: Any, prompt: str) -> tuple[Tensor, Tensor]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    if tuple(input_ids.shape) != tuple(attention_mask.shape):
        raise RuntimeError("Tokenizer returned mismatched IDs and attention mask.")
    if input_ids.shape[0] != 1:
        raise RuntimeError("The smoke test requires a single exact-length prompt.")
    if not 1 <= input_ids.shape[1] <= 8192:
        raise RuntimeError(
            f"Tokenized prompt length must lie in [1, 8192], got {input_ids.shape[1]}."
        )
    if not bool((attention_mask == 1).all().item()):
        raise RuntimeError("Tokenizer unexpectedly padded or masked the prompt.")
    return input_ids, attention_mask


def _normalize_sha256(value: str, *, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal characters.")
    return normalized


def _verify_hash(actual: str, expected: str | None, *, name: str) -> bool:
    if expected is None:
        return False
    normalized = _normalize_sha256(expected, name=name)
    if actual != normalized:
        raise RuntimeError(
            f"{name} mismatch: expected {normalized}, got {actual}."
        )
    return True


def hash_nvarc_directory(path: Path) -> dict[str, Any]:
    """Hash NVARC weights and tokenizer/config files into one manifest identity."""

    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Expected a local NVARC directory: {root}")
    metadata_names = {
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.json",
    }
    files = {
        item
        for item in root.rglob("*")
        if item.is_file()
        and (
            item.name in metadata_names
            or item.suffix == ".safetensors"
            or item.name.startswith("pytorch_model")
        )
    }
    if not any(
        item.suffix == ".safetensors" or item.name.startswith("pytorch_model")
        for item in files
    ):
        raise RuntimeError(f"No NVARC weight files were found below {root}.")

    entries: list[dict[str, Any]] = []
    manifest_digest = hashlib.sha256()
    for item in sorted(files):
        relative = item.relative_to(root).as_posix()
        digest = sha256_file(item)
        size = item.stat().st_size
        entries.append({"path": relative, "bytes": size, "sha256": digest})
        manifest_digest.update(f"{relative}\0{size}\0{digest}\n".encode())
    return {
        "path": str(root),
        "manifest_sha256": manifest_digest.hexdigest(),
        "files": entries,
    }


def _component_manifest(module: nn.Module) -> dict[str, Any]:
    parameters = tuple(module.parameters())
    buffers = tuple(module.buffers())
    return {
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "parameter_devices": sorted({str(parameter.device) for parameter in parameters}),
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
        "buffer_dtypes": sorted({str(buffer.dtype) for buffer in buffers}),
        "training": module.training,
    }


def _cuda_devices(*values: str) -> tuple[torch.device, ...]:
    devices: dict[int, torch.device] = {}
    for value in values:
        device = torch.device(value)
        if device.type != "cuda":
            continue
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device} was requested but CUDA is unavailable.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device} does not exist; found {torch.cuda.device_count()} GPUs."
            )
        devices[index] = torch.device("cuda", index)
    return tuple(devices[index] for index in sorted(devices))


def _reset_cuda_peaks(devices: tuple[torch.device, ...]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _memory_snapshot(devices: tuple[torch.device, ...]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for device in devices:
        torch.cuda.synchronize(device)
        properties = torch.cuda.get_device_properties(device)
        snapshot[str(device)] = {
            "name": properties.name,
            "total_bytes": properties.total_memory,
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    return snapshot


def _gradient_audit(model: DiffusionARCModel) -> dict[str, Any]:
    frozen_with_grad = [
        f"{prefix}.{name}"
        for prefix, module in (("conditioner", model.conditioner), ("vae", model.vae))
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    ]
    missing: list[str] = []
    non_finite: list[str] = []
    tensors_with_grad = 0
    elements_with_grad = 0
    norm_squared = 0.0
    for name, parameter in model.flux.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        tensors_with_grad += 1
        elements_with_grad += gradient.numel()
        if not bool(torch.isfinite(gradient).all().item()):
            non_finite.append(name)
            continue
        norm = torch.linalg.vector_norm(gradient.detach(), dtype=torch.float32)
        norm_squared += float(norm.item()) ** 2

    report = {
        "frozen_parameters_with_grad": frozen_with_grad[:20],
        "missing_flux_gradients": missing[:20],
        "non_finite_flux_gradients": non_finite[:20],
        "flux_gradient_tensor_count": tensors_with_grad,
        "flux_gradient_element_count": elements_with_grad,
        "flux_global_grad_norm": math.sqrt(norm_squared),
    }
    if frozen_with_grad:
        raise RuntimeError("Frozen NVARC/VAE parameters received gradients.")
    if missing:
        raise RuntimeError(f"{len(missing)} trainable FLUX parameters have no gradient.")
    if non_finite:
        raise FloatingPointError(
            f"{len(non_finite)} FLUX gradients contain non-finite values."
        )
    if not math.isfinite(report["flux_global_grad_norm"]):
        raise FloatingPointError("FLUX global gradient norm is non-finite.")
    if report["flux_global_grad_norm"] <= 0:
        raise RuntimeError("FLUX global gradient norm is zero.")
    return report


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in (
        "accelerate",
        "einops",
        "safetensors",
        "torch",
        "transformers",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= args.flow_time <= 1.0 or not math.isfinite(args.flow_time):
        raise ValueError("--flow-time must be finite and lie in [0, 1].")
    if (
        isinstance(args.generation_steps, bool)
        or not isinstance(args.generation_steps, int)
        or args.generation_steps <= 0
    ):
        raise ValueError("--generation-steps must be a positive integer.")

    task, target, task_id = load_arc_example(
        args.task_json,
        args.solution_json,
        task_id=args.task_id,
        test_index=args.test_index,
    )
    prompt = serialize_arc_task(task, test_index=args.test_index)

    hash_requested = not args.skip_checkpoint_hashes
    identities: dict[str, Any] = {}
    if hash_requested or args.expected_nvarc_manifest_sha256:
        identities["nvarc"] = hash_nvarc_directory(args.nvarc_path)
        identities["nvarc"]["declared_source"] = args.nvarc_source
        identities["nvarc"]["expected_hash_verified"] = _verify_hash(
            identities["nvarc"]["manifest_sha256"],
            args.expected_nvarc_manifest_sha256,
            name="expected NVARC manifest SHA-256",
        )
    if hash_requested or args.expected_flux_sha256:
        actual = sha256_file(args.flux_path)
        identities["flux"] = {
            "path": str(args.flux_path.expanduser().resolve()),
            "sha256": actual,
            "expected_hash_verified": _verify_hash(
                actual,
                args.expected_flux_sha256,
                name="expected FLUX SHA-256",
            ),
        }
    if hash_requested or args.expected_vae_sha256:
        actual = sha256_file(args.vae_path)
        identities["vae"] = {
            "path": str(args.vae_path.expanduser().resolve()),
            "sha256": actual,
            "expected_hash_verified": _verify_hash(
                actual,
                args.expected_vae_sha256,
                name="expected VAE SHA-256",
            ),
        }

    cuda_devices = _cuda_devices(
        args.nvarc_device,
        args.flux_device,
        args.vae_device,
    )
    _reset_cuda_peaks(cuda_devices)
    memory_before_load = _memory_snapshot(cuda_devices)

    nvarc, tokenizer, nvarc_contract = load_nvarc_local(
        args.nvarc_path,
        device=args.nvarc_device,
        dtype=torch.bfloat16,
    )
    flux = load_bfl_flux2_klein_4b_local(
        args.flux_path,
        device=args.flux_device,
        dtype=torch.bfloat16,
        # The runner already verified the user-selected hash above.
        expected_sha256=None,
    )
    vae = load_bfl_autoencoder_local(
        args.vae_path,
        device=args.vae_device,
        dtype=torch.float32,
        expected_sha256=None,
    )
    model = DiffusionARCModel(
        conditioner=NVARCConditioner(nvarc),
        flux=flux,
        vae=vae,
    ).train()
    model.assert_parameter_boundaries()
    memory_after_load = _memory_snapshot(cuda_devices)

    input_ids, attention_mask = _tokenize_exact(tokenizer, prompt)
    flux_parameter = next(model.flux.parameters())
    flux_device = flux_parameter.device
    flux_dtype = flux_parameter.dtype
    generator = torch.Generator(device=flux_device).manual_seed(args.seed)
    noise = torch.randn(
        (1, ARC_IMAGE_TOKENS, ARC_LATENT_CHANNELS),
        device=flux_device,
        dtype=flux_dtype,
        generator=generator,
    )
    timesteps = torch.tensor(
        [args.flow_time],
        device=flux_device,
        dtype=flux_dtype,
    )

    model.zero_grad(set_to_none=True)
    _reset_cuda_peaks(cuda_devices)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_grids=target,
        timesteps=timesteps,
        noise=noise,
        return_details=True,
    )
    output.loss.backward()
    gradient_report = _gradient_audit(model)
    memory_after_backward = _memory_snapshot(cuda_devices)

    if (
        output.predicted_velocity is None
        or output.clean_latents is None
        or output.noisy_latents is None
    ):
        raise AssertionError("Detailed smoke-test tensors were not returned.")
    training_tensor_report = {
        "context": [1, int(input_ids.shape[1]), 7680],
        "clean_latents": list(output.clean_latents.shape),
        "noisy_latents": list(output.noisy_latents.shape),
        "predicted_velocity": list(output.predicted_velocity.shape),
        "loss_dtype": str(output.loss.dtype),
        "loss": float(output.loss.detach().item()),
        "all_reported_tensors_finite": True,
    }

    model.zero_grad(set_to_none=True)
    del output
    for device in cuda_devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    model.eval()
    _reset_cuda_peaks(cuda_devices)
    generation = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_steps=args.generation_steps,
        noise=noise,
    )
    memory_after_generation = _memory_snapshot(cuda_devices)
    decode = generation.decodes[0]
    generation_report = {
        "steps": args.generation_steps,
        "latents_shape": list(generation.latents.shape),
        "images_shape": list(generation.images.shape),
        "schedule_shape": list(generation.timesteps.shape),
        "schedule_start": float(generation.timesteps[0].item()),
        "schedule_end": float(generation.timesteps[-1].item()),
        "strict_decode": {
            "valid": decode.valid,
            "height": decode.height,
            "width": decode.width,
            "reason": decode.reason,
            "grid": (
                decode.grid.detach().cpu().tolist()
                if decode.grid is not None
                else None
            ),
        },
        "interface_only": True,
    }
    model.train()

    identity_checks = {
        name: bool(identity.get("expected_hash_verified"))
        for name, identity in identities.items()
    }
    identity_verified = (
        identity_checks.get("nvarc", False)
        and identity_checks.get("flux", False)
        and identity_checks.get("vae", False)
    )
    return {
        "status": (
            "passed"
            if identity_verified
            else "passed_with_unverified_checkpoint_identity"
        ),
        "scope": "milestone_1_complete_pipeline_smoke",
        "milestone_complete": identity_verified,
        "task_id": task_id,
        "test_index": args.test_index,
        "seed": args.seed,
        "bfl_source_commit": BFL_SOURCE_COMMIT,
        "package_versions": _package_versions(),
        "torch_cuda_version": torch.version.cuda,
        "checkpoint_identities": identities,
        "checkpoint_identity": {
            "fully_verified_against_expected_hashes": identity_verified,
            "component_checks": identity_checks,
            "required_for_final_milestone_evidence": True,
        },
        "nvarc_contract": nvarc_contract,
        "prompt": {
            "characters": len(prompt),
            "tokens": int(input_ids.shape[1]),
            "padding_tokens": int((attention_mask == 0).sum().item()),
            "test_output_excluded_by_serializer": True,
        },
        "components": {
            "nvarc": _component_manifest(model.nvarc),
            "flux": _component_manifest(model.flux),
            "vae": _component_manifest(model.vae),
        },
        "tensors": training_tensor_report,
        "generation": generation_report,
        "gradients": gradient_report,
        "memory": {
            "before_load": memory_before_load,
            "after_load": memory_after_load,
            "forward_backward_peak": memory_after_backward,
            "generation_peak": memory_after_generation,
        },
        "validated_boundaries": {
            "nvarc_frozen": not any(
                parameter.requires_grad for parameter in model.nvarc.parameters()
            ),
            "vae_frozen": not any(
                parameter.requires_grad for parameter in model.vae.parameters()
            ),
            "all_flux_parameters_trainable": all(
                parameter.requires_grad for parameter in model.flux.parameters()
            ),
            "nvarc_eval": not model.nvarc.training,
            "vae_eval": not model.vae.training,
            "conditioned_only_no_cfg": True,
        },
        "limitations": [
            "A passing run validates interfaces and gradient boundaries only.",
            (
                "The default one-step generation validates execution and strict "
                "decoding, but not Base-model sample quality."
            ),
            (
                "Milestone 1 is complete only after this report is produced "
                "with real checkpoints and verified identities on Kaggle."
            ),
            "Tiny-set overfitting and generation quality remain Milestone 2 work.",
        ],
    }


def _write_report(report: dict[str, Any], path: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if path is not None:
        resolved = path.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(rendered + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        report = run(args)
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_report(failure, args.report_path)
        raise
    _write_report(report, args.report_path)


if __name__ == "__main__":
    main()
