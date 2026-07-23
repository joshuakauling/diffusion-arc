#!/usr/bin/env python3
"""Run a deterministic FLUX.2 VAE round-trip on synthetic ARC grids."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import diffusers
import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image, ImageDraw


MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
MODEL_REVISION = "8c44a2fbef88fae175da65df054db0f901aa9747"
EXPECTED_VAE_SHA256 = "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"

GRID_SIZE = 30
ALLOWED_CELL_SIZES = (8, 16, 32)
PAD_CLASS = 10
SEED = 20260722

# ARC colors 0-9 use the common ARC visualization palette. White is reserved
# for the padding/outside class in this diagnostic.
PALETTE = np.asarray(
    [
        (0x00, 0x00, 0x00),
        (0x00, 0x74, 0xD9),
        (0xFF, 0x41, 0x36),
        (0x2E, 0xCC, 0x40),
        (0xFF, 0xDC, 0x00),
        (0xAA, 0xAA, 0xAA),
        (0xF0, 0x12, 0xBE),
        (0xFF, 0x85, 0x1B),
        (0x7F, 0xDB, 0xFF),
        (0x87, 0x0C, 0x25),
        (0xFF, 0xFF, 0xFF),
    ],
    dtype=np.uint8,
)


@dataclass
class PatternResult:
    name: str
    cell_size: int
    grid: np.ndarray
    input_rgb: np.ndarray
    reconstruction_rgb: np.ndarray
    predicted_grid: np.ndarray
    transformer_latent: np.ndarray
    metrics: dict[str, float | int | list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument(
        "--cell-size",
        type=int,
        choices=ALLOWED_CELL_SIZES,
        default=16,
        help="Rendered pixels per ARC cell; 8, 16, and 32 yield 240, 480, and 960 pixels.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to artifacts/vae-smoke/<width>x<height>.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse network access and load the VAE from the local cache.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError("This PyTorch build does not include MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available on this machine.")
    return torch.device(requested)


def make_patterns() -> dict[str, np.ndarray]:
    rows = np.arange(GRID_SIZE)[:, None]
    cols = np.arange(GRID_SIZE)[None, :]

    palette_bands = np.broadcast_to((cols * len(PALETTE) // GRID_SIZE), (GRID_SIZE, GRID_SIZE)).copy()

    isolated_cells = np.full((GRID_SIZE, GRID_SIZE), PAD_CLASS, dtype=np.int64)
    positions = [
        (2, 2), (2, 14), (2, 27), (8, 8), (8, 21), (14, 2),
        (14, 14), (14, 27), (21, 8), (21, 21), (27, 14),
    ]
    for color, (row, col) in enumerate(positions):
        isolated_cells[row, col] = color

    thin_lines = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    for color in range(1, len(PALETTE)):
        coordinate = 2 + (color - 1) * 3
        if coordinate < GRID_SIZE:
            thin_lines[coordinate, :] = color
            thin_lines[:, coordinate] = color

    checkerboard = ((rows * 3 + cols * 5) % len(PALETTE)).astype(np.int64)

    rng = np.random.default_rng(SEED)
    random_balanced = rng.permutation(
        np.resize(np.arange(len(PALETTE), dtype=np.int64), GRID_SIZE * GRID_SIZE)
    ).reshape(GRID_SIZE, GRID_SIZE)

    return {
        "palette-bands": palette_bands,
        "isolated-cells": isolated_cells,
        "thin-lines": thin_lines,
        "checkerboard": checkerboard,
        "random-balanced": random_balanced,
    }


def render_grid(grid: np.ndarray, cell_size: int) -> np.ndarray:
    assert grid.shape == (GRID_SIZE, GRID_SIZE)
    assert cell_size in ALLOWED_CELL_SIZES
    assert int(grid.min()) >= 0 and int(grid.max()) < len(PALETTE)
    cell_rgb = PALETTE[grid]
    return np.repeat(np.repeat(cell_rgb, cell_size, axis=0), cell_size, axis=1)


def image_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32).div(127.5).sub(1.0)


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    assert height % 2 == 0 and width % 2 == 0
    latents = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch, channels * 4, height // 2, width // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    assert channels % 4 == 0
    latents = latents.reshape(batch, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch, channels // 4, height * 2, width * 2)


def normalize_for_transformer(vae: AutoencoderKLFlux2, latents: torch.Tensor) -> torch.Tensor:
    patched = patchify_latents(latents)
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(patched)
    std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1).to(patched) + vae.config.batch_norm_eps
    )
    return (patched - mean) / std


def denormalize_from_transformer(
    vae: AutoencoderKLFlux2, transformer_latents: torch.Tensor
) -> torch.Tensor:
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(transformer_latents)
    std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1).to(transformer_latents)
        + vae.config.batch_norm_eps
    )
    return unpatchify_latents(transformer_latents * std + mean)


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu()[0].permute(1, 2, 0).numpy()
    return np.clip((array + 1.0) * 127.5, 0.0, 255.0).round().astype(np.uint8)


def classify_cells(
    reconstruction_rgb: np.ndarray, cell_size: int
) -> tuple[np.ndarray, np.ndarray]:
    image_size = GRID_SIZE * cell_size
    assert reconstruction_rgb.shape == (image_size, image_size, 3)
    cells = reconstruction_rgb.reshape(
        GRID_SIZE, cell_size, GRID_SIZE, cell_size, 3
    ).mean(axis=(1, 3))
    distances = ((cells[:, :, None, :] - PALETTE[None, None, :, :]) ** 2).sum(axis=-1)
    return distances.argmin(axis=-1).astype(np.int64), cells


def compute_metrics(
    expected_grid: np.ndarray,
    predicted_grid: np.ndarray,
    input_rgb: np.ndarray,
    reconstruction_rgb: np.ndarray,
) -> dict[str, float | int | list[int]]:
    input_float = input_rgb.astype(np.float32) / 255.0
    reconstruction_float = reconstruction_rgb.astype(np.float32) / 255.0
    error = reconstruction_float - input_float
    mse = float(np.mean(error**2))
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    correct = expected_grid == predicted_grid
    return {
        "correct_cells": int(correct.sum()),
        "total_cells": int(correct.size),
        "cell_accuracy": float(correct.mean()),
        "exact_grid": int(bool(correct.all())),
        "pixel_mae": float(np.mean(np.abs(error))),
        "pixel_mse": mse,
        "pixel_psnr_db": psnr,
        "wrong_cell_indices": np.flatnonzero(~correct).astype(int).tolist(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_cached_weights(cache_dir: Path) -> Path:
    candidates = sorted(cache_dir.rglob("vae/diffusion_pytorch_model.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"Could not locate cached VAE weights below {cache_dir}")
    return candidates[-1]


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def run_pattern(
    name: str,
    grid: np.ndarray,
    cell_size: int,
    vae: AutoencoderKLFlux2,
    device: torch.device,
) -> PatternResult:
    image_size = GRID_SIZE * cell_size
    vae_latent_size = image_size // 8
    transformer_latent_size = image_size // 16
    assert image_size % 16 == 0

    input_rgb = render_grid(grid, cell_size)
    sample = image_tensor(input_rgb, device)
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        posterior = vae.encode(sample).latent_dist
        vae_latents = posterior.mode()
        transformer_latents = normalize_for_transformer(vae, vae_latents)
        recovered_vae_latents = denormalize_from_transformer(vae, transformer_latents)
        reconstruction = vae.decode(recovered_vae_latents).sample
    synchronize(device)
    duration = time.perf_counter() - started

    assert tuple(sample.shape) == (1, 3, image_size, image_size)
    assert tuple(vae_latents.shape) == (1, 32, vae_latent_size, vae_latent_size)
    assert tuple(transformer_latents.shape) == (
        1,
        128,
        transformer_latent_size,
        transformer_latent_size,
    )
    assert tuple(reconstruction.shape) == tuple(sample.shape)
    for tensor_name, tensor in {
        "sample": sample,
        "vae_latents": vae_latents,
        "transformer_latents": transformer_latents,
        "reconstruction": reconstruction,
    }.items():
        if not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError(f"Non-finite values in {tensor_name}")

    reconstruction_rgb = tensor_to_rgb(reconstruction)
    predicted_grid, _ = classify_cells(reconstruction_rgb, cell_size)
    metrics = compute_metrics(grid, predicted_grid, input_rgb, reconstruction_rgb)
    metrics["duration_seconds"] = float(duration)
    metrics["latent_roundtrip_max_abs_error"] = float(
        (recovered_vae_latents - vae_latents).abs().max().item()
    )
    return PatternResult(
        name=name,
        cell_size=cell_size,
        grid=grid,
        input_rgb=input_rgb,
        reconstruction_rgb=reconstruction_rgb,
        predicted_grid=predicted_grid,
        transformer_latent=transformer_latents.detach().float().cpu()[0].numpy(),
        metrics=metrics,
    )


def deterministic_encode_error(
    vae: AutoencoderKLFlux2, rgb: np.ndarray, device: torch.device
) -> float:
    sample = image_tensor(rgb, device)
    with torch.inference_mode():
        first = vae.encode(sample).latent_dist.mode()
        second = vae.encode(sample).latent_dist.mode()
    return float((first - second).abs().max().item())


def confusion_matrix(results: list[PatternResult]) -> np.ndarray:
    expected = np.concatenate([result.grid.ravel() for result in results])
    predicted = np.concatenate([result.predicted_grid.ravel() for result in results])
    matrix = np.zeros((len(PALETTE), len(PALETTE)), dtype=np.int64)
    np.add.at(matrix, (expected, predicted), 1)
    return matrix


def labeled_panel(rgb: np.ndarray, label: str, width: int | None = None) -> Image.Image:
    image = Image.fromarray(rgb)
    if width is not None and width != image.width:
        height = round(image.height * width / image.width)
        image = image.resize((width, height), Image.Resampling.NEAREST)
    label_height = 28
    panel = Image.new("RGB", (image.width, image.height + label_height), "white")
    panel.paste(image, (0, label_height))
    ImageDraw.Draw(panel).text((8, 8), label, fill="black")
    return panel


def quantized_with_error_boxes(result: PatternResult) -> np.ndarray:
    rgb = render_grid(result.predicted_grid, result.cell_size)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    wrong = np.argwhere(result.grid != result.predicted_grid)
    for row, col in wrong:
        left = int(col) * result.cell_size
        top = int(row) * result.cell_size
        draw.rectangle(
            (
                left,
                top,
                left + result.cell_size - 1,
                top + result.cell_size - 1,
            ),
            outline=(255, 0, 0),
            width=max(1, result.cell_size // 5),
        )
    return np.asarray(image)


def absolute_difference(result: PatternResult) -> np.ndarray:
    difference = np.abs(
        result.reconstruction_rgb.astype(np.int16) - result.input_rgb.astype(np.int16)
    )
    return np.clip(difference * 4, 0, 255).astype(np.uint8)


def comparison_panels(result: PatternResult, panel_width: int | None = None) -> list[Image.Image]:
    accuracy = 100.0 * float(result.metrics["cell_accuracy"])
    return [
        labeled_panel(result.input_rgb, f"{result.name}: input", panel_width),
        labeled_panel(result.reconstruction_rgb, "VAE reconstruction", panel_width),
        labeled_panel(
            quantized_with_error_boxes(result),
            f"nearest palette ({accuracy:.2f}% cells)",
            panel_width,
        ),
        labeled_panel(absolute_difference(result), "absolute RGB error (4x)", panel_width),
    ]


def join_images(images: list[Image.Image], columns: int, gap: int = 10) -> Image.Image:
    rows = math.ceil(len(images) / columns)
    widths = [image.width for image in images]
    heights = [image.height for image in images]
    cell_width = max(widths)
    cell_height = max(heights)
    canvas = Image.new(
        "RGB",
        (columns * cell_width + (columns - 1) * gap, rows * cell_height + (rows - 1) * gap),
        (224, 224, 224),
    )
    for index, image in enumerate(images):
        row, col = divmod(index, columns)
        canvas.paste(image, (col * (cell_width + gap), row * (cell_height + gap)))
    return canvas


def save_comparisons(results: list[PatternResult], output_dir: Path) -> None:
    overview_panels: list[Image.Image] = []
    for result in results:
        full_panels = comparison_panels(result)
        join_images(full_panels, columns=4).save(output_dir / f"{result.name}.png")
        overview_panels.extend(comparison_panels(result, panel_width=240))
    join_images(overview_panels, columns=4).save(output_dir / "overview.png")


def latent_pca_images(results: list[PatternResult]) -> list[np.ndarray]:
    token_sets = [
        result.transformer_latent.transpose(1, 2, 0).reshape(-1, 128) for result in results
    ]
    tokens = np.concatenate(token_sets, axis=0).astype(np.float64)
    centered = tokens - tokens.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    components = eigenvectors[:, np.argsort(eigenvalues)[-3:][::-1]]
    projected = centered @ components
    low = np.percentile(projected, 1.0, axis=0)
    high = np.percentile(projected, 99.0, axis=0)
    scaled = np.clip((projected - low) / np.maximum(high - low, 1e-12), 0.0, 1.0)

    images: list[np.ndarray] = []
    offset = 0
    for token_set in token_sets:
        count = token_set.shape[0]
        result = results[len(images)]
        _, latent_height, latent_width = result.transformer_latent.shape
        rgb = (
            scaled[offset : offset + count].reshape(latent_height, latent_width, 3) * 255
        ).astype(np.uint8)
        scale_h = result.input_rgb.shape[0] // latent_height
        scale_w = result.input_rgb.shape[1] // latent_width
        assert scale_h == 16 and scale_w == 16
        images.append(np.repeat(np.repeat(rgb, scale_h, axis=0), scale_w, axis=1))
        offset += count
    return images


def save_latent_visualization(results: list[PatternResult], output_dir: Path) -> None:
    images = latent_pca_images(results)
    panels = [
        labeled_panel(rgb, f"{result.name}: latent PCA", width=320)
        for result, rgb in zip(results, images)
    ]
    join_images(panels, columns=2).save(output_dir / "latent-pca.png")


def environment_metadata(device: torch.device) -> dict[str, object]:
    metadata: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "numpy": np.__version__,
        "pillow": importlib.metadata.version("Pillow"),
        "safetensors": importlib.metadata.version("safetensors"),
        "device": str(device),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    if device.type == "mps" and hasattr(torch.backends.mps, "get_name"):
        metadata["mps_name"] = torch.backends.mps.get_name()
    return metadata


def main() -> None:
    args = parse_args()
    image_size = GRID_SIZE * args.cell_size
    vae_latent_size = image_size // 8
    transformer_latent_size = image_size // 16
    if args.output_dir is None:
        args.output_dir = Path(f"artifacts/vae-smoke/{image_size}x{image_size}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print(f"Loading {MODEL_ID} VAE at revision {MODEL_REVISION} on {device}...")
    vae = AutoencoderKLFlux2.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        revision=MODEL_REVISION,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)

    if vae.training:
        raise AssertionError("VAE must be in eval mode.")
    if any(parameter.requires_grad for parameter in vae.parameters()):
        raise AssertionError("Every VAE parameter must be frozen.")
    if next(vae.parameters()).dtype != torch.float32:
        raise AssertionError("The smoke test requires FP32 VAE weights.")

    weight_path = locate_cached_weights(args.cache_dir)
    weight_sha256 = sha256_file(weight_path)
    if weight_sha256 != EXPECTED_VAE_SHA256:
        raise RuntimeError(
            f"Unexpected VAE SHA256: {weight_sha256}; expected {EXPECTED_VAE_SHA256}"
        )

    patterns = make_patterns()
    results: list[PatternResult] = []
    for name, grid in patterns.items():
        print(f"Running {name} at {image_size}x{image_size}...")
        results.append(run_pattern(name, grid, args.cell_size, vae, device))
        if device.type == "mps":
            torch.mps.empty_cache()

    determinism_error = deterministic_encode_error(vae, results[0].input_rgb, device)
    if determinism_error > 1e-6:
        raise AssertionError(f"Encode is not deterministic: max error {determinism_error}")

    matrix = confusion_matrix(results)
    total_correct = int(np.trace(matrix))
    total_cells = int(matrix.sum())
    save_comparisons(results, args.output_dir)
    save_latent_visualization(results, args.output_dir)

    report = {
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "subfolder": "vae",
            "class": type(vae).__name__,
            "weight_file": str(weight_path.resolve()),
            "weight_bytes": weight_path.stat().st_size,
            "weight_sha256": weight_sha256,
            "parameter_count": sum(parameter.numel() for parameter in vae.parameters()),
            "frozen": not any(parameter.requires_grad for parameter in vae.parameters()),
            "eval_mode": not vae.training,
        },
        "environment": environment_metadata(device),
        "command": [sys.executable, *sys.argv],
        "configuration": {
            "seed": SEED,
            "grid_size": GRID_SIZE,
            "cell_size": args.cell_size,
            "image_size": image_size,
            "pad_class": PAD_CLASS,
            "pad_rgb": PALETTE[PAD_CLASS].astype(int).tolist(),
            "input_shape": [1, 3, image_size, image_size],
            "vae_latent_shape": [1, 32, vae_latent_size, vae_latent_size],
            "transformer_latent_shape": [
                1,
                128,
                transformer_latent_size,
                transformer_latent_size,
            ],
            "dtype": "float32",
            "posterior_statistic": "mode",
        },
        "determinism_max_abs_error": determinism_error,
        "aggregate": {
            "correct_cells": total_correct,
            "total_cells": total_cells,
            "cell_accuracy": total_correct / total_cells,
            "exact_grids": sum(int(result.metrics["exact_grid"]) for result in results),
            "total_grids": len(results),
            "confusion_matrix": matrix.tolist(),
        },
        "patterns": {result.name: result.metrics for result in results},
        "artifacts": [
            "overview.png",
            "latent-pca.png",
            *[f"{result.name}.png" for result in results],
        ],
    }
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["aggregate"], indent=2))
    print(f"Metrics: {report_path}")


if __name__ == "__main__":
    main()
