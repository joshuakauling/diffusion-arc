#!/usr/bin/env python3
"""Walk through scripts.train.main on CPU with explicit, tiny test fixtures.

The production config parser, runner, data loader, packing, FLUX blocks, loss,
backward, AdamW and scheduler all execute. Hardware selection, checkpoint loading,
the source dataset and frozen encoders are substituted locally. The fixture uses
FP32 CPU attention instead of the production recipe's BF16 CUDA/FlashAttention.
This is a debugger walkthrough, not a numerical or performance benchmark.
"""

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch

from scripts import train
from diffusion_arc.training import training_loop
from diffusion_arc.training.setup import TrainingModels
from tests.model.test_flux_packing import SegmentedReference
from tests.training.test_ddp_runner import Sources
from tests.training.test_ddp_trainer import FrozenText, FrozenVae, _model, _render


def write_debug_config(directory: Path) -> Path:
    """Prepare an explicit two-update recipe for the ordinary CLI to read."""
    template = ROOT / "configs/training/benchmarks/single_gpu_b3_smoke.json"
    payload = json.loads(template.read_text())
    payload["name"] = "cpu_debug_walkthrough"
    payload["models"].update(
        nvarc_checkpoint=str(directory / "synthetic-nvarc"),
        flux_checkpoint=str(directory / "tiny-flux"),
    )
    payload["data"].update(
        prepared_split_directory=str(directory / "synthetic-sources"),
        expected_task_count=2, num_workers=0, prefetch_factor=None,
    )
    payload["loop"].update(max_steps=2, optimizer_batch_size=3, vae_microbatch_size=2)
    payload["execution"]["flux_max_joint_tokens"] = 9300
    payload["flow"]["rng_scheme"] = "global_step_target_v1"
    payload["scheduler"] = {"type": "cosine", "warmup_steps": 1, "total_steps": 2}
    for service in ("benchmark", "logging", "validation", "checkpointing"):
        payload[service] = None
    path = directory / "training.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_debug_models(config, *, device, transformer_checkpoint=None):
    """Construct tiny random FLUX blocks and frozen encoder doubles without I/O."""
    return TrainingModels(
        tokenizer=object(), nvarc_encoder=FrozenText(), vae_encoder=FrozenVae(),
        vae_decoder=None, flux_transformer=_model().to(device),
        flux_compute_dtype=torch.float32, flux_packing_backend=SegmentedReference(),
    )


def main() -> int:
    """Install CPU fixtures only for this process, then enter the real training CLI."""
    print("CPU walkthrough: tiny FP32 FLUX, synthetic sources and frozen encoder doubles.\n"
          "CUDA/BF16 settings in the parsed production recipe are replaced by CPU fixtures.\n"
          "Two real optimizer updates; no downloads, logging services or checkpoints.")
    with tempfile.TemporaryDirectory(prefix="diffusion-arc-debug-") as temporary, ExitStack() as stack:
        config_path = write_debug_config(Path(temporary))
        stack.enter_context(patch.object(training_loop, "prepare_device", return_value=torch.device("cpu")))
        stack.enter_context(patch.object(training_loop, "load_models", load_debug_models))
        stack.enter_context(patch(
            "diffusion_arc.training.distributed_data.ArcTrainingDataset",
            side_effect=lambda **kwargs: Sources(source_count=2),
        ))
        stack.enter_context(patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render))
        return train.main(["--config", str(config_path)])


if __name__ == "__main__":
    raise SystemExit(main())
