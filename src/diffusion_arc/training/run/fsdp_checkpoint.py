"""Read and write local FSDP2 shards using PyTorch distributed checkpointing."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)


def export_sharded_transformer(checkpoint_path: Path, output_directory: Path) -> None:
    """Explicitly consolidate model shards into an HF export on CPU only.

    The caller budgets CPU memory and scratch disk for the complete weights.
    Optimizer shards are never loaded and the source checkpoint is unchanged.
    """
    from diffusers import Flux2Transformer2DModel
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    checkpoint_path, output_directory = Path(checkpoint_path).resolve(), Path(output_directory).resolve()
    if output_directory.exists():
        raise FileExistsError(f"transformer export already exists: {output_directory}")
    if output_directory.is_relative_to(checkpoint_path):
        raise ValueError("export directory must be outside the source checkpoint")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fsdp-export-", dir=output_directory.parent) as scratch:
        scratch = Path(scratch)
        weights = scratch / "model.pt"
        # The format utility disables distributed communication internally;
        # its empty-state planner allocates tensors on the current default device.
        with torch.device("cpu"):
            dcp_to_torch_save(checkpoint_path / "model", weights)
        state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        config = Flux2Transformer2DModel.load_config(checkpoint_path / "transformer", local_files_only=True)
        with torch.device("meta"):
            transformer = Flux2Transformer2DModel.from_config(config)
        # Assignment preserves the stored parameter precision without an extra copy.
        transformer.load_state_dict(state["model"], strict=True, assign=True)
        transformer.save_pretrained(scratch / "transformer", safe_serialization=True,
                                    max_shard_size="5GB", push_to_hub=False)
        del transformer, state
        (scratch / "transformer").rename(output_directory)


def save_sharded_state(path: Path, transformer, optimizer, process_group) -> dict:
    """Keep model shards separate so optimizer state can later be pruned."""
    if not optimizer.state:
        raise ValueError("FSDP2 checkpoint requires initialized AdamW state")
    options = StateDictOptions(full_state_dict=False)
    model_state = get_model_state_dict(transformer, options=options)
    dcp.save({"model": model_state}, checkpoint_id=path / "model", process_group=process_group)
    del model_state
    optimizer_state = get_optimizer_state_dict(transformer, optimizer, options=options)
    dcp.save({"optimizer": optimizer_state}, checkpoint_id=path / "optimizer", process_group=process_group)
    # Only small contracts go in training_state.pt; no DTensors are gathered.
    return {
        "param_groups": deepcopy(optimizer_state["param_groups"]),
        "state": {
            name: {"shape": list(moments["exp_avg"].shape), "dtype": str(moments["exp_avg"].dtype)}
            for name, moments in optimizer_state["state"].items()
        },
    }


def restore_sharded_model(path: Path, transformer, process_group) -> None:
    """Fill the model's allocated DTensor shards without materializing full weights."""
    options = StateDictOptions(full_state_dict=False, strict=True)
    state = {"model": get_model_state_dict(transformer, options=options)}
    dcp.load(state, checkpoint_id=path / "model", process_group=process_group)
    set_model_state_dict(transformer, state["model"], options=options)


def restore_sharded_optimizer(path: Path, transformer, optimizer, process_group, summary: dict) -> None:
    """Allocate fresh AdamW shard buffers and restore them without an optimizer step."""
    if not isinstance(optimizer, torch.optim.AdamW) or optimizer.state:
        raise ValueError("FSDP2 restore requires a fresh AdamW optimizer")
    names = {id(parameter): name for name, parameter in transformer.named_parameters()}
    expected_names = set(names.values())
    if set(summary["state"]) != expected_names:
        raise ValueError("FSDP2 checkpoint must contain moments for every trainable parameter")
    for saved, group in zip(summary["param_groups"], optimizer.param_groups, strict=True):
        if saved["params"] != [names[id(parameter)] for parameter in group["params"]]:
            raise ValueError("FSDP2 optimizer parameter names or order differ")
        if group.get("amsgrad"):
            raise ValueError("FSDP2 checkpoint restore requires the configured non-AMSGrad AdamW")
        for parameter in group["params"]:
            # get_optimizer_state_dict otherwise initializes AdamW with a hidden
            # zero-learning-rate step. Allocate its slots explicitly instead.
            step_device = parameter.device if group.get("capturable") or group.get("fused") else "cpu"
            optimizer.state[parameter] = {
                "step": torch.zeros((), dtype=torch.float32, device=step_device),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
    options = StateDictOptions(full_state_dict=False, strict=True)
    state = {"optimizer": get_optimizer_state_dict(transformer, optimizer, options=options)}
    dcp.load(state, checkpoint_id=path / "optimizer", process_group=process_group)
    loaded = state["optimizer"]
    if loaded["param_groups"] != summary["param_groups"] or set(loaded["state"]) != expected_names:
        raise ValueError("FSDP2 optimizer shard metadata differs from the checkpoint contract")
    set_optimizer_state_dict(transformer, optimizer, loaded, options=options)
