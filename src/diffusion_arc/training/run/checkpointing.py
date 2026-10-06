"""Atomic full-state checkpoints at completed single-GPU, DDP or FSDP2 updates."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..config import TrainingConfig

if TYPE_CHECKING:
    from ..runner import RunProgress


@dataclass(frozen=True)
class Checkpoint:
    """Verified on-disk inventory and metadata; large optimizer tensors load later."""

    path: Path
    metadata: dict


def normalize_checkpoint_recipe(recipe: dict) -> dict:
    """Migrate historical saved recipes; never complete a new user configuration."""
    recipe = deepcopy(recipe)
    schema = recipe.get("schema_version")
    if schema == 2:
        return recipe
    if schema != 1:
        raise ValueError("unsupported saved training recipe schema")
    recipe["schema_version"] = 2
    recipe["data"]["source"] = "epoch_pool"
    recipe["benchmark"] = None
    targets = (recipe["data"]["expected_task_count"]
               * (recipe["data"]["pairs_per_sequence"] - recipe["loop"]["min_context_pairs_before_target"]))
    batch = recipe["loop"]["optimizer_batch_size"]
    expected = {"max_optimizer_steps": (targets + batch - 1) // batch,
                "expected_target_count": targets,
                "expected_final_batch_target_count": (targets - 1) % batch + 1}
    if any(recipe["loop"].get(name) != value for name, value in expected.items()):
        raise ValueError("historical saved recipe has inconsistent epoch counts")
    for name in expected:
        recipe["loop"].pop(name)
    recipe["loop"].update(epochs=1, max_steps=None)
    recipe["execution"].setdefault("flux_max_spread_tokens", 1_000)
    recipe["execution"].setdefault("flux_packing_algorithm", "legacy_bfd_v1")
    recipe["flow"].setdefault("rng_scheme", "legacy_stream_v1")
    return recipe


def load_saved_training_config(path: Path) -> TrainingConfig:
    """Read an existing run receipt, explicitly migrating its historical recipe."""
    from ..config import parse_training_config

    path = Path(path).resolve()
    payload = normalize_checkpoint_recipe(json.loads(path.read_text()))
    return parse_training_config(payload, config_path=path)


def checkpoint_identities_match(saved: dict, current: dict) -> bool:
    """Compare complete identities with the explicit legacy recipe normalization."""
    return ({**saved, "recipe": normalize_checkpoint_recipe(saved["recipe"])}
            == {**current, "recipe": normalize_checkpoint_recipe(current["recipe"])})


def _config_execution(config: TrainingConfig) -> dict:
    return {"strategy": config.recipe["execution"]["distributed_strategy"],
            "world_size": config.recipe["deployment"]["gpu_count"]}


def _checkpoint_execution(metadata: dict) -> dict:
    execution = metadata.get("execution")
    if execution is None and metadata["schema_version"] == 1:
        execution = {"strategy": "single_process", "world_size": 1}
    if (not isinstance(execution, dict) or set(execution) != {"strategy", "world_size"}
            or execution["strategy"] not in ("single_process", "ddp", "fsdp2")
            or type(execution["world_size"]) is not int or execution["world_size"] < 1
            or (execution["strategy"] == "single_process" and execution["world_size"] != 1)):
        raise ValueError("checkpoint execution strategy or world size is invalid")
    return execution


def _runtime_execution(runtime) -> dict:
    return ({"strategy": "single_process", "world_size": 1} if runtime is None
            else {"strategy": runtime.strategy, "world_size": runtime.world_size})


def checkpoint_identity(config: TrainingConfig, *, data_fingerprint: str, validation_manifest: dict | None) -> dict:
    """Bind continuation to the recipe, ordered training content and validation split.

    Deployment and logging may change when restarting. Model/data paths remain
    part of the contract; referenced frozen model files must remain immutable.
    """
    recipe = config.recipe_dict()
    for name in ("deployment", "logging", "checkpointing", "name"):
        recipe.pop(name)
    recipe["models"].pop("resume_from")
    encoded = json.dumps(validation_manifest, sort_keys=True, separators=(",", ":")).encode()
    return {"recipe": recipe, "data_sha256": data_fingerprint,
            "validation_sha256": hashlib.sha256(encoded).hexdigest()}


def load_checkpoint_metadata(path: Path, config: TrainingConfig, *, require_training_state: bool = True) -> Checkpoint:
    """Reject incomplete inventories, incompatible recipes and invalid data cursors.

    File sizes detect missing/truncated writes, not arbitrary same-size bit rot.
    Diffusers and PyTorch validate their serialized payloads when loading them.
    """
    path = Path(path).resolve()
    metadata = json.loads((path / "checkpoint.json").read_text())
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] not in (1, 2, 3):
        raise ValueError("unsupported checkpoint schema")
    execution = _checkpoint_execution(metadata)
    if execution != _config_execution(config):
        raise ValueError("checkpoint execution strategy or world size differs from this run")
    saved_recipe = metadata["recipe"]
    if execution != {"strategy": saved_recipe["execution"]["distributed_strategy"],
                     "world_size": saved_recipe["deployment"]["gpu_count"]}:
        raise ValueError("checkpoint execution metadata differs from its recipe")
    weights_only = metadata.get("weights_only", False)
    if type(weights_only) is not bool:
        raise ValueError("checkpoint weights_only must be boolean")
    if weights_only and require_training_state:
        raise ValueError("checkpoint retains model weights only; its training state was removed")
    inventory = metadata["files"]
    required = {"transformer/config.json"} | (set() if weights_only else {"training_state.pt"})
    if execution["strategy"] == "fsdp2":
        if metadata["schema_version"] != 3:
            raise ValueError("FSDP2 requires distributed checkpoint schema 3")
        required.add("model/.metadata")
        if not weights_only:
            required.add("optimizer/.metadata")
        elif any(name.startswith("optimizer/") for name in inventory):
            raise ValueError("weights-only checkpoint must not inventory optimizer shards")
        has_weights = any(name.startswith("model/") and name.endswith(".distcp") for name in inventory)
        if not weights_only and not any(name.startswith("optimizer/") and name.endswith(".distcp")
                                        for name in inventory):
            raise ValueError("checkpoint optimizer shards are missing")
    else:
        has_weights = any(name.startswith("transformer/") and name.endswith(".safetensors")
                          for name in inventory)
    if weights_only and "training_state.pt" in inventory:
        raise ValueError("weights-only checkpoint must not inventory a training state")
    actual = {str(file.relative_to(path)) for file in path.rglob("*") if file.is_file()}
    if (not required <= inventory.keys() or actual != set(inventory) | {"checkpoint.json"}
            or not has_weights):
        raise ValueError("incomplete checkpoint inventory")
    for name, size in inventory.items():
        file = path / name
        if not file.resolve().is_relative_to(path) or file.is_symlink():
            raise ValueError("checkpoint inventory must contain local regular files")
        if type(size) is not int or size <= 0 or file.stat().st_size != size:
            raise ValueError(f"checkpoint file size differs: {name}")
    saved = metadata["identity"]
    current = checkpoint_identity(config, data_fingerprint=saved["data_sha256"], validation_manifest={})
    if normalize_checkpoint_recipe(saved["recipe"]) != normalize_checkpoint_recipe(current["recipe"]):
        raise ValueError("checkpoint training recipe differs from this run")
    progress = metadata["progress"]
    step = progress["optimizer_step"]
    epoch = progress.get("epoch", 1)
    if type(epoch) is not int or epoch < 1:
        raise ValueError("checkpoint epoch must be a positive integer")
    epoch_steps = config.epoch_optimizer_steps
    previous_steps = (epoch - 1) * epoch_steps
    optimizer_start = progress.get("optimizer_start_step", 0)
    expected_start = config.recipe["models"].get("completed_epochs", 0) * epoch_steps
    if type(optimizer_start) is not int or optimizer_start != expected_start or optimizer_start > previous_steps:
        raise ValueError("checkpoint optimizer start differs from its initialization recipe")
    if type(step) is not int or not previous_steps < step <= previous_steps + config.loop.max_optimizer_steps:
        raise ValueError("checkpoint optimizer step is outside this epoch")
    epoch_targets = config.recipe["data"]["expected_task_count"] * config.targets_per_sequence
    expected_targets = ((epoch - 1) * epoch_targets
                        + min((step - previous_steps) * config.loop.optimizer_batch_size, epoch_targets))
    if (type(progress["targets_seen"]) is not int or type(progress["sources_seen"]) is not int
            or progress["targets_seen"] != expected_targets
            or progress["sources_seen"] * config.targets_per_sequence != expected_targets):
        raise ValueError("checkpoint data cursor is not a completed optimizer batch")
    if any(type(value) is not int or value < 0 for value in progress["workload_totals"].values()):
        raise ValueError("checkpoint workload counters must be nonnegative integers")
    run = metadata["wandb_run"]
    if run is None:
        return Checkpoint(path, metadata)
    if not isinstance(run, dict) or not all(isinstance(run[name], str) and run[name] for name in ("id", "project", "entity")):
        raise ValueError("checkpoint must identify its W&B run")
    if not run["id"].isascii() or not all(char.isalnum() or char in "-_" for char in run["id"]):
        raise ValueError("checkpoint W&B run ID is not a directory name")
    return Checkpoint(path, metadata)


def remove_training_state(checkpoint: Checkpoint) -> int:
    """Turn a verified full checkpoint into retained weights; caller must hold the workspace lock.

    Move the state out of the inventory before publishing the weights-only marker.
    If publication fails, restore it. Only delete the state after publication;
    model files and their HF or distributed layout are never changed.
    """
    path = checkpoint.path
    metadata_path = path / "checkpoint.json"
    if json.loads(metadata_path.read_text()) != checkpoint.metadata:
        raise ValueError("checkpoint metadata changed since verification")
    if checkpoint.metadata.get("weights_only", False):
        return 0
    if _checkpoint_execution(checkpoint.metadata)["strategy"] == "fsdp2":
        return _remove_sharded_training_state(checkpoint)
    state = path / "training_state.pt"
    size = checkpoint.metadata["files"][state.name]
    if state.is_symlink() or state.stat().st_size != size:
        raise ValueError("checkpoint training state changed since verification")
    pending_state = path.parent / f".{path.name}-training-state-removal.pt"
    pending_metadata = path.parent / f".{path.name}-weights-only.json"
    if pending_state.exists() or pending_metadata.exists():
        raise FileExistsError("an earlier checkpoint cleanup needs inspection")
    metadata = dict(checkpoint.metadata, weights_only=True,
                    files={name: size for name, size in checkpoint.metadata["files"].items()
                           if name != "training_state.pt"})
    try:
        with pending_metadata.open("x") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        state.rename(pending_state)
        try:
            pending_metadata.replace(metadata_path)
        except BaseException:
            pending_state.rename(state)
            raise
    finally:
        pending_metadata.unlink(missing_ok=True)
    pending_state.unlink()
    return size


def _remove_sharded_training_state(checkpoint: Checkpoint) -> int:
    """Prune optimizer shards and the sidecar while retaining distributed weights."""
    path = checkpoint.path
    removed = {name: size for name, size in checkpoint.metadata["files"].items()
               if name == "training_state.pt" or name.startswith("optimizer/")}
    for name, size in removed.items():
        file = path / name
        if file.is_symlink() or file.stat().st_size != size:
            raise ValueError("checkpoint training state changed since verification")
    pending = path.parent / f".{path.name}-training-state-removal"
    pending.mkdir()  # An interrupted cleanup requires inspection before retrying.
    metadata = dict(checkpoint.metadata, weights_only=True,
                    files={name: size for name, size in checkpoint.metadata["files"].items()
                           if name not in removed})
    moved = []
    try:
        with (pending / "checkpoint.json").open("x") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for name in ("training_state.pt", "optimizer"):
            (path / name).rename(pending / name)
            moved.append(name)
        (pending / "checkpoint.json").replace(path / "checkpoint.json")
    except BaseException:
        for name in reversed(moved):
            (pending / name).rename(path / name)
        (pending / "checkpoint.json").unlink(missing_ok=True)
        pending.rmdir()
        raise
    shutil.rmtree(pending)
    return sum(removed.values())


def validate_epoch_continuation(checkpoint: Checkpoint, config: TrainingConfig) -> None:
    """Allow a completed epoch to feed a new augmentation view of the same experiment."""
    if _checkpoint_execution(checkpoint.metadata) != _config_execution(config):
        raise ValueError("checkpoint execution strategy or world size differs from this run")
    previous = normalize_checkpoint_recipe(checkpoint.metadata["identity"]["recipe"])
    current = normalize_checkpoint_recipe(
        checkpoint_identity(config, data_fingerprint="", validation_manifest={})["recipe"])
    if previous.pop("seed") == current.pop("seed"):
        raise ValueError("the next epoch requires a different augmentation seed")
    if previous["data"].pop("prepared_split_directory") == current["data"].pop("prepared_split_directory"):
        raise ValueError("the next epoch requires a separate prepared data directory")
    if previous != current:
        raise ValueError("only the augmentation seed and prepared view may change between epochs")
    progress = checkpoint.metadata["progress"]
    if (config.recipe["data"]["source"] != "epoch_pool"
            or progress["optimizer_step"] != progress.get("epoch", 1) * config.epoch_optimizer_steps):
        raise ValueError("finish the previous epoch before starting the next one")


def _capture_rng(flow_generator: torch.Generator, device: torch.device) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(), "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "flow": flow_generator.get_state(), "flow_device_type": flow_generator.device.type,
        # Object collectives cannot pickle UInt32Storage on every supported Torch
        # build. Signed int64 preserves every NumPy uint32 key exactly.
        "numpy": (numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)), *numpy_state[2:]),
    }


def _restore_rng(state: dict, flow_generator: torch.Generator, device: torch.device) -> None:
    if state["flow_device_type"] != flow_generator.device.type or (state["torch_cuda"] is None) != (device.type != "cuda"):
        raise ValueError("checkpoint RNG device type differs from this run")
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"], device)
    flow_generator.set_state(state["flow"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype(np.uint32), *numpy_state[2:]))


def restore_training_state(checkpoint: Checkpoint, *, transformer, optimizer, scheduler,
                           flow_generator: torch.Generator, device: torch.device, runtime=None) -> None:
    """Restore AdamW/LR and RNGs after model loading and frozen-cache preparation.

    Single/DDP weights have already been loaded by load_models. FSDP2 loads its
    model and optimizer shards here, after fully_shard and optimizer creation.
    No pending gradients are saved; continuation starts with the
    next complete optimizer batch. Load scheduler state after constructing it,
    then optimizer state to retain the next update's stored learning rate.
    """
    execution = _checkpoint_execution(checkpoint.metadata)
    if execution != _runtime_execution(runtime):
        raise ValueError("checkpoint execution strategy or world size differs from this runtime")
    state = torch.load(checkpoint.path / "training_state.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not {"parameter_names", "optimizer", "scheduler"} <= state.keys():
        raise ValueError("checkpoint training state is incomplete")
    sharded = execution["strategy"] == "fsdp2"
    if execution["strategy"] in ("ddp", "fsdp2"):
        states = state.get("rng_by_rank")
        if not isinstance(states, list) or len(states) != execution["world_size"]:
            raise ValueError("checkpoint rank RNG states are incomplete")
        rng = states[runtime.rank]
    else:
        rng = state.get("rng")
        states = [rng]
    if any(not isinstance(item, dict) or set(item) != {
            "python", "torch_cpu", "torch_cuda", "flow", "flow_device_type", "numpy"} for item in states):
        raise ValueError("checkpoint RNG state is incomplete")
    if (rng["flow_device_type"] != flow_generator.device.type
            or (rng["torch_cuda"] is None) != (device.type != "cuda")):
        raise ValueError("checkpoint RNG device type differs from this run")
    if state["parameter_names"] != [name for name, _ in transformer.named_parameters()]:
        raise ValueError("checkpoint optimizer parameter order differs from the transformer")
    progress = checkpoint.metadata["progress"]
    step = progress["optimizer_step"] - progress.get("optimizer_start_step", 0)
    stored_scheduler = state["scheduler"]
    if (stored_scheduler is None) != (scheduler is None):
        raise ValueError("checkpoint scheduler differs from the recipe")
    groups = state["optimizer"]["param_groups"]
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("checkpoint optimizer parameter groups differ")
    if scheduler is None:
        expected_lrs = [group["lr"] for group in optimizer.param_groups]
    else:
        if stored_scheduler["last_epoch"] != step:
            raise ValueError("checkpoint scheduler step differs from optimizer progress")
        progress_fields = {"last_epoch", "_step_count", "_last_lr", "_get_lr_called_within_step"}
        current_settings = {key: value for key, value in scheduler.state_dict().items()
                            if key not in progress_fields}
        saved_settings = {key: value for key, value in stored_scheduler.items()
                          if key not in progress_fields}
        if saved_settings != current_settings:
            raise ValueError("checkpoint scheduler settings differ from the recipe")
        # Evaluate the configured schedule at the saved step without advancing
        # either the live scheduler or its optimizer.
        probe = copy(scheduler)
        probe.load_state_dict(deepcopy(stored_scheduler))
        probe._get_lr_called_within_step = True
        expected_lrs = probe.get_lr()
        if stored_scheduler["_last_lr"] != expected_lrs:
            raise ValueError("checkpoint learning rate differs from scheduler progress")
    if [group["lr"] for group in groups] != expected_lrs:
        raise ValueError("checkpoint learning rate differs from scheduler progress")
    if set(state["optimizer"]["state"]) != set(checkpoint.metadata["optimizer_state_keys"]):
        raise ValueError("checkpoint optimizer states are incomplete")
    for stored, current in zip(groups, optimizer.param_groups, strict=True):
        if len(stored["params"]) != len(current["params"]):
            raise ValueError("checkpoint optimizer parameter counts differ")
        if any(stored[name] != value for name, value in current.items() if name not in ("params", "lr")):
            raise ValueError("checkpoint optimizer settings differ from the recipe")
        for key, parameter in zip(stored["params"], current["params"], strict=True):
            moments = state["optimizer"]["state"].get(key)
            if moments is not None:
                if sharded:
                    if (moments.get("shape") != list(parameter.shape)
                            or moments.get("dtype") != str(parameter.dtype)):
                        raise ValueError("checkpoint optimizer moment shape or dtype differs")
                    continue
                if not {"exp_avg", "exp_avg_sq", "step"} <= moments.keys():
                    raise ValueError("checkpoint optimizer moments are incomplete")
                if any(moments[name].shape != parameter.shape or moments[name].dtype != parameter.dtype
                       for name in ("exp_avg", "exp_avg_sq")):
                    raise ValueError("checkpoint optimizer moment shape or dtype differs")
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if sharded:
        from .fsdp_checkpoint import restore_sharded_optimizer

        restore_model_weights(checkpoint, transformer=transformer, runtime=runtime)
        restore_sharded_optimizer(checkpoint.path, transformer, optimizer, runtime.process_group,
                                  state["optimizer"])
    else:
        optimizer.load_state_dict(state["optimizer"])
    optimizer.zero_grad(set_to_none=execution["strategy"] != "ddp")
    _restore_rng(rng, flow_generator, device)


def restore_model_weights(checkpoint: Checkpoint, *, transformer, runtime) -> None:
    """Collectively restore FSDP2 weights after constructing and sharding the model."""
    from .fsdp_checkpoint import restore_sharded_model

    if (_checkpoint_execution(checkpoint.metadata) != _runtime_execution(runtime)
            or runtime.strategy != "fsdp2"):
        raise ValueError("sharded model restore requires the checkpoint's FSDP2 runtime")
    restore_sharded_model(checkpoint.path, transformer, runtime.process_group)


def save_checkpoint(progress: RunProgress, *, config: TrainingConfig, transformer, optimizer,
                    scheduler, flow_generator: torch.Generator, device: torch.device,
                    identity: dict, wandb_run: dict | None, runtime=None,
                    completed_update: bool = False) -> dict[str, float | int | str]:
    """Publish one full checkpoint after all writes finish, then apply retention.

    HF safetensors (single/DDP) or distributed tensor shards (FSDP2) store only
    the trainable FLUX transformer. Frozen NVARC/VAE weights and CPU evaluation
    caches are referenced by the recipe.
    Checkpoints live in a per-W&B-run directory, or under "local" without W&B;
    retention only touches completed checkpoints of that run. Failed temporary writes never replace a good save.

    Every distributed rank enters this function. The caller explicitly confirms a
    completed update because DDP retains zeroed gradient bucket views. No
    gradient tensor scan is needed to re-prove that training-loop boundary.
    """
    execution = _runtime_execution(runtime)
    if execution != _config_execution(config):
        raise ValueError("checkpoint execution strategy or world size differs from the recipe")
    error = None
    if execution["strategy"] in ("ddp", "fsdp2"):
        if completed_update is not True:
            error = f"{execution['strategy'].upper()} checkpoint requires an explicitly completed optimizer update"
    elif any(parameter.grad is not None for parameter in transformer.parameters()):
        error = "checkpoint requires cleared gradients at an optimizer boundary"
    if scheduler is not None and scheduler.last_epoch != progress.optimizer_step - progress.optimizer_start_step:
        error = "checkpoint scheduler is not at the completed optimizer step"
    rng = _capture_rng(flow_generator, device)
    try:
        gathered = ([{"error": error, "rng": rng}] if runtime is None
                    else runtime.all_gather_objects({"error": error, "rng": rng}))
        errors = [f"rank {rank}: {item['error']}" for rank, item in enumerate(gathered) if item["error"]]
        if errors:
            raise ValueError("; ".join(errors))
        if len(gathered) != execution["world_size"]:
            raise ValueError("checkpoint rank RNG states are incomplete")
        rng_payload = ({"rng_by_rank": [item["rng"] for item in gathered]}
                       if execution["strategy"] in ("ddp", "fsdp2") else {"rng": rng})

        if execution["strategy"] == "fsdp2":
            from .fsdp_checkpoint import save_sharded_state

            def prepare_directory():
                directory = config.recipe["deployment"]["output_directory"] / "checkpoints" / (wandb_run["id"] if wandb_run else "local")
                directory.mkdir(parents=True, exist_ok=True)
                destination = directory / f"step-{progress.optimizer_step:06d}"
                if destination.exists():
                    raise FileExistsError(f"checkpoint already exists: {destination}")
                return str(Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=directory)))

            started = time.perf_counter()
            temporary = Path(runtime.run_on_primary_and_broadcast(prepare_directory))
            # Distributed writes finish before rank zero publishes the directory.
            # Failed collective writes remain hidden for diagnosis; never publish them.
            optimizer_summary = save_sharded_state(temporary, transformer, optimizer, runtime.process_group)
            return runtime.run_on_primary_and_broadcast(lambda: _publish_checkpoint(
                progress, config=config, transformer=transformer, optimizer=optimizer,
                scheduler=scheduler, device=device, identity=identity, wandb_run=wandb_run,
                execution=execution, rng_payload=rng_payload, prepared_directory=temporary,
                optimizer_summary=optimizer_summary, started=started))

        def publish():
            return _publish_checkpoint(progress, config=config, transformer=transformer,
                                       optimizer=optimizer, scheduler=scheduler, device=device,
                                       identity=identity, wandb_run=wandb_run,
                                       execution=execution, rng_payload=rng_payload)

        return publish() if runtime is None else runtime.run_on_primary_and_broadcast(publish)
    finally:
        _restore_rng(rng, flow_generator, device)


def _publish_checkpoint(progress, *, config, transformer, optimizer, scheduler, device,
                        identity, wandb_run, execution, rng_payload, prepared_directory=None,
                        optimizer_summary=None, started=None):
    """Publish shared metadata after local HF or collective FSDP2 weight writes."""
    started = time.perf_counter() if started is None else started
    directory = config.recipe["deployment"]["output_directory"] / "checkpoints" / (wandb_run["id"] if wandb_run else "local")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"step-{progress.optimizer_step:06d}"
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    temporary = (Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=directory))
                 if prepared_directory is None else prepared_directory)
    try:
        if prepared_directory is None:
            transformer.save_pretrained(temporary / "transformer", safe_serialization=True,
                                        max_shard_size="5GB", push_to_hub=False)
            optimizer_state = optimizer.state_dict()
        else:
            transformer.save_config(temporary / "transformer")
            optimizer_state = optimizer_summary
        torch.save({"optimizer": optimizer_state, "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    **rng_payload, "parameter_names": [name for name, _ in transformer.named_parameters()]},
                   temporary / "training_state.pt")
        files = {str(file.relative_to(temporary)): file.stat().st_size
                 for file in temporary.rglob("*") if file.is_file()}
        metadata = {
            "schema_version": 3 if execution["strategy"] == "fsdp2" else 2,
            "progress": asdict(progress), "identity": identity,
            "execution": execution,
            "optimizer_state_keys": list(optimizer_state["state"]),
            "recipe": config.recipe_dict(), "wandb_run": wandb_run, "files": files,
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "device": str(device),
                        "packages": {name: version(name) for name in ("diffusers", "transformers", "safetensors")}},
        }
        sources = Path(__file__).resolve().parents[3]
        source_hash = hashlib.sha256()
        for file in sorted(sources.rglob("*.py")):
            source_hash.update(str(file.relative_to(sources)).encode() + b"\0" + file.read_bytes())
        metadata["runtime"]["source_sha256"] = source_hash.hexdigest()
        (temporary / "checkpoint.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        # Flush file contents before the directory rename publishes the inventory.
        for name in (*files, "checkpoint.json"):
            with (temporary / name).open("rb") as handle:
                os.fsync(handle.fileno())
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    size = sum(files.values()) + (destination / "checkpoint.json").stat().st_size
    keep_last = config.recipe["checkpointing"]["keep_last"]
    if keep_last is not None:
        completed = []
        for path in sorted(directory.glob("step-*")):
            if path.is_symlink() or not (path / "checkpoint.json").is_file():
                continue
            try:
                marker = json.loads((path / "checkpoint.json").read_text())
            except (ValueError, OSError):
                continue
            if marker.get("wandb_run") == wandb_run:
                completed.append(path)
        for old in completed[:-keep_last]:
            # Refuse deletion of incompatible or incomplete saves, even inside our run directory.
            saved = load_checkpoint_metadata(old, config, require_training_state=False)
            if saved.metadata["wandb_run"] != wandb_run:
                raise ValueError("retention encountered a checkpoint from another run")
            shutil.rmtree(old)
    values = {"checkpoint/seconds": time.perf_counter() - started,
              "checkpoint/size_bytes": size, "checkpoint/path": str(destination)}
    logging = config.recipe["logging"]
    return values if logging is None else {name: values[name] for name in logging["checkpoint_metrics"]}


__all__ = ["Checkpoint", "checkpoint_identity", "load_checkpoint_metadata", "restore_training_state", "save_checkpoint",
           "validate_epoch_continuation", "remove_training_state", "normalize_checkpoint_recipe",
           "checkpoint_identities_match", "restore_model_weights", "load_saved_training_config"]
