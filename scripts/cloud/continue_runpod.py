#!/usr/bin/env python3
"""Start one additional augmented epoch from a completed RunPod run's full checkpoint.

With --ssh, upload local source and launch a detached job through launch_runpod.
Inside the Pod, reuse local models/data, AdamW/LR/RNG state and the existing W&B
run. --run-name identifies a new local output directory, not a new W&B run.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.cloud import train_runpod as pod


def load_parent(run_directory: Path):
    """Select the final complete checkpoint without accepting a weights-only export."""
    from diffusion_arc.training.data import ArcTrainingDataset
    config = pod.load_saved_config(run_directory / "training.json")
    pod.require_epoch_launch_config(config)
    if config.recipe["deployment"]["output_directory"] != run_directory.resolve():
        raise ValueError("Parent config points to another run directory")
    checkpoints = pod.run_checkpoints(config)
    if not checkpoints:
        raise ValueError("The previous run has no complete local checkpoint including optimizer state")
    checkpoint = checkpoints[-1]
    progress = checkpoint.metadata["progress"]
    if progress["optimizer_step"] != progress.get("epoch", 1) * config.epoch_optimizer_steps:
        raise ValueError("Finish the previous epoch before starting a new one")
    # Prepared manifests identify the source data without reopening every task.
    dataset = ArcTrainingDataset(config.data.split_directory, config.data.view_name, tokenizer=None)
    expected = checkpoint.metadata["identity"]["data_sha256"]
    if dataset.fingerprint_for(expected) != expected:
        raise ValueError("Parent training data differs from the saved checkpoint")
    return config, checkpoint


def continuation_config(parent_config, checkpoint, run_directory: Path, *, seed: int | None, resume: Path | None):
    """Derive the same recipe with a fresh data seed/view and separate local outputs."""
    from diffusion_arc.training.config import load_training_config
    from diffusion_arc.training.run.checkpointing import validate_epoch_continuation

    path = run_directory / "training.json"
    existing = pod.load_saved_config(path) if path.exists() else None
    if seed is None:
        seed = existing.seed if existing else parent_config.seed + 1
    payload = parent_config.recipe_dict()
    payload["seed"] = seed
    payload["models"]["resume_from"] = None
    payload["deployment"]["output_directory"] = str(run_directory)
    payload["data"]["prepared_split_directory"] = str(run_directory / "data/splits/train")
    run = checkpoint.metadata["wandb_run"]
    payload["logging"].update(project=run["project"], entity=run["entity"])
    # Validate without publishing a partial or incompatible destination.
    with TemporaryDirectory() as temporary:
        candidate = Path(temporary) / "training.json"
        pod.write_json(candidate, payload)
        config = replace(load_training_config(candidate, expected_type="training_run"), path=path)
        validate_epoch_continuation(checkpoint, config)
        if existing:
            saved = existing.recipe_dict()
            saved["models"]["resume_from"] = None
            if saved != config.recipe_dict():
                raise ValueError("Existing continuation config differs; use another local run name")
        if resume:
            payload["models"]["resume_from"] = str(resume.resolve())
            pod.write_json(candidate, payload)
            config = replace(load_training_config(candidate, expected_type="training_run"), path=path)
    return config


def prepare_next_view(parent_config, config) -> None:
    """Resample augmentation keys over the prepared source pool without reading task files."""
    from scripts.data.prepare_nvarc_benchmark import source_input_hashes
    from scripts.data.prepare_training_data import prepare_next_epoch_data

    data = config.recipe["data"]
    parent_training_data_manifest = json.loads((parent_config.data.split_directory / "manifest.json").read_text())
    inputs = source_input_hashes(data["source_archive"], data["task_index"], data["official_training_keys"])
    if inputs != parent_training_data_manifest["inputs_sha256"]:
        raise ValueError("Training source pool changed since the previous epoch")
    split = config.data.split_directory
    if not split.parent.parent.exists():
        prepare_next_epoch_data(parent_config, config)
    training_data_manifest = json.loads((split / "manifest.json").read_text())
    if (training_data_manifest["inputs_sha256"] != inputs or training_data_manifest["seed"] != config.seed
            or training_data_manifest["source_count"] != data["expected_task_count"]
            or training_data_manifest["augmentation"]["geometry"] != "uniform_D4"):
        raise ValueError("Prepared continuation view differs from its recipe")
    # Identity and original task contents stay fixed; only augmentation keys/order change.
    contents = lambda rows: {row["task_key"]: row["converted_sha256"] for row in rows}
    if contents(training_data_manifest["sources"]) != contents(parent_training_data_manifest["sources"]):
        raise ValueError("The next epoch must use exactly the same underlying training tasks")
    pod.verify_file(split / f"augmented_keys/{config.data.view_name}.json", training_data_manifest["keys_sha256"])
    print(f"Prepared augmentation seed {config.seed}; reusing {training_data_manifest['source_count']} source tasks", flush=True)


def train(config_path: str, parent_run: str) -> None:
    """Load models in a child process so all GPU memory is released before uploading."""
    from diffusion_arc.training.config import load_training_config
    from diffusion_arc.training.runner import run_training

    config = load_training_config(Path(config_path), expected_type="training_run")
    _, checkpoint = load_parent(Path(parent_run))
    run_training(config, previous_epoch_checkpoint=None if config.recipe["models"]["resume_from"] else checkpoint)


def check_wandb_run(api, checkpoint, *, last_allowed_step: int) -> None:
    """Reject active or newer continuations with a read-only lookup before resuming W&B."""
    identity = checkpoint.metadata["wandb_run"]
    run = api.run("/".join(identity[key] for key in ("entity", "project", "id")))
    if run.state == "running":
        raise ValueError("The W&B run is still active; wait for the previous process to finish")
    if run.lastHistoryStep > last_allowed_step:
        raise ValueError("W&B already contains later steps; choose the latest completed epoch")


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--ssh" or arg.startswith("--ssh=") for arg in arguments):
        from scripts.cloud.launch_runpod import main as launch

        return launch(["continue", *arguments])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", required=True, help="Name of the completed local source run")
    parser.add_argument("--run-name", required=True, help="New local directory; W&B resumes the existing run")
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--augmentation-seed", type=int, help="Defaults to the previous epoch's seed + 1")
    parser.add_argument("--resume", type=Path, help="Latest complete checkpoint of this additional epoch")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-repo", help="Defaults to the previous run's HF destination")
    args = parser.parse_args(arguments)
    for name in (args.from_run, args.run_name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
            parser.error("Run names must contain 1–80 letters, digits, underscores or hyphens")
    if args.from_run == args.run_name:
        parser.error("Use a separate local --run-name for the next epoch")
    pod.require_secrets("HF_TOKEN", "WANDB_API_KEY")
    workspace = args.workspace.resolve()
    os.environ["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    os.environ["HF_HOME"] = str(workspace / ".cache/huggingface")
    from huggingface_hub import HfApi

    provider = pod.load_saved_config(workspace / "runs" / args.from_run / "training.json").recipe["deployment"]["provider"]
    with pod.workspace_lock(workspace, local_only=provider == "modal"):
        parent_run = workspace / "runs" / args.from_run
        parent_config, checkpoint = load_parent(parent_run)
        run_directory = workspace / "runs" / args.run_name
        config = continuation_config(parent_config, checkpoint, run_directory,
                                     seed=args.augmentation_seed, resume=args.resume)
        start_step = pod.validate_start(config, run_directory, args.resume)
        parent_startup = json.loads((parent_run / "startup.json").read_text())
        lineage = {"checkpoint": str(checkpoint.path),
                   "checkpoint_metadata_sha256": pod.sha256_file(checkpoint.path / "checkpoint.json")}
        startup_path = run_directory / "startup.json"
        saved = json.loads(startup_path.read_text()) if startup_path.exists() else None
        output_repo = args.output_repo or (saved or parent_startup)["output_repo"]
        if saved is not None:
            if saved["parent"] != lineage or saved["output_repo"] != output_repo:
                raise ValueError("Continuation parent or upload destination changed")
        pod.prepare_runtime()
        api = HfApi(token=os.environ["HF_TOKEN"])
        pod.prepare_output_repository(api, output_repo, "model")
        import wandb

        wandb_api = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=30)
        if not wandb_api.viewer:
            raise ValueError("W&B credentials could not be verified")
        parent_step = checkpoint.metadata["progress"]["optimizer_step"]
        check_wandb_run(wandb_api, checkpoint, last_allowed_step=(
            parent_step + config.loop.max_optimizer_steps if args.resume else parent_step))
        pod.preflight_gpu_and_disk(config, workspace, start_step)
        pod.write_json(config.path, config.recipe_dict())
        prepare_next_view(parent_config, config)
        pod.write_json(startup_path, {
            **parent_startup, "parent": lineage, "output_repo": output_repo,
            "execution": {"strategy": config.recipe["execution"]["distributed_strategy"],
                          "gpu_count": config.recipe["deployment"]["gpu_count"]},
            "augmentation_seed": config.seed, "launcher_sha256": pod.sha256_file(Path(__file__)),
            "source_snapshot": json.loads((ROOT / "_source.json").read_text()) if (ROOT / "_source.json").exists() else None,
        })
        if args.prepare_only:
            print(f"Next epoch prepared with augmentation seed {config.seed}: {config.path}", flush=True)
            return 0
        pod.write_json(run_directory / "training-started.json", {
            "parent": lineage, "resume_from": config.recipe_dict()["models"]["resume_from"],
        })
        subprocess.run(pod.training_command(config, [
            sys.executable, "-c",
            "from scripts.cloud.continue_runpod import train; import sys; train(*sys.argv[1:])",
            str(config.path), str(parent_run)]), check=True, cwd=ROOT)
        pod.upload_weights(config, api=api, output_repo=output_repo)
        suffix = " The Pod remains running." if config.recipe["deployment"]["provider"] == "runpod" else ""
        print("Additional epoch and weight uploads completed." + suffix, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
