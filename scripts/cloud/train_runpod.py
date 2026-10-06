#!/usr/bin/env python3
"""Prepare and train inside an existing RunPod; publish transformer weights to HF."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from tempfile import TemporaryDirectory, gettempdir
from urllib.request import urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for directory in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

FLUX_REPO = "black-forest-labs/FLUX.2-klein-base-4B"
FLUX_REVISION = "8c44a2fbef88fae175da65df054db0f901aa9747"
NVARC_HANDLE = "sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
DATA_HANDLE = "joshuakauling/nvarc-data/versions/1"
EMBEDDED_MODELS = Path("/opt/models")
NVARC_WEIGHT_SIZES = {
    "model-00001-of-00002.safetensors": 4_996_836_472,
    "model-00002-of-00002.safetensors": 2_270_397_024,
}
# Published metadata plus the deterministically derived 14,490-task index.
DATA_SHA256 = {
    "nvarc_train_tasks.jsonl": "157293e949bb0e57c72ac26dbabd28a287c6fb03eb598d6a0a59f38e33d9b550",
    "nvarc_train_arc_agi_1_training_only_tasks.jsonl": "385141b83e7ee2e17a625caa1ae04405fa34373c8e7717d3d3a551f5860a189e",
    "arc_agi_1_training_keys.json": "d4319ec6951cd46129556f8ad4c4c8d149fab39dec20863cacea8ec000d4dc19",
}


def require_secrets(*names: str) -> None:
    """Fail together on absent, empty or unresolved RunPod secret references."""
    missing = [name for name in names if not os.environ.get(name, "").strip()
               or os.environ[name].strip().startswith("{{")]
    if missing:
        raise ValueError("Missing environment secrets: " + ", ".join(missing))
    print("Required secrets are present: " + ", ".join(names), flush=True)


def write_json(path: Path, value: object) -> None:
    """Replace a small local receipt only after its complete JSON has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_file(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def embedded_model_paths(*, verify_payloads: bool = True) -> dict[str, Path] | None:
    """Check the pinned image receipt, optionally auditing all bundled model files."""
    manifest_path = EMBEDDED_MODELS / "manifest.json"
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return None
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Embedded model manifest must be a regular file")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported embedded model manifest")
    identities = {
        "nvarc": {"handle": NVARC_HANDLE, "path": "nvarc-qwen3-4b-grids15-sft139"},
        "flux": {"repo_id": FLUX_REPO, "revision": FLUX_REVISION, "path": "flux2-klein-base-4b"},
    }
    paths = {}
    print("Reading pinned model receipt bundled in the image", flush=True)
    for component, identity in identities.items():
        entry = manifest.get(component)
        if not isinstance(entry, dict) or any(entry.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Embedded {component} source differs from the pinned recipe")
        root = EMBEDDED_MODELS / entry["path"]
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"Embedded {component} directory is missing or is a symlink")
        files = entry.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError(f"Embedded {component} file inventory is empty or invalid")
        # Walk without following symlinks, and reject files absent from the build receipt.
        if verify_payloads:
            inventory = set()
            for path in root.rglob("*"):
                if path.is_symlink() or (not path.is_dir() and not path.is_file()):
                    raise ValueError(f"Embedded model contains a non-regular entry: {path}")
                if path.is_file():
                    inventory.add(path.relative_to(root).as_posix())
            if inventory != set(files):
                raise ValueError(f"Embedded {component} file inventory differs from its manifest")
        for name, record in files.items():
            if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                    or not isinstance(record, dict) or type(record.get("bytes")) is not int
                    or record["bytes"] <= 0 or not isinstance(record.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
                raise ValueError(f"Invalid embedded model file receipt: {name}")
            if verify_payloads:
                path = root / name
                if path.stat().st_size != record["bytes"]:
                    raise ValueError(f"Embedded model file is incomplete: {path}")
                verify_file(path, record["sha256"])
        if component == "nvarc":
            required = {"config.json", "tokenizer.json", "tokenizer_config.json",
                        "model.safetensors.index.json", *NVARC_WEIGHT_SIZES}
            if not required.issubset(files):
                raise ValueError("Embedded NVARC is missing model or tokenizer files")
            for name, size in NVARC_WEIGHT_SIZES.items():
                if files[name]["bytes"] != size:
                    raise ValueError(f"Embedded NVARC weight size differs from the pinned model: {name}")
        else:
            for subfolder in ("transformer", "vae"):
                if (f"{subfolder}/config.json" not in files
                        or not any(Path(name).parent == Path(subfolder) and name.endswith(".safetensors")
                                   for name in files)):
                    raise ValueError(f"Embedded FLUX is missing {subfolder} config or weights")
        for name in files:
            if name.endswith(".json"):
                payload = json.loads((root / name).read_text())
                if name.endswith(".safetensors.index.json"):
                    weights = payload.get("weight_map") if isinstance(payload, dict) else None
                    if not isinstance(weights, dict) or not weights:
                        raise ValueError(f"Embedded weight index is empty or invalid: {name}")
                    for shard in weights.values():
                        if not isinstance(shard, str) or (Path(name).parent / shard).as_posix() not in files:
                            raise ValueError(f"Embedded weight index references a missing shard: {name}")
        paths[f"{component}_checkpoint"] = root.resolve()
    return paths


def prepare_runtime() -> None:
    """Check the image's training dependencies; add only the missing pinned Kaggle client."""
    from importlib.metadata import PackageNotFoundError, distributions, version
    import tomllib

    requirements = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    for requirement in requirements:
        name, expected = requirement.split("==")
        try:
            actual = version(name)
        except PackageNotFoundError:
            raise RuntimeError(f"Image is missing {requirement}; prepare the runtime before training") from None
        if actual.split("+")[0] != expected:
            raise RuntimeError(f"Image has {name}=={actual}; this source snapshot requires {requirement}")
    try:
        installed = version("kagglehub")
    except PackageNotFoundError:
        installed = None
    if installed is None:
        # Preserve every installed package, including torch/FA3 and their dependencies.
        with TemporaryDirectory() as temporary:
            constraints = Path(temporary) / "constraints.txt"
            constraints.write_text("\n".join(sorted(
                f"{distribution.metadata['Name']}=={distribution.version}"
                for distribution in distributions() if distribution.metadata["Name"]
            )) + "\n")
            print("Installing missing kagglehub==1.0.2 without changing installed packages", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                            "--constraint", str(constraints), "kagglehub==1.0.2"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    elif installed != "1.0.2":
        raise RuntimeError(f"Expected kagglehub==1.0.2, found {installed}")


@contextmanager
def workspace_lock(workspace: Path, *, local_only: bool = False):
    """Serialize local launchers; Modal additionally serializes Sandboxes by Volume name."""
    import fcntl

    if not workspace.is_dir():
        raise ValueError(f"Workspace disk must already be mounted: {workspace}")
    # Modal Volumes do not support flock. Its Sandbox name provides the cross-container
    # exclusion; this /tmp lock only coordinates subprocesses inside that container.
    lock = (Path(gettempdir()) / ("diffusion-arc-" + hashlib.sha256(str(workspace).encode()).hexdigest() + ".lock")
            if local_only else workspace / ".diffusion-arc.lock")
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another launcher is using this workspace") from None
        yield


def prepare_output_repository(api, repo_id: str, repo_type: str) -> None:
    """Verify write access, preserving existing visibility; create missing repos as private."""
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
    from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status
    from requests.exceptions import ConnectionError, Timeout

    for attempt in range(3):
        try:
            try:
                info = api.repo_info(repo_id, repo_type=repo_type, timeout=15)
            except RepositoryNotFoundError as error:
                # An unauthorized response must not be treated as a missing repository.
                if error.response is None or error.response.status_code != 404:
                    raise
                api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
                info = api.repo_info(repo_id, repo_type=repo_type, timeout=15)
            # Hub 0.36's auth_check has no write argument; use the Hub endpoint.
            response = get_session().get(
                f"{api.endpoint}/api/{repo_type}s/{repo_id}/auth-check", params={"write": "true"},
                headers=build_hf_headers(token=api.token), timeout=15,
            )
            hf_raise_for_status(response)
            visibility = "private" if info.private else "public"
            print(f"HF output repository: {repo_id} ({visibility}); write access verified", flush=True)
            return
        except (HfHubHTTPError, ConnectionError, Timeout) as error:
            status = error.response.status_code if getattr(error, "response", None) is not None else None
            if isinstance(error, HfHubHTTPError) and status not in (500, 502, 503, 504):
                raise
            reason = f"HTTP {status}" if status is not None else type(error).__name__
            if attempt == 2:
                raise RuntimeError(
                    f"Hugging Face could not prepare {repo_id} after 3 attempts ({reason}). "
                    "Check https://status.huggingface.co/ and retry when the Hub is available."
                ) from error
            delay = 2 ** (attempt + 1)
            print(f"Hugging Face output check failed ({reason}); retry {attempt + 2}/3 in {delay}s", flush=True)
            time.sleep(delay)


def load_saved_config(path: Path):
    """Load a published run receipt with explicit historical schema migration."""
    from diffusion_arc.training.run.checkpointing import load_saved_training_config

    return load_saved_training_config(path)


def require_epoch_launch_config(config) -> None:
    """Keep this launcher explicitly scoped to uploaded, logged full epochs."""
    recipe = config.recipe
    if recipe["data"]["source"] != "epoch_pool" or recipe["loop"]["epochs"] != 1:
        raise ValueError("This cloud launcher requires one complete epoch_pool epoch")
    if recipe["logging"] is None or recipe["checkpointing"] is None:
        raise ValueError("This cloud launcher requires logging and checkpointing for weight uploads")
    if config.benchmark is not None:
        raise ValueError("Use the benchmark launcher for a configured benchmark")


def resolve_config(args, run_directory: Path, *, model_paths: dict[str, Path] | None = None):
    """Change deployment paths and account settings while retaining the training recipe."""
    from diffusion_arc.training.config import load_training_config

    template = load_training_config(args.config, expected_type="training_run")
    require_epoch_launch_config(template)
    payload = template.recipe_dict()
    root = args.workspace
    payload["models"].update(
        nvarc_checkpoint=str(root / "models/nvarc-qwen3-4b-grids15-sft139"),
        flux_checkpoint=str(root / "models/flux2-klein-base-4b"),
        resume_from=str(args.resume.resolve()) if args.resume else None,
    )
    if model_paths is not None:
        payload["models"].update({key: str(path) for key, path in model_paths.items()})
    # The historical source_archive field also accepts the extracted Kaggle task tree.
    payload["data"]["source_archive"] = str(root / "data/nvarc_data/kaggle/nvarc-all-classified")
    for key in ("task_index", "official_training_keys"):
        payload["data"][key] = str(root / "data/nvarc_data" / Path(payload["data"][key]).name)
    payload["data"]["prepared_split_directory"] = str(root / "data/training/arc_agi_1_epoch/splits/train")
    if payload["models"]["initialization"] == "transformer_weights":
        payload["data"]["prepared_split_directory"] = str(run_directory / "data/splits/train")
    if payload["validation"] is not None:
        payload["validation"]["bundle"] = str(root / "data/benchmarks/arc_agi_1_validation")
    payload["deployment"]["output_directory"] = str(run_directory)
    payload["logging"].update(project=args.wandb_project, entity=args.wandb_entity)
    # All paths are absolute already. Validate before publishing the resolved file.
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "training.json"
        write_json(path, payload)
        config = replace(load_training_config(path, expected_type="training_run"),
                         path=run_directory / "training.json")
    if config.loop.max_optimizer_steps not in config.checkpoint_steps:
        raise ValueError("RunPod upload requires a final local checkpoint")
    if config.recipe["logging"]["mode"] != "online":
        raise ValueError("RunPod training requires online W&B logging")
    return config


def run_checkpoints(config, *, include_weights_only: bool = False) -> list:
    """Verify one run's saves; omit explicitly pruned training states from resume candidates."""
    from diffusion_arc.training.run.checkpointing import load_checkpoint_metadata

    root = config.recipe["deployment"]["output_directory"] / "checkpoints"
    paths = sorted(root.glob("*/step-*/checkpoint.json"))
    runs = {path.parent.parent.name for path in paths}
    if len(runs) > 1:
        raise ValueError("Output directory contains multiple W&B runs; use a separate run name")
    checkpoints = [load_checkpoint_metadata(path.parent, config, require_training_state=False) for path in paths]
    for checkpoint in checkpoints:
        if checkpoint.metadata["wandb_run"]["id"] != checkpoint.path.parent.name:
            raise ValueError("Checkpoint path and W&B run identity disagree")
    return sorted((item for item in checkpoints if include_weights_only or not item.metadata.get("weights_only", False)),
                  key=lambda item: item.metadata["progress"]["optimizer_step"])


def validate_start(config, run_directory: Path, resume: Path | None) -> int:
    """Require an explicit latest-checkpoint resume after training has been attempted."""
    checkpoints = run_checkpoints(config)
    if resume:
        if not checkpoints or checkpoints[-1].path != resume.resolve():
            raise ValueError("Resume must select the latest complete checkpoint in this run directory")
        run = checkpoints[-1].metadata["wandb_run"]
        logging = config.recipe["logging"]
        if logging["project"] != run["project"] or logging["entity"] not in (None, run["entity"]):
            raise ValueError("Resume W&B project/entity must match the checkpoint")
        progress = checkpoints[-1].metadata["progress"]
        step = progress["optimizer_step"] - (progress.get("epoch", 1) - 1) * config.epoch_optimizer_steps
        if step == config.loop.max_optimizer_steps:
            raise ValueError("Epoch is complete; use the upload command to retry weight uploads")
        return step
    if checkpoints or (run_directory / "training-started.json").exists():
        raise ValueError("Training was already attempted; use --resume or a new --run-name")
    return 0


def preflight_disk(config, workspace: Path, start_step: int) -> None:
    """Reserve space for every remaining full checkpoint, including a safety margin."""
    future_saves = sum(step > start_step for step in config.checkpoint_steps)
    # Full FP32 transformer + AdamW is about 48 GB per save; retain headroom.
    needed_gib = 55 * future_saves + 25
    if config.recipe["deployment"]["provider"] == "modal":
        # Modal Volumes grow on demand; statfs reports placeholder capacity, not free space.
        print(f"Modal Volume: budget about {needed_gib} GiB for remaining checkpoints; "
              "no fixed-capacity disk check", flush=True)
        return
    free_gib = shutil.disk_usage(workspace).free / 1024**3
    if free_gib < needed_gib:
        raise ValueError(f"Workspace needs {needed_gib} GiB free; found {free_gib:.1f} GiB")
    print(f"Disk preflight: {free_gib:.1f} GiB free; budget {needed_gib} GiB", flush=True)


def preflight_gpu_and_disk(config, workspace: Path, start_step: int) -> None:
    """Check checkpoint headroom and exercise the existing FA3 forward/backward preflight."""
    if config.recipe["deployment"]["gpu_type"] != "H200":
        raise ValueError("This launcher supports H200 training recipes")
    preflight_disk(config, workspace, start_step)
    gpu_count = config.recipe["deployment"]["gpu_count"]
    subprocess.run([sys.executable, "-c", "\n".join((
        "import torch",
        "from diffusion_arc.model.flux.packing import Flash3Backend",
        "assert torch.cuda.is_available(), 'CUDA is unavailable'",
        f"assert torch.cuda.device_count() == {gpu_count}, 'Expose exactly {gpu_count} H200 GPUs'",
        f"for index in range({gpu_count}):",
        "    assert 'H200' in torch.cuda.get_device_name(index), 'This recipe expects H200 GPUs'",
        "    assert torch.cuda.get_device_capability(index) == (9, 0), 'Hopper SM90 required'",
        "    with torch.cuda.device(index):",
        "        Flash3Backend(f'cuda:{index}')",
        "print('GPU/FA3 forward and backward preflight passed', flush=True)",
    ))], check=True, cwd=REPOSITORY_ROOT)


def training_command(config, command: list[str]) -> list[str]:
    """Launch one training process per configured GPU, preserving stage arguments."""
    strategy = config.recipe["execution"]["distributed_strategy"]
    if strategy == "single_process":
        return list(command)
    if strategy not in ("ddp", "fsdp2"):
        raise ValueError(f"Unsupported training execution strategy: {strategy}")
    count = config.recipe["deployment"]["gpu_count"]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc-per-node={count}", "--max-restarts=0", "--no-python", *command]


def download_models(config, *, workspace: Path, include_transformer: bool = True) -> None:
    """Reuse the pinned SDK caches; fetch only the FLUX components training actually loads."""
    import kagglehub
    from huggingface_hub import snapshot_download

    print("Preparing pinned NVARC and FLUX checkpoints", flush=True)
    nvarc = Path(kagglehub.model_download(NVARC_HANDLE))
    for name, size in NVARC_WEIGHT_SIZES.items():
        if (nvarc / name).stat().st_size != size:
            raise ValueError(f"NVARC weight file is incomplete: {name}")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        json.loads((nvarc / name).read_text())
    # Config loading resolves symlinks, so manage this alias by its workspace path.
    destination = workspace / "models/nvarc-qwen3-4b-grids15-sft139"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() or destination.resolve() != nvarc.resolve():
            raise ValueError(f"NVARC path does not point to the pinned download: {destination}")
    else:
        destination.symlink_to(nvarc, target_is_directory=True)
    patterns = ["vae/*.json", "vae/*.safetensors"]
    if include_transformer:
        patterns = ["transformer/*.json", "transformer/*.safetensors", *patterns]
    snapshot_download(FLUX_REPO, revision=FLUX_REVISION,
                      local_dir=config.models.flux_checkpoint, token=os.environ["HF_TOKEN"],
                      allow_patterns=patterns)


def preflight_data_access() -> None:
    """Verify private Kaggle access with a tiny pinned file before large downloads."""
    import kagglehub

    keys = Path(kagglehub.dataset_download(DATA_HANDLE, path="arc_agi_1_training_keys.json"))
    verify_file(keys, DATA_SHA256["arc_agi_1_training_keys.json"])


def prepare_data(config, *, workspace: Path) -> None:
    """Prepare missing bundles; trust recorded source hashes when reusing prepared tasks."""
    require_epoch_launch_config(config)
    import kagglehub
    from scripts.data.build_nvarc_arc_agi_1_training_only import build_subset
    from scripts.data.prepare_nvarc_benchmark import source_input_hashes
    from scripts.data.prepare_training_data import prepare_training_data
    from scripts.data.prepare_arc_validation_benchmark import ARCHIVE_URL, ARCHIVE_SHA256, prepare
    from diffusion_arc.training.run.validation import validate_validation_bundle

    source = workspace / "data/nvarc_data"
    downloaded = Path(kagglehub.dataset_download(DATA_HANDLE, output_dir=str(source / "kaggle")))
    if downloaded.resolve() != (source / "kaggle").resolve():
        raise ValueError("Kaggle download did not use the configured workspace directory")
    original_index = downloaded / "nvarc-all-classified/metadata/views/nvarc_train_tasks.jsonl"
    official_keys = downloaded / "arc_agi_1_training_keys.json"
    verify_file(original_index, DATA_SHA256[original_index.name])
    verify_file(official_keys, DATA_SHA256[official_keys.name])
    data = config.recipe["data"]
    if not data["official_training_keys"].exists():
        with TemporaryDirectory(dir=source) as temporary:
            staged = Path(temporary) / official_keys.name
            shutil.copyfile(official_keys, staged)
            staged.replace(data["official_training_keys"])
    verify_file(data["official_training_keys"], DATA_SHA256[official_keys.name])
    if not data["task_index"].exists():
        build_subset(source_tasks=original_index, official_keys_path=data["official_training_keys"],
                     output_tasks=data["task_index"],
                     output_keys=source / "nvarc_train_arc_agi_1_training_only_keys.json", overwrite=False)
    verify_file(data["task_index"], DATA_SHA256[data["task_index"].name])
    split = config.data.split_directory
    if not split.parent.parent.exists():
        prepare_training_data(config, config_path=config.path)
    else:
        manifest = json.loads((split / "manifest.json").read_text())
        expected_inputs = source_input_hashes(data["source_archive"], data["task_index"], data["official_training_keys"])
        if (manifest["source_format"] != "directory" or manifest["inputs_sha256"] != expected_inputs
                or manifest["seed"] != config.seed
                or manifest["source_count"] != config.recipe["data"]["expected_task_count"]
                or manifest["augmentation"]["geometry"] != "uniform_D4"):
            raise ValueError("Existing training bundle differs from this recipe")
        verify_file(split / f"augmented_keys/{config.data.view_name}.json", manifest["keys_sha256"])
        print(f"Reusing {manifest['source_count']} prepared tasks without scanning task files", flush=True)
    validation = config.recipe["validation"]
    if validation is None:
        return
    if not validation["bundle"].exists():
        # A failed transfer or preparation never publishes a partial validation bundle.
        with TemporaryDirectory(dir=workspace) as temporary:
            archive = Path(temporary) / "arc-agi-1.tar.gz"
            with urlopen(ARCHIVE_URL, timeout=60) as response, archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            verify_file(archive, ARCHIVE_SHA256)
            bundle = Path(temporary) / "validation"
            prepare(archive, bundle, seed=validation["split_seed"])
            validation["bundle"].parent.mkdir(parents=True, exist_ok=True)
            bundle.rename(validation["bundle"])
    validate_validation_bundle(validation["bundle"], split_seed=validation["split_seed"])


@contextmanager
def _transformer_upload_files(checkpoint):
    """Expose HF weights, explicitly consolidating FSDP shards on CPU for export."""
    if checkpoint.metadata.get("execution", {}).get("strategy") != "fsdp2":
        names = [Path(name).name for name in sorted(checkpoint.metadata["files"])
                 if Path(name).parent == Path("transformer")]
        yield checkpoint.path / "transformer", names
        return
    from diffusion_arc.training.run.fsdp_checkpoint import export_sharded_transformer

    model_bytes = sum(size for name, size in checkpoint.metadata["files"].items()
                      if name.startswith("model/"))
    if shutil.disk_usage(checkpoint.path.parent).free < 2 * model_bytes + 1024**3:
        raise ValueError("FSDP2 HF export requires scratch space for CPU weights and safetensors")
    with TemporaryDirectory(prefix=".fsdp-export-", dir=checkpoint.path.parent) as temporary:
        exported = Path(temporary) / "transformer"
        print(f"Exporting FSDP2 model shards to HF weights on CPU: {checkpoint.path.name}", flush=True)
        export_sharded_transformer(checkpoint.path, exported)
        yield exported, sorted(path.name for path in exported.iterdir() if path.is_file())


def upload_weights(config, *, api, output_repo: str) -> None:
    """Upload all retained transformer versions, never optimizer/RNG states or task data."""
    from huggingface_hub import CommitOperationAdd

    checkpoints = run_checkpoints(config, include_weights_only=True)
    if not checkpoints:
        raise ValueError("No complete checkpoints found for upload")
    receipts = config.recipe["deployment"]["output_directory"] / "uploads"
    for checkpoint in checkpoints:
        run_id = checkpoint.metadata["wandb_run"]["id"]
        prefix = f"{run_id}/{checkpoint.path.name}/transformer"
        with _transformer_upload_files(checkpoint) as (directory, names):
            operations = []
            for name in names:
                if (name == "config.json" or name.endswith(".safetensors")
                        or name.endswith(".safetensors.index.json")):
                    operations.append(CommitOperationAdd(
                        path_in_repo=f"{prefix}/{name}", path_or_fileobj=directory / name,
                    ))
            if not any(name.endswith(".safetensors") for name in names):
                raise ValueError("Transformer upload requires weight tensors, not only a config")
            print(f"Uploading weights: {output_repo}/{prefix}", flush=True)
            commit = api.create_commit(repo_id=output_repo, operations=operations,
                                       commit_message=f"Save {run_id} {checkpoint.path.name} transformer")
        write_json(receipts / f"{checkpoint.path.name}.json", {
            "repo_id": output_repo, "path_in_repo": prefix, "commit": commit.oid,
            "checkpoint": str(checkpoint.path),
            "files": [operation.path_in_repo for operation in operations],
        })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "upload"))
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--run-name", help="Unique directory name for this training run")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/training/arc_agi_1_epoch.json")
    parser.add_argument("--resume", type=Path, help="Latest full local checkpoint of this run")
    parser.add_argument("--prepare-only", action="store_true", help="Run preflight and downloads without training")
    parser.add_argument("--output-repo", default=os.environ.get("HF_OUTPUT_REPO"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "diffusion-arc"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    args = parser.parse_args(argv)
    # Do not import service/GPU SDKs, touch the workspace or transfer anything before this check.
    require_secrets(*(("HF_TOKEN", "WANDB_API_KEY", "KAGGLE_API_TOKEN")
                      if args.command == "run" else ("HF_TOKEN",)))
    # Child processes must import this uploaded snapshot, not the code bundled in the image.
    os.environ["PYTHONPATH"] = os.pathsep.join((str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)))
    if args.command != "run" and (args.resume or args.prepare_only):
        parser.error("--resume and --prepare-only apply only to run")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name or ""):
        parser.error("--run-name must be 1–80 letters, digits, underscores or hyphens")
    args.workspace = args.workspace.resolve()
    os.environ["HF_HOME"] = str(args.workspace / ".cache/huggingface")
    os.environ["KAGGLEHUB_CACHE"] = str(args.workspace / ".cache/kagglehub")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    username = api.whoami()["name"]
    output_repo = args.output_repo or f"{username}/diffusion-arc-weights"

    with workspace_lock(args.workspace):
        run_directory = args.workspace / "runs" / args.run_name
        startup_path = run_directory / "startup.json"
        if args.output_repo is None and startup_path.exists():
            output_repo = json.loads(startup_path.read_text())["output_repo"]
        if args.command == "upload":
            from diffusion_arc.training.config import load_training_config

            config = load_saved_config(run_directory / "training.json")
            if config.recipe["deployment"]["output_directory"] != run_directory:
                raise ValueError("Saved config points to another run directory")
            prepare_output_repository(api, output_repo, "model")
            upload_weights(config, api=api, output_repo=output_repo)
            return 0

        model_paths = embedded_model_paths()
        config = resolve_config(args, run_directory, model_paths=model_paths)
        start_step = validate_start(config, run_directory, args.resume)
        prepare_runtime()
        prepare_output_repository(api, output_repo, "model")
        if model_paths is None:
            api.auth_check(FLUX_REPO)
        import wandb

        if not wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=30).viewer:
            raise ValueError("W&B credentials could not be verified")
        preflight_gpu_and_disk(config, args.workspace, start_step)
        import kagglehub

        # A tiny file checks Kaggle access before transferring model weights or data.
        if model_paths is None:
            json.loads(Path(kagglehub.model_download(NVARC_HANDLE, path="config.json")).read_text())
        preflight_data_access()
        if model_paths is None:
            download_models(config, workspace=args.workspace)
        # Freeze the final cache paths before preparation, training and checkpoint checks.
        config = resolve_config(args, run_directory, model_paths=model_paths)
        write_json(config.path, config.recipe_dict())
        prepare_data(config, workspace=args.workspace)
        write_json(run_directory / "startup.json", {
            "flux": {"repo_id": FLUX_REPO, "revision": FLUX_REVISION}, "nvarc": NVARC_HANDLE,
            "models": {"provider": "image" if model_paths is not None else "download",
                       "manifest_sha256": sha256_file(EMBEDDED_MODELS / "manifest.json")
                       if model_paths is not None else None},
            "data": {"provider": "kaggle", "handle": DATA_HANDLE, "sha256": DATA_SHA256},
            "output_repo": output_repo, "launcher_sha256": sha256_file(Path(__file__)),
            "execution": {"strategy": config.recipe["execution"]["distributed_strategy"],
                          "gpu_count": config.recipe["deployment"]["gpu_count"]},
            "source_snapshot": (json.loads((REPOSITORY_ROOT / "_source.json").read_text())
                                if (REPOSITORY_ROOT / "_source.json").exists() else None),
        })
        if args.prepare_only:
            print(f"Preparation complete. Resolved config: {config.path}", flush=True)
            return 0
        write_json(run_directory / "training-started.json", {"resume_from": config.recipe_dict()["models"]["resume_from"]})
        subprocess.run(training_command(config, [sys.executable, str(REPOSITORY_ROOT / "scripts/train.py"),
                                                 "--config", str(config.path)]),
                       check=True, cwd=REPOSITORY_ROOT)
        upload_weights(config, api=api, output_repo=output_repo)
        print("Training and weight uploads completed. The Pod remains running.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
