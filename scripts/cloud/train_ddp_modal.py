#!/usr/bin/env python3
"""Launch detached distributed diagnostics or benchmark repetitions on Modal H200s."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import modal


REMOTE_ROOT = Path("/opt/diffusion-arc")
LOCAL_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_ROOT
VOLUME_ROOT = Path("/mnt/diffusion-arc-models")
BUNDLE = "data/benchmarks/single_gpu_b0"
CONFIG = "configs/training/benchmarks/ddp_training.json"
NUMERICS_CRITERIA = "configs/training/benchmarks/ddp_numerics.json"
TIMEOUTS = {"smoke": 600, "benchmark": 1800, "diagnostic": 600, "memory": 1800,
            "profile": 900, "numerics": 1800}
ALLOCATOR_CONFIG = "expandable_segments:True"

sys.path.insert(0, str(LOCAL_ROOT))
sys.path.insert(0, str(LOCAL_ROOT / "src"))
from scripts.cloud._training_image import FA3_REVISION, training_image

app = modal.App("diffusion-arc-ddp")
model_volume = modal.Volume.from_name("diffusion-arc-models", create_if_missing=False)
image = training_image(LOCAL_ROOT).workdir(REMOTE_ROOT)
for directory in ("src", "scripts", "configs", BUNDLE):
    image = image.add_local_dir(
        LOCAL_ROOT / directory, remote_path=str(REMOTE_ROOT / directory),
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )


def resources(gpus: int, mode: str, repetitions: int = 1) -> dict:
    """Keep CPU and host-memory budgets fixed per GPU across comparisons."""
    if type(gpus) is not int or gpus not in (1, 2, 4, 8):
        raise ValueError("--gpus must be 1, 2, 4 or 8")
    if mode not in TIMEOUTS:
        raise ValueError("--mode must be smoke, benchmark, diagnostic, memory, profile or numerics")
    if mode == "numerics" and gpus != 2:
        raise ValueError("--mode numerics requires exactly 2 GPUs")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    cpu, memory = 8.0 * gpus, 64 * 1024 * gpus
    return dict(gpu=f"H200:{gpus}", cpu=(cpu, cpu), memory=(memory, memory),
                timeout=TIMEOUTS[mode] * (repetitions if mode == "benchmark" else 1))


def check_inputs(repository: Path, config: str, *, remote: bool = False):
    """Validate the uploaded recipe and data without loading model weights."""
    from diffusion_arc.training.benchmark_config import load_training_benchmark_config
    from diffusion_arc.training.data import ArcTrainingDataset
    from diffusion_arc.training.source_lengths import load_source_token_lengths

    repository = repository.resolve()
    config_path = (repository / config).resolve()
    config_path.relative_to(repository / "configs")
    experiment = load_training_benchmark_config(config_path)
    training = experiment.training
    training.path.relative_to(repository / "configs")
    expected_split = repository / BUNDLE / "splits/train"
    if training.data.split_directory != expected_split:
        raise ValueError(f"this launcher uploads only the benchmark bundle at {expected_split}")
    if not (expected_split / "config.json").is_file():
        raise FileNotFoundError(f"benchmark data is not prepared: {expected_split}")
    # Dataset construction validates source files and split boundaries; tokenization
    # is deferred to __getitem__, so this preflight needs no tokenizer or weights.
    dataset = ArcTrainingDataset(expected_split, training.data.view_name, tokenizer=None)
    if (len(dataset.keys) != experiment.expected_source_count
            or len({key.task_id for key in dataset.keys}) != experiment.expected_source_count):
        raise ValueError(f"benchmark view requires {experiment.expected_source_count} distinct tasks")
    if any(len(key.augmentation.pair_indices) != training.recipe["data"]["pairs_per_sequence"]
           for key in dataset.keys):
        raise ValueError("benchmark pair counts do not match the training recipe")
    if training.rank_batch_assignment in ("token_balanced_v1", "joint_packing_v1"):
        load_source_token_lengths(dataset,
            min_context_pairs_before_target=training.loop.min_context_pairs_before_target)
    for checkpoint in (training.models.nvarc_checkpoint, training.models.flux_checkpoint):
        checkpoint.relative_to((repository / "checkpoints").resolve())
        if remote and not checkpoint.is_dir():
            raise FileNotFoundError(f"required cached model is missing: {checkpoint}")
    return experiment


def run_worker(command: list[str], *, timeout: int) -> None:
    """Bound torchrun and terminate its entire worker group on interruption."""
    environment = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                   "PYTORCH_ALLOC_CONF": ALLOCATOR_CONFIG,
                   "PYTHONUNBUFFERED": "1",
                   "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1"}
    process = subprocess.Popen(command, cwd=REMOTE_ROOT, env=environment, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def check_numerics_inputs(repository: Path, experiment):
    """Validate the fixed numerical comparison before allocating or loading GPUs."""
    from diffusion_arc.training.run.numerics import NumericalCriteria

    if experiment.precompute_packing:
        raise ValueError("numerics does not support precompute_packing; use a benchmark or smoke run")
    if experiment.training.rank_batch_assignment == "joint_packing_v1":
        raise ValueError("numerics does not support background joint packing; use a benchmark or smoke run")
    if experiment.fsdp_gradient_sync != "last_microbatch":
        raise ValueError("numerics does not support fsdp_gradient_sync=per_microbatch; use a benchmark or smoke run")
    strategy = experiment.training.recipe["execution"]["distributed_strategy"]
    if strategy not in ("ddp", "fsdp2"):
        raise ValueError("numerics requires a ddp or fsdp2 benchmark configuration")
    if experiment.training.recipe["deployment"]["gpu_count"] != 2:
        raise ValueError("numerics requires configured gpu_count=2")
    criteria = NumericalCriteria.load(repository / NUMERICS_CRITERIA)
    if criteria.steps > experiment.benchmark_steps:
        raise ValueError("numerics requires enough configured batches for every comparison step")
    return criteria


def run_numerics(experiment, destination: Path, launch: dict, *, deadline: float) -> None:
    """Run fresh comparison processes, retaining large tensor captures only locally."""
    criteria = check_numerics_inputs(REMOTE_ROOT, experiment)
    criteria_path = REMOTE_ROOT / NUMERICS_CRITERIA
    launch["numerical_criteria"] = json.loads(criteria_path.read_text())
    launch["phases"] = []
    output = destination / "training"
    include_fsdp2 = experiment.training.recipe["execution"]["distributed_strategy"] == "fsdp2"
    phases = ("reference", "ddp", "fsdp2", "compare") if include_fsdp2 else ("reference", "ddp", "compare")
    # Scratch files never enter the mounted volume and are removed after any failure.
    with tempfile.TemporaryDirectory(prefix="diffusion-arc-numerics-") as scratch:
        for phase in phases:
            command = [sys.executable, "-B", "-u"]
            if phase in ("ddp", "fsdp2"):
                command.extend(["-m", "torch.distributed.run", "--standalone",
                                "--nproc-per-node=2", "--max-restarts=0"])
            command.extend([str(REMOTE_ROOT / "scripts/diagnostics/validate_ddp.py"),
                            "--phase", phase, "--config", str(experiment.path),
                            "--criteria", str(criteria_path), "--output-dir", str(output),
                            "--scratch-dir", scratch])
            if phase == "compare" and include_fsdp2:
                command.append("--include-fsdp2")
            entry = dict(phase=phase, status="running", command=command)
            launch["phases"].append(entry)
            (destination / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
            print(f"Starting numerical comparison phase {phase}: {output}", flush=True)
            try:
                remaining = int(deadline - time.monotonic())
                if remaining <= 0:
                    raise TimeoutError("shared numerical comparison time budget exhausted")
                run_worker(command, timeout=remaining)
                entry["status"] = "complete"
            except BaseException as error:
                entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
        summary = json.loads((output / "summary.json").read_text())
        if (summary["event"] != "complete" or summary["optimizer_steps"] != criteria.steps
                or summary["status"] not in ("passed", "failed")):
            raise RuntimeError("numerical comparison did not complete the requested training window")
        launch["validation_status"] = summary["status"]


@app.function(
    image=image, **resources(1, "smoke"), max_containers=1, retries=0,
    scaledown_window=2, volumes={str(VOLUME_ROOT): model_volume},
)
def train_ddp(config: str, gpus: int, mode: str, metadata: dict, diagnostic_batch: int = 0) -> str:
    """Keep the complete torchrun lifecycle inside the detached Modal function."""
    # Preserve the public Modal argument while using an explicit batch number internally.
    diagnostic_batch_number = diagnostic_batch
    started = time.monotonic()
    allocation = resources(gpus, mode)
    result_strategy = "ddp"
    if mode == "numerics":
        from diffusion_arc.training.benchmark_config import load_training_benchmark_config

        selected = load_training_benchmark_config(REMOTE_ROOT / config)
        check_numerics_inputs(REMOTE_ROOT, selected)
        result_strategy = selected.training.recipe["execution"]["distributed_strategy"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = VOLUME_ROOT / "results" / result_strategy / f"{run_id}-{mode}-{gpus}gpu"
    destination.mkdir(parents=True, exist_ok=False)
    launch = dict(status="running", metadata=metadata, resources=allocation,
                  fa3_revision=FA3_REVISION, mode=mode, gpus=gpus,
                  allocator_config=ALLOCATOR_CONFIG)
    try:
        link = REMOTE_ROOT / "checkpoints"
        if link.is_symlink():
            if link.resolve() != VOLUME_ROOT:
                raise RuntimeError(f"unexpected checkpoint symlink: {link}")
        elif link.exists():
            raise RuntimeError(f"checkpoint path already exists: {link}")
        else:
            link.symlink_to(VOLUME_ROOT, target_is_directory=True)
        experiment = check_inputs(REMOTE_ROOT, config, remote=True)
        if mode == "profile" and experiment.warmup_steps < 1:
            raise ValueError("profile requires at least one configured warmup step")
        launch["benchmark_config"] = json.loads(experiment.path.read_text())
        if mode == "numerics":
            run_numerics(experiment, destination, launch,
                         deadline=started + TIMEOUTS[mode] - 90)
            launch["status"] = "complete"
            return str(destination)
        repetitions = experiment.repetitions if mode == "benchmark" else 1
        launch["resources"] = resources(gpus, mode, repetitions)
        launch["repetitions"] = []
        steps = (experiment.warmup_steps + 1 if mode == "profile" else
                 3 if mode == "diagnostic" else
                 experiment.smoke_steps if mode == "smoke" else experiment.benchmark_steps)
        completed = []
        for index in range(1, repetitions + 1):
            output = destination / (f"repetition-{index}" if mode == "benchmark" else "training")
            command = [sys.executable, "-B", "-u", "-m", "torch.distributed.run",
                       "--standalone", f"--nproc-per-node={gpus}", "--max-restarts=0",
                       str(REMOTE_ROOT / "scripts/benchmark_training.py"),
                       "--config", str(experiment.path), "--mode", mode,
                       "--output-dir", str(output)]
            if mode == "diagnostic":
                command.extend(["--diagnostic-batch", str(diagnostic_batch_number)])
            entry = dict(repetition=index, status="running", command=command)
            launch["repetitions"].append(entry)
            (destination / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
            print(f"Starting DDP repetition {index}/{repetitions}: {output}", flush=True)
            try:
                # A new torchrun resets ranks, model, optimizer, seeds and CUDA allocator.
                run_worker(command, timeout=TIMEOUTS[mode] - 90)
                summary = json.loads((output / "summary.json").read_text())
                if summary["event"] != "complete" or summary["optimizer_steps"] != steps:
                    raise RuntimeError("torchrun did not complete the requested training window")
                if mode == "profile":
                    for rank in range(gpus):
                        trace = output / f"rank-{rank}.trace.json.gz"
                        if not trace.is_file() or trace.stat().st_size == 0:
                            raise RuntimeError(f"Torch profiler trace missing for rank {rank}")
                if mode == "benchmark":
                    from diffusion_arc.training.run.benchmarking import BenchmarkWindow, summarize_repetitions
                    series = summarize_repetitions(
                        completed + [output], expected_repetitions=repetitions,
                        window=BenchmarkWindow(experiment.warmup_steps, experiment.measured_steps),
                    )
                    (destination / "summary.json").write_text(json.dumps(series, indent=2) + "\n")
                completed.append(output)
                entry["status"] = "complete"
            except BaseException as error:
                entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            if index < repetitions:
                (destination / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
                model_volume.commit()
        launch["status"] = "complete"
    except BaseException as error:
        launch.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            (destination / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
        finally:
            model_volume.commit()
        print(f"Persistent {result_strategy.upper()} results: {destination}", flush=True)
    return str(destination)


@app.local_entrypoint()
def main(gpus: int = 1, mode: str = "smoke", config: str = CONFIG, diagnostic_batch: int = 0) -> None:
    """Check local inputs before requesting a GPU; never prepare or download data."""
    # Preserve the Modal --diagnostic-batch option at the public boundary.
    diagnostic_batch_number = diagnostic_batch
    allocation = resources(gpus, mode)
    experiment = check_inputs(LOCAL_ROOT, config)
    if mode != "numerics" and gpus != experiment.training.recipe["deployment"]["gpu_count"]:
        raise ValueError("--gpus must match the configuration's execution.gpu_count")
    allocation = resources(gpus, mode, experiment.repetitions)
    if mode == "profile" and experiment.warmup_steps < 1:
        raise ValueError("profile requires at least one configured warmup step")
    if mode == "numerics":
        check_numerics_inputs(LOCAL_ROOT, experiment)
    if mode == "diagnostic":
        from diffusion_arc.training.distributed_data import plan_distributed_batches
        counts = (experiment.training.recipe["data"]["pairs_per_sequence"]
                  - experiment.training.loop.min_context_pairs_before_target,)
        plans = plan_distributed_batches(counts * experiment.expected_source_count,
                                        optimizer_batch_size=experiment.training.loop.optimizer_batch_size,
                                        rank=0, world_size=gpus)
        if type(diagnostic_batch_number) is not int or not 1 <= diagnostic_batch_number <= len(plans):
            raise ValueError("--diagnostic-batch must select an existing one-based global batch")
    elif diagnostic_batch_number != 0:
        raise ValueError("--diagnostic-batch requires --mode diagnostic")

    def git(*arguments):
        result = subprocess.run(["git", *arguments], cwd=LOCAL_ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    metadata = dict(git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--porcelain"),
                    command=sys.argv, modal_version=modal.__version__, image_id=image.object_id)
    result = train_ddp.with_options(**allocation).remote(
        str(experiment.path.relative_to(LOCAL_ROOT.resolve())), gpus, mode, metadata, diagnostic_batch_number,
    )
    result_strategy = experiment.training.recipe["execution"]["distributed_strategy"] if mode == "numerics" else "ddp"
    print(f"Modal {result_strategy.upper()} run complete: {result}")
