#!/usr/bin/env python3
"""Capture three warmed-up B3 updates with Nsight Systems on one Modal H200."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import modal

REMOTE_REPOSITORY_ROOT = Path("/opt/diffusion-arc")
LOCAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPOSITORY_ROOT
VOLUME_ROOT = Path("/mnt/diffusion-arc-models")
DEFAULT_CONFIG = "configs/training/profiling/single_gpu_b3_torch_profile.json"
# Pinned CLI package from NVIDIA's Ubuntu 24.04 developer-tools repository.
NSYS_PACKAGE = "NsightSystems-linux-cli-public-2025.6.1.190-3689520.deb"
NSYS_URL = "https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/" + NSYS_PACKAGE
RUN_TIMEOUT_SECONDS = 10 * 60

sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT / "src"))
from scripts.cloud._training_image import training_image, FA3_REVISION, CUDA_IMAGE


def _ignore_data(path: Path) -> bool:
    included = Path("benchmarks/single_gpu_b0")
    return not (path.is_relative_to(included) or included.is_relative_to(path))


def validate_config(repository: Path, config: str) -> Path:
    """Reject unrelated workloads before allocating a GPU."""
    path = (repository / config).resolve()
    if Path(config).is_absolute() or not path.is_relative_to(repository.resolve()):
        raise ValueError("config must be a repository-relative path")
    from diffusion_arc.training.config import load_training_config

    validated = load_training_config(path, expected_type="training_run")
    payload = json.loads(path.read_text())
    expected = (
        validated.seed == 167, validated.device == "cuda:0",
        validated.nvarc_max_tokens == 134400,
        validated.loop.max_optimizer_steps == 7,
        validated.loop.optimizer_batch_size == 48,
        validated.loop.min_context_pairs_before_target == 2,
        validated.loop.vae_microbatch_size == 48,
        validated.flux_packing.max_tokens == 12288,
        validated.benchmark is None,
        validated.optimizer.learning_rate == 1e-5,
        validated.optimizer.beta1 == 0.9, validated.optimizer.beta2 == 0.999,
        validated.optimizer.epsilon == 1e-8, validated.optimizer.weight_decay == 0.01,
        payload["scheduler"] == {"type": "cosine", "warmup_steps": 213, "total_steps": 7245},
        payload["execution"]["distributed_strategy"] == "single_process",
        payload["loss"]["type"] == "velocity_mse",
        payload["flow"]["time_distribution"] == "uniform",
        payload["flow"]["rng_scheme"] == "legacy_stream_v1",
    )
    if not all(expected):
        raise ValueError("Nsight launcher requires the seven-update B3 profiler configuration")
    expected_models = {"nvarc_dtype": "bfloat16", "vae_dtype": "float32",
                       "flux_parameter_dtype": "float32", "flux_compute_dtype": "bfloat16"}
    if any(payload["models"][key] != value for key, value in expected_models.items()):
        raise ValueError("Nsight launcher requires B3 model precision")
    for key, name in (("nvarc_checkpoint", "nvarc-qwen3-4b-grids15-sft139"),
                      ("flux_checkpoint", "flux2-klein-base-4b")):
        if Path(payload["models"][key]).is_absolute() or getattr(validated.models, key) != (repository / "checkpoints" / name).resolve():
            raise ValueError("Nsight launcher requires repository B3 checkpoint paths")
    bundle = repository / "data/benchmarks/single_gpu_b0"
    if (validated.data.split_directory != (bundle / "splits/train").resolve()
            or validated.data.view_name != "train" or validated.data.shuffle
            or validated.data.num_workers != 2 or validated.data.prefetch_factor != 8):
        raise ValueError("Nsight launcher requires the B0 training bundle and B3 loader settings")
    for relative in ("config.json", "manifest.json", "train.json", "validation.json", "augmented_keys/train.json"):
        required = validated.data.split_directory / relative
        if not required.is_file():
            raise FileNotFoundError(f"missing B0 split file: {required}")
    for task_id in json.loads((validated.data.split_directory / "train.json").read_text()):
        required = bundle / "raw/nvarc_train_arc_agi_1_training_only/training" / f"{task_id}.json"
        if not required.is_file():
            raise FileNotFoundError(f"missing B0 raw task: {required}")
    return path


def profile_command(config: Path, output: Path) -> list[str]:
    # These options are documented for the pinned 2025.6 CLI. GPU sampling stays
    # limited to the GPUs assigned to this container through CUDA visibility.
    return ["nsys", "profile", "--trace=cuda,nvtx",
            "--sample=none", "--cpuctxsw=none",
            "--python-sampling=false", "--cuda-memory-usage=false",
            "--gpu-metrics-devices=cuda-visible", "--gpu-metrics-frequency=10000",
            "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--kill=none",
            f"--output={output / 'trace'}", sys.executable, "-B",
            str(REMOTE_REPOSITORY_ROOT / "scripts/profile_single_gpu.py"),
            "--config", str(config), "--profiler", "nsys"]


app = modal.App("diffusion-arc-nsys-profile")
model_volume = modal.Volume.from_name("diffusion-arc-models", create_if_missing=False)
image = (
    training_image(LOCAL_REPOSITORY_ROOT)
    .apt_install("curl", "ca-certificates")
    .run_commands(
        f"curl --fail --location --retry 3 {NSYS_URL} --output /tmp/nsys.deb",
        "apt-get update && apt-get install -y /tmp/nsys.deb && rm /tmp/nsys.deb",
        "nsys --version",
    )
    .workdir(REMOTE_REPOSITORY_ROOT)
)
for directory in ("src", "scripts", "configs"):
    image = image.add_local_dir(LOCAL_REPOSITORY_ROOT / directory,
                                remote_path=str(REMOTE_REPOSITORY_ROOT / directory),
                                ignore=["**/__pycache__/**", "**/*.pyc"])
image = image.add_local_dir(LOCAL_REPOSITORY_ROOT / "data",
                            remote_path=str(REMOTE_REPOSITORY_ROOT / "data"), ignore=_ignore_data)


def _capture(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    return result.stdout.strip()



def _preflight(output: Path) -> list[dict]:
    """Save capability diagnostics; a failed command never disables a collector."""
    diagnostics = []
    for name, command in (
        ("environment", ["nsys", "status", "--environment"]),
        ("gpu-metrics-devices", ["nsys", "profile", "--gpu-metrics-devices=help"]),
        ("gpu-metrics-sets", ["nsys", "profile", "--gpu-metrics-devices=cuda-visible",
                              "--gpu-metrics-set=help"]),
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        filename = f"preflight-{name}.log"
        report = result.stdout + result.stderr
        (output / filename).write_text(report)
        # Retain unsupported CPU checks as evidence; these collectors are disabled.
        cpu_checks = (
            "Linux perf_event_open syscall available",
            "Sampling trigger event available",
            "CPU Profiling Environment (process-tree)",
        )
        cpu_failures = [line.strip() for line in report.splitlines()
                    if name == "environment" and any(check in line for check in cpu_checks)
                    and line.rsplit(":", 1)[-1].strip().lower().startswith("fail")]
        diagnostics.append({"command": command, "returncode": result.returncode,
                            "log": filename, "disabled_cpu_failures": cpu_failures})
    return diagnostics


def _capture_evidence(log: str) -> dict:
    """Report requested collectors separately from data that has been inspected.

    CLI success and a nonempty report establish neither counter availability nor
    event coverage. Preserve all messages and require report inspection rather
    than treating the absence of a known warning string as evidence of support.
    """
    messages = [line for line in log.splitlines()
                if any(word in line.lower() for word in
                       ("warning", "error", "unsupported", "not supported", "permission", "disabled"))]
    return {
        "status": "requires_report_review",
        "requested": ["cuda_api_and_kernels", "nvtx", "gpu_metrics"],
        "disabled": ["cpu_sampling", "cpu_context_switches", "python_sampling",
                     "python_gil", "os_runtime", "file_access", "cuda_memory_usage"],
        "verified": [],
        "diagnostic_messages": messages,
        "limitations": [
            "Collector flags express requests, not proof that data was collected.",
            "Inspect report Diagnostics Summary and each requested timeline track before interpreting results.",
            "Available SM, Tensor and DRAM counters depend on the GPU metric set and permissions.",
            "No samples/events can mean no activity in this short window or unavailable collection.",
            "CPU visibility is limited to CUDA API calls and NVTX launch ranges; CPU/Python sampling and OS runtime tracing are disabled.",
        ],
    }

def _run_profile(config: Path, output: Path) -> str:
    """Persist diagnostic artifacts even when profiling fails or times out."""
    output.mkdir(parents=True, exist_ok=False)
    command = profile_command(config, output)
    metadata = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                "command": command, "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "fa3_revision": FA3_REVISION, "cuda_image": CUDA_IMAGE,
                "nsys_package": NSYS_PACKAGE, "warmup_updates": 4, "captured_updates": 3,
                "source_sha256": {str(path.relative_to(REMOTE_REPOSITORY_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for folder in ("src", "scripts", "configs", "data/benchmarks/single_gpu_b0")
                                  for path in sorted((REMOTE_REPOSITORY_ROOT / folder).rglob("*"))
                                  if path.is_file() and path.suffix in (".py", ".json")},
                "python": sys.version, "timeout_seconds": RUN_TIMEOUT_SECONDS}
    (output / "config.json").write_bytes(config.read_bytes())
    started = time.monotonic()
    try:
        metadata["collection"] = _capture_evidence("")
        metadata["preflight"] = _preflight(output)
        if any(item["returncode"] != 0 for item in metadata["preflight"]):
            raise RuntimeError("Nsight capability preflight failed; inspect preflight logs before launching training")
        metadata["nsys_version"] = _capture(["nsys", "--version"])
        metadata["gpu"] = _capture(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"])
        metadata["packages"] = _capture([sys.executable, "-m", "pip", "freeze"])
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                           PYTHONPATH=str(REMOTE_REPOSITORY_ROOT / "src"),
                           DIFFUSION_ARC_PROFILER_OUTPUT_ROOT=str(output))
        with (output / "run.log").open("w") as log:
            process = subprocess.Popen(command, cwd=REMOTE_REPOSITORY_ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=RUN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                # Stop the complete process group, including the training child.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                raise
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        report = output / "trace.nsys-rep"
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError("Nsight did not produce a nonempty trace.nsys-rep; inspect run.log")
        training = json.loads((output / "training-summary.json").read_text())
        if (training.get("status") != "complete" or training.get("completed_updates") != 7
                or training.get("recorded_updates") != 3):
            raise RuntimeError("Nsight training did not complete seven updates and capture three")
        metadata["status"] = "capture_complete_unverified"
        metadata["report_bytes"] = report.stat().st_size
        return str(report)
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if (output / "run.log").is_file():
            metadata["collection"] = _capture_evidence((output / "run.log").read_text(errors="replace"))
            for message in metadata["collection"]["diagnostic_messages"]:
                print(f"Nsight diagnostic: {message}", flush=True)
        print("Requested collector coverage is unverified; inspect preflight logs and report Diagnostics Summary.", flush=True)
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "run-summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
        model_volume.commit()
        print(f"Persistent Nsight artifacts: {output}", flush=True)


@app.function(image=image, gpu="H200", cpu=8.0, memory=64 * 1024, timeout=15 * 60,
              max_containers=1, retries=0, volumes={str(VOLUME_ROOT): model_volume})
def profile_single_gpu(config: str = DEFAULT_CONFIG) -> str:
    path = validate_config(REMOTE_REPOSITORY_ROOT, config)
    checkpoint_link = REMOTE_REPOSITORY_ROOT / "checkpoints"
    if checkpoint_link.is_symlink():
        if checkpoint_link.resolve() != VOLUME_ROOT:
            raise RuntimeError("unexpected checkpoint symlink target")
    elif checkpoint_link.exists():
        raise RuntimeError("checkpoint path already exists")
    else:
        checkpoint_link.symlink_to(VOLUME_ROOT, target_is_directory=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return _run_profile(path, VOLUME_ROOT / "results/nsys-profiler" / run_id)


@app.local_entrypoint()
def main(config: str = DEFAULT_CONFIG) -> None:
    validate_config(LOCAL_REPOSITORY_ROOT, config)
    print(f"Modal Nsight report: {profile_single_gpu.remote(config)}")
