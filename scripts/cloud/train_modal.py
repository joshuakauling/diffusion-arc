#!/usr/bin/env python3
"""Prepare on CPU, then train two epochs in a bounded Modal H200 Sandbox.

Run with ordinary Python, not `modal run`. --dry-run is entirely local.
The remote entrypoint owns training and uploads, so a disconnected terminal
cannot leave an idle GPU behind. Full checkpoints stay on the named Volume.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.cloud import train_runpod as shared

DEFAULT_CONFIG = ROOT / "configs/training/arc_agi_1_epoch5_high_noise.json"
REMOTE_ROOT = "/opt/diffusion-arc"
WORKSPACE = Path("/workspace")
SECRETS = ("HF_TOKEN", "WANDB_API_KEY", "KAGGLE_API_TOKEN")
PREPARE_SECONDS = 2 * 60 * 60
TRAIN_SECONDS = 8 * 60 * 60
SHUTDOWN_SECONDS = 5 * 60


def plan_run(args) -> dict:
    """Resolve the two-epoch experiment without importing Modal or accessing accounts."""
    from diffusion_arc.training.config import load_training_config

    config_path = args.config.resolve()
    if not config_path.is_relative_to(ROOT / "configs"):
        raise ValueError("--config must be inside this repository's configs directory")
    config = load_training_config(config_path, expected_type="training_run")
    shared.require_epoch_launch_config(config)
    recipe = config.recipe_dict()
    if (recipe["deployment"]["provider"] != "modal"
            or recipe["deployment"]["gpu_type"] != "H200"
            or recipe["models"]["initialization"] != "transformer_weights"
            or recipe["models"]["resume_from"] is not None):
        raise ValueError("Modal launcher requires fresh H200 transformer_weights initialization")
    if not recipe["logging"].get("resume_parent_run"):
        raise ValueError("Both epochs must continue the parent W&B run")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,55}", args.run_name):
        raise ValueError("--run-name must contain 1–56 letters, digits, underscores or hyphens")
    first = recipe["models"]["completed_epochs"] + 1
    return {
        "schema_version": 1, "run_name": args.run_name,
        "profile": args.profile, "expected_workspace": args.expected_workspace,
        "environment": args.environment, "volume": args.volume, "secret": args.secret,
        "config": str(config_path.relative_to(ROOT)),
        "execution": {"strategy": recipe["execution"]["distributed_strategy"],
                      "gpu_count": recipe["deployment"]["gpu_count"]},
        "output_repo": recipe["models"]["transformer_source"]["repo_id"],
        "transformer_source": recipe["models"]["transformer_source"],
        "frozen_sources": {"nvarc": shared.NVARC_HANDLE, "vae_repo": shared.FLUX_REPO,
                           "vae_revision": shared.FLUX_REVISION},
        "wandb": {key: recipe["logging"][key] for key in ("entity", "project", "parent_run_id")},
        "epochs": [{"epoch": epoch, "run_name": f"{args.run_name}-epoch{epoch}",
                    "seed": config.seed + offset,
                    "final_step": epoch * config.epoch_optimizer_steps,
                    "optimizer_reset": offset == 0} for offset, epoch in enumerate((first, first + 1))],
        "limits": {"prepare_seconds": PREPARE_SECONDS, "train_seconds": TRAIN_SECONDS},
        "checkpoint_budget_gib": 2 * 55 * len(config.checkpoint_steps) + 25,
        "required_secrets": list(SECRETS),
    }


def connect_account(profile: str, expected_workspace: str):
    """Verify the explicitly selected account before creating objects or building images."""
    if any(os.environ.get(key) for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")):
        raise ValueError("Unset MODAL_TOKEN_ID/MODAL_TOKEN_SECRET: they override --profile")
    if "modal" in sys.modules:
        raise RuntimeError("Start this launcher with ordinary Python in a fresh process")
    os.environ["MODAL_PROFILE"] = profile
    modal = importlib.import_module("modal")
    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    if workspace.name != expected_workspace:
        raise ValueError(f"Modal account mismatch: expected {expected_workspace}, found {workspace.name}")
    print(f"Verified Modal workspace: {workspace.name}; profile: {profile}", flush=True)
    return modal


def phase_commands(plan: dict, phase: str, *, root: Path, workspace: Path) -> list[list[str]]:
    """Prepare assets on CPU; require their reuse when the GPU worker starts."""
    first, second = plan["epochs"]
    first_command = [sys.executable, str(root / "scripts/cloud/finetune_runpod.py"),
                     "--config", str(root / plan["config"]), "--workspace", str(workspace),
                     "--run-name", first["run_name"], "--output-repo", plan["output_repo"],
                     "--wandb-project", plan["wandb"]["project"]]
    if plan["wandb"]["entity"]:
        first_command += ["--wandb-entity", plan["wandb"]["entity"]]
    if phase == "prepare":
        return [[*first_command, "--prepare-only", "--cpu-preparation"]]
    return [[*first_command, "--use-prepared"],
            [sys.executable, str(root / "scripts/cloud/continue_runpod.py"),
             "--workspace", str(workspace), "--from-run", first["run_name"],
             "--run-name", second["run_name"], "--augmentation-seed", str(second["seed"]),
             "--output-repo", plan["output_repo"]]]


def run_command(command: list[str], *, log, timeout: float) -> None:
    """Stream a child to console and persistent log; bound its whole process group."""
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               start_new_session=True)

    def forward():
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)

    reader = threading.Thread(target=forward, daemon=True)
    reader.start()
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        reader.join(timeout=5)
        process.stdout.close()


def worker(plan: dict, phase: str, *, workspace: Path = WORKSPACE) -> int:
    """Own the remote lifecycle through uploads and exit, even without a local client."""
    job = workspace / "modal-jobs" / plan["run_name"]
    job.mkdir(parents=True, exist_ok=True)
    status_path = job / f"{phase}.json"
    status = {"status": "running", "phase": phase, "started_at": utc_now()}
    try:
        if phase == "train":
            prepared = json.loads((job / "prepare.json").read_text())
            if prepared["status"] != "succeeded" or prepared["plan"] != plan:
                raise ValueError("CPU preparation must complete with exactly this launch plan")
        shared.require_secrets(*SECRETS)
        # Existing launchers also reject a new attempt over an incomplete training run.
        shared.write_json(job / "plan.json", plan)
        shared.write_json(status_path, {**status, "plan": plan})
        deadline = time.monotonic() + plan["limits"][f"{phase}_seconds"] - SHUTDOWN_SECONDS
        with (job / f"{phase}.log").open("a", buffering=1) as log:
            for index, command in enumerate(phase_commands(plan, phase, root=ROOT, workspace=workspace)):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{phase} exceeded its total time limit")
                status["command_index"] = index
                shared.write_json(status_path, {**status, "plan": plan})
                run_command(command, log=log, timeout=remaining)
        status["status"] = "succeeded"
        return 0
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_at"] = utc_now()
        shared.write_json(status_path, {**status, "plan": plan})
        # v2 Volumes support explicit commits via sync, without account tokens in the worker.
        try:
            subprocess.run(["sync", str(workspace)], check=True, timeout=SHUTDOWN_SECONDS - 60)
        except Exception as error:
            status.update(status="failed", commit_error=f"{type(error).__name__}: {error}")
            shared.write_json(status_path, {**status, "plan": plan})
            raise
        print(f"Modal {phase}: {status['status']}. Entrypoint exits; Sandbox terminates.", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def follow(sandbox) -> int:
    """Follow logs without depending on stream EOF to notice container completion."""
    def forward(stream):
        try:
            for line in stream:
                print(line, end="", flush=True)
        except Exception as error:
            print(f"Log stream interrupted ({type(error).__name__}); checking Sandbox status", flush=True)

    readers = [threading.Thread(target=forward, args=(stream,), daemon=True)
               for stream in (sandbox.stdout, sandbox.stderr)]
    for reader in readers:
        reader.start()
    while (code := sandbox.poll()) is None:
        time.sleep(2)
    for reader in readers:
        reader.join(timeout=2)
    # The entrypoint already ended; this also makes explicit cleanup idempotent.
    sandbox.terminate(wait=True)
    return code


def phase_resources(plan: dict, phase: str) -> dict:
    """Reserve CPU preparation once and one training resource share per GPU."""
    execution = plan["execution"]
    count = execution["gpu_count"]
    strategy = execution["strategy"]
    if (strategy not in ("single_process", "ddp", "fsdp2") or type(count) is not int or count < 1
            or (strategy == "single_process" and count != 1)):
        raise ValueError("Invalid training strategy or GPU count in launch plan")
    if phase == "prepare":
        return {"cpu": (4, 4), "memory": (32768, 32768)}
    if phase != "train":
        raise ValueError(f"Unsupported launch phase: {phase}")
    return {"gpu": "H200" if count == 1 else f"H200:{count}",
            "cpu": (8 * count, 8 * count), "memory": (65536 * count, 65536 * count)}


def execute(plan: dict) -> None:
    """Build on CPU, finish CPU preparation, then launch one self-terminating H200 job."""
    resources = {phase: phase_resources(plan, phase) for phase in ("prepare", "train")}
    modal = connect_account(plan["profile"], plan["expected_workspace"])
    secret = modal.Secret.from_name(plan["secret"], environment_name=plan["environment"],
                                    required_keys=list(SECRETS))
    secret.hydrate()  # Fail missing credentials before any build or compute allocation.
    from scripts.cloud._training_image import training_image
    from scripts.cloud.launch_runpod import source_snapshot

    content, source_snapshot_manifest = source_snapshot(ROOT)
    if plan["config"] not in source_snapshot_manifest["files"]:
        raise ValueError("Config is missing from the source snapshot")
    plan = {**plan, "source": source_snapshot_manifest}
    receipt = ROOT / "artifacts/modal-launches" / f"{plan['run_name']}-{time.time_ns()}"
    receipt.mkdir(parents=True)
    (receipt / "source.tar.gz").write_bytes(content)
    shared.write_json(receipt / "launch.json", plan)
    volume = modal.Volume.from_name(plan["volume"], environment_name=plan["environment"],
                                    create_if_missing=True, version=2)
    app = modal.App.lookup("diffusion-arc-training", environment_name=plan["environment"],
                           create_if_missing=True)
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
            archive.extractall(snapshot, filter="data")
        image = (training_image(snapshot).pip_install("kagglehub==1.0.2")
                 .env({"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0",
                       "PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}",
                       "HF_HOME": "/workspace/.cache/huggingface",
                       "KAGGLEHUB_CACHE": "/workspace/.cache/kagglehub"})
                 .workdir(REMOTE_ROOT).add_local_dir(snapshot, REMOTE_ROOT))
        for phase in ("prepare", "train"):
            with modal.enable_output():
                sandbox = modal.Sandbox.create(
                    "python", f"{REMOTE_ROOT}/scripts/cloud/train_modal.py", "worker",
                    "--phase", phase, "--plan-json", json.dumps(plan),
                    # Volumes lack distributed file locks. One server-side name serializes
                    # all preparation/training writers to this Volume, across run names.
                    app=app, name="arc-train-" + hashlib.sha256(plan["volume"].encode()).hexdigest()[:20],
                    image=image,
                    secrets=[secret], volumes={str(WORKSPACE): volume},
                    timeout=plan["limits"][f"{phase}_seconds"], **resources[phase])
            shared.write_json(receipt / f"{phase}.json", {"sandbox_id": sandbox.object_id})
            print(f"Modal {phase}: {sandbox.object_id}; receipt: {receipt}", flush=True)
            print("Closing this terminal does not stop this Sandbox. It exits on completion or timeout.", flush=True)
            if phase == "prepare":
                print("Keep this terminal connected until the H200 stage has been launched.", flush=True)
            try:
                code = follow(sandbox)
                if code:
                    raise RuntimeError(f"Modal {phase} failed (exit {code}); logs/checkpoints remain on {plan['volume']}")
            finally:
                sandbox.detach()
    print("Both epochs and HF uploads completed. CPU and GPU Sandboxes terminated.", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Prepare on CPU and train two more epochs")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--run-name", default="arc1-epochs5-6")
    run.add_argument("--volume", default="diffusion-arc-training-v2")
    run.add_argument("--secret", default="custom-secret")
    run.add_argument("--dry-run", action="store_true", help="Print the plan; no Modal import, API calls or downloads")
    commands = [run]
    for name in ("logs", "stop"):
        command = sub.add_parser(name)
        command.add_argument("--sandbox-id", required=True)
        commands.append(command)
    for command in commands:
        command.add_argument("--profile", required=True)
        command.add_argument("--expected-workspace", required=True)
        command.add_argument("--environment", required=True)
    internal = sub.add_parser("worker", help=argparse.SUPPRESS)
    internal.add_argument("--phase", choices=("prepare", "train"), required=True)
    internal.add_argument("--plan-json", required=True)
    args = parser.parse_args(argv)
    if args.command == "worker":
        return worker(json.loads(args.plan_json), args.phase)
    if args.command == "run":
        plan = plan_run(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
        else:
            execute(plan)
        return 0
    modal = connect_account(args.profile, args.expected_workspace)
    sandbox = modal.Sandbox.from_id(args.sandbox_id)
    try:
        if args.command == "stop":
            sandbox.terminate(wait=True)
            return 0
        return follow(sandbox)
    finally:
        sandbox.detach()


if __name__ == "__main__":
    raise SystemExit(main())
