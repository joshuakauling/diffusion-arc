#!/usr/bin/env python3
"""Download pinned initial weights and write the manifest consumed by the Pod launcher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory


FLUX_REPO = "black-forest-labs/FLUX.2-klein-base-4B"
FLUX_REVISION = "8c44a2fbef88fae175da65df054db0f901aa9747"
NVARC_HANDLE = "sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
NVARC_WEIGHT_SIZES = {
    "model-00001-of-00002.safetensors": 4_996_836_472,
    "model-00002-of-00002.safetensors": 2_270_397_024,
}


def file_inventory(directory: Path) -> dict:
    """Record every regular model file; incomplete downloads fail before image publication."""
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected model symlink: {path}")
        if path.is_file():
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            files[path.relative_to(directory).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest}
    if not files:
        raise ValueError(f"Empty model directory: {directory}")
    return files


def build_models(destination: Path) -> dict:
    """Download to temporary caches, retaining only model assets and provenance in the image."""
    import os
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=False)
    nvarc = destination / "nvarc-qwen3-4b-grids15-sft139"
    flux = destination / "flux2-klein-base-4b"
    with TemporaryDirectory() as cache:
        os.environ["KAGGLEHUB_CACHE"] = str(Path(cache) / "kaggle")
        import kagglehub

        print("Downloading pinned NVARC initial weights", flush=True)
        source = Path(kagglehub.model_download(NVARC_HANDLE))
        shutil.copytree(source, nvarc, ignore=shutil.ignore_patterns(".cache", ".git"))
        for name, size in NVARC_WEIGHT_SIZES.items():
            if (nvarc / name).stat().st_size != size:
                raise ValueError(f"Unexpected NVARC weight size: {name}")
        print("Downloading pinned FLUX transformer and VAE", flush=True)
        snapshot_download(
            FLUX_REPO, revision=FLUX_REVISION, local_dir=flux, cache_dir=str(Path(cache) / "hf"),
            token=os.environ["HF_TOKEN"], allow_patterns=[
                "transformer/*.json", "transformer/*.safetensors", "vae/*.json", "vae/*.safetensors",
                "LICENSE*", "NOTICE*", "README.md",
            ],
        )
    shutil.rmtree(flux / ".cache", ignore_errors=True)
    for root, configs in ((nvarc, ("config.json", "tokenizer.json", "tokenizer_config.json",
                                  "model.safetensors.index.json")),
                          (flux, ("transformer/config.json", "vae/config.json"))):
        for name in configs:
            json.loads((root / name).read_text())
    for root in (nvarc, flux / "transformer", flux / "vae"):
        indices = list(root.glob("*.safetensors.index.json"))
        if indices:
            for index in indices:
                shards = set(json.loads(index.read_text())["weight_map"].values())
                if not shards or any(not (root / shard).is_file() for shard in shards):
                    raise ValueError(f"Incomplete model shards: {root.name}")
        elif not any(root.glob("*.safetensors")):
            raise ValueError(f"Missing model weights: {root.name}")
    manifest = {
        "schema_version": 1,
        "nvarc": {"handle": NVARC_HANDLE, "path": nvarc.name, "files": file_inventory(nvarc)},
        "flux": {"repo_id": FLUX_REPO, "revision": FLUX_REVISION, "path": flux.name,
                 "files": file_inventory(flux)},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("Initial model bundle complete: " + str(sum(
        item["bytes"] for component in ("nvarc", "flux") for item in manifest[component]["files"].values()
    )) + " bytes", flush=True)
    return manifest


if __name__ == "__main__":
    build_models(Path(sys.argv[1]))
