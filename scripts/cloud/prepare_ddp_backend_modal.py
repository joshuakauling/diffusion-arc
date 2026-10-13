#!/usr/bin/env python3
"""Stream the original base FLUX transformer between existing Modal accounts on CPU."""

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import modal


SOURCE_PROFILE = "joshua-kauling01"
SOURCE_ENVIRONMENT = "diffusion-arc"
SOURCE_VOLUME = "diffusion-arc-models"
SOURCE_DIRECTORY = "flux2-klein-base-4b/transformer"
SOURCE_REVISION = "8c44a2fbef88fae175da65df054db0f901aa9747"
TARGET_WORKSPACE = "joshuakauling"
TARGET_VOLUME = "diffusion-arc-training-v2"
TARGET_DIRECTORY = Path("/workspace/models/flux2-klein-base-4b/transformer")
WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
MAX_CONFIG_BYTES = 128 * 1024
MIN_WEIGHT_BYTES = 1 << 30
CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES = 128 * 1024 * 1024


class TransformerTransferError(RuntimeError):
    """Report transfer progress and failure categories without capability URLs."""


def create_download_session():
    """Use Modal's HTTP/TLS setup in a session owned by this transfer's event loop."""
    from modal._utils.http_utils import _http_client_with_tls

    return _http_client_with_tls(timeout=60)


def safe_error_details(error: Exception) -> str:
    """Keep useful error types and numeric codes while excluding URL-bearing text."""
    details = type(error).__name__
    for name in ("status", "errno"):
        value = getattr(error, name, None)
        if type(value) is int:
            details += f" {name}={value}"
    return details


async def download_transformer_blocks(download_urls: tuple[str, ...], target,
                                      expected_size: int) -> None:
    """Stream ordered blocks through Modal's HTTP client with sanitized failures."""
    size, next_progress = 0, PROGRESS_BYTES
    print(f"Transferring original FLUX transformer: {expected_size / (1 << 30):.2f} GiB "
          f"in {len(download_urls)} blocks", flush=True)
    async with create_download_session() as session:
        for index, download_url in enumerate(download_urls, start=1):
            try:
                async with session.get(download_url) as response:
                    if response.status >= 400:
                        raise TransformerTransferError(f"HTTP {response.status}")
                    async for chunk in response.content.iter_chunked(CHUNK_BYTES):
                        size += len(chunk)
                        if size > expected_size:
                            raise TransformerTransferError("Downloaded bytes exceed expected size")
                        target.write(chunk)
            except Exception as error:
                details = str(error) if isinstance(error, TransformerTransferError) else safe_error_details(error)
                raise TransformerTransferError(
                    f"Download block {index}/{len(download_urls)} failed: {details}; "
                    f"received {size}/{expected_size} bytes"
                ) from None
            if size >= next_progress or index == len(download_urls):
                print(f"Transferred {size / (1 << 30):.2f}/{expected_size / (1 << 30):.2f} GiB "
                      f"({index}/{len(download_urls)} blocks)", flush=True)
                next_progress = size + PROGRESS_BYTES
    if size != expected_size:
        raise TransformerTransferError(f"Downloaded transformer is truncated: {size}/{expected_size} bytes")


def reject_environment_credentials() -> None:
    """Keep CLI profile selection authoritative without changing global profiles."""
    if any(os.environ.get(key) for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")):
        raise ValueError("Unset MODAL_TOKEN_ID/MODAL_TOKEN_SECRET: they override --profile")


def validate_model_metadata(config_bytes: bytes, expected_size: int) -> None:
    """Reject implausible weights and malformed or oversized transformer config data."""
    if type(expected_size) is not int or expected_size < MIN_WEIGHT_BYTES:
        raise ValueError("Source transformer weights are missing or too small")
    if not isinstance(config_bytes, bytes) or not 0 < len(config_bytes) <= MAX_CONFIG_BYTES:
        raise ValueError("Source transformer config must be a small JSON file")
    try:
        config = json.loads(config_bytes)
    except (ValueError, UnicodeDecodeError):
        raise ValueError("Source transformer config must contain valid JSON") from None
    if not isinstance(config, dict) or not config:
        raise ValueError("Source transformer config must contain a nonempty JSON object")


async def load_source_model_metadata() -> tuple[bytes, tuple[str, ...], int]:
    """Use explicit local source credentials for read-only Modal 1.5.5 metadata RPCs."""
    from modal.client import _Client
    from modal.config import Config
    from modal_proto import api_pb2

    reject_environment_credentials()
    config = Config()
    settings = {key: config.get(key, profile=SOURCE_PROFILE, use_env=False)
                for key in ("server_url", "token_id", "token_secret")}
    if not settings["token_id"] or not settings["token_secret"]:
        raise ValueError("The source Modal profile requires configured token credentials")
    async with _Client(settings["server_url"], api_pb2.CLIENT_TYPE_CLIENT,
                       (settings["token_id"], settings["token_secret"])) as client:
        volume = await client.stub.VolumeGetOrCreate(api_pb2.VolumeGetOrCreateRequest(
            deployment_name=SOURCE_VOLUME, environment_name=SOURCE_ENVIRONMENT,
            object_creation_type=api_pb2.OBJECT_CREATION_TYPE_UNSPECIFIED,
        ))
        source_config = await client.stub.VolumeGetFile(api_pb2.VolumeGetFileRequest(
            volume_id=volume.volume_id, path=f"{SOURCE_DIRECTORY}/config.json",
            len=MAX_CONFIG_BYTES + 1,
        ))
        weights = await client.stub.VolumeGetFile2(api_pb2.VolumeGetFile2Request(
            volume_id=volume.volume_id, path=f"{SOURCE_DIRECTORY}/{WEIGHTS_NAME}",
        ))
        config_bytes = source_config.data
        if len(config_bytes) != source_config.size:
            raise ValueError("Source transformer config is incomplete")
        validate_model_metadata(config_bytes, weights.size)
        if weights.start != 0 or weights.len != weights.size or not weights.get_urls:
            raise ValueError("Source transformer metadata must cover the complete file")
        return config_bytes, tuple(weights.get_urls), weights.size


def install_transformer(config_bytes: bytes, download_urls: tuple[str, ...], expected_size: int,
                        *, directory: Path = TARGET_DIRECTORY) -> dict:
    """Publish complete streamed weights; reuse matching files and never replace mismatches."""
    validate_model_metadata(config_bytes, expected_size)
    config_path, weights_path = directory / "config.json", directory / WEIGHTS_NAME
    if config_path.exists() and (config_path.stat().st_size != len(config_bytes)
                                 or config_path.read_bytes() != config_bytes):
        raise ValueError("Existing target transformer config differs from the source")
    if weights_path.exists() and (not config_path.is_file()
                                  or weights_path.stat().st_size != expected_size):
        raise ValueError("Existing target transformer weights are incomplete or mismatched")
    status = "already_present"
    if not weights_path.exists():
        if not download_urls or isinstance(download_urls, str):
            raise ValueError("Source transformer download requires HTTPS block URLs")
        for download_url in download_urls:
            url = urlsplit(download_url)
            if url.scheme != "https" or not url.netloc:
                raise ValueError("Source transformer download requires HTTPS block URLs")
        directory.mkdir(parents=True, exist_ok=True)
        partial = weights_path.with_name(WEIGHTS_NAME + ".partial")
        config_partial = config_path.with_name("config.json.partial")
        partial.unlink(missing_ok=True)
        phase = "creating the temporary weights file"
        try:
            with partial.open("xb") as target:
                phase = "downloading weights"
                asyncio.run(download_transformer_blocks(download_urls, target, expected_size))
                phase = "flushing weights"
                target.flush()
                os.fsync(target.fileno())
            if weights_path.exists():
                raise TransformerTransferError("Target transformer weights appeared during transfer")
            phase = "publishing model files"
            if not config_path.exists():
                config_partial.unlink(missing_ok=True)
                config_partial.write_bytes(config_bytes)
                config_partial.rename(config_path)
            partial.rename(weights_path)
            status = "transferred"
        except Exception as error:
            details = str(error) if isinstance(error, TransformerTransferError) else safe_error_details(error)
            raise TransformerTransferError(
                f"Transformer transfer failed while {phase}: {details}; final weights were not published"
            ) from None
        finally:
            partial.unlink(missing_ok=True)
            config_partial.unlink(missing_ok=True)
    receipt = {"source_volume": SOURCE_VOLUME, "source_path": f"{SOURCE_DIRECTORY}/{WEIGHTS_NAME}",
               "source_revision": SOURCE_REVISION, "weight_bytes": expected_size}
    receipt_path = directory / "base_transformer_transfer.json"
    receipt_partial = receipt_path.with_suffix(".json.partial")
    receipt_partial.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    receipt_partial.replace(receipt_path)
    return {"status": status, **receipt}


if modal.is_local():
    reject_environment_credentials()
app = modal.App("diffusion-arc-ddp-backend-preparation")
target_volume = modal.Volume.from_name(TARGET_VOLUME, environment_name="main", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, cpu=(2.0, 2.0), memory=(4096, 4096), timeout=900,
              max_containers=1, retries=0, volumes={"/workspace": target_volume})
def prepare_backend(config_bytes: bytes, download_urls: tuple[str, ...], expected_size: int) -> dict:
    result = install_transformer(config_bytes, download_urls, expected_size)
    target_volume.commit()
    print("Original FLUX transformer ready in target volume", flush=True)
    return result


@app.local_entrypoint()
def main():
    reject_environment_credentials()
    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    if workspace.name != TARGET_WORKSPACE:
        raise ValueError(f"Target Modal workspace must be {TARGET_WORKSPACE}")
    metadata = asyncio.run(asyncio.wait_for(load_source_model_metadata(), timeout=60))
    result = prepare_backend.remote(*metadata)
    print(json.dumps(result, sort_keys=True), flush=True)
