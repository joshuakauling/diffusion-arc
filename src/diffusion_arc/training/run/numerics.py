"""Bounded CPU comparisons of full gradients and exact model-state fingerprints."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class NumericalCriteria:
    schema_version: int = 1
    steps: int = 3
    loss_rtol: float = 0.001
    gradient_relative_l2_max: float = 0.05
    chunk_elements: int = 1_048_576

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("numerical criteria require schema_version=1")
        if type(self.steps) is not int or self.steps != 3:
            raise ValueError("this diagnostic requires exactly three optimizer steps")
        if type(self.chunk_elements) is not int or not 1 <= self.chunk_elements <= 4_194_304:
            raise ValueError("chunk_elements must be between 1 and 4194304")
        for name in ("loss_rtol", "gradient_relative_l2_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("numerical criteria must specify exactly the supported fields")
        return cls(**payload)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def cpu_chunks(tensor, chunk_elements):
    """Copy bounded views only; do not allocate a complete extra CUDA tensor."""
    tensor = tensor.detach()
    if not tensor.is_contiguous():
        raise ValueError("numerical capture requires contiguous tensors")
    flat = tensor.view(-1)
    for start in range(0, flat.numel(), chunk_elements):
        yield flat[start:start + chunk_elements].to(device="cpu", copy=True)


def _header(name, tensor):
    return json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode() + b"\0"


def state_fingerprint(module, chunk_elements):
    """Hash all parameters and buffers, checking every value for finiteness."""
    digest, count = hashlib.sha256(), 0
    for name, value in module.state_dict().items():
        digest.update(_header(name, value))
        for chunk in cpu_chunks(value, chunk_elements):
            if not bool(torch.isfinite(chunk).all()):
                raise FloatingPointError(f"non-finite state: {name}")
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
            count += chunk.numel()
    return dict(sha256=digest.hexdigest(), elements=count)


def tensor_fingerprint(value, chunk_elements):
    digest = hashlib.sha256(_header("tensor", value))
    for chunk in cpu_chunks(value, chunk_elements):
        if not bool(torch.isfinite(chunk).all()):
            raise FloatingPointError("non-finite flow draw")
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _block(name):
    parts = name.split(".")
    return ".".join(parts[:2]) if len(parts) > 1 and parts[1].isdigit() else parts[0]


@dataclass
class VectorDifference:
    """Accumulate FP64 squared norms and dot products without retaining vectors."""
    reference_squared: float = 0.0
    actual_squared: float = 0.0
    difference_squared: float = 0.0
    dot: float = 0.0
    max_absolute: float = 0.0
    elements: int = 0

    def add(self, reference, actual):
        if reference.shape != actual.shape:
            raise ValueError("gradient comparison shapes differ")
        reference, actual = reference.double(), actual.double()
        if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
            raise FloatingPointError("non-finite gradient in comparison")
        difference = actual - reference
        self.reference_squared += float(reference.square().sum())
        self.actual_squared += float(actual.square().sum())
        self.difference_squared += float(difference.square().sum())
        self.dot += float((reference * actual).sum())
        self.max_absolute = max(self.max_absolute, float(difference.abs().max()))
        self.elements += reference.numel()

    def result(self):
        ref, actual, difference = (math.sqrt(x) for x in
                                   (self.reference_squared, self.actual_squared, self.difference_squared))
        relative = difference / ref if ref else (0.0 if difference == 0 else None)
        return dict(reference_norm=ref, actual_norm=actual, difference_norm=difference,
                    relative_l2=relative, cosine_similarity=max(-1.0, min(1.0, self.dot / (ref * actual)))
                    if ref and actual else None, max_absolute=self.max_absolute, elements=self.elements)


def inspect_gradients(module, chunk_elements, *, save_to=None, compare_to=None):
    """Fingerprint every gradient; optionally save or compare the FP32 reference.

    The binary reference is temporary job-local scratch. Only rank zero needs to
    read it; both DDP ranks independently fingerprint their actual gradients.
    """
    if save_to is not None and compare_to is not None:
        raise ValueError("save and compare are mutually exclusive")
    layout = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or parameter.grad.dtype != torch.float32:
            raise ValueError(f"expected a complete FP32 gradient: {name}")
        layout.append(dict(name=name, shape=list(parameter.shape), elements=parameter.numel()))
    if not layout:
        raise ValueError("no trainable parameters")
    reference = None
    if save_to is not None:
        Path(save_to).mkdir(parents=True, exist_ok=False)
        stream = (Path(save_to) / "gradients.bin").open("xb")
    elif compare_to is not None:
        reference = json.loads((Path(compare_to) / "gradients.json").read_text())
        path = Path(compare_to) / "gradients.bin"
        if reference["layout"] != layout or path.stat().st_size != 4 * sum(x["elements"] for x in layout):
            raise ValueError("reference gradient layout or file size differs")
        stream = path.open("rb")
    else:
        stream = None
    digest, reference_digest = hashlib.sha256(), hashlib.sha256()
    total, blocks, squared_norm = VectorDifference(), {}, 0.0
    try:
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            value = parameter.grad
            digest.update(_header(name, value))
            reference_digest.update(_header(name, value))
            for chunk in cpu_chunks(value, chunk_elements):
                if not bool(torch.isfinite(chunk).all()):
                    raise FloatingPointError(f"non-finite gradient: {name}")
                raw = chunk.numpy().tobytes()
                digest.update(raw)
                squared_norm += float(chunk.double().square().sum())
                if save_to is not None:
                    stream.write(raw)
                elif compare_to is not None:
                    raw_reference = stream.read(len(raw))
                    reference_digest.update(raw_reference)
                    ref = torch.from_numpy(np.frombuffer(raw_reference, dtype=np.float32).copy())
                    total.add(ref, chunk)
                    blocks.setdefault(_block(name), VectorDifference()).add(ref, chunk)
    finally:
        if stream is not None:
            stream.close()
    result = dict(sha256=digest.hexdigest(), norm=math.sqrt(squared_norm),
                  elements=sum(x["elements"] for x in layout))
    if save_to is not None:
        write_json(Path(save_to) / "gradients.json", dict(layout=layout, **result))
    if compare_to is not None:
        if reference_digest.hexdigest() != reference["sha256"]:
            raise ValueError("reference gradient checksum mismatch")
        result.update(comparison=total.result(), blocks={k: v.result() for k, v in blocks.items()})
    return result


def compare_records(reference, distributed, criteria):
    """Apply preregistered criteria; ingestion or identity failures cannot pass."""
    if len(reference) != criteria.steps or len(distributed) != criteria.steps:
        raise ValueError("both phases must complete exactly the configured steps")
    rows, failures = [], []
    for index, (plain, ddp) in enumerate(zip(reference, distributed, strict=True), 1):
        if plain["step"] != index or ddp["step"] != index:
            raise ValueError("missing or reordered optimizer steps")
        peers = ddp["ranks"]
        if len(plain["ranks"]) != 1 or [r["rank"] for r in peers] != [0, 1]:
            raise ValueError("expected one plain process and two DDP ranks")
        expected, actual = plain["loss"], ddp["loss"]
        if not math.isfinite(expected) or not math.isfinite(actual):
            raise ValueError("non-finite reported loss")
        delta = abs(actual - expected)
        relative = delta / abs(expected) if expected else (0.0 if delta == 0 else None)
        checks = dict(loss=relative is not None and relative <= criteria.loss_rtol,
                      rank_parameters=peers[0]["parameters"] == peers[1]["parameters"],
                      frozen=all(r["frozen_unchanged"] and r["frozen_no_grad"] for r in plain["ranks"] + peers),
                      learning_rate=all(r["learning_rates"] == plain["ranks"][0]["learning_rates"] for r in peers))
        for key in ("inputs", "draws"):
            checks[key] = ([v for r in plain["ranks"] for v in r[key]]
                           == [v for r in peers for v in r[key]])
        gradient = None
        if index == 1:
            checks["rank_gradients"] = peers[0]["gradients"]["sha256"] == peers[1]["gradients"]["sha256"]
            gradient = peers[0]["gradients"]["comparison"]
            checks["gradient_l2"] = (gradient["relative_l2"] is not None
                                     and gradient["relative_l2"] <= criteria.gradient_relative_l2_max)
        failures.extend(f"step_{index}/{key}" for key, passed in checks.items() if not passed)
        rows.append(dict(step=index, reference_loss=expected, ddp_loss=actual,
                         absolute_loss_difference=delta, relative_loss_difference=relative,
                         checks=checks, gradient=gradient))
    return dict(event="complete", status="failed" if failures else "passed", optimizer_steps=criteria.steps,
                failures=failures, steps=rows,
                gradient_blocks=distributed[0]["ranks"][0]["gradients"]["blocks"])
