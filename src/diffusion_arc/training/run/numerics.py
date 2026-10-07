"""Bounded CPU comparisons of full gradients and exact model-state fingerprints."""

from dataclasses import dataclass
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.distributed.tensor import DTensor


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
    """Hash parameters and buffers, or this rank's FSDP shards, checking finiteness."""
    digest, count = hashlib.sha256(), 0
    for name, value in module.state_dict().items():
        digest.update(_header(name, value))
        if isinstance(value, DTensor):
            value = value.to_local()
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


def inspect_gradients(module, chunk_elements, *, save_to=None, compare_to=None, compare_to_ddp=None):
    """Capture complete gradients, gathering FSDP one parameter at a time.

    Every FSDP rank must participate in the gathers. Only rank zero reads the
    temporary CPU references; captures never enter the mounted results volume.
    """
    layout = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or parameter.grad.dtype != torch.float32:
            raise ValueError(f"expected a complete FP32 gradient: {name}")
        layout.append(dict(name=name, shape=list(parameter.shape), elements=parameter.numel()))
    if not layout:
        raise ValueError("no trainable parameters")
    digest, squared_norm = hashlib.sha256(), 0.0
    references = {}
    with ExitStack() as stack:
        capture = None
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=False)
            capture = stack.enter_context((Path(save_to) / "gradients.bin").open("xb"))
        for label, directory in (("reference", compare_to), ("ddp", compare_to_ddp)):
            if directory is None:
                continue
            reference = json.loads((Path(directory) / "gradients.json").read_text())
            path = Path(directory) / "gradients.bin"
            if reference["layout"] != layout or path.stat().st_size != 4 * sum(x["elements"] for x in layout):
                raise ValueError("reference gradient layout or file size differs")
            references[label] = dict(metadata=reference, stream=stack.enter_context(path.open("rb")),
                                     digest=hashlib.sha256(), total=VectorDifference(), blocks={})
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            value = parameter.grad
            # Bound extra GPU storage to one parameter, never a full-model gather.
            if isinstance(value, DTensor):
                value = value.full_tensor()
            digest.update(_header(name, value))
            for reference in references.values():
                reference["digest"].update(_header(name, value))
            for chunk in cpu_chunks(value, chunk_elements):
                if not bool(torch.isfinite(chunk).all()):
                    raise FloatingPointError(f"non-finite gradient: {name}")
                raw = chunk.numpy().tobytes()
                digest.update(raw)
                squared_norm += float(chunk.double().square().sum())
                if capture is not None:
                    capture.write(raw)
                for reference in references.values():
                    raw_reference = reference["stream"].read(len(raw))
                    reference["digest"].update(raw_reference)
                    ref = torch.from_numpy(np.frombuffer(raw_reference, dtype=np.float32).copy())
                    reference["total"].add(ref, chunk)
                    reference["blocks"].setdefault(_block(name), VectorDifference()).add(ref, chunk)
            del value
    result = dict(sha256=digest.hexdigest(), norm=math.sqrt(squared_norm),
                  elements=sum(x["elements"] for x in layout))
    if save_to is not None:
        write_json(Path(save_to) / "gradients.json", dict(layout=layout, **result))
    for label, reference in references.items():
        if reference["digest"].hexdigest() != reference["metadata"]["sha256"]:
            raise ValueError("reference gradient checksum mismatch")
        prefix = "" if label == "reference" else "ddp_"
        result[prefix + "comparison"] = reference["total"].result()
        result[prefix + "blocks"] = {k: v.result() for k, v in reference["blocks"].items()}
    return result


def compare_records(reference, distributed, criteria, *, strategy="ddp", reference_strategy="single_process"):
    """Apply preregistered criteria; ingestion or identity failures cannot pass."""
    if len(reference) != criteria.steps or len(distributed) != criteria.steps:
        raise ValueError("both phases must complete exactly the configured steps")
    if strategy not in ("ddp", "fsdp2") or reference_strategy not in ("single_process", "ddp"):
        raise ValueError("unsupported numerical comparison strategies")
    rows, failures = [], []
    for index, (plain, ddp) in enumerate(zip(reference, distributed, strict=True), 1):
        if plain["step"] != index or ddp["step"] != index:
            raise ValueError("missing or reordered optimizer steps")
        peers = ddp["ranks"]
        reference_ranks = [0] if reference_strategy == "single_process" else [0, 1]
        if [r["rank"] for r in plain["ranks"]] != reference_ranks or [r["rank"] for r in peers] != [0, 1]:
            raise ValueError("unexpected reference or distributed ranks")
        expected, actual = plain["loss"], ddp["loss"]
        if not math.isfinite(expected) or not math.isfinite(actual):
            raise ValueError("non-finite reported loss")
        delta = abs(actual - expected)
        relative = delta / abs(expected) if expected else (0.0 if delta == 0 else None)
        checks = dict(loss=relative is not None and relative <= criteria.loss_rtol,
                      frozen=all(r["frozen_unchanged"] and r["frozen_no_grad"] for r in plain["ranks"] + peers),
                      learning_rate=all(r["learning_rates"] == plain["ranks"][0]["learning_rates"] for r in peers))
        if strategy == "ddp":
            checks["rank_parameters"] = peers[0]["parameters"] == peers[1]["parameters"]
        for key in ("inputs", "draws"):
            checks[key] = ([v for r in plain["ranks"] for v in r[key]]
                           == [v for r in peers for v in r[key]])
        gradient = None
        if index == 1:
            if strategy == "ddp":
                checks["rank_gradients"] = peers[0]["gradients"]["sha256"] == peers[1]["gradients"]["sha256"]
            key = "ddp_comparison" if reference_strategy == "ddp" else "comparison"
            gradient = peers[0]["gradients"][key]
            checks["gradient_l2"] = (gradient["relative_l2"] is not None
                                     and gradient["relative_l2"] <= criteria.gradient_relative_l2_max)
        failures.extend(f"step_{index}/{key}" for key, passed in checks.items() if not passed)
        rows.append(dict(step=index, reference_loss=expected, **{f"{strategy}_loss": actual},
                         absolute_loss_difference=delta, relative_loss_difference=relative,
                         checks=checks, gradient=gradient))
    return dict(event="complete", status="failed" if failures else "passed", optimizer_steps=criteria.steps,
                failures=failures, steps=rows,
                gradient_blocks=distributed[0]["ranks"][0]["gradients"][
                    "ddp_blocks" if reference_strategy == "ddp" else "blocks"])
