"""Compare the real Trainer with a global-loss oracle, then on two CPU ranks.

Use tiny FP32 FLUX blocks, explicit CPU attention and frozen encoder doubles.
Fixed global flow draws and small prescribed packs isolate DDP reduction from
data assignment, endpoint sampling and the production packing heuristic.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion_arc.model.contracts import NvarcBatchOutput
from diffusion_arc.model.flux.packing import PackedFluxForward, predict_packed_flux_velocity
from diffusion_arc.model.flux.transformer import Flux2Transformer2DModel
from diffusion_arc.training.step.encoding import encode_training_batch
from diffusion_arc.training.step.flow_sampling import sample_endpoint_flow_batches, select_flow_batch
from diffusion_arc.training.step.flux_batching import FluxTargetBatch
from diffusion_arc.training.step.flux_microbatch import build_flux_microbatch
from diffusion_arc.training.step.loss import RegionLossWeights, RegionVelocityLoss, velocity_mse
from diffusion_arc.training.step.lr_schedule import LinearWarmupConstantLR
from diffusion_arc.training.trainer import Trainer
from tests.model.test_flux_packing import SegmentedReference
from tests.training.test_trainer import _sample


# Set before execution; the oracle and trainer use different FP32 sum orders.
GRAD_TOLERANCE = dict(atol=3e-6, rtol=3e-4)
PARAM_TOLERANCE = dict(atol=2e-6, rtol=2e-5)
LOSS_TOLERANCE = dict(atol=3e-6, rtol=3e-5)


class FrozenText(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1), requires_grad=False)

    def forward(self, input_ids, sequence_offsets):
        states = torch.sin(input_ids.float()[:, None] + torch.arange(12)) * self.scale
        offsets = sequence_offsets.tolist()
        return NvarcBatchOutput(tuple(states[a:b].clone() for a, b in zip(offsets, offsets[1:])))


class FrozenVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2), requires_grad=False)

    def forward(self, images):
        values = images.mean(dim=(1, 2, 3)) * self.scale
        return values[:, None, None].expand(-1, 900, 128).clone()


def _render(grid):
    return torch.full((3, 2, 2), float(grid[0][0]) / 10)


def _model(initial=None):
    model = Flux2Transformer2DModel(
        in_channels=128, out_channels=128, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=1, joint_attention_dim=12,
        axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False,
    ).train()
    if initial is not None:
        model.load_state_dict(initial)
    return model


def _objective(name):
    return velocity_mse if name == "mse" else RegionVelocityLoss(RegionLossWeights())


def _plan(prefixes, _policy=None, **_):
    return tuple(FluxTargetBatch(tuple(range(i, min(i + 2, len(prefixes)))),
                                sum(n + 900 for n in prefixes[i:i + 2]))
                 for i in range(0, len(prefixes), 2))


def _global_plan(frame):
    batches, offset = [], 0
    for source in frame["sources"]:
        for batch in _plan(source.prefix_token_ends):
            batches.append(replace(batch, target_indices=tuple(offset + i for i in batch.target_indices)))
        offset += len(source.target_grids)
    return tuple(batches)


def _encode(sources):
    with patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render):
        return encode_training_batch(sources, nvarc_encoder=FrozenText(), vae_encoder=FrozenVae())


def _frames():
    generator = torch.Generator().manual_seed(168)
    frames = []
    for step, counts in enumerate(((1, 3), (4, 1), (2, 3))):
        sources = []
        for rank, count in enumerate(counts):
            source = _sample(count)
            grids = []
            for target in range(count):
                height, width = ((1, 1), (30, 30), (2, 3))[target % 3]
                grids.append(tuple(tuple((step + rank + target) % 10 for _ in range(width))
                                   for _ in range(height)))
            sources.append(replace(source, target_grids=tuple(grids),
                                   input_ids=tuple(rank * 11 + i for i in source.input_ids)))
        encoded = _encode(tuple(sources))
        clean = tuple(target.clean_latent for target in encoded.inputs)
        flows = sample_endpoint_flow_batches(clean, endpoint_count=2, generator=generator)
        frames.append(dict(sources=tuple(sources), clean=clean, flows=flows))
    return tuple(frames)


def _gradients(model):
    return {name: p.grad.detach().clone() for name, p in model.named_parameters()}


def _oracle(initial, frames, objective):
    """Call the raw packed prediction and differentiate one global mean loss."""
    model = _model(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    scheduler = LinearWarmupConstantLR(optimizer, warmup_steps=2)
    snapshots = []
    for frame in frames:
        targets = _encode(frame["sources"]).inputs
        predictions = [None] * len(targets)
        for batch in _global_plan(frame):
            micro = build_flux_microbatch(tuple(targets[i] for i in batch.target_indices), max_joint_tokens=12288)
            flow = select_flow_batch(frame["flows"], batch.target_indices)
            prediction = predict_packed_flux_velocity(
                model, flow.noisy_latents, micro.text, micro.image_position_ids,
                flow.timesteps, micro.layout, compute_dtype=torch.float32, backend=SegmentedReference(),
            )
            for slot, index in enumerate(batch.target_indices):
                predictions[index] = prediction[slot:slot + 1]
        result = objective(torch.cat(predictions), torch.cat([f.target_velocity for f in frame["flows"]]),
                           tuple(target.grid_shape for target in targets))
        result.loss.backward()
        snapshot = dict(gradients=_gradients(model), loss=float(result.loss.detach()),
                        per_target=result.per_target.detach().clone(),
                        learning_rates=tuple(g["lr"] for g in optimizer.param_groups))
        snapshot["norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, foreach=False))
        optimizer.step()
        scheduler.step()
        snapshot.update(parameters=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()),
                        scheduler=deepcopy(scheduler.state_dict()))
        snapshots.append(snapshot)
        optimizer.zero_grad(set_to_none=True)
    return snapshots


def _trainer(model, sampler, objective, backend):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    scheduler = LinearWarmupConstantLR(optimizer, warmup_steps=2)
    trainer = Trainer(
        nvarc_encoder=FrozenText(), vae_encoder=FrozenVae(), flux_transformer=model,
        optimizer=optimizer, scheduler=scheduler, flow_sampler=sampler, loss_function=objective,
        flux_compute_dtype=torch.float32, flux_packing_backend=backend, max_grad_norm=0.1,
    )
    return trainer, optimizer, scheduler


def _check_updates(initial, frames, snapshots, objective, rank=None, *, bucket_views=False):
    model, backend = _model(initial), SegmentedReference()
    forward = PackedFluxForward(model, compute_dtype=torch.float32, backend=backend)
    if rank is not None:
        forward = DDP(forward, forward_sync_buffers=False, gradient_as_bucket_view=bucket_views)
    cursor = 0

    def sampler(clean):
        nonlocal cursor
        frame = frames[cursor]
        start = 0 if rank in (None, 0) else len(frame["sources"][0].target_grids)
        stop = len(frame["flows"]) if rank is None else start + len(frame["sources"][rank].target_grids)
        torch.testing.assert_close(tuple(clean), frame["clean"][start:stop], atol=0, rtol=0)
        cursor += 1
        return frame["flows"][start:stop]

    trainer, optimizer, scheduler = _trainer(forward, sampler, objective, backend)
    frozen = (trainer.nvarc_encoder, trainer.vae_encoder)
    frozen_states = tuple(deepcopy(m.state_dict()) for m in frozen)
    gradients, sync_states = [], []
    clip = torch.nn.utils.clip_grad_norm_

    def capture_clip(parameters, **kwargs):
        gradients.append(_gradients(model))
        return clip(parameters, **kwargs)

    handle = forward.register_forward_pre_hook(
        lambda module, _args: sync_states.append(module.require_backward_grad_sync if rank is not None else True)
    )
    try:
        for frame, expected in zip(frames, snapshots, strict=True):
            sources = frame["sources"] if rank is None else (frame["sources"][rank],)
            planner = (lambda *_, **__: _global_plan(frame)) if rank is None else _plan
            with patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render), \
                    patch("diffusion_arc.training.trainer.plan_flux_batches", planner), \
                    patch("torch.nn.utils.clip_grad_norm_", capture_clip):
                result = trainer.step(sources)
            torch.testing.assert_close(gradients[-1], expected["gradients"], **GRAD_TOLERANCE)
            assert trainer.completed_update
            torch.testing.assert_close(model.state_dict(), expected["parameters"], **PARAM_TOLERANCE)
            torch.testing.assert_close(optimizer.state_dict(), expected["optimizer"], **GRAD_TOLERANCE)
            torch.testing.assert_close(scheduler.state_dict(), expected["scheduler"], atol=0, rtol=0)
            torch.testing.assert_close(result.gradient_norm, expected["norm"], **LOSS_TOLERANCE)
            assert result.learning_rates == expected["learning_rates"]
            assert result.gradient_clipped == (expected["norm"] > 0.1)
            start = 0 if rank in (None, 0) else len(frame["sources"][0].target_grids)
            stop = start + result.target_count
            torch.testing.assert_close(torch.tensor([t.loss for t in result.targets]),
                                       expected["per_target"][start:stop], **LOSS_TOLERANCE)
            assert [t.timestep for t in result.targets] == [float(f.timesteps[0]) for f in frame["flows"][start:stop]]
            torch.testing.assert_close(result.loss, float(expected["per_target"][start:stop].mean()), **LOSS_TOLERANCE)
            assert len(sync_states) == result.flux_forward_calls
            if rank is not None:
                assert sync_states == [False] * (result.flux_forward_calls - 1) + [True]
                for tensor in (*model.parameters(), *gradients[-1].values()):
                    peer = tensor.detach().clone()
                    dist.broadcast(peer, src=0)
                    torch.testing.assert_close(tensor, peer, atol=0, rtol=0)
            sync_states.clear()
            if bucket_views:
                assert all(p.grad is not None and p.grad._is_view() and not bool(p.grad.count_nonzero())
                           for p in model.parameters())
            else:
                assert all(p.grad is None for p in model.parameters())
            json.dumps(asdict(result), allow_nan=False)
        for module, state in zip(frozen, frozen_states, strict=True):
            torch.testing.assert_close(module.state_dict(), state, atol=0, rtol=0)
            assert not module.training
            assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    finally:
        handle.remove()


def _check_failure(initial, frame, rank, failure):
    model, backend = _model(initial), SegmentedReference()
    forward = DDP(PackedFluxForward(model, compute_dtype=torch.float32, backend=backend), forward_sync_buffers=False)
    start = 0 if rank == 0 else len(frame["sources"][0].target_grids)
    flows = frame["flows"][start:start + len(frame["sources"][rank].target_grids)]

    def objective(*args):
        result = velocity_mse(*args)
        if failure == "loss" and rank == 0:
            # Finite gradients with a NaN loss must still prevent both updates.
            result = replace(result, loss=result.loss + float("nan"))
        return result

    trainer, optimizer, scheduler = _trainer(forward, lambda _: flows, objective, backend)
    scheduler_before = deepcopy(scheduler.state_dict())
    hook = (next(model.parameters()).register_hook(lambda g: torch.full_like(g, float("nan")))
            if failure == "gradient" and rank == 0 else None)
    sources = () if failure == "empty" and rank == 0 else (frame["sources"][rank],)
    error = ValueError if failure == "empty" else (FloatingPointError if failure == "loss" else RuntimeError)
    try:
        with patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render), \
                patch("diffusion_arc.training.trainer.plan_flux_batches", _plan):
            try:
                trainer.step(sources)
            except error as caught:
                expected_message = "each DDP rank" if failure == "empty" else "non-finite"
                assert expected_message in str(caught), str(caught)
            else:
                raise AssertionError(f"rank {rank} did not reject {failure}")
        torch.testing.assert_close(model.state_dict(), initial, atol=0, rtol=0)
        assert not optimizer.state
        assert scheduler.state_dict() == scheduler_before
        assert not trainer.completed_update
    finally:
        if hook is not None:
            hook.remove()


def _worker(rank, rendezvous, initial, frames, expected):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        for name, snapshots in expected.items():
            _check_updates(initial, frames, snapshots, _objective(name), rank)
        _check_updates(initial, frames, expected["mse"], _objective("mse"), rank, bucket_views=True)
        for failure in ("empty", "loss", "gradient"):
            _check_failure(initial, frames[0], rank, failure)
        for settings in ({"forward_sync_buffers": True},
                         {"forward_sync_buffers": False, "static_graph": True}):
            backend = SegmentedReference()
            forward = DDP(PackedFluxForward(_model(initial), compute_dtype=torch.float32, backend=backend), **settings)
            try:
                _trainer(forward, lambda _: (), velocity_mse, backend)
            except ValueError as error:
                assert "forward_sync_buffers=False and static_graph=False" in str(error)
            else:
                raise AssertionError("unsupported DDP configuration was accepted")
    finally:
        dist.destroy_process_group()


class DdpTrainerTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)

    def test_single_then_two_ranks_match_global_reference_and_abort_together(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(167)
            initial, frames = deepcopy(_model().state_dict()), _frames()
            expected = {name: _oracle(initial, frames, _objective(name)) for name in ("mse", "regions")}
            # Establish single-process correctness before any process group exists.
            for name, snapshots in expected.items():
                _check_updates(initial, frames, snapshots, _objective(name))
            with tempfile.TemporaryDirectory(prefix="trainer-ddp-") as directory:
                processes = mp.spawn(_worker, args=((Path(directory) / "store").as_uri(), initial, frames, expected),
                                     nprocs=2, join=False)
                try:
                    deadline = time.monotonic() + 90
                    while not processes.join(timeout=1):
                        if time.monotonic() > deadline:
                            raise TimeoutError("DDP trainer check exceeded 90 seconds")
                finally:
                    for process in processes.processes:
                        if process.is_alive():
                            process.terminate()
                    for process in processes.processes:
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=5)

    def test_rejects_inconsistent_packed_forward_settings(self):
        backend = SegmentedReference()
        for compute_dtype, packed_backend in ((torch.bfloat16, backend), (torch.float32, object())):
            forward = PackedFluxForward(_model(), compute_dtype=compute_dtype, backend=packed_backend)
            with self.assertRaisesRegex(ValueError, "precision/backend"):
                _trainer(forward, lambda _: (), velocity_mse, backend)


if __name__ == "__main__":
    unittest.main()
