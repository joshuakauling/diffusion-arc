"""Packed microbatch ownership, numerical gradients, and explicit loss invocation."""
from dataclasses import replace
from copy import deepcopy
import gc
import unittest
from unittest.mock import patch
import weakref

import torch
from diffusion_arc.model import TARGET_LATENT_CHANNELS, TARGET_LATENT_TOKEN_COUNT
from diffusion_arc.model.contracts import FluxSpatialShape
from diffusion_arc.model.flux.transformer import Flux2Transformer2DModel
from diffusion_arc.model.flux.packing import PackedFluxForward
from diffusion_arc.training.step.encoding import FluxInput, NvarcSequenceState
from diffusion_arc.training.step.flow_sampling import UniformFlowSampler, select_flow_batch
from diffusion_arc.training.step.flux_microbatch import FluxMicrobatch, build_flux_microbatch, compute_flux_microbatch_loss
from diffusion_arc.training.step.loss import RegionLossWeights, RegionVelocityLoss, velocity_mse
from tests.model.test_flux_packing import SegmentedReference

def _flux_inputs() -> tuple[FluxInput, ...]:
    prefix_ends = (2, 4, 6)
    sequence_state = NvarcSequenceState(
        hidden_states=torch.arange(24, dtype=torch.float32).reshape(6, 4),
        pending_prefix_ends=prefix_ends,
    )
    return tuple(
        FluxInput(
            nvarc_sequence_state=sequence_state,
            prefix_end=prefix_end,
            clean_latent=torch.full(
                (TARGET_LATENT_TOKEN_COUNT, TARGET_LATENT_CHANNELS),
                float(target_position + 1),
            ),
         grid_shape=(1, 1), context_pair_count=1)
        for target_position, prefix_end in enumerate(prefix_ends)
    )


class FluxMicrobatchTests(unittest.TestCase):
    def test_packed_copy_preserves_segments_and_releases_consumed_source(self) -> None:
        inputs = _flux_inputs()
        state = inputs[0].nvarc_sequence_state
        hidden_ref = weakref.ref(state._hidden_states)
        microbatch = build_flux_microbatch((inputs[2], inputs[0]), max_joint_tokens=12288)
        self.assertIsInstance(microbatch, FluxMicrobatch)
        self.assertEqual(microbatch.text.shape, (8, 4))
        self.assertEqual(microbatch.layout.cumulative.tolist(), [0, 906, 1808])
        self.assertEqual(microbatch.layout.text_positions[:, 3].tolist(), [0, 1, 2, 3, 4, 5, 0, 1])
        self.assertEqual(microbatch.layout.text_lengths, (6, 2))
        self.assertEqual(microbatch.image_position_ids.shape, (2, 900, 4))
        torch.testing.assert_close(microbatch.text[:2], microbatch.text[-2:], rtol=0, atol=0)
        self.assertEqual(state.pending_prefix_ends, frozenset({4}))
        build_flux_microbatch((inputs[1],), max_joint_tokens=12288)
        self.assertTrue(state.is_released)
        gc.collect()
        self.assertIsNone(hidden_ref())
        self.assertEqual(microbatch.target_count, 2)


    def test_packed_budget_and_failed_metadata_allocation_preserve_owners(self) -> None:
        inputs = _flux_inputs()
        state = inputs[0].nvarc_sequence_state
        for kwargs in ({"max_joint_tokens": 1000},):
            with self.assertRaises(ValueError):
                build_flux_microbatch(inputs, **kwargs)
            self.assertEqual(state.pending_prefix_ends, frozenset({2, 4, 6}))
        with patch("diffusion_arc.training.step.flux_microbatch.packing_layout", side_effect=torch.OutOfMemoryError("test")):
            with self.assertRaises(torch.OutOfMemoryError):
                build_flux_microbatch(inputs, max_joint_tokens=12288)
        self.assertEqual(state.pending_prefix_ends, frozenset({2, 4, 6}))
        self.assertFalse(state.is_released)


    def test_sequence_owner_rejects_autograd_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "autograd graph"):
            NvarcSequenceState(
                hidden_states=torch.ones((2, 4), requires_grad=True),
                pending_prefix_ends=(2,),
            )


    def test_failed_position_allocation_does_not_consume_any_input(self) -> None:
        inputs = _flux_inputs()
        sequence_state = inputs[0].nvarc_sequence_state
        pending_before = sequence_state.pending_prefix_ends

        with patch('diffusion_arc.training.step.flux_microbatch.build_flux_image_position_ids',
                   side_effect=torch.OutOfMemoryError('test')):
            with self.assertRaises(torch.OutOfMemoryError):
                build_flux_microbatch(inputs, max_joint_tokens=12288)

        self.assertEqual(sequence_state.pending_prefix_ends, pending_before)
        self.assertFalse(sequence_state.is_released)
        self.assertGreater(sequence_state.allocated_bytes, 0)


    def test_releases_independent_sequence_owners_separately(self) -> None:
        first_source_inputs = _flux_inputs()
        second_source_inputs = _flux_inputs()
        first_state = first_source_inputs[0].nvarc_sequence_state
        second_state = second_source_inputs[0].nvarc_sequence_state

        microbatch = build_flux_microbatch(
            (*first_source_inputs, second_source_inputs[0])
        , max_joint_tokens=12288)

        self.assertEqual(microbatch.target_count, 4)
        self.assertTrue(first_state.is_released)
        self.assertEqual(first_state.pending_prefix_ends, frozenset())
        self.assertFalse(second_state.is_released)
        self.assertEqual(
            second_state.pending_prefix_ends,
            frozenset({4, 6}),
        )


    def test_finished_microbatch_retains_neither_source_tensor_nor_owner(self) -> None:
        inputs = _flux_inputs()
        sequence_state = inputs[0].nvarc_sequence_state
        state_reference = weakref.ref(sequence_state)
        hidden_states = sequence_state._hidden_states
        self.assertIsNotNone(hidden_states)
        hidden_state_reference = weakref.ref(hidden_states)
        del hidden_states

        microbatch = build_flux_microbatch(inputs, max_joint_tokens=12288)
        self.assertTrue(sequence_state.is_released)
        gc.collect()
        self.assertIsNone(hidden_state_reference())

        del inputs
        del sequence_state
        gc.collect()

        self.assertIsNone(state_reference())
        self.assertEqual(microbatch.target_count, 3)


    def test_rejects_empty_duplicate_and_unknown_inputs_without_consuming(self) -> None:
        inputs = _flux_inputs()
        sequence_state = inputs[0].nvarc_sequence_state
        pending_before = sequence_state.pending_prefix_ends

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            build_flux_microbatch((), max_joint_tokens=12288)
        with self.assertRaisesRegex(ValueError, "prefix ends must be unique"):
            build_flux_microbatch((inputs[0], inputs[0]), max_joint_tokens=12288)

        unknown = replace(inputs[0], prefix_end=5)
        with self.assertRaisesRegex(RuntimeError, "not pending"):
            build_flux_microbatch((unknown,), max_joint_tokens=12288)

        self.assertEqual(sequence_state.pending_prefix_ends, pending_before)
        self.assertFalse(sequence_state.is_released)



class PackedForwardTests(unittest.TestCase):
    def test_real_flux_packing_preserves_loss_and_gradients_across_groupings(self):
        torch.manual_seed(53)
        initial = Flux2Transformer2DModel(
            in_channels=128, out_channels=128, num_layers=1, num_single_layers=1,
            attention_head_dim=8, num_attention_heads=1, joint_attention_dim=16,
            axes_dims_rope=(2,2,2,2), guidance_embeds=False,
        )
        hidden = torch.randn(6,16)
        latents = tuple(torch.randn(4,128) for _ in range(3))
        flows = UniformFlowSampler(torch.Generator().manual_seed(81))(latents)
        def run(groups):
            model = deepcopy(initial)
            forward = PackedFluxForward(
                model, compute_dtype=torch.bfloat16, backend=SegmentedReference(),
            )
            state = NvarcSequenceState(hidden, (2,4,6))
            inputs = tuple(FluxInput(state, end, latent, (2,2), 1) for end,latent in zip((2,4,6),latents))
            total = 0.0
            for indices in groups:
                result = compute_flux_microbatch_loss(
                    tuple(inputs[i] for i in indices), select_flow_batch(flows, indices),
                    transformer=forward,
                    max_joint_tokens=12288, loss_function=velocity_mse, spatial_shape=FluxSpatialShape(2,2),
                )
                self.assertIsNone(result.regions)
                self.assertEqual(result.loss.dtype, torch.float32)
                (result.loss * len(indices) / 3).backward()
                total += float(result.loss.detach()) * len(indices) / 3
            self.assertTrue(state.is_released)
            gradients = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            self.assertTrue(bool(torch.isfinite(gradients).all()))
            self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
            return total, gradients
        reference = run(((0,),(1,),(2,)))
        grouped = run(((2,1),(0,)))
        self.assertAlmostEqual(reference[0], grouped[0], delta=0.002)
        self.assertLess(float(torch.linalg.vector_norm(reference[1] - grouped[1]) /
                              torch.linalg.vector_norm(reference[1])), 0.02)

    def test_fixed_flow_and_grid_metadata_are_only_passed_to_their_components(self):
        for objective in (velocity_mse, RegionVelocityLoss(RegionLossWeights())):
            with self.subTest(objective=objective):
                inputs = tuple(replace(item, grid_shape=shape) for item, shape in
                               zip(_flux_inputs(), ((2,3), (7,1), (30,30)), strict=True))
                flows = UniformFlowSampler(torch.Generator().manual_seed(2))([i.clean_latent for i in inputs])
                order = (2,0,1)
                flow = select_flow_batch(flows, order)
                model = torch.nn.Linear(128,128)
                loss_calls = []
                def forward(transformer, noisy, text, positions, timesteps, layout, *, compute_dtype, backend):
                    self.assertIs(transformer, model)
                    self.assertIs(noisy, flow.noisy_latents)
                    self.assertIs(timesteps, flow.timesteps)
                    self.assertEqual(layout.text_lengths, (6,2,4))
                    return transformer(noisy)
                def loss(prediction, target_velocity, grid_shapes):
                    self.assertIs(target_velocity, flow.target_velocity)
                    self.assertEqual(grid_shapes, ((30,30), (2,3), (7,1)))
                    loss_calls.append(grid_shapes)
                    return objective(prediction, target_velocity, grid_shapes)
                with patch('diffusion_arc.model.flux.packing.predict_packed_flux_velocity', side_effect=forward) as predict:
                    result = compute_flux_microbatch_loss(tuple(inputs[i] for i in order), flow,
                                                          transformer=PackedFluxForward(model, compute_dtype=torch.bfloat16,
                                                                                        backend=object()), max_joint_tokens=12288,
                                                          loss_function=loss)
                self.assertEqual(predict.call_count, 1)
                self.assertEqual(len(loss_calls), 1)
                self.assertEqual(result.regions is None, objective is velocity_mse)
                result.loss.backward()
                self.assertIsNotNone(model.weight.grad)
                self.assertTrue(all(i.nvarc_sequence_state.is_released for i in inputs))


if __name__ == '__main__':
    unittest.main()
