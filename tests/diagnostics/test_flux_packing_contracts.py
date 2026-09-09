"""Exact structural packing oracles using independently labelled synthetic tokens.

The padding control exercises the mask and gather/scatter primitives only;
it does not validate a production attention-only packing adapter.
"""

from dataclasses import replace
from itertools import permutations
import unittest

import torch

from diffusion_arc.model.flux_conditioning import build_flux_joint_attention_mask
from scripts.diagnostics._flux_packing import packing_layout
from scripts.diagnostics._flux_packing_preflight import validate_layout


def labelled_tokens(lengths_by_owner, image_tokens, sample_order):
    """Describe source and expected destinations by identity, without layout indices."""
    text = [(owner, "text", position) for owner in sample_order
            for position in range(lengths_by_owner[owner])]
    image = [(owner, "image", position) for owner in sample_order
             for position in range(image_tokens)]
    segments = [[(owner, "text", position) for position in range(lengths_by_owner[owner])]
                + [(owner, "image", position) for position in range(image_tokens)]
                for owner in sample_order]
    return text, image, segments


def token_value(token):
    owner, modality, position = token
    return owner * 100_000 + (10_000 if modality == "image" else 0) + position


def token_position(token):
    owner, modality, position = token
    return (0, 0, 0, position) if modality == "text" else (owner, position // 30, position % 30, 0)


def upstream_value(token):
    owner, modality, position = token
    return owner * 4 + (2 if modality == "image" else -2) + position / 2


class FluxPackingContractTests(unittest.TestCase):
    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def assert_labelled_layout(self, lengths_by_owner, image_tokens, sample_order):
        text, image, segments = labelled_tokens(lengths_by_owner, image_tokens, sample_order)
        source = text + image
        expected_joint = [token for segment in segments for token in segment]
        layout = packing_layout(tuple(lengths_by_owner[owner] for owner in sample_order), image_tokens, "cpu")
        validate_layout(layout)
        source_values = torch.tensor([token_value(token) for token in source], dtype=torch.int64)
        packed = source_values.index_select(0, layout.joint_order)
        self.assert_exact(packed, torch.tensor([token_value(token) for token in expected_joint]))
        self.assertEqual(len(set(packed.tolist())), len(source))
        self.assertEqual(set(packed.tolist()), set(source_values.tolist()))
        self.assert_exact(packed.index_select(0, layout.text_indices), source_values[:len(text)])
        self.assert_exact(packed.index_select(0, layout.image_indices), source_values[len(text):])

        # Expected positions and owners come from token identities, never inverses
        # or boundaries computed by the packing implementation.
        positions = torch.tensor([token_position(token) for token in source])
        self.assert_exact(layout.text_positions, positions[:len(text)])
        self.assert_exact(positions.index_select(0, layout.joint_order),
                          torch.tensor([token_position(token) for token in expected_joint]))
        expected_boundaries = [0]
        for segment in segments:
            expected_boundaries.append(expected_boundaries[-1] + len(segment))
        self.assertEqual(layout.cumulative.dtype, torch.int32)
        self.assertEqual(layout.cumulative.tolist(), expected_boundaries)
        self.assertEqual(layout.maximum, max(map(len, segments)))
        row_by_owner = {owner: row for row, owner in enumerate(sample_order)}
        for actual, tokens in ((layout.text_samples, text), (layout.image_samples, image),
                               (layout.joint_samples, expected_joint)):
            self.assertEqual(actual.dtype, torch.int64)
            self.assertEqual(actual.tolist(), [row_by_owner[token[0]] for token in tokens])
        for row, segment in enumerate(segments):
            start, end = layout.cumulative[row:row + 2].tolist()
            self.assertEqual(packed[start:end].tolist(), [token_value(token) for token in segment])
        return layout, text, image, expected_joint

    def test_hand_written_three_segment_order_and_inverse(self):
        layout = packing_layout((2, 1, 3), 2, "cpu")
        # Source is [A0,A1,B0,C0,C1,C2, a0,a1,b0,b1,c0,c1].
        self.assertEqual(layout.joint_order.tolist(), [0, 1, 6, 7, 2, 8, 9, 3, 4, 5, 10, 11])
        self.assertEqual(layout.text_indices.tolist(), [0, 1, 4, 7, 8, 9])
        self.assertEqual(layout.image_indices.tolist(), [2, 3, 5, 6, 10, 11])
        self.assertEqual(layout.cumulative.tolist(), [0, 4, 7, 12])
        self.assertEqual(layout.text_positions.tolist(),
                         [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0],
                          [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]])

    def test_asymmetric_segments_and_every_sample_permutation(self):
        lengths = {11: 3, 22: 1, 33: 4}
        for order in permutations(lengths):
            with self.subTest(order=order):
                self.assert_labelled_layout(lengths, 2, order)

    def test_text_and_joint_boundaries_around_128_and_256(self):
        for lengths, image_tokens in (((127, 128, 129, 255, 256, 257), 900),
                                       ((125, 126, 127, 253, 254, 255), 2)):
            by_owner = dict(enumerate(lengths, start=1))
            for order in (tuple(by_owner), tuple(reversed(by_owner))):
                with self.subTest(lengths=lengths, image_tokens=image_tokens, order=order):
                    self.assert_labelled_layout(by_owner, image_tokens, order)

    def test_8400_text_tokens_remain_separate_from_900_image_tokens(self):
        for order in ((1, 2, 3), (3, 1, 2)):
            with self.subTest(order=order):
                layout, _, _, _ = self.assert_labelled_layout({1: 8400, 2: 1, 3: 257}, 900, order)
                self.assertEqual(layout.maximum, 9300)
                self.assertEqual(layout.cumulative[-1].item(), 11_358)

    def test_unique_permutation_gather_and_inverse_have_analytic_gradients(self):
        for order in ((1, 2, 3), (3, 1, 2), (3, 2, 1)):
            with self.subTest(order=order):
                layout, text, image, joint = self.assert_labelled_layout({1: 2, 2: 1, 3: 3}, 2, order)
                source_tokens = text + image
                source = torch.tensor([token_value(token) for token in source_tokens],
                                      dtype=torch.float64, requires_grad=True)
                packed = source.index_select(0, layout.joint_order)
                packed_upstream = torch.tensor([upstream_value(token) for token in joint], dtype=torch.float64)
                (packed * packed_upstream).sum().backward()
                self.assert_exact(source.grad, torch.tensor([upstream_value(token) for token in source_tokens],
                                                           dtype=torch.float64))

                independent_packed = torch.tensor([token_value(token) for token in joint],
                                                 dtype=torch.float64, requires_grad=True)
                restored = torch.cat((independent_packed.index_select(0, layout.text_indices),
                                      independent_packed.index_select(0, layout.image_indices)))
                source_upstream = torch.tensor([upstream_value(token) for token in source_tokens], dtype=torch.float64)
                (restored * source_upstream).sum().backward()
                self.assert_exact(independent_packed.grad, packed_upstream)

    def test_repeated_timestep_ownership_sums_gradients_instead_of_permuting(self):
        layout = packing_layout((2, 1, 3), 2, "cpu")
        controls = (
            (layout.text_samples, [11, 11, 23, 37, 37, 37], [1, 2, 3, 4, 5, 6], [3, 3, 15]),
            (layout.image_samples, [11, 11, 23, 23, 37, 37], [7, 8, 9, 10, 11, 12], [15, 19, 23]),
            (layout.joint_samples, [11, 11, 11, 11, 23, 23, 23, 37, 37, 37, 37, 37],
             [-1, 2, 3, -4, 5, 6, -7, 8, 9, -10, 11, 12], [0, 4, 30]),
        )
        accumulated = torch.zeros(3, dtype=torch.float64)
        for indices, expected_values, upstream, expected_gradient in controls:
            with self.subTest(ownership=expected_values):
                timesteps = torch.tensor([11., 23., 37.], dtype=torch.float64, requires_grad=True)
                expanded = timesteps.index_select(0, indices)
                self.assert_exact(expanded, torch.tensor(expected_values, dtype=torch.float64))
                (expanded * torch.tensor(upstream, dtype=torch.float64)).sum().backward()
                self.assert_exact(timesteps.grad, torch.tensor(expected_gradient, dtype=torch.float64))
                accumulated += timesteps.grad
        self.assert_exact(accumulated, torch.tensor([18., 26., 68.], dtype=torch.float64))

    def test_padding_mask_and_gather_scatter_primitives_exclude_middle_padding(self):
        # This pointwise toy operation isolates the primitive contract. It is
        # deliberately not an attention backend or an attention-only adapter.
        text_mask = torch.tensor([[True, True, False, False, False],
                                  [True, True, True, True, False],
                                  [True, False, False, False, False]])
        joint_mask = build_flux_joint_attention_mask(text_mask, 2)[:, 0, 0, :]
        indices = joint_mask.flatten().nonzero().flatten()
        self.assertEqual(indices.tolist(), [0, 1, 5, 6, 7, 8, 9, 10, 12, 13, 14, 19, 20])
        expected_tokens = [101, 102, 151, 152, 201, 202, 203, 204, 251, 252, 301, 351, 352]
        values = [[101, 102, -999, -998, -997, 151, 152],
                  [201, 202, 203, 204, -996, 251, 252],
                  [301, -995, -994, -993, -992, 351, 352]]
        expected_gradient = torch.tensor([[2, 4, 0, 0, 0, 12, 14],
                                          [16, 18, 20, 22, 0, 26, 28],
                                          [30, 0, 0, 0, 0, 40, 42]], dtype=torch.float64)
        previous_output = None
        for padding_value in (None, 1e9):
            padded = torch.tensor(values, dtype=torch.float64)
            if padding_value is not None:
                padded[~joint_mask] = padding_value
            padded.requires_grad_()
            gathered = padded.flatten().index_select(0, indices)
            self.assert_exact(gathered, torch.tensor(expected_tokens, dtype=torch.float64))
            restored = torch.zeros_like(padded).flatten().index_copy(0, indices, 2 * gathered).reshape_as(padded)
            self.assert_exact(restored[~joint_mask], torch.zeros_like(restored[~joint_mask]))
            self.assert_exact(restored[joint_mask], 2 * torch.tensor(expected_tokens, dtype=torch.float64))
            weights = torch.arange(1, 22, dtype=torch.float64).reshape_as(padded)
            (restored * weights).sum().backward()
            self.assert_exact(padded.grad, expected_gradient)
            if previous_output is not None:
                self.assert_exact(restored, previous_output)
            previous_output = restored.detach()

    def test_constructor_rejects_nonpositive_or_noninteger_lengths(self):
        for lengths, image_tokens in (((), 2), ((0, 1), 2), ((-1, 2), 2),
                                      ((True, 2), 2), ((1.5, 2), 2), (("2",), 2),
                                      ((None,), 2), ((1,), 0), ((1,), -2),
                                      ((1,), True), ((1,), 2.0), ((1,), "2")):
            with self.subTest(lengths=lengths, image_tokens=image_tokens):
                with self.assertRaisesRegex(ValueError, "positive text and image lengths"):
                    packing_layout(lengths, image_tokens, "cpu")

    def test_preflight_rejects_invalid_length_limits_and_metadata_types(self):
        valid = packing_layout((2, 1, 3), 2, "cpu")
        for lengths in ((), (0, 1), (-1, 2), (True, 2), (1.5, 2), ("2",), (None,), [2, 1, 3]):
            with self.subTest(text_lengths=lengths):
                with self.assertRaisesRegex(ValueError, "text_lengths"):
                    validate_layout(replace(valid, text_lengths=lengths))
        for image_tokens in (0, -1, True, 2.0, "2", None, 901):
            with self.subTest(image_tokens=image_tokens):
                with self.assertRaisesRegex(ValueError, "image_tokens"):
                    validate_layout(replace(valid, image_tokens=image_tokens))
        with self.assertRaisesRegex(ValueError, "8400"):
            validate_layout(packing_layout((8401,), 900, "cpu"))
        with self.assertRaisesRegex(ValueError, "PackingLayout"):
            validate_layout(object())
        # Oversized external metadata must fail without creating its index arrays.
        excessive_segments = (8400,) * 256_000
        with self.assertRaisesRegex(ValueError, "int32 capacity"):
            validate_layout(replace(valid, text_lengths=excessive_segments, image_tokens=900))

    def test_preflight_rejects_mutated_boundaries_ownership_inverses_and_positions(self):
        valid = packing_layout((2, 1, 3), 2, "cpu")
        mutations = (
            ("cumulative", [1, 4, 7, 12]),
            ("cumulative", [0, 5, 7, 12]),
            ("cumulative", [0, 4, 3, 12]),
            ("cumulative", [0, 4, 7, 11]),
            ("joint_order", [1, 0, 6, 7, 2, 8, 9, 3, 4, 5, 10, 11]),
            ("joint_order", [0, 0, 6, 7, 2, 8, 9, 3, 4, 5, 10, 11]),
            ("text_indices", [0, 1, 4, 7, 8, 12]),
            ("image_indices", [2, 3, 6, 5, 10, 11]),
            ("text_samples", [0, 0, 1, 2, 2, 1]),
            ("image_samples", [0, 0, 2, 1, 2, 2]),
            ("joint_samples", [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            ("text_positions", [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2],
                                [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]),
            ("text_positions", [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]),
        )
        for field, values in mutations:
            with self.subTest(field=field, values=values):
                changed = torch.tensor(values, dtype=getattr(valid, field).dtype)
                with self.assertRaisesRegex(ValueError, field):
                    validate_layout(replace(valid, **{field: changed}))
        # Frozen dataclasses do not freeze the tensor storage they contain.
        valid.cumulative[1] = 5
        with self.assertRaisesRegex(ValueError, "cumulative"):
            validate_layout(valid)

    def test_preflight_checks_each_tensor_dtype_shape_and_materialization(self):
        valid = packing_layout((2, 1, 3), 2, "cpu")
        for field in ("cumulative", "joint_order", "text_indices", "image_indices",
                      "text_samples", "image_samples", "joint_samples", "text_positions"):
            tensor = getattr(valid, field)
            invalid_dtype = torch.int64 if tensor.dtype == torch.int32 else torch.int32
            for malformed in (tensor.to(invalid_dtype), tensor.unsqueeze(0), tensor.tolist(),
                              torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")):
                with self.subTest(field=field, invalid_type=type(malformed)):
                    with self.assertRaisesRegex(ValueError, field):
                        validate_layout(replace(valid, **{field: malformed}))


if __name__ == "__main__":
    unittest.main()
