from __future__ import annotations

import types
import unittest

import torch
from torch import nn

from diffusion_arc.nvarc_conditioning import NVARCConditioner


class FakeNVARC(nn.Module):
    def __init__(self, hidden_size: int = 4, hidden_count: int = 6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.hidden_size = hidden_size
        self.hidden_count = hidden_count
        self.dropout = nn.Dropout(p=0.5)
        self.seen_training_modes: list[bool] = []

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        output_hidden_states,
        use_cache,
        return_dict,
    ):
        self.seen_training_modes.append(self.training)
        assert output_hidden_states
        assert not use_cache
        assert return_dict
        assert input_ids.shape == attention_mask.shape
        batch, length = input_ids.shape
        states = tuple(
            torch.full(
                (batch, length, self.hidden_size),
                float(index),
                device=input_ids.device,
            )
            * self.weight
            for index in range(self.hidden_count)
        )
        return types.SimpleNamespace(hidden_states=states)


class NVARCConditionerTest(unittest.TestCase):
    def test_concatenates_selected_layers_in_order_and_freezes_model(self):
        nvarc = FakeNVARC()
        conditioner = NVARCConditioner(
            nvarc,
            layer_indices=(1, 3, 5),
            hidden_size=4,
            context_dim=12,
        )
        conditioner.train()

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        attention_mask = torch.ones_like(input_ids)
        context = conditioner(input_ids, attention_mask)

        self.assertEqual(context.shape, (1, 3, 12))
        torch.testing.assert_close(context[..., :4], torch.ones(1, 3, 4))
        torch.testing.assert_close(context[..., 4:8], torch.full((1, 3, 4), 3.0))
        torch.testing.assert_close(context[..., 8:], torch.full((1, 3, 4), 5.0))
        self.assertFalse(context.requires_grad)
        self.assertFalse(nvarc.training)
        self.assertEqual(nvarc.seen_training_modes, [False])
        self.assertFalse(any(parameter.requires_grad for parameter in nvarc.parameters()))

    def test_rejects_padding_by_default(self):
        conditioner = NVARCConditioner(
            FakeNVARC(),
            layer_indices=(1, 3, 5),
            hidden_size=4,
            context_dim=12,
        )
        input_ids = torch.tensor([[1, 2, 0]])
        attention_mask = torch.tensor([[1, 1, 0]])

        with self.assertRaisesRegex(ValueError, "Padded or masked NVARC contexts"):
            conditioner(input_ids, attention_mask)

    def test_rejects_batch_greater_than_one(self):
        conditioner = NVARCConditioner(
            FakeNVARC(),
            layer_indices=(1, 3, 5),
            hidden_size=4,
            context_dim=12,
        )
        input_ids = torch.tensor([[1, 2], [3, 4]])

        with self.assertRaisesRegex(ValueError, "batch size 1"):
            conditioner(input_ids, torch.ones_like(input_ids))

    def test_rejects_wrong_hidden_width(self):
        conditioner = NVARCConditioner(
            FakeNVARC(hidden_size=5),
            layer_indices=(1, 3, 5),
            hidden_size=4,
            context_dim=12,
        )
        input_ids = torch.tensor([[1, 2]])

        with self.assertRaisesRegex(RuntimeError, "hidden state 1"):
            conditioner(input_ids, torch.ones_like(input_ids))

    def test_rejects_overlong_context_without_truncating(self):
        conditioner = NVARCConditioner(
            FakeNVARC(),
            layer_indices=(1, 3, 5),
            hidden_size=4,
            context_dim=12,
            max_length=2,
        )
        input_ids = torch.tensor([[1, 2, 3]])

        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            conditioner(input_ids, torch.ones_like(input_ids))


if __name__ == "__main__":
    unittest.main()
