"""Tests for target-blind batched NVARC execution."""

from __future__ import annotations

import types
import weakref
import unittest

import torch
from torch import Tensor, nn

from diffusion_arc.model import (
    NVARC_HIDDEN_SIZE,
    NVARC_HIDDEN_STATE_INDICES,
    NvarcBatchOutput,
    NvarcEncoder,
)
from unittest.mock import patch
from contextlib import ExitStack
from diffusion_arc.model.nvarc.attention import register_nvarc_attention, NVARC_ATTENTION_IMPLEMENTATION


class FakeEmbedding(nn.Embedding):
    def __init__(self, owner):
        super().__init__(16, 1)
        self.owner = weakref.ref(owner)

    def forward(self, input_ids):
        model = self.owner()
        model.forward_calls += 1
        model.seen_training_modes.append(model.training)
        model.seen_grad_modes.append(torch.is_grad_enabled())
        model.last_input_ids = input_ids.detach().cpu().clone()
        return input_ids.float().unsqueeze(-1).expand(-1, -1, NVARC_HIDDEN_SIZE)


class FakeBlock(nn.Module):
    def __init__(self, owner, index):
        super().__init__()
        self.owner = weakref.ref(owner)
        self.index = index

    def forward(self, hidden_states, **kwargs):
        if self.index == 1:
            self.owner().record_attention(**kwargs)
        return hidden_states + 1


class FakeRotary(nn.Module):
    def forward(self, hidden_states, position_ids):
        return None, None


class FakeNvarc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(_attn_implementation=NVARC_ATTENTION_IMPLEMENTATION)
        self.embed_tokens = FakeEmbedding(self)
        self.rotary_emb = FakeRotary()
        self.layers = nn.ModuleList(FakeBlock(self, index) for index in range(1, 37))
        self.forward_calls = 0
        self.seen_training_modes: list[bool] = []
        self.seen_grad_modes: list[bool] = []
        self.last_input_ids: Tensor | None = None
        self.last_attention_mask: Tensor | None = None
        self.last_use_cache: bool | None = None

    @property
    def base_model(self):
        return self

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def record_attention(self, **kwargs):
        self.last_attention_mask = kwargs["attention_mask"]
        self.last_use_cache = kwargs["use_cache"]
        self.metadata = kwargs

    def forward(self, **kwargs):
        raise AssertionError("the encoder must call only the required decoder blocks")


def reference_varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k,
                     max_seqlen_q, max_seqlen_k, softmax_scale, causal):
    """CPU reference for the kernel interface, with independent causal sources."""
    assert cu_seqlens_q.dtype == torch.int32
    torch.testing.assert_close(cu_seqlens_q, cu_seqlens_k)
    outputs = []
    offsets = cu_seqlens_q.tolist()
    assert max_seqlen_q == max_seqlen_k == max(b - a for a, b in zip(offsets, offsets[1:]))
    for start, end in zip(offsets, offsets[1:]):
        output = torch.nn.functional.scaled_dot_product_attention(
            q[start:end].transpose(0, 1), k[start:end].transpose(0, 1),
            v[start:end].transpose(0, 1), is_causal=causal,
            scale=softmax_scale, enable_gqa=True,
        )
        outputs.append(output.transpose(0, 1))
    return torch.cat(outputs)


class NvarcEncoderTests(unittest.TestCase):
    def test_packed_positions_boundaries_layers_and_compact_frozen_outputs(self):
        model = FakeNvarc()
        runner = NvarcEncoder(model)
        runner.train(True)
        ids = torch.tensor([1, 2, 3, 4, 5, 6])
        output = runner(ids, torch.tensor([0, 2, 6]))
        self.assertEqual(model.last_input_ids.tolist(), [[1, 2, 3, 4, 5, 6]])
        self.assertIsNone(model.last_attention_mask)
        self.assertEqual(model.metadata["position_ids"].tolist(), [[0, 1, 0, 1, 2, 3]])
        self.assertEqual(model.metadata["cu_seq_lens_q"].tolist(), [0, 2, 6])
        self.assertEqual(model.metadata["cu_seq_lens_q"].dtype, torch.int32)
        self.assertEqual(model.metadata["max_length_q"], 4)
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(model.seen_training_modes, [False])
        self.assertEqual(model.seen_grad_modes, [False])
        self.assertFalse(model.last_use_cache)
        self.assertTrue(all(not p.requires_grad for p in runner.parameters()))
        for states, tokens in zip(output.sequence_hidden_states, (ids[:2], ids[2:])):
            expected = torch.cat(tuple((tokens[:, None] + i).expand(-1, NVARC_HIDDEN_SIZE)
                                       for i in NVARC_HIDDEN_STATE_INDICES), -1).float()
            torch.testing.assert_close(states, expected)
            self.assertEqual(states.untyped_storage().nbytes(), states.numel() * states.element_size())
            self.assertFalse(states.requires_grad)

    def test_real_qwen_packed_equals_separate_and_isolates_sources(self):
        from transformers import Qwen3Config, Qwen3Model
        with patch("diffusion_arc.model.nvarc.attention.import_module",
                   return_value=types.SimpleNamespace(flash_attn_varlen_func=reference_varlen)):
            register_nvarc_attention()
        torch.manual_seed(11)
        config = Qwen3Config(vocab_size=16, hidden_size=8, intermediate_size=16,
                            num_hidden_layers=28, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=4, max_position_embeddings=32)
        config._attn_implementation = NVARC_ATTENTION_IMPLEMENTATION
        model = Qwen3Model(config)
        runner = NvarcEncoder(model)
        ids = torch.tensor([1, 2, 3, 4, 5, 6])
        actual = runner(ids, torch.tensor([0, 2, 6]))
        # Ordinary SDPA on independent sequences is the reference, including RoPE.
        model.config._attn_implementation = "sdpa"
        for index, tokens in enumerate((ids[:2], ids[2:])):
            with torch.no_grad():
                reference = model(tokens[None], output_hidden_states=True, use_cache=False)
            states = torch.cat(tuple(reference.hidden_states[i][0] for i in NVARC_HIDDEN_STATE_INDICES), -1)
            torch.testing.assert_close(actual.sequence_hidden_states[index], states, atol=3e-6, rtol=3e-6)
        model.config._attn_implementation = NVARC_ATTENTION_IMPLEMENTATION
        changed = runner(torch.tensor([7, 8, 3, 4, 5, 6]), torch.tensor([0, 2, 6]))
        torch.testing.assert_close(changed.sequence_hidden_states[1], actual.sequence_hidden_states[1], rtol=0, atol=0)
        tail = runner(torch.tensor([1, 2, 3, 4, 5, 9]), torch.tensor([0, 2, 6]))
        torch.testing.assert_close(tail.sequence_hidden_states[1][:-1], actual.sequence_hidden_states[1][:-1], rtol=0, atol=0)

    def test_early_exit_matches_full_qwen_and_skips_unused_modules(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        with patch("diffusion_arc.model.nvarc.attention.import_module",
                   return_value=types.SimpleNamespace(flash_attn_varlen_func=reference_varlen)):
            register_nvarc_attention()
        torch.manual_seed(23)
        config = Qwen3Config(vocab_size=16, hidden_size=8, intermediate_size=16,
                            num_hidden_layers=36, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=4, max_position_embeddings=32)
        config._attn_implementation = NVARC_ATTENTION_IMPLEMENTATION
        model = Qwen3ForCausalLM(config)
        runner = NvarcEncoder(model)
        # Make accidental final normalization easy to detect.
        with torch.no_grad():
            model.model.norm.weight.fill_(2)
        for lengths in ((2, 4), (1, 3, 2), (5,)):
            with self.subTest(lengths=lengths):
                offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
                ids = torch.arange(sum(lengths)) % 16
                positions = torch.cat(tuple(torch.arange(n) for n in lengths))[None]
                cumulative = offsets.to(torch.int32)
                with torch.no_grad():
                    full = model(
                        ids[None], position_ids=positions, attention_mask=None,
                        cu_seq_lens_q=cumulative, cu_seq_lens_k=cumulative,
                        max_length_q=max(lengths), max_length_k=max(lengths),
                        output_hidden_states=True, use_cache=False, return_dict=True,
                    )
                with ExitStack() as stack:
                    calls = [stack.enter_context(patch.object(block, "forward", wraps=block.forward))
                             for block in model.model.layers]
                    norm = stack.enter_context(patch.object(model.model.norm, "forward",
                        side_effect=AssertionError("final normalization must not run")))
                    head = stack.enter_context(patch.object(model.lm_head, "forward",
                        side_effect=AssertionError("LM head must not run")))
                    actual = runner(ids, offsets)
                    self.assertEqual([call.call_count for call in calls], [1] * 27 + [0] * 9)
                    norm.assert_not_called()
                    head.assert_not_called()
                for states, start, end in zip(actual.sequence_hidden_states, offsets[:-1], offsets[1:]):
                    expected = torch.cat(tuple(full.hidden_states[i][0, start:end]
                                               for i in NVARC_HIDDEN_STATE_INDICES), -1)
                    torch.testing.assert_close(states, expected, rtol=0, atol=0)
                    self.assertFalse(states.requires_grad)
                    self.assertEqual(states.untyped_storage().nbytes(), states.numel() * states.element_size())
                # This demonstrates why using HF's normalized final state would be wrong.
                raw_last = full.hidden_states[27]
                self.assertFalse(torch.equal(raw_last, model.model.norm(raw_last)))
        self.assertEqual(model.config.num_hidden_layers, 36)
        self.assertEqual(len(model.model.layers), 36)

    def test_refuses_unregistered_or_dense_backend(self):
        model = FakeNvarc()
        model.config._attn_implementation = "sdpa"
        with self.assertRaisesRegex(ValueError, "nvarc_fa3"):
            NvarcEncoder(model)

    def test_missing_fa3_fails_without_fallback(self):
        with patch("diffusion_arc.model.nvarc.attention.import_module", side_effect=ImportError("FA3 missing")):
            with self.assertRaisesRegex(ImportError, "FA3 missing"):
                register_nvarc_attention()
