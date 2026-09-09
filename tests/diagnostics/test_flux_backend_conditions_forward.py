"""Real tiny FLUX forwards check passive capture and attention-only routing."""

import unittest
from unittest.mock import patch

import torch

from scripts.diagnostics import probe_flux_backend_conditions as probe
from scripts.diagnostics import _flux_bfl_forward as observed
from scripts.diagnostics._flux_bfl_weights import build_bfl_model


class BackendConditionsForwardTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        torch.manual_seed(167)
        self.model = probe.df.Flux2Transformer2DModel(
            in_channels=8, out_channels=8, num_layers=5, num_single_layers=20,
            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=12,
            axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False,
        ).eval().requires_grad_(False)
        self.inputs = dict(images=torch.randn(1, 3, 8), text=torch.randn(1, 2, 12),
                           image_positions=torch.zeros(1, 3, 4, dtype=torch.int64),
                           text_positions=torch.zeros(1, 2, 4, dtype=torch.int64),
                           timesteps=torch.tensor([.5]), target=torch.randn(1, 3, 8),
                           text_mask=torch.ones(1, 2, dtype=torch.bool), lengths=(2,), labels=("tiny",))

    def test_native_capture_is_passive_and_all_25_backend_calls_are_routed(self):
        with observed.controlled_precision("bf16", torch.device("cpu")):
            expected = observed.forward_call(self.model, "diffusers", self.inputs)
        row, values, qkv = probe.model_case(self.model, "diffusers", self.inputs, "bf16", "native_math", capture_qkv=True)
        torch.testing.assert_close(values["tiny"]["velocity"]["image"], expected[0], atol=0, rtol=0)
        self.assertEqual(row["attention_calls"], 25)
        self.assertEqual(tuple(qkv), probe.BLOCKS)
        for tensors in qkv.values():
            self.assertTrue(all(t.dtype == torch.bfloat16 and t.is_contiguous() and not t.requires_grad for t in tensors))
        with patch.object(probe.replay, "attention_forward", wraps=probe.replay.attention_forward) as dispatch:
            routed, outputs, _ = probe.model_case(self.model, "diffusers", self.inputs, "bf16", "bf16_math")
        self.assertEqual(dispatch.call_count, 25)
        self.assertEqual(routed["input_fingerprints"], row["input_fingerprints"])
        torch.testing.assert_close(outputs["tiny"]["velocity"]["image"], expected[0], atol=0, rtol=0)

    def test_actual_bfl_and_diffusers_fp32_mapping_matches_selected_boundaries(self):
        bfl, _ = build_bfl_model(self.model)
        row, bfl_values, _ = probe.model_case(bfl, "bfl", self.inputs, "fp32")
        df_row, values, _ = probe.model_case(self.model, "diffusers", self.inputs, "fp32", "native_math")
        compared = probe.compare_model(values, bfl_values, "tiny")
        self.assertLess(compared["maximum_observed_relative_l2"], 2e-5)
        self.assertEqual(row["backend"], "native_math")
        self.assertEqual(row["input_fingerprints"], df_row["input_fingerprints"])
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_injected_backend_failure_restores_dispatch_and_capture_hooks(self):
        original = probe.df.dispatch_attention_fn
        before = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in self.model.modules()]
        with patch.object(probe.replay, "attention_forward", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                probe.model_case(self.model, "diffusers", self.inputs, "bf16", "bf16_math")
        self.assertIs(probe.df.dispatch_attention_fn, original)
        self.assertEqual(before, [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in self.model.modules()])
        self.assertTrue(torch.is_grad_enabled())


if __name__ == "__main__":
    unittest.main()
