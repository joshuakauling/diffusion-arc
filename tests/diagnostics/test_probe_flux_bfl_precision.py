"""Validate passive observations and isolated native precision interventions."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from scripts.diagnostics import probe_flux_bfl_precision as probe


class BFLPrecisionTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        self.model, self.reference, self.inputs, _ = probe.fixture(37)

    def unobserved(self, implementation, precision):
        dtype = torch.float32 if precision == "fp32" else torch.bfloat16
        image, text, image_mod, text_mod = (
            self.inputs[key].to(dtype) for key in ("image", "text", "image_mod", "text_mod"))
        with probe.controlled_precision(precision, torch.device("cpu")):
            if implementation == "bfl":
                pe_image, pe_text, _, _ = probe.rope_coefficients(self.reference, self.inputs)
                image_out, text_out, _ = self.reference.double_blocks[0].forward_kv_extract(
                    image, text, pe_image, pe_text, probe.df.Flux2Modulation.split(image_mod, 2),
                    probe.df.Flux2Modulation.split(text_mod, 2), num_ref_tokens=0)
            else:
                ids = torch.cat((self.inputs["text_ids"], self.inputs["image_ids"]), dim=1)[0]
                text_out, image_out = self.model.transformer_blocks[0](
                    image, text, image_mod, text_mod, image_rotary_emb=self.model.pos_embed(ids))
        return image_out, text_out

    def test_observation_preserves_native_block_outputs_bitwise(self):
        for precision in ("fp32", "bf16"):
            for implementation in ("bfl", "diffusers"):
                with self.subTest(precision=precision, implementation=implementation):
                    plain = self.unobserved(implementation, precision)
                    observed = probe.block_run(self.model, self.reference, self.inputs,
                                               precision, "native", implementation)
                    self.assertEqual(tuple(observed), probe.STAGES)
                    for stream, expected in zip(("image", "text"), plain):
                        actual = observed[f"block.{stream}"]
                        self.assertEqual(actual.dtype, expected.dtype)
                        self.assertTrue(torch.equal(actual, expected))

    def test_norm_and_rope_interventions_preserve_upstream_values_and_match_bfl(self):
        reference = probe.block_run(self.model, self.reference, self.inputs, "bf16", "native", "bfl")
        native = probe.block_run(self.model, self.reference, self.inputs, "bf16", "native")
        corrected = probe.block_run(self.model, self.reference, self.inputs, "bf16", "bfl_norm+bfl_rope")
        preceding = probe.STAGES[:probe.STAGES.index("normalized.image.q")]
        self.assertTrue(all(torch.equal(native[name], corrected[name]) for name in preceding))
        self.assertFalse(torch.equal(native["normalized.image.q"], reference["normalized.image.q"]))
        self.assertTrue(all(torch.equal(corrected[name], reference[name]) for name in probe.STAGES))
        self.assertTrue(all(p.dtype == torch.float32 for p in self.model.parameters()))

    def test_exception_restores_hooks_global_functions_and_norm_methods(self):
        functions = (probe.bfl.apply_rope, probe.bfl.causal_attn_fn, probe.df.apply_rotary_emb,
                     probe.df.dispatch_attention_fn, probe.df._get_qkv_projections)
        modules = [*self.model.modules(), *self.reference.modules()]
        state = [(len(m._forward_hooks), len(m._forward_pre_hooks), m.__dict__.get("forward")) for m in modules]
        for implementation, variant, failing in (
            ("bfl", "native", self.reference.double_blocks[0].img_attn.proj),
            ("diffusers", "bfl_norm+fused_qkv+bfl_rope", self.model.transformer_blocks[0].attn.to_out[0]),
        ):
            with self.subTest(implementation=implementation), ExitStack() as stack:
                stack.enter_context(patch.object(failing, "forward", side_effect=RuntimeError("injected")))
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    probe.block_run(self.model, self.reference, self.inputs, "bf16", variant, implementation)
            self.assertEqual(functions, (probe.bfl.apply_rope, probe.bfl.causal_attn_fn,
                             probe.df.apply_rotary_emb, probe.df.dispatch_attention_fn,
                             probe.df._get_qkv_projections))
            self.assertEqual(state, [(len(m._forward_hooks), len(m._forward_pre_hooks),
                                     m.__dict__.get("forward")) for m in modules])

    def test_cuda_norm_replay_records_dtypes_and_distinguishes_final_cast_from_bfl(self):
        result = probe.cuda_norm_replay(self.reference, self.inputs, 37)
        self.assertIn("CPU replay", result["scope"])
        native = result["comparisons"]["cuda_native_norm"]
        final_cast = result["comparisons"]["cuda_norm_final_cast"]
        self.assertEqual(native["normalized_q"]["actual_dtype"], "torch.float32")
        self.assertEqual(final_cast["normalized_q"]["actual_dtype"], "torch.bfloat16")
        for rows in (native, final_cast):
            self.assertEqual(rows["kernel_q"]["actual_dtype"], "torch.bfloat16")
            self.assertGreater(rows["kernel_q"]["relative_l2"], 0)
            self.assertGreater(rows["attention"]["relative_l2"], 0)


if __name__ == "__main__":
    unittest.main()
