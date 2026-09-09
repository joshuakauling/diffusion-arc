"""Verify the pinned BFL reference, lossless mapping, and real CPU forwards."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers

from diffusion_arc.model.flux_transformer import Flux2Transformer2DModel
from scripts.diagnostics import _flux_bfl_weights as weights


def tiny_model(**overrides):
    return Flux2Transformer2DModel(**dict(
        dict(in_channels=8, out_channels=8, num_layers=2, num_single_layers=2,
             attention_head_dim=16, num_attention_heads=2, joint_attention_dim=12,
             axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False), **overrides))


class BFLWeightTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        torch.manual_seed(472)

    def test_pinned_source_rejects_modified_code_and_self_consistent_rewritten_manifest(self):
        self.assertTrue(weights.validate_source()["validated"])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name in (*weights.SOURCE_FILES, "source.json"):
                shutil.copyfile(weights.SOURCE_DIRECTORY / name, directory / name)
            (directory / "model.py").write_text("# changed source\n")
            with self.assertRaisesRegex(ValueError, "source hash mismatch: model.py"):
                weights.validate_source(directory)
            manifest = json.loads((directory / "source.json").read_text())
            manifest["files"]["model.py"]["sha256"] = weights.hashlib.sha256((directory / "model.py").read_bytes()).hexdigest()
            (directory / "source.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "source metadata mismatch: model.py"):
                weights.validate_source(directory)

    def test_complete_lossless_mapping_matches_official_inverse_and_preserves_source(self):
        for guidance in (False, True):
            with self.subTest(guidance=guidance):
                source = tiny_model(guidance_embeds=guidance)
                source.train()
                before = {name: value.clone() for name, value in source.state_dict().items()}
                rng = torch.random.get_rng_state().clone()
                bfl, report = weights.build_bfl_model(source)
                torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)
                self.assertTrue(source.training)
                self.assertTrue(all(parameter.requires_grad for parameter in source.parameters()))
                self.assertFalse(bfl.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in bfl.parameters()))
                self.assertTrue(report["complete"] and report["lossless"])
                self.assertEqual(report["source_numel"], report["target_numel"])
                self.assertEqual(report["source_parameter_count"], len(before))
                self.assertEqual(report["target_parameter_count"], len(bfl.state_dict()))
                self.assertGreater(report["shared_storage_bytes"], 0)
                self.assertGreater(report["copied_storage_bytes"], 0)
                self.assertIsNot(bfl.img_in.weight, source.x_embedder.weight)
                self.assertEqual(bfl.img_in.weight.data_ptr(), source.x_embedder.weight.data_ptr())
                # This independent, installed upstream converter reverses QKV and final scale/shift.
                recovered = convert_flux2_transformer_checkpoint_to_diffusers(dict(bfl.state_dict()))
                self.assertEqual(recovered.keys(), before.keys())
                for name, original in before.items():
                    with self.subTest(parameter=name):
                        torch.testing.assert_close(recovered[name], original, atol=0, rtol=0)
                        torch.testing.assert_close(source.state_dict()[name], original, atol=0, rtol=0)
                json.dumps(report, allow_nan=False)

    def test_mapping_preserves_bfloat16_weights_without_implicit_fp32_conversion(self):
        source = tiny_model().to(torch.bfloat16)
        bfl, report = weights.build_bfl_model(source)
        recovered = convert_flux2_transformer_checkpoint_to_diffusers(dict(bfl.state_dict()))
        self.assertEqual(report["dtype"], "torch.bfloat16")
        for name, original in source.state_dict().items():
            self.assertEqual(recovered[name].dtype, original.dtype)
            torch.testing.assert_close(recovered[name], original, atol=0, rtol=0)

    def test_actual_bfl_and_diffusers_fp32_forward_parity(self):
        for guidance_enabled in (False, True):
            source = tiny_model(guidance_embeds=guidance_enabled).eval()
            bfl, _ = weights.build_bfl_model(source)
            for batch_size, text_length in ((1, 2), (2, 5)):
                with self.subTest(guidance=guidance_enabled, batch=batch_size, text=text_length):
                    image, text = torch.randn(batch_size, 3, 8), torch.randn(batch_size, text_length, 12)
                    # Diffusers uses IDs from row zero for the entire batch.
                    image_ids = torch.randint(-3, 15, (1, 3, 4)).repeat(batch_size, 1, 1)
                    text_ids = torch.randint(-3, 15, (1, text_length, 4)).repeat(batch_size, 1, 1)
                    timesteps = torch.linspace(.13, .71, batch_size)
                    guidance = torch.linspace(1.7, 3.4, batch_size) if guidance_enabled else None
                    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), \
                         patch("torch.autograd.backward", side_effect=AssertionError("unexpected backward")), \
                         patch("torch.autograd.grad", side_effect=AssertionError("unexpected gradient")):
                        actual = bfl(image, image_ids, timesteps, text, text_ids, guidance)
                        expected = source(hidden_states=image, encoder_hidden_states=text, timestep=timesteps,
                                          img_ids=image_ids, txt_ids=text_ids, guidance=guidance).sample
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                    self.assertTrue(all(parameter.grad is None for parameter in source.parameters()))

    def test_missing_extra_wrong_shape_mixed_dtype_and_meta_weights_are_rejected(self):
        source = tiny_model()
        state = source.state_dict()
        key = "transformer_blocks.0.attn.to_k.weight"
        bad_states = []
        missing = dict(state)
        missing.pop(key)
        bad_states.append((missing, "coverage mismatch"))
        bad_states.append(({**state, "unmapped.weight": torch.zeros(1)}, "coverage mismatch"))
        bad_states.append(({**state, key: state[key][:-1]}, "Weight shape mismatch"))
        bad_states.append(({**state, key: state[key].double()}, "one dtype and device"))
        bad_states.append(({name: value.to("meta") for name, value in state.items()}, "materialized"))
        for bad, message in bad_states:
            with self.subTest(message=message), patch.object(source, "state_dict", return_value=bad):
                with self.assertRaisesRegex(ValueError, message):
                    weights.build_bfl_model(source)

    def test_configuration_and_live_normalization_contracts_are_checked(self):
        for overrides, field in ((dict(patch_size=2), "patch_size"), (dict(out_channels=16), "out_channels"),
                                 (dict(timestep_guidance_channels=128), "timestep_guidance_channels"),
                                 (dict(eps=1e-5), "eps")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                weights.build_bfl_model(tiny_model(**overrides))
        model = tiny_model()
        model.transformer_blocks[0].attn.norm_q.eps = 1e-5
        with self.assertRaisesRegex(ValueError, "normalization eps"):
            weights.build_bfl_model(model)
        model = tiny_model()
        model.pos_embed.theta = 10000
        with self.assertRaisesRegex(ValueError, "RoPE module"):
            weights.build_bfl_model(model)


if __name__ == "__main__":
    unittest.main()
