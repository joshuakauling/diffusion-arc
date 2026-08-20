"""Parity tests for the repository-local FLUX.2 transformer source."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest

import torch
from diffusers import Flux2Transformer2DModel as UpstreamFlux2Transformer2DModel

from diffusion_arc.model.flux_transformer import (
    Flux2Transformer2DModel as LocalFlux2Transformer2DModel,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SOURCE = (
    REPOSITORY_ROOT
    / "src"
    / "diffusion_arc"
    / "_third_party"
    / "diffusers_flux2"
    / "transformer_flux2.py"
)
UPSTREAM_SHA256 = "0a3c56ed3ca28a416492b50953a433f20b8a89fd26c7b7a49b1b6b45d5236006"
LOCAL_SHA256 = "20a84b9cd636e5a69ef4d82591f3c964427bb5da3afa1f0a6c933aed463eb50f"


def _tiny_model(model_class):
    return model_class(
        in_channels=8,
        out_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    ).eval()


def _inputs() -> dict[str, torch.Tensor | None | dict[str, torch.Tensor]]:
    torch.manual_seed(19)
    image_states = torch.randn(2, 3, 8)
    context = torch.randn(2, 4, 12)
    image_ids = torch.zeros((2, 3, 4), dtype=torch.int64)
    image_ids[:, :, 2] = torch.arange(3)
    text_ids = torch.zeros((2, 4, 4), dtype=torch.int64)
    text_ids[:, :, 3] = torch.arange(4)
    attention_mask = torch.tensor(
        [
            [[[True, True, False, False, True, True, True]]],
            [[[True, True, True, True, True, True, True]]],
        ]
    )
    return {
        "hidden_states": image_states,
        "encoder_hidden_states": context,
        "timestep": torch.tensor([0.25, 0.75]),
        "img_ids": image_ids,
        "txt_ids": text_ids,
        "guidance": None,
        "joint_attention_kwargs": {"attention_mask": attention_mask},
    }


class FluxTransformerSourceTests(unittest.TestCase):
    def test_sources_match_pinned_hashes(self) -> None:
        upstream_source = Path(inspect.getfile(UpstreamFlux2Transformer2DModel))

        self.assertEqual(
            hashlib.sha256(upstream_source.read_bytes()).hexdigest(),
            UPSTREAM_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(LOCAL_SOURCE.read_bytes()).hexdigest(),
            LOCAL_SHA256,
        )

    def test_local_model_matches_upstream_outputs_and_gradients(self) -> None:
        torch.manual_seed(11)
        upstream = _tiny_model(UpstreamFlux2Transformer2DModel)
        local = _tiny_model(LocalFlux2Transformer2DModel)
        local.load_state_dict(upstream.state_dict(), strict=True)

        upstream_output = upstream(**_inputs()).sample
        local_output = local(**_inputs()).sample
        torch.testing.assert_close(local_output, upstream_output, atol=0.0, rtol=0.0)

        upstream_output.square().mean().backward()
        local_output.square().mean().backward()
        upstream_parameters = dict(upstream.named_parameters())
        local_parameters = dict(local.named_parameters())
        self.assertEqual(upstream_parameters.keys(), local_parameters.keys())
        for name in upstream_parameters:
            with self.subTest(parameter=name):
                torch.testing.assert_close(
                    local_parameters[name].grad,
                    upstream_parameters[name].grad,
                    atol=0.0,
                    rtol=0.0,
                )

    def test_loads_diffusers_checkpoint_layout(self) -> None:
        torch.manual_seed(23)
        upstream = _tiny_model(UpstreamFlux2Transformer2DModel)

        with tempfile.TemporaryDirectory() as checkpoint_directory:
            upstream.save_pretrained(checkpoint_directory)
            local = LocalFlux2Transformer2DModel.from_pretrained(checkpoint_directory)

        self.assertEqual(local.state_dict().keys(), upstream.state_dict().keys())
        for name, value in upstream.state_dict().items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(local.state_dict()[name], value)


if __name__ == "__main__":
    unittest.main()
