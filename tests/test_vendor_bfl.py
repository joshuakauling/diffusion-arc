from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import torch

from diffusion_arc._vendor.bfl_flux2.model import Flux2, Flux2Params, Klein4BParams
from diffusion_arc.positions import make_image_ids, make_text_ids


VENDOR_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "diffusion_arc"
    / "_vendor"
    / "bfl_flux2"
)


class BFLVendorTest(unittest.TestCase):
    def test_vendored_sources_match_pinned_upstream_hashes(self):
        expected = {
            "model.py": "e062691d789ec80a04a5fbc421b09d3b7e264305be1dfcb6da65d68c259e18c1",
            "autoencoder.py": (
                "bb578f78d097a88f6e7e8aa01032c10f55659729b85203ae2d5852a7889ac96a"
            ),
            "LICENSE.md": (
                "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
            ),
        }

        for name, digest in expected.items():
            actual = hashlib.sha256((VENDOR_ROOT / name).read_bytes()).hexdigest()
            self.assertEqual(actual, digest)

    def test_tiny_official_flux_forward_backward(self):
        model = Flux2(
            Flux2Params(
                in_channels=8,
                context_in_dim=12,
                hidden_size=32,
                num_heads=2,
                depth=1,
                depth_single_blocks=1,
                axes_dim=[4, 4, 4, 4],
                theta=2000,
                mlp_ratio=1.0,
                use_guidance_embed=False,
            )
        )
        image = torch.randn(1, 4, 8)
        context = torch.randn(1, 3, 12)

        velocity = model(
            x=image,
            x_ids=make_image_ids(1, 2, 2),
            timesteps=torch.tensor([0.5]),
            ctx=context,
            ctx_ids=make_text_ids(1, 3),
            guidance=None,
        )
        velocity.square().mean().backward()

        self.assertEqual(velocity.shape, image.shape)
        self.assertTrue(torch.isfinite(velocity).all())
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(
            all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        )

    def test_klein_4b_parameter_contract(self):
        params = Klein4BParams()

        self.assertEqual(params.in_channels, 128)
        self.assertEqual(params.context_in_dim, 7680)
        self.assertEqual(params.hidden_size, 3072)
        self.assertEqual(params.num_heads, 24)
        self.assertEqual(params.depth, 5)
        self.assertEqual(params.depth_single_blocks, 20)
        self.assertFalse(params.use_guidance_embed)


if __name__ == "__main__":
    unittest.main()
