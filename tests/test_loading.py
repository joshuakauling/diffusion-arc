from __future__ import annotations

import types
import unittest
from inspect import signature
from pathlib import Path

import torch
from torch import nn

from diffusion_arc.loading import (
    OFFICIAL_FLUX2_KLEIN_BASE_4B_SHA256,
    OFFICIAL_FLUX2_VAE_SHA256,
    load_bfl_autoencoder_local,
    load_bfl_flux2_klein_4b_local,
    verify_nvarc_checkpoint_contract,
)


class FakeTokenizer:
    pad_token_id = 13
    eos_token_id = 15
    unk_token_id = None
    model_max_length = 262_144

    def __len__(self):
        return 16

    def convert_tokens_to_ids(self, token: str) -> int:
        mapping = {str(index): index for index in range(10)}
        mapping.update(
            {
                "Ċ": 10,
                "user": 11,
                "assistant": 12,
                "<|endoftext|>": 13,
                "<|im_start|>": 14,
                "<|im_end|>": 15,
            }
        )
        return mapping.get(token, -1)

    def encode(self, text: str, *, add_special_tokens: bool):
        self.last_text = text
        self.last_add_special_tokens = add_special_tokens
        return [14, 11, 10, 0, 15, 14, 12, 10]


class FakeNVARC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_embeddings = nn.Embedding(16, 2560)
        self.output_embeddings = nn.Linear(2560, 16, bias=False)
        self.config = types.SimpleNamespace(
            hidden_size=2560,
            num_hidden_layers=36,
            max_position_embeddings=262_144,
        )
        self.to(dtype=torch.bfloat16)

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings


class LoadingContractTest(unittest.TestCase):
    def test_pins_official_bfl_checkpoint_hashes(self):
        self.assertEqual(len(OFFICIAL_FLUX2_KLEIN_BASE_4B_SHA256), 64)
        self.assertEqual(len(OFFICIAL_FLUX2_VAE_SHA256), 64)
        self.assertNotEqual(
            OFFICIAL_FLUX2_KLEIN_BASE_4B_SHA256,
            OFFICIAL_FLUX2_VAE_SHA256,
        )
        self.assertEqual(
            signature(load_bfl_flux2_klein_4b_local)
            .parameters["expected_sha256"]
            .default,
            OFFICIAL_FLUX2_KLEIN_BASE_4B_SHA256,
        )
        self.assertEqual(
            signature(load_bfl_autoencoder_local)
            .parameters["expected_sha256"]
            .default,
            OFFICIAL_FLUX2_VAE_SHA256,
        )

    def test_verifies_released_nvarc_contract(self):
        contract = verify_nvarc_checkpoint_contract(FakeNVARC(), FakeTokenizer())

        self.assertEqual(contract["input_embedding_rows"], 16)
        self.assertEqual(contract["output_embedding_rows"], 16)
        self.assertEqual(contract["hidden_size"], 2560)
        self.assertEqual(contract["num_hidden_layers"], 36)
        self.assertEqual(contract["max_position_embeddings"], 262_144)
        self.assertEqual(contract["tokenizer_size"], 16)
        self.assertEqual(contract["tokenizer_model_max_length"], 262_144)
        self.assertEqual(contract["parameter_dtypes"], ["torch.bfloat16"])

    def test_rejects_wrong_nvarc_dtype(self):
        model = FakeNVARC().float()

        with self.assertRaisesRegex(RuntimeError, "must be torch.bfloat16"):
            verify_nvarc_checkpoint_contract(model, FakeTokenizer())

    def test_local_bfl_loaders_fail_before_any_download(self):
        missing = Path("/definitely/missing/diffusion-arc-weights.safetensors")

        with self.assertRaises(FileNotFoundError):
            load_bfl_flux2_klein_4b_local(missing, device="cpu")
        with self.assertRaises(FileNotFoundError):
            load_bfl_autoencoder_local(missing, device="cpu")


if __name__ == "__main__":
    unittest.main()
