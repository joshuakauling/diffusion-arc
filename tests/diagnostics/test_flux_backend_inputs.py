"""Input identity, target exclusion, and native Qwen contract regression checks."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from scripts.diagnostics import _flux_backend_inputs as inputs


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "configs/diagnostics/flux_backend_inputs.json"


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        return [ord(c) for c in text]


class NativeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.template_kwargs = messages, kwargs
        return "native image prompt"

    def __call__(self, text, **kwargs):
        self.tokenizer_kwargs = kwargs
        return {"input_ids": torch.arange(512).unsqueeze(0),
                "attention_mask": (torch.arange(512) < 7).unsqueeze(0)}


class NativeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)

    @property
    def dtype(self):
        return self.weight.dtype

    @property
    def device(self):
        return self.weight.device

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(hidden_states=tuple(
            torch.full((1, 512, 2560), i, dtype=torch.bfloat16) if i in (9, 18, 27) else None
            for i in range(28)
        ))


class BackendInputTests(unittest.TestCase):
    def test_fixture_matches_actual_report_and_completed_exposures(self):
        fixture = inputs.load_fixture(FIXTURE_PATH)
        report_path = ROOT / fixture["overfit"]["report_path"]
        self.assertEqual(hashlib.sha256(report_path.read_bytes()).hexdigest(),
                         fixture["overfit"]["report_sha256"])
        report = json.loads(report_path.read_text())
        self.assertEqual(fixture["overfit"]["checkpoint_manifest"],
                         report["stages"]["augmentation-generalization-overfit"]["saved_checkpoint"]["manifest"])
        for view in fixture["arc_views"]:
            matching = [b["step"] for b in report["training_schedule"]["batches"] if view["key"] in b["keys"]]
            self.assertEqual(matching, view["historical_continuation_exposure_steps"])
            task_path = ROOT / view["source_file"]
            if task_path.is_file():
                self.assertEqual(hashlib.sha256(task_path.read_bytes()).hexdigest(), view["source_file_sha256"])
        self.assertEqual(fixture["arc_views"][0]["historical_cumulative_exposure_audit"]["total"], 64)

    def test_fixture_rejects_changed_grids_leakage_and_false_seen_labels(self):
        original = inputs.load_fixture(FIXTURE_PATH)
        mutations = (
            lambda f: f["arc_views"][0]["source_content"]["train"][0]["input"][0].__setitem__(0, 8),
            lambda f: f["source_split"].__setitem__("split", "evaluation"),
            lambda f: f["source_split"]["train_task_ids"].remove("3c9b0459"),
            lambda f: f["arc_views"][0]["historical_cumulative_exposure_audit"].__setitem__("total", 0),
            lambda f: f["preprocessing"].__setitem__("palette", "standard_arc"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            for mutate in mutations:
                fixture = copy.deepcopy(original)
                mutate(fixture)
                path.write_text(json.dumps(fixture))
                with self.assertRaises(ValueError):
                    inputs.load_fixture(path)

    def test_arc_inputs_exclude_target_and_share_noise_across_views_and_timesteps(self):
        fixture = inputs.load_fixture(FIXTURE_PATH)
        for view in fixture["arc_views"]:
            view["prefix_ids_sha256"], view["expected_prefix_tokens"] = None, None
        calls, vae_inputs = [], []

        def runner(ids, offsets):
            calls.append(ids.clone())
            self.assertEqual(offsets.tolist(), [0, ids.numel()])
            return SimpleNamespace(sequence_hidden_states=(
                ids.to(torch.bfloat16).unsqueeze(-1).expand(-1, 7680),
            ))

        def vae(images):
            vae_inputs.append(images.clone())
            return torch.full((1, 900, 128), float(len(vae_inputs)), requires_grad=True)

        result = inputs.prepare_arc_inputs(fixture, tokenizer=CharacterTokenizer(), nvarc_runner=runner,
                                          vae_encoder=vae, seed=167)
        self.assertEqual(len(result), 6)
        noise = inputs.shared_noise(167)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(vae_inputs), 3)
        for index, view in enumerate(fixture["arc_views"]):
            rendered = inputs._render_view(view)
            prefix = rendered.prompt[:rendered.targets[-1].prefix_character_end]
            self.assertEqual(calls[index].tolist(), [ord(c) for c in prefix])
            self.assertTrue(prefix.endswith("<|im_start|>assistant\n"))
            endpoint, middle = result[2 * index:2 * index + 2]
            self.assertTrue(torch.equal(endpoint.images, noise))
            torch.testing.assert_close(middle.images, (noise + (index + 1)) / 2)
            torch.testing.assert_close(endpoint.target_velocity, noise - (index + 1))
            self.assertEqual(endpoint.text.dtype, torch.float32)
            self.assertEqual(endpoint.images.device.type, "cpu")
            self.assertFalse(endpoint.target_velocity.requires_grad)
            self.assertEqual(endpoint.text_ids[0, -1].tolist(), [0, 0, 0, len(prefix) - 1])
            self.assertEqual(endpoint.image_ids[0, -1].tolist(), [0, 29, 29, 0])

    def test_arc_prefix_identity_checked_before_frozen_model_execution(self):
        fixture = inputs.load_fixture(FIXTURE_PATH)
        with self.assertRaisesRegex(ValueError, "token IDs differ"):
            inputs.prepare_arc_inputs(fixture, tokenizer=CharacterTokenizer(), nvarc_runner=None, vae_encoder=None)

    def test_native_calls_exact_helper_retains_padding_and_has_no_target(self):
        fixture = inputs.load_fixture(FIXTURE_PATH)
        encoder, tokenizer = NativeEncoder(), NativeTokenizer()
        result = inputs.prepare_native_input(fixture, text_encoder=encoder, tokenizer=tokenizer, seed=167)
        self.assertIsNone(result.target_velocity)
        self.assertEqual(result.text.shape, (1, 512, 7680))
        self.assertEqual(result.text.dtype, torch.float32)
        for index, layer in enumerate((9, 18, 27)):
            self.assertTrue(torch.all(result.text[:, :, index * 2560:(index + 1) * 2560] == layer))
        self.assertEqual(tokenizer.template_kwargs,
                         {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})
        self.assertEqual(tokenizer.tokenizer_kwargs,
                         {"return_tensors": "pt", "padding": "max_length", "truncation": True, "max_length": 512})
        self.assertEqual(result.metadata["native_qwen_call"]["unpadded_tokens"], 7)
        self.assertEqual(result.text_ids[0, -1].tolist(), [0, 0, 0, 511])
        self.assertTrue(torch.equal(result.images, inputs.shared_noise(167)))
        self.assertFalse(encoder._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
