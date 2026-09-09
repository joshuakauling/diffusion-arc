"""Check passive captures, actual FLUX replays, and analytical ownership sums."""

import copy
import json
import unittest
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from diffusion_arc.model.flux_transformer import Flux2Transformer2DModel
from scripts.diagnostics import _flux_operation_replay as replay


def fixture():
    model = Flux2Transformer2DModel(in_channels=8, out_channels=8, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=12,
        axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval()
    calls = []
    for length, timestep in ((2, .17), (5, .83)):
        calls.append(dict(hidden_states=torch.randn(1, 3, 8), encoder_hidden_states=torch.randn(1, length, 12),
                          timestep=torch.tensor([timestep]), img_ids=torch.zeros(1, 3, 4, dtype=torch.int64),
                          txt_ids=torch.arange(length)[:, None].expand(length, 4)[None], guidance=None))
    return model, calls


def hook_counts(model):
    return [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()]


def captured(model, calls):
    captures, outputs = [], []
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cpu", dtype=torch.bfloat16):
        for call in calls:
            with replay.OperationCapture(model) as capture:
                outputs.append(model(**call).sample)
            captures.append(capture)
    return tuple(captures), outputs


class OperationReplayTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        torch.manual_seed(327)

    def test_capture_is_passive_has_correct_tuple_mapping_and_restores_hooks(self):
        model, calls = fixture()
        before = hook_counts(model)
        captures, actual = captured(model, calls)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cpu", dtype=torch.bfloat16):
            expected = [model(**call).sample for call in calls]
        self.assertEqual(hook_counts(model), before)
        for capture, output, original, length in zip(captures, actual, expected, (2, 5)):
            torch.testing.assert_close(output, original, rtol=0, atol=0)
            torch.testing.assert_close(capture.values["projection.output"], output, rtol=0, atol=0)
            self.assertEqual(capture.values["first.text.attention"].shape, (1, length, 16))
            self.assertEqual(capture.values["first.image.attention"].shape, (1, 3, 16))
            self.assertEqual(capture.values["first.text.output"].shape, (1, length, 16))
            self.assertEqual(len(capture.values), 28)
            self.assertTrue(all(not value.requires_grad for value in capture.values.values()))
            self.assertEqual(capture.values["modulation.image.output"].dtype, torch.bfloat16)

    def test_frozen_replays_match_original_operations_and_repeat_with_sample_identity(self):
        model, calls = fixture()
        captures, _ = captured(model, calls)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        rng = torch.random.get_rng_state().clone()
        with patch.object(model, "forward", side_effect=AssertionError("unexpected full forward")), \
             patch.object(model.transformer_blocks[0].attn, "forward", side_effect=AssertionError("unexpected attention")):
            result = replay.operation_replays(model, captures)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(replay.validate_operation_result(result))
        self.assertTrue(replay.validate_operation_result(json.loads(json.dumps(result, sort_keys=True))))
        self.assertTrue(result["input_hashes_unchanged"])
        self.assertEqual(len(result["forward_cases"]), 130)
        self.assertEqual(len(result["reduction_cases"]), 72)
        self.assertTrue(result["capture_replay_bitwise_equal"])
        for operation, labels in result["capture_replay"].items():
            for label, stages in labels.items():
                for stage, metrics in stages.items():
                    self.assertTrue(metrics["bitwise_equal"], (operation, label, stage, metrics))
        for row in result["forward_cases"]:
            if row["operation"] == "projection" and row["layout"] == "ab":
                self.assertEqual(row["input_shapes"]["input"], [2, 3, 16])
            if row["operation"] == "text.norm" and row["layout"] == "ba":
                self.assertEqual(row["input_shapes"]["input"], [1, 7, 16])
                self.assertEqual(row["input_shapes"]["gate"], [1, 7, 16])
                self.assertEqual(row["input_strides"]["gate"][-2], 6 * 16)
            if row["layout"] == "ba":
                self.assertEqual(row["labels"], ["B", "A"])
            if row["layout"] == "ab_repeat":
                for stages in row["comparisons"]["ab"].values():
                    for metrics in stages.values():
                        self.assertTrue(metrics["bitwise_equal"])
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad)
        self.assertEqual(len({row["seed"] for row in result["reduction_cases"]}), 3)
        for module in replay.MODULATIONS:
            for seed in replay.DEFAULT_SEEDS:
                rows = [row for row in result["reduction_cases"] if row["module"] == module and row["seed"] == seed]
                self.assertEqual({row["precision"] for row in rows}, {"native_bf16", "fp32_reduction_diagnostic"})
                self.assertTrue(all(row["upstream_identity"] == rows[0]["upstream_identity"] for row in rows))
        json.dumps(result, allow_nan=False)

    def test_result_validator_rejects_missing_nonfinite_and_unfaithful_evidence(self):
        model, calls = fixture()
        captures, _ = captured(model, calls)
        result = replay.operation_replays(model, captures)
        missing = copy.deepcopy(result)
        missing["forward_cases"][2]["comparisons"]["singletons"]["A"]["output"].pop("reference_l2")
        with self.assertRaisesRegex(ValueError, "metric fields"):
            replay.validate_operation_result(missing)
        nonfinite = copy.deepcopy(result)
        nonfinite["reduction_cases"][0]["comparisons"]["fp32_broadcast"]["A"]["max_abs"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            replay.validate_operation_result(nonfinite)
        unfaithful = copy.deepcopy(result)
        unfaithful["capture_replay"]["projection"]["A"]["output"]["bitwise_equal"] = False
        with self.assertRaisesRegex(ValueError, "same-shape"):
            replay.validate_operation_result(unfaithful)
        incomplete = copy.deepcopy(result)
        incomplete["reduction_cases"].pop()
        with self.assertRaisesRegex(ValueError, "reduction cases"):
            replay.validate_operation_result(incomplete)
        upstream = copy.deepcopy(result)
        upstream["reduction_cases"][1]["upstream_identity"] = {
            **result["reduction_cases"][1]["upstream_identity"],
            "A": {**result["reduction_cases"][1]["upstream_identity"]["A"], "sha256": "f" * 64}}
        with self.assertRaisesRegex(ValueError, "upstream changed"):
            replay.validate_operation_result(upstream)

    def test_precision_flags_are_restored_when_replay_raises(self):
        model, calls = fixture()
        captures, _ = captured(model, calls)
        before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                  torch.backends.cudnn.allow_tf32)
        with patch.object(replay, "_forward_replays", side_effect=RuntimeError("injected replay failure")):
            with self.assertRaisesRegex(RuntimeError, "injected replay failure"):
                replay.operation_replays(model, captures)
        self.assertEqual((torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                          torch.backends.cudnn.allow_tf32), before)

    def test_repeated_owners_sum_the_same_labelled_upstream_and_do_not_mix_samples(self):
        values = (torch.tensor([[10., 20.]], dtype=torch.bfloat16),
                  torch.tensor([[30., 40.]], dtype=torch.bfloat16))
        upstream = (torch.tensor([[[1., 2.], [4., 8.]]], dtype=torch.bfloat16),
                    torch.tensor([[[16., 32.], [64., 128.], [0., 1.]]], dtype=torch.bfloat16))
        expected = (torch.tensor([[5., 10.]], dtype=torch.bfloat16),
                    torch.tensor([[80., 161.]], dtype=torch.bfloat16))
        for indexed in (False, True):
            for order in ((0, 1), (1, 0)):
                actual = replay.reduction_gradient(values, upstream, order, indexed=indexed)
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
        zero_b = (upstream[0], torch.zeros_like(upstream[1]))
        for indexed in (False, True):
            actual = replay.reduction_gradient(values, zero_b, (1, 0), indexed=indexed)
            self.assertEqual(torch.count_nonzero(actual[1]).item(), 0)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "each sample once"):
            replay.reduction_gradient(values, upstream, (0, 0), indexed=True)

    def test_fp32_reduction_diagnostic_promotes_identical_upstream_before_summing(self):
        values = (torch.zeros(1, 1, dtype=torch.bfloat16), torch.zeros(1, 1, dtype=torch.bfloat16))
        upstream = (torch.ones(1, 257, 1, dtype=torch.bfloat16), torch.zeros(1, 3, 1, dtype=torch.bfloat16))
        native = replay.reduction_gradient(values, upstream, (0, 1), indexed=False)
        diagnostic = replay.reduction_gradient(tuple(value.float() for value in values),
                                               tuple(value.float() for value in upstream), (0, 1), indexed=False)
        self.assertEqual(native[0].dtype, torch.bfloat16)
        self.assertEqual(diagnostic[0].dtype, torch.float32)
        self.assertEqual(native[0].item(), 256.)
        self.assertEqual(diagnostic[0].item(), 257.)
        self.assertEqual(native[1].item(), 0.)
        self.assertEqual(diagnostic[1].item(), 0.)

    def test_failed_or_incomplete_capture_cleans_up_and_cannot_replay(self):
        model, calls = fixture()
        before = hook_counts(model)
        capture = replay.OperationCapture(model)
        with self.assertRaisesRegex(RuntimeError, "injected"), capture, \
             torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16), \
             patch.object(model.transformer_blocks[0].ff_context, "forward", side_effect=RuntimeError("injected")):
            model(**calls[0])
        self.assertEqual(hook_counts(model), before)
        self.assertFalse(capture.completed)
        with self.assertRaisesRegex(ValueError, "complete singleton captures"):
            replay.operation_replays(model, (capture, capture))
        empty = replay.OperationCapture(model)
        with self.assertRaisesRegex(ValueError, "incomplete operation capture"), empty:
            pass
        self.assertEqual(hook_counts(model), before)

    def test_replay_rejects_wrong_model_precision_and_seed_contracts(self):
        model, calls = fixture()
        captures, _ = captured(model, calls)
        other, _ = fixture()
        with self.assertRaisesRegex(ValueError, "this model"):
            replay.operation_replays(other, captures)
        with self.assertRaisesRegex(ValueError, "three distinct"):
            replay.operation_replays(model, captures, seeds=(1, 1, 2))
        captures[0].values["modulation.image.output"] = captures[0].values["modulation.image.output"].float()
        with self.assertRaisesRegex(ValueError, "production BF16"):
            replay.operation_replays(model, captures)


if __name__ == "__main__":
    unittest.main()
