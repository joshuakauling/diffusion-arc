"""Packing kernel contracts with independent CPU Math and fake CUDA bindings."""

from contextlib import nullcontext
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.diagnostics import _flux_packing_replay as replay


class PackingReplayTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        generator = torch.Generator().manual_seed(37)
        self.captures = tuple(tuple(torch.randn(1, length, 2, 8, dtype=torch.bfloat16,
                                               generator=generator) for _ in range(3))
                              for length in (3, 5))

    def test_math_three_seeds_pack_permute_repeat_isolate_and_compare_directly(self):
        partials = []
        def save(result):
            partials.append(json.loads(json.dumps(result, allow_nan=False)))
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            result = replay.replay_pair(self.captures, seeds=(11, 13, 17), profile=True,
                                        backends=("fp32_math", "bf16_math"), on_result=save)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["evaluations_completed"], 36)
        self.assertEqual(result["evaluations_attempted"], 36)
        self.assertEqual(result["plan"]["total_evaluations_maximum"], 36)
        self.assertEqual(result["segment_lengths"], [3, 5])
        self.assertEqual(len(result["rows"]), 6)
        self.assertEqual(len(result["pairwise"]), 3)
        self.assertTrue(result["input_hashes_unchanged"])
        self.assertFalse(result["training_qualified"])
        self.assertTrue(replay.validate_replay_result(result, seeds=(11, 13, 17), profile=True,
                                                      backends=("fp32_math", "bf16_math")))
        self.assertEqual(partials[-1], result)
        self.assertEqual(partials[0]["status"], "running")
        for row in result["rows"]:
            self.assertEqual(row["status"], "complete")
            for kind in ("packed_vs_singletons", "reversed_vs_singletons", "reversed_vs_packed", "repeat_vs_packed"):
                for sample in ("A", "B"):
                    for metric in row[kind][sample].values():
                        self.assertEqual(metric["max_abs"], 0)
            if row["seed"] == 11:
                self.assertTrue(row["dispatch"]["backend_verified"])
                self.assertEqual(row["dispatch"]["attention_calls"], 2)
                self.assertEqual(row["dispatch"]["layout"], "segmented_sdpa_control")
                self.assertTrue(row["isolation"]["A_only_upstream"]["exact_zero"])
                for changed in ("changed_A", "changed_B"):
                    self.assertTrue(row["isolation"][changed]["all_tensors_exact"])
        for row in result["pairwise"]:
            self.assertEqual((row["actual"], row["reference"]), ("bf16_math", "fp32_math"))
            self.assertEqual(set(row["comparisons"]), {"A", "B"})
            self.assertLess(row["comparisons"]["B"]["dQ"]["relative_l2"], .03)
        for sample in self.captures:
            for tensor in sample:
                self.assertIsNone(tensor.grad)

    def test_varlen_boundaries_restore_asymmetric_sample_order_and_backward(self):
        for name in ("fa2", "fa3", "fa4"):
            seen = []
            def kernel(q, k, v, *, deterministic=False, **kwargs):
                seen.append((q.clone(), kwargs, deterministic))
                output = q + 2 * k + 3 * v
                return (output, None) if name == "fa4" else output
            backend = SimpleNamespace(kernel=kernel)
            incoming = tuple(torch.full_like(sample[0], index + 1) for index, sample in enumerate(self.captures))
            with patch.object(replay, "_flash", return_value=backend):
                values, evidence = replay._evaluate(name, self.captures, incoming, (1, 0), deterministic=True)
            self.assertEqual(seen[0][1]["cu_seqlens_q"].tolist(), [0, 5, 8])
            self.assertEqual(seen[0][1]["cu_seqlens_k"].dtype, torch.int32)
            self.assertEqual(seen[0][1]["max_seqlen_q"], 5)
            self.assertFalse(seen[0][1]["causal"])
            self.assertTrue(seen[0][2])
            self.assertEqual(evidence["layout"], "native_varlen")
            for index, label in enumerate(("A", "B")):
                for actual, scale in zip(values[label][1:], (1, 2, 3)):
                    torch.testing.assert_close(actual, torch.full_like(actual, (index + 1) * scale),
                                               atol=0, rtol=0)
                q, k, v = self.captures[index]
                torch.testing.assert_close(values[label][0], (q + 2 * k + 3 * v).squeeze(0), atol=0, rtol=0)

    def test_production_omits_deterministic_and_singletons_use_same_varlen_binding(self):
        calls = []
        def kernel(q, k, v, **kwargs):
            calls.append(kwargs)
            return q + k + v
        with patch.object(replay, "_flash", return_value=SimpleNamespace(kernel=kernel)):
            for lengths in ((3,), (5,), (3, 5), (5, 3)):
                value = torch.ones(sum(lengths), 2, 8, dtype=torch.bfloat16)
                replay.attention_forward("fa3", value, value, value, lengths)
        self.assertEqual([row["cu_seqlens_q"].tolist() for row in calls], [[0, 3], [0, 5], [0, 3, 8], [0, 5, 8]])
        self.assertTrue(all("deterministic" not in row for row in calls))

    def test_cudnn_forces_two_separate_noncausal_calls(self):
        value = torch.ones(8, 2, 8, dtype=torch.bfloat16)
        with patch.object(replay, "sdpa_kernel", return_value=nullcontext()) as selection:
            with patch.object(replay.F, "scaled_dot_product_attention", side_effect=lambda q, k, v, **kw: q) as call:
                actual = replay.attention_forward("cudnn", value, value, value, (3, 5))
        selection.assert_called_once_with(replay.SDPBackend.CUDNN_ATTENTION)
        self.assertEqual(call.call_count, 2)
        self.assertEqual([args.args[0].shape[2] for args in call.call_args_list], [3, 5])
        self.assertEqual(call.call_args.kwargs, dict(dropout_p=0.0, is_causal=False, scale=8 ** -.5))
        torch.testing.assert_close(actual, value, atol=0, rtol=0)

    def test_identical_upstream_is_promoted_and_associated_with_sample(self):
        def linear(name, q, k, v, lengths, **kwargs):
            return q + k + v + (10 if name == "bf16_math" else 0)
        with patch.object(replay, "attention_forward", side_effect=linear):
            result = replay.replay_pair(self.captures, seeds=(19,), backends=("fp32_math", "bf16_math"))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["rows"][0]["upstreams"], result["rows"][1]["upstreams"])
        for sample in result["pairwise"][0]["comparisons"].values():
            self.assertGreater(sample["output"]["rms"], 9)
            self.assertTrue(all(sample[field]["max_abs"] == 0 for field in ("dQ", "dK", "dV")))

    def test_cross_sample_leak_is_invalid_and_other_backend_continues(self):
        original = replay.attention_forward
        def leaking(name, q, k, v, lengths, **kwargs):
            if name == "fp32_math" and len(lengths) == 2:
                return q + k + v + q.mean()
            return original(name, q, k, v, lengths, **kwargs)
        with patch.object(replay, "attention_forward", side_effect=leaking):
            result = replay.replay_pair(self.captures, seeds=(19,), backends=("fp32_math", "bf16_math"))
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertIn("affected B output", result["rows"][0]["error"])
        self.assertEqual(result["rows"][1]["status"], "complete")
        self.assertEqual(result["pairwise"], [])
        self.assertFalse(replay.validate_replay_result(result, seeds=(19,), backends=("fp32_math", "bf16_math")))

    def test_disconnected_gradient_detects_backward_only_leak(self):
        class Leaking(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, k, v):
                return q + k + v

            @staticmethod
            def backward(ctx, incoming):
                return incoming + incoming.mean(), incoming, incoming
        with patch.object(replay, "attention_forward", side_effect=lambda name, q, k, v, lengths, **kw: Leaking.apply(q, k, v)):
            result = replay.replay_pair(self.captures, seeds=(19,), backends=("fp32_math",))
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertFalse(result["rows"][0]["isolation"]["A_only_upstream"]["exact_zero"])

    def test_mutation_is_reported_without_changing_caller_or_precision_state(self):
        saved = tuple(tuple(value.clone() for value in sample) for sample in self.captures)
        previous = torch.get_float32_matmul_precision()
        def mutating(name, q, k, v, lengths, **kwargs):
            with torch.no_grad():
                q.add_(1)
            return q + k + v
        with patch.object(replay, "attention_forward", side_effect=mutating):
            result = replay.replay_pair(self.captures, seeds=(19,), backends=("fp32_math",))
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertIn("mutated a Q/K/V", result["errors"][0]["error"])
        self.assertTrue(result["input_hashes_unchanged"])
        self.assertEqual(torch.get_float32_matmul_precision(), previous)
        for original, before in zip(self.captures, saved):
            for actual, expected in zip(original, before):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_deterministic_binding_support_is_explicit_and_production_stays_default(self):
        seen = []
        def kernel(q, k, v, *, deterministic=False, **kwargs):
            seen.append(deterministic)
            return q + k + v
        # The fake binding runs on CPU; the normal public preflight rejects real
        # CUDA backends here, as verified separately in the ingestion test.
        with patch.object(replay, "_validate"), patch.object(replay, "_flash", return_value=SimpleNamespace(kernel=kernel)):
            result = replay.replay_pair(self.captures, seeds=(19,), backends=("fp32_math", "bf16_math", "fa3"))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["evaluations_completed"], 29)
        self.assertEqual(seen, [False] * 8 + [True] * 5)
        self.assertEqual(result["deterministic"][-1]["status"], "complete")
        self.assertTrue(result["deterministic"][-1]["repeat_exact"])
        self.assertTrue(replay.validate_replay_result(result, seeds=(19,), backends=("fp32_math", "bf16_math", "fa3")))

    def test_missing_or_unsupported_deterministic_argument_is_not_silently_dropped(self):
        for kind in ("absent", "unsupported"):
            def absent(q, k, v, **kwargs):
                return q + k + v
            def unsupported(q, k, v, *, deterministic=False, **kwargs):
                if deterministic:
                    raise NotImplementedError("deterministic backward is not supported")
                return q + k + v
            kernel = absent if kind == "absent" else unsupported
            with self.subTest(kind=kind), patch.object(replay, "_validate"), patch.object(
                    replay, "_flash", return_value=SimpleNamespace(kernel=kernel)):
                result = replay.replay_pair(self.captures, seeds=(19,), backends=("fp32_math", "bf16_math", "fa3"))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["evaluations_completed"], 24)
            self.assertEqual(result["evaluations_attempted"], 24 if kind == "absent" else 25)
            self.assertEqual(result["deterministic"][-1]["status"], "unsupported")
            self.assertIn("deterministic", result["deterministic"][-1]["reason"])
            self.assertTrue(replay.validate_replay_result(result, seeds=(19,), backends=("fp32_math", "bf16_math", "fa3")))

    def test_completeness_rejects_missing_metrics_duplicate_cases_and_wrong_dispatch(self):
        result = replay.replay_pair(self.captures, seeds=(19,), profile=True, backends=("fp32_math", "bf16_math"))
        corruptions = {
            "removed tensor": lambda value: value["rows"][0]["packed_vs_singletons"]["A"].pop("dQ"),
            "removed metric": lambda value: value["pairwise"][0]["comparisons"]["B"]["output"].pop("rms"),
            "nonfinite": lambda value: value["rows"][0]["reversed_vs_packed"]["A"]["output"].update(max_abs=float("nan")),
            "missing pair": lambda value: value["pairwise"].clear(),
            "duplicate row": lambda value: value["rows"].append(deepcopy(value["rows"][0])),
            "wrong count": lambda value: value.update(evaluations_completed=100),
            "changed upstream": lambda value: value["rows"][1]["upstreams"]["A"].update(sha256="0" * 64),
            "missing backward": lambda value: value["rows"][0]["dispatch"]["operators"].pop("aten::_softmax_backward_data"),
            "missing disconnected": lambda value: value["rows"][0]["isolation"]["A_only_upstream"]["gradients"].pop("dK"),
            "missing deterministic": lambda value: value["deterministic"].clear(),
        }
        for label, corrupt in corruptions.items():
            bad = deepcopy(result)
            # Runtime rows intentionally share immutable upstream identities;
            # serialized artifacts do not, so mirror that persistence boundary.
            bad = json.loads(json.dumps(bad))
            corrupt(bad)
            with self.subTest(label=label), self.assertRaises(ValueError):
                replay.validate_replay_result(bad, seeds=(19,), profile=True, backends=("fp32_math", "bf16_math"))

    def test_configuration_and_ingestion_reject_invalid_inputs(self):
        self.assertEqual(replay.evaluation_plan((1, 2, 3))["total_evaluations_maximum"], 123)
        for seeds in ((), (1, 1), (-1,), (True,)):
            with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, "63-bit"):
                replay.replay_pair(self.captures, seeds=seeds, backends=("fp32_math",))
        with self.assertRaisesRegex(ValueError, "require CUDA"):
            replay.replay_pair(self.captures, seeds=(1,))
        with self.assertRaisesRegex(ValueError, "contiguous BF16"):
            replay.replay_pair(tuple(tuple(value.float() for value in sample) for sample in self.captures),
                               seeds=(1,), backends=("fp32_math",))
        with self.assertRaisesRegex(ValueError, "exactly two"):
            replay.replay_pair(self.captures[:1], seeds=(1,), backends=("fp32_math",))
        self.captures[0][0][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            replay.replay_pair(self.captures, seeds=(1,), backends=("fp32_math",))
