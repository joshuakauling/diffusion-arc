"""Padding placement, actual attention VJPs and bounded replay coordination."""

import copy
import json
import unittest
from unittest.mock import patch

import torch

from scripts.diagnostics import _flux_boundary_replay as boundary
from scripts.diagnostics import _flux_packing_replay as replay


def operands():
    generator = torch.Generator().manual_seed(193)
    captures = tuple(tuple(torch.randn(1, length, 1, 4, generator=generator, dtype=torch.bfloat16)
                           for _ in range(3)) for length in (4, 5))
    upstreams = tuple(torch.randn(sample[0].shape, generator=generator, dtype=torch.bfloat16) for sample in captures)
    return captures, upstreams


def records():
    captures, upstreams = operands()
    return {label: {stage: {"qkv": tuple(value.clone() for value in captures[index]),
                           "upstream": upstreams[index].clone()}
                    for stage in boundary.STAGES}
            for index, label in enumerate(replay.SAMPLES)}


class BoundaryReplayTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)

    def test_middle_padding_keeps_all_image_tokens_and_zeroes_only_padding_upstream(self):
        captures, upstreams = operands()
        padded, incoming, valid = boundary._pad_inputs(captures, upstreams, (2, 3), 5)
        self.assertEqual(valid.tolist(), [[True, True, False, False, False, True, True],
                                         [True, True, True, False, False, True, True]])
        for index in range(2):
            for actual, source in zip(padded, captures[index], strict=True):
                torch.testing.assert_close(actual[index, valid[index]], source[0], atol=0, rtol=0)
                self.assertEqual(actual[index, ~valid[index]].count_nonzero().item(), 0)
            torch.testing.assert_close(incoming[index, valid[index]], upstreams[index][0], atol=0, rtol=0)
            self.assertEqual(incoming[index, ~valid[index]].count_nonzero().item(), 0)

    def test_real_padded_backward_matches_unpadded_attention_and_has_no_padding_gradients(self):
        captures, upstreams = operands()
        original = copy.deepcopy((captures, upstreams))
        padded, evidence = boundary._evaluate_padded(captures, upstreams, (2, 3), 5,
                                                      profile=True, backend="bf16_math")
        for index, label in enumerate(replay.SAMPLES):
            singleton, _ = replay._evaluate("bf16_math", captures, upstreams, (index,))
            for actual, expected in zip(padded[label], singleton[label], strict=True):
                torch.testing.assert_close(actual, expected, atol=.008, rtol=.008)
            self.assertTrue(all(row["nonzero_count"] == 0 for row in evidence["padding_gradients"][label].values()))
        self.assertTrue(evidence["backend_verified"])
        self.assertEqual(evidence["attention_calls"], 1)
        self.assertEqual(evidence["padding_token_counts"], [3, 2])
        for actual_samples, original_samples in zip((captures, upstreams), original, strict=True):
            for actual, expected in zip(actual_samples, original_samples, strict=True):
                if isinstance(actual, tuple):
                    for value, reference in zip(actual, expected, strict=True):
                        torch.testing.assert_close(value, reference, atol=0, rtol=0)
                else:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_coordinator_uses_real_upstreams_repeats_every_case_and_stays_within_budget(self):
        source = records()
        evaluated, padded_calls, progress = [], [], []
        original_evaluate = replay._evaluate
        original_padded = boundary._evaluate_padded

        def cpu_evaluate(name, captures, upstreams, order, *, profile=False):
            evaluated.append((name, order, profile))
            for index, label in enumerate(replay.SAMPLES):
                torch.testing.assert_close(upstreams[index], source[label]["double.0"]["upstream"], atol=0, rtol=0)
            return original_evaluate("bf16_math", captures, upstreams, order, profile=False)

        def cpu_padded(captures, upstreams, text_lengths, padded_length, *, profile=False):
            padded_calls.append(profile)
            return original_padded(captures, upstreams, text_lengths, padded_length, profile=False, backend="bf16_math")

        with patch.object(replay, "_evaluate", side_effect=cpu_evaluate), \
             patch.object(boundary, "_evaluate_padded", side_effect=cpu_padded):
            result = boundary.replay_boundaries(source, text_lengths={"A": 2, "B": 3}, device="cpu", padded_length=5,
                                               edge_width=1, on_result=lambda row: progress.append((
                                                   row["evaluations_completed"], row["status"])))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["evaluations_attempted"], 72)
        self.assertEqual(result["evaluations_completed"], 72)
        self.assertEqual(len(evaluated), 64)
        self.assertEqual(len(padded_calls), 8)
        self.assertEqual(sum(row[2] for row in evaluated) + sum(padded_calls), 3)
        self.assertEqual(progress[0], (0, "running"))
        self.assertEqual(progress[-1], (72, "complete"))
        for stage in result["stages"].values():
            for backend, row in stage["backends"].items():
                self.assertEqual(len(row["executions"]), 2)
                self.assertEqual(set(row["repeat_comparisons"]),
                                 {"singletons", "ab", "ba", "padded"} if backend == "cudnn" else {"singletons", "ab", "ba"})
                for comparison in row["repeat_comparisons"].values():
                    for sample in comparison.values():
                        self.assertTrue(all(field["summary"]["max_abs"] == 0 for field in sample.values()))
                for execution in row["executions"]:
                    self.assertIn("ba_vs_ab", execution["comparisons"])
        json.dumps(result, allow_nan=False)

    def test_bad_inputs_fail_before_compute_and_runtime_failure_publishes_without_retry(self):
        source = records()
        with patch.object(replay, "_evaluate") as evaluate:
            broken = copy.deepcopy(source)
            broken["B"]["single.19"]["upstream"] = broken["B"]["single.19"]["upstream"].float()
            with self.assertRaisesRegex(ValueError, "CPU BF16"):
                boundary.replay_boundaries(broken, text_lengths={"A": 2, "B": 3}, device="cpu", padded_length=5)
            evaluate.assert_not_called()
        progress = []
        with patch.object(replay, "_evaluate", side_effect=RuntimeError("kernel failure")) as evaluate:
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                boundary.replay_boundaries(source, text_lengths={"A": 2, "B": 3}, device="cpu", padded_length=5,
                                           on_result=lambda row: progress.append((row["status"], row["evaluations_attempted"],
                                                                                 row["evaluations_completed"])))
            self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(progress[-1], ("failed", 1, 0))


if __name__ == "__main__":
    unittest.main()
