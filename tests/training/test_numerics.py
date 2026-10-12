"""Known-vector oracles for full-gradient comparisons and failure criteria."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from diffusion_arc.training.run.numerics import (
    NumericalCriteria, VectorDifference, compare_records, inspect_gradients, state_fingerprint,
)


class NumericsTests(unittest.TestCase):
    def test_full_vector_metric_detects_direction_change_with_identical_norm(self):
        metric = VectorDifference()
        metric.add(torch.tensor([3.0]), torch.tensor([-3.0]))
        metric.add(torch.tensor([4.0]), torch.tensor([4.0]))
        result = metric.result()
        self.assertEqual(result["reference_norm"], 5)
        self.assertEqual(result["actual_norm"], 5)
        self.assertEqual(result["difference_norm"], 6)
        self.assertAlmostEqual(result["relative_l2"], 1.2)
        self.assertAlmostEqual(result["cosine_similarity"], 7 / 25)

    def test_zero_reference_and_nonfinite_do_not_silently_pass(self):
        metric = VectorDifference()
        metric.add(torch.zeros(2), torch.zeros(2))
        self.assertEqual(metric.result()["relative_l2"], 0)
        metric.add(torch.zeros(1), torch.ones(1))
        self.assertIsNone(metric.result()["relative_l2"])
        json.dumps(metric.result(), allow_nan=False)
        with self.assertRaises(FloatingPointError):
            metric.add(torch.ones(1), torch.tensor([float("nan")]))

    def test_gradient_capture_matches_explicit_vectors_and_checks_corruption(self):
        model = nn.Linear(3, 2)
        model.weight.grad = torch.arange(6, dtype=torch.float32).view(2, 3)
        model.bias.grad = torch.tensor([1.0, -2.0])
        reference = torch.cat([p.grad.flatten() for p in model.parameters()]).double()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gradients"
            captured = inspect_gradients(model, 2, save_to=path)
            self.assertEqual((path / "gradients.bin").stat().st_size, reference.numel() * 4)
            equal = inspect_gradients(model, 3, compare_to=path)
            self.assertEqual(equal["sha256"], captured["sha256"])
            self.assertEqual(equal["comparison"]["relative_l2"], 0)
            model.weight.grad.mul_(2)
            actual = torch.cat([p.grad.flatten() for p in model.parameters()]).double()
            changed = inspect_gradients(model, 3, compare_to=path)
            expected = float(torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference))
            self.assertAlmostEqual(changed["comparison"]["relative_l2"], expected)
            self.assertEqual(set(changed["blocks"]), {"weight", "bias"})
            with (path / "gradients.bin").open("r+b") as stream:
                stream.write(torch.tensor([1.0]).numpy().tobytes())
            with self.assertRaisesRegex(ValueError, "checksum"):
                inspect_gradients(model, 3, compare_to=path)
        model.bias.grad = None
        with self.assertRaisesRegex(ValueError, "complete FP32"):
            inspect_gradients(model, 3)

    def test_state_fingerprints_cover_bf16_buffers_and_parameters(self):
        model = nn.Linear(2, 2).bfloat16()
        model.register_buffer("counter", torch.tensor(1))
        before = state_fingerprint(model, 2)
        self.assertEqual(before, state_fingerprint(model, 3))
        model.counter.add_(1)
        self.assertNotEqual(before, state_fingerprint(model, 2))
        with torch.no_grad():
            model.weight[0, 0] = float("inf")
        with self.assertRaises(FloatingPointError):
            state_fingerprint(model, 2)

    def test_capture_can_compare_single_gpu_and_ddp_references_in_one_pass(self):
        model = nn.Linear(3, 2)
        model.weight.grad = torch.arange(6, dtype=torch.float32).view(2, 3)
        model.bias.grad = torch.tensor([1.0, -2.0])
        reference = torch.cat([p.grad.flatten() for p in model.parameters()]).double()
        with tempfile.TemporaryDirectory() as temporary:
            single, ddp = Path(temporary) / "single", Path(temporary) / "ddp"
            inspect_gradients(model, 2, save_to=single)
            for parameter in model.parameters():
                parameter.grad.mul_(1.01)
            captured = inspect_gradients(model, 3, save_to=ddp, compare_to=single)
            self.assertAlmostEqual(captured["comparison"]["relative_l2"], 0.01, places=6)
            ddp_vector = torch.cat([p.grad.flatten() for p in model.parameters()]).double()
            model.weight.grad[0, 1].mul_(-1)
            actual = torch.cat([p.grad.flatten() for p in model.parameters()]).double()
            compared = inspect_gradients(model, 2, compare_to=single, compare_to_ddp=ddp)
            for key, expected in (("comparison", reference), ("ddp_comparison", ddp_vector)):
                self.assertAlmostEqual(compared[key]["relative_l2"],
                                       float(torch.linalg.vector_norm(actual - expected)
                                             / torch.linalg.vector_norm(expected)))
            with (ddp / "gradients.bin").open("r+b") as stream:
                stream.write(torch.tensor([100.0]).numpy().tobytes())
            with self.assertRaisesRegex(ValueError, "checksum"):
                inspect_gradients(model, 2, compare_to=single, compare_to_ddp=ddp)

    def test_criteria_are_explicit_finite_and_bounded(self):
        for changes in (dict(loss_rtol=float("nan")), dict(steps=0), dict(steps=True),
                        dict(gradient_relative_l2_max=0), dict(chunk_elements=2**30)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                NumericalCriteria(**changes)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "criteria.json"
            path.write_text(json.dumps(asdict(NumericalCriteria())))
            self.assertEqual(NumericalCriteria.load(path), NumericalCriteria())
            path.write_text('{"steps":3}')
            with self.assertRaises(ValueError):
                NumericalCriteria.load(path)

    def test_gates_reject_loss_gradient_rank_input_and_frozen_mismatches(self):
        reference, distributed = [], []
        for step in range(1, 4):
            common = dict(parameters="same", frozen_unchanged=True, frozen_no_grad=True,
                          learning_rates=[step * 1e-7])
            reference.append(dict(step=step, loss=1.0, targets=2, ranks=[dict(rank=0, inputs=[dict(index=0), dict(index=1)],
                                  draws=[dict(target=0), dict(target=1)], **common)]))
            gradient = dict(sha256="same", comparison=dict(relative_l2=0.01), blocks={})
            distributed.append(dict(step=step, loss=1.0001, targets=2, ranks=[
                dict(rank=i, inputs=[dict(index=i)], draws=[dict(target=i)],
                     gradients=deepcopy(gradient), **common) for i in range(2)]))
        self.assertEqual(compare_records(reference, distributed, NumericalCriteria())["status"], "passed")
        changed = deepcopy(distributed)
        changed[0]["ranks"][0]["gradients"]["comparison"]["relative_l2"] = 0.1
        changed[0]["ranks"][1]["gradients"]["sha256"] = "different"
        changed[1]["ranks"][1].update(parameters="different", frozen_no_grad=False, draws=[dict(target=0)])
        changed[2]["loss"] = 1.1
        result = compare_records(reference, changed, NumericalCriteria())
        self.assertEqual(set(result["failures"]), {"step_1/gradient_l2", "step_1/rank_gradients",
                         "step_2/rank_parameters", "step_2/frozen", "step_2/draws", "step_3/loss"})
        with self.assertRaises(ValueError):
            compare_records(reference[:2], distributed, NumericalCriteria())

        sharded = deepcopy(distributed)
        for row in sharded:
            row["ranks"][1]["parameters"] = "a different shard"
        sharded[0]["ranks"][0]["gradients"].update(ddp_comparison=dict(relative_l2=0.2), ddp_blocks={})
        self.assertEqual(compare_records(reference, sharded, NumericalCriteria(), strategy="fsdp2")["status"], "passed")
        direct = compare_records(distributed, sharded, NumericalCriteria(),
                                 strategy="fsdp2", reference_strategy="ddp")
        self.assertEqual(direct["failures"], ["step_1/gradient_l2"])

        permuted = deepcopy(distributed)
        for row in permuted:
            row["ranks"][0]["inputs"], row["ranks"][1]["inputs"] = row["ranks"][1]["inputs"], row["ranks"][0]["inputs"]
            row["ranks"][0]["draws"], row["ranks"][1]["draws"] = row["ranks"][1]["draws"], row["ranks"][0]["draws"]
        self.assertEqual(compare_records(reference, permuted, NumericalCriteria())["status"], "passed")
        permuted[0]["ranks"][0]["inputs"] = [dict(index=0)]
        self.assertEqual(compare_records(reference, permuted, NumericalCriteria())["failures"], ["step_1/inputs"])


if __name__ == "__main__":
    unittest.main()
