"""Exercise the reference chain and reject incomplete or misleading evidence."""

import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.diagnostics import probe_flux_bfl as probe
from scripts.diagnostics import _flux_bfl_forward as forward
from scripts.diagnostics._flux_bfl_weights import build_bfl_model
from tests.diagnostics.test_flux_packing_forward import fixture

CONFIG = probe.REPOSITORY_ROOT / "configs/diagnostics/flux_bfl.json"


class BflProbeTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(22)
        self.policy, self.training = probe.load_probe_config(CONFIG, require_checkpoints=False)

    def result(self):
        model, source = fixture(double=1, single=1, head_dim=8)
        source["image_positions"][1] = source["image_positions"][0]
        data = forward.prepare_comparison_inputs(source)
        bfl, _ = build_bfl_model(model)
        policy = replace(self.policy, input_policy=replace(self.policy.input_policy,
            groups=({**self.policy.input_policy.groups[0], "padded_length": 7},)))
        cases = probe.run_comparisons(bfl, model, data, policy)
        plan = {**probe.build_plan(self.policy, self.training), "stages": forward.stage_order(1, 1)}
        return dict(status="complete", phase="complete", cases=cases, plan=plan)

    def test_ten_real_forward_calls_reuse_tensors_across_implementations_and_precisions(self):
        with patch("torch.autograd.backward", side_effect=AssertionError("unexpected backward")):
            result = self.result()
        probe.validate_complete(result)
        rows = {row["variant"]: row for row in result["cases"]}
        for precision in ("fp32", "bf16"):
            for label in ("a", "b"):
                self.assertEqual(rows[f"{precision}/bfl_{label}"]["input_identity"],
                                 rows[f"{precision}/diffusers_{label}"]["input_identity"])
        for variant in probe.VARIANTS:
            self.assertEqual(rows[f"fp32/{variant}"]["input_identity"],
                             rows[f"bf16/{variant}"]["input_identity"])
        for row in result["cases"][:5]:
            for samples in row["comparison_summary"].values():
                self.assertTrue(all(sample["passed"] for sample in samples.values()))
        json.dumps(result, allow_nan=False)

    def test_complete_validation_rejects_missing_or_falsified_comparison_evidence(self):
        valid = self.result()
        for damage in ("summary_label", "stream", "precision", "passed", "case"):
            result = copy.deepcopy(valid)
            row = result["cases"][2]
            if damage == "summary_label":
                row["comparison_summary"]["bfl_singletons"].clear()
            elif damage == "stream":
                row["comparisons"]["bfl_singletons"]["A"]["double.0"].pop("text")
            elif damage == "precision":
                row["precision"] = "bf16"
            elif damage == "passed":
                row["comparison_summary"]["bfl_singletons"]["A"]["passed"] = False
            else:
                result["cases"].pop()
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                probe.validate_complete(result)

    def test_call_tensor_mutation_fails_before_it_can_contaminate_next_reference(self):
        model, source = fixture(double=1, single=1, head_dim=8)
        source["image_positions"][1] = source["image_positions"][0]
        data = forward.prepare_comparison_inputs(source)
        bfl, _ = build_bfl_model(model)
        real = forward.run_case
        def mutate(model, implementation, inputs, *args, **kwargs):
            result = real(model, implementation, inputs, *args, **kwargs)
            inputs["text"].add_(1)
            return result
        with patch.object(forward, "run_case", side_effect=mutate), self.assertRaisesRegex(ValueError, "mutated"):
            probe.run_comparisons(bfl, model, data, self.policy)

    def test_plan_is_offline_and_preserves_recorded_exposures(self):
        with patch.object(probe, "run_probe", side_effect=AssertionError("unexpected execution")), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(probe.main(["--plan", "--config", str(CONFIG)]), 0)
        plan = json.loads(output.getvalue())["plan"]
        self.assertEqual(plan["target_indices"], [15, 12])
        self.assertEqual(plan["text_lengths"], [468, 406])
        self.assertEqual(plan["batch_padded_text_length"], 487)
        self.assertEqual(len(plan["cases"]), 10)
        self.assertEqual(len(plan["stages"]), 30)
        self.assertEqual(set(probe.config_hashes(CONFIG, self.policy)), {"probe", "inputs", "training"})

    def test_configuration_rejects_unsupported_precision_and_invalid_limits(self):
        original = json.loads(CONFIG.read_text())
        for key, value in (("max_duration_seconds", 481), ("max_duration_seconds", True),
                           ("sdpa_backend", "auto"), ("precisions", ["bf16", "fp32"]),
                           ("numerical_tolerances", {"fp32": {}, "bf16": {}})):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.json"
                path.write_text(json.dumps({**original, key: value}))
                with self.assertRaises(ValueError):
                    probe.load_probe_config(path, require_checkpoints=False)

    def test_supervisor_preserves_partial_cases_and_demotes_timeout_success(self):
        for timed_out, returncode in ((True, -9), (False, 1)):
            with self.subTest(timed_out=timed_out), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "result"
                def worker(command, **kwargs):
                    self.assertEqual(kwargs["timeout_seconds"], 480)
                    (destination / "result.json").write_text(json.dumps(dict(
                        status="complete", phase="bf16/diffusers_a", numerical_passed=True,
                        cases=[{"variant": "partial", "comparison_summary": {}}])))
                    return returncode, timed_out
                with patch.object(probe, "load_probe_config", return_value=(self.policy, self.training)), \
                     patch.object(probe, "source_hashes", return_value={}), \
                     patch.object(probe, "dataset_hashes", return_value={}), \
                     patch.object(probe.inputs_probe.shapes, "launch_worker", side_effect=worker):
                    result = probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination),
                                             metadata={"source_sha256": {}})
                self.assertEqual(result["status"], "timeout" if timed_out else "error")
                self.assertNotIn("numerical_passed", result)
                self.assertEqual(result["cases"][0]["variant"], "partial")
                self.assertIn("not established", (destination / "report.md").read_text())
                self.assertTrue((destination / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
