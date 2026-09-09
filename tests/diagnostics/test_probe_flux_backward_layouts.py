"""Validate the approved matrix and fail-closed handling of partial GPU evidence."""

import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.diagnostics import probe_flux_backward_layouts as probe

CONFIG = probe.ROOT / "configs/diagnostics/flux_backward_layouts.json"


def complete_result(policy):
    """Minimal internally consistent evidence for testing missing-field rejection."""
    parameter_count = 3875544576
    counts = [parameter_count // 169] * 169
    counts[0] += parameter_count % 169
    parameters = {f"parameter.{index}": probe.execution.metrics.metrics(dict(
        error_squared=1., reference_squared=2., actual_squared=3., dot=2., count=count, max_abs=.1))
        for index, count in enumerate(counts)}
    total = probe.execution.metrics.aggregate(parameters.values())
    total["norm_ratio"] = (total["actual_squared"] / total["reference_squared"]) ** .5
    boundaries = {stage: {field: {label: dict(sequence_length=length if stage == "text.mlp.linear_out" else length + 900,
        text_length=length, edge_width=32) for label, length in (("A", 468), ("B", 406))}
        for field in ("output", "gradient")} for stage in (*probe.execution.BLOCKS, "text.mlp.linear_out")}
    cases = []
    for spec in probe.execution.case_plan():
        events = 25 * (2 if "packed" in spec["variant"] and spec["variant"].startswith("cudnn") else spec["backward_calls"])
        operators = ({"aten::_scaled_dot_product_cudnn_attention": events,
                      "aten::_scaled_dot_product_cudnn_attention_backward": events}
                     if spec["variant"].startswith("cudnn") else {
                      "flash_attn_3::_flash_attn_forward": events, "flash_attn_3::_flash_attn_backward": events})
        cases.append({**spec, "sample_weights": {"A": 1 / 48, "B": 1 / 48}, "dispatch_verified": True,
            "operators": operators, "parameter_count": parameter_count, "gradient_tensors": 169,
            "gradient_norm": 1., "padding_checks": [],
            "comparisons": {name: dict(gradients=dict(total=total, parameters=parameters), boundaries=boundaries)
                            for name in spec["references"]}})
    stages = {}
    for stage in probe.execution.BLOCKS:
        backends = {}
        for backend in ("cudnn", "fa3"):
            layouts = ["singletons", "ab", "ba"] + (["padded"] if backend == "cudnn" else [])
            comparisons = ["ab_vs_singletons", "ba_vs_singletons", "ba_vs_ab"] + (["padded_vs_singletons"] if backend == "cudnn" else [])
            executions = []
            for repeat in range(2):
                dispatch = {layout: {"profiled": False} for layout in layouts}
                if stage == "double.0" and repeat == 0:
                    for layout in (["singletons", "padded"] if backend == "cudnn" else ["ab"]):
                        dispatch[layout] = dict(profiled=True, backend_verified=True, cuda_kernel_names=["actual-kernel"])
                executions.append(dict(repetition=repeat, comparisons=dict.fromkeys(comparisons, {}), dispatch=dispatch))
            backends[backend] = dict(executions=executions, repeat_comparisons=dict.fromkeys(layouts, {}))
        stages[stage] = dict(backends=backends)
    return dict(status="complete", plan=probe.build_plan(policy),
        checkpoints=dict.fromkeys(("flux", "nvarc", "vae"), {}), weights_unchanged=True,
        inputs_unchanged=True, max_gradient_references_resident=2, cases=cases,
        kernel_boundaries=dict(status="complete", evaluations_completed=72, evaluations_attempted=72, stages=stages))


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = probe.load_probe_config(CONFIG, require_checkpoints=False)

    def test_plan_is_offline_and_has_fixed_backward_budget(self):
        with patch.object(probe, "run_probe", side_effect=AssertionError("execution")), \
                patch.object(probe, "worker", side_effect=AssertionError("worker")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(probe.main(["--plan", "--config", str(CONFIG)]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual((plan["full_model_backward_calls"], plan["logical_batches"]), (22, 16))
        self.assertEqual(plan["gradient_references_resident_max"], 2)
        self.assertEqual((plan["max_worker_seconds"], plan["max_function_seconds"], plan["max_lifecycle_seconds"]),
                         (840, 960, 1200))
        self.assertEqual(plan["optimizer_updates"], 0)
        self.assertFalse(plan["model_downloads"])
        self.assertEqual(plan["retries"], 0)

    def test_config_rejects_changed_scope_and_types_before_model_loading(self):
        variants = [dict(max_duration_seconds=841), dict(max_duration_seconds=True), dict(repetitions=1),
            dict(global_batch_size=2), dict(padded_length=468), dict(text_lengths=[468., 406.]),
            dict(selected_labels=["B", "A"]), dict(capture_blocks=["double.0"]), dict(unplanned=True)]
        for update in variants:
            with self.subTest(update=update), tempfile.TemporaryDirectory() as directory:
                raw = {**self.policy.raw, "packing_fixture": str(self.policy.fixture), **update}
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    probe.load_probe_config(path, require_checkpoints=False)
        for raw in (None, [], 1, "invalid"):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    probe.load_probe_config(path, require_checkpoints=False)

    def test_complete_evidence_rejects_missing_gradients_dispatch_boundaries_and_replays(self):
        good = complete_result(self.policy)
        self.assertTrue(probe.validate_complete(good, self.policy))
        mutations = [lambda r: r["cases"].pop(), lambda r: r.update(weights_unchanged=False),
            lambda r: r.update(max_gradient_references_resident=3),
            lambda r: r["cases"][0]["sample_weights"].update(B=1 / 24),
            lambda r: r["cases"][0].update(dispatch_verified=False),
            lambda r: r["cases"][0]["operators"].update({"aten::_scaled_dot_product_cudnn_attention": 49}),
            lambda r: r["cases"][1]["comparisons"]["cudnn_single/0"]["gradients"]["parameters"].pop("parameter.0"),
            lambda r: r["cases"][1]["comparisons"]["cudnn_single/0"]["gradients"]["total"].update(error_squared=999),
            lambda r: r["cases"][1]["comparisons"]["cudnn_single/0"]["boundaries"]["double.0"]["gradient"].pop("B"),
            lambda r: r["cases"][1]["comparisons"]["cudnn_single/0"]["boundaries"]["double.0"]["gradient"]["B"].update(text_length=468),
            lambda r: r["cases"][0]["padding_checks"].append(dict(nonzero_gradient_elements=1)),
            lambda r: r["kernel_boundaries"].update(evaluations_completed=71),
            lambda r: r["kernel_boundaries"]["stages"].pop("double.4"),
            lambda r: r["kernel_boundaries"]["stages"]["double.0"]["backends"].pop("fa3"),
            lambda r: r["kernel_boundaries"]["stages"]["double.0"]["backends"]["fa3"]["executions"][0]["dispatch"]["ab"].update(cuda_kernel_names=[])]
        for mutate in mutations:
            broken = copy.deepcopy(good)
            mutate(broken)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                probe.validate_complete(broken, self.policy)

    def test_timeout_saves_partial_evidence_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "results"

            def launch(command, **options):
                self.assertIn("--worker", command)
                self.assertEqual(options["timeout_seconds"], 840)
                self.assertEqual(options["progress_marker"], probe.PROGRESS_MARKER)
                probe.worker_tools._write_json(destination / "result.json", dict(status="running", cases=[],
                    phase="backward/cudnn_padded_batch/0", worker_elapsed_seconds=840))
                return -9, True

            with patch.object(probe, "load_probe_config", return_value=self.policy), \
                    patch.object(probe, "source_metadata", return_value={}), \
                    patch.object(probe, "config_hashes", return_value={}), \
                    patch.object(probe, "dataset_hashes", return_value={}), \
                    patch.object(probe.worker_tools, "launch_worker", side_effect=launch) as worker:
                result = probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination))
            worker.assert_called_once()
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(result["phase"], "backward/cudnn_padded_batch/0")
            self.assertTrue(result["timed_out"])
            self.assertEqual(json.loads((destination / "manifest.json").read_text())["status"], "timeout")
            self.assertEqual(json.loads((destination / "summary.json").read_text())["worker_returncode"], -9)
            self.assertIn("840 seconds", (destination / "report.md").read_text())

    def test_source_mismatch_stops_before_directory_creation_or_worker_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "results"
            with patch.object(probe, "load_probe_config", return_value=self.policy), \
                    patch.object(probe, "source_metadata", return_value={"source_sha256": {"code": "correct"}}), \
                    patch.object(probe, "config_hashes", return_value={}), \
                    patch.object(probe, "dataset_hashes", return_value={}), \
                    patch.object(probe.worker_tools, "launch_worker") as worker:
                with self.assertRaisesRegex(ValueError, "differs from launch manifest"):
                    probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination),
                                    metadata={"source_sha256": {"code": "different"}})
            worker.assert_not_called()
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
