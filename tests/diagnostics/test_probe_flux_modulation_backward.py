"""Verify the bounded intervention matrix and reject incomplete causal evidence."""

import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.diagnostics import probe_flux_modulation_backward as probe

CONFIG = probe.ROOT / "configs/diagnostics/flux_modulation_backward.json"


def complete_result(policy):
    count = 3875544576
    names = [*(f"{name}.weight" for name in probe.MODULATIONS), *(f"other.{i}.weight" for i in range(166))]
    parameters = {name: probe.execution.metrics.metrics(dict(error_squared=1., reference_squared=2.,
        actual_squared=3., dot=2., count=count // 169 + (count % 169 if i == 0 else 0), max_abs=.1))
        for i, name in enumerate(names)}
    gradients = dict(parameters=parameters, total=probe.execution.metrics.aggregate(parameters.values()))
    gradients["total"]["norm_ratio"] = (gradients["total"]["actual_squared"] / gradients["total"]["reference_squared"]) ** .5
    def identity(shape):
        return dict(shape=shape, dtype="torch.bfloat16", sha256="a" * 64)
    forward = dict(velocity={label: identity([900, 128]) for label in "AB"},
        activations={stage: {label: identity([length if stage == "text.mlp.linear_out" else length + 900, 3072])
                            for label, length in (("A", 468), ("B", 406))}
                     for stage in (*probe.execution.BLOCKS, "text.mlp.linear_out")})
    cases = []
    for spec in probe.case_plan():
        n = 3 if spec["modulation_reduction"] == "fp32" else 0
        cases.append({**spec, "forward_identity": copy.deepcopy(forward), "forward_unchanged": True,
            "sample_weights": {"A": 1 / 48, "B": 1 / 48}, "losses": {"A": .1, "B": .2},
            "dispatch_verified": True, "parameter_count": count, "gradient_tensors": 169,
            "gradient_norm": 1., "padding_checks": [],
            "operators": {"flash_attn_3::_flash_attn_forward": 25 * spec["backward_calls"],
                          "flash_attn_3::_flash_attn_backward": 25 * spec["backward_calls"]},
            "modulation_reduction_evidence": dict(forward_calls=n, backward_calls=n,
                source_dtypes=["torch.bfloat16"] * n, reduction_dtype="torch.float32" if n else None),
            "comparisons": {name: dict(gradients=copy.deepcopy(gradients), modulation=probe.modulation_metrics(gradients))
                            for name in spec["references"]}})
    return dict(status="complete", plan=probe.build_plan(policy), checkpoints=dict.fromkeys(("flux", "nvarc", "vae"), {}),
                weights_unchanged=True, inputs_unchanged=True, max_gradient_references_resident=2, cases=cases)


class ModulationProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = probe.load_probe_config(CONFIG, require_checkpoints=False)

    def test_offline_plan_covers_repeats_reversal_and_at_most_two_full_gradient_snapshots(self):
        with patch.object(probe, "run_probe", side_effect=AssertionError("execution")), \
                patch.object(probe, "worker", side_effect=AssertionError("worker")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(probe.main(["--plan", "--config", str(CONFIG)]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual((plan["logical_batches"], plan["full_model_backward_calls"]), (8, 10))
        self.assertEqual((plan["max_worker_seconds"], plan["max_function_seconds"], plan["max_lifecycle_seconds"]), (420, 480, 600))
        self.assertEqual((plan["optimizer_updates"], plan["kernel_replay_evaluations"], plan["retries"]), (0, 0, 0))
        self.assertFalse(plan["model_downloads"])
        self.assertLess(plan["estimated_compute_ceiling_usd"], 1.)
        cache, peak = set(), 0
        for index, spec in enumerate(plan["cases"]):
            self.assertTrue(set(spec["references"]).issubset(cache))
            future = probe.execution.retained_after(plan["cases"], index)
            cache.intersection_update(future)
            if spec["name"] in future:
                cache.add(spec["name"])
            peak = max(peak, len(cache))
        self.assertEqual(peak, 2)
        self.assertEqual(cache, set())
        for row in plan["cases"]:
            if row["name"] != "single/0":
                self.assertIn("single/0", row["references"])
        self.assertEqual(plan["cases"][6]["labels"], ["B", "A"])
        self.assertIn("packed_fp32_ab/0", plan["cases"][6]["references"])

    def test_config_rejects_changed_precision_backend_repetitions_and_budget(self):
        raw = json.loads(CONFIG.read_text())
        for update in (dict(backend="cudnn"), dict(modulation_reduction="native"), dict(repetitions=1),
                       dict(max_duration_seconds=421), dict(max_duration_seconds=True), dict(schema_version=True),
                       dict(unplanned=True)):
            with self.subTest(update=update), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps({**raw, "backward_fixture": str(self.policy.config_paths[1]), **update}))
                with self.assertRaises(ValueError):
                    probe.load_probe_config(path, require_checkpoints=False)

    def test_evidence_rejects_forward_changes_inactive_intervention_and_incomplete_gradients(self):
        good = complete_result(self.policy)
        self.assertTrue(probe.validate_complete(good, self.policy))
        mutations = [lambda r: r["cases"].pop(), lambda r: r.update(weights_unchanged=False),
            lambda r: r.update(max_gradient_references_resident=3),
            lambda r: r["cases"][4].update(forward_unchanged=False),
            lambda r: r["cases"][4]["forward_identity"]["velocity"]["B"].update(sha256="b" * 64),
            lambda r: r["cases"][4]["forward_identity"]["activations"].pop("double.4"),
            lambda r: r["cases"][4]["forward_identity"]["activations"]["double.4"]["B"].update(shape=[1368, 3072]),
            lambda r: r["cases"][4]["losses"].update(B=.21),
            lambda r: r["cases"][4]["modulation_reduction_evidence"].update(backward_calls=0),
            lambda r: r["cases"][4]["modulation_reduction_evidence"].update(source_dtypes=["torch.float32"] * 3),
            lambda r: r["cases"][4]["operators"].update({"flash_attn_3::_flash_attn_backward": 24}),
            lambda r: r["cases"][4]["sample_weights"].update(B=1 / 24),
            lambda r: r["cases"][4]["comparisons"]["single/0"]["gradients"]["parameters"].pop("other.0.weight"),
            lambda r: r["cases"][4]["comparisons"]["single/0"]["gradients"]["total"].update(error_squared=999),
            lambda r: r["cases"][4]["comparisons"]["single/0"]["modulation"]["all_modulation"].update(error_squared=999)]
        for mutate in mutations:
            broken = copy.deepcopy(good)
            mutate(broken)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                probe.validate_complete(broken, self.policy)

    def test_timeout_preserves_partial_evidence_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "results"
            def launch(command, **options):
                self.assertIn("--worker", command)
                self.assertEqual(options["timeout_seconds"], 420)
                probe.worker_tools._write_json(destination / "result.json", dict(status="running", cases=[],
                    phase="backward/packed_fp32_ab/0", worker_elapsed_seconds=420))
                return -9, True
            with patch.object(probe, "load_probe_config", return_value=self.policy), \
                    patch.object(probe, "source_metadata", return_value={}), \
                    patch.object(probe, "config_hashes", return_value={}), \
                    patch.object(probe, "dataset_hashes", return_value={}), \
                    patch.object(probe.worker_tools, "launch_worker", side_effect=launch) as worker:
                result = probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination))
            worker.assert_called_once()
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(result["phase"], "backward/packed_fp32_ab/0")
            self.assertEqual(json.loads((destination / "manifest.json").read_text())["status"], "timeout")
            self.assertIn("420 seconds", (destination / "report.md").read_text())

    def test_manifest_mismatch_stops_before_submission_or_directory_creation(self):
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
