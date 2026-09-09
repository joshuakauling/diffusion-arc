"""Exercise the offline condition plan and bounded diagnostic supervisor."""

import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from scripts.diagnostics import probe_flux_backend_conditions as probe


CONFIG = probe.ROOT / "configs/diagnostics/flux_backend_conditions.json"


class ConditionsProbeTests(unittest.TestCase):
    def setUp(self):
        self.policy = probe.load_probe_config(CONFIG, require_checkpoints=False)

    def write_config(self, folder, change=None):
        raw = copy.deepcopy(self.policy.raw)
        raw["models"] = {name: str(path) for name, path in self.policy.models.items()}
        raw["inputs_fixture"] = str(self.policy.fixture_paths[0])
        if change is not None:
            change(raw)
        path = Path(folder) / "probe.json"
        path.write_text(json.dumps(raw, sort_keys=True))
        return path

    def test_sorted_config_preserves_named_checkpoints_and_complete_offline_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = probe.load_probe_config(self.write_config(directory), require_checkpoints=False)
        self.assertEqual(policy.models, self.policy.models)
        with patch.object(probe, "run_probe", side_effect=AssertionError("unexpected execution")), \
             patch("sys.argv", ["probe", "--plan", "--config", str(CONFIG)]), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(probe.main(), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(len(plan["conditions"]), 13)
        self.assertEqual(plan["full_model_forwards"], 104)
        self.assertEqual(plan["kernel_replays"], 52)
        self.assertEqual(plan["kernel_forward_backward_pairs"], 312)
        self.assertEqual(plan["model_backward_calls"], 0)
        self.assertEqual(plan["optimizer_updates"], 0)
        self.assertEqual(plan["conditions"][0], {"checkpoint": "base", "input": "native_prompt_t1"})
        self.assertEqual({row["input"] for row in plan["conditions"] if row["checkpoint"] == "overfit"}, {
            f"arc_{category}_t{timestep}" for category in ("seen", "unseen", "other")
            for timestep in ("1", "0.5")})

    def test_configuration_rejects_changed_scope_limits_and_checkpoint_lineage(self):
        mutations = {
            "budget": lambda raw: raw.update(max_duration_seconds=1801),
            "bool_budget": lambda raw: raw.update(max_duration_seconds=True),
            "seed": lambda raw: raw.update(seed=True),
            "timestep": lambda raw: raw.update(timesteps=[1.0, 0.25]),
            "backend": lambda raw: raw["backends"].pop(),
            "capture": lambda raw: raw["capture_blocks"].pop(),
            "threshold": lambda raw: raw["thresholds"].update(kernel_gradient_relative_l2=float("nan")),
            "duplicate_checkpoint": lambda raw: raw["models"].update(nvarc_checkpoint=raw["models"]["base_checkpoint"]),
            "overfit_lineage": lambda raw: raw["models"].update(overfit_checkpoint="/tmp/unrelated-checkpoint"),
            "unknown_field": lambda raw: raw.update(extra=True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = self.write_config(directory, mutate)
                with self.assertRaises(ValueError):
                    probe.load_probe_config(path, require_checkpoints=False)

    def test_checkpoint_manifest_checks_actual_weights_and_rejects_external_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            weights = root / "model-00001-of-00001.safetensors"
            weights.write_bytes(b"checkpoint payload")
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"weight": weights.name}}))
            evidence = probe.checkpoint_evidence(root)
            self.assertEqual(set(evidence["files"]), {"config.json", index.name, weights.name})
            self.assertFalse(evidence["manifest_verified"])
            self.assertTrue(probe.checkpoint_evidence(root, expected=evidence["files"])["manifest_verified"])
            weights.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "differ"):
                probe.checkpoint_evidence(root, expected=evidence["files"])
            index.write_text(json.dumps({"weight_map": {"weight": "../other.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "nonlocal"):
                probe.checkpoint_evidence(root)

    def complete_result(self):
        metric = probe.replay.tensor_metrics(torch.ones(2), torch.ones(2))
        model_comparisons = {
            "bfl_fp32": (), "bfl_bf16": ("bfl_fp32",), "diffusers_fp32": ("bfl_fp32",),
            "bf16_math": ("diffusers_fp32", "bfl_bf16"),
            **{name: ("diffusers_fp32", "bf16_math") for name in ("cudnn", "fa2", "fa3", "fa4")},
        }
        measured = dict(velocity=metric, maximum_observed_relative_l2=0,
                        stages={name: metric for name in (
                            "velocity/image", *(f"{block}/{stream}" for block in probe.BLOCKS
                                               for stream in ("text", "image")))})
        conditions = []
        for item in probe.build_plan(self.policy)["conditions"]:
            identity = {"images": {"sha256": item["input"]}}
            condition = dict(**item, models={}, kernels={})
            for variant, references in model_comparisons.items():
                condition["models"][variant] = dict(attention_calls=25, input_fingerprints=identity,
                    comparisons={reference: copy.deepcopy(measured) for reference in references})
            for block in probe.BLOCKS:
                kernel = dict(input_hashes_unchanged=True, inputs={"q": "bf16-input"},
                              fp32_inputs={"q": "fp32-promoted-input"},
                              upstream_gradient="bf16-upstream", fp32_upstream_gradient="fp32-promoted-upstream",
                              backends={})
                for backend in probe.replay.BACKENDS:
                    promoted = backend == "fp32_math"
                    kernel["backends"][backend] = dict(
                        **{name: copy.deepcopy(metric) for name in ("output", "dQ", "dK", "dV")},
                        comparisons_vs_bf16_math={name: copy.deepcopy(metric) for name in ("output", "dQ", "dK", "dV")},
                        input_fingerprints=kernel["fp32_inputs" if promoted else "inputs"],
                        upstream_fingerprint=kernel["fp32_upstream_gradient" if promoted else "upstream_gradient"],
                        profiled=not conditions and block == probe.BLOCKS[0],
                        backend_verified=not conditions and block == probe.BLOCKS[0])
                condition["kernels"][block] = kernel
            conditions.append(condition)
        return dict(status="complete", conditions=conditions)

    def test_complete_evidence_survives_sorted_json_roundtrip(self):
        restored = json.loads(json.dumps(self.complete_result(), sort_keys=True, allow_nan=False))
        self.assertTrue(probe.validate_complete(restored, self.policy))

    def test_validator_rejects_incomplete_metrics_identity_and_dispatch_evidence(self):
        for damage in ("condition", "model", "boundary", "reference", "model_input", "checkpoint_input",
                       "kernel_backend", "kernel_input", "upstream", "gradient", "nonfinite", "profile"):
            result = copy.deepcopy(self.complete_result())
            condition = result["conditions"][0]
            model = condition["models"]["fa3"]
            kernel = condition["kernels"]["double.0"]
            if damage == "condition":
                result["conditions"].pop()
            elif damage == "model":
                condition["models"].pop("fa4")
            elif damage == "boundary":
                model["comparisons"]["bf16_math"]["stages"].pop("single.19/text")
            elif damage == "reference":
                model["comparisons"].pop("bf16_math")
            elif damage == "model_input":
                model["input_fingerprints"] = {"images": {"sha256": "different"}}
            elif damage == "checkpoint_input":
                other = next(row for row in result["conditions"] if row["checkpoint"] == "overfit")
                for row in other["models"].values():
                    row["input_fingerprints"] = {"images": {"sha256": "different"}}
            elif damage == "kernel_backend":
                kernel["backends"].pop("fa3")
            elif damage == "kernel_input":
                kernel["backends"]["fa3"]["input_fingerprints"] = {"q": "different"}
            elif damage == "upstream":
                kernel["backends"]["fa3"]["upstream_fingerprint"] = "different"
            elif damage == "gradient":
                kernel["backends"]["fa3"]["dQ"]["count"] = 0
            elif damage == "nonfinite":
                kernel["backends"]["fa3"]["dK"]["rms"] = float("nan")
            else:
                kernel["backends"]["fa3"]["backend_verified"] = False
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                probe.validate_complete(result, self.policy)

    def test_numerical_screening_is_independent_of_execution_success(self):
        result = self.complete_result()
        result["conditions"][0]["kernels"]["double.0"]["backends"]["fa3"]["dQ"]["relative_l2"] = 0.5
        self.assertTrue(probe.validate_complete(result, self.policy))
        screening = probe.numerical_screening(result, self.policy)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(screening["exceeded_count"], 1)
        self.assertEqual(screening["exceeded"][0]["metric"], "dQ")
        self.assertEqual(screening["exceeded"][0]["threshold"], 0.02)

    def test_supervisor_preserves_partial_evidence_and_demotes_false_success(self):
        for mode in ("timeout", "nonzero", "incomplete", "exception", "missing"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "results"
                progress = Mock()

                def worker(command, **kwargs):
                    self.assertEqual(kwargs["timeout_seconds"], 1800)
                    self.assertEqual(kwargs["progress_marker"], probe.PROGRESS_MARKER)
                    self.assertIn("--worker", command)
                    self.assertIs(kwargs["on_progress"], progress)
                    if mode == "exception":
                        raise RuntimeError("worker launch failed")
                    if mode != "missing":
                        probe.worker_tools._write_json(destination / "result.json", {
                            "status": "complete", "phase": "kernel/base/arc_seen_t1/double.0",
                            "conditions": [{"checkpoint": "base", "input": "arc_seen_t1",
                                            "models": {}, "kernels": {}}],
                        })
                    return (-9 if mode == "timeout" else 1 if mode == "nonzero" else 0), mode == "timeout"

                with patch.object(probe, "load_probe_config", return_value=self.policy), \
                     patch.object(probe, "source_metadata", return_value={"source_sha256": {}}), \
                     patch.object(probe, "config_hashes", return_value={}), \
                     patch.object(probe, "dataset_hashes", return_value={}), \
                     patch.object(probe.worker_tools, "launch_worker", side_effect=worker):
                    result = probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination),
                                             metadata={"source_sha256": {}}, on_progress=progress)
                self.assertEqual(result["status"], "timeout" if mode == "timeout" else "error")
                if mode not in ("exception", "missing"):
                    self.assertEqual(result["conditions"][0]["input"], "arc_seen_t1")
                self.assertIn("error", result)
                self.assertTrue((destination / "report.md").is_file())
                self.assertEqual(json.loads((destination / "manifest.json").read_text())["status"], result["status"])
                self.assertEqual(json.loads((destination / "summary.json").read_text())["status"], result["status"])
                progress.assert_called_once()

    def test_changed_source_manifest_fails_before_worker_or_artifact_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "results"
            with patch.object(probe, "load_probe_config", return_value=self.policy), \
                 patch.object(probe, "source_metadata", return_value={"source_sha256": {"file": "current"}}), \
                 patch.object(probe, "config_hashes", return_value={}), \
                 patch.object(probe, "dataset_hashes", return_value={}), \
                 patch.object(probe.worker_tools, "launch_worker") as launch, \
                 self.assertRaisesRegex(ValueError, "manifest"):
                probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination),
                                metadata={"source_sha256": {"file": "previous"}})
            launch.assert_not_called()
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
