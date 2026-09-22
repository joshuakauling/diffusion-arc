"""Paired RunPod checkpoint experiments without remote compute or model downloads."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.cloud import diagnose_runpod as launcher
from scripts.cloud import launch_runpod as transport


CONFIG = launcher.ROOT / "configs/diagnostics/sampling_checkpoints_4530_4830.json"


class CheckpointComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.config = launcher.load_config(CONFIG)

    def test_comparison_dry_run_keeps_fixed_inputs_and_uses_existing_proxy_transport(self):
        stdout = io.StringIO()
        with patch.object(transport, "remote_python", side_effect=AssertionError("SSH")), \
                patch.object(launcher, "transformer_metadata", side_effect=AssertionError("Hub")), \
                redirect_stdout(stdout):
            code = launcher.main(["--ssh", "ssh test@ssh.runpod.io -i ~/.ssh/id_rsa",
                                  "--config", str(CONFIG), "--run-name", "compare", "--dry-run"])
        plan = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(plan["diagnostic"], self.config)
        self.assertEqual(self.config["generation_steps"], [50, 4])
        self.assertEqual(self.config["sampling_seeds"], [20167, 20168])
        self.assertEqual((self.config["task_pool"], self.config["task_count"]), ("validation", 100))
        self.assertEqual(list(self.config["checkpoints"]), ["step-004530", "step-004830"])
        self.assertEqual(plan["arguments"][0], "diagnose")

    def test_source_check_never_downloads_or_starts_a_worker(self):
        with patch.object(launcher, "transformer_metadata", return_value=("c" * 40, [1], 100)) as metadata, \
                patch.object(launcher, "supervise", side_effect=AssertionError("worker")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.main(["--config", str(CONFIG), "--check-sources"]), 0)
        self.assertEqual(metadata.call_count, 2)

    def test_ambiguous_comparisons_fail_before_ssh(self):
        specs = self.config["checkpoints"]
        invalid = [{"checkpoint": specs["step-004530"]},
                   {"checkpoints": {"../outside": specs["step-004530"], "valid": specs["step-004830"]}},
                   {"checkpoints": {"one": specs["step-004530"], "two": specs["step-004530"]}},
                   {"task_pool": "test"}, {"start_times": [1., .9]}]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                launcher.validate_config({**self.config, **change}, CONFIG)

    def write_child_results(self, folder, config):
        queries = [{"task_id": "aaaaaaaa", "query_index": 0},
                   {"task_id": "aaaaaaaa", "query_index": 1},
                   {"task_id": "bbbbbbbb", "query_index": 0}]
        first = folder.name == "step-004530"
        records = []
        for query in queries:
            base = {**query, "kind": "clean_roundtrip", "schedule": None, "start_t": None,
                    "sampling_steps": None, "seed": None, "exact_grid": False,
                    "cell_accuracy": None, "shape_match": False, "prediction_valid": True,
                    "latent_mse": 1., "initial_v_mse": 1., "one_step_z0_mse": 1.}
            records.append(base)
            for steps in config["generation_steps"]:
                for index, seed in enumerate(config["sampling_seeds"]):
                    for kind in ("noisy_start", "denoised"):
                        # Second checkpoint solves A's queries under different seeds: not a solved task.
                        exact = (query["task_id"] == "aaaaaaaa" and
                                 (index == 0 if first else index == query["query_index"]))
                        exact |= not first and query["task_id"] == "bbbbbbbb" and index == 0
                        records.append({**base, "kind": kind, "schedule": "linear", "start_t": 1.,
                                        "sampling_steps": steps, "seed": seed,
                                        "exact_grid": exact if kind == "denoised" else False})
        launcher.pod.write_json(folder / "report.json", {
            "status": "succeeded", "queries": queries, "selected_task_ids": ["aaaaaaaa", "bbbbbbbb"],
            "dataset_manifest": {"split_seed": 167},
            "trajectories": [{"start_t": 1., "steps": n} for n in config["generation_steps"]],
            "preparation": {"frozen_sources": {"nvarc": "same-pinned-model"}},
        })
        (folder / "records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
        return 0

    def test_supervised_comparison_pairs_tasks_without_mixing_queries_across_seeds(self):
        with patch.object(launcher, "resolve_checkpoints", return_value=self.config), \
                patch.object(launcher, "supervise", side_effect=lambda path, workspace, output, config:
                             self.write_child_results(output, config)) as supervise, redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run_comparison(CONFIG, self.output, self.output, self.config), 0)
        self.assertEqual(supervise.call_count, 2)
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["status"], "succeeded")
        self.assertEqual(report["completed_checkpoints"], list(self.config["checkpoints"]))
        for row in report["paired_tasks"]:
            self.assertEqual(row["first_only"], ["aaaaaaaa"])
            self.assertEqual(row["second_only"], ["bbbbbbbb"])
            self.assertEqual(row["both"], [])
        for call in supervise.call_args_list:
            child = launcher.load_config(call.args[0])
            self.assertEqual(child["sampling_seeds"], self.config["sampling_seeds"])
            self.assertEqual(child["generation_steps"], self.config["generation_steps"])
        broken = self.output / "step-004830/records.jsonl"
        broken.write_text("\n".join(broken.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            launcher.summarize_comparison(self.output, self.config)

    def test_missing_second_checkpoint_prevents_first_evaluation(self):
        with patch.object(launcher, "transformer_metadata", side_effect=[("a" * 40, [], 1), ValueError("missing 4830")]), \
                patch.object(launcher, "supervise") as supervise, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "missing 4830"):
                launcher.run_comparison(CONFIG, self.output, self.output, self.config)
        supervise.assert_not_called()
        self.assertEqual(json.loads((self.output / "report.json").read_text())["status"], "failed")

    def test_failed_child_preserves_results_and_prevents_second_evaluation(self):
        with patch.object(launcher, "resolve_checkpoints", return_value=self.config), \
                patch.object(launcher, "supervise", return_value=124) as supervise, redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run_comparison(CONFIG, self.output, self.output, self.config), 124)
        supervise.assert_called_once()
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["status"], "timed_out")
        self.assertEqual(report["active_checkpoint"], "step-004530")
        self.assertEqual(report["completed_checkpoints"], [])


if __name__ == "__main__":
    unittest.main()
