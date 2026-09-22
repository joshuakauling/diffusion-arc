"""Launcher boundaries, immutable downloads, and diagnostic result accounting."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.cloud import diagnose_runpod as launcher
from scripts.cloud import launch_runpod as transport
from scripts import diagnose_partial_denoising as worker


class DiagnosticLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = launcher.load_config(launcher.DEFAULT_CONFIG)

    def test_local_dry_run_selects_export_and_transfers_required_sources_without_network(self):
        stdout = io.StringIO()
        with patch.object(transport, "remote_python", side_effect=AssertionError("SSH attempted")), \
                redirect_stdout(stdout):
            code = launcher.main(["--ssh", "ssh root@192.0.2.1 -p 22001", "--run-name", "partial", "--dry-run"])
        plan = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(plan["arguments"], ["diagnose", "--run-name", "partial", "--workspace", "/workspace",
                                           "--config", "configs/diagnostics/partial_denoising_b3.json"])
        self.assertEqual(plan["diagnostic"]["checkpoint"]["subfolder"], "m7y2mfn2/step-001812/transformer")
        _, manifest = transport.source_snapshot(transport.ROOT)
        for path in ("scripts/cloud/diagnose_runpod.py", "scripts/diagnose_partial_denoising.py",
                     "src/diffusion_arc/diagnostics/partial_denoising.py", "configs/diagnostics/partial_denoising_b3.json"):
            self.assertIn(path, manifest["files"])

    def test_bad_time_grid_checkpoint_paths_and_unbounded_runtime_fail_locally(self):
        for field, value in (("start_times", [1., .123]), ("start_times", [1., .5, .5]),
                             ("max_runtime_seconds", 1801), ("max_setup_seconds", 1801),
                             ("sampling_seeds", [1, 1]), ("split_seed", 168),
                             ("task_count", 100), ("task_pool", "test"),
                             ("generation_schedules", ["cosine_high_noise"]),
                             ("generation_schedules", ["linear", "unknown"]),
                             ("checkpoint", {**self.config["checkpoint"], "subfolder": "../transformer"})):
            with self.subTest(field=field):
                config = {**self.config, field: value,
                          "training_config": str(launcher.ROOT / "configs/training/benchmarks/single_gpu_b3.json")}
                path = self.root / "invalid.json"
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    launcher.load_config(path)

    def test_epoch4_dry_run_selects_all_validation_tasks_and_three_trajectories(self):
        stdout = io.StringIO()
        with patch.object(transport, "remote_python", side_effect=AssertionError("SSH attempted")), \
                redirect_stdout(stdout):
            code = launcher.main(["--ssh", "ssh root@192.0.2.1 -p 22001", "--run-name", "epoch4",
                                  "--config", "configs/diagnostics/sampling_epoch4.json", "--dry-run"])
        plan = json.loads(stdout.getvalue())
        config = plan["diagnostic"]
        self.assertEqual(code, 0)
        self.assertEqual((config["task_pool"], config["task_count"]), ("validation", 100))
        self.assertEqual(config["checkpoint"]["subfolder"], "m7y2mfn2/step-003624/transformer")
        self.assertEqual(config["checkpoint"]["revision"], "a0080ba91538ab001fe958d2632a2f6a0cf71fb5")
        self.assertEqual(worker.diagnostic_variants(config),
                         [("linear", 1., 50), ("linear", .9, 45), ("cosine_high_noise", 1., 50)])

    def test_step_sweep_dry_run_preserves_the_existing_checkpoint_tasks_and_seeds(self):
        stdout = io.StringIO()
        with patch.object(transport, "remote_python", side_effect=AssertionError("SSH attempted")), \
                redirect_stdout(stdout):
            code = launcher.main(["--ssh", "ssh root@192.0.2.1 -p 22001", "--run-name", "epoch4-steps",
                                  "--config", "configs/diagnostics/sampling_steps_epoch4.json", "--dry-run"])
        config = json.loads(stdout.getvalue())["diagnostic"]
        reference = launcher.load_config(launcher.ROOT / "configs/diagnostics/sampling_epoch4.json")
        self.assertEqual(code, 0)
        for key in ("checkpoint", "task_pool", "task_count", "split_seed", "sampling_seeds", "training_config"):
            self.assertEqual(config[key], reference[key])
        self.assertEqual(worker.diagnostic_variants(config), [("linear", 1., n) for n in (1, 4, 10)])

    def test_step_sweep_rejects_ambiguous_or_invalid_trajectories_before_connecting(self):
        base = launcher.load_config(launcher.ROOT / "configs/diagnostics/sampling_steps_epoch4.json")
        invalid = [{"generation_steps": value} for value in ([], [0], [True], [1.5], [1, 1], [1001], None)]
        invalid += [{"start_times": [1.]}, {"steps_per_unit": 50}, {"generation_schedules": ["linear"]}]
        path = self.root / "invalid.json"
        for change in invalid:
            with self.subTest(change=change):
                path.write_text(json.dumps({**base, **change,
                    "training_config": str(launcher.ROOT / "configs/training/benchmarks/single_gpu_b3.json")}))
                with self.assertRaises(ValueError):
                    launcher.load_config(path)

    def hub_fixture(self):
        spec = self.config["checkpoint"]
        prefix = spec["subfolder"] + "/"
        payloads = {prefix + "config.json": b'{"in_channels":128}',
                    prefix + "diffusion_pytorch_model.safetensors": b"test weights"}
        entries = []
        for name, content in payloads.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            entries.append(SimpleNamespace(rfilename=name, size=len(content),
                                           lfs=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest())))
        entries.extend((SimpleNamespace(rfilename="other-run/transformer/model.safetensors", size=10**12),
                        SimpleNamespace(rfilename=prefix + "training_state.pt", size=10**12),
                        SimpleNamespace(rfilename=prefix + "nested/model.safetensors", size=10**12)))
        return spec, SimpleNamespace(sha="a" * 40, siblings=entries), payloads

    def test_download_is_pinned_and_limited_to_direct_transformer_files(self):
        spec, info, payloads = self.hub_fixture()
        api = Mock()
        api.model_info.return_value = info
        with patch("huggingface_hub.HfApi", return_value=api), \
                patch("huggingface_hub.snapshot_download", return_value=str(self.root)) as download, \
                patch.object(launcher.pod, "sha256_file", side_effect=AssertionError("weight rehash")), \
                patch.dict(os.environ, {"HF_TOKEN": "fixture-token"}), redirect_stdout(io.StringIO()):
            folder, identity = launcher.download_transformer(spec, workspace=self.root, max_gib=24)
        self.assertEqual(folder, self.root / spec["subfolder"])
        self.assertEqual(identity["resolved_revision"], info.sha)
        self.assertEqual(identity["bytes"], sum(map(len, payloads.values())))
        self.assertEqual(download.call_args.kwargs["revision"], info.sha)
        self.assertEqual(set(download.call_args.kwargs["allow_patterns"]), set(payloads))
        self.assertNotIn("fixture-token", json.dumps(identity))
        self.assertEqual(identity["files"]["diffusion_pytorch_model.safetensors"]["hub_sha256"],
                         info.siblings[1].lfs.sha256)

    def test_oversized_or_truncated_exports_are_rejected(self):
        spec, info, _ = self.hub_fixture()
        api = Mock()
        api.model_info.return_value = info
        with patch("huggingface_hub.HfApi", return_value=api), \
                patch("huggingface_hub.snapshot_download", return_value=str(self.root)) as download, \
                patch.dict(os.environ, {"HF_TOKEN": "fixture-token"}), redirect_stdout(io.StringIO()):
            old_size = info.siblings[1].size
            info.siblings[1].size = 25 * 1024**3
            with self.assertRaisesRegex(ValueError, "24 GiB"):
                launcher.download_transformer(spec, workspace=self.root, max_gib=24)
            download.assert_not_called()
            info.siblings[1].size = old_size
            (self.root / info.siblings[1].rfilename).write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "Incomplete downloaded file"):
                launcher.download_transformer(spec, workspace=self.root, max_gib=24)

    def test_elastic_volume_skips_statfs_but_keeps_size_checks(self):
        spec, info, _ = self.hub_fixture()
        api = Mock()
        api.model_info.return_value = info
        with patch("huggingface_hub.HfApi", return_value=api), \
                patch("huggingface_hub.snapshot_download", return_value=str(self.root)), \
                patch.dict(os.environ, {"HF_TOKEN": "fixture-token"}), \
                patch.object(launcher.shutil, "disk_usage", return_value=SimpleNamespace(free=0)) as disk, \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Insufficient workspace"):
                launcher.download_transformer(spec, workspace=self.root, max_gib=24)
            disk.reset_mock()
            launcher.download_transformer(spec, workspace=self.root, max_gib=24, elastic_volume=True)
            disk.assert_not_called()
            (self.root / info.siblings[1].rfilename).write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "Incomplete downloaded file"):
                launcher.download_transformer(spec, workspace=self.root, max_gib=24, elastic_volume=True)

    def test_diagnostic_preflight_needs_only_hf_and_dispatches_the_diagnostic_entrypoint(self):
        with patch.object(transport, "pod_environment", return_value={"HF_TOKEN": "fixture-token"}), \
                redirect_stdout(io.StringIO()):
            transport.remote_preflight("diagnose", str(self.root))
        job = self.root / "job"
        job.mkdir()
        (job / "request.json").write_text(json.dumps({"source_directory": str(launcher.ROOT),
            "arguments": ["diagnose", "--run-name", "fixture"]}))
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
            self.assertEqual(transport.run_remote_job(str(job)), 0)
        command = run.call_args.args[0]
        self.assertEqual(Path(command[1]), launcher.ROOT / "scripts/cloud/diagnose_runpod.py")
        self.assertEqual(command[2:], ["--run-name", "fixture"])

    def test_watchdog_terminates_child_during_setup_and_after_models_load(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        script = scripts / "diagnose_partial_denoising.py"
        script.write_text("\n".join((
            "import json, os, pathlib, sys, time",
            "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])",
            "(output / 'pid').write_text(str(os.getpid()))",
            "(output / 'report.json').write_text(json.dumps({'status': 'running'}))",
            "if output.name == 'ready':",
            "    (output / 'models-ready.json').write_text(json.dumps({'monotonic_seconds': time.monotonic()}))",
            "time.sleep(60)",
        )))
        for phase, directory in (("setup", "setup"), ("diagnostic", "ready")):
            with self.subTest(phase=phase):
                output = self.root / directory
                output.mkdir()
                with patch.object(launcher, "ROOT", self.root), redirect_stdout(io.StringIO()):
                    result = launcher.supervise(self.root / "unused.json", self.root, output,
                                               {"max_setup_seconds": .15, "max_runtime_seconds": .15})
                self.assertEqual(result, 124)
                self.assertEqual(json.loads((output / "timeout.json").read_text())["phase"], phase)
                self.assertEqual(json.loads((output / "report.json").read_text())["status"], "timed_out")
                with self.assertRaises(ProcessLookupError):
                    os.kill(int((output / "pid").read_text()), 0)

    def test_sigterm_to_supervisor_releases_the_diagnostic_process(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        (scripts / "diagnose_partial_denoising.py").write_text("\n".join((
            "import os, pathlib, sys, time",
            "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])",
            "(output / 'child-pid').write_text(str(os.getpid()))",
            "time.sleep(60)",
        )))
        program = "\n".join((
            "import sys",
            "from pathlib import Path",
            "from scripts.cloud import diagnose_runpod as launcher",
            "launcher.ROOT = Path(sys.argv[1])",
            "launcher.supervise(launcher.ROOT / 'config.json', launcher.ROOT, launcher.ROOT,",
            "                   {'max_setup_seconds': 60, 'max_runtime_seconds': 60})",
        ))
        process = subprocess.Popen([sys.executable, "-B", "-c", program, str(self.root)], cwd=launcher.ROOT)
        child_pid = None
        try:
            deadline = time.monotonic() + 5
            while not (self.root / "child-pid").exists():
                if time.monotonic() > deadline:
                    self.fail("supervised child did not start")
                time.sleep(.01)
            child_pid = int((self.root / "child-pid").read_text())
            process.terminate()
            self.assertEqual(process.wait(timeout=5), 143)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if child_pid is not None:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


class DiagnosticSummaryTests(unittest.TestCase):
    def test_task_success_requires_all_queries_under_same_seed_and_cells_exclude_wrong_shapes(self):
        config = {"start_times": [1.], "steps_per_unit": 50, "sampling_seeds": [20167, 20168],
                  "generation_schedules": ["linear", "cosine_high_noise"]}
        queries = [{"task_id": "multi", "query_index": index} for index in (0, 1)]
        records = []
        for query in queries:
            base = {**query, "prediction_valid": True, "shape_match": True, "cell_accuracy": 1.,
                    "exact_grid": True, "latent_mse": 0., "initial_v_mse": .4, "one_step_z0_mse": .4}
            records.append({**base, "kind": "clean_roundtrip", "start_t": None, "seed": None,
                            "schedule": None, "sampling_steps": None})
            for seed in config["sampling_seeds"]:
                for schedule in config["generation_schedules"]:
                    for kind in ("noisy_start", "denoised"):
                        # Linear seeds solve different queries; cosine's second seed solves both.
                        correct = ((seed == 20167) == (query["query_index"] == 0)
                                   if schedule == "linear" else seed == 20168)
                        records.append({**base, "kind": kind, "start_t": 1., "seed": seed, "schedule": schedule,
                                        "sampling_steps": 50,
                                        "exact_grid": correct, "shape_match": correct,
                                        "cell_accuracy": 1. if correct else None})
        summary = worker.summarize(records, queries, config)
        row = next(row for row in summary if row["kind"] == "denoised" and row["schedule"] == "linear")
        self.assertEqual(row["exact_grid_rate"], .5)
        self.assertEqual(row["task_exact_rate"], 0.)
        self.assertEqual(row["cell_accuracy_query_samples"], 2)
        self.assertEqual(row["cell_accuracy_on_correct_shapes"], 1.)
        self.assertEqual((row["pass_at_1"], row["pass_at_2"]), (0., 0.))
        cosine = next(row for row in summary if row["kind"] == "denoised" and row["schedule"] == "cosine_high_noise")
        self.assertEqual((cosine["pass_at_1"], cosine["pass_at_2"]), (0., 1.))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            worker.summarize(records[:-1], queries, config)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            worker.summarize([*records, records[0]], queries, config)

    def test_cpu_worker_pipeline_keeps_all_queries_of_selected_tasks_and_writes_artifacts(self):
        self.run_cpu_worker(launcher.ROOT / "configs/diagnostics/sampling_epoch4.json", [50, 45, 50])

    def test_cpu_step_sweep_loads_models_once_and_reports_each_step_count_separately(self):
        self.run_cpu_worker(launcher.ROOT / "configs/diagnostics/sampling_steps_epoch4.json", [1, 4, 10])

    def run_cpu_worker(self, config_path, expected_steps):
        from dataclasses import replace
        from PIL import Image
        import torch
        from diffusion_arc.diagnostics import partial_denoising
        from diffusion_arc.training.run import validation
        from tests.inference.test_partial_denoising import target, Decoder, Transformer

        config = launcher.load_config(config_path)
        config.update(task_count=2,
                      training_config=str(launcher.ROOT / "configs/training/benchmarks/single_gpu_b3.json"))
        targets = (target(), replace(target(), query_index=1), target(1))
        samples = [SimpleNamespace(task_id=t.task_id, query_index=t.query_index) for t in targets]
        samples.append(SimpleNamespace(task_id="not-selected", query_index=0))
        grids = [t.expected_grid for t in targets] + [((9,),)]
        models = SimpleNamespace(flux_transformer=Transformer(), vae_decoder=Decoder(),
                                 nvarc_encoder=object(), vae_encoder=object(), tokenizer=object(),
                                 flux_compute_dtype=torch.bfloat16, flux_packing_backend=object())
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "config.json"
            path.write_text(json.dumps(config))
            prepared = {"models": {"nvarc_checkpoint": "unused", "flux_checkpoint": "unused"},
                        "transformer": "selected-transformer", "bundle": "unused-bundle"}
            (output / "prepared.json").write_text(json.dumps(prepared))
            with patch.object(launcher, "prepare_job"), patch.object(worker, "ROOT", output), \
                    patch("diffusion_arc.training.setup.load_models", return_value=models) as load, \
                    patch("diffusion_arc.training.setup.prepare_device", return_value=torch.device("cpu")), \
                    patch("diffusion_arc.training.setup.seed_process"), \
                    patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "get_device_name", return_value="CPU fixture"), \
                    patch.object(validation, "validate_validation_bundle", return_value={
                        "sampling_task_ids": [targets[0].task_id],
                        "validation_task_ids": [targets[0].task_id, targets[2].task_id, "not-selected"]}), \
                    patch.object(validation, "tokenize_validation_targets", return_value=(samples, grids)), \
                    patch.object(validation, "prepare_validation_cache", return_value=targets) as cache, \
                    patch.object(validation, "load_sampling_input_grids", return_value={(t.task_id, t.query_index): ((0,),) for t in targets}), \
                    patch.object(partial_denoising, "predict_packed_flux_velocity", side_effect=lambda _, z, *a, **k: z * 0), \
                    redirect_stdout(io.StringIO()):
                worker.run(path, output, output)
            self.assertEqual([(sample.task_id, sample.query_index) for sample in cache.call_args.args[0]],
                             [(t.task_id, t.query_index) for t in targets])
            self.assertEqual(load.call_args.kwargs["transformer_checkpoint"], "selected-transformer")
            self.assertFalse(models.flux_transformer.anchor.requires_grad)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "succeeded")
            self.assertEqual(report["record_count"], 39)
            load.assert_called_once()
            cache.assert_called_once()
            self.assertEqual([item["steps"] for item in report["trajectories"]], expected_steps)
            self.assertEqual([item["sampling_steps"] for item in report["summaries"] if item["kind"] == "denoised"], expected_steps)
            self.assertEqual(report["planned_flux_forward_calls"], len(report["packs"]) * 2 * sum(expected_steps))
            self.assertEqual(len((output / "records.jsonl").read_text().splitlines()), 39)
            self.assertEqual(len(list((output / "comparisons").glob("*.png"))), 6)
            with Image.open(next((output / "comparisons").glob("*.png"))) as panel:
                panel.verify()


if __name__ == "__main__":
    unittest.main()
