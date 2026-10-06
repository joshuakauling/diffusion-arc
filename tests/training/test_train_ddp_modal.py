"""Offline checks of DDP cloud dispatch, input preflight and process cleanup."""

import importlib.util
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.training.test_data import make_split_config, make_task_payload, write_json
from tests.training.test_ddp_benchmarking import write_repetition
from tests.training.test_training_modal import _App, RecordingImage


ROOT = Path(__file__).resolve().parents[2]


class TrainDdpModalTests(unittest.TestCase):
    def setUp(self):
        self.volume, self.image = Mock(), RecordingImage()
        volume_factory = Mock(return_value=self.volume)
        fake_modal = SimpleNamespace(
            App=_App, is_local=lambda: True, __version__="test",
            Image=SimpleNamespace(from_registry=Mock(return_value=self.image)),
            Volume=SimpleNamespace(from_name=volume_factory),
        )
        spec = importlib.util.spec_from_file_location(
            "_ddp_modal_test", ROOT / "scripts/cloud/train_ddp_modal.py",
        )
        self.launcher = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modal": fake_modal}), patch.object(sys, "path", list(sys.path)):
            spec.loader.exec_module(self.launcher)
        volume_factory.assert_called_once_with("diffusion-arc-models", create_if_missing=False)

    def make_repository(self, directory, *, warmup_steps=0, measured_steps=1):
        repository = Path(directory).resolve()
        config = repository / self.launcher.CONFIG
        config.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "configs/training/arc_agi_1_epoch.json", config.parent.parent / "arc_agi_1_epoch.json")
        benchmark = json.loads((ROOT / self.launcher.CONFIG).read_text())
        benchmark.update(smoke_steps=1, warmup_steps=warmup_steps, measured_steps=measured_steps)
        benchmark["data"]["expected_source_count"] = 16 * (warmup_steps + measured_steps)
        write_json(config, benchmark)
        write_json(repository / self.launcher.NUMERICS_CRITERIA, dict(
            schema_version=1, steps=3, loss_rtol=0.001,
            gradient_relative_l2_max=0.05, chunk_elements=1048576,
        ))
        bundle = repository / self.launcher.BUNDLE
        split = bundle / "splits/train"
        data_config = make_split_config()
        data_config["limits"]["max_pairs"] = 5
        write_json(split / "config.json", data_config)
        ids = [f"task-{index}" for index in range(benchmark["data"]["expected_source_count"])]
        write_json(split / "train.json", ids)
        write_json(split / "validation.json", [])
        write_json(split / "augmented_keys/train.json",
                   [f"{task_id}_ID_0123456789_0,1,2,3,4" for task_id in ids])
        for task_id in ids:
            write_json(bundle / "raw/arc-agi-2/training" / f"{task_id}.json",
                       make_task_payload(((0, 1),) * 5))
        return repository

    def test_cloud_resources_and_dispatch_follow_gpu_count_and_mode(self):
        options = self.launcher.app.function_options
        self.assertEqual(options["volumes"], {str(self.launcher.VOLUME_ROOT): self.volume})
        self.assertEqual((options["max_containers"], options["retries"]), (1, 0))
        mounts = [args[0].relative_to(ROOT).as_posix()
                  for call, args, _ in self.image.calls if call == "add_local_dir"]
        self.assertEqual(mounts, ["src", "scripts", "configs", self.launcher.BUNDLE])
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory)
            for gpus, mode, timeout in ((1, "smoke", 600), (2, "benchmark", 5400), (2, "memory", 1800)):
                remote = Mock()
                with self.subTest(gpus=gpus, mode=mode), \
                        patch.object(self.launcher, "LOCAL_ROOT", repository), \
                        patch.object(self.launcher, "train_ddp", remote):
                    self.launcher.main(gpus=gpus, mode=mode)
                remote.with_options.assert_called_once_with(
                    gpu=f"H200:{gpus}", cpu=(8.0 * gpus,) * 2,
                    memory=(65536 * gpus,) * 2, timeout=timeout,
                )
                self.assertEqual(remote.with_options.return_value.remote.call_args.args[:3],
                                 (self.launcher.CONFIG, gpus, mode))

    def test_missing_data_or_invalid_options_never_dispatch(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(self.launcher, "train_ddp") as remote:
            repository = self.make_repository(directory)
            shutil.rmtree(repository / self.launcher.BUNDLE / "raw")
            with patch.object(self.launcher, "LOCAL_ROOT", repository):
                with self.assertRaisesRegex(FileNotFoundError, "raw/arc-agi-2/training/task-0.json"):
                    self.launcher.main()
                for options in (dict(gpus=4), dict(gpus=True), dict(mode="nsys")):
                    with self.subTest(options=options), self.assertRaises(ValueError):
                        self.launcher.main(**options)
            remote.with_options.assert_not_called()

    def test_numerics_requires_two_gpus_and_valid_criteria_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory, measured_steps=3)
            with patch.object(self.launcher, "LOCAL_ROOT", repository), \
                    patch.object(self.launcher, "train_ddp") as remote:
                self.launcher.main(gpus=2, mode="numerics")
                remote.with_options.assert_called_once_with(
                    gpu="H200:2", cpu=(16.0, 16.0), memory=(131072, 131072), timeout=1800,
                )
                remote.reset_mock()
                with self.assertRaisesRegex(ValueError, "exactly 2 GPUs"):
                    self.launcher.main(gpus=1, mode="numerics")
                (repository / self.launcher.NUMERICS_CRITERIA).unlink()
                with self.assertRaises(FileNotFoundError):
                    self.launcher.main(gpus=2, mode="numerics")
                remote.with_options.assert_not_called()
        self.assertEqual(self.launcher.resources(2, "numerics", repetitions=7)["timeout"], 1800)

    def test_numerics_runs_plain_then_ddp_with_one_budget_and_removes_tensor_scratch(self):
        for result_status in ("passed", "failed"):
            with self.subTest(result_status=result_status), tempfile.TemporaryDirectory() as directory:
                repository = self.make_repository(Path(directory) / "repository", measured_steps=3)
                volume = Path(directory).resolve() / "volume"
                experiment = self.launcher.check_inputs(repository, self.launcher.CONFIG)
                for checkpoint in (experiment.training.models.nvarc_checkpoint,
                                   experiment.training.models.flux_checkpoint):
                    (volume / checkpoint.name).mkdir(parents=True)
                commands, timeouts = [], []

                def worker(command, *, timeout):
                    commands.append(command)
                    timeouts.append(timeout)
                    phase = command[command.index("--phase") + 1]
                    scratch = Path(command[command.index("--scratch-dir") + 1])
                    self.assertTrue(scratch.is_dir())
                    self.assertFalse(scratch.is_relative_to(volume))
                    self.assertEqual(command[command.index("--criteria") + 1],
                                     str(repository / self.launcher.NUMERICS_CRITERIA))
                    if phase == "reference":
                        (scratch / "gradients.bin").write_bytes(b"test tensor")
                    else:
                        self.assertTrue((scratch / "gradients.bin").is_file())
                    if phase == "compare":
                        output = Path(command[command.index("--output-dir") + 1])
                        write_json(output / "summary.json", dict(
                            event="complete", status=result_status, optimizer_steps=3,
                        ))

                with patch.object(self.launcher, "REMOTE_ROOT", repository), \
                        patch.object(self.launcher, "VOLUME_ROOT", volume), \
                        patch.object(self.launcher.time, "monotonic", side_effect=[0, 10, 120, 900]), \
                        patch.object(self.launcher, "run_worker", side_effect=worker):
                    self.launcher.train_ddp(self.launcher.CONFIG, 2, "numerics", {})
                self.assertEqual(timeouts, [1700, 1590, 810])
                self.assertEqual([command[command.index("--phase") + 1] for command in commands],
                                 ["reference", "ddp", "compare"])
                self.assertNotIn("torch.distributed.run", commands[0])
                self.assertNotIn("torch.distributed.run", commands[2])
                self.assertIn("--nproc-per-node=2", commands[1])
                self.assertIn("--max-restarts=0", commands[1])
                scratch_paths = {command[command.index("--scratch-dir") + 1] for command in commands}
                self.assertEqual(len(scratch_paths), 1)
                self.assertFalse(Path(scratch_paths.pop()).exists())
                self.assertFalse(list(volume.rglob("*.bin")))
                self.volume.commit.assert_called_once_with()
                self.volume.reset_mock()
                launch = json.loads(next((volume / "results/ddp").glob("*/launch.json")).read_text())
                self.assertEqual(launch["status"], "complete")
                self.assertEqual(launch["validation_status"], result_status)
                self.assertEqual([phase["status"] for phase in launch["phases"]], ["complete"] * 3)

    def test_numerics_stops_on_failure_or_shared_deadline_and_commits_partial_artifacts(self):
        for outcome in ("failure", "deadline"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                repository = self.make_repository(Path(directory) / "repository", measured_steps=3)
                volume = Path(directory).resolve() / "volume"
                experiment = self.launcher.check_inputs(repository, self.launcher.CONFIG)
                for checkpoint in (experiment.training.models.nvarc_checkpoint,
                                   experiment.training.models.flux_checkpoint):
                    (volume / checkpoint.name).mkdir(parents=True)
                scratch_paths = []

                def worker(command, *, timeout):
                    scratch = Path(command[command.index("--scratch-dir") + 1])
                    scratch_paths.append(scratch)
                    (scratch / "gradient.bin").write_bytes(b"partial capture")
                    output = Path(command[command.index("--output-dir") + 1])
                    write_json(output / "partial.json", {"event": "started"})
                    if outcome == "failure":
                        raise subprocess.CalledProcessError(1, command)

                expected_error = subprocess.CalledProcessError if outcome == "failure" else TimeoutError
                with patch.object(self.launcher, "REMOTE_ROOT", repository), \
                        patch.object(self.launcher, "VOLUME_ROOT", volume), \
                        patch.object(self.launcher.time, "monotonic", side_effect=[0, 10, 1711]), \
                        patch.object(self.launcher, "run_worker", side_effect=worker) as run:
                    with self.assertRaises(expected_error):
                        self.launcher.train_ddp(self.launcher.CONFIG, 2, "numerics", {})
                    run.assert_called_once()
                self.assertFalse(scratch_paths[0].exists())
                self.volume.commit.assert_called_once_with()
                self.volume.reset_mock()
                launch_path = next((volume / "results/ddp").glob("*/launch.json"))
                launch = json.loads(launch_path.read_text())
                self.assertEqual(launch["status"], "failed")
                self.assertEqual(launch["phases"][-1]["status"], "failed")
                self.assertTrue((launch_path.parent / "training/partial.json").is_file())

    def test_diagnostic_dispatch_and_batch_bounds_are_checked_before_gpu_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory)
            with patch.object(self.launcher, "LOCAL_ROOT", repository), \
                    patch.object(self.launcher, "train_ddp") as remote:
                self.launcher.main(gpus=2, mode="diagnostic", diagnostic_batch=1)
                self.assertEqual(remote.with_options.call_args.kwargs["timeout"], 600)
                self.assertEqual(remote.with_options.return_value.remote.call_args.args[-1], 1)
                remote.reset_mock()
                for options in (dict(mode="diagnostic", diagnostic_batch=2),
                                dict(mode="diagnostic"), dict(mode="smoke", diagnostic_batch=1)):
                    with self.assertRaises(ValueError):
                        self.launcher.main(**options)
                remote.with_options.assert_not_called()

    def test_preflight_preserves_split_boundary_and_requires_remote_models(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory)
            config = self.launcher.CONFIG
            self.assertEqual(self.launcher.check_inputs(repository, config).expected_source_count, 16)
            with self.assertRaisesRegex(FileNotFoundError, "cached model"):
                self.launcher.check_inputs(repository, config, remote=True)
            write_json(repository / self.launcher.BUNDLE / "splits/train/validation.json", ["task-0"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.launcher.check_inputs(repository, config)

    def test_remote_persists_success_and_failure_and_checks_complete_window(self):
        for outcome in ("complete", "failed", "incomplete", "diagnostic", "memory"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                repository = self.make_repository(Path(directory) / "repository")
                volume = Path(directory).resolve() / "volume"
                volume.mkdir()
                experiment = self.launcher.check_inputs(repository, self.launcher.CONFIG)
                for checkpoint in (experiment.training.models.nvarc_checkpoint,
                                   experiment.training.models.flux_checkpoint):
                    (volume / checkpoint.name).mkdir()

                def worker(command, *, timeout):
                    self.assertIn("--nproc-per-node=2", command)
                    self.assertIn("torch.distributed.run", command)
                    self.assertIn("--max-restarts=0", command)
                    self.assertEqual(timeout, 1710 if outcome == "memory" else 510)
                    if outcome == "memory":
                        self.assertEqual(command[command.index("--mode") + 1], "memory")
                    if outcome == "diagnostic":
                        self.assertEqual(command[command.index("--diagnostic-batch") + 1], "1")
                    output = Path(command[command.index("--output-dir") + 1])
                    write_json(output / "summary.json", dict(
                        event="complete", optimizer_steps=(3 if outcome == "diagnostic" else
                                                          0 if outcome == "incomplete" else 1),
                    ))
                    if outcome == "failed":
                        raise subprocess.CalledProcessError(1, command)

                self.volume.reset_mock()
                with patch.object(self.launcher, "REMOTE_ROOT", repository), \
                        patch.object(self.launcher, "VOLUME_ROOT", volume), \
                        patch.object(self.launcher, "run_worker", side_effect=worker):
                    if outcome in ("complete", "diagnostic", "memory"):
                        self.launcher.train_ddp(self.launcher.CONFIG, 2,
                                               outcome if outcome in ("diagnostic", "memory") else "smoke", {}, 1)
                    else:
                        error = subprocess.CalledProcessError if outcome == "failed" else RuntimeError
                        with self.assertRaises(error):
                            self.launcher.train_ddp(self.launcher.CONFIG, 2, "smoke", {})
                self.volume.commit.assert_called_once_with()
                launch = json.loads(next((volume / "results/ddp").glob("*/launch.json")).read_text())
                self.assertEqual(launch["status"], "complete" if outcome in ("complete", "diagnostic", "memory") else "failed")
                self.assertEqual(launch["allocator_config"], "expandable_segments:True")
                self.assertEqual((repository / "checkpoints").resolve(), volume)

    def test_benchmark_restarts_workers_sequentially_and_preserves_partial_results(self):
        for fail_second in (False, True):
            with self.subTest(fail_second=fail_second), tempfile.TemporaryDirectory() as directory:
                repository = self.make_repository(Path(directory) / "repository")
                volume = Path(directory).resolve() / "volume"
                experiment = self.launcher.check_inputs(repository, self.launcher.CONFIG)
                for checkpoint in (experiment.training.models.nvarc_checkpoint,
                                   experiment.training.models.flux_checkpoint):
                    (volume / checkpoint.name).mkdir(parents=True)
                commands = []

                def worker(command, *, timeout):
                    self.assertEqual(timeout, 1710)
                    self.assertIn("--nproc-per-node=2", command)
                    output = Path(command[command.index("--output-dir") + 1])
                    if commands:
                        previous = Path(commands[-1][-1])
                        self.assertTrue((previous / "summary.json").is_file())
                        self.assertEqual(self.volume.commit.call_count, len(commands))
                        self.assertEqual(command[:-1], commands[-1][:-1])
                    commands.append(command)
                    if fail_second and len(commands) == 2:
                        raise subprocess.CalledProcessError(1, command)
                    write_repetition(output, [2 ** len(commands)], warmup_steps=0)

                self.volume.reset_mock()
                with patch.object(self.launcher, "REMOTE_ROOT", repository), \
                        patch.object(self.launcher, "VOLUME_ROOT", volume), \
                        patch.object(self.launcher, "run_worker", side_effect=worker):
                    if fail_second:
                        with self.assertRaises(subprocess.CalledProcessError):
                            self.launcher.train_ddp(self.launcher.CONFIG, 2, "benchmark", {})
                    else:
                        self.launcher.train_ddp(self.launcher.CONFIG, 2, "benchmark", {})
                root = next((volume / "results/ddp").iterdir())
                launch = json.loads((root / "launch.json").read_text())
                result = json.loads((root / "summary.json").read_text())
                self.assertEqual(launch["resources"]["timeout"], 5400)
                self.assertEqual(len(commands), 2 if fail_second else 3)
                self.assertEqual(self.volume.commit.call_count, len(commands))
                self.assertEqual(result["completed_repetitions"], 1 if fail_second else 3)
                self.assertEqual(result["status"], "partial" if fail_second else "complete")
                self.assertEqual(launch["status"], "failed" if fail_second else "complete")
                if not fail_second:
                    self.assertAlmostEqual(result["benchmark"]["targets_per_second"], 144 / 14)

    def test_profile_runs_once_and_requires_a_trace_from_every_rank(self):
        for missing_rank in (None, 1):
            with self.subTest(missing_rank=missing_rank), tempfile.TemporaryDirectory() as directory:
                repository = self.make_repository(Path(directory) / "repository", warmup_steps=1)
                volume = Path(directory).resolve() / "volume"
                experiment = self.launcher.check_inputs(repository, self.launcher.CONFIG)
                for checkpoint in (experiment.training.models.nvarc_checkpoint,
                                   experiment.training.models.flux_checkpoint):
                    (volume / checkpoint.name).mkdir(parents=True)
                with patch.object(self.launcher, "LOCAL_ROOT", repository), \
                        patch.object(self.launcher, "train_ddp") as remote:
                    self.launcher.main(gpus=2, mode="profile")
                    self.assertEqual(remote.with_options.call_args.kwargs["timeout"], 900)

                def worker(command, *, timeout):
                    self.assertEqual(timeout, 810)
                    output = Path(command[-1])
                    write_json(output / "summary.json", dict(event="complete", optimizer_steps=2))
                    for rank in range(2):
                        if rank != missing_rank:
                            (output / f"rank-{rank}.trace.json.gz").write_bytes(b"trace")

                with patch.object(self.launcher, "REMOTE_ROOT", repository), \
                        patch.object(self.launcher, "VOLUME_ROOT", volume), \
                        patch.object(self.launcher, "run_worker", side_effect=worker) as run:
                    if missing_rank is None:
                        self.launcher.train_ddp(self.launcher.CONFIG, 2, "profile", {})
                    else:
                        with self.assertRaisesRegex(RuntimeError, "trace missing for rank 1"):
                            self.launcher.train_ddp(self.launcher.CONFIG, 2, "profile", {})
                    run.assert_called_once()

    def test_worker_environment_failure_and_timeout_kill_all_ranks(self):
        process = Mock(pid=123)
        process.wait.side_effect = [subprocess.TimeoutExpired("torchrun", 510),
                                   subprocess.TimeoutExpired("torchrun", 10), -9]
        process.poll.return_value = None
        with patch.dict("os.environ", {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "16",
                                       "PYTORCH_ALLOC_CONF": "expandable_segments:False"}), \
                patch.object(self.launcher.subprocess, "Popen", return_value=process) as popen, \
                patch.object(self.launcher.os, "killpg") as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                self.launcher.run_worker(["python", "-m", "torch.distributed.run"], timeout=510)
        self.assertEqual([call.args for call in kill.call_args_list],
                         [(123, signal.SIGTERM), (123, signal.SIGKILL)])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["OMP_NUM_THREADS"], "8")
        self.assertEqual(environment["MKL_NUM_THREADS"], "8")
        self.assertEqual(environment["PYTORCH_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        process.wait.side_effect = None
        process.wait.return_value = 2
        with patch.object(self.launcher.subprocess, "Popen", return_value=process):
            with self.assertRaises(subprocess.CalledProcessError):
                self.launcher.run_worker(["torchrun"], timeout=510)


if __name__ == "__main__":
    unittest.main()
