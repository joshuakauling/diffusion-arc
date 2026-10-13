"""Offline checks of DDP cloud dispatch, input preflight and process cleanup."""

import importlib.util
import json
import os
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
FSDP_CONFIG = "configs/training/benchmarks/fsdp2_training.json"
PRECOMPUTED_CONFIG = "configs/training/benchmarks/b15_fsdp2_14k_precomputed_packing.json"
SCALING_CONFIGS = {
    4: "configs/training/benchmarks/ddp_4gpu_12k_background_packing.json",
    8: "configs/training/benchmarks/ddp_8gpu_12k_background_packing.json",
}


class TrainDdpModalTests(unittest.TestCase):
    def setUp(self):
        self.load_launcher()

    def load_launcher(self, *, environment=None, scaling_bundles=()):
        self.volume, self.image = Mock(), RecordingImage()
        self.volume_factory = Mock(return_value=self.volume)
        fake_modal = SimpleNamespace(
            App=_App, is_local=lambda: True, __version__="test",
            Image=SimpleNamespace(from_registry=Mock(return_value=self.image)),
            Volume=SimpleNamespace(from_name=self.volume_factory),
        )
        spec = importlib.util.spec_from_file_location(
            "_ddp_modal_test", ROOT / "scripts/cloud/train_ddp_modal.py",
        )
        self.launcher = importlib.util.module_from_spec(spec)
        backend_keys = {"DIFFUSION_ARC_MODAL_VOLUME", "DIFFUSION_ARC_MODAL_VOLUME_ROOT",
                        "DIFFUSION_ARC_MODAL_MODEL_SUBDIRECTORY"}
        backend_environment = {key: value for key, value in os.environ.items() if key not in backend_keys}
        backend_environment.update(environment or {})
        available_bundles = {ROOT / bundle for bundle in scaling_bundles}
        with patch.dict(sys.modules, {"modal": fake_modal}), \
                patch.dict("os.environ", backend_environment, clear=True), \
                patch.object(sys, "path", list(sys.path)), \
                patch.object(Path, "is_dir", autospec=True,
                             side_effect=lambda path: path in available_bundles):
            spec.loader.exec_module(self.launcher)
        self.volume_factory.assert_called_once_with(
            backend_environment.get("DIFFUSION_ARC_MODAL_VOLUME", "diffusion-arc-models"), create_if_missing=False,
        )

    def make_repository(self, directory, *, warmup_steps=0, measured_steps=1,
                        config=None, benchmark_root=None):
        repository = Path(directory).resolve()
        config_name = config or self.launcher.CONFIG
        config_path = repository / config_name
        config_path.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "configs/training/arc_agi_1_epoch.json", config_path.parent.parent / "arc_agi_1_epoch.json")
        benchmark = json.loads((ROOT / config_name).read_text())
        benchmark.update(smoke_steps=1, warmup_steps=warmup_steps, measured_steps=measured_steps)
        targets_per_batch = benchmark.get("loop", {}).get("optimizer_batch_size", 48)
        benchmark["data"]["expected_source_count"] = targets_per_batch // 3 * (warmup_steps + measured_steps)
        if benchmark_root is not None:
            benchmark["data"]["split_directory"] = str(repository / benchmark_root / "splits/train")
        write_json(config_path, benchmark)
        write_json(repository / self.launcher.NUMERICS_CRITERIA, dict(
            schema_version=1, steps=3, loss_rtol=0.001,
            gradient_relative_l2_max=0.05, chunk_elements=1048576,
        ))
        split = (config_path.parent / benchmark["data"]["split_directory"]).resolve()
        bundle = split.parent.parent
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
        self.assertEqual(self.launcher.VOLUME_NAME, "diffusion-arc-models")
        self.assertEqual(self.launcher.VOLUME_ROOT, Path("/mnt/diffusion-arc-models"))
        self.assertEqual(self.launcher.MODEL_SUBDIRECTORY, Path("."))
        options = self.launcher.app.function_options
        self.assertEqual(options["volumes"], {str(self.launcher.VOLUME_ROOT): self.volume})
        self.assertEqual((options["max_containers"], options["retries"]), (1, 0))
        mounts = [args[0].relative_to(ROOT).as_posix()
                  for call, args, _ in self.image.calls if call == "add_local_dir"]
        self.assertEqual(mounts, ["src", "scripts", "configs", self.launcher.BUNDLE])
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory)
            for gpus, mode, timeout in ((1, "smoke", 600), (2, "benchmark", 5400), (2, "memory", 1800),
                                        (4, "smoke", 600), (8, "smoke", 600)):
                config_path = repository / self.launcher.CONFIG
                payload = json.loads(config_path.read_text())
                payload["execution"]["gpu_count"] = gpus
                write_json(config_path, payload)
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

    def test_new_backend_is_propagated_to_worker_image_and_mount(self):
        backend = {
            "DIFFUSION_ARC_MODAL_VOLUME": "diffusion-arc-training-v2",
            "DIFFUSION_ARC_MODAL_VOLUME_ROOT": "/workspace",
            "DIFFUSION_ARC_MODAL_MODEL_SUBDIRECTORY": "models",
        }
        self.load_launcher(environment=backend)
        self.assertEqual(self.launcher.app.function_options["volumes"], {"/workspace": self.volume})
        self.assertEqual(self.launcher.VOLUME_ROOT / self.launcher.MODEL_SUBDIRECTORY, Path("/workspace/models"))
        environment = {key: value for call, args, _ in self.image.calls if call == "env"
                       for key, value in args[0].items()}
        for name, value in backend.items():
            self.assertEqual(environment[name], value)

    def test_invalid_backend_paths_fail_before_selecting_a_volume(self):
        for environment in (
            {"DIFFUSION_ARC_MODAL_VOLUME": ""},
            {"DIFFUSION_ARC_MODAL_VOLUME_ROOT": "workspace"},
            {"DIFFUSION_ARC_MODAL_VOLUME_ROOT": "/workspace/../outside"},
            {"DIFFUSION_ARC_MODAL_MODEL_SUBDIRECTORY": "/models"},
            {"DIFFUSION_ARC_MODAL_MODEL_SUBDIRECTORY": "../models"},
            {"DIFFUSION_ARC_MODAL_MODEL_SUBDIRECTORY": "models/../outside"},
            {"DIFFUSION_ARC_MODAL_MODEL_SUBDIRECTORY": ""},
        ):
            with self.subTest(environment=environment), self.assertRaisesRegex(ValueError, "DIFFUSION_ARC_MODAL"):
                self.load_launcher(environment=environment)
            self.volume_factory.assert_not_called()

    def test_image_uploads_only_the_baseline_and_present_explicit_scaling_bundles(self):
        for bundles in ((), ("data/benchmarks/ddp_4gpu",),
                        ("data/benchmarks/ddp_4gpu", "data/benchmarks/ddp_8gpu")):
            with self.subTest(bundles=bundles):
                self.load_launcher(scaling_bundles=bundles)
                uploads = [args[0].relative_to(ROOT).as_posix()
                           for call, args, _ in self.image.calls if call == "add_local_dir"]
                self.assertEqual(uploads, ["src", "scripts", "configs", self.launcher.BUNDLE, *bundles])
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            (repository / "data/benchmarks/unsupported").mkdir(parents=True)
            self.assertEqual(self.launcher.uploaded_benchmark_bundles(repository), (self.launcher.BUNDLE,))

    def test_scaling_configs_require_their_prepared_bundle_and_token_lengths_before_dispatch(self):
        from diffusion_arc.training.data import ArcTrainingDataset
        from diffusion_arc.training.source_lengths import write_source_token_lengths

        for gpus, config_name in SCALING_CONFIGS.items():
            with self.subTest(gpus=gpus), tempfile.TemporaryDirectory() as directory:
                repository = self.make_repository(directory, config=config_name)
                split = repository / f"data/benchmarks/ddp_{gpus}gpu/splits/train"
                with patch.object(self.launcher, "LOCAL_ROOT", repository), \
                        patch.object(self.launcher, "train_ddp") as remote:
                    with self.assertRaisesRegex(FileNotFoundError, "Missing source token lengths"):
                        self.launcher.main(gpus=gpus, config=config_name)
                    remote.with_options.assert_not_called()
                    dataset = ArcTrainingDataset(split, "train", tokenizer=None)
                    write_source_token_lengths(dataset, ((10, 20, 30),) * len(dataset.keys),
                        min_context_pairs_before_target=2, provenance={})
                    experiment = self.launcher.check_inputs(repository, config_name)
                    self.assertEqual(experiment.expected_source_count, 8 * gpus)
                    self.assertEqual(experiment.training.loop.optimizer_batch_size, 24 * gpus)
                    self.assertEqual(experiment.training.recipe["flow"]["endpoint_targets_per_full_batch"], 2 * gpus)
                    self.launcher.main(gpus=gpus, config=config_name)
                    self.assertEqual(remote.with_options.return_value.remote.call_args.args[:3],
                                     (config_name, gpus, "smoke"))
                    remote.reset_mock()
                    shutil.rmtree(split.parent.parent)
                    with self.assertRaisesRegex(ValueError, "uploads only"):
                        self.launcher.main(gpus=gpus, config=config_name)
                    remote.with_options.assert_not_called()

    def test_preflight_rejects_unsupported_benchmark_bundle_even_when_prepared(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory, benchmark_root="data/benchmarks/unsupported")
            with self.assertRaisesRegex(ValueError, "uploads only"):
                self.launcher.check_inputs(repository, self.launcher.CONFIG)

    def test_missing_data_or_invalid_options_never_dispatch(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(self.launcher, "train_ddp") as remote:
            repository = self.make_repository(directory)
            shutil.rmtree(repository / self.launcher.BUNDLE / "raw")
            with patch.object(self.launcher, "LOCAL_ROOT", repository):
                with self.assertRaisesRegex(FileNotFoundError, "raw/arc-agi-2/training/task-0.json"):
                    self.launcher.main()
                for options in (dict(gpus=3), dict(gpus=True), dict(mode="nsys")):
                    with self.subTest(options=options), self.assertRaises(ValueError):
                        self.launcher.main(**options)
            remote.with_options.assert_not_called()

    def test_gpu_count_mismatch_never_dispatches(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory)
            with patch.object(self.launcher, "LOCAL_ROOT", repository), \
                    patch.object(self.launcher, "train_ddp") as remote:
                with self.assertRaisesRegex(ValueError, "must match"):
                    self.launcher.main(gpus=4, mode="smoke")
                remote.with_options.assert_not_called()

    def test_background_packing_requires_metadata_and_rejects_numerics_before_dispatch(self):
        from diffusion_arc.training.data import ArcTrainingDataset
        from diffusion_arc.training.source_lengths import write_source_token_lengths

        config_name = "configs/training/benchmarks/fsdp2_12k_background_packing.json"
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory, config=config_name)
            with patch.object(self.launcher, "LOCAL_ROOT", repository), \
                    patch.object(self.launcher, "train_ddp") as remote:
                with self.assertRaisesRegex(FileNotFoundError, "Missing source token lengths"):
                    self.launcher.main(gpus=2, mode="smoke", config=config_name)
                remote.with_options.assert_not_called()
                dataset = ArcTrainingDataset(repository / self.launcher.BUNDLE / "splits/train", "train", tokenizer=None)
                write_source_token_lengths(dataset, ((10, 20, 30),) * len(dataset.keys),
                    min_context_pairs_before_target=2, provenance={})
                self.launcher.main(gpus=2, mode="smoke", config=config_name)
                remote.with_options.assert_called_once()
                remote.reset_mock()
                with self.assertRaisesRegex(ValueError, "numerics does not support background joint packing"):
                    self.launcher.main(gpus=2, mode="numerics", config=config_name)
                remote.with_options.assert_not_called()

    def test_token_balancing_requires_matching_prepared_lengths_before_dispatch(self):
        from diffusion_arc.training.data import ArcTrainingDataset
        from diffusion_arc.training.source_lengths import write_source_token_lengths

        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory)
            path = repository / self.launcher.CONFIG
            payload = json.loads(path.read_text())
            payload["execution"]["rank_batch_assignment"] = "token_balanced_v1"
            write_json(path, payload)
            with patch.object(self.launcher, "LOCAL_ROOT", repository), \
                    patch.object(self.launcher, "train_ddp") as remote:
                with self.assertRaisesRegex(FileNotFoundError, "Missing source token lengths"):
                    self.launcher.main(gpus=2, mode="benchmark")
                remote.with_options.assert_not_called()
                dataset = ArcTrainingDataset(repository / self.launcher.BUNDLE / "splits/train", "train", tokenizer=None)
                lengths = write_source_token_lengths(dataset, ((10, 20, 30),) * len(dataset.keys),
                    min_context_pairs_before_target=2, provenance={})
                self.launcher.main(gpus=2, mode="benchmark")
                remote.with_options.assert_called_once()
                remote.reset_mock()
                metadata = json.loads(lengths.read_text())
                metadata["data_fingerprint"] = "other epoch"
                write_json(lengths, metadata)
                with self.assertRaisesRegex(ValueError, "Stale or mismatched"):
                    self.launcher.main(gpus=2, mode="benchmark")
                remote.with_options.assert_not_called()

    def test_precomputed_packing_preflight_and_dispatch_require_prepared_metadata(self):
        from diffusion_arc.training.data import ArcTrainingDataset
        from diffusion_arc.training.source_lengths import write_source_token_lengths

        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory, config=PRECOMPUTED_CONFIG)
            with patch.object(self.launcher, "LOCAL_ROOT", repository), \
                    patch.object(self.launcher, "train_ddp") as remote:
                with self.assertRaisesRegex(FileNotFoundError, "Missing source token lengths"):
                    self.launcher.main(gpus=2, mode="benchmark", config=PRECOMPUTED_CONFIG)
                remote.with_options.assert_not_called()
                dataset = ArcTrainingDataset(repository / self.launcher.BUNDLE / "splits/train", "train", tokenizer=None)
                write_source_token_lengths(dataset, ((10, 20, 30),) * len(dataset.keys),
                    min_context_pairs_before_target=2, provenance={})
                self.launcher.main(gpus=2, mode="benchmark", config=PRECOMPUTED_CONFIG)
                remote.with_options.assert_called_once_with(
                    gpu="H200:2", cpu=(16.0, 16.0), memory=(131072, 131072), timeout=1800,
                )
                self.assertEqual(remote.with_options.return_value.remote.call_args.args[:3],
                                 (PRECOMPUTED_CONFIG, 2, "benchmark"))
                remote.reset_mock()
                with self.assertRaisesRegex(ValueError, "numerics does not support precompute_packing"):
                    self.launcher.main(gpus=2, mode="numerics", config=PRECOMPUTED_CONFIG)
                remote.with_options.assert_not_called()

    def test_per_microbatch_policy_is_rejected_by_numerics_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory, config=FSDP_CONFIG)
            path = repository / FSDP_CONFIG
            payload = json.loads(path.read_text())
            payload["fsdp_gradient_sync"] = "per_microbatch"
            write_json(path, payload)
            with patch.object(self.launcher, "LOCAL_ROOT", repository), \
                    patch.object(self.launcher, "train_ddp") as remote:
                with self.assertRaisesRegex(ValueError, "numerics does not support fsdp_gradient_sync"):
                    self.launcher.main(gpus=2, mode="numerics", config=FSDP_CONFIG)
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

    def test_fsdp_numerics_selects_config_and_rejects_incompatible_execution_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory, measured_steps=3, config=FSDP_CONFIG)
            config_path = repository / FSDP_CONFIG
            payload = json.loads(config_path.read_text())
            with patch.object(self.launcher, "LOCAL_ROOT", repository), \
                    patch.object(self.launcher, "train_ddp") as remote:
                self.launcher.main(gpus=2, mode="numerics", config=FSDP_CONFIG)
                remote.with_options.assert_called_once_with(
                    gpu="H200:2", cpu=(16.0, 16.0), memory=(131072, 131072), timeout=1800,
                )
                self.assertEqual(remote.with_options.return_value.remote.call_args.args[:3],
                                 (FSDP_CONFIG, 2, "numerics"))
                remote.reset_mock()
                for strategy, gpu_count, error in (
                    ("fsdp2", 1, "configured gpu_count=2"),
                    ("fsdp2", 4, "configured gpu_count=2"),
                    ("single_process", 1, "ddp or fsdp2"),
                ):
                    payload["execution"].update(distributed_strategy=strategy, gpu_count=gpu_count)
                    write_json(config_path, payload)
                    with self.subTest(strategy=strategy, gpu_count=gpu_count), \
                            self.assertRaisesRegex(ValueError, error):
                        self.launcher.main(gpus=2, mode="numerics", config=FSDP_CONFIG)
                remote.with_options.assert_not_called()

    def test_numerics_runs_selected_phases_with_one_budget_and_removes_tensor_scratch(self):
        for config, result_status in ((config, status) for config in (self.launcher.CONFIG, FSDP_CONFIG)
                                     for status in ("passed", "failed")):
            with self.subTest(config=config, result_status=result_status), tempfile.TemporaryDirectory() as directory:
                repository = self.make_repository(Path(directory) / "repository", measured_steps=3, config=config)
                volume = Path(directory).resolve() / "volume"
                experiment = self.launcher.check_inputs(repository, config)
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
                    self.assertEqual(command[command.index("--config") + 1], str(repository / config))
                    if phase == "reference":
                        (scratch / "gradients.bin").write_bytes(b"test tensor")
                    else:
                        self.assertTrue((scratch / "gradients.bin").is_file())
                    if phase == "compare":
                        output = Path(command[command.index("--output-dir") + 1])
                        write_json(output / "summary.json", dict(
                            event="complete", status=result_status, optimizer_steps=3,
                        ))

                include_fsdp2 = config == FSDP_CONFIG
                times = [0, 10, 120, 400, 900] if include_fsdp2 else [0, 10, 120, 900]
                phases = ["reference", "ddp", "fsdp2", "compare"] if include_fsdp2 else ["reference", "ddp", "compare"]
                with patch.object(self.launcher, "REMOTE_ROOT", repository), \
                        patch.object(self.launcher, "VOLUME_ROOT", volume), \
                        patch.object(self.launcher.time, "monotonic", side_effect=times), \
                        patch.object(self.launcher, "run_worker", side_effect=worker):
                    self.launcher.train_ddp(config, 2, "numerics", {})
                self.assertEqual(timeouts, [1700, 1590, 1310, 810] if include_fsdp2 else [1700, 1590, 810])
                self.assertEqual([command[command.index("--phase") + 1] for command in commands],
                                 phases)
                self.assertNotIn("torch.distributed.run", commands[0])
                self.assertNotIn("torch.distributed.run", commands[-1])
                for command in commands[1:-1]:
                    self.assertIn("torch.distributed.run", command)
                    self.assertIn("--nproc-per-node=2", command)
                    self.assertIn("--max-restarts=0", command)
                for command in commands[:-1]:
                    self.assertNotIn("--include-fsdp2", command)
                self.assertEqual("--include-fsdp2" in commands[-1], include_fsdp2)
                scratch_paths = {command[command.index("--scratch-dir") + 1] for command in commands}
                self.assertEqual(len(scratch_paths), 1)
                self.assertFalse(Path(scratch_paths.pop()).exists())
                self.assertFalse(list(volume.rglob("*.bin")))
                self.volume.commit.assert_called_once_with()
                self.volume.reset_mock()
                result_strategy = "fsdp2" if include_fsdp2 else "ddp"
                launch_path = next((volume / "results" / result_strategy).glob("*/launch.json"))
                self.assertTrue(launch_path.parent.name.endswith("-numerics-2gpu"))
                launch = json.loads(launch_path.read_text())
                self.assertEqual(launch["status"], "complete")
                self.assertEqual(launch["validation_status"], result_status)
                self.assertEqual([phase["status"] for phase in launch["phases"]], ["complete"] * len(phases))

    def test_numerics_stops_on_failure_or_shared_deadline_and_commits_partial_artifacts(self):
        for config, outcome in ((config, outcome) for config in (self.launcher.CONFIG, FSDP_CONFIG)
                                for outcome in ("failure", "deadline")):
            with self.subTest(config=config, outcome=outcome), tempfile.TemporaryDirectory() as directory:
                repository = self.make_repository(Path(directory) / "repository", measured_steps=3, config=config)
                volume = Path(directory).resolve() / "volume"
                experiment = self.launcher.check_inputs(repository, config)
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
                    if outcome == "failure" and len(scratch_paths) == (3 if config == FSDP_CONFIG else 1):
                        raise subprocess.CalledProcessError(1, command)

                expected_error = subprocess.CalledProcessError if outcome == "failure" else TimeoutError
                include_fsdp2 = config == FSDP_CONFIG
                times = ([0, 10, 120, 900 if outcome == "failure" else 1711] if include_fsdp2
                         else [0, 10, 1711])
                expected_calls = (3 if outcome == "failure" else 2) if include_fsdp2 else 1
                with patch.object(self.launcher, "REMOTE_ROOT", repository), \
                        patch.object(self.launcher, "VOLUME_ROOT", volume), \
                        patch.object(self.launcher.time, "monotonic", side_effect=times), \
                        patch.object(self.launcher, "run_worker", side_effect=worker) as run:
                    with self.assertRaises(expected_error):
                        self.launcher.train_ddp(config, 2, "numerics", {})
                    self.assertEqual(run.call_count, expected_calls)
                self.assertFalse(scratch_paths[0].exists())
                self.volume.commit.assert_called_once_with()
                self.volume.reset_mock()
                result_strategy = "fsdp2" if include_fsdp2 else "ddp"
                launch_path = next((volume / "results" / result_strategy).glob("*/launch.json"))
                launch = json.loads(launch_path.read_text())
                self.assertEqual(launch["status"], "failed")
                self.assertEqual(launch["phases"][-1]["status"], "failed")
                self.assertEqual(launch["phases"][-1]["phase"], "fsdp2" if include_fsdp2 else
                                 "reference" if outcome == "failure" else "ddp")
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
            with patch.object(self.launcher, "VOLUME_ROOT", repository / "checkpoints"), \
                    self.assertRaisesRegex(FileNotFoundError, "cached model"):
                self.launcher.check_inputs(repository, config, remote=True)
            write_json(repository / self.launcher.BUNDLE / "splits/train/validation.json", ["task-0"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.launcher.check_inputs(repository, config)

    def test_new_model_alias_resolves_nvarc_cache_inside_the_volume_and_records_backend(self):
        self.load_launcher(environment={
            "DIFFUSION_ARC_MODAL_VOLUME": "diffusion-arc-training-v2",
            "DIFFUSION_ARC_MODAL_VOLUME_ROOT": "/workspace",
            "DIFFUSION_ARC_MODAL_MODEL_SUBDIRECTORY": "models",
        })
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(Path(directory) / "repository")
            volume = Path(directory).resolve() / "workspace"
            models = volume / "models"
            (models / "flux2-klein-base-4b").mkdir(parents=True)
            cached_nvarc = volume / ".cache/kagglehub/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
            cached_nvarc.mkdir(parents=True)
            (models / "nvarc-qwen3-4b-grids15-sft139").symlink_to(cached_nvarc, target_is_directory=True)

            def worker(command, *, timeout):
                experiment = self.launcher.check_inputs(repository, self.launcher.CONFIG, remote=True)
                self.assertEqual(experiment.training.models.nvarc_checkpoint, cached_nvarc)
                self.assertEqual(experiment.training.models.flux_checkpoint, models / "flux2-klein-base-4b")
                write_json(Path(command[-1]) / "summary.json", dict(event="complete", optimizer_steps=1))

            with patch.object(self.launcher, "REMOTE_ROOT", repository), \
                    patch.object(self.launcher, "VOLUME_ROOT", volume), \
                    patch.object(self.launcher, "run_worker", side_effect=worker) as run:
                self.launcher.train_ddp(self.launcher.CONFIG, 2, "smoke", {})
                run.assert_called_once()
                with self.assertRaises(ValueError):
                    self.launcher.check_inputs(repository, self.launcher.CONFIG)
            self.assertEqual((repository / "checkpoints").resolve(), models)
            launch = json.loads(next((volume / "results/ddp").glob("*/launch.json")).read_text())
            self.assertEqual(launch["model_volume"], dict(
                name="diffusion-arc-training-v2", mount_path=str(volume), model_subdirectory="models",
            ))

    def test_remote_preflight_rejects_a_model_alias_outside_the_mounted_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(Path(directory) / "repository")
            volume = Path(directory).resolve() / "volume"
            volume.mkdir()
            outside = Path(directory).resolve() / "outside-models"
            outside.mkdir()
            (volume / "nvarc-qwen3-4b-grids15-sft139").symlink_to(outside, target_is_directory=True)
            (volume / "flux2-klein-base-4b").mkdir()
            (repository / "checkpoints").symlink_to(volume, target_is_directory=True)
            with patch.object(self.launcher, "VOLUME_ROOT", volume), self.assertRaises(ValueError):
                self.launcher.check_inputs(repository, self.launcher.CONFIG, remote=True)

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
