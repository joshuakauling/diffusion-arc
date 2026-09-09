"""Offline coverage of image reuse, provenance, persistence and lifecycle cancellation."""

import asyncio
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[2]
MODULE = "scripts.diagnostics.probe_flux_sequence_capacity"
CONFIG = "configs/diagnostics/flux_sequence_capacity.json"
IMAGE_ID = "im-R5CncDboPfMQryWnq0UZn4"


class _Image:
    object_id = IMAGE_ID

    def __init__(self):
        self.mounts = []

    def add_local_dir(self, path, **options):
        self.mounts.append(("directory", path, options))
        return self

    def add_local_file(self, path, **options):
        self.mounts.append(("file", path, options))
        return self


class _App:
    app_id = "ap-bounded-test"

    def __init__(self, name):
        self.name = name

    def function(self, **options):
        self.function_options = options

        def decorate(function):
            function.spawn = SimpleNamespace(aio=AsyncMock())
            return function

        return decorate

    def local_entrypoint(self):
        return lambda function: function


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.image = _Image()
        self.volume = SimpleNamespace(commit=SimpleNamespace(aio=AsyncMock()))
        self.modal = SimpleNamespace(App=_App, is_local=lambda: True, __version__="test",
            Image=SimpleNamespace(from_id=Mock(return_value=self.image)),
            Volume=SimpleNamespace(from_name=Mock(return_value=self.volume)))
        source = ROOT / "scripts/cloud/probe_flux_sequence_capacity_modal.py"
        spec = importlib.util.spec_from_file_location("_sequence_capacity_launcher_test", source)
        self.launcher = importlib.util.module_from_spec(spec)
        self.paths = patch.object(sys, "path", list(sys.path))
        self.paths.start()
        self.addCleanup(self.paths.stop)
        with patch.dict(sys.modules, {"modal": self.modal}):
            spec.loader.exec_module(self.launcher)

    def test_existing_image_is_reused_with_mounts_and_no_build_methods(self):
        self.assertEqual(self.launcher.app.name, "diffusion-arc-flux-sequence-capacity")
        self.modal.Image.from_id.assert_called_once_with(IMAGE_ID)
        self.assertEqual([str(path.relative_to(ROOT)) for _, path, _ in self.image.mounts],
                         ["src", "scripts", "configs", "pyproject.toml", "data/benchmarks/single_gpu_b0"])
        self.assertTrue(all(options["copy"] is False for _, _, options in self.image.mounts))
        self.modal.Volume.from_name.assert_called_once_with("diffusion-arc-models",
            environment_name="diffusion-arc", create_if_missing=False)

    def test_resources_cap_usage_and_avoid_premium_placement_or_retries(self):
        options = self.launcher.app.function_options
        self.assertEqual(options["gpu"], "H200")
        self.assertEqual(options["cpu"], (8.0, 8.0))
        self.assertEqual(options["memory"], (65536, 65536))
        self.assertEqual((options["timeout"], options["startup_timeout"]), (390, 60))
        self.assertEqual((options["max_containers"], options["min_containers"], options["buffer_containers"]), (1, 0, 0))
        self.assertEqual(options["retries"], 0)
        self.assertIs(options["nonpreemptible"], False)
        self.assertNotIn("region", options)
        self.assertLessEqual(options["scaledown_window"], 2)
        self.assertEqual(options["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(options["env"]["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(self.launcher.LIFECYCLE_SECONDS, 480)
        self.assertEqual(self.launcher.CLEANUP_SECONDS, 30)
        self.assertEqual(self.launcher.MAX_WORKER_SECONDS, 330)
        self.assertEqual(self.launcher.FINAL_COMMIT_SECONDS, 60)
        self.assertLessEqual(options["startup_timeout"] + options["timeout"] + options["scaledown_window"],
                             self.launcher.LIFECYCLE_SECONDS)
        self.assertAlmostEqual(self.launcher.BUDGET["maximum_list_price_estimate_usd"], 0.7237824)
        self.assertEqual(self.launcher.BUDGET["planned_ceiling_usd"], 1.00)

    def test_original_budget_includes_the_previous_capacity_attempt(self):
        budget = self.launcher.BUDGET
        self.assertEqual(budget["prior_attempt"]["run_id"], "20260909T061831725274Z")
        self.assertEqual(budget["prior_attempt"]["lifecycle_seconds"], 121)
        self.assertAlmostEqual(budget["prior_attempt"]["list_price_estimate_usd"], 0.18245348)
        self.assertEqual(budget["lifecycle_seconds"], 480)
        self.assertEqual(budget["cumulative_lifecycle_seconds"], 601)
        self.assertEqual(budget["approved_cumulative_seconds"], 660)
        self.assertLessEqual(budget["cumulative_lifecycle_seconds"], budget["approved_cumulative_seconds"])
        self.assertAlmostEqual(budget["cumulative_maximum_list_price_estimate_usd"], 0.90623588)
        self.assertAlmostEqual(budget["maximum_list_price_estimate_usd"] +
                               budget["prior_attempt"]["list_price_estimate_usd"],
                               budget["cumulative_maximum_list_price_estimate_usd"])
        self.assertLessEqual(budget["cumulative_maximum_list_price_estimate_usd"],
                             budget["planned_ceiling_usd"])

    def _policy(self, repository):
        return dict(config_paths=(repository / CONFIG,),
            checkpoint_paths=(repository / "checkpoints/model",), max_duration_seconds=330,
            training=SimpleNamespace(data=SimpleNamespace(
                split_directory=repository / "data/benchmarks/single_gpu_b0/splits/train")))

    def test_local_preflight_rejects_escaping_paths_and_over_budget_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            valid = self._policy(repository)
            variants = [dict(config_paths=(Path("/tmp/outside.json"),)),
                        dict(checkpoint_paths=(Path("/tmp/outside-model"),)),
                        dict(max_duration_seconds=331), dict(max_duration_seconds=True),
                        dict(max_duration_seconds=0), dict(training=SimpleNamespace(data=SimpleNamespace(
                            split_directory=repository / "data/other/splits/train")))]
            for change in variants:
                with self.subTest(change=change), patch.dict(sys.modules, {MODULE: SimpleNamespace(
                        load_probe_config=Mock(return_value=SimpleNamespace(**{**valid, **change})))}):
                    with self.assertRaises(ValueError):
                        self.launcher._validate_layout(repository, CONFIG,
                            checkpoint_root=repository / "checkpoints", require_checkpoints=False)

    def test_remote_preflight_requires_cached_checkpoint_without_downloading(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            policy = SimpleNamespace(**self._policy(repository))
            loader = Mock(return_value=policy)
            with patch.dict(sys.modules, {MODULE: SimpleNamespace(load_probe_config=loader)}):
                with self.assertRaisesRegex(FileNotFoundError, "checkpoint directory"):
                    self.launcher._validate_layout(repository, CONFIG,
                        checkpoint_root=repository / "checkpoints", require_checkpoints=True)
                checkpoint = policy.checkpoint_paths[0]
                checkpoint.mkdir(parents=True)
                path, actual = self.launcher._validate_layout(repository, CONFIG,
                    checkpoint_root=repository / "checkpoints", require_checkpoints=True)
            loader.assert_called_with(repository / CONFIG, require_checkpoints=True)
            self.assertEqual(path, repository / CONFIG)
            self.assertIs(actual, policy)

    def test_launch_sends_source_config_data_image_and_run_identity(self):
        probe = SimpleNamespace(source_metadata=Mock(return_value={"source_sha256": {"source": "sha"}}),
            config_hashes=Mock(return_value={"config": "sha"}), dataset_hashes=Mock(return_value={"data": "sha"}))
        with patch.dict(sys.modules, {MODULE: probe}), \
                patch.object(self.launcher, "_validate_layout", return_value=(ROOT / CONFIG, object())), \
                patch.object(self.launcher, "_bounded_remote_call", new_callable=AsyncMock) as invoke:
            self.launcher.main()
        config, metadata = invoke.call_args.args
        self.assertEqual(config, CONFIG)
        self.assertEqual(metadata["source_sha256"], {"source": "sha"})
        self.assertEqual(metadata["config_sha256"], {"config": "sha"})
        self.assertEqual(metadata["dataset_sha256"], {"data": "sha"})
        self.assertEqual(metadata["image_id"], IMAGE_ID)
        self.assertEqual(metadata["modal_app_id"], "ap-bounded-test")
        self.assertEqual(metadata["budget"], self.launcher.BUDGET)
        self.assertRegex(metadata["run_id"], r"^\d{8}T\d{12}Z$")

    def test_changed_image_is_rejected_before_submission(self):
        self.image.object_id = "im-other"
        with patch.dict(sys.modules, {MODULE: SimpleNamespace(config_hashes=Mock(), dataset_hashes=Mock(), source_metadata=Mock())}), \
                patch.object(self.launcher, "_validate_layout", return_value=(ROOT / CONFIG, object())), \
                patch.object(self.launcher, "_bounded_remote_call", new_callable=AsyncMock) as invoke:
            with self.assertRaisesRegex(RuntimeError, "image ID"):
                self.launcher.main()
        invoke.assert_not_called()

    def test_progress_never_commits_and_final_commit_occurs_once_on_success_and_failure(self):
        for status in ("complete", "error"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                repository, volume = Path(directory) / "repository", Path(directory) / "volume"
                repository.mkdir()
                volume.mkdir()
                observed = []

                def run_probe(args, **options):
                    self.assertEqual(args.config, repository / CONFIG)
                    self.assertEqual(args.output_dir,
                        volume / "results/flux-sequence-capacity/20260909T010203123456Z")
                    self.assertEqual(options["metadata"]["image_id"], IMAGE_ID)
                    self.assertEqual(options["metadata"]["modal_resources"], self.launcher.RESOURCES)
                    self.assertEqual(options["metadata"]["budget"], self.launcher.BUDGET)
                    for _ in range(10):
                        options["on_progress"]()
                    observed.append(self.volume.commit.aio.await_count)
                    return {"status": status}

                self.volume.commit.aio.reset_mock()
                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                        patch.object(self.launcher, "VOLUME_ROOT", volume), \
                        patch.object(self.launcher, "_validate_layout", return_value=(repository / CONFIG, object())), \
                        patch.dict(sys.modules, {MODULE: SimpleNamespace(run_probe=run_probe)}):
                    if status == "complete":
                        destination = self.launcher.probe_flux_sequence_capacity(CONFIG,
                            {"image_id": IMAGE_ID, "run_id": "20260909T010203123456Z"})
                        self.assertEqual(destination,
                            str(volume / "results/flux-sequence-capacity/20260909T010203123456Z"))
                    else:
                        with self.assertRaisesRegex(RuntimeError, "sequence capacity error"):
                            self.launcher.probe_flux_sequence_capacity(CONFIG,
                                {"image_id": IMAGE_ID, "run_id": "20260909T010203123456Z"})
                self.assertEqual(observed, [0])
                self.volume.commit.aio.assert_awaited_once()

    def test_remote_metadata_is_validated_before_work_or_filesystem_mutation(self):
        probe = Mock()
        invalid = [dict(image_id="im-other", run_id="20260909T010203123456Z"),
                   dict(image_id=IMAGE_ID, run_id="../outside"),
                   dict(image_id=IMAGE_ID, run_id=123)]
        with patch.dict(sys.modules, {MODULE: SimpleNamespace(run_probe=probe)}), \
                patch.object(self.launcher, "_validate_layout") as validate:
            for metadata in invalid:
                with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                    self.launcher.probe_flux_sequence_capacity(CONFIG, metadata)
        validate.assert_not_called()
        probe.assert_not_called()
        self.volume.commit.aio.assert_not_called()

    def test_existing_run_directory_prevents_repeating_persisted_work(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, volume = Path(directory) / "repository", Path(directory) / "volume"
            repository.mkdir()
            run_id = "20260909T010203123456Z"
            (volume / "results/flux-sequence-capacity" / run_id).mkdir(parents=True)
            probe = Mock()
            with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                    patch.object(self.launcher, "VOLUME_ROOT", volume), \
                    patch.object(self.launcher, "_validate_layout", return_value=(repository / CONFIG, object())), \
                    patch.dict(sys.modules, {MODULE: SimpleNamespace(run_probe=probe)}):
                with self.assertRaisesRegex(RuntimeError, "refusing to repeat"):
                    self.launcher.probe_flux_sequence_capacity(CONFIG, {"image_id": IMAGE_ID, "run_id": run_id})
            probe.assert_not_called()

    def _call(self, *, result="result-path", error=None):
        call = SimpleNamespace(get=SimpleNamespace(aio=AsyncMock(return_value=result, side_effect=error)),
            cancel=SimpleNamespace(aio=AsyncMock()))
        self.launcher.probe_flux_sequence_capacity.spawn.aio.return_value = call
        return call

    def test_submission_and_result_share_one_deadline(self):
        call = self._call()
        with patch.object(self.launcher, "_stop_app", new_callable=AsyncMock) as stop:
            result = asyncio.run(self.launcher._bounded_remote_call(CONFIG, {"modal_app_id": "ap-one"}))
        self.assertEqual(result, "result-path")
        self.launcher.probe_flux_sequence_capacity.spawn.aio.assert_awaited_once()
        self.assertGreater(call.get.aio.call_args.kwargs["timeout"], 449)
        self.assertLessEqual(call.get.aio.call_args.kwargs["timeout"], 450)
        call.cancel.aio.assert_not_called()
        stop.assert_not_called()

    def test_result_timeout_cancels_containers_and_stops_only_submitted_app(self):
        call = self._call(error=TimeoutError("deadline"))
        with patch.object(self.launcher, "_stop_app", new_callable=AsyncMock) as stop:
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                asyncio.run(self.launcher._bounded_remote_call(CONFIG, {"modal_app_id": "ap-one"}))
        call.cancel.aio.assert_awaited_once_with(terminate_containers=True)
        stop.assert_awaited_once_with("ap-one")
        self.launcher.probe_flux_sequence_capacity.spawn.aio.assert_awaited_once()

    def test_hanging_submission_is_also_bounded_and_stops_app(self):
        async def never_submit(*_args):
            await asyncio.Event().wait()

        self.launcher.probe_flux_sequence_capacity.spawn.aio.side_effect = never_submit
        with patch.object(self.launcher, "LIFECYCLE_SECONDS", 0.01), \
                patch.object(self.launcher, "CLEANUP_SECONDS", 0), \
                patch.object(self.launcher, "_stop_app", new_callable=AsyncMock) as stop:
            with self.assertRaises(TimeoutError):
                asyncio.run(self.launcher._bounded_remote_call(CONFIG, {"modal_app_id": "ap-one"}))
        stop.assert_awaited_once_with("ap-one")

    def test_failed_call_cancellation_still_stops_app_and_preserves_original_error(self):
        call = self._call(error=RuntimeError("worker failed"))
        call.cancel.aio.side_effect = OSError("cancellation unavailable")
        with patch.object(self.launcher, "_stop_app", new_callable=AsyncMock) as stop:
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                asyncio.run(self.launcher._bounded_remote_call(CONFIG, {"modal_app_id": "ap-one"}))
        stop.assert_awaited_once_with("ap-one")

    def test_final_commit_has_an_independent_timeout(self):
        async def never_commit():
            await asyncio.Event().wait()

        self.volume.commit.aio.side_effect = never_commit
        with patch.object(self.launcher, "FINAL_COMMIT_SECONDS", 0.01):
            with self.assertRaises(TimeoutError):
                asyncio.run(self.launcher._commit_final())

    def test_stop_uses_explicit_environment_and_noninteractive_app_id(self):
        process = SimpleNamespace(communicate=AsyncMock(return_value=(b"Stopped", b"")), returncode=0)
        with patch.object(self.launcher.asyncio, "create_subprocess_exec", new_callable=AsyncMock,
                          return_value=process) as spawn:
            asyncio.run(self.launcher._stop_app("ap-one"))
        self.assertEqual(spawn.call_args.args,
            (sys.executable, "-m", "modal", "app", "stop", "ap-one", "--env", "diffusion-arc", "--yes"))


if __name__ == "__main__":
    unittest.main()
