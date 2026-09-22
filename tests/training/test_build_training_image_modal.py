"""Local checks for build uploads, registry inputs and paid VM cleanup; no cloud calls."""

from contextlib import nullcontext
import gzip
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.cloud import build_training_image_modal as launcher
from scripts.cloud import build_image_models as models
from scripts.cloud import train_runpod


HAS_MODAL = importlib.util.find_spec("modal") is not None


class ModelBundleTests(unittest.TestCase):
    def test_downloaded_bundle_passes_pod_validation_without_caches_or_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "kaggle-source"
            source.mkdir()
            sizes = {name: 3 for name in models.NVARC_WEIGHT_SIZES}
            for name in sizes:
                (source / name).write_bytes(b"123")
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                (source / name).write_text("{}")
            (source / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {str(i): name for i, name in enumerate(sizes)},
            }))
            (source / ".cache").mkdir()
            (source / ".cache/private").write_text("must-not-persist")

            def download_flux(repo, *, local_dir, revision, **kwargs):
                self.assertEqual(repo, train_runpod.FLUX_REPO)
                self.assertEqual(revision, train_runpod.FLUX_REVISION)
                for component in ("transformer", "vae"):
                    directory = local_dir / component
                    directory.mkdir(parents=True)
                    (directory / "config.json").write_text("{}")
                    (directory / "diffusion_pytorch_model.safetensors").write_bytes(b"123")
                (local_dir / ".cache").mkdir()
                (local_dir / ".cache/private").write_text("must-not-persist")

            destination = root / "models"
            kaggle = SimpleNamespace(model_download=Mock(return_value=str(source)))
            with patch.dict("sys.modules", {"kagglehub": kaggle}), \
                 patch.dict("os.environ", {"HF_TOKEN": "must-not-persist"}), \
                 patch("huggingface_hub.snapshot_download", side_effect=download_flux), \
                 patch.object(models, "NVARC_WEIGHT_SIZES", sizes), \
                 patch.object(train_runpod, "NVARC_WEIGHT_SIZES", sizes), \
                 patch.object(train_runpod, "EMBEDDED_MODELS", destination), \
                 patch("sys.stdout", io.StringIO()):
                manifest = models.build_models(destination)
                paths = train_runpod.embedded_model_paths()
            kaggle.model_download.assert_called_once_with(train_runpod.NVARC_HANDLE)
            self.assertEqual(paths["flux_checkpoint"], destination / "flux2-klein-base-4b")
            self.assertNotIn("must-not-persist", json.dumps(manifest))
            self.assertFalse(list(destination.rglob(".cache")))
            for name in manifest["nvarc"]["files"]:
                self.assertNotIn("private", name)


class CommandOutputTests(unittest.TestCase):
    def test_both_streams_are_saved_and_arguments_remain_literal(self):
        log, output = io.StringIO(), io.StringIO()
        processes = []

        def local_exec(*command, **options):
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, bufsize=options["bufsize"])
            processes.append(process)
            self.addCleanup(Path(command[4]).unlink, missing_ok=True)
            return process

        sandbox = SimpleNamespace(exec=local_exec, filesystem=SimpleNamespace(read_text=lambda path: Path(path).read_text()))
        literal = '$(printf expanded); "quoted"'
        try:
            with patch("sys.stdout", output):
                launcher.run_command(sandbox, "sh", "-c",
                                     'printf "%s\\n" "$1"; printf "build warning\\n" >&2',
                                     "example", literal, log=log, timeout=10)
            self.assertEqual(log.getvalue(), literal + "\nbuild warning\n")
            self.assertEqual(output.getvalue(), log.getvalue())
            with patch("sys.stdout", output), self.assertRaisesRegex(RuntimeError, "exit code 7"):
                launcher.run_command(sandbox, "sh", "-c", 'echo failed >&2; exit 7',
                                     log=log, timeout=10)
            self.assertTrue(log.getvalue().endswith("failed\n"))
        finally:
            for process in processes:
                process.wait(timeout=10)
                process.stdout.close()
                process.stderr.close()

    def test_finished_process_does_not_wait_for_modal_log_stream_eof(self):
        process = Mock(returncode=0)
        process.poll.return_value = 0
        sandbox = Mock()
        sandbox.exec.return_value = process
        sandbox.filesystem.read_text.return_value = "Published successfully\n"
        log = io.StringIO()
        with patch("sys.stdout", io.StringIO()):
            launcher.run_command(sandbox, "docker", "build", log=log, timeout=10)
        self.assertEqual(log.getvalue(), "Published successfully\n")
        process.stdout.assert_not_called()
        process.wait.assert_called_once()


class ImageReferenceTests(unittest.TestCase):
    def test_explicit_registry_and_version_are_required(self):
        self.assertEqual(launcher.registry_for_image("ghcr.io/owner/repo:run-v1"), "ghcr.io")
        self.assertEqual(launcher.registry_for_image("registry.example:5000/repo:v1"), "registry.example:5000")
        for image in ("repo", "owner/repo:v1", "ghcr.io/owner/repo", "ghcr.io/owner/repo:latest",
                      "ghcr.io/owner/repo:buildcache", "https://ghcr.io/owner/repo:v1",
                      "ghcr.io/owner/repo:v1;echo-secret", "ghcr.io/owner/repo:v1\n"):
            with self.subTest(image=image), self.assertRaises(ValueError):
                launcher.registry_for_image(image)

    def test_invalid_build_input_fails_before_preparation_or_cloud_work(self):
        with patch.object(launcher, "build_context") as prepare, patch.object(launcher, "execute") as execute:
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                launcher.main(["build"])
            with self.assertRaises(ValueError):
                launcher.main(["build", "--image", "bad-reference"])
        prepare.assert_not_called()
        execute.assert_not_called()

    def test_smoke_dry_run_is_local_and_bounded(self):
        output = io.StringIO()
        with patch.object(launcher, "execute") as execute, patch("sys.stdout", output):
            self.assertEqual(launcher.main(["smoke", "--dry-run"]), 0)
        execute.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["resources"], {"cpu": [1.0, 1.0], "memory": 2048, "timeout": 300})
        self.assertIsNone(plan["registry_secret"])


@unittest.skipUnless(HAS_MODAL, "Modal SDK is needed for its Docker ignore matcher")
class BuildContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / ".dockerignore").write_text((launcher.ROOT / ".dockerignore").read_text())
        for name in ("Dockerfile", "scripts/cloud/build_image_models.py", "pyproject.toml", "README.md", "scripts/train_single_gpu.py",
                     "src/diffusion_arc/__init__.py", "configs/run.json", "src/uncommitted.py",
                     "src/.env", "src/.env.production", "src/model.safetensors", "src/state.pt",
                     "src/__pycache__/module.pyc", "data/private.json", "checkpoints/weights.pth"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("example\n")

    def test_archive_filters_before_upload_and_hashes_actual_uncommitted_content(self):
        data, manifest = launcher.build_context(self.root)
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(data))) as archive:
            names = set(archive.getnames())
            self.assertEqual(names, set(manifest["files"]))
            self.assertEqual(names, launcher.CONTEXT_FILES)
            self.assertEqual(archive.extractfile("scripts/cloud/build_image_models.py").read(), b"example\n")
            for name in ("src/.env", "src/.env.production", "src/model.safetensors", "src/state.pt",
                         "src/__pycache__/module.pyc", "data/private.json", "checkpoints/weights.pth", ".git/config"):
                self.assertNotIn(name, names)
        self.assertEqual(launcher.build_context(self.root)[0], data)
        (self.root / "scripts/cloud/build_image_models.py").write_text("changed\n")
        self.assertNotEqual(launcher.build_context(self.root)[1]["sha256"], manifest["sha256"])

    def test_symlink_cannot_import_a_file_outside_the_context(self):
        script = self.root / "scripts/cloud/build_image_models.py"
        script.unlink()
        script.symlink_to(self.root / "data/private.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            launcher.build_context(self.root)

    def test_missing_required_input_is_rejected(self):
        (self.root / "Dockerfile").unlink()
        with self.assertRaisesRegex(ValueError, "Dockerfile"):
            launcher.build_context(self.root)


class SandboxLifecycleTests(unittest.TestCase):
    def fake_modal(self, sandbox):
        builder = Mock()
        builder.env.return_value = builder
        builder.apt_install.return_value = builder
        builder.run_commands.return_value = builder
        return SimpleNamespace(
            Image=SimpleNamespace(from_registry=Mock(return_value=builder)),
            App=SimpleNamespace(lookup=Mock()), Sandbox=SimpleNamespace(create=Mock(return_value=sandbox)),
            Secret=SimpleNamespace(from_name=Mock(), from_dict=Mock()), enable_output=nullcontext, __version__="test",
        )

    def test_smoke_failure_and_interruption_terminate_vm_without_publishing(self):
        for error in (RuntimeError("failed smoke"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temp:
                sandbox = Mock(object_id="sb-test")
                modal = self.fake_modal(sandbox)
                plan = {"command": "smoke", "status": "prepared"}
                with patch.dict("sys.modules", {"modal": modal}), \
                     patch.object(launcher, "run_command", side_effect=error), \
                     patch("sys.stdout", io.StringIO()), self.assertRaises(type(error)):
                    launcher.execute(plan, None, Path(temp))
                sandbox.terminate.assert_called_once_with(wait=True)
                modal.Secret.from_name.assert_not_called()
                self.assertEqual(json.loads((Path(temp) / "build.json").read_text())["status"], "failed")

    def test_build_auth_is_scoped_to_login_and_success_records_verified_digest(self):
        sandbox = Mock(object_id="sb-test")
        digest = "sha256:" + "a" * 64
        sandbox.filesystem.read_text.side_effect = [json.dumps({"containerimage.digest": digest}), "{}", "{}",
                                                  json.dumps({key: {"files": {"weights": {"bytes": 123}}}
                                                              for key in ("nvarc", "flux")})]
        modal = self.fake_modal(sandbox)
        plan = {"command": "build", "registry_secret": "test-registry", "registry": "ghcr.io",
                "image": "ghcr.io/owner/repo:v1", "source_revision": "abc", "context": {"sha256": "def"}}
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"modal": modal}), \
             patch.object(launcher, "model_credentials", return_value={"HF_TOKEN": "never-record-this"}), \
             patch.object(launcher, "run_command") as run, patch("sys.stdout", io.StringIO()):
            launcher.execute(plan, b"context", Path(temp))
            result = json.loads((Path(temp) / "build.json").read_text())
        commands = [call.args[1:] for call in run.call_args_list]
        login = next(call for call in run.call_args_list if launcher.REGISTRY_LOGIN in call.args)
        self.assertEqual(login.kwargs["secrets"], [modal.Secret.from_name.return_value])
        for call in run.call_args_list:
            if "--secret" in call.args:
                self.assertEqual(call.kwargs["secrets"], [modal.Secret.from_dict.return_value])
                self.assertNotIn("never-record-this", str(call.args))
            elif call is not login:
                self.assertFalse(call.kwargs["secrets"])
        build = next(command for command in commands if "--push" in command)
        self.assertIn("--metadata-file", build)
        self.assertIn("type=registry,ref=ghcr.io/owner/repo:models-buildcache,mode=min", build)
        self.assertIn(("docker", "buildx", "imagetools", "inspect", "ghcr.io/owner/repo@" + digest), commands)
        self.assertEqual(result["model_bytes"], 246)
        self.assertNotIn("never-record-this", json.dumps(result))
        self.assertEqual(result["runpod_image"], "ghcr.io/owner/repo@" + digest)
        self.assertEqual(result["status"], "succeeded")
        options = modal.Sandbox.create.call_args.kwargs
        self.assertEqual(options["cpu"], (4.0, 4.0))
        self.assertEqual(options["memory"], 32768)
        self.assertEqual(options["timeout"], 7200)
        self.assertNotIn("gpu", options)
        self.assertNotIn("secrets", options)
        sandbox.terminate.assert_called_once_with(wait=True)

    def test_missing_registry_secret_fails_before_vm_allocation(self):
        sandbox = Mock()
        modal = self.fake_modal(sandbox)
        modal.Secret.from_name.return_value.hydrate.side_effect = RuntimeError("missing secret")
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"modal": modal}), \
             self.assertRaisesRegex(RuntimeError, "missing secret"):
            launcher.execute({"command": "build", "registry_secret": "missing"}, b"", Path(temp))
        modal.Sandbox.create.assert_not_called()

    def test_missing_model_credentials_fails_before_vm_allocation(self):
        modal = self.fake_modal(Mock())
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"modal": modal}), \
             patch.object(launcher, "model_credentials", side_effect=ValueError("missing model credentials")), \
             self.assertRaisesRegex(ValueError, "missing model credentials"):
            launcher.execute({"command": "build", "registry_secret": "present"}, b"", Path(temp))
        modal.Sandbox.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
