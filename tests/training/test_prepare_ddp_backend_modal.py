"""Offline checks of the CPU model bridge, bounded streaming, and account separation."""

import asyncio
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

from tests.training.test_training_modal import _App, RecordingImage


ROOT = Path(__file__).resolve().parents[2]
CONFIG = b'{"_class_name": "Flux2Transformer2DModel"}'
WEIGHTS = b"original-base-weights"
SIGNED_URL = "https://storage.example/weights?signature=never-print-this"
SIGNED_URLS = (SIGNED_URL,)


class DownloadResponse:
    def __init__(self, data, status=200):
        self.data, self.status = data, status
        self.content = self
        self.iter_chunked = Mock(wraps=self.chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def chunks(self, size):
        for offset in range(0, len(self.data), size):
            yield self.data[offset:offset + size]


class DownloadSession:
    def __init__(self, responses):
        self.get = Mock(side_effect=responses)
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True


class BackendPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "transformer"
        self.volume = Mock()
        self.image = RecordingImage()
        self.volume_factory = Mock(return_value=self.volume)
        self.modal = SimpleNamespace(
            App=_App, is_local=lambda: True,
            Image=SimpleNamespace(debian_slim=Mock(return_value=self.image)),
            Volume=SimpleNamespace(from_name=self.volume_factory),
            Workspace=SimpleNamespace(from_context=Mock(return_value=Mock(name="workspace"))),
        )
        self.launcher = self.load_launcher()
        self.launcher.MIN_WEIGHT_BYTES = 8

    def load_launcher(self, environment=None):
        spec = importlib.util.spec_from_file_location(
            "_prepare_backend_test", ROOT / "scripts/cloud/prepare_ddp_backend_modal.py",
        )
        module = importlib.util.module_from_spec(spec)
        clean_environment = {key: value for key, value in os.environ.items()
                             if key not in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")}
        clean_environment.update(environment or {})
        with patch.dict(sys.modules, {"modal": self.modal}), \
                patch.dict(os.environ, clean_environment, clear=True):
            spec.loader.exec_module(module)
        return module

    def install(self, config=CONFIG, weights=WEIGHTS, expected_size=None):
        response = DownloadResponse(weights)
        self.response = response
        self.session = DownloadSession([response])
        with patch.object(self.launcher, "create_download_session", return_value=self.session):
            self.download = self.session.get
            return self.launcher.install_transformer(
                config, SIGNED_URLS, len(WEIGHTS) if expected_size is None else expected_size,
                directory=self.directory,
            )

    def test_streams_complete_weights_and_saves_sanitized_provenance(self):
        result = self.install()
        self.assertEqual((self.directory / self.launcher.WEIGHTS_NAME).read_bytes(), WEIGHTS)
        self.assertEqual((self.directory / "config.json").read_bytes(), CONFIG)
        self.assertFalse(list(self.directory.glob("*.partial")))
        self.assertEqual(result["status"], "transferred")
        self.assertEqual(result["source_revision"], self.launcher.SOURCE_REVISION)
        self.assertEqual(result["weight_bytes"], len(WEIGHTS))
        receipt = json.loads((self.directory / "base_transformer_transfer.json").read_text())
        self.assertEqual(receipt["source_path"],
                         "flux2-klein-base-4b/transformer/diffusion_pytorch_model.safetensors")
        self.assertNotIn(SIGNED_URL, json.dumps(result) + json.dumps(receipt))
        self.download.assert_called_once_with(SIGNED_URL)
        self.assertTrue(all(item.args == (self.launcher.CHUNK_BYTES,)
                            for item in self.response.iter_chunked.call_args_list))
        self.assertTrue(self.session.closed)

    def test_matching_existing_files_skip_the_download(self):
        self.install()
        with patch.object(self.launcher, "create_download_session") as download:
            result = self.launcher.install_transformer(
                CONFIG, (), len(WEIGHTS), directory=self.directory,
            )
        self.assertEqual(result["status"], "already_present")
        download.assert_not_called()

    def test_multiple_blocks_are_streamed_in_source_order(self):
        urls = (SIGNED_URL + "&part=1", SIGNED_URL + "&part=2")
        responses = [DownloadResponse(WEIGHTS[:8], status=206), DownloadResponse(WEIGHTS[8:])]
        session = DownloadSession(responses)
        with patch.object(self.launcher, "create_download_session", return_value=session):
            result = self.launcher.install_transformer(CONFIG, urls, len(WEIGHTS), directory=self.directory)
        self.assertEqual((self.directory / self.launcher.WEIGHTS_NAME).read_bytes(), WEIGHTS)
        self.assertEqual(session.get.call_args_list, [call(url) for url in urls])
        self.assertTrue(session.closed)
        self.assertNotIn("signature", json.dumps(result))

    def test_failure_in_second_block_removes_the_incomplete_transfer(self):
        response = DownloadResponse(WEIGHTS[:8])
        urls = (SIGNED_URL + "&part=1", SIGNED_URL + "&part=2")
        session = DownloadSession([response, OSError(urls[1])])
        with patch.object(self.launcher, "create_download_session", return_value=session), \
                self.assertRaises(RuntimeError) as error:
            self.launcher.install_transformer(CONFIG, urls, len(WEIGHTS), directory=self.directory)
        self.assertFalse((self.directory / self.launcher.WEIGHTS_NAME).exists())
        self.assertFalse(list(self.directory.glob("*.partial")))
        self.assertNotIn("signature", str(error.exception))
        self.assertIn("block 2/2", str(error.exception))
        self.assertIn("OSError", str(error.exception))
        self.assertIn(f"received 8/{len(WEIGHTS)} bytes", str(error.exception))
        self.assertTrue(session.closed)

    def test_http_error_reports_status_and_block_without_exposing_download_capabilities(self):
        session = DownloadSession([DownloadResponse(b"", status=403)])
        output = io.StringIO()
        with patch.object(self.launcher, "create_download_session", return_value=session), \
                patch("sys.stdout", output), self.assertRaises(RuntimeError) as error:
            self.launcher.install_transformer(CONFIG, SIGNED_URLS, len(WEIGHTS), directory=self.directory)
        self.assertIn("HTTP 403", str(error.exception))
        self.assertIn("block 1/1", str(error.exception))
        self.assertIn(f"received 0/{len(WEIGHTS)} bytes", str(error.exception))
        self.assertNotIn(SIGNED_URL, str(error.exception) + output.getvalue())
        self.assertFalse((self.directory / self.launcher.WEIGHTS_NAME).exists())
        self.assertFalse(list(self.directory.glob("*.partial")))
        self.assertTrue(session.closed)

    def test_existing_mismatches_are_preserved_without_a_download(self):
        self.directory.mkdir()
        config_path = self.directory / "config.json"
        weights_path = self.directory / self.launcher.WEIGHTS_NAME
        for config, weights in ((b'{"wrong": true}', WEIGHTS), (CONFIG, b"wrong"), (None, WEIGHTS)):
            config_path.unlink(missing_ok=True)
            if config is not None:
                config_path.write_bytes(config)
            weights_path.write_bytes(weights)
            with self.subTest(config=config, weights=weights), \
                    patch.object(self.launcher, "create_download_session") as download:
                with self.assertRaisesRegex(ValueError, "Existing target"):
                    self.launcher.install_transformer(CONFIG, SIGNED_URLS, len(WEIGHTS), directory=self.directory)
                download.assert_not_called()
                self.assertEqual(weights_path.read_bytes(), weights)
                if config is not None:
                    self.assertEqual(config_path.read_bytes(), config)

    def test_short_long_and_failed_downloads_publish_no_final_weights_or_urls(self):
        for content in (WEIGHTS[:-1], WEIGHTS + b"extra"):
            with self.subTest(content=content), self.assertRaisesRegex(RuntimeError, "not published"):
                self.install(weights=content)
            self.assertFalse((self.directory / self.launcher.WEIGHTS_NAME).exists())
            self.assertFalse(list(self.directory.glob("*.partial")))
        session = DownloadSession([OSError(SIGNED_URL)])
        with patch.object(self.launcher, "create_download_session", return_value=session), \
                self.assertRaises(RuntimeError) as error:
            self.launcher.install_transformer(CONFIG, SIGNED_URLS, len(WEIGHTS), directory=self.directory)
        self.assertNotIn(SIGNED_URL, str(error.exception))
        self.assertTrue(error.exception.__suppress_context__)

    def test_cleanup_replaces_only_this_scripts_partial_file(self):
        self.directory.mkdir()
        (self.directory / (self.launcher.WEIGHTS_NAME + ".partial")).write_bytes(b"stale")
        unrelated = self.directory / "other-model.partial"
        unrelated.write_bytes(b"keep")
        self.install()
        self.assertEqual(unrelated.read_bytes(), b"keep")

    def test_invalid_source_metadata_is_rejected_before_http_or_target_writes(self):
        for config, size in ((b"", len(WEIGHTS)), (b"invalid", len(WEIGHTS)),
                             (b"[]", len(WEIGHTS)), (b"{}", len(WEIGHTS)),
                             (CONFIG, 0), (CONFIG, 7), (CONFIG, True), (CONFIG, 20.0)):
            with self.subTest(config=config, size=size), \
                    patch.object(self.launcher, "create_download_session") as download:
                with self.assertRaises(ValueError):
                    self.launcher.install_transformer(config, SIGNED_URLS, size, directory=self.directory)
                download.assert_not_called()
                self.assertFalse(self.directory.exists())

    def test_worker_has_bounded_cpu_resources_and_commits_only_after_success(self):
        self.volume_factory.assert_called_once_with(
            "diffusion-arc-training-v2", environment_name="main", create_if_missing=False,
        )
        options = self.launcher.app.function_options
        self.assertEqual(options["cpu"], (2.0, 2.0))
        self.assertEqual(options["memory"], (4096, 4096))
        self.assertEqual(options["timeout"], 900)
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["volumes"], {"/workspace": self.volume})
        self.assertNotIn("gpu", options)
        self.assertNotIn("secrets", options)
        with patch.object(self.launcher, "install_transformer", return_value={"status": "transferred"}):
            self.assertEqual(self.launcher.prepare_backend(CONFIG, SIGNED_URLS, len(WEIGHTS)),
                             {"status": "transferred"})
        self.volume.commit.assert_called_once_with()
        self.volume.commit.reset_mock()
        with patch.object(self.launcher, "install_transformer", side_effect=ValueError("mismatch")), \
                self.assertRaises(ValueError):
            self.launcher.prepare_backend(CONFIG, SIGNED_URLS, len(WEIGHTS))
        self.volume.commit.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("modal_proto"), "Modal SDK required for RPC contracts")
    def test_source_client_reads_the_old_profile_explicitly_and_closes(self):
        from modal_proto import api_pb2

        settings = {"server_url": "https://api.modal.com", "token_id": "local-only-id",
                    "token_secret": "local-only-secret"}
        config = SimpleNamespace(get=Mock(side_effect=lambda key, **kwargs: settings[key]))
        stub = SimpleNamespace(
            VolumeGetOrCreate=AsyncMock(return_value=SimpleNamespace(volume_id="vo-source")),
            VolumeGetFile=AsyncMock(return_value=SimpleNamespace(data=CONFIG, size=len(CONFIG))),
            VolumeGetFile2=AsyncMock(return_value=SimpleNamespace(
                size=len(WEIGHTS), start=0, len=len(WEIGHTS), get_urls=SIGNED_URLS,
            )),
        )
        client = SimpleNamespace(stub=stub)
        context = SimpleNamespace(__aenter__=AsyncMock(return_value=client), __aexit__=AsyncMock())
        class ClientContext:
            async def __aenter__(self):
                return await context.__aenter__()
            async def __aexit__(self, *args):
                await context.__aexit__(*args)
        factory = Mock(return_value=ClientContext())
        with patch.dict(sys.modules, {
            "modal.config": SimpleNamespace(Config=Mock(return_value=config)),
            "modal.client": SimpleNamespace(_Client=factory),
        }), patch.dict(os.environ, {}, clear=True):
            result = asyncio.run(self.launcher.load_source_model_metadata())
        self.assertEqual(result, (CONFIG, SIGNED_URLS, len(WEIGHTS)))
        self.assertNotIn("local-only", repr(result))
        self.assertEqual(config.get.call_args_list, [
            call(key, profile="joshua-kauling01", use_env=False)
            for key in ("server_url", "token_id", "token_secret")
        ])
        factory.assert_called_once_with("https://api.modal.com", api_pb2.CLIENT_TYPE_CLIENT,
                                        ("local-only-id", "local-only-secret"))
        request = stub.VolumeGetOrCreate.call_args.args[0]
        self.assertEqual(request.environment_name, "diffusion-arc")
        self.assertEqual(request.object_creation_type, api_pb2.OBJECT_CREATION_TYPE_UNSPECIFIED)
        self.assertEqual(stub.VolumeGetFile2.call_args.args[0].path,
                         "flux2-klein-base-4b/transformer/" + self.launcher.WEIGHTS_NAME)
        context.__aexit__.assert_awaited_once()

    def test_entrypoint_forwards_only_metadata_and_prints_no_signed_url(self):
        workspace = SimpleNamespace(name="joshuakauling", hydrate=Mock())
        self.modal.Workspace.from_context.return_value = workspace
        remote = Mock(return_value={"status": "transferred", "weight_bytes": len(WEIGHTS)})
        output = io.StringIO()
        with patch.object(self.launcher, "load_source_model_metadata",
                          AsyncMock(return_value=(CONFIG, SIGNED_URLS, len(WEIGHTS)))), \
                patch.object(self.launcher.prepare_backend, "remote", remote, create=True), \
                patch.dict(os.environ, {}, clear=True), patch("sys.stdout", output):
            self.launcher.main()
        remote.assert_called_once_with(CONFIG, SIGNED_URLS, len(WEIGHTS))
        self.assertNotIn(SIGNED_URL, output.getvalue())
        self.assertNotIn("signature", output.getvalue())

    def test_environment_tokens_fail_before_loading_or_dispatch(self):
        for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "override --profile"):
                self.load_launcher({key: "never-print"})
        self.load_launcher({"MODAL_PROFILE": "joshuakauling"})
        self.modal.Workspace.from_context.return_value = SimpleNamespace(name="wrong", hydrate=Mock())
        with patch.object(self.launcher, "load_source_model_metadata") as metadata, \
                patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "workspace"):
            self.launcher.main()
        metadata.assert_not_called()


if __name__ == "__main__":
    unittest.main()
