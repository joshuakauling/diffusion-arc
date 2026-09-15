"""CPU checks of validation split isolation, state residency and timing scope."""

import hashlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import benchmark_validation as benchmark
from scripts.data import prepare_arc_validation_benchmark as preparation
from diffusion_arc.training.run import validation
import torch


def fixture_archive(path):
    with tarfile.open(path, "w:gz") as archive:
        for index in range(400):
            pair = {"input": [[index % 10]], "output": [[(index + 1) % 10]]}
            payload = json.dumps({"train": [pair, pair], "test": [pair] * (1 + (index % 19 == 0))}).encode()
            info = tarfile.TarInfo(f"ARC-AGI-1.0.2/data/evaluation/{index:08x}.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidationBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "fixture.tar.gz"
        self.digest = fixture_archive(self.archive)
        self.bundle = self.root / "bundle"
        self.patchers = [patch.object(preparation, "ARCHIVE_SHA256", self.digest),
                         patch.object(validation, "ARC_AGI_1_ARCHIVE_SHA256", self.digest)]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.manifest = preparation.prepare(self.archive, self.bundle)

    def test_split_is_reproducible_excludes_test_payloads_and_counts_every_query(self):
        second = preparation.prepare(self.archive, self.root / "second")
        self.assertEqual(second, self.manifest)
        self.assertEqual(benchmark.validate_bundle(self.bundle), self.manifest)
        self.assertEqual(len(list((self.bundle / "tasks").glob("*.json"))), 100)
        for key in self.manifest["final_test_task_ids"]:
            self.assertFalse((self.bundle / "tasks" / f"{key}.json").exists())
        self.assertGreater(self.manifest["validation_output_count"], 100)
        with self.assertRaises(FileExistsError):
            preparation.prepare(self.archive, self.bundle)

    def test_unpinned_archive_and_changed_validation_payload_are_rejected(self):
        with patch.object(preparation, "ARCHIVE_SHA256", "wrong"), self.assertRaisesRegex(ValueError, "pinned"):
            preparation.prepare(self.archive, self.root / "wrong")
        key = self.manifest["validation_task_ids"][0]
        (self.bundle / "tasks" / f"{key}.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "payload changed"):
            benchmark.validate_bundle(self.bundle)

    def test_shared_bundle_preflight_enforces_requested_seed_and_hash_inventory(self):
        with self.assertRaisesRegex(ValueError, "split seed"):
            validation.validate_validation_bundle(self.bundle, split_seed=168)
        manifest = dict(self.manifest, task_sha256=dict(self.manifest["task_sha256"]))
        manifest["task_sha256"].pop(self.manifest["validation_task_ids"][0])
        (self.bundle / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "hash inventory"):
            validation.validate_validation_bundle(self.bundle)

    def test_verified_local_directory_produces_same_split_without_archive_download(self):
        evaluation = self.root / "evaluation"
        evaluation.mkdir()
        with tarfile.open(self.archive) as archive:
            for member in archive:
                (evaluation / Path(member.name).name).write_bytes(archive.extractfile(member).read())
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in evaluation.glob("*.json")}
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with patch.object(preparation, "TASK_FILES_SHA256", digest):
            local = preparation.prepare_directory(evaluation, self.root / "from-local")
            for key in ("validation_task_ids", "final_test_task_ids", "sampling_task_ids", "task_sha256"):
                self.assertEqual(local[key], self.manifest[key])
            benchmark.validate_bundle(self.root / "from-local")
            (evaluation / "00000000.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "baseline inventory"):
                preparation.prepare_directory(evaluation, self.root / "modified")

    def test_final_test_payloads_and_repartitioning_are_rejected(self):
        key = self.manifest["final_test_task_ids"][0]
        path = self.bundle / "tasks" / f"{key}.json"
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "only the 100"):
            benchmark.validate_bundle(self.bundle)
        path.unlink()
        manifest = dict(self.manifest, validation_task_ids=list(reversed(self.manifest["validation_task_ids"])))
        manifest["sampling_task_ids"] = manifest["validation_task_ids"][:32]
        (self.bundle / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "seeded split"):
            benchmark.validate_bundle(self.bundle)

    def test_tokenization_uses_all_queries_without_query_output_conditioning(self):
        tokenizer = SimpleNamespace(encode=lambda text, **kw: tuple(text.encode()))
        samples, grids = benchmark.tokenize_targets(self.bundle, self.manifest, tokenizer)
        self.assertEqual(len(samples), self.manifest["validation_output_count"])
        selected = self.manifest["validation_task_ids"][0]
        path = self.bundle / "tasks" / f"{selected}.json"
        payload = json.loads(path.read_text())
        payload["test"][0]["output"] = [[9, 9], [9, 9]]
        path.write_text(json.dumps(payload))
        changed_samples, changed_grids = benchmark.tokenize_targets(self.bundle, self.manifest, tokenizer)
        self.assertEqual(samples, changed_samples)
        self.assertNotEqual(grids, changed_grids)

    def test_long_prompts_fail_preflight_instead_of_dropping_or_truncating_tasks(self):
        tokenizer = SimpleNamespace(encode=lambda text, **kw: (1,) * 8401)
        with self.assertRaisesRegex(ValueError, "no truncation or replacement"):
            benchmark.tokenize_targets(self.bundle, self.manifest, tokenizer)

    def test_resident_adam_buffers_match_real_adam_initialization_without_weight_change(self):
        parameter = torch.nn.Parameter(torch.arange(5, dtype=torch.float32))
        reference = torch.nn.Parameter(parameter.detach().clone())
        optimizer = torch.optim.AdamW([parameter], lr=0.0, foreach=False)
        real = torch.optim.AdamW([reference], lr=0.0, foreach=False)
        before = parameter.detach().clone()
        benchmark.allocate_adam_state(optimizer)
        reference.grad = torch.zeros_like(reference)
        real.step()
        torch.testing.assert_close(parameter, before, atol=0, rtol=0)
        self.assertIsNone(parameter.grad)
        for field in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(optimizer.state[parameter][field], real.state[reference][field])
        self.assertEqual(optimizer.state[parameter]["step"].item(), 0)
        with self.assertRaisesRegex(ValueError, "freshly"):
            benchmark.allocate_adam_state(optimizer)

    def test_summary_excludes_setup_warmup_and_pairs_repetitions(self):
        rows = [{"phase": name, "seconds": seconds} for name, seconds in (
            ("setup/models", 100), ("warmup/loss/0", 90), ("warmup/sampling/0", 80),
            ("measure/loss/1", 2), ("measure/sampling/1", 20),
            ("measure/loss/2", 4), ("measure/sampling/2", 24))]
        result = benchmark.summarize(rows)
        self.assertEqual(result["validation_round_seconds"], 25)
        self.assertEqual(result["loss_seconds"], 3)
        self.assertEqual(result["sampling_seconds"], 22)
        with self.assertRaisesRegex(ValueError, "paired"):
            benchmark.summarize(rows[:-1])

    def test_checkpoint_identity_detects_weight_changes_at_the_same_path(self):
        nvarc, flux = self.root / "nvarc", self.root / "flux"
        for directory in (nvarc, flux / "transformer", flux / "vae"):
            directory.mkdir(parents=True)
            (directory / "model.safetensors").write_bytes(b"old weights")
            (directory / "config.json").write_text("{}")
        runtime = SimpleNamespace(models=SimpleNamespace(nvarc_checkpoint=nvarc, flux_checkpoint=flux))
        before = benchmark.checkpoint_identity(runtime)
        (nvarc / "model.safetensors").write_bytes(b"new weights")
        after = benchmark.checkpoint_identity(runtime)
        self.assertNotEqual(before["nvarc"], after["nvarc"])
        self.assertEqual(before["flux_transformer"], after["flux_transformer"])

    def test_checked_in_configuration_has_agreed_workload(self):
        path = Path(__file__).resolve().parents[2] / "configs/validation/benchmark_b3.json"
        config, runtime, _ = benchmark.load_config(path)
        self.assertEqual(config["sampling_steps"], 40)
        self.assertEqual(runtime.flux_packing.max_tokens, 12288)
        self.assertEqual(runtime.nvarc_max_tokens, 134400)
        with patch.object(benchmark, "load_training_config", return_value=replace(runtime, nvarc_max_tokens=8400)):
            with self.assertRaisesRegex(ValueError, "134400-token NVARC budget"):
                benchmark.load_config(path)


if __name__ == "__main__":
    unittest.main()
