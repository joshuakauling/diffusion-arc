"""Keep prepared-bundle startup limited to manifest and split-boundary checks."""

import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from diffusion_arc.training.run import validation


def _manifest(payload: bytes) -> dict:
    ordered = [f"{index:08x}" for index in range(400)]
    random.Random(167).shuffle(ordered)
    return {
        "schema_version": 1,
        "dataset": "ARC-AGI-1",
        "release": "v1.0.2",
        "archive_sha256": validation.ARC_AGI_1_ARCHIVE_SHA256,
        "split_seed": 167,
        "validation_task_ids": ordered[:100],
        "sampling_task_ids": ordered[:32],
        "final_test_task_ids": ordered[100:],
        "validation_output_count": 100,
        "sampling_output_count": 32,
        "task_sha256": {key: hashlib.sha256(payload).hexdigest() for key in ordered[:100]},
    }


class ValidationManifestChecksTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.bundle = Path(temporary.name)
        self.payload = b'{"train": [], "test": [{"input": [[0]], "output": [[1]]}]}'
        self.manifest = _manifest(self.payload)
        self.write_manifest(self.manifest)

    def write_manifest(self, manifest):
        (self.bundle / "manifest.json").write_text(json.dumps(manifest))

    def assert_metadata_rejected(self, manifest, message, *, split_seed=167):
        self.write_manifest(manifest)
        # Invalid metadata must fail before any payload audit, in either mode.
        with patch.object(Path, "iterdir", side_effect=AssertionError("directory scan")), \
                patch.object(Path, "read_bytes", side_effect=AssertionError("payload read")):
            for verify_payloads in (False, True):
                with self.subTest(verify_payloads=verify_payloads), self.assertRaisesRegex(ValueError, message):
                    validation.validate_validation_bundle(
                        self.bundle, split_seed=split_seed, verify_payloads=verify_payloads,
                    )

    def test_metadata_only_reads_manifest_without_scanning_or_hashing_payloads(self):
        original_read_text = Path.read_text
        reads = []

        def read_manifest(path, *args, **kwargs):
            self.assertEqual(path, self.bundle / "manifest.json")
            reads.append(path)
            return original_read_text(path, *args, **kwargs)

        # No tasks directory is needed until a selected task is consumed.
        with patch.object(Path, "read_text", read_manifest), \
                patch.object(Path, "read_bytes", side_effect=AssertionError("payload read")), \
                patch.object(Path, "iterdir", side_effect=AssertionError("directory scan")), \
                patch.object(validation.hashlib, "sha256", side_effect=AssertionError("payload hash")):
            actual = validation.validate_validation_bundle(self.bundle, verify_payloads=False)
        self.assertEqual(actual, self.manifest)
        self.assertEqual(reads, [self.bundle / "manifest.json"])

    def test_metadata_only_keeps_source_and_requested_seed_protections(self):
        self.assert_metadata_rejected(
            dict(self.manifest, archive_sha256="0" * 64), "unexpected source",
        )
        self.assert_metadata_rejected(self.manifest, "split seed", split_seed=168)

    def test_metadata_only_rejects_overlap_repartitioning_and_final_test_sampling(self):
        final_test = self.manifest["final_test_task_ids"].copy()
        final_test[0] = self.manifest["validation_task_ids"][0]
        self.assert_metadata_rejected(dict(self.manifest, final_test_task_ids=final_test), "100/300")

        reordered = list(reversed(self.manifest["validation_task_ids"]))
        self.assert_metadata_rejected(
            dict(self.manifest, validation_task_ids=reordered, sampling_task_ids=reordered[:32]),
            "seeded split",
        )
        sampling = self.manifest["sampling_task_ids"].copy()
        sampling[0] = self.manifest["final_test_task_ids"][0]
        self.assert_metadata_rejected(dict(self.manifest, sampling_task_ids=sampling), "first 32")

    def test_metadata_only_requires_hash_inventory_to_cover_only_validation(self):
        hashes = self.manifest["task_sha256"].copy()
        hashes.pop(self.manifest["validation_task_ids"][0])
        self.assert_metadata_rejected(dict(self.manifest, task_sha256=hashes), "hash inventory")
        hashes[self.manifest["final_test_task_ids"][0]] = "0" * 64
        self.assert_metadata_rejected(dict(self.manifest, task_sha256=hashes), "hash inventory")

    def test_malformed_manifest_fields_fail_before_payload_access(self):
        cases = [
            ([], "must be an object"),
            (dict(self.manifest, schema_version=True), "manifest schema"),
            (dict(self.manifest, dataset="other"), "dataset release"),
            (dict(self.manifest, split_seed=True), "split seed"),
            (dict(self.manifest, validation_task_ids=None), "must be lists"),
            (dict(self.manifest, sampling_task_ids=[{}]), "hexadecimal"),
            (dict(self.manifest, task_sha256=[]), "hash inventory"),
            (dict(self.manifest, task_sha256=dict.fromkeys(self.manifest["validation_task_ids"], "wrong")),
             "SHA256 digests"),
            (dict(self.manifest, validation_output_count=True), "query counts"),
            (dict(self.manifest, sampling_output_count=None), "query counts"),
            (dict(self.manifest, sampling_output_count=31), "query counts"),
            (dict(self.manifest, validation_output_count=99), "query counts"),
            (dict(self.manifest, sampling_output_count=101), "query counts"),
        ]
        for manifest, message in cases:
            with self.subTest(manifest=manifest, message=message):
                self.assert_metadata_rejected(manifest, message)

    def test_default_mode_still_detects_changed_payloads(self):
        tasks = self.bundle / "tasks"
        tasks.mkdir()
        for key in self.manifest["validation_task_ids"]:
            (tasks / f"{key}.json").write_bytes(self.payload)
        self.assertEqual(validation.validate_validation_bundle(self.bundle), self.manifest)
        changed = self.manifest["validation_task_ids"][0]
        (tasks / f"{changed}.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "payload changed"):
            validation.validate_validation_bundle(self.bundle)


if __name__ == "__main__":
    unittest.main()
