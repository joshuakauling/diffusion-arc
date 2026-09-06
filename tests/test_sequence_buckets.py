"""Checks for immutable sequence buckets and explicit overflow handling."""

from dataclasses import FrozenInstanceError
import unittest

from diffusion_arc.sequence_buckets import SequenceLengthBuckets


class SequenceLengthBucketTests(unittest.TestCase):
    def test_selects_smallest_covering_boundary_including_exact_hits(self):
        buckets = SequenceLengthBuckets((256, 512, 8400))
        for length, expected in ((1, 256), (256, 256), (257, 512),
                                 (512, 512), (513, 8400), (8400, 8400)):
            with self.subTest(length=length):
                self.assertEqual(buckets.padded_length(length), expected)
        with self.assertRaisesRegex(ValueError, "exceeds the largest bucket"):
            buckets.padded_length(8401)
        self.assertEqual(SequenceLengthBuckets().padded_length(454), 454)

    def test_rejects_invalid_boundaries_once_at_construction(self):
        for value in ((), (0,), (-1,), (8, 8), (16, 8)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SequenceLengthBuckets(value)
        for value in ([8, 16], (True,), (8.0,), ("8",)):
            with self.subTest(value=value), self.assertRaises(TypeError):
                SequenceLengthBuckets(value)
        buckets = SequenceLengthBuckets((8, 16))
        with self.assertRaises(FrozenInstanceError):
            buckets.boundaries = (32,)


if __name__ == "__main__":
    unittest.main()
