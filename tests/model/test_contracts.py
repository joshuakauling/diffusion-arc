"""Tests for shared model contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch

from diffusion_arc.model import (
    NVARC_CONTEXT_WIDTH,
    NVARC_HIDDEN_SIZE,
    NVARC_HIDDEN_STATE_INDICES,
    NVARC_MAX_TOKENS,
    NVARC_PAD_TOKEN_ID,
    NVARC_VOCAB_SIZE,
    NvarcBatchOutput,
    TARGET_IMAGE_HEIGHT,
    TARGET_IMAGE_SHAPE,
    TARGET_IMAGE_WIDTH,
    validate_packed_source_sequences,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

class ModelContractTests(unittest.TestCase):
    def test_nvarc_execution_constants_match_the_fixed_interface(self) -> None:
        self.assertEqual(NVARC_PAD_TOKEN_ID, 13)
        self.assertEqual(NVARC_HIDDEN_SIZE, 2560)
        self.assertEqual(NVARC_HIDDEN_STATE_INDICES, (9, 18, 27))
        self.assertEqual(NVARC_CONTEXT_WIDTH, 7680)

    def test_packed_source_validation_returns_individual_lengths(self) -> None:
        input_ids = torch.arange(9, dtype=torch.int64).remainder(NVARC_VOCAB_SIZE)
        sequence_offsets = torch.tensor([0, 2, 7, 9], dtype=torch.int64)

        source_lengths = validate_packed_source_sequences(
            input_ids=input_ids,
            sequence_offsets=sequence_offsets,
        )

        self.assertEqual(source_lengths, (2, 5, 2))

    def test_nvarc_batch_output_contract_is_frozen(self) -> None:
        output = NvarcBatchOutput(
            sequence_hidden_states=(torch.zeros((2, NVARC_CONTEXT_WIDTH)),),
        )

        with self.assertRaises(FrozenInstanceError):
            output.sequence_hidden_states = ()  # type: ignore[misc]

    def test_contract_constants_match_arc_data(self) -> None:
        from diffusion_arc.arc_data.image.canvas import CANVAS_SIZE, CELL_SIZE
        from diffusion_arc.arc_data.text.tokenization import (
            NVARC_MAX_TOKENS as ARC_DATA_NVARC_MAX_TOKENS,
        )

        self.assertEqual(TARGET_IMAGE_HEIGHT, CANVAS_SIZE * CELL_SIZE)
        self.assertEqual(TARGET_IMAGE_WIDTH, CANVAS_SIZE * CELL_SIZE)
        self.assertEqual(NVARC_MAX_TOKENS, ARC_DATA_NVARC_MAX_TOKENS)

    def test_importing_model_does_not_import_heavy_model_dependencies(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import diffusion_arc.model; "
                    "assert not any(name == 'transformers' or "
                    "name.startswith('transformers.') for name in sys.modules); "
                    "assert not any(name.startswith('diffusion_arc._vendor') "
                    "or name.startswith('diffusion_arc._third_party') "
                    "or 'bfl_flux' in name for name in sys.modules)"
                ),
            ],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
