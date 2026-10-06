"""Tests for target-free NVARC conditioning preparation."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from diffusion_arc.arc_data import ArcTokenizedEvaluationSample
from diffusion_arc.inference import PackedGenerationInput, prepare_evaluation_text_inputs
from diffusion_arc.inference.conditioning import sample_query_latent_noise
from diffusion_arc.model.contracts import TARGET_LATENT_SHAPE
from diffusion_arc.model import NvarcBatchOutput


class RecordingEvaluationRunner(nn.Module):
    def __init__(self, *, output_count_delta: int = 0) -> None:
        super().__init__()
        self.output_count_delta = output_count_delta
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        sequence_offsets: torch.Tensor,
    ) -> NvarcBatchOutput:
        self.calls.append((input_ids.clone(), sequence_offsets.clone()))
        lengths = tuple(
            end - start
            for start, end in zip(
                sequence_offsets.tolist(),
                sequence_offsets.tolist()[1:],
            )
        )
        count = max(0, len(lengths) + self.output_count_delta)
        hidden_states = tuple(
            torch.full((lengths[index], 4), float(index + 1))
            for index in range(count)
        )
        return NvarcBatchOutput(sequence_hidden_states=hidden_states)


def _sample(task_id: str, query_index: int, input_ids: tuple[int, ...]):
    return ArcTokenizedEvaluationSample(
        task_id=task_id,
        query_index=query_index,
        input_ids=input_ids,
    )


class EvaluationConditioningTests(unittest.TestCase):
    def test_packs_target_free_prompts_and_pads_exact_hidden_states(self) -> None:
        samples = (
            _sample("task-a", 0, (1, 2)),
            _sample("task-b", 0, (3, 4, 5)),
        )
        runner = RecordingEvaluationRunner()

        result = prepare_evaluation_text_inputs(
            samples,
            nvarc_encoder=runner,  # type: ignore[arg-type]
        )

        self.assertEqual(len(runner.calls), 1)
        torch.testing.assert_close(
            runner.calls[0][0],
            torch.tensor((1, 2, 3, 4, 5), dtype=torch.int64),
        )
        torch.testing.assert_close(
            runner.calls[0][1],
            torch.tensor((0, 2, 5), dtype=torch.int64),
        )
        self.assertEqual(result.encoder_hidden_states.shape, (2, 3, 4))
        torch.testing.assert_close(
            result.text_attention_mask,
            torch.tensor(((True, True, False), (True, True, True))),
        )
        torch.testing.assert_close(
            result.encoder_hidden_states[0, 2],
            torch.zeros(4),
        )

    def test_rejects_empty_invalid_and_mismatched_batches(self) -> None:
        runner = RecordingEvaluationRunner()
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            prepare_evaluation_text_inputs((), nvarc_encoder=runner)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "ArcTokenizedEvaluationSample"):
            prepare_evaluation_text_inputs((object(),), nvarc_encoder=runner)  # type: ignore[arg-type]

        with self.assertRaisesRegex(RuntimeError, "output count"):
            prepare_evaluation_text_inputs(
                (_sample("task-a", 0, (1, 2)),),
                nvarc_encoder=RecordingEvaluationRunner(output_count_delta=-1),  # type: ignore[arg-type]
            )


    def test_query_noise_preserves_legacy_seed_assignment(self) -> None:
        query = PackedGenerationInput("task-0", 0, torch.zeros((2, 7680)))
        # Golden derived seeds from the original packed validation implementation.
        cases = (
            (50, ("generation", 0), 6863582346321245173),
            (51, ("generation", 1), 7407952957187489372),
            (167, ("loss", 0.1), 3735880390092699040),
        )
        for seed, purpose, expected_seed in cases:
            with self.subTest(seed=seed, purpose=purpose):
                expected = torch.randn(
                    TARGET_LATENT_SHAPE,
                    generator=torch.Generator().manual_seed(expected_seed),
                )
                actual = sample_query_latent_noise(query, seed, purpose, "cpu")
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
