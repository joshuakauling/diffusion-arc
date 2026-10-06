"""Numerical and boundary tests for oracle partial-denoising diagnostics."""

import json
import unittest
from unittest.mock import patch

import torch
from torch import nn

from diffusion_arc.arc_data import render_grid_canvas
from diffusion_arc.diagnostics import partial_denoising as diagnostic
from diffusion_arc.inference.conditioning import sample_query_latent_noise
from diffusion_arc.inference.generation import generate_packed_arc_samples, linear_timestep_schedule
from diffusion_arc.model.contracts import TARGET_LATENT_SHAPE
from diffusion_arc.training.run.validation import ValidationTarget


def target(index=0, *, clean_value=0.35, grid=((1, 2), (3, 4))):
    return ValidationTarget(
        f"task-{index}", 0, torch.full((2 + index, 7680), index / 10.0),
        torch.full(TARGET_LATENT_SHAPE, clean_value), grid,
    )


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))


class Decoder(nn.Module):
    def __init__(self, grid=((1, 2), (3, 4))):
        super().__init__()
        self.image = render_grid_canvas(grid)
        self.states = []

    def forward(self, latents, *, spatial_shape):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert (spatial_shape.height, spatial_shape.width) == (30, 30)
        self.states.append(latents.detach().clone())
        return self.image.unsqueeze(0).expand(len(latents), -1, -1, -1)


class PartialDenoisingTests(unittest.TestCase):
    def setUp(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        self.transformer = Transformer().train()
        self.decoder = Decoder().train()
        self.target = target()

    def iterate(self, *, transformer=None, decoder=None, targets=None, packs=None, **overrides):
        return diagnostic.iter_partial_denoising(
            transformer or self.transformer, decoder or self.decoder,
            targets or (self.target,), ((0,),) if packs is None else packs,
            **{
                "start_times": (1., .9, .7, .5, .3, .1), "steps_per_unit": 40,
                "seeds": (20167,), "compute_dtype": torch.bfloat16, "backend": object(),
                **overrides,
            },
        )

    def test_constant_oracle_velocity_recovers_nonzero_target_at_every_start(self):
        clean = self.target.clean_latent.unsqueeze(0)
        noise = sample_query_latent_noise(
            self.target.conditioning, 20167, ("generation", 0), "cpu",
        ).unsqueeze(0)
        true_velocity = noise - clean
        calls = []

        def predict(model, state, text, positions, times, layout, **kwargs):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(state.dtype, torch.float32)
            self.assertEqual(kwargs["compute_dtype"], torch.bfloat16)
            calls.append((times[0].item(), state.clone()))
            return true_velocity

        rng = torch.random.get_rng_state().clone()
        with patch.object(diagnostic, "predict_packed_flux_velocity", side_effect=predict):
            records = list(self.iterate())
        self.assertEqual(len(records), 13)
        self.assertEqual(records[0]["kind"], "clean_roundtrip")
        self.assertIsNone(records[0]["start_t"])
        self.assertIsNone(records[0]["seed"])
        self.assertEqual(records[0]["latent_mse"], 0.)
        offset = 0
        full_schedule = linear_timestep_schedule(40)
        for index, steps in enumerate((40, 36, 28, 20, 12, 4)):
            initial_record, final_record = records[1 + index * 2:3 + index * 2]
            schedule = full_schedule[40 - steps:]
            expected_initial = (1 - schedule[0]) * clean + schedule[0] * noise
            torch.testing.assert_close(self.decoder.states[1 + index * 2], expected_initial)
            torch.testing.assert_close(self.decoder.states[2 + index * 2], clean, atol=2e-6, rtol=0)
            self.assertEqual([time for time, _ in calls[offset:offset + steps]], schedule[:-1].tolist())
            torch.testing.assert_close(calls[offset][1], expected_initial)
            self.assertEqual(final_record["steps"], steps)
            self.assertEqual(initial_record["steps"], 0)
            self.assertLess(final_record["latent_mse"], 1e-12)
            self.assertEqual(final_record["initial_v_mse"], 0.)
            self.assertLess(final_record["one_step_z0_mse"], 1e-12)
            self.assertTrue(final_record["exact_grid"])
            offset += steps
        json.dumps(records, allow_nan=False)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        self.assertTrue(self.transformer.training)
        self.assertTrue(self.decoder.training)

    def test_one_step_error_is_t_squared_velocity_error(self):
        with patch.object(diagnostic, "predict_packed_flux_velocity", side_effect=lambda _, state, *a, **k:
                          torch.full_like(state, 0.375, dtype=torch.bfloat16)):
            records = list(self.iterate(start_times=(.5, .1), seeds=(20167, 20168)))
        finals = [record for record in records if record["kind"] == "denoised"]
        for record in finals:
            self.assertAlmostEqual(record["one_step_z0_mse"],
                                   record["start_t"] ** 2 * record["initial_v_mse"], places=6)
        self.assertNotEqual(finals[0]["latent_mse"], finals[2]["latent_mse"])

    def test_paired_schedules_recover_target_with_equal_forward_budgets(self):
        plans = diagnostic.diagnostic_schedules((1., .9), 50, ("linear", "cosine_high_noise"))
        self.assertEqual([len(times) - 1 for _, _, times in plans], [50, 45, 50])
        linear, cosine = plans[0][2], plans[2][2]
        self.assertEqual((cosine[0].item(), cosine[-1].item()), (1., 0.))
        self.assertTrue(bool((cosine[:-1] > cosine[1:]).all()))
        self.assertEqual(int((linear[:-1] >= .8).sum()), 11)
        self.assertEqual(int((cosine[:-1] >= .8).sum()), 21)

        noise = sample_query_latent_noise(self.target.conditioning, 20167, ("generation", 0), "cpu").unsqueeze(0)
        velocity = noise - self.target.clean_latent.unsqueeze(0)
        calls = []

        def predict(_, state, text, positions, times, *args, **kwargs):
            calls.append((times[0].item(), state.clone()))
            return velocity

        with patch.object(diagnostic, "predict_packed_flux_velocity", side_effect=predict):
            records = list(self.iterate(start_times=(1., .9), steps_per_unit=50,
                                       generation_schedules=("linear", "cosine_high_noise")))
        self.assertEqual([time for time, _ in calls],
                         [time for _, _, times in plans for time in times[:-1].tolist()])
        torch.testing.assert_close(calls[0][1], calls[95][1], rtol=0, atol=0)
        for record in records:
            if record["kind"] == "denoised":
                self.assertLess(record["latent_mse"], 1e-11)

    def test_full_noise_model_inputs_do_not_depend_on_reference_target_for_either_schedule(self):
        traces = []
        for clean_value in (.35, -2.):
            calls = []

            def predict(_, state, text, positions, times, *args, **kwargs):
                calls.append(state.clone())
                return torch.tanh(state) * times[0]

            with patch.object(diagnostic, "predict_packed_flux_velocity", side_effect=predict):
                list(self.iterate(targets=(target(clean_value=clean_value),), start_times=(1.,),
                                  steps_per_unit=5, seeds=(20167, 20168),
                                  generation_schedules=("linear", "cosine_high_noise")))
            traces.append(calls)
        for before, after in zip(*traces, strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_t1_matches_existing_packed_sampling_with_real_transformer(self):
        from tests.model.test_flux_packing import SegmentedReference, tiny_model

        model = tiny_model(in_channels=128, out_channels=128, joint_attention_dim=7680)
        targets = (target(0), target(1, clean_value=-1.25))
        packs = ((1, 0),)
        seeds = (20167, 20168)
        normal_decoder = Decoder().eval()
        expected = generate_packed_arc_samples(
            model, normal_decoder, tuple(value.conditioning for value in targets), packs,
            steps=2, seeds=seeds, compute_dtype=torch.float32, backend=SegmentedReference(),
        )
        records = list(self.iterate(
            transformer=model, targets=targets, packs=packs, start_times=(1.,),
            steps_per_unit=2, seeds=seeds, compute_dtype=torch.float32, backend=SegmentedReference(),
        ))
        for sample_index, seed in enumerate(seeds):
            torch.testing.assert_close(self.decoder.states[2 + 2 * sample_index],
                                       normal_decoder.states[sample_index], rtol=0, atol=0)
            for index, value in enumerate(targets):
                row = next(row for row in records if row["kind"] == "denoised"
                           and row["seed"] == seed and row["task_id"] == value.task_id)
                self.assertEqual(row["predicted_grid"], [list(line) for line in expected[index][sample_index].grid])

    def test_step_sweep_matches_individual_production_runs_and_direct_one_step_estimate(self):
        from tests.model.test_flux_packing import SegmentedReference, tiny_model

        model = tiny_model(in_channels=128, out_channels=128, joint_attention_dim=7680)
        seeds = (20167, 20168)
        counts = (1, 4, 10)
        records = list(self.iterate(
            transformer=model, start_times=(1.,), steps_per_unit=None, generation_steps=counts,
            seeds=seeds, compute_dtype=torch.float32, backend=SegmentedReference(),
        ))
        finals = [r for r in records if r["kind"] == "denoised"]
        self.assertEqual([r["sampling_steps"] for r in finals], list(counts) * 2)
        for variant_index, steps in enumerate(counts):
            decoder = Decoder().eval()
            generate_packed_arc_samples(model, decoder, (self.target.conditioning,), ((0,),),
                steps=steps, seeds=seeds, compute_dtype=torch.float32, backend=SegmentedReference())
            for seed_index in range(2):
                offset = 1 + seed_index * 6 + variant_index * 2
                torch.testing.assert_close(self.decoder.states[offset + 1], decoder.states[seed_index], rtol=0, atol=0)
                torch.testing.assert_close(self.decoder.states[offset], self.decoder.states[1 + seed_index * 6], rtol=0, atol=0)
        self.assertFalse(torch.equal(self.decoder.states[1], self.decoder.states[7]))
        for record in finals:
            if record["sampling_steps"] == 1:
                self.assertEqual(record["latent_mse"], record["one_step_z0_mse"])

    def test_step_sweep_calls_all_timesteps_and_keeps_initial_noise_paired(self):
        times, starts = [], []

        def predict(_, state, text, positions, t, *args, **kwargs):
            times.append(t[0].item())
            if t[0] == 1:
                starts.append(state.clone())
            return torch.tanh(state) * t[0]

        with patch.object(diagnostic, "predict_packed_flux_velocity", side_effect=predict):
            list(self.iterate(start_times=(1.,), steps_per_unit=None, generation_steps=(1, 4, 10)))
        self.assertEqual(times, [t for steps in (1, 4, 10) for t in linear_timestep_schedule(steps)[:-1].tolist()])
        for state in starts[1:]:
            torch.testing.assert_close(state, starts[0], rtol=0, atol=0)

    def test_every_yield_restores_modes_and_inference_context(self):
        with patch.object(diagnostic, "predict_packed_flux_velocity", side_effect=lambda _, state, *a, **k: state * 0):
            iterator = self.iterate(start_times=(.5,), steps_per_unit=2)
            for expected_kind in ("clean_roundtrip", "noisy_start", "denoised"):
                row = next(iterator)
                self.assertEqual(row["kind"], expected_kind)
                self.assertTrue(self.transformer.training)
                self.assertTrue(self.decoder.training)
                self.assertTrue(torch.is_grad_enabled())
                self.assertFalse(torch.is_inference_mode_enabled())
            iterator.close()

    def test_forward_failure_restores_modes_and_keeps_completed_controls(self):
        with patch.object(diagnostic, "predict_packed_flux_velocity", side_effect=RuntimeError("injected")):
            iterator = self.iterate(start_times=(.5,))
            self.assertEqual(next(iterator)["kind"], "clean_roundtrip")
            self.assertEqual(next(iterator)["kind"], "noisy_start")
            with self.assertRaisesRegex(RuntimeError, "injected"):
                next(iterator)
        self.assertTrue(self.transformer.training)
        self.assertTrue(self.decoder.training)
        self.assertFalse(torch.is_inference_mode_enabled())

    def test_bad_configuration_fails_before_any_model_forward(self):
        cases = (
            {"start_times": ()}, {"start_times": (0.,)}, {"start_times": (-.1,)},
            {"start_times": (1.1,)}, {"start_times": (float("nan"),)},
            {"start_times": (True,)}, {"start_times": (.51,)}, {"start_times": (.5, .5)},
            {"steps_per_unit": 0}, {"steps_per_unit": True}, {"seeds": ()},
            {"seeds": (1, 1)}, {"seeds": (-1,)}, {"seeds": (True,)},
            {"packs": ((0, 0),)}, {"backend": None}, {"compute_dtype": torch.float16},
            {"generation_schedules": ("linear", "unknown")},
            {"start_times": (.9,), "generation_schedules": ("linear", "cosine_high_noise")},
            {"start_times": (1.,), "steps_per_unit": None, "generation_steps": ()},
            {"start_times": (1.,), "steps_per_unit": None, "generation_steps": (1, 1)},
            {"start_times": (1.,), "steps_per_unit": None, "generation_steps": (True,)},
            {"start_times": (1.,), "generation_steps": (1, 4, 10)},
        )
        with patch.object(diagnostic, "predict_packed_flux_velocity") as predict:
            for arguments in cases:
                with self.subTest(arguments=arguments), self.assertRaises((ValueError, TypeError)):
                    list(self.iterate(**arguments))
            predict.assert_not_called()
        self.assertEqual(self.decoder.states, [])

    def test_shape_mismatch_has_no_cell_accuracy(self):
        self.decoder = Decoder(grid=((1,),))
        row = next(self.iterate())
        self.assertTrue(row["prediction_valid"])
        self.assertFalse(row["shape_match"])
        self.assertFalse(row["exact_grid"])
        self.assertIsNone(row["cell_accuracy"])
        self.assertEqual(row["predicted_grid"], [[1]])

    def test_nonfinite_velocity_fails_without_emitting_denoised_result(self):
        with patch.object(diagnostic, "predict_packed_flux_velocity", side_effect=lambda _, state, *a, **k:
                          torch.full_like(state, float("nan"))):
            iterator = self.iterate(start_times=(.5,))
            self.assertEqual(next(iterator)["kind"], "clean_roundtrip")
            self.assertEqual(next(iterator)["kind"], "noisy_start")
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                next(iterator)
        self.assertEqual(len(self.decoder.states), 2)
        self.assertTrue(self.transformer.training)
        self.assertTrue(self.decoder.training)


if __name__ == "__main__":
    unittest.main()
