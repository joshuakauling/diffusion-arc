"""CPU checks for held-out packed loss, RNG isolation, and strict sampling."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from torch import nn
import torch

from diffusion_arc.arc_data import CANVAS_SIZE
from diffusion_arc.arc_data.image.canvas import decode_arc_grids, render_grid_canvas
from diffusion_arc.arc_data.image.palette import PAD_CLASS
from diffusion_arc.arc_data.text.tokenization import ArcTokenizedEvaluationSample
from diffusion_arc.inference import generation
from diffusion_arc.model.contracts import TARGET_LATENT_SHAPE
from diffusion_arc.training.run import validation
from diffusion_arc.training.step.loss import RegionLossWeights


def _target(index=0, *, task_id=None, query_index=0, side=3, length=2, value=0.0):
    return validation.ValidationTarget(
        task_id=task_id or f"task-{index}", query_index=query_index,
        text=torch.full((length, 7680), value, dtype=torch.bfloat16),
        clean_latent=torch.zeros(TARGET_LATENT_SHAPE),
        expected_grid=tuple((index % 10,) * side for _ in range(side)),
    )


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))


class _Decoder(nn.Module):
    def __init__(self, grid=((0, 0, 0),) * 3):
        super().__init__()
        self.image = render_grid_canvas(grid)
        self.states = []

    def forward(self, latents, *, spatial_shape):
        assert not torch.is_grad_enabled()
        assert (spatial_shape.height, spatial_shape.width) == (30, 30)
        self.states.append(latents.detach().clone())
        return self.image.unsqueeze(0).expand(latents.shape[0], -1, -1, -1)


class _NvarcCacheEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.output_refs = []

    def forward(self, ids, offsets):
        assert not torch.is_grad_enabled() and not self.training
        assert all(reference() is None for reference in self.output_refs)
        self.calls.append(offsets.tolist())
        states = tuple(torch.full((end - start, 7680), float(ids[start]), dtype=torch.bfloat16)
                       for start, end in zip(offsets.tolist(), offsets.tolist()[1:]))
        self.output_refs = [weakref.ref(state) for state in states]
        return SimpleNamespace(sequence_hidden_states=states)


class _VaeCacheEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.calls = []

    def forward(self, images):
        assert not torch.is_grad_enabled() and not self.training
        self.calls.append(len(images))
        return images[:, 0, 0, 0, None, None].expand(-1, *TARGET_LATENT_SHAPE) + self.anchor


def _prediction(grid):
    canvas = torch.full((1, 30, 30), PAD_CLASS, dtype=torch.int64)
    canvas[0, :len(grid), :len(grid[0])] = torch.tensor(grid)
    return decode_arc_grids(canvas)[0]


def _class_canvas(grid: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    canvas = torch.full(
        (CANVAS_SIZE, CANVAS_SIZE),
        PAD_CLASS,
        dtype=torch.int64,
    )
    values = torch.tensor(grid, dtype=torch.int64)
    canvas[: values.shape[0], : values.shape[1]] = values
    return canvas


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.transformer = _Transformer().train()
        self.arguments = {"compute_dtype": torch.float32, "backend": object()}

    def loss(self, targets, packs, **overrides):
        arguments = {"timesteps": validation.VALIDATION_TIMESTEPS, "seed": 167,
                     **self.arguments, **overrides}
        return validation.evaluate_packed_loss(self.transformer, targets, packs, **arguments)

    def generate(self, targets, packs, decoder=None, **overrides):
        arguments = {"steps": 4, "seeds": (50, 51), **self.arguments, **overrides}
        return validation.generate_packed_samples(
            self.transformer, decoder or _Decoder(), targets, packs, **arguments,
        )

    def test_sampling_inputs_load_only_selected_tasks_without_query_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "tasks").mkdir()
            # Held-out outputs may be absent or malformed: presentation only needs inputs.
            (bundle / "tasks" / "00000001.json").write_text(json.dumps({
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3, 4]], "output": "never read"}, {"input": [[5], [6]]}],
            }))
            (bundle / "tasks" / "00000002.json").write_text(json.dumps({
                "train": [], "test": [{"input": [[7]]}],
            }))
            (bundle / "tasks" / "00000003.json").write_text("not JSON")
            with patch.object(validation, "load_evaluation_targets", side_effect=AssertionError("target access")), \
                    patch.object(validation, "load_evaluation_task", wraps=validation.load_evaluation_task) as load:
                inputs = validation.load_sampling_input_grids(bundle, ("00000002", "00000001"))
            self.assertEqual([call.args[0].stem for call in load.call_args_list], ["00000002", "00000001"])
        self.assertEqual(inputs, {
            ("00000002", 0): ((7,),),
            ("00000001", 0): ((3, 4),),
            ("00000001", 1): ((5,), (6,)),
        })

    def test_cache_batches_by_tokens_and_images_without_changing_query_alignment(self):
        samples = tuple(ArcTokenizedEvaluationSample(f"task-{i // 2}", i % 2, (i,) * length)
                        for i, length in enumerate((1, 2, 3, 1)))
        grids = tuple(((i,),) for i in range(4))
        rng = torch.random.get_rng_state().clone()
        nvarc, vae = _NvarcCacheEncoder().eval(), _VaeCacheEncoder().eval()
        cache = validation.prepare_validation_cache(
            samples, grids, nvarc_encoder=nvarc, vae_encoder=vae,
            nvarc_max_tokens=4, vae_batch_size=3,
        )
        self.assertEqual(nvarc.calls, [[0, 1, 3], [0, 3, 4]])
        self.assertEqual(vae.calls, [3, 1])
        self.assertTrue(all(reference() is None for reference in nvarc.output_refs))
        for i, target in enumerate(cache):
            self.assertEqual((target.task_id, target.query_index), (samples[i].task_id, samples[i].query_index))
            self.assertEqual(target.expected_grid, grids[i])
            self.assertEqual(target.text.shape[0], len(samples[i].input_ids))
            self.assertEqual(target.text[0, 0], i)
            self.assertEqual(target.clean_latent[0, 0], render_grid_canvas(grids[i])[0, 0, 0])
            for tensor in (target.text, target.clean_latent):
                self.assertEqual(tensor.device.type, "cpu")
                self.assertFalse(tensor.requires_grad)
                self.assertIsNone(tensor.grad_fn)
        # A different physical execution must produce the same logical caches.
        other = validation.prepare_validation_cache(
            samples, grids, nvarc_encoder=nvarc, vae_encoder=vae,
            nvarc_max_tokens=7, vae_batch_size=4,
        )
        for left, right in zip(cache, other, strict=True):
            torch.testing.assert_close(left.text, right.text, rtol=0, atol=0)
            torch.testing.assert_close(left.clean_latent, right.clean_latent, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        self.assertIsNone(vae.anchor.grad)

    def test_cache_preflights_all_lengths_before_encoding_any_query(self):
        nvarc, vae = _NvarcCacheEncoder().eval(), _VaeCacheEncoder().eval()
        for budget, length, message in ((3, 4, "nvarc_max_tokens=3"), (134400, 8401, "source limit 8400")):
            samples = (ArcTokenizedEvaluationSample("short", 0, (1,)),
                       ArcTokenizedEvaluationSample("long", 0, (2,) * length))
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, message):
                validation.prepare_validation_cache(
                    samples, (((0,),),) * 2, nvarc_encoder=nvarc, vae_encoder=vae,
                    nvarc_max_tokens=budget, vae_batch_size=1,
                )
        self.assertEqual(nvarc.calls, [])
        self.assertEqual(vae.calls, [])

    def test_cache_requires_frozen_encoders_in_eval_mode(self):
        nvarc, vae = _NvarcCacheEncoder().eval(), _VaeCacheEncoder().eval()
        for training, trainable in ((True, False), (False, True)):
            nvarc.train(training)
            vae.anchor.requires_grad_(trainable)
            with self.subTest(training=training), self.assertRaisesRegex(ValueError, "frozen and in evaluation"):
                validation.prepare_validation_cache(
                    (ArcTokenizedEvaluationSample("query", 0, (1,)),), (((0,),),),
                    nvarc_encoder=nvarc, vae_encoder=vae, nvarc_max_tokens=10, vae_batch_size=1,
                )
        self.assertEqual(nvarc.calls, [])
        self.assertEqual(vae.calls, [])

    def test_cache_contract_rejects_gradients_shapes_and_nonfinite_values(self):
        target = _target()
        cases = (
            {"clean_latent": torch.zeros(TARGET_LATENT_SHAPE, requires_grad=True)},
            {"clean_latent": torch.zeros(TARGET_LATENT_SHAPE, dtype=torch.bfloat16)},
            {"text": torch.zeros((2, 4))},
            {"text": torch.full((2, 7680), float("nan"))},
            {"expected_grid": ((10,),)},
        )
        for changes in cases:
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                replace(target, **changes)

    def test_best_fit_packs_are_stable_complete_and_respect_budget(self):
        targets = tuple(_target(i, length=length) for i, length in enumerate((6, 5, 4, 4)))
        packs = validation.plan_validation_packs(targets, max_joint_tokens=1810)
        self.assertEqual(packs, ((0, 2), (1, 3)))
        self.assertEqual(sorted(i for pack in packs for i in pack), list(range(4)))
        for pack in packs:
            self.assertLessEqual(sum(900 + targets[i].text.shape[0] for i in pack), 1810)
        with self.assertRaisesRegex(ValueError, "exceeding budget"):
            validation.plan_validation_packs(targets, max_joint_tokens=905)
        with self.assertRaisesRegex(ValueError, "unique"):
            validation.plan_validation_packs((targets[0], targets[0]), max_joint_tokens=1810)


    def test_validation_uses_configurable_region_weights_and_prepares_each_pack_once(self):
        residual = torch.ones((1, 30, 30, 128))
        residual[:, :3, :3] = 2.0
        residual = residual.reshape(1, 900, 128)

        def predict(_, state, _text, _positions, times, _layout, **kwargs):
            return state / times[:, None, None] + residual

        weights = RegionLossWeights(.7, .2, .1)
        with patch.object(validation, "predict_packed_flux_velocity", side_effect=predict), \
                patch.object(validation, "build_region_loss_reduction", wraps=validation.build_region_loss_reduction) as prepare:
            result = self.loss((_target(side=3),), ((0,),), weights=weights)
        prepare.assert_called_once()
        for row in result["per_timestep"]:
            self.assertAlmostEqual(row["canvas_loss"], 1.03, places=5)
            self.assertAlmostEqual(row["grid_loss"], 4.0, places=5)
            self.assertAlmostEqual(row["boundary_loss"], 1.0, places=5)
            self.assertAlmostEqual(row["weighted_loss"], .7 * 1.03 + .2 * 4 + .1, places=5)

    def test_loss_averages_targets_equally_across_small_tail_pack(self):
        targets = tuple(_target(i, value=value, length=i + 2) for i, value in enumerate((1, 1, 3)))

        def predict(transformer, state, text, positions, times, layout, **_):
            self.assertFalse(transformer.training)
            self.assertFalse(torch.is_grad_enabled())
            offset = 0
            errors = []
            for length in layout.text_lengths:
                errors.append(text[offset, 0].float())
                offset += length
            return state / times[:, None, None] + torch.stack(errors)[:, None, None]

        rng = torch.random.get_rng_state().clone()
        with patch.object(validation, "predict_packed_flux_velocity", side_effect=predict):
            result = self.loss(targets, ((0, 1), (2,)))
        self.assertEqual(result["actual_forward_calls"], 12)
        self.assertEqual(result["target_forward_evaluations"], 18)
        self.assertEqual(result["nvarc_tokens"], 0)
        self.assertEqual(result["flux_text_tokens"], (2 + 3 + 4) * 6)
        self.assertEqual(result["flux_image_tokens"], 3 * 900 * 6)
        self.assertEqual(result["flux_joint_tokens"], (2 + 3 + 4 + 3 * 900) * 6)
        self.assertEqual(result["flux_forward_calls"], 12)
        for time in result["per_timestep"]:
            self.assertAlmostEqual(time["weighted_loss"], 11 / 3, places=5)
        self.assertAlmostEqual(result["combined_loss"], 11 / 3, places=5)
        self.assertTrue(self.transformer.training)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        json.dumps(result, allow_nan=False)

    def test_loss_rng_and_metrics_do_not_depend_on_pack_order(self):
        targets = tuple(_target(i, value=i) for i in range(3))
        seen = []

        def predict(_transformer, state, text, positions, times, layout, **_):
            offset = 0
            for index, length in enumerate(layout.text_lengths):
                seen.append((int(text[offset, 0]), float(times[index]), state[index].clone()))
                offset += length
            return torch.zeros_like(state)

        with patch.object(validation, "predict_packed_flux_velocity", side_effect=predict):
            first = self.loss(targets, ((0, 1), (2,)))
            first_seen, seen = seen, []
            second = self.loss(targets, ((2, 0, 1),))
        self.assertEqual(first["per_timestep"], second["per_timestep"])
        for left, right in zip(sorted(first_seen, key=lambda v: v[:2]),
                               sorted(seen, key=lambda v: v[:2]), strict=True):
            torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)
        self.assertNotEqual(first["actual_forward_calls"], second["actual_forward_calls"])

    def test_full_canvas_loss_omits_absent_boundary_and_remains_finite(self):
        with patch.object(validation, "predict_packed_flux_velocity", side_effect=lambda _, x, *a, **k: x * 0):
            result = self.loss((_target(side=30),), ((0,),))
        self.assertEqual(result["boundary_target_count"], 0)
        for row in result["per_timestep"]:
            self.assertIsNone(row["boundary_loss"])
            self.assertAlmostEqual(row["canvas_loss"], row["weighted_loss"], places=6)
        json.dumps(result, allow_nan=False)


    def test_generation_is_independent_of_targets_and_packing(self):
        targets = tuple(_target(i) for i in range(3))
        first_decoder, second_decoder = _Decoder(), _Decoder()
        changed = tuple(replace(target, clean_latent=torch.full(TARGET_LATENT_SHAPE, 99.),
                                expected_grid=((9,),)) for target in targets)
        with patch.object(generation, "predict_packed_flux_velocity", side_effect=lambda _, x, *a, **k: x * 0):
            self.generate(targets, ((0, 1), (2,)), first_decoder, steps=1)
            self.generate(changed, ((2, 0, 1),), second_decoder, steps=1)
        # Decoder calls are pack-major, sample-minor, with targets in pack order.
        for sample in range(2):
            original = torch.cat((first_decoder.states[sample], first_decoder.states[2 + sample]))
            reordered = second_decoder.states[sample][[1, 2, 0]]
            torch.testing.assert_close(original, reordered, rtol=0, atol=0)

    def test_generation_never_reads_supervision_before_both_decodes(self):
        completed_decodes = []

        class GuardedTarget(validation.ValidationTarget):
            guard_enabled = False

            def __getattribute__(self, name):
                if name in ("clean_latent", "expected_grid") and self.guard_enabled:
                    if name == "clean_latent" or len(completed_decodes) != 2:
                        raise AssertionError("generation accessed supervision")
                return super().__getattribute__(name)

        base = _target()
        target = GuardedTarget(base.task_id, base.query_index, base.text,
                               base.clean_latent, base.expected_grid)
        object.__setattr__(target, "guard_enabled", True)
        decoder = _Decoder()
        decode = generation.decode_arc_grids

        def record_decode(canvases):
            completed_decodes.append(True)
            return decode(canvases)

        with patch.object(generation, "predict_packed_flux_velocity", side_effect=lambda _, x, *a, **k: x * 0), \
                patch.object(generation, "decode_arc_grids", side_effect=record_decode):
            result = self.generate((target,), ((0,),), decoder, steps=1)
        self.assertEqual(result["pass_at_1"], 1.0)

    def test_strict_task_pass_two_requires_shared_successful_sample_index(self):
        targets = (_target(0, task_id="multi", query_index=0),
                   _target(1, task_id="multi", query_index=1))
        correct0, correct1 = (_prediction(target.expected_grid) for target in targets)
        incorrect = _prediction(((9,),))
        decoded = ((correct0, incorrect), (incorrect, correct1))
        with patch.object(generation, "predict_packed_flux_velocity", side_effect=lambda _, x, *a, **k: x * 0), \
                patch.object(generation, "decode_arc_grids", side_effect=decoded):
            result = self.generate(targets, ((0, 1),), steps=1)
        self.assertEqual(result["task_count"], 1)
        self.assertEqual(result["target_count"], 2)
        self.assertEqual(result["pass_at_2"], 0.0)
        self.assertEqual(result["tasks"][0]["sample_exact"], [False, False])
        self.assertEqual(result["prediction_count"], 4)
        self.assertEqual(result["shape_correct_count"], 2)
        self.assertEqual(result["shape_accuracy"], 0.5)
        self.assertEqual(result["cell_accuracy"], 1.0)
        self.assertTrue(result["queries"][0]["samples"][0]["exact_grid"])
        self.assertTrue(result["queries"][1]["samples"][1]["exact_grid"])

    def test_sampling_diagnostics_weight_predictions_equally_and_reuse_each_decode(self):
        targets = (_target(0, side=1, length=3), _target(1, side=2, length=5))
        small_wrong = _prediction(((9,),))
        large_partial = _prediction(((1, 1), (1, 9)))
        decoder = _Decoder()
        with patch.object(generation, "predict_packed_flux_velocity",
                          side_effect=lambda _, x, *a, **k: x * 0) as forward, \
                patch.object(generation, "decode_arc_grids",
                             side_effect=((small_wrong,), (small_wrong,),
                                          (large_partial,), (large_partial,))) as decode:
            result = self.generate(targets, ((0,), (1,)), decoder, steps=3)

        self.assertEqual(result["prediction_count"], 4)
        self.assertEqual(result["shape_correct_count"], 4)
        self.assertEqual(result["shape_accuracy"], 1.0)
        self.assertEqual(result["invalid_grid_rate"], 0.0)
        self.assertEqual(result["cell_accuracy"], 0.375)  # Mean of 0, 0, .75, .75, not 6/10 cells.
        self.assertEqual(forward.call_count, 12)
        self.assertEqual(result["actual_forward_calls"], 12)
        self.assertEqual(result["flux_forward_calls"], 12)
        self.assertEqual(result["target_forward_evaluations"], 12)
        self.assertEqual(result["nvarc_tokens"], 0)
        self.assertEqual(result["flux_text_tokens"], (3 + 5) * 2 * 3)
        self.assertEqual(result["flux_image_tokens"], 2 * 900 * 2 * 3)
        self.assertEqual(result["flux_joint_tokens"], (3 + 5 + 2 * 900) * 2 * 3)
        self.assertEqual(decode.call_count, 4)
        self.assertEqual(len(decoder.states), 4)
        self.assertEqual(result["queries"][1]["samples"][0]["class_canvas"],
                         large_partial.class_canvas.tolist())
        json.dumps(result, allow_nan=False)

    def test_sampling_preserves_invalid_canvas_and_omits_undefined_cell_accuracy(self):
        canvas = _class_canvas(((1, 1), (1, 1)))
        canvas[0, 1] = PAD_CLASS
        invalid = decode_arc_grids(canvas.unsqueeze(0))[0]
        self.assertFalse(invalid.valid)
        wrong_shape = _prediction(((1,),))
        with patch.object(validation, "generate_packed_arc_samples",
                          return_value=((invalid, wrong_shape),)) as generate:
            result = self.generate((_target(side=2),), ((0,),))
        generate.assert_called_once()
        self.assertEqual(result["prediction_count"], 2)
        self.assertEqual(result["shape_correct_count"], 0)
        self.assertEqual(result["shape_accuracy"], 0.0)
        self.assertEqual(result["invalid_grid_rate"], 0.5)
        self.assertIsNone(result["cell_accuracy"])
        sample = result["queries"][0]["samples"][0]
        self.assertFalse(sample["prediction_valid"])
        self.assertEqual(sample["class_canvas"], canvas.tolist())
        self.assertEqual(len(sample["class_canvas"]), 30)
        self.assertTrue(all(len(row) == 30 for row in sample["class_canvas"]))
        restored = json.loads(json.dumps(result, allow_nan=False))
        self.assertEqual(restored["queries"][0]["samples"][0]["class_canvas"], canvas.tolist())

    def test_restores_mode_after_failures_and_rejects_nonfinite_loss(self):
        target = _target()
        for module, evaluate in ((validation, lambda: self.loss((target,), ((0,),))),
                                 (generation, lambda: self.generate((target,), ((0,),)))):
            for initial_mode in (True, False):
                self.transformer.train(initial_mode)
                with patch.object(module, "predict_packed_flux_velocity", side_effect=RuntimeError("failure")):
                    with self.assertRaisesRegex(RuntimeError, "failure"):
                        evaluate()
                self.assertEqual(self.transformer.training, initial_mode)
        with patch.object(validation, "predict_packed_flux_velocity",
                          side_effect=lambda _, x, *a, **k: torch.full_like(x, float("nan"))):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                self.loss((target,), ((0,),))

    def test_rejects_invalid_protocol_before_forward(self):
        target = _target()
        with patch.object(validation, "predict_packed_flux_velocity") as forward:
            for packs in ((), ((0,), (0,)), ((), (0,)), ((1,),), ((True,),)):
                with self.subTest(packs=packs), self.assertRaises(ValueError):
                    self.loss((target,), packs)
            with self.assertRaises(ValueError):
                self.loss((target,), ((0,),), timesteps=(.5, 1.))
            with self.assertRaises(ValueError):
                self.generate((target,), ((0,),), seeds=(50, 50))
            with self.assertRaises(ValueError):
                self.generate((target,), ((0,),), steps=0)
            forward.assert_not_called()


    def test_exact_grid_diagnostics_distinguish_all_failure_modes(self) -> None:
        exact = _class_canvas(((1, 2), (3, 4)))
        wrong_cell = _class_canvas(((1, 2), (3, 0)))
        wrong_shape = _class_canvas(((1, 2),))
        all_pad = torch.full_like(exact, PAD_CLASS)
        predictions = decode_arc_grids(
            torch.stack((exact, wrong_cell, wrong_shape, all_pad))
        )
        expected = ((1, 2), (3, 4))

        diagnostics = validation.diagnose_exact_grids(
            predictions,
            (expected, expected, expected, expected),
        )

        self.assertTrue(diagnostics[0].exact_grid)
        self.assertEqual(diagnostics[0].cell_accuracy, 1.0)
        self.assertTrue(diagnostics[1].shape_match)
        self.assertEqual(diagnostics[1].cell_accuracy, 0.75)
        self.assertFalse(diagnostics[1].exact_grid)
        self.assertEqual(diagnostics[2].predicted_shape, (1, 2))
        self.assertFalse(diagnostics[2].shape_match)
        self.assertIsNone(diagnostics[2].cell_accuracy)
        self.assertFalse(diagnostics[3].prediction_valid)
        self.assertEqual(diagnostics[3].invalid_reason, "all_pad")
        self.assertIsNone(diagnostics[3].predicted_shape)
        self.assertFalse(diagnostics[3].exact_grid)


if __name__ == "__main__":
    unittest.main()
