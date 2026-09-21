"""Packed trainer ownership, numerical updates, and recipe integration."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import Tensor, nn

from diffusion_arc.arc_data.augmentation import ArcAugmentationKey, ArcAugmentationSpec, ArcGeometry
from diffusion_arc.arc_data.text.tokenization import ArcTokenizedTrainingSample
from diffusion_arc.model import NvarcBatchOutput, TARGET_LATENT_CHANNELS, TARGET_LATENT_TOKEN_COUNT
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.step.flow_sampling import EndpointFlowPolicy, EndpointFlowSampler, UniformFlowSampler, build_flow_sampler
from diffusion_arc.training.step.flux_microbatch import build_flux_microbatch, compute_flux_microbatch_loss
from diffusion_arc.training.step.flux_batching import FluxPackingPolicy
from diffusion_arc.training.step.loss import LossResult, RegionLossWeights, RegionVelocityLoss, compute_region_velocity_loss, velocity_mse, build_velocity_loss
from diffusion_arc.training.step.lr_schedule import LinearWarmupConstantLR, build_lr_scheduler
from diffusion_arc.training.setup import TrainingModels
from diffusion_arc.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[2]


def baseline_config(**loop):
    config = load_training_config(ROOT / "configs/training/benchmarks/single_gpu_b3.json")
    settings = dict(max_optimizer_steps=3, optimizer_batch_size=4,
                    min_context_pairs_before_target=1)
    settings.update(loop)
    return replace(config, loop=replace(config.loop, **settings))


def models_for(transformer, *, nvarc=None, vae=None):
    return TrainingModels(tokenizer=None, nvarc_encoder=nvarc if nvarc is not None else RecordingNvarcEncoder(),
                          vae_encoder=vae if vae is not None else NumberedVaeEncoder(), vae_decoder=None,
                          flux_transformer=transformer, flux_compute_dtype=torch.bfloat16,
                          flux_packing_backend=object())

class RecordingNvarcEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[Tensor, Tensor]] = []

    def forward(
        self,
        input_ids: Tensor,
        sequence_offsets: Tensor,
    ) -> NvarcBatchOutput:
        self.calls.append((input_ids.clone(), sequence_offsets.clone()))
        hidden_states = torch.arange(
            input_ids.numel() * 4,
            dtype=torch.float32,
        ).reshape(input_ids.numel(), 4)
        offsets = sequence_offsets.tolist()
        return NvarcBatchOutput(
            sequence_hidden_states=tuple(
                hidden_states[start:end].clone()
                for start, end in zip(offsets, offsets[1:])
            )
        )


class NumberedVaeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[Tensor] = []
        self.next_target_value = 1

    def forward(self, target_images: Tensor) -> Tensor:
        self.calls.append(target_images.clone())
        clean_latents = []
        for _ in range(target_images.shape[0]):
            clean_latents.append(
                torch.full(
                    (TARGET_LATENT_TOKEN_COUNT, TARGET_LATENT_CHANNELS),
                    float(self.next_target_value),
                )
            )
            self.next_target_value += 1
        return torch.stack(clean_latents)


class ScalarTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


def _sample(target_count: int) -> ArcTokenizedTrainingSample:
    prefix_ends = tuple(2 * (index + 1) for index in range(target_count))
    return ArcTokenizedTrainingSample(
        key=ArcAugmentationKey(
            task_id="abc123ef",
            augmentation=ArcAugmentationSpec(
                geometry=ArcGeometry.IDENTITY,
                color_mapping=tuple(range(10)),
                pair_indices=tuple(range(target_count + 1)),
            ),
        ),
        input_ids=tuple(range(prefix_ends[-1] + 1)),
        prefix_token_ends=prefix_ends,
        target_grids=tuple(((index,),) for index in range(target_count)),
    )


class ToyVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.bias = torch.nn.Parameter(torch.tensor(-0.1))

    def forward(self, state):
        return state * self.weight + self.bias


class ScaledVae(NumberedVaeEncoder):
    def forward(self, images):
        return super().forward(images) / 100


def _samples(count=26):
    shapes = ((1, 1), (29, 29), (30, 30), (30, 1), (1, 30), (7, 13))
    samples = []
    for index in range(count):
        sample = _sample(4)
        ends = tuple(value + (index % 4) * 200 for value in (300, 900, 1900, 3900))
        grids = []
        for target in range(4):
            height, width = shapes[(index * 3 + target) % len(shapes)]
            grids.append(tuple(tuple((index + target) % 10 for _ in range(width)) for _ in range(height)))
        samples.append(replace(sample, prefix_token_ends=ends, input_ids=(0,) * (ends[-1] + 1),
                               target_grids=tuple(grids)))
    return tuple(samples)


def make_trainer(models, optimizer, *, sampler=None, scheduler=None, **settings):
    return Trainer(
        nvarc_encoder=models.nvarc_encoder, vae_encoder=models.vae_encoder,
        flux_transformer=models.flux_transformer, optimizer=optimizer, scheduler=scheduler,
        flow_sampler=sampler if sampler is not None else UniformFlowSampler(torch.Generator().manual_seed(17)),
        loss_function=settings.pop('loss_function', velocity_mse),
        flux_compute_dtype=models.flux_compute_dtype, flux_packing_backend=models.flux_packing_backend,
        **settings,
    )


class RecordingLoss:
    def __init__(self):
        self.owners = []

    def __call__(self, inputs, flow, *, transformer, max_joint_tokens, **kwargs):
        self.owners.extend(weakref.ref(item.nvarc_sequence_state) for item in inputs)
        build_flux_microbatch(inputs, max_joint_tokens=max_joint_tokens)
        values = transformer.transformer.weight * torch.stack(tuple(item.clean_latent[0, 0] for item in inputs))
        return LossResult(values.mean(), values)


class PackedTrainerTests(unittest.TestCase):
    def test_one_call_encodes_one_update_and_returns_target_measurements_in_logical_order(self):
        model = ScalarTransformer()
        model.weight.grad = torch.tensor(100.0)
        models = models_for(model)
        for module in (models.nvarc_encoder, models.vae_encoder):
            module.register_forward_pre_hook(lambda *_args: self.assertIsNone(model.weight.grad))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = make_trainer(models, optimizer, nvarc_max_tokens=4, vae_microbatch_size=2)
        self.assertIsNone(model.weight.grad)
        self.assertTrue(model.training)
        self.assertFalse(models.nvarc_encoder.training)
        self.assertFalse(models.vae_encoder.training)
        compute = RecordingLoss()
        with patch('diffusion_arc.training.step.encoding._render_valid_grid_canvas', return_value=torch.zeros(1,1,1)), \
             patch('diffusion_arc.training.trainer.compute_flux_microbatch_loss', new=compute):
            first = trainer.step((_sample(2), _sample(1)))
            self.assertIsNone(model.weight.grad)
            self.assertAlmostEqual(float(model.weight.detach()), 0.8)
            second = trainer.step((_sample(2),))
        self.assertEqual([first.target_count, second.target_count], [3,2])
        self.assertEqual([first.source_count, second.source_count], [2,1])
        self.assertEqual(first.diagnostics.preparation.nvarc_batches, ((0,), (1,)))
        self.assertEqual((first.nvarc_calls, first.nvarc_tokens, first.vae_calls), (2,6,2))
        self.assertEqual((first.flux_text_tokens, first.flux_image_tokens, first.flux_joint_tokens), (8,2700,2708))
        self.assertEqual(first.flux_forward_calls, len(first.diagnostics.flux_batches))
        self.assertEqual(first.diagnostics.target_prefix_tokens, (2,4,2))
        self.assertEqual(first.diagnostics.flux_batches[0].target_indices, (1,0,2))
        self.assertEqual(tuple(target.loss for target in first.targets), (1.0,2.0,3.0))
        self.assertEqual(tuple(target.context_pair_count for target in first.targets), (1,2,1))
        for target in first.targets:
            self.assertLess(target.timestep, 1)
            self.assertTrue(target.has_boundary)
            for field in ('mse_canvas', 'mse_grid', 'mse_boundary', 'contribution_canvas',
                          'contribution_grid', 'contribution_boundary'):
                self.assertIsNone(getattr(target, field))
        self.assertAlmostEqual(first.loss, 2.0)
        self.assertAlmostEqual(second.loss, 3.6, places=6)
        self.assertAlmostEqual(second.gradient_norm, 4.5)
        self.assertAlmostEqual(float(model.weight.detach()), 0.35, places=6)
        self.assertGreater(first.step_seconds, 0)
        self.assertIsNone(model.weight.grad)
        self.assertTrue(all(ref() is None for ref in compute.owners))
        # A directly serializable result cannot retain GPU tensors or autograd owners.
        json.dumps(asdict(first), allow_nan=False)
        self.assertFalse(hasattr(trainer, 'train'))
        self.assertFalse(hasattr(trainer, 'config'))

    def test_clipping_and_used_learning_rate_follow_exactly_one_update(self):
        model = ScalarTransformer()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = LinearWarmupConstantLR(optimizer, warmup_steps=2)
        trainer = make_trainer(models_for(model), optimizer, scheduler=scheduler, max_grad_norm=1.0)
        with patch('diffusion_arc.training.step.encoding._render_valid_grid_canvas', return_value=torch.zeros(1,1,1)), \
             patch('diffusion_arc.training.trainer.compute_flux_microbatch_loss', side_effect=RecordingLoss()):
            first = trainer.step((_sample(2),))
            second = trainer.step((_sample(2),))
        self.assertEqual(first.learning_rates, (0.05,))
        self.assertEqual(second.learning_rates, (0.1,))
        self.assertEqual(first.gradient_norm, 1.5)
        self.assertTrue(first.gradient_clipped)
        self.assertEqual(second.gradient_norm, 3.5)
        self.assertAlmostEqual(float(model.weight.detach()), 0.85, places=6)
        self.assertEqual(scheduler.last_epoch, 2)

    def test_nonfinite_loss_or_gradient_aborts_before_optimizer_and_scheduler(self):
        for failure in ('loss', 'gradient'):
            with self.subTest(failure=failure):
                model = ScalarTransformer()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                scheduler = LinearWarmupConstantLR(optimizer, warmup_steps=2)
                trainer = make_trainer(models_for(model), optimizer, scheduler=scheduler)
                calls = []
                def compute(inputs, flow, **kwargs):
                    result = RecordingLoss()(inputs, flow, **kwargs)
                    calls.append(1)
                    if failure == 'gradient':
                        value = torch.sqrt(model.weight - model.weight)
                        return LossResult(value, value.expand(len(inputs)))
                    if len(calls) == 2:
                        return LossResult(result.loss * float('inf'), result.per_target)
                    return result
                sample = replace(_sample(2), input_ids=(0,) * 8401, prefix_token_ends=(8000,8400))
                with patch('diffusion_arc.training.step.encoding._render_valid_grid_canvas', return_value=torch.zeros(1,1,1)), \
                     patch('diffusion_arc.training.trainer.compute_flux_microbatch_loss', side_effect=compute):
                    error = FloatingPointError if failure == 'loss' else RuntimeError
                    with self.assertRaisesRegex(error, 'non-finite'):
                        trainer.step((sample,))
                self.assertEqual(float(model.weight.detach()), 1.0)
                self.assertEqual(scheduler.last_epoch, 0)
                if failure == 'loss':
                    self.assertEqual(len(calls), 2)

    def test_empty_or_ineligible_step_does_no_model_or_optimizer_work(self):
        model = ScalarTransformer()
        models = models_for(model)
        trainer = make_trainer(models, torch.optim.SGD(model.parameters(), lr=0.1),
                               min_context_pairs_before_target=2)
        for sources in ((), (_sample(1),)):
            with self.assertRaises(ValueError):
                trainer.step(sources)
        self.assertEqual(models.nvarc_encoder.calls, [])
        self.assertEqual(models.vae_encoder.calls, [])
        self.assertEqual(float(model.weight.detach()), 1.0)
        self.assertIsNone(model.weight.grad)

    def test_constructor_rejects_invalid_component_ownership_and_execution_limits(self):
        model = ScalarTransformer()
        models = models_for(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        foreign = torch.optim.SGD(ScalarTransformer().parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, 'scheduler must belong'):
            make_trainer(models, optimizer, scheduler=LinearWarmupConstantLR(foreign, warmup_steps=2))
        with self.assertRaisesRegex(ValueError, 'preflighted'):
            make_trainer(replace(models, flux_packing_backend=None), optimizer)
        with self.assertRaisesRegex(ValueError, 'frozen'):
            make_trainer(replace(models, nvarc_encoder=nn.Linear(1,1)), optimizer)
        for settings in ({'nvarc_max_tokens':0}, {'vae_microbatch_size':0},
                         {'min_context_pairs_before_target':0}, {'max_grad_norm':float('nan')}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                make_trainer(models, optimizer, **settings)

    def test_sampler_and_loss_are_supplied_independently(self):
        results = {}
        for endpoint in (False, True):
            for regional in (False, True):
                model = ToyVelocity()
                generator = torch.Generator().manual_seed(167)
                sampler = (EndpointFlowSampler(generator, EndpointFlowPolicy(6,2,3,1)) if endpoint
                           else UniformFlowSampler(generator))
                draws = []
                def sample(latents):
                    flows = sampler(latents)
                    draws.extend(tuple((f.timesteps.clone(), f.noisy_latents.clone()) for f in flows))
                    return flows
                trainer = make_trainer(
                    models_for(model, vae=ScaledVae()), torch.optim.SGD(model.parameters(),lr=0.001),
                    sampler=sample, min_context_pairs_before_target=2,
                    loss_function=RegionVelocityLoss(RegionLossWeights()) if regional else velocity_mse,
                )
                with patch('diffusion_arc.training.step.encoding._render_valid_grid_canvas', return_value=torch.zeros(1,1,1)), \
                     patch('diffusion_arc.model.flux.packing.predict_packed_flux_velocity', side_effect=lambda m,z,*a,**k:m(z)):
                    metrics = [trainer.step(_samples(3)[:2]), trainer.step(_samples(3)[2:])]
                self.assertEqual([sum(t.timestep == 1 for t in m.targets) for m in metrics],
                                 [2,1] if endpoint else [0,0])
                for step, boundaries in zip(metrics, ((True,False,True,True,True,True),
                                                     (True,False,True)), strict=True):
                    for target in step.targets:
                        self.assertEqual(target.mse_grid is not None, regional)
                    self.assertEqual(tuple(t.has_boundary for t in step.targets), boundaries)
                results[endpoint,regional] = draws, generator.get_state(), metrics
        for endpoint in (False,True):
            plain, regional = results[endpoint,False], results[endpoint,True]
            torch.testing.assert_close(plain[:2], regional[:2], rtol=0, atol=0)
            self.assertNotEqual(plain[2][0].loss, regional[2][0].loss)


class TrainingRecipeTests(unittest.TestCase):
    def setUp(self):
        self.samples = _samples()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        payload = json.loads((ROOT / 'configs/training/arc_agi_1_epoch.json').read_text())
        payload['data']['expected_task_count'] = 26
        payload['loop'].update(max_optimizer_steps=2, expected_target_count=78,
                               expected_final_batch_target_count=30)
        payload['scheduler'].update(warmup_steps=2, first_update_warmup_fraction=0.5)
        path = Path(self.temp.name) / 'recipe.json'
        path.write_text(json.dumps(payload))
        self.config = load_training_config(path, expected_type='training_run')

    def run_core(self, *, packing=12288, nvarc_budget=134400, vae_size=48):
        model = ToyVelocity()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = build_lr_scheduler(self.config, optimizer=optimizer)
        generator = torch.Generator().manual_seed(167)
        metrics, plans, flows, shapes, updates = [], [], {}, {}, []
        def record_loss(inputs, flow, **kwargs):
            for slot, target in enumerate(inputs):
                identity = round(float(target.clean_latent[0,0]) * 100)
                flows[identity] = tuple(getattr(flow, field)[slot].clone() for field in
                                        ('noisy_latents', 'timesteps', 'target_velocity'))
                shapes[identity] = target.grid_shape
            return compute_flux_microbatch_loss(inputs, flow, **kwargs)
        trainer = make_trainer(
            models_for(model, vae=ScaledVae()), optimizer, scheduler=scheduler,
            sampler=build_flow_sampler(self.config,generator=generator),
            loss_function=build_velocity_loss(self.config), min_context_pairs_before_target=2,
            max_grad_norm=1.0, flux_max_joint_tokens=packing,
            nvarc_max_tokens=nvarc_budget, vae_microbatch_size=vae_size,
        )
        with patch('diffusion_arc.training.step.encoding._render_valid_grid_canvas', return_value=torch.zeros(1,1,1)), \
             patch('diffusion_arc.training.trainer.compute_flux_microbatch_loss', side_effect=record_loss), \
             patch('diffusion_arc.model.flux.packing.predict_packed_flux_velocity', side_effect=lambda m,z,*a,**k:m(z)):
            for sources in (self.samples[:16], self.samples[16:]):
                step = trainer.step(sources)
                metrics.append(step)
                plans.append(step.diagnostics.flux_batches)
                updates.append(tuple(p.detach().clone() for p in model.parameters()))
        return metrics, plans, flows, shapes, updates, generator.get_state()

    def test_recipe_updates_and_each_target_measurement_match_unfragmented_reference(self):
        metrics, plans, flows, shapes, updates, _ = self.run_core()
        self.assertEqual([step.target_count for step in metrics], [48,30])
        self.assertEqual([step.learning_rates for step in metrics], [(5e-6,),(1e-5,)])
        self.assertEqual(sorted(flows), list(range(1,79)))
        reference = ToyVelocity()
        optimizer = torch.optim.AdamW(reference.parameters(),lr=1e-5)
        scheduler = LinearWarmupConstantLR(optimizer,warmup_steps=2)
        for index, identities in enumerate((range(1,49),range(49,79))):
            step = metrics[index]
            state = torch.stack([flows[i][0] for i in identities])
            velocity = torch.stack([flows[i][2] for i in identities])
            times = torch.stack([flows[i][1] for i in identities])
            regions = compute_region_velocity_loss(reference(state),velocity,[shapes[i] for i in identities])
            loss = regions.per_target.mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(reference.parameters(),1.0,foreach=False)
            optimizer.step()
            scheduler.step()
            self.assertAlmostEqual(float(loss.detach()),step.loss,delta=3e-6)
            self.assertAlmostEqual(float(norm),step.gradient_norm,delta=3e-6)
            self.assertEqual(step.gradient_clipped,bool(norm>1))
            self.assertEqual(step.flux_forward_calls,len(plans[index]))
            self.assertEqual(len(step.targets), step.target_count)
            self.assertEqual(sum(t.timestep == 1 for t in step.targets),4 if index==0 else 3)
            for slot, target in enumerate(step.targets):
                self.assertEqual(target.timestep, float(times[slot]))
                self.assertEqual(target.context_pair_count, 2 + slot % 3)
                self.assertEqual(target.has_boundary, bool(regions.has_boundary[slot]))
                self.assertAlmostEqual(target.loss, float(regions.per_target[slot].detach()), delta=3e-6)
                for column, region in enumerate(('canvas', 'grid', 'boundary')):
                    mean = getattr(target, 'mse_' + region)
                    if region == 'boundary' and not target.has_boundary:
                        self.assertIsNone(mean)
                    else:
                        self.assertAlmostEqual(mean, float(regions.region_means[slot,column].detach()), delta=3e-6)
                    self.assertAlmostEqual(getattr(target, 'contribution_' + region),
                                           float(regions.contributions[slot,column].detach()), delta=3e-6)
                self.assertAlmostEqual(target.contribution_canvas + target.contribution_grid
                                       + target.contribution_boundary, target.loss, delta=3e-6)
            for actual, expected in zip(updates[index],reference.parameters()):
                torch.testing.assert_close(actual,expected,rtol=2e-6,atol=1e-7)
            optimizer.zero_grad(set_to_none=True)
            json.dumps(asdict(step),allow_nan=False)

    def test_nvarc_vae_and_flux_splits_preserve_rng_targets_and_updates(self):
        reference = self.run_core(packing=1_000_000)
        for budget in (9300,12288):
            with self.subTest(budget=budget):
                packed = self.run_core(packing=budget,nvarc_budget=9000,vae_size=7)
                self.assertEqual(packed[3],reference[3])
                torch.testing.assert_close(packed[5],reference[5],rtol=0,atol=0)
                torch.testing.assert_close(packed[2],reference[2],rtol=0,atol=0)
                self.assertGreater(packed[0][0].nvarc_calls,1)
                self.assertEqual(packed[0][0].vae_calls,7)
                for actual, expected, update, ref_update in zip(packed[0],reference[0],packed[4],reference[4]):
                    self.assertAlmostEqual(actual.loss,expected.loss,delta=3e-6)
                    self.assertAlmostEqual(actual.gradient_norm,expected.gradient_norm,delta=3e-6)
                    torch.testing.assert_close(update,ref_update,rtol=2e-6,atol=1e-7)
                    for target, ref_target in zip(actual.targets, expected.targets, strict=True):
                        self.assertEqual(target.timestep, ref_target.timestep)
                        self.assertEqual(target.context_pair_count, ref_target.context_pair_count)
                        self.assertEqual(target.has_boundary, ref_target.has_boundary)
                        for field in ('loss', 'mse_canvas', 'mse_grid', 'mse_boundary',
                                      'contribution_canvas', 'contribution_grid', 'contribution_boundary'):
                            value, ref_value = getattr(target, field), getattr(ref_target, field)
                            if ref_value is None:
                                self.assertIsNone(value)
                            else:
                                self.assertAlmostEqual(value, ref_value, delta=3e-6)

    def test_absent_boundary_is_not_logged_as_zero_error(self):
        self.samples = tuple(replace(sample,target_grids=tuple(tuple((0,)*30 for _ in range(30))
                                    for _ in range(4))) for sample in self.samples)
        metrics, *_ = self.run_core()
        for step in metrics:
            for target in step.targets:
                self.assertFalse(target.has_boundary)
                self.assertIsNone(target.mse_boundary)
                self.assertEqual(target.contribution_boundary,0)
            self.assertGreater(step.loss,0)


if __name__ == '__main__':
    unittest.main()
