"""Training and benchmark entrypoints share settings, updates and stop boundaries."""

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from diffusion_arc.training import training_loop, runner
from diffusion_arc.training.benchmark_runner import run_benchmark
from diffusion_arc.training.config import load_training_config, parse_training_config
from diffusion_arc.training.distributed import TrainingRuntime
from diffusion_arc.training.run.experiment import BenchmarkInstrumentation
from diffusion_arc.training.step.flow_sampling import UniformFlowSampler, build_flow_policy
from diffusion_arc.training.step.loss import velocity_mse


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/training/benchmarks/single_gpu_b3_smoke.json'


class QuietExperiment(BenchmarkInstrumentation):
    """Record public measurement boundaries without touching GPU instrumentation."""

    def start(self, stack, config, models, dataset):
        self.seen_config = config
        self.updates = []

    def before_step(self):
        pass

    def completed_step(self):
        pass

    def after_step(self, run_step, batch, result, progress):
        self.updates.append((batch.epoch_batch_index, result.loss, progress.optimizer_step))

    def finish(self, progress):
        self.final = progress


class SharedTrainingLoopTests(unittest.TestCase):
    def config(self, **loop):
        payload = load_training_config(CONFIG).recipe_dict()
        payload['data']['expected_task_count'] = 4
        payload['benchmark'] = None
        payload['loop'].update(optimizer_batch_size=3, max_steps=2, **loop)
        return parse_training_config(payload, config_path=CONFIG)

    def harness(self, stack, config, *, available=4):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = Mock()
        sequences = [SimpleNamespace(target_grids=(1, 2, 3, 4)) for _ in range(available)]
        dataset = SimpleNamespace(keys=[SimpleNamespace(task_id=str(i),
            augmentation=SimpleNamespace(pair_indices=tuple(range(5)))) for i in range(4)],
            fingerprint='prepared-manifest-id')
        class Loader:
            def __iter__(self):
                return iter(sequences)
        loader = Loader()
        loader.dataset = dataset
        models = SimpleNamespace(flux_transformer=model, tokenizer=object(), nvarc_encoder=object(),
            vae_encoder=object(), flux_compute_dtype=torch.bfloat16, flux_packing_backend=object())
        for name, value in [('load_models', models), ('build_dataloader', loader),
                            ('build_optimizer', optimizer), ('build_lr_scheduler', scheduler)]:
            stack.enter_context(patch.object(training_loop, name, return_value=value))
        # A tiny actual gradient update exercises the shared loop without model assets.
        def step(batch):
            loss = model.weight.square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            return SimpleNamespace(loss=float(loss.detach()), source_count=1, target_count=3,
                step_seconds=1.0, padding_forward_calls=0,
                nvarc_tokens=5, flux_text_tokens=6, flux_image_tokens=2700,
                flux_joint_tokens=2706, flux_forward_calls=1)
        trainer = stack.enter_context(patch.object(training_loop, 'Trainer', return_value=SimpleNamespace(step=step)))
        return model, scheduler, trainer

    def test_training_and_benchmark_have_identical_updates_and_explicit_components(self):
        config = self.config()
        runtime = TrainingRuntime(0, 0, 1, torch.device('cpu'), None, 'single_process')
        with ExitStack() as stack:
            model, scheduler, trainer = self.harness(stack, config)
            logger = stack.enter_context(patch.object(runner, 'WandbLogger'))
            saves = stack.enter_context(patch.object(runner, 'save_checkpoint'))
            train_progress = runner.run_training(config, runtime=runtime)
            training_weight = model.weight.detach().clone()
            self.assertEqual(scheduler.step.call_count, 2)
            self.assertIs(trainer.call_args.kwargs['loss_function'], velocity_mse)
            self.assertIsInstance(trainer.call_args.kwargs['flow_sampler'], UniformFlowSampler)
            logger.assert_not_called()
            saves.assert_not_called()
        experiment = QuietExperiment(max_steps=2)
        with ExitStack() as stack:
            model, scheduler, trainer = self.harness(stack, config)
            benchmark_progress = run_benchmark(config, experiment=experiment, runtime=runtime)
            torch.testing.assert_close(model.weight, training_weight)
            self.assertEqual(scheduler.step.call_count, 2)
            self.assertEqual([row[0] for row in experiment.updates], [0, 1])
            self.assertIs(experiment.seen_config, config)
        self.assertEqual(train_progress, benchmark_progress)
        self.assertEqual((train_progress.optimizer_step, train_progress.sources_seen), (2, 2))
        self.assertEqual(config.epoch_optimizer_steps, 4)

    def test_short_stream_fails_instead_of_reporting_a_complete_run(self):
        config = self.config()
        runtime = TrainingRuntime(0, 0, 1, torch.device('cpu'), None, 'single_process')
        with ExitStack() as stack:
            self.harness(stack, config, available=1)
            with self.assertRaisesRegex(ValueError, 'exhausted'):
                runner.run_training(config, runtime=runtime)

    def test_uniform_flow_has_zero_endpoint_quotas_for_indexed_execution(self):
        config = self.config()
        policy = build_flow_policy(config)
        self.assertEqual(policy.endpoint_count(3), 0)
        self.assertEqual(policy.high_noise_count(3), 0)

    def test_runtime_must_match_explicit_strategy(self):
        config = self.config()
        runtime = TrainingRuntime(0, 0, 2, torch.device('cpu'), None, 'ddp')
        with self.assertRaisesRegex(ValueError, 'differs'):
            with training_loop.runtime_context(config, runtime):
                pass


if __name__ == '__main__':
    unittest.main()
