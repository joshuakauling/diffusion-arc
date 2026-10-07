"""Exercise the real diagnostic sequence with small CPU models before CUDA use."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusion_arc.training.distributed import TrainingRuntime
from diffusion_arc.training.config import parse_training_config
from diffusion_arc.training.numerical_runner import run_numerical_phase, compare_numerical_phases
from diffusion_arc.training.run.numerics import NumericalCriteria
from diffusion_arc.training.step.flux_batching import FluxMicrobatchPlan
from tests.model.test_flux_packing import SegmentedReference
from tests.training.test_ddp_runner import Sources, _config, _plain_reference
from tests.training.test_ddp_trainer import _model, _render, FrozenText, FrozenVae


class UnevenSources(Sources):
    """Give rank one longer prefixes so FSDP must pad rank zero's pack schedule."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if index % 4 >= 2:
            sample = replace(sample, prefix_token_ends=(100, 200, 300, 400),
                             input_ids=tuple(index * 11 + i for i in range(401)))
        return sample


def _uneven_packs(prefixes, *_args, **_kwargs):
    """Prescribe small packs to exercise dummy backwards without large CPU attention."""
    width = 1 if max(prefixes) >= 100 else 2
    return tuple(FluxMicrobatchPlan(tuple(range(start, min(start + width, len(prefixes)))),
                                   sum(prefix + 900 for prefix in prefixes[start:start + width]))
                 for start in range(0, len(prefixes), width))


def _execute(phase, directory, initial, rank=0, *, changed_initial=False, include_fsdp2=False):
    torch.set_num_threads(1)
    backend, model = SegmentedReference(), _model(initial)
    if changed_initial:
        with torch.no_grad():
            next(model.parameters()).add_(0.1)
    models = SimpleNamespace(tokenizer=object(), flux_transformer=model, nvarc_encoder=FrozenText(),
                             vae_encoder=FrozenVae(), flux_compute_dtype=torch.float32,
                             flux_packing_backend=backend)
    config = _config(source_count=12, world_size=2)
    if include_fsdp2:
        recipe = config.recipe_dict()
        recipe["execution"].update(distributed_strategy="fsdp2", flux_packing_algorithm="legacy_bfd_v1",
                                   fsdp_reshard_after_forward=False)
        config = parse_training_config(recipe, config_path=config.path)

    @contextmanager
    def runtime(phase_config, **kwargs):
        dist.init_process_group("gloo", init_method=(Path(directory) / f"gloo-store-{phase}").as_uri(),
                                rank=rank, world_size=2, timeout=timedelta(seconds=60))
        try:
            yield TrainingRuntime(rank, rank, 2, torch.device("cpu"), dist.group.WORLD,
                                  phase_config.recipe["execution"]["distributed_strategy"])
        finally:
            dist.destroy_process_group()

    with patch("diffusion_arc.training.numerical_runner.prepare_device", return_value=torch.device("cpu")), \
         patch("diffusion_arc.training.numerical_runner.distributed_runtime", runtime), \
         patch("diffusion_arc.training.numerical_runner.load_models", return_value=models), \
         patch("diffusion_arc.training.numerical_runner.version", return_value="test"), \
         patch("diffusion_arc.training.numerical_runner.print", create=True), \
         patch("diffusion_arc.training.distributed_data.ArcTrainingDataset",
               return_value=(UnevenSources if include_fsdp2 else Sources)(source_count=12)), \
         (patch("diffusion_arc.training.trainer.plan_flux_batches", side_effect=_uneven_packs)
          if include_fsdp2 else nullcontext()), \
         patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render):
        run_numerical_phase(SimpleNamespace(training=config, expected_source_count=12), NumericalCriteria(),
                            phase=phase, output_directory=Path(directory) / "output",
                            scratch_directory=Path(directory) / "scratch",
                            source_identity=dict(source_sha256="test-source"))


def _worker(rank, directory, initial, phase="ddp", include_fsdp2=False):
    _execute(phase, directory, initial, rank, include_fsdp2=include_fsdp2)


def _run_distributed(directory, initial, *, phase="ddp", include_fsdp2=False):
    processes = mp.spawn(_worker, args=(directory, initial, phase, include_fsdp2), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 120
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("numerical diagnostic CPU test exceeded 120 seconds")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


class NumericalRunnerTests(unittest.TestCase):
    def test_plain_reference_then_two_ranks_and_actual_artifact_comparison(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)
        torch.manual_seed(167)
        initial = deepcopy(_model().state_dict())
        expected, losses = _plain_reference(initial, source_count=12)
        with tempfile.TemporaryDirectory() as directory:
            _execute("reference", directory, initial)
            root = Path(directory) / "output"
            rows = [json.loads(s) for s in (root / "reference/steps.jsonl").read_text().splitlines()]
            torch.testing.assert_close(torch.tensor([r["loss"] for r in rows]), torch.tensor(losses))
            for row in rows:
                for rank in row["ranks"]:
                    self.assertEqual(set(rank["packs"]),
                                     {"preparation", "flux_batches", "target_prefix_tokens"})
                    self.assertEqual(sorted(index for plan in rank["packs"]["flux_batches"]
                                            for index in plan["target_indices"]),
                                     list(range(len(rank["target_metrics"]))))
            self.assertFalse(json.loads((root / "reference/manifest.json").read_text())["ddp"])
            self.assertFalse(dist.is_initialized())
            _run_distributed(directory, initial)
            result = compare_numerical_phases(root, NumericalCriteria())
            self.assertEqual(result["status"], "passed", result["failures"])
            self.assertLess(result["steps"][0]["gradient"]["relative_l2"], 1e-4)
            self.assertTrue(all(r["checks"]["rank_parameters"] for r in result["steps"]))
            manifest_path = root / "ddp/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["contract"]["initial"]["flux"]["sha256"] = "different"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "contracts differ"):
                compare_numerical_phases(root, NumericalCriteria())

    def test_three_strategies_compare_full_gradients_with_real_fsdp_shards_and_dummy_calls(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)
        torch.manual_seed(167)
        initial = deepcopy(_model().state_dict())
        with tempfile.TemporaryDirectory() as directory:
            _execute("reference", directory, initial, include_fsdp2=True)
            _run_distributed(directory, initial, include_fsdp2=True)
            _run_distributed(directory, initial, phase="fsdp2", include_fsdp2=True)
            root = Path(directory) / "output"
            result = compare_numerical_phases(root, NumericalCriteria(), include_fsdp2=True)
            self.assertEqual(result["status"], "passed", result["failures"])
            self.assertEqual(set(result["comparisons"]),
                             {"single_gpu_vs_ddp", "single_gpu_vs_fsdp2", "ddp_vs_fsdp2"})
            for comparison in result["comparisons"].values():
                self.assertLess(comparison["steps"][0]["gradient"]["relative_l2"], 1e-4)
            rows = [json.loads(line) for line in (root / "fsdp2/steps.jsonl").read_text().splitlines()]
            for row in rows:
                self.assertGreater(row["ranks"][0]["padding_forward_calls"], 0)
                self.assertEqual(row["ranks"][1]["padding_forward_calls"], 0)
                self.assertNotEqual(row["ranks"][0]["parameters"], row["ranks"][1]["parameters"])
                self.assertEqual(sum(rank["parameters"]["elements"] for rank in row["ranks"]),
                                 sum(t.numel() for t in initial.values()))
            self.assertNotIn("rank_parameters", result["comparisons"]["single_gpu_vs_fsdp2"]["steps"][0]["checks"])
            manifest = json.loads((root / "reference/manifest.json").read_text())
            self.assertEqual(manifest["effective_recipe"]["execution"]["distributed_strategy"], "single_process")
            self.assertEqual(manifest["effective_recipe"]["deployment"]["gpu_count"], 1)
            self.assertNotIn("fsdp_reshard_after_forward", manifest["effective_recipe"]["execution"])
            self.assertFalse(manifest["contract"]["input_recipe"]["execution"]["fsdp_reshard_after_forward"])
            fsdp_manifest = root / "fsdp2/manifest.json"
            manifest = json.loads(fsdp_manifest.read_text())
            self.assertFalse(manifest["effective_recipe"]["execution"]["fsdp_reshard_after_forward"])
            manifest["contract"]["initial"]["flux"]["sha256"] = "different"
            fsdp_manifest.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "contracts differ"):
                compare_numerical_phases(root, NumericalCriteria(), include_fsdp2=True)


if __name__ == "__main__":
    unittest.main()
