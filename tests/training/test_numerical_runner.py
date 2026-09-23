"""Exercise the real diagnostic sequence with small CPU models before CUDA use."""

from contextlib import contextmanager
from copy import deepcopy
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

from diffusion_arc.training.distributed import DistributedRuntime
from diffusion_arc.training.numerical_runner import run_numerical_phase, compare_numerical_phases
from diffusion_arc.training.run.numerics import NumericalCriteria
from tests.model.test_flux_packing import SegmentedReference
from tests.training.test_ddp_runner import Sources, _config, _plain_reference
from tests.training.test_ddp_trainer import _model, _render, FrozenText, FrozenVae


def _execute(phase, directory, initial, rank=0, *, changed_initial=False):
    torch.set_num_threads(1)
    backend, model = SegmentedReference(), _model(initial)
    if changed_initial:
        with torch.no_grad():
            next(model.parameters()).add_(0.1)
    models = SimpleNamespace(tokenizer=object(), flux_transformer=model, nvarc_encoder=FrozenText(),
                             vae_encoder=FrozenVae(), flux_compute_dtype=torch.float32,
                             flux_packing_backend=backend)

    @contextmanager
    def runtime(_, **kwargs):
        dist.init_process_group("gloo", init_method=(Path(directory) / "gloo-store").as_uri(),
                                rank=rank, world_size=2, timeout=timedelta(seconds=60))
        try:
            yield DistributedRuntime(rank, rank, 2, torch.device("cpu"), dist.group.WORLD)
        finally:
            dist.destroy_process_group()

    with patch("diffusion_arc.training.numerical_runner.prepare_device", return_value=torch.device("cpu")), \
         patch("diffusion_arc.training.numerical_runner.distributed_runtime", runtime), \
         patch("diffusion_arc.training.numerical_runner.load_models", return_value=models), \
         patch("diffusion_arc.training.numerical_runner.version", return_value="test"), \
         patch("diffusion_arc.training.numerical_runner.print", create=True), \
         patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=Sources(source_count=12)), \
         patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render):
        run_numerical_phase(SimpleNamespace(training=_config(), expected_source_count=12), NumericalCriteria(),
                            phase=phase, output_directory=Path(directory) / "output",
                            scratch_directory=Path(directory) / "scratch",
                            source_identity=dict(source_sha256="test-source"))


def _worker(rank, directory, initial):
    _execute("ddp", directory, initial, rank)


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
            self.assertFalse(json.loads((root / "reference/manifest.json").read_text())["ddp"])
            self.assertFalse(dist.is_initialized())
            processes = mp.spawn(_worker, args=(directory, initial), nprocs=2, join=False)
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


if __name__ == "__main__":
    unittest.main()
