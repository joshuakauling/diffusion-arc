"""CPU validation of the diagnostic's isolation, grouping, and failure handling."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from diffusion_arc.model.nvarc_encoder import NvarcBatchRunner
from diffusion_arc.training.nvarc_schedule import NvarcSourceBatch
from scripts.diagnostics import probe_nvarc_packing as probe

CONFIG = probe.REPOSITORY_ROOT / "configs/diagnostics/nvarc_packing.json"


class PackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(19)
        config = Qwen3Config(
            vocab_size=16, hidden_size=16, intermediate_size=24, num_hidden_layers=36,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8, pad_token_id=None,
        )
        config._attn_implementation = "sdpa"
        cls.model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        cls.runner = NvarcBatchRunner(cls.model)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_packed_boundaries_positions_and_cpu_metadata(self):
        sources = (torch.tensor([1, 2, 3]), torch.tensor([4, 5]))
        args, offsets = probe.packed_arguments(sources, "cpu")
        self.assertEqual(offsets, [0, 3, 5])
        self.assertEqual(args["input_ids"].tolist(), [[1, 2, 3, 4, 5]])
        self.assertEqual(args["position_ids"].tolist(), [[0, 1, 2, 0, 1]])
        self.assertEqual(args["cu_seq_lens_q"].tolist(), [0, 3, 5])
        self.assertEqual(args["cu_seq_lens_q"].dtype, torch.int32)
        self.assertIs(args["cu_seq_lens_q"], args["cu_seq_lens_k"])
        self.assertEqual(args["max_length_q"], 3)
        self.assertIsNone(args["attention_mask"])

    def test_padding_uses_tokenizer_contract_when_model_config_has_no_pad(self):
        sources = (torch.tensor([1, 2, 3]), torch.tensor([4, 5]))
        layout = probe.packing_layout((3, 2), "source", 8)
        self.assertIsNone(self.model.config.pad_token_id)
        args, _ = probe.packed_arguments(sources, "cpu", layout=layout)
        self.assertEqual(args["input_ids"][0, 3:8].tolist(), [13] * 5)
        self.assertEqual(args["input_ids"][0, 10:16].tolist(), [13] * 6)
        with patch.object(probe, "packed_arguments", wraps=probe.packed_arguments) as assemble:
            states = probe.run_packed(self.model, sources, layout=layout)
        self.assertEqual(assemble.call_args.kwargs["pad_token_id"], 13)
        self.assertEqual([state.shape[0] for state in states], [3, 2])
        probe.check_isolation(self.model, sources)

    def test_runtime_accepts_unset_model_pad_and_keeps_head_dimension_check(self):
        _, training = probe.load_probe_config(CONFIG, require_checkpoint=False)
        for head_dim in (128, 64):
            with self.subTest(head_dim=head_dim), contextlib.ExitStack() as stack:
                model = SimpleNamespace(config=SimpleNamespace(head_dim=head_dim, pad_token_id=None))
                tokenizer = SimpleNamespace(pad_token_id=13, model_max_length=8400)
                stack.enter_context(patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model))
                stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer))
                preflight = stack.enter_context(patch.object(probe, "_validate_nvarc_checkpoint"))
                runner = stack.enter_context(patch.object(probe, "NvarcBatchRunner"))
                for method, value in (("is_available", True), ("is_bf16_supported", True), ("set_device", None)):
                    stack.enter_context(patch.object(torch.cuda, method, return_value=value))
                stack.enter_context(patch.object(torch, "manual_seed"))
                if head_dim == 128:
                    result = probe.load_runtime(training)
                    self.assertIs(result[1], model)
                    self.assertIs(result[3], tokenizer)
                    self.assertIsNone(model.config.pad_token_id)
                    self.assertEqual(tokenizer.model_max_length, 9330)
                    runner.assert_called_once_with(model)
                else:
                    with self.assertRaisesRegex(ValueError, "head_dim=128, received 64"):
                        probe.load_runtime(training)
                    runner.assert_not_called()
                preflight.assert_called_once_with(model, tokenizer,
                                                  expected_device=torch.device("cuda:0"),
                                                  expected_dtype=torch.bfloat16)

    def test_real_tiny_qwen_packing_matches_independent_sources(self):
        # CPU SDPA consumes the reset-position block mask. This validates the
        # diagnostic's data layout with real Qwen layers, not CUDA FlashAttention.
        sources = (torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6]), torch.tensor([7, 8, 9]))
        actual = probe.run_packed(self.model, sources)
        for source, state in zip(sources, actual):
            reference = self.runner(source, torch.tensor([0, len(source)])).sequence_hidden_states[0]
            self.assertEqual(state.shape, (len(source), 48))
            self.assertFalse(state.requires_grad)
            torch.testing.assert_close(state, reference, atol=1e-6, rtol=1e-5)

    def test_alignment_restores_real_outputs_and_keeps_sources_isolated(self):
        sources = (torch.tensor([1, 2, 3, 4, 5]), torch.tensor([6, 7, 8]))
        reference = probe.run_packed(self.model, sources)
        for mode in ("total", "source"):
            for alignment in (8, 32):
                with self.subTest(mode=mode, alignment=alignment):
                    layout = probe.packing_layout((5, 3), mode, alignment)
                    actual = probe.run_packed(self.model, sources, layout=layout)
                    alternate = probe.run_packed(self.model, sources, layout=layout, pad_token_id=12)
                    for a, b, c in zip(actual, reference, alternate):
                        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
                        torch.testing.assert_close(a, c, atol=0, rtol=0)
        probe.check_isolation(self.model, sources)

    def test_padding_layout_and_cudnn_offsets_use_physical_elements(self):
        sources = (torch.arange(301), torch.arange(502))
        layout = probe.packing_layout((301, 502), "source", 8)
        args, offsets = probe.packed_arguments(sources, "cpu", layout=layout,
                                               pad_token_id=999, cudnn_heads=(8, 2, 128))
        self.assertEqual(offsets, [0, 304, 808])
        self.assertEqual(args["input_ids"][0, 301:304].tolist(), [999] * 3)
        self.assertEqual(args["position_ids"][0, 304].item(), 0)
        self.assertEqual(args["input_ids"][0, 304:806].tolist(), sources[1].tolist())
        metadata = args["nvarc_cudnn_metadata"]
        self.assertEqual(metadata["q_offsets"].flatten().tolist(), [n * 8 * 128 for n in offsets])
        self.assertEqual(metadata["kv_offsets"].flatten().tolist(), [n * 2 * 128 for n in offsets])
        self.assertEqual(metadata["lengths"].flatten().tolist(), [304, 504])
        self.assertEqual(metadata["lengths"].dtype, torch.int32)
        self.assertEqual(metadata["q_offsets"].dtype, torch.int64)
        total = probe.packing_layout((301, 502), "total", 32)
        self.assertEqual(total["segment_lengths"], (301, 502, 29))
        self.assertEqual(probe.packing_layout((8400,), "source", 32)["total_tokens"], 8416)

    def test_real_sources_replay_keys_trim_targets_and_reject_drift(self):
        class Dataset:
            keys = tuple(f"key{i}" for i in range(32))

            def __getitem__(self, index):
                return SimpleNamespace(input_ids=(index, 2, 3, 4, 5, 6, 7),
                                       prefix_token_ends=(1, 3, 5), target_grids=("a", "b", "c"))

        window = probe.Window("real", 2, (5,) * 16, Dataset.keys[16:])
        sources = probe.real_sources(Dataset(), window, min_context_pairs=2)
        self.assertEqual([s.tolist() for s in sources], [[i, 2, 3, 4, 5] for i in range(16, 32)])
        for changed in (replace(window, source_keys=("wrong",) * 16),
                        replace(window, source_lengths=(7,) * 16)):
            with self.assertRaises(ValueError):
                probe.real_sources(Dataset(), changed, min_context_pairs=2)

    def test_perturbing_one_source_does_not_change_another_source(self):
        sources = (torch.tensor([1, 2, 3]), torch.tensor([7, 8, 9, 0]))
        before = probe.run_packed(self.model, sources)
        after = probe.run_packed(self.model, (torch.tensor([4, 5, 6]), sources[1]))
        self.assertFalse(torch.equal(before[0], after[0]))
        torch.testing.assert_close(before[1], after[1], atol=0, rtol=0)

    def test_grouped_runner_restores_original_order_and_matches_packing(self):
        sources = tuple(torch.tensor(row) for row in ([1, 2], [3, 4, 5, 6], [7, 8, 9], [0]))
        groups = (NvarcSourceBatch((1, 2), 6), NvarcSourceBatch((0, 3), 3))
        actual = probe.run_grouped(self.runner, sources, groups)
        expected = probe.run_packed(self.model, sources)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_comparison_reports_differences_and_rejects_invalid_outputs(self):
        sources = (torch.tensor([1, 2]),)
        states = probe.collect_outputs(lambda: probe.run_packed(self.model, sources), sources, self.model)
        result = probe.numerical_difference(states, states)
        self.assertEqual(result["relative_l2"], 0)
        self.assertFalse(result["agreement_threshold_applied"])
        with self.assertRaisesRegex(ValueError, "invalid frozen"):
            probe.collect_outputs(lambda: (torch.full((2, 48), float("nan")),), sources, self.model)
        with self.assertRaises(ValueError):
            probe.numerical_difference(states, ())

    def test_config_plan_matches_all_sources_and_rotates_variant_order(self):
        policy, training = probe.load_probe_config(CONFIG, require_checkpoint=False)
        plan = probe.build_plan(policy, training)
        self.assertEqual((policy.warmup_calls, policy.measured_calls, policy.max_duration_seconds), (2, 5, 600))
        self.assertEqual([p["optimizer_step"] for p in plan], [35, 37, 31, 34])
        self.assertEqual(max(plan[2]["source_lengths"]), 8400)
        self.assertEqual(plan[3]["real_tokens"], 53367)
        for entry in plan:
            indices = [i for batch in entry["b2_batches"] for i in batch["source_indices"]]
            self.assertEqual(sorted(indices), list(range(16)))
            self.assertEqual(entry["packed_tokens"], sum(entry["source_lengths"]))
            self.assertEqual(entry["packed_calls"], 1)
        self.assertEqual({p["cases"][0]["backend"] for p in plan}, set(probe.BACKENDS))
        self.assertEqual([len(p["cases"]) for p in plan], [16, 13, 13, 16])
        for entry in plan:
            self.assertEqual(len([c for c in entry["cases"] if c["backend"] == "cudnn"]), 1)
            for case in entry["cases"]:
                self.assertEqual(case["layout"]["total_tokens"], sum(case["layout"]["segment_lengths"]))
        median_fa2 = next(c for c in plan[1]["cases"] if c["variant"] == "fa2_none")
        self.assertEqual(median_fa2["aliases"], ["fa2_total_8"])

    def test_invalid_config_inputs_fail_before_model_loading(self):
        raw = json.loads(CONFIG.read_text())
        for field, value in (("warmup_calls", True), ("measured_calls", 0), ("windows", []),
                             ("backends", ["unknown"]), ("backends", ["fa2", "fa2"]),
                             ("alignments", [16]), ("alignments", [32, 8])):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "probe.json"
                path.write_text(json.dumps({**raw, field: value}))
                with self.assertRaises(ValueError):
                    probe.load_probe_config(path, require_checkpoint=False)

    def test_dry_run_writes_nothing_and_loads_no_models(self):
        with patch.object(probe, "load_runtime") as load, patch.object(probe, "run_probe") as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe.main(["--config", str(CONFIG), "--dry-run"]), 0)
            load.assert_not_called()
            run.assert_not_called()

    def test_backend_observation_rejects_fallback_or_partial_flash_execution(self):
        fa2 = {"nvarc_probe::fa2": 36, "flash_attn::_flash_attn_varlen_forward": 36}
        probe.validate_backend_events(fa2, "fa2", layers=36)
        probe.validate_backend_events({"nvarc_probe::fa3": 36, "flash_attn_3::_flash_attn_forward": 36},
                                      "fa3", layers=36)
        for backend in ("fa4", "cudnn"):
            probe.validate_backend_events({f"nvarc_probe::{backend}": 36}, backend, layers=36)
        for counts in ({**fa2, "flash_attn::_flash_attn_varlen_forward": 35},
                       {**fa2, "nvarc_probe::fa2": 35},
                       {**fa2, "aten::_scaled_dot_product_attention_math": 1},
                       {**fa2, "flash_attn_3::_flash_attn_forward": 1}):
            with self.assertRaises(RuntimeError):
                probe.validate_backend_events(counts, "fa2", layers=36)

    def test_worker_completes_sweep_or_preserves_results_and_stops_on_oom(self):
        policy, training = probe.load_probe_config(CONFIG, require_checkpoint=False)
        policy = replace(policy, windows=(probe.Window("tiny", 1, (3, 2), ("a", "b")),), backends=("fa2",))
        cases = len(probe.build_plan(policy, training)[0]["cases"])
        sources = (torch.tensor([1, 2, 3]), torch.tensor([4, 5]))
        for fail in (False, True):
            with self.subTest(oom=fail), tempfile.TemporaryDirectory() as directory:
                with contextlib.ExitStack() as stack:
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                    values = {
                        "load_probe_config": (policy, training),
                        "load_runtime": (torch.device("cpu"), self.model, self.runner, None),
                        "real_sources": sources, "ArcTrainingDataset": Mock(),
                        "schedule_nvarc_sources": (NvarcSourceBatch((0, 1), 4),),
                        "_environment_metadata": {}, "_memory": {}, "version": "test",
                        "select_backend": None, "inspect_backend": {"test_operator": 36},
                        "PackedAttention": SimpleNamespace(name="sdpa", binding="mock", install=Mock(), close=Mock()),
                        "check_isolation": {"mock": "passed"},
                    }
                    mocks = {name: stack.enter_context(patch.object(probe, name, return_value=value))
                             for name, value in values.items()}
                    for name, value in {
                        "Event": None, "empty_cache": None, "get_device_name": "CPU mock",
                        "get_device_properties": SimpleNamespace(total_memory=1024),
                        "get_device_capability": (9, 0),
                    }.items():
                        stack.enter_context(patch.object(torch.cuda, name, return_value=value))
                    observations = []

                    def measure(call, device, events):
                        if fail and len(observations) == policy.measured_calls + 1:
                            raise torch.OutOfMemoryError("simulated allocation failure")
                        output = call()
                        self.assertEqual([len(state) for state in output], [3, 2])
                        row = {"wall_ms": 10, "cuda_span_ms": 8,
                               "peak_allocated_bytes": 100, "peak_reserved_bytes": 200}
                        observations.append(row)
                        return row

                    stack.enter_context(patch.object(probe, "measure_window", side_effect=measure))
                    result = probe.run_worker(SimpleNamespace(config=CONFIG, output_dir=Path(directory)))
                self.assertEqual(json.loads((Path(directory) / "result.json").read_text())["status"],
                                 result["status"])
                self.assertEqual(result["cases"][0]["status"], "complete")
                if fail:
                    self.assertEqual(result["status"], "oom")
                    self.assertIn("simulated allocation failure", result["error"])
                    self.assertEqual(len(result["cases"]), 2)
                    self.assertEqual(mocks["inspect_backend"].call_count, 1)
                else:
                    self.assertEqual(result["status"], "complete")
                    self.assertEqual(len(observations), cases * (policy.measured_calls + 1))
                    self.assertEqual(mocks["inspect_backend"].call_count, cases)
                    self.assertTrue(all(c["status"] == "complete" for c in result["cases"]))
                    self.assertTrue(all(c["hidden_state_difference"]["max_abs"] < 1e-6
                                        for c in result["cases"]))

    def test_worker_timeout_and_progress_streaming(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = Mock()
            code = f"print('{probe.PROGRESS_MARKER} complete', flush=True)"
            rc, expired = probe.launch_worker(
                [sys.executable, "-c", code], log_path=root / "success.log",
                timeout_seconds=5, on_progress=progress,
            )
            self.assertEqual(rc, 0)
            self.assertFalse(expired)
            progress.assert_called_once()
            rc, expired = probe.launch_worker(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                log_path=root / "timeout.log", timeout_seconds=0.1, on_progress=Mock(),
            )
            self.assertNotEqual(rc, 0)
            self.assertTrue(expired)

    def test_supervisor_rejects_incomplete_results_and_preserves_failure(self):
        policy, training = probe.load_probe_config(CONFIG, require_checkpoint=False)
        for status in ("error", "complete", "timeout"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "result"
                partial = {"status": status, "phase": "flash_packed/comparison", "cases": [],
                           "error": "original worker failure"}

                def worker(*args, **kwargs):
                    probe._write_json(destination / "result.json", partial)
                    kwargs["on_progress"]()
                    return (1, status == "timeout")

                progress = Mock()
                with patch.object(probe, "load_probe_config", return_value=(policy, training)), patch.object(
                    probe, "launch_worker", side_effect=worker,
                ):
                    result = probe.run_probe(
                        SimpleNamespace(config=CONFIG, output_dir=destination),
                        metadata={"source_sha256": probe.source_hashes()}, on_progress=progress,
                    )
                self.assertEqual(result["status"], "timeout" if status == "timeout" else "error")
                self.assertTrue((destination / "report.md").exists())
                self.assertEqual(json.loads((destination / "result.json").read_text()), partial)
                self.assertEqual(progress.call_count, 3)


if __name__ == "__main__":
    unittest.main()
