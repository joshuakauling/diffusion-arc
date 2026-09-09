"""Validate the first-wave plan, independent inputs and bounded supervisor."""

import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from scripts.diagnostics import probe_attention_execution as probe

CONFIG = probe.ROOT / "configs/diagnostics/attention_execution.json"


class AttentionExecutionTests(unittest.TestCase):
    def setUp(self):
        self.policy = probe.load_probe_config(CONFIG, require_checkpoints=False)

    def test_offline_plan_has_frozen_scope_without_execution(self):
        with patch.object(probe, "run_probe", side_effect=AssertionError("execution")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(probe.main(["--plan", "--config", str(CONFIG)]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["kernel_pair_cases"], 12)
        self.assertEqual(plan["production_forward_vjp_evaluations"], 1296)
        self.assertEqual(plan["full_model_singleton_forwards"], 6)
        self.assertEqual(plan["optimizer_updates"], 0)
        self.assertEqual(plan["max_worker_seconds"], 750)
        self.assertEqual(plan["max_function_seconds"], 870)
        self.assertEqual([g["selected_text_lengths"] for g in plan["groups"]],
                         [[468, 406], [2740, 2526], [4745, 4227]])

    def test_configuration_rejects_changed_scope_and_invalid_limits(self):
        mutations = [lambda r: r.update(max_duration_seconds=901),
                     lambda r: r.update(max_duration_seconds=True),
                     lambda r: r.update(upstream_seeds=[167, 167, 169]),
                     lambda r: r.update(upstream_seeds=[167, True, 169]),
                     lambda r: r["groups"].pop(), lambda r: r["backends"].pop(),
                     lambda r: r["capture_blocks"].pop(), lambda r: r.update(extra=1)]
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                raw = copy.deepcopy(self.policy.raw)
                raw["packing_fixture"] = str(self.policy.fixture)
                mutate(raw)
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    probe.load_probe_config(path, require_checkpoints=False)

    def test_singleton_reference_constructs_positions_without_packer(self):
        prefixes = (torch.arange(3 * 7680).reshape(3, 7680).to(torch.bfloat16),
                    torch.arange(5 * 7680).reshape(5, 7680).to(torch.bfloat16))
        clean = torch.randn(2, 900, 128)
        stored = (prefixes, clean, "source")
        with patch.object(probe.packing, "packing_layout", side_effect=AssertionError("packer used")):
            first = probe.singleton_inputs(stored, seed=167, device=torch.device("cpu"))
            second = probe.singleton_inputs(stored, seed=167, device=torch.device("cpu"))
        for index, (a, b) in enumerate(zip(first, second)):
            self.assertEqual(probe.observed.call_identity(a), probe.observed.call_identity(b))
            self.assertTrue(torch.equal(a["text"][0], prefixes[index]))
            self.assertEqual(a["text_positions"][0, :, -1].tolist(), list(range(len(prefixes[index]))))
            self.assertTrue(a["text_mask"].all())
        self.assertFalse(torch.equal(first[0]["images"], first[1]["images"]))

    def test_ingestion_rejects_missing_targets_width_dtype_and_nonfinite_values(self):
        prefixes = (torch.zeros(2, 7680, dtype=torch.bfloat16), torch.zeros(3, 7680, dtype=torch.bfloat16))
        clean = torch.zeros(2, 900, 128)
        for text, images in ((prefixes, clean[:1]), (prefixes, clean.to(torch.bfloat16)),
                (tuple(p[:, :12] for p in prefixes), clean), (tuple(p.float() for p in prefixes), clean),
                (prefixes, clean + float("nan"))):
            with self.subTest(shapes=[p.shape for p in text]), self.assertRaises(ValueError):
                probe.singleton_inputs((text, images, "source"), seed=167, device=torch.device("cpu"))

    def test_capture_runs_original_25_block_singletons_and_preserves_each_completion(self):
        from diffusion_arc.model.flux_transformer import Flux2Transformer2DModel
        from scripts.diagnostics._flux_backend_replay import attention_forward

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        model = Flux2Transformer2DModel(in_channels=8, out_channels=8, num_layers=5, num_single_layers=20,
            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=12,
            axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval().requires_grad_(False)
        inputs = tuple(dict(images=torch.randn(1, 3, 8), text=torch.randn(1, length, 12),
            timesteps=torch.tensor([.3]), target=torch.zeros(1, 3, 8),
            image_positions=torch.zeros(1, 3, 4, dtype=torch.int64),
            text_positions=torch.zeros(1, length, 4, dtype=torch.int64), labels=(label,),
            lengths=(length,), text_mask=torch.ones(1, length, dtype=torch.bool))
            for label, length in (("A", 2), ("B", 5)))
        completed = []
        with patch.object(probe.conditions.replay, "attention_forward",
                side_effect=lambda name, *qkv: attention_forward("bf16_math", *qkv)), \
                patch.object(probe.packing, "packed_velocity", side_effect=AssertionError("packed forward")):
            rows, qkv, captures = probe.capture_pair(model, inputs, operations=True,
                on_result=lambda rows: completed.append([row["label"] for row in rows]))
        self.assertEqual(completed, [["A"], ["A", "B"]])
        self.assertEqual([len(row["attention_calls"]) for row in rows], [25, 25])
        self.assertTrue(all(set(sample) == set(probe.BLOCKS) for sample in qkv))
        self.assertTrue(all(capture.completed for capture in captures))
        self.assertTrue(all(value.dtype == torch.bfloat16 for sample in qkv for tensors in sample.values() for value in tensors))

    def test_frozen_backends_scope_cudnn_to_nvarc_and_math_to_fp32_vae(self):
        def flags():
            return (torch.backends.cuda.cudnn_sdp_enabled(), torch.backends.cuda.math_sdp_enabled())
        before = flags()
        observations = []
        def encode(images):
            observations.append(("vae", flags(), images.dtype))
            return images
        models = SimpleNamespace(tokenizer=object(), nvarc_runner=object(), vae_encoder=encode)
        def prepare(policy, training, actual):
            self.assertIs(actual.nvarc_runner, models.nvarc_runner)
            observations.append(("nvarc", flags(), torch.bfloat16))
            actual.vae_encoder(torch.zeros(1, dtype=torch.float32))
            observations.append(("nvarc_after_vae", flags(), torch.bfloat16))
            return "prepared"
        with patch.object(probe.packing, "prepare_inputs", side_effect=prepare):
            self.assertEqual(probe.prepare_frozen_inputs(object(), object(), models), "prepared")
        self.assertEqual(observations, [("nvarc", (True, False), torch.bfloat16),
            ("vae", (False, True), torch.float32), ("nvarc_after_vae", (True, False), torch.bfloat16)])
        self.assertEqual(flags(), before)

    def test_completeness_requires_named_blocks_and_delegates_full_metric_validation(self):
        from scripts.diagnostics import _flux_packing_replay, _flux_operation_replay

        stages = [*(f"double.{i}" for i in range(5)), *(f"single.{i}" for i in range(20))]
        result = dict(status="complete", plan=probe.build_plan(self.policy),
            checkpoints={name: {} for name in ("flux", "nvarc", "vae")}, groups=[dict(name=name,
                singletons=[dict(label=label, attention_calls=stages, inputs={"images": "hash"}, loss=1., timestep=.3)
                            for label in ("A", "B")], kernels={block: {"case": block} for block in probe.BLOCKS},
                operations={"case": "operations"}) for name in probe.GROUPS])
        with patch.object(_flux_packing_replay, "validate_replay_result", return_value=True) as kernel, \
                patch.object(_flux_operation_replay, "validate_operation_result", return_value=True) as operation:
            self.assertTrue(probe.validate_complete(json.loads(json.dumps(result, sort_keys=True)), self.policy))
            self.assertEqual(kernel.call_count, 12)
            self.assertEqual(operation.call_count, 1)
            bad = copy.deepcopy(result)
            bad["groups"][0]["singletons"][0]["attention_calls"][0] = "single.0"
            with self.assertRaisesRegex(ValueError, "capture"):
                probe.validate_complete(bad, self.policy)
            bad = copy.deepcopy(result)
            bad["groups"][1]["kernels"].pop("single.19")
            with self.assertRaisesRegex(ValueError, "block"):
                probe.validate_complete(bad, self.policy)
            kernel.side_effect = ValueError("missing backward dispatch")
            with self.assertRaisesRegex(ValueError, "backward dispatch"):
                probe.validate_complete(result, self.policy)

    def test_supervisor_preserves_partial_results_and_does_not_retry(self):
        for returncode, timed_out, status in ((-9, True, "timeout"), (1, False, "error")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = SimpleNamespace(config=CONFIG, output_dir=output)

                def launch(*args, **kwargs):
                    probe.worker_tools._write_json(output / "result.json", dict(
                        status="running", phase="kernel/short_b4/double.0", groups=[{"name": "short_b4"}]))
                    self.assertEqual(kwargs["timeout_seconds"], 750)
                    return returncode, timed_out

                with patch.object(probe, "load_probe_config", return_value=self.policy), \
                        patch.object(probe, "source_metadata", return_value={"source_sha256": {}}), \
                        patch.object(probe, "config_hashes", return_value={}), \
                        patch.object(probe, "dataset_hashes", return_value={}), \
                        patch.object(probe.worker_tools, "launch_worker", side_effect=launch) as run:
                    result = probe.run_probe(args)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["groups"], [{"name": "short_b4"}])
                self.assertEqual(run.call_count, 1)
                self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], status)

    def test_source_identity_mismatch_prevents_worker_launch(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(probe, "load_probe_config", return_value=self.policy), \
                patch.object(probe, "source_metadata", return_value={"source_sha256": {"new": "sha"}}), \
                patch.object(probe, "config_hashes", return_value={}), \
                patch.object(probe, "dataset_hashes", return_value={}), \
                patch.object(probe.worker_tools, "launch_worker") as launch:
            with self.assertRaisesRegex(ValueError, "manifest"):
                probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=Path(directory) / "run"),
                                metadata={"source_sha256": {"old": "sha"}})
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
