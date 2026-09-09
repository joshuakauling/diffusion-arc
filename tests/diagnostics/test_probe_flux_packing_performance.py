"""Contracts for comparable B2 timing, numerical intervention and bounded partial evidence."""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from diffusion_arc.model.contracts import FlowMatchingBatch
from scripts.diagnostics import probe_flux_packing_performance as probe

CONFIG = probe.ROOT / "configs/diagnostics/flux_packing_performance.json"


def policy():
    return probe.load_probe_config(CONFIG, require_checkpoints=False)


def timing():
    return {**{k: 1.0 for k in probe.packing.TIMINGS}, "loss": .25,
            "peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048}


def observation(variant):
    active = variant != "sdpa_b2"
    return dict(forward_calls=3 if active else 0, backward_calls=3 if active else 0,
                source_dtypes=["torch.bfloat16"] * 3 if active else [],
                reduction_dtype="torch.float32" if active else None)


def complete_result(p):
    plan = probe.build_plan(p)
    groups = []
    for spec in plan["groups"]:
        cases = []
        for v in plan["variants"]:
            cases.append(dict(variant=v, status="complete", comparison={"passed": True},
                gradient_health={"finite": True}, modulation_observation=observation(v),
                warmup=[timing() for _ in range(p.raw["warmup_calls"])],
                calls=[dict(round=r, call=c, phase="measured", **timing())
                       for r in range(p.raw["rounds"]) for c in range(p.raw["measured_calls"])]))
        groups.append(dict(**spec, status="complete", inputs_unchanged=True, cases=cases))
    return dict(status="complete", plan=plan, groups=groups, weights_unchanged=True,
                gradient_storage_unchanged=True, checkpoints={k: {} for k in ("flux", "nvarc", "vae")},
                backend_profiles={v: {"verified": True} for v in plan["variants"]})


class PerformanceTests(unittest.TestCase):
    def test_plan_keeps_exact_existing_groups_and_separates_timing_from_warmup(self):
        p = policy()
        plan = probe.build_plan(p)
        original, training = probe.packing.load_probe_config(p.config_paths[1], require_checkpoints=False)
        expected = {g["name"]: g for g in probe.packing.build_plan(original, training)}
        for group in plan["groups"]:
            for key in ("target_indices", "padded_length", "text_lengths", "batch_size", "real_joint_tokens"):
                self.assertEqual(group[key], expected[group["name"]][key])
        self.assertEqual(plan["timed_calls"], 144)
        self.assertEqual(plan["warmup_calls"], 24)
        self.assertEqual(plan["backend_profiles"], 4)
        self.assertFalse(plan["capacity_search"])
        self.assertFalse(plan["training_qualified"])

    def test_configuration_rejects_unapproved_groups_precision_and_budget(self):
        raw = json.loads(CONFIG.read_text())
        raw["packing_fixture"] = str(CONFIG.parent / raw["packing_fixture"])
        for key, value in (("groups", ["new_pack"]), ("groups", ["short_b4", "short_b4"]),
                           ("modulation_reduction", "native"), ("max_duration_seconds", 421),
                           ("measured_calls", True), ("backends", ["fa3"])):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps({**raw, key: value}))
                with self.assertRaises(ValueError):
                    probe.load_probe_config(path, require_checkpoints=False)

    def test_measurement_preserves_weights_gradient_storage_and_global_loss_weight(self):
        model = torch.nn.Linear(1, 128, bias=False)
        model.weight.data.fill_(.5)
        model.weight.grad = torch.zeros_like(model.weight)
        model.gradient_checkpointing = False
        model.transformer_blocks, model.single_transformer_blocks = [], []
        pointer = model.weight.grad.data_ptr()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0)
        clean = torch.zeros(2, 900, 128)
        flow = FlowMatchingBatch(clean, torch.tensor([.2, .8]), clean)
        data = dict(clean=clean, flow=flow, prefixes=(torch.ones(2, 12), torch.ones(5, 12)))
        group = dict(padded_length=6, text_lengths=[2, 5], batch_size=2)
        seen = []
        def forward(m, variant, assembled, current, dtype, *, modulation_reduction):
            self.assertEqual(modulation_reduction, "fp32")
            self.assertIs(current["flow"], flow)
            if variant == "sdpa_b2":
                self.assertEqual(assembled.text_attention_mask.sum(1).tolist(), [2, 5])
            else:
                self.assertEqual(tuple(assembled.shape), (7, 12))
                self.assertEqual(current["layout"].cumulative.tolist(), [0, 902, 1807])
                seen.append(current["layout"])
            return m(torch.ones(2, 900, 1))
        event = lambda **kw: SimpleNamespace(record=lambda: None, elapsed_time=lambda other: 1.0)
        with ExitStack() as stack:
            for name in ("synchronize", "reset_peak_memory_stats"):
                stack.enter_context(patch.object(torch.cuda, name))
            stack.enter_context(patch.object(torch.cuda, "Event", side_effect=event))
            stack.enter_context(patch.object(probe.worker_tools, "_memory", return_value={}))
            stack.enter_context(patch.object(probe.packing, "forward_velocity", side_effect=forward))
            step = stack.enter_context(patch.object(optimizer, "step"))
            for variant, backend in (("sdpa_b2", None), ("fa3", object()), ("fa3", object())):
                row, output = probe.measure_call(model, optimizer, variant, group, data, backend,
                                                compute_dtype=torch.bfloat16)
                self.assertAlmostEqual(row["loss"], .25)
                self.assertFalse(output.requires_grad)
                self.assertEqual(model.weight.grad.data_ptr(), pointer)
                torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, (2 / 48) / 128))
            step.assert_not_called()
        self.assertIsNot(seen[0], seen[1])
        self.assertTrue(torch.equal(model.weight, torch.full_like(model.weight, .5)))

    def test_packed_forward_delivers_fp32_option_to_actual_model_entry(self):
        data = dict(flow=SimpleNamespace(noisy_latents=torch.zeros(1), timesteps=torch.zeros(1)),
                    image_positions=object(), layout=object())
        with patch.object(probe.packing, "packed_velocity", return_value=torch.ones(1)) as forward:
            probe.packing.forward_velocity(object(), "fa3", object(), data, torch.bfloat16,
                                          modulation_reduction="fp32")
        self.assertEqual(forward.call_args.kwargs, {"modulation_reduction": "fp32"})

    def test_health_rejects_missing_or_nonfinite_gradients_and_inactive_intervention(self):
        model = torch.nn.Linear(1, 1)
        with self.assertRaises(ValueError):
            probe.gradient_health(model)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        self.assertEqual(probe.gradient_health(model)["tensors"], 2)
        model.bias.grad.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            probe.gradient_health(model)
        with self.assertRaises(ValueError):
            probe.validate_observation(observation("sdpa_b2"), "fa3")

    def test_cudnn_evidence_uses_actual_backend_counts_despite_nested_autocast_entry(self):
        # Observed in the earlier full-model padded-batch backward run.
        counts = {"aten::scaled_dot_product_attention": 50,
                  "aten::_scaled_dot_product_cudnn_attention": 25,
                  "aten::_scaled_dot_product_cudnn_attention_backward": 25}
        probe.validate_backend_evidence(counts, "sdpa_b2", 25)
        for changed in ({**counts, "aten::_scaled_dot_product_cudnn_attention_backward": 0},
                        {**counts, "aten::_scaled_dot_product_attention_math": 25},
                        {**counts, "aten::_scaled_dot_product_cudnn_attention": 50}):
            with self.assertRaises(ValueError):
                probe.validate_backend_evidence(changed, "sdpa_b2", 25)

    def test_completion_rejects_partial_duplicate_changed_or_invalid_samples(self):
        p = policy()
        result = complete_result(p)
        probe.validate_complete(result, p)
        changes = [lambda r: r["groups"][0]["cases"][0]["calls"].pop(),
                   lambda r: r["groups"][0]["cases"][0]["calls"][0].update(forward_ms=float("nan")),
                   lambda r: r["groups"][0]["cases"][0]["calls"][0].update(call=1),
                   lambda r: r["groups"][0].update(target_indices=[0]),
                   lambda r: r["groups"][0].update(inputs_unchanged=False),
                   lambda r: r["backend_profiles"].pop("fa4"),
                   lambda r: r.update(gradient_storage_unchanged=False)]
        for change in changes:
            broken = copy.deepcopy(result)
            change(broken)
            with self.assertRaises(ValueError):
                probe.validate_complete(broken, p)

    def test_supervisor_timeout_retains_finished_groups_and_phase(self):
        p = policy()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result"
            partial = complete_result(p)
            partial.update(status="running", phase="long_b2/fa4/warmup")
            partial["groups"] = partial["groups"][:2]
            def launch(*args, **kwargs):
                self.assertEqual(kwargs["timeout_seconds"], 330)
                probe.worker_tools._write_json(destination / "result.json", partial)
                return -9, True
            with patch.object(probe, "load_probe_config", return_value=p), \
                 patch.object(probe, "source_metadata", return_value={}), \
                 patch.object(probe, "config_hashes", return_value={}), \
                 patch.object(probe, "dataset_hashes", return_value={}), \
                 patch.object(probe.worker_tools, "launch_worker", side_effect=launch):
                result = probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination))
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(result["phase"], partial["phase"])
            saved = json.loads((destination / "summary.json").read_text())
            self.assertEqual(len(saved["rows"]), 8)
            self.assertTrue(all(r["complete"] for r in saved["rows"]))


if __name__ == "__main__":
    unittest.main()
