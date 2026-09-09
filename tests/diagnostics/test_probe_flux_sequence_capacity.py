"""Check single-sequence construction, bracket search, failure recovery and saved evidence."""

from pathlib import Path
from types import SimpleNamespace
import copy
import gc
import json
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch

from scripts.diagnostics import probe_flux_sequence_capacity as probe

CONFIG = probe.ROOT / "configs/diagnostics/flux_sequence_capacity.json"


def policy():
    return probe.load_probe_config(CONFIG, require_checkpoints=False)


def attempt_row(length, threshold=12000):
    spec = probe.sequence_spec(length)
    if length > threshold:
        return dict(status="oom", spec=spec, phase="forward_backward", error="allocation failed")
    return dict(status="passed", spec=spec, calls=[{**{k: 1.0 for k in probe.packing.TIMINGS},
        "loss": .5, "peak_allocated_bytes": 1024}], inputs_unchanged=True,
        gradient_health={"finite": True}, modulation_observation=dict(forward_calls=3, backward_calls=3,
        source_dtypes=["torch.bfloat16"] * 3, reduction_dtype="torch.float32"),
        cu_seqlens=[0, length], maximum_segment=length)


def search(threshold=12000, **kwargs):
    return probe.search_lengths(policy().raw, lambda length: attempt_row(length, threshold),
                                kwargs.get("recover", lambda: None), kwargs.get("publish", lambda _: None))


class SequenceCapacityTests(unittest.TestCase):
    def test_inputs_grow_in_one_segment_with_continuous_positions_and_fixed_flow(self):
        template = torch.arange(12).reshape(3, 4).to(torch.bfloat16)
        clean = torch.ones(900, 128)
        short = probe.synthetic_data(template, clean, 1024, 170, max_text_tokens=10000)
        long = probe.synthetic_data(template, clean, 9472, 170, max_text_tokens=10000)
        self.assertEqual(len(long["prefixes"]), 1)
        self.assertEqual(long["layout"].cumulative.tolist(), [0, 9472])
        self.assertEqual(long["layout"].maximum, 9472)
        self.assertEqual(long["layout"].text_lengths, (8572,))
        self.assertEqual(long["layout"].text_positions[-1].tolist(), [0, 0, 0, 8571])
        self.assertEqual(long["layout"].joint_samples.unique().tolist(), [0])
        torch.testing.assert_close(short["prefixes"][0], long["prefixes"][0][:124], rtol=0, atol=0)
        torch.testing.assert_close(long["prefixes"][0][8400], template[0], rtol=0, atol=0)
        for field in ("noisy_latents", "timesteps", "target_velocity"):
            torch.testing.assert_close(getattr(short["flow"], field), getattr(long["flow"], field), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "8400"):
            probe.performance.validate_layout(long["layout"])
        probe.performance.validate_layout(long["layout"], max_text_tokens=10000)

    def test_search_refines_between_adjacent_lengths_and_cleans_before_every_call(self):
        events = []
        def attempt(length):
            self.assertEqual(events[-1], "clean")
            events.append(length)
            return attempt_row(length)
        result = probe.search_lengths(policy().raw, attempt, lambda: events.append("clean"), lambda _: None)
        self.assertEqual(result["largest_passed"], 11776)
        self.assertEqual(result["smallest_oom"], 12032)
        self.assertEqual(result["resolution_tokens"], 256)
        self.assertEqual(result["status"], "bounded")
        self.assertEqual(events[-8:], ["clean", 11776] * 3 + ["clean", 12032])
        self.assertEqual([r["search_phase"] for r in result["attempts"][-4:]],
                         ["confirm_lower"] * 3 + ["confirm_upper"])

    def test_no_upper_failure_is_only_a_lower_bound(self):
        result = search(100000)
        self.assertEqual(result["status"], "lower_bound_only")
        self.assertEqual(result["largest_passed"], 32768)
        self.assertIsNone(result["smallest_oom"])
        self.assertNotIn("upper_confirmation", result)

    def test_start_failure_establishes_a_minimum_control_before_refinement(self):
        result = search(3000)
        self.assertEqual(result["attempts"][1]["search_phase"], "minimum_control")
        self.assertEqual(result["largest_passed"], 2816)
        self.assertEqual(result["smallest_oom"], 3072)
        with self.assertRaisesRegex(RuntimeError, "no lower bound"):
            search(0)

    def test_nonreproducible_upper_failure_is_not_claimed_as_a_maximum(self):
        counts = {}
        def attempt(length):
            counts[length] = counts.get(length, 0) + 1
            return attempt_row(length, 32768 if length == 12032 and counts[length] == 2 else 12000)
        published = []
        with self.assertRaisesRegex(RuntimeError, "did not reproduce"):
            probe.search_lengths(policy().raw, attempt, lambda: None, lambda r: published.append(copy.deepcopy(r)))
        self.assertEqual(published[-1]["status"], "running")
        self.assertEqual(published[-1]["upper_confirmation"]["status"], "passed")

    def test_failed_lower_confirmation_is_not_claimed_as_a_capacity(self):
        counts = {}
        def attempt(length):
            counts[length] = counts.get(length, 0) + 1
            return attempt_row(length, 0 if length == 11776 and counts[length] > 1 else 12000)
        with self.assertRaisesRegex(RuntimeError, "failed fresh-cache confirmation"):
            probe.search_lengths(policy().raw, attempt, lambda: None, lambda _: None)

    def test_oom_discards_graph_and_other_runtime_errors_are_fatal(self):
        references = []
        model = torch.nn.Linear(1, 1)
        models = SimpleNamespace(transformer=model)
        def data(*args, **kwargs):
            x = torch.ones(1, 1, requires_grad=True)
            references.append(weakref.ref(x))
            return {"x": x}
        def fail(*args, **kwargs):
            output = model(args[4]["x"])
            references.append(weakref.ref(output))
            raise torch.OutOfMemoryError("allocation failed")
        with patch.object(probe, "synthetic_data", side_effect=data), \
             patch.object(probe.performance, "input_identity", new=lambda _: {}), \
             patch.object(probe, "allocator_state", return_value={}), \
             patch.object(probe.performance, "measure_call", side_effect=fail) as measured:
            result = probe.attempt_length(models, None, None, torch.zeros(1), None, 1024, policy())
            self.assertEqual(result["status"], "oom")
            # Mock call histories themselves retain inputs; discard those test-only references.
            measured.reset_mock()
            gc.collect()
            self.assertTrue(all(ref() is None for ref in references))
            measured.side_effect = RuntimeError("kernel contract failed")
            with self.assertRaisesRegex(RuntimeError, "kernel contract failed"):
                probe.attempt_length(models, None, None, torch.zeros(1), None, 1024, policy())

    def test_configuration_rejects_backend_budget_and_multi_image_changes(self):
        raw = json.loads(CONFIG.read_text())
        raw["packing_fixture"] = str(CONFIG.parent / raw["packing_fixture"])
        for change in ({"backend": "fa4"}, {"image_tokens": 1800}, {"max_duration_seconds": 331},
                       {"start_joint_tokens": True}, {"quantum": 512}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps({**raw, **change}))
                with self.assertRaises(ValueError):
                    probe.load_probe_config(path, require_checkpoints=False)

    def test_result_replay_rejects_missing_calls_and_multiple_segments(self):
        p = policy()
        result = dict(status="complete", plan=probe.build_plan(p), weights_unchanged=True,
            gradient_storage_unchanged=True, template_unchanged=True,
            backend_preflight={"passed": True}, checkpoints={k: {} for k in ("flux", "nvarc", "vae")},
            control={"backend_profile": {"verified": True}}, search=search())
        probe.validate_complete(result, p)
        bad = copy.deepcopy(result)
        bad["search"]["attempts"][0]["cu_seqlens"] = [0, 4096, 8192]
        with self.assertRaisesRegex(ValueError, "one complete"):
            probe.validate_complete(bad, p)
        bad = copy.deepcopy(result)
        bad["search"]["attempts"].pop(2)
        with self.assertRaises(ValueError):
            probe.validate_complete(bad, p)

    def test_timeout_saves_partial_search_and_reports_no_complete_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            def launch(*args, **kwargs):
                (output / "result.json").write_text(json.dumps(dict(status="running", phase="length/16384/running",
                    search={"status": "running", "attempts": [{**attempt_row(8192), "search_phase": "coarse"}]})))
                return -15, True
            with patch.object(probe, "load_probe_config", return_value=policy()), \
                 patch.object(probe, "source_metadata", return_value={}), \
                 patch.object(probe, "config_hashes", return_value={}), \
                 patch.object(probe, "dataset_hashes", return_value={}), \
                 patch.object(probe.worker_tools, "launch_worker", side_effect=launch):
                result = probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=output))
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(json.loads((output / "summary.json").read_text())["search"]["attempts"][0]["spec"]["real_joint_tokens"], 8192)
            self.assertNotIn("Confirmed fitting length", (output / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
