"""Validate whole-example growth, precise reduction switching and OOM recovery evidence."""

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

from scripts.diagnostics import probe_flux_packed_capacity as probe

CONFIG = probe.ROOT / "configs/diagnostics/flux_packed_capacity.json"


def policy():
    return probe.load_probe_config(CONFIG, require_checkpoints=False)


def timing():
    return {**{k: 1.0 for k in probe.packing.TIMINGS}, "loss": .5,
            "peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048}


def observation(reduction="fp32"):
    active = reduction == "fp32"
    return dict(forward_calls=3 if active else 0, backward_calls=3 if active else 0,
        source_dtypes=["torch.bfloat16"] * 3 if active else [], reduction_dtype="torch.float32" if active else None)


def passed(spec, count=1):
    return dict(status="passed", spec=spec, calls=[timing() for _ in range(count)],
                inputs_unchanged=True, gradient_health={"finite": True}, modulation_observation=observation())


def complete_result(p):
    plan = probe.build_plan(p)
    comparison = dict(status="complete", spec=plan["comparison"], inputs_unchanged=True,
                      backend_profile={"verified": True}, cases=[])
    for reduction in ("native", "fp32"):
        comparison["cases"].append(dict(reduction=reduction, forward_identical=True,
            gradient_health={"finite": True}, modulation_observation=observation(reduction),
            warmup=[timing(), timing()], calls=[dict(round=r, call=c, **timing()) for r in range(4) for c in range(5)]))
    families = []
    for f in plan["families"]:
        first, second = f["candidates"][:2]
        oom = dict(status="oom", spec=second, phase="forward_backward", error="allocation failed")
        families.append(dict(name=f["name"], status="bounded", attempts=[passed(first), oom],
            first_oom=oom, largest_passed=first, confirmation=passed(first, 3)))
    return dict(status="complete", plan=plan, comparison=comparison, families=families,
                weights_unchanged=True, gradient_storage_unchanged=True,
                backend_preflight={"passed": True}, checkpoints={k: {} for k in ("flux", "nvarc", "vae")})


class CapacityTests(unittest.TestCase):
    def test_plan_preserves_comparison_and_adds_only_complete_examples(self):
        p = policy()
        plan = probe.build_plan(p)
        self.assertEqual(plan["comparison"]["text_lengths"], [2077, 2017, 2009, 1956])
        self.assertEqual(plan["comparison"]["real_joint_tokens"], 11659)
        self.assertEqual(plan["timed_comparison_calls"], 40)
        self.assertEqual(plan["warmup_comparison_calls"], 4)
        self.assertEqual(plan["round_orders"], [["native", "fp32"], ["fp32", "native"]] * 2)
        for family in plan["families"]:
            for first, second in zip(family["candidates"], family["candidates"][1:]):
                self.assertEqual(first["target_indices"], second["target_indices"][:-1])
                self.assertEqual(first["text_lengths"], second["text_lengths"][:-1])
                self.assertEqual(second["real_joint_tokens"] - first["real_joint_tokens"], second["text_lengths"][-1] + 900)
        self.assertEqual(plan["families"][1]["candidates"][0]["text_lengths"], [8400])

    def test_config_rejects_changed_control_duplicate_targets_and_over_budget(self):
        raw = json.loads(CONFIG.read_text())
        raw["packing_fixture"] = str(CONFIG.parent / raw["packing_fixture"])
        changes = [lambda r: r.update(max_duration_seconds=481), lambda r: r.update(backend="fa4"),
                   lambda r: r["families"][0]["target_indices"].__setitem__(0, 0),
                   lambda r: r["families"][1]["target_indices"].__setitem__(1, 14),
                   lambda r: r["families"][1].update(start_count=2)]
        for change in changes:
            candidate = copy.deepcopy(raw)
            change(candidate)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(candidate))
                with self.assertRaises(ValueError):
                    probe.load_probe_config(path, require_checkpoints=False)

    def test_call_switches_only_requested_reduction(self):
        args = [object() for _ in range(5)]
        with patch.object(probe.performance, "measure_call", return_value=(timing(), torch.zeros(1))) as measured:
            for reduction in ("native", "fp32"):
                probe.call(*args, reduction)
        a, b = measured.call_args_list
        self.assertEqual(a.args, b.args)
        self.assertEqual(a.kwargs, dict(compute_dtype=torch.bfloat16, modulation_reduction="native"))
        self.assertEqual(b.kwargs, dict(compute_dtype=torch.bfloat16, modulation_reduction="fp32"))

    def test_search_stops_at_first_oom_then_confirms_last_fit(self):
        family = probe.build_plan(policy())["families"][0]
        events = []
        second = family["candidates"][1]
        def attempt(spec, repeats):
            events.append((spec["name"], repeats))
            return dict(status="oom", spec=spec, phase="forward_backward") if spec == second else passed(spec, repeats)
        def recover():
            events.append("recover")
        result = probe.search_family(family, attempt, recover, confirmation_calls=3, publish=lambda _: None)
        first = family["candidates"][0]
        self.assertEqual(events, [(first["name"], 1), (second["name"], 1), "recover", "recover", (first["name"], 3)])
        self.assertEqual(result["status"], "bounded")
        self.assertEqual(result["largest_passed"], first)
        self.assertEqual(result["confirmation"]["calls"], [timing()] * 3)

    def test_search_distinguishes_lower_bound_from_maximum_and_rejects_failed_recovery(self):
        family = probe.build_plan(policy())["families"][0]
        result = probe.search_family(family, lambda spec, count: passed(spec, count), lambda: None,
                                     confirmation_calls=3, publish=lambda _: None)
        self.assertEqual(result["status"], "lower_bound_only")
        self.assertNotIn("first_oom", result)
        def fail_confirmation(spec, count):
            return dict(status="oom", spec=spec) if count == 3 else passed(spec)
        with self.assertRaisesRegex(RuntimeError, "failed confirmation"):
            probe.search_family(family, fail_confirmation, lambda: None,
                                 confirmation_calls=3, publish=lambda _: None)
        with self.assertRaisesRegex(RuntimeError, "initial capacity control"):
            probe.search_family(family, lambda spec, count: {"status": "oom", "spec": spec}, lambda: None,
                                 confirmation_calls=3, publish=lambda _: None)

    def test_allocation_oom_drops_failed_graph_and_other_errors_are_not_capacity_bounds(self):
        model = torch.nn.Linear(1, 1)
        models = SimpleNamespace(transformer=model)
        references = []
        def data(*args):
            x = torch.ones(1, 1, requires_grad=True)
            references.append(weakref.ref(x))
            return {"x": x}
        def failed_call(model, optimizer, backend, spec, data, reduction):
            output = model(data["x"])
            references.append(weakref.ref(output))
            raise torch.OutOfMemoryError("deliberate allocation failure")
        with patch.object(probe, "candidate_data", side_effect=data), \
             patch.object(probe.performance, "input_identity", return_value={}), \
             patch.object(probe, "call", side_effect=failed_call):
            result = probe.attempt_candidate(models, object(), object(), object(), {}, 1, 1)
        gc.collect()
        self.assertEqual(result["status"], "oom")
        self.assertTrue(all(ref() is None for ref in references))
        with patch.object(probe, "candidate_data", side_effect=RuntimeError("kernel bug")):
            with self.assertRaisesRegex(RuntimeError, "kernel bug"):
                probe.attempt_candidate(models, object(), object(), object(), {}, 1, 1)
        with patch.object(probe, "candidate_data", side_effect=torch.OutOfMemoryError("setup")):
            with self.assertRaisesRegex(RuntimeError, "outside measured"):
                probe.attempt_candidate(models, object(), object(), object(), {}, 1, 1)

    def test_completion_requires_confirmed_bound_and_all_comparison_samples(self):
        p = policy()
        original = complete_result(p)
        probe.validate_complete(original, p)
        changes = [lambda r: r["comparison"]["cases"][0]["calls"].pop(),
                   lambda r: r["comparison"]["cases"][0]["calls"][0].update(round=1),
                   lambda r: r["comparison"]["cases"][0].update(forward_identical=False),
                   lambda r: r["families"][0]["confirmation"]["calls"].pop(),
                   lambda r: r["families"][0].update(status="lower_bound_only"),
                   lambda r: r["families"][0].update(largest_passed=r["families"][0]["first_oom"]["spec"]),
                   lambda r: r.update(gradient_storage_unchanged=False)]
        for change in changes:
            altered = copy.deepcopy(original)
            change(altered)
            with self.assertRaises(ValueError):
                probe.validate_complete(altered, p)

    def test_timeout_retains_completed_comparison_and_capacity_progress(self):
        p = policy()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result"
            partial = complete_result(p)
            partial.update(status="running", phase="capacity/long_source_3/1_calls")
            partial["families"] = partial["families"][:1]
            def launch(*args, **kwargs):
                self.assertEqual(kwargs["timeout_seconds"], 480)
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
            self.assertEqual([r["measured_calls"] for r in saved["rows"]], [20, 20])
            self.assertEqual(len(saved["families"]), 1)


if __name__ == "__main__":
    unittest.main()
