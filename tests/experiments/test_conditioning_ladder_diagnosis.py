from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest

from PIL import Image
from safetensors.torch import save_file
import torch

from diffusion_arc.arc_data import PAD_CLASS


RUNNER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "experiments"
    / "diagnose_conditioning_ladder_checkpoints.py"
)


def _parent_report() -> dict:
    train_names = (
        "train-id",
        "train-r90",
        "train-r180",
        "train-r270",
        "train-fh",
        "train-fv",
        "train-fd1",
        "train-fd2",
    )
    unseen_names = tuple(name.replace("train-", "unseen-") for name in train_names)
    order = train_names + unseen_names + ("canonical",)
    stage_specs = (
        ("one-condition", ("canonical",), 256),
        ("two-condition-routing", ("canonical", "train-id"), 512),
        ("eight-condition-overfit", train_names, 512),
    )
    return {
        "status": "COMPLETE",
        "run_id": "training-run",
        "source": {"resolved_commit": "a" * 40},
        "conditions": {
            "order": list(order),
            "items": {name: {} for name in order},
        },
        "protocol": {
            "stages": [
                {
                    "name": name,
                    "condition_names": list(condition_names),
                    "optimizer_steps": steps,
                }
                for name, condition_names, steps in stage_specs
            ]
        },
        "stages": {
            name: {
                "status": "COMPLETE",
                "saved_checkpoint": {
                    "state_scope": "transformer_only",
                    "step": steps,
                    "manifest": {"config.json": {"bytes": 1, "sha256": "x"}},
                },
            }
            for name, _condition_names, steps in stage_specs
        },
    }


class ConditioningLadderDiagnosisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.namespace = runpy.run_path(str(RUNNER))

    def test_parent_report_and_nonduplicated_stage_groups(self) -> None:
        report = _parent_report()
        self.namespace["_validate_parent_report"](
            report,
            training_run_id="training-run",
            expected_source_commit="a" * 40,
        )

        stage_groups = self.namespace["_stage_groups"]
        one = stage_groups(report, "one-condition")
        eight = stage_groups(report, "eight-condition-overfit")

        self.assertEqual(one["seen"], ("canonical",))
        self.assertNotIn("original", one)
        self.assertEqual(len(one["unseen"]), 8)
        self.assertEqual(eight["original"], ("canonical",))
        flattened = tuple(name for names in eight.values() for name in names)
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_parent_report_rejects_source_or_checkpoint_step_drift(self) -> None:
        validate = self.namespace["_validate_parent_report"]
        with self.assertRaisesRegex(RuntimeError, "source commit"):
            validate(
                _parent_report(),
                training_run_id="training-run",
                expected_source_commit="b" * 40,
            )

        report = copy.deepcopy(_parent_report())
        report["stages"]["one-condition"]["saved_checkpoint"]["step"] = 255
        with self.assertRaisesRegex(RuntimeError, "Checkpoint step"):
            validate(
                report,
                training_run_id="training-run",
                expected_source_commit="a" * 40,
            )

    def test_user_facing_names_use_original_seen_and_unseen(self) -> None:
        display_name = self.namespace["_display_name"]
        self.assertEqual(display_name("canonical"), "Original")
        self.assertEqual(display_name("train-r90"), "Gesehen · r90")
        self.assertEqual(display_name("unseen-fh"), "Ungesehen · fh")

    def test_partial_schedules_preserve_maximum_full_run_step_size(self) -> None:
        schedule_function = self.namespace["_partial_timestep_schedule"]
        schedule_function.__globals__["DEVICE"] = torch.device("cpu")

        expected_intervals = {0.5: 25, 0.9: 45, 0.99: 50, 1.0: 50}
        for start, interval_count in expected_intervals.items():
            with self.subTest(start=start):
                schedule = schedule_function(start)
                self.assertEqual(len(schedule) - 1, interval_count)
                self.assertAlmostEqual(float(schedule[0].item()), start)
                self.assertEqual(float(schedule[-1].item()), 0.0)
                self.assertLessEqual(
                    float((schedule[:-1] - schedule[1:]).max().item()),
                    0.0200001,
                )

    def test_shared_noise_is_paired_across_batch_sizes_and_rows(self) -> None:
        shared_noise = self.namespace["_shared_standard_noise"]
        first = shared_noise(
            batch_size=1,
            latent_shape=(5, 3),
            seed=167,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        second = shared_noise(
            batch_size=8,
            latent_shape=(5, 3),
            seed=167,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(
            second,
            second[:1].expand_as(second),
        )

    def test_failure_report_survives_nonfinite_measurement(self) -> None:
        record_failure = self.namespace["_record_failure"]
        function_globals = record_failure.__globals__
        self.addCleanup(function_globals.pop, "REPORT", None)
        self.addCleanup(function_globals.pop, "REPORT_PATH", None)
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "report.json"
            function_globals["REPORT_PATH"] = report_path
            function_globals["REPORT"] = {
                "status": "RUNNING",
                "measurement": float("nan"),
                "failures": [],
            }

            record_failure("probe", FloatingPointError("non-finite velocity"))

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["failed_stage"], "probe")
            self.assertEqual(report["measurement"], "NaN")
            self.assertEqual(report["failures"][0]["type"], "FloatingPointError")

    def test_canvas_metrics_separate_grid_and_padding(self) -> None:
        metrics = self.namespace["_canvas_metrics"]
        expected = torch.full((30, 30), PAD_CLASS, dtype=torch.int64)
        expected[:2, :2] = torch.tensor(((1, 2), (3, 4)))
        predicted = expected.clone()
        predicted[0, 1] = 9
        predicted[20, 20] = 0

        result = metrics(predicted, expected, ((1, 2), (3, 4)))

        self.assertEqual(result["grid_cells_correct"], 3)
        self.assertEqual(result["grid_cells_total"], 4)
        self.assertEqual(result["padding_cells_correct"], 895)
        self.assertEqual(result["padding_cells_total"], 896)
        self.assertFalse(result["exact_class_canvas"])

    def test_checkpoint_validation_checks_safetensors_and_hashes(self) -> None:
        validate = self.namespace["_validate_checkpoint_directory"]
        sha256_file = self.namespace["sha256_file"]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text("{}", encoding="utf-8")
            shard = directory / "diffusion_pytorch_model-00001-of-00001.safetensors"
            save_file({"weight": torch.arange(4)}, shard)
            index = {
                "metadata": {},
                "weight_map": {"weight": shard.name},
            }
            (directory / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps(index),
                encoding="utf-8",
            )
            manifest = {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in directory.iterdir()
            }

            self.assertEqual(validate(directory, manifest), manifest)
            corrupt = copy.deepcopy(manifest)
            corrupt["config.json"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                validate(directory, corrupt)

    def test_comparison_sheet_renders_unicode_labels_and_both_regions(self) -> None:
        save_sheet = self.namespace["_save_comparison_sheet"]
        expected = torch.full((1, 30, 30), PAD_CLASS, dtype=torch.int64)
        expected[0, :2, :2] = torch.tensor(((1, 2), (3, 4)))
        predicted = expected.clone()
        predicted[0, 0, 1] = 9
        palette = self.namespace["TARGET_PALETTE_RGB"]
        rgb = palette[predicted].repeat_interleave(16, dim=1).repeat_interleave(
            16, dim=2
        ).permute(0, 3, 1, 2)
        regions = self.namespace["_canvas_metrics"](
            predicted[0], expected[0], ((1, 2), (3, 4))
        )
        records = [
            {
                "exact_grid": {"exact_grid": False},
                "regions": regions,
            }
        ]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sheet.png"
            save_sheet(
                path=path,
                title="Checkpoint · t=1.00",
                row_labels=("Gesehen · r90",),
                expected_grids=(((1, 2), (3, 4)),),
                expected_canvases=expected,
                vae_class_canvases=expected,
                predicted_rgb_images=rgb,
                predicted_class_canvases=predicted,
                records=records,
                canvas_view=False,
            )

            with Image.open(path) as rendered:
                self.assertEqual(rendered.mode, "RGB")
                self.assertGreater(rendered.width, 1000)
                self.assertGreater(rendered.height, 250)


if __name__ == "__main__":
    unittest.main()
