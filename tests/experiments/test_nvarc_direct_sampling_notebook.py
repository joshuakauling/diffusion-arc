from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = (
    ROOT / "experiments/inference/kaggle/nvarc_direct_sampling"
    / "nvarc_direct_sampling_kaggle.ipynb"
)
REFERENCE = ROOT / "experiments/inference/nvarc_frozen_inference.ipynb"


def source(cell: dict) -> str:
    return "".join(cell["source"])


def python_source(cell: dict) -> str:
    text = source(cell)
    if text.startswith("%%writefile "):
        return text.split("\n", 1)[1]
    return text


class GridTokenizer:
    """Minimal local tokenizer for prompt and grid tests, not model inference."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return "".join("\n" if token == 10 else str(token) for token in tokens)


class NvarcDirectSamplingNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        cls.namespace = {"__name__": "nvarc_direct_sampling_notebook_test"}
        # Only define the embedded module; never load Unsloth or a checkpoint.
        exec(
            compile(python_source(cls.notebook["cells"][4]), str(NOTEBOOK), "exec"),
            cls.namespace,
        )

    def configuration(self, rerun: bool = False) -> dict:
        namespace = {}
        # Execute the real first cell without writing into /kaggle or changing RNG.
        with (
            patch.dict(os.environ, {"KAGGLE_IS_COMPETITION_RERUN": str(int(rerun))}),
            patch("pathlib.Path.mkdir") as mkdir,
            patch("pathlib.Path.write_text") as write_text,
            patch("random.seed"),
        ):
            exec(
                compile(source(self.notebook["cells"][2]), "configuration", "exec"),
                namespace,
            )
        mkdir.assert_called_once()
        write_text.assert_called_once()
        self.assertEqual(json.loads(write_text.call_args.args[0]), namespace["config"])
        return namespace

    def test_only_cell_sources_differ_from_reference(self) -> None:
        notebook_structure = copy.deepcopy(self.notebook)
        reference_structure = copy.deepcopy(self.reference)
        self.assertEqual(len(notebook_structure["cells"]), len(reference_structure["cells"]))
        for cell, reference_cell in zip(
            notebook_structure["cells"], reference_structure["cells"]
        ):
            lines = cell.pop("source")
            reference_lines = reference_cell.pop("source")
            self.assertIsInstance(lines, type(reference_lines))
            self.assertIsInstance(lines, list)
            self.assertTrue(all(isinstance(line, str) for line in lines))
        # Compare every field, not only selected metadata: no added cell IDs,
        # schema updates, output changes, or other notebook normalization.
        self.assertEqual(notebook_structure, reference_structure)

    def test_all_code_cells_compile(self) -> None:
        helper_files = []
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            text = source(cell)
            if text.startswith("%%writefile "):
                helper_files.append(text.splitlines()[0].split()[1])
            with self.subTest(cell=index):
                compile(python_source(cell), f"cell_{index}", "exec")
        self.assertEqual(
            helper_files,
            ["nvarc_direct_sampling.py", "nvarc_direct_sampling_runner.py"],
        )

    def test_preflight_and_gpu_startup_match_working_reference(self) -> None:
        self.assertEqual(
            source(self.notebook["cells"][3]), source(self.reference["cells"][3])
        )
        reference_runner = (
            source(self.reference["cells"][5])
            .replace("nvarc_runner.py", "nvarc_direct_sampling_runner.py")
            .replace("from nvarc_frozen import", "from nvarc_direct_sampling import")
        )
        self.assertEqual(source(self.notebook["cells"][5]), reference_runner)
        self.assertEqual(
            source(self.notebook["cells"][7]),
            source(self.reference["cells"][7]).replace(
                "nvarc_runner.py", "nvarc_direct_sampling_runner.py"
            ),
        )

    def test_first_cells_run_with_simulated_kaggle_inputs(self) -> None:
        namespace = self.configuration()
        versions = {"torch": "test", "transformers": "test", "unsloth": "test"}
        with (
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.write_text") as write_text,
            patch("importlib.metadata.version", side_effect=versions.__getitem__),
        ):
            exec(
                compile(source(self.notebook["cells"][3]), "preflight", "exec"),
                namespace,
            )
        for package, version in versions.items():
            self.assertEqual(namespace["environment"][package], version)
        self.assertEqual(
            json.loads(write_text.call_args.args[0]), namespace["environment"]
        )

    def test_default_is_two_draws_without_extra_inference_methods(self) -> None:
        config = self.configuration()["config"]
        self.assertEqual(config["run_variant"], "S2_A")
        self.assertEqual(config["attempt_seed_labels"], ["A", "B"])
        self.assertEqual(config["evaluation_task_limit"], 4)
        self.assertEqual(
            config["model_path"],
            "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        )
        self.assertEqual(config["protocol"]["attempts_per_output"], 2)
        self.assertEqual(config["protocol"]["unique_draws_per_output"], 2)
        self.assertTrue(config["protocol"]["canonical_input_only"])
        self.assertTrue(config["protocol"]["direct_ancestral_sampling"])
        for name in (
            "test_time_training", "test_time_augmentation", "dfs_or_beam_search",
            "candidate_pool", "rescoring_or_ranking", "voting",
            "invalid_sample_retry", "target_shape_available_to_decoder",
        ):
            with self.subTest(method=name):
                self.assertFalse(config["protocol"][name])

    def test_competition_rerun_uses_test_data_without_labels(self) -> None:
        for rerun in (False, True):
            with self.subTest(rerun=rerun):
                config = self.configuration(rerun=rerun)["config"]
                mode = "submission" if rerun else "evaluation"
                split = "test" if rerun else "evaluation"
                self.assertEqual(config["run_mode"], mode)
                self.assertTrue(
                    config["challenge_path"].endswith(f"arc-agi_{split}_challenges.json")
                )
                if rerun:
                    self.assertIsNone(config["solutions_path"])
                else:
                    self.assertTrue(
                        config["solutions_path"].endswith("arc-agi_evaluation_solutions.json")
                    )

    def solve_output(self, labels: list[str], tokens: list[list[int]]):
        config = self.configuration()["config"]
        config["attempt_seed_labels"] = labels
        task = {
            "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
            "test": [{"input": [[3, 4]]}],
        }
        original_task = copy.deepcopy(task)
        formatter = self.namespace["QwenFormatter"](GridTokenizer())
        sample = Mock(side_effect=[
            {
                "tokens": sequence,
                "sampling_nll": 0.0,
                "stop_reason": "eos" if sequence[-1] == 15 else "max_new_tokens",
                "ended_with_eos": sequence[-1] == 15,
            }
            for sequence in tokens
        ])
        with patch.dict(self.namespace, {"autoregressive_sample": sample}):
            output = self.namespace["solve_output"](
                None, formatter, "task-one", task, 0, config, float("inf")
            )
        self.assertEqual(task, original_task)
        self.assertEqual(output["dropped_demonstrations"], 0)
        self.assertEqual(output["retained_demonstrations"], 1)
        return output, sample

    def test_pass2_samples_same_canonical_prompt_with_distinct_stable_seeds(self) -> None:
        output, sample = self.solve_output(["A", "B"], [[1, 15], [2, 15]])
        self.assertEqual(sample.call_count, 2)
        first, second = [call.kwargs for call in sample.call_args_list]
        prompt = (
            "<|im_start|>user\n12<|im_end|><|im_start|>assistant\n21<|im_end|>"
            "<|im_start|>user\n34<|im_end|><|im_start|>assistant\n"
        )
        self.assertEqual(first["prefix_tokens"], GridTokenizer().encode(prompt))
        self.assertEqual(first["prefix_tokens"], second["prefix_tokens"])
        self.assertNotEqual(first["seed"], second["seed"])
        for label, call in zip(("A", "B"), (first, second)):
            expected_seed = self.namespace["stable_seed"](
                "nvarc-direct-sampling-v1", 42, "task-one", 0, label
            )
            self.assertEqual(call["seed"], expected_seed)
        self.assertEqual(output["attempts"], [[[1]], [[2]]])
        self.assertEqual(output["attempt_valid"], [True, True])
        repeated, _ = self.solve_output(["A", "B"], [[1, 15], [2, 15]])
        self.assertEqual(output["samples"], repeated["samples"])

    def test_pass1_draws_once_and_reuses_draw_a(self) -> None:
        output, sample = self.solve_output(["A", "A"], [[1, 15]])
        self.assertEqual(sample.call_count, 1)
        self.assertEqual(output["attempts"], [[[1]], [[1]]])
        self.assertEqual(list(output["samples"]), ["A"])

    def test_duplicate_pass2_outputs_do_not_trigger_resampling(self) -> None:
        output, sample = self.solve_output(["A", "B"], [[1, 15], [1, 15]])
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(output["attempts"], [[[1]], [[1]]])
        self.assertEqual(list(output["samples"]), ["A", "B"])

    def test_invalid_sample_is_not_retried_or_credited_as_a_placeholder_match(self) -> None:
        output, sample = self.solve_output(["A", "B"], [[1], [2, 15]])
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(output["attempts"], [[[0]], [[2]]])
        self.assertEqual(output["attempt_valid"], [False, True])
        metrics = self.namespace["evaluate_predictions"](
            ["task-one"], {}, {"task-one": [[[0]]]},
            {"task-one": {"outputs": [output]}},
        )
        self.assertEqual(metrics["correct_outputs_at_1"], 0)
        self.assertEqual(metrics["correct_outputs_at_2"], 0)

    def test_submission_has_two_attempts_for_every_test_input(self) -> None:
        output, _ = self.solve_output(["A", "B"], [[1, 15], [2, 15]])
        challenges = {
            "task-one": {"test": [{"input": [[3]]}, {"input": [[4]]}]},
            "task-two": {"test": [{"input": [[5]]}]},
        }
        submission = self.namespace["make_submission"](
            challenges, {"task-one": {"outputs": [output]}}, [[0]]
        )
        self.assertEqual(
            submission["task-one"][0], {"attempt_1": [[1]], "attempt_2": [[2]]}
        )
        for task_id, task in challenges.items():
            self.assertEqual(len(submission[task_id]), len(task["test"]))
            for attempts in submission[task_id]:
                self.assertEqual(set(attempts), {"attempt_1", "attempt_2"})
        self.assertEqual(
            submission["task-one"][1], {"attempt_1": [[0]], "attempt_2": [[0]]}
        )
        self.assertEqual(
            submission["task-two"][0], {"attempt_1": [[0]], "attempt_2": [[0]]}
        )


if __name__ == "__main__":
    unittest.main()
