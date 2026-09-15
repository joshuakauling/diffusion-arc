"""Keep prediction code independent of supervision and CPU config loading light."""

import ast
from importlib.util import resolve_name
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ModuleBoundaryTests(unittest.TestCase):
    def test_model_and_inference_do_not_depend_on_training(self):
        for package in ("model", "inference"):
            for path in (ROOT / "src/diffusion_arc" / package).rglob("*.py"):
                with self.subTest(path=path):
                    parent = ".".join(path.relative_to(ROOT / "src").parent.parts)
                    for node in ast.walk(ast.parse(path.read_text())):
                        if isinstance(node, ast.Import):
                            imports = [alias.name for alias in node.names]
                        elif isinstance(node, ast.ImportFrom):
                            module = resolve_name("." * node.level + (node.module or ""), parent)
                            imports = [module, *(module + "." + alias.name for alias in node.names)]
                        else:
                            continue
                        self.assertFalse(any(name == "diffusion_arc.training" or
                                             name.startswith("diffusion_arc.training.")
                                             for name in imports), imports)

    def test_config_loading_needs_no_tensor_or_model_runtime(self):
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys; sys.path.insert(0, 'src'); "
                "from diffusion_arc.training.config import load_training_config; "
                "import diffusion_arc.training.step, diffusion_arc.training.run; "
                "config = load_training_config('configs/training/arc_agi_1_epoch.json'); "
                "assert config.loop.optimizer_batch_size == 48; "
                "assert config.recipe_dict()['scheduler']['warmup_steps'] == 50; "
                "assert not {'torch', 'transformers', 'diffusers'}.intersection(sys.modules)"
            )], cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
