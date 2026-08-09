# Initial ARC-AGI-2 Debug Dataset

## Status and purpose

This document defines the first fixed data subsets for interface checks and
Milestone-2 tiny-set overfitting. They are deliberately small debugging sets,
not representative training or validation splits.

The authoritative project scope and milestone order remain in `README.md`.

## Source identity and leakage boundary

The samples come only from the public ARC-AGI-2 training split stored in the
separate Kaggle-format files:

- `arc-agi_training_challenges.json`
  - SHA-256:
    `779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5`
- `arc-agi_training_solutions.json`
  - SHA-256:
    `9f07a38bd25af5e83aa5bf85c5cb1a1fefdb30f6a755256fa65429e697ca97f9`

The selected challenge records contain demonstrations and test inputs but no
test outputs. Targets are resolved separately from the solution file. The
official evaluation and test splits are not used.

The machine-readable source of truth is
`configs/datasets/arc_agi_2_initial.json`.

## Ordered samples and nested subsets

| Order | Task ID | Label | Test input | Target |
|---:|---|---|---:|---:|
| 1 | `3c9b0459` | 180-degree rotation | 3x3 | 3x3 |
| 2 | `0d3d703e` | fixed color mapping | 3x3 | 3x3 |
| 3 | `8be77c9e` | vertical mirror | 3x3 | 6x3 |
| 4 | `c9e6f938` | horizontal mirror | 3x3 | 3x6 |
| 5 | `00576224` | periodic pattern expansion | 2x2 | 6x6 |
| 6 | `0520fde7` | panel intersection | 3x7 | 3x3 |
| 7 | `88a62173` | unique quadrant selection | 5x5 | 2x2 |
| 8 | `d631b094` | count nonzero cells | 3x3 | 1x5 |

The subsets are strict prefixes of this order:

- `smoke_1`: sample 1;
- `tiny_4`: samples 1 through 4;
- `tiny_8`: samples 1 through 8.

Every selected task has one test example, two to four demonstrations, a
serialized prompt no longer than 384 characters, and no grid dimension above
seven cells.

## Sample contract

One model sample is identified by `(task_id, test_index)`. The dataset resolver
returns:

- the serialized demonstrations and selected test input as the NVARC prompt;
- the selected target grid from the separate solution source;
- task identity, shapes, prompt character count, and other audit metadata.

It does not tokenize prompts, render canvases, encode VAE latents, batch
examples, or augment grids. Those operations belong to later training work.

## Validation

With the benchmark files available at their repository-local paths, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_initial_dataset.py
```

For attached files in another environment, pass explicit paths:

```bash
PYTHONPATH=src python scripts/validate_initial_dataset.py \
  --challenges /path/to/arc-agi_training_challenges.json \
  --solutions /path/to/arc-agi_training_solutions.json
```

The validator verifies the pinned source hashes, manifest schema, task and
solution presence, test indices, grid schema, leakage boundary, nesting, and
selection constraints before printing a JSON audit report.

## Limitations

- Prompt sizes are currently measured in characters. Exact NVARC token counts
  require the pinned NVARC tokenizer and must be verified in the model runtime.
- Small ARC grids shorten the conditioning context and simplify debugging, but
  every target is still padded to the fixed 30x30 canvas and therefore uses 900
  FLUX image tokens.
- Results on these curated tasks must not be reported as held-out
  generalization or benchmark performance.
