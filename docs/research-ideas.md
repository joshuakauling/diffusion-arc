# Research Ideas

Compact notes for possible future experiments. Their research benefit is
untested; implementation support is noted separately. The root
[README](../README.md) remains the source of truth for project scope. Reference
images and output heads are possible extensions beyond the initial scope.

## 2026-09-07 — Later NVARC hidden states (idea 1)

Try conditioning FLUX on NVARC hidden states `18/27/36` instead of `9/18/27`.
Later layers may contain more useful information for solving ARC tasks;
the original FLUX layer selection may not be optimal for ARC-specialized NVARC.
Using layer 36 would require running all NVARC blocks again, giving up B3's
early exit after block 27. Compare any quality gain with that additional cost.

Layer selection is now configuration-driven: set
`execution.nvarc_hidden_layers` to `[18, 27, 36]` and
`execution.nvarc_early_exit_layer` to `36`. Three states retain the existing
7,680-wide conditioning interface. Use raw block outputs, including layer 36,
without final normalization or the language-model head. Configuration support
does not establish a learning improvement; see the
[NVARC runner contract](training/nvarc-batch-runner.md).

## VAE-encoded reference grids (idea 2)

Supply the query input grid as a clean, frozen-VAE-encoded reference image in
addition to NVARC text conditioning. The hypothesis is that an explicit spatial
representation helps FLUX copy, move, and compare cells and objects.

Start with the query input only. Give reference and generated-image tokens
distinct image identities while retaining their row/column coordinates. Apply
flow noise and the velocity loss only to the generated output tokens. Query
output colors and dimensions must never enter the reference conditioning.
Demonstration input/output grids are a later extension using only examples
already visible in the prompt.

Measure quality, token count, memory, and complete-step time. This requires
changes to image-token packing, positions, and output selection; existing
assumptions about exactly 900 image tokens per target need review. Combine with
idea 3 only after measuring each separately, since both expose grid geometry.

## Cell coordinates for text conditioning (idea 3)

FLUX currently receives text positions `(0, 0, 0, s)` and output-image positions
`(0, row, column, 0)`, where `s` is the position within a conditioning sequence.
Test `(0, row, column, s)` for text tokens representing ARC cells. Reset
row/column coordinates for each serialized grid; leave structural tokens at
their existing coordinates and preserve the sequence coordinate everywhere.
NVARC's own causal positions and hidden-state computation remain unchanged.

The hypothesis is that shared spatial axes make grid structure easier to use.
Derive coordinates from the visible serialized prompt, retain them alongside
the hidden states, and slice both at the same target-prefix boundary. Use the
same convention in training, validation, and inference. Verify correspondence
after augmentation, row separators, and prefix slicing before a training run.

A later variant could add a grid identity on the first axis. A numerical RoPE
coordinate is not a categorical input/output-role embedding, so these should
be separate experiments. Sharing coordinates also does not imply that input
and output objects necessarily occupy the same cells. This ablation needs code
changes but retains the current model widths and token count.

## Direct color readout and auxiliary color loss (idea 5)

Keep the frozen VAE encoder and learn a small decoder from packed latents to
the ten ARC colors plus PAD. The clean flow target remains the existing
`z_0 = VAE_encode(render(output_grid))`, including packing and normalization,
with shape `[900, 128]`. These are contextual vectors at cell-aligned spatial
positions, not a fixed embedding for each color; see
[target-image encoding](training/flux-image-encoding.md).

First train only a shared linear `128 -> 11` classifier on clean training
latents and integer cell labels. Try a small MLP or local spatial decoder only
if the linear readout is insufficient. Evaluate cell accuracy, per-class/PAD
errors, inferred shape, and exact-grid reconstruction on held-out validation
tasks, compared with the existing VAE decoder. Success on clean latents is a
readout feasibility result; also test predicted latents before claiming a
generation improvement.

Then test a color loss that reaches FLUX. Under the current flow convention,
`z_t = (1 - t) * z_0 + t * noise`, form the clean estimate
`z_0_hat = z_t - t * v_theta(z_t, t)` and apply the head to that estimate:

```text
logits = color_head(z_0_hat)                  # [900, 11]
loss = flow_loss + lambda_color * CE(logits, output_cell_ids)
```

Preserve gradients through the clean estimate into FLUX; training the head
only on true clean latents cannot improve FLUX itself. Balance grid and PAD
contributions so blank canvas does not dominate. The VAE encoder stays frozen.
At inference, classify the final sampled latent and infer dimensions from PAD,
without ground-truth shape information.

This could replace RGB decoding and nearest-palette classification, and may
reduce dependence on palette-specific decoding behavior. It does not remove
the palette or VAE encoder used to construct flow targets. Keep the existing
decoder as a comparison and distinguish readout gains from changes in learned
generation.

### Separate extension: explicit color embeddings without a VAE

To remove the VAE encoder as well, define an explicit codebook, for example
eleven fixed, equally scaled orthogonal vectors in 128 dimensions, and set
`z_0[row, column] = E[cell_id]`. This supplies a deterministic flow target
without image rendering. A learned codebook is a further experiment requiring
care with changing target scale and collapse.

Although the 128-wide interface can stay intact, these targets have a different
distribution from pretrained FLUX latents. This is a larger representation
change than adding a color head. It could remove VAE and palette machinery
entirely, but compatibility with FLUX pretraining must be measured separately.

## First comparison: two separate 512-step ablations

Prioritize ideas 1 and 3, each in isolation against the existing 512-step
baseline. Start each from the same initial FLUX weights, not from the baseline's
step-512 checkpoint. Reuse the same first 512 optimizer batches, including task
order, augmentation, target selection, global batch size, seeds, flow draws,
optimizer, and learning-rate schedule. Stop at 512 updates without shortening
the schedule's original horizon. No full epoch or new training subset is needed.

At a global batch of 48 targets this is 24,576 target exposures per arm, not
24,576 distinct tasks. Reuse the recorded baseline only where its recipe and
data order match; missing diagnostics can be evaluated on retained checkpoints.
Record source/configuration identity and compare matching checkpoints and
validation noise draws.

Use held-out validation grid velocity MSE at `t=0.9` and `t=1.0` as the primary
early signals. Inspect the existing six-timestep curve and boundary/canvas
losses for regressions, alongside training loss and complete-step time/memory.
An optional swapped-conditioning probe can test whether the model uses the
prompt, but must be measured for the baseline too. Early loss improvements are
screening evidence, not proof of better exact-grid generation; reserve sampling
comparisons for promising candidates and confirm small differences across seeds.

Idea 2 needs more implementation work. The clean-latent readout test for idea 5
can be evaluated independently before adding an auxiliary loss to FLUX. Keep
held-out tasks and all their augmentations out of training, fit heads only on
training data, and do not use final-test tasks for any of these decisions.

## Distributed load balancing

Assign sequences to GPUs using their expected number of FLUX calls to keep
work balanced. Compare predicted and observed per-rank step time while
preserving the logical global batch and per-target random draws.
