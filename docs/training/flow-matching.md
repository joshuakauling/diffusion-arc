# Flow-Matching Training State

For the current end-to-end training path and documentation reading order, see
the [training system overview](README.md).

## Status and scope

`sample_flow_matching_batch` is the implemented stochastic boundary between
clean normalized target latents and one straight-path FLUX training state. It
draws independent Gaussian noise and a uniform normalized flow time for every
target exposure, constructs the noisy transformer input, and returns the
corresponding velocity target.

`flow_matching_loss` computes the initial unweighted velocity objective in
FP32. Neither function executes the FLUX transformer, chooses an optimizer, or
performs an update.

The training-side `compute_flux_microbatch_loss` operation composes these
helpers with one prepared microbatch: it executes FLUX with the supplied
flow draws and returns the unscaled mean loss with its optional regional metrics.
Training uses independent packed attention segments throughout.
Flow draws are assigned in FIFO order before physical grouping, preserving
each target's timestep and noise.

The samplers and immutable `FlowMatchingBatch` contract live in
[`training/step/flow_sampling.py`](../../src/diffusion_arc/training/step/flow_sampling.py).
The plain and regional objectives live in
[`training/step/loss.py`](../../src/diffusion_arc/training/step/loss.py).
The complete microbatch composition lives in
[`training/step/flux_microbatch.py`](../../src/diffusion_arc/training/step/flux_microbatch.py).
This page describes the plain baseline; the
[first epoch recipe](arc-agi-1-recipe.md) adds regional weighting and a fixed
quota of exact `t=1` targets through the same training loop.

## Input and per-target sampling

The sampler consumes the clean normalized output of `FluxVaeEncoder`:

```text
z_0: [K, 900, 128], floating point
```

For every one of the `K` target exposures it independently draws:

```text
epsilon ~ Normal(0, I),  shape [900, 128]
t       ~ Uniform(0, 1), one scalar
```

Noise inherits the clean latent device and dtype. Timesteps are sampled as
FP32 on the same device, then converted to the latent dtype only for the
interpolation arithmetic. A caller may supply a device-compatible
`torch.Generator` so the training runner can control and record random state.

The released FLUX.2 inference code does not disclose the original training
timestep density or any timestep-dependent loss weighting. Uniform sampling
is therefore the explicit Milestone 1 baseline rather than a claim about the
checkpoint's pretraining recipe. Alternative densities remain a later
configuration-driven training-recipe experiment.

## Straight-path state and supervision

For each target exposure the sampler constructs:

```text
z_t = (1 - t) * z_0 + t * epsilon
u_t = epsilon - z_0
```

Under this convention, `t=0` is clean data and `t=1` is Gaussian noise. The
same scalar `t` is broadcast across all 900 tokens and 128 channels of one
target, while different targets receive independent times and noise tensors.

The returned value is:

```python
FlowMatchingBatch(
    noisy_latents=...,      # z_t, [K, 900, 128], latent dtype
    timesteps=...,          # t,   [K], torch.float32
    target_velocity=...,    # u_t, [K, 900, 128], latent dtype
)
```

The normalized `[0,1]` timesteps are passed to the Diffusers FLUX.2
transformer without pre-scaling. Its forward method performs the internal
factor-of-1000 conversion used by the timestep embedding.

The sampled noise is not retained as a separate public field because training
needs only `z_t`, `t`, and `u_t` after construction. `FluxMicrobatch` contains
text conditioning, the attention layout and image position IDs. It does not
copy clean target latents; image tensors come from the supplied `FlowMatchingBatch`.

## Velocity loss

Given the transformer prediction `v_theta`, the initial objective is:

```text
mean((v_theta - u_t) ** 2)
```

`flow_matching_loss` casts both tensors to FP32 before the subtraction and
mean reduction. Since every target has the fixed shape `[900,128]`, the global
element mean gives every target exposure equal weight. The cast does not break
autograd; gradients flow through the prediction into the complete FLUX
transformer. The frozen VAE and NVARC outputs do not require gradients.

There is no timestep weighting, auxiliary target, color loss, shape loss, or
per-target weighting in this baseline.

## Complete microbatch computation

`Trainer` receives the sampler and loss as ready-to-use components. Each
`compute_flux_microbatch_loss` call receives the model, preflighted
`packing_backend`, token budget and loss function, together with its selected `FluxInput` objects and the corresponding `flow_batch`:

```text
FIFO targets -> per-target flow draws -> select draws for this packed microbatch
    -> FLUX velocity prediction with separate FA3 attention segments
    -> selected FP32 velocity objective
```

Flow draws are required; the helper never samples or changes the time distribution.
The selected loss receives only predicted velocity, target velocity and the
output-grid dimensions in microbatch order. It has no dependency on `FluxInput`
or NVARC state ownership, and grid dimensions never reach the transformer.
The helper builds one packed `FluxMicrobatch`. Only the transformer forward runs inside an autocast region
using the supplied compute dtype. Persistent transformer
parameters retain their loader-established FP32 dtype, while the loss is
computed after leaving autocast and explicitly uses FP32 inputs. The function
returns a `LossResult` containing the scalar mean, per-target losses and any
already-computed regional reductions. The trainer returns these measurements
as CPU-only `TargetMetrics` in logical target order, alongside each target's
time and demonstration count. Grouping and averaging for logging belong to the
caller. The microbatch helper
does not scale the loss for an optimizer batch or call `backward()`.

## Validation boundaries

The hot path checks only the narrow structural contract needed to avoid silent
broadcasting or dtype errors: fixed latent shape, non-empty target count,
floating latent dtype, FP32 persistent FLUX parameter storage, matching
prediction and target shapes, and matching loss devices. It does not scan
complete tensors for finite values on every training step. Explicit preflights
and the training runner provide finite-value diagnostics and recorded RNG seeds.
The trainer rejects non-finite losses before backward and non-finite gradient
norms before the optimizer update. It clears gradients even when a step fails.

Unit tests establish:

- exact straight-path interpolation and velocity supervision;
- independent per-target timestep shape and broadcasting;
- deterministic outputs from identically seeded generators;
- latent dtype and device preservation with FP32 timesteps;
- immutable output contracts;
- BF16-autocast compute with FP32 parameters and gradients;
- unweighted FP32 MSE with finite prediction gradients; and
- one complete two-target, 900-image-token tiny FLUX forward and backward.

## Explicit non-goals

The complete microbatch computation does not:

- load or select a FLUX checkpoint;
- choose a non-uniform timestep density or loss weighting;
- select the logical targets belonging to an optimizer batch;
- scale losses, call `backward()`, accumulate gradients, or update parameters;
- configure gradient scaling, gradient checkpointing, or distributed execution;
  or
- define the inference-time ODE schedule.
