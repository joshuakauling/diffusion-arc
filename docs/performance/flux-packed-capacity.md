# FA3 reduction performance and packed capacity

This experiment measures the cost of the three explicit FP32 modulation-gradient
sums and bounds full-model packing capacity using real ARC examples. It follows
the [fixed-microbatch performance comparison](flux-packing-performance.md).

For the controlled length limit of **one attention sequence**, use the later
[single-sequence capacity test](flux-sequence-capacity.md). The whole-example
growth experiment below does not isolate that quantity.

The reduction comparison uses the same four medium B2 targets: 2,077, 2,017,
2,009 and 1,956 text tokens, plus 900 image tokens per target (11,659 total).
Only `modulation_reduction` changes between `native` and `fp32`; FA3, BF16
compute, FP32 parameters, loss weighting, inputs and example layout are shared.
Two warmups per variant precede four alternating rounds of five measured calls
per variant. Forward, backward, synchronized assembly-plus-F/B wall time and
peak allocated/reserved memory are recorded. Output and loss must remain exactly
equal at the warmup control; finite complete FP32 gradients are checked outside
timing. FA3 forward/backward dispatch is profiled separately using CPU events.

The capacity phase always uses FP32 modulation sums. Two fixed source pools
provide whole examples: one starts with the comparison group and appends shorter
real targets, while the other starts with the 8,400-token source and appends
additional real targets. Every original prefix, latent, timestep and noise draw
remains assigned to its example as the pack grows. The first model-execution
allocation OOM stops each growth sequence. Its failed graph is released, cached
allocations are cleared, and the last successful pack must pass three further
forward/backward calls. Other runtime failures do not count as capacity bounds.

The result is a largest confirmed pack and the first failing next composition
for each recorded growth sequence, not a universal or token-exact limit. All
segment lengths and total text-plus-image tokens are retained in the result.
Resident memory includes frozen NVARC/VAE, FP32 FLUX parameters and gradients,
initialized AdamW moments and the existing preparation reserve. One LR=0 setup
step initializes optimizer storage. No optimizer steps occur during comparison
or search. Checkpointing and TF32 are disabled. Timing excludes gradient reset,
input preparation and optimizer work.

The single H200 launcher reuses image `im-R5CncDboPfMQryWnq0UZn4`, eight CPU
cores and 64 GiB RAM. The worker deadline is 480 seconds, function deadline 540
seconds, startup deadline 90 seconds and lifecycle envelope 660 seconds. There
are no image builds, downloads or automatic retries. At
[Modal's published rates](https://modal.com/pricing), the lifecycle resource
estimate is at most $0.995201, leaving about $0.324799 of the reported $1.32
balance. Actual account charges may differ from this conservative estimate.

Configuration: `configs/diagnostics/flux_packed_capacity.json`.
Launcher: `scripts/cloud/probe_flux_packed_capacity_modal.py`.
Artifacts: `results/flux-packed-capacity/<run-id>` on the model volume.
Source/config/data/checkpoint hashes, environment, seeds, raw timings, backend
events, allocation errors and confirmation results are persisted with each run.

Local validation: 39 targeted tests passed, covering whole-example growth,
reduction switching, failed-graph release, OOM classification, confirmation,
partial-result persistence, launcher limits and the existing numerical helper.

## Results

Run `20260909T061831725274Z` completed successfully. Its worker took 85.96
seconds, supervisor 93.52 seconds, and app lifecycle 121 seconds. App
`ap-KL31EJYKrhjnPnwSPWUGTT` is stopped with zero tasks. The lifecycle-based
resource estimate is $0.182453; actual account billing was not queried.

Median performance at 11,659 joint tokens, from 20 measurements per variant:

| Modulation reduction | Forward ms | Backward ms | Assembly + F/B ms | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Native gather backward | 222.71 | 434.53 | 660.15 | 129.9319 |
| Explicit FP32 segment sums | 222.33 | 432.03 | 657.81 | 129.9319 |

FP32 has a 0.355% lower total median and 0.575% lower backward median. The
paired round differences range from FP32 being 0.239% slower to 0.698% faster.
This establishes no meaningful performance penalty at the tested shape, but
does not establish a reliable speedup. Peak allocated and reserved memory are
exactly equal between variants (139,513,324,544 and 144,596,533,248 bytes).
The forward output fingerprint and loss are identical, and both warmup controls
have finite gradients for all 169 parameter tensors / 3,875,544,576 parameters.

Capacity results use FP32 modulation sums throughout:

| Family | Confirmed text segment lengths | Joint tokens | Confirmation peak GiB | First failed text lengths | Failed joint tokens |
| --- | --- | ---: | ---: | --- | ---: |
| Typical | 2077, 2017, 2009, 1956 | 11,659 | 129.95 | 2077, 2017, 2009, 1956, 617 | 13,176 |
| Long source | 8400 | 9,300 | 118.68 | 8400, 2356 | 12,556 |

Both allocation failures occurred in measured model execution. Each previous
successful pack subsequently passed all three confirmations after failed-graph
release and allocator cache cleanup. Weights, inputs and persistent FP32
gradient storage remained unchanged. Local validation of the retrieved artifacts
matches all 84 source, three configuration and 965 dataset hashes.

## Interpretation and next validation

Keep explicit FP32 modulation sums: the earlier gradient improvement has no
measurable cost in peak memory and no meaningful timing penalty at this shape.
There is no performance reason here to restore the native gather reduction.

The failed larger packs are **observed execution limits, not definitive physical
capacity limits**. The typical failure requested 618 MiB with 131.22 GiB allocated
by PyTorch and 7.53 GiB reserved but unused. The long-source failure requested
664 MiB with 131.41 GiB allocated and 7.42 GiB reserved but unused. Device free
memory was about 303 and 221 MiB respectively. These figures suggest allocator
fragmentation, but no memory snapshot or exact allocator-setting capture was
made, so the cause is not proven. The failed compositions were not retried after
cleanup; only the smaller successful controls were rerun.

PyTorch documents how changing allocation sizes can leave unused slices in
partially occupied segments, and provides experimental `expandable_segments`
to address this pattern. See the
[allocator documentation](https://docs.pytorch.org/docs/2.14/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf).
That documentation describes the mechanism; the measured runtime was PyTorch
2.13.0, and support/effect on this image remains to be tested.

The later single-sequence test instead isolates length and records allocator
state, with cache cleanup before every probe and confirmation of both sides
of the boundary. It establishes a fitting length of 13,312 joint tokens and
a reproducible OOM at 13,568. Neither these whole-example results nor the old
B2 shape catalog should be treated as universal hardware limits. No allocator
setting or production scheduling policy was changed by either experiment.
