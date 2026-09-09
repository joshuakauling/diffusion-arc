# B2 preparation policy

This first B2 implementation reduces NVARC padding within each consecutive
16-source preparation window and overlaps CPU data loading with GPU work.
It retains B1's 48-target optimizer batches, FLUX microbatch size 1, FLUX padding
boundaries, and VAE batches of up to 48 targets. This preparation-only config
remains available; the combined B2 config additionally uses the
[measured FLUX shape catalog and scheduler](flux-batch-shapes.md#measured-results-and-initial-b2-catalog).

## NVARC shapes and scheduling

The catalog contains explicit `(batch_size, max_tokens)` pairs. A boundary is
available only at the batch size in its row; the two columns are not a cross
product.

| Batch size | Token upper bounds |
| --- | --- |
| 2 | 864, 3022, 4250, 6010, 6870, 8400 |
| 4 | 1083, 1431, 1718, 2190, 2824, 3122, 3372, 3780, 4204, 4507, 4836, 5380 |
| 8 | 2528, 2702 |

Eligibility is applied before scheduling. The scheduler stably sorts eligible
sources by descending consumed prefix length, then partitions them into
contiguous groups using dynamic programming. Its integer objective is:

`sum(batch_size * padded_length) + 2560 * number_of_calls`

Ties prefer fewer calls, followed by deterministic traversal in ascending batch
size and boundary. Regular NVARC batches pad explicitly to their selected
upper bounds, mask padding, and return only real hidden-state positions.

An odd eligible-source count uses exactly one additional singleton at its real
source length, without extra padding. Its cost is the real length plus the same
per-call weight. The scheduler chooses which source runs alone as part of the
whole-window optimization. This covers short stream tails and windows with an
ineligible source. Rare singletons may introduce new attention shapes; their
lengths are not restricted to the regular catalog boundaries. Even windows never
use this exception. There are no dummy sources, dropped sources, or groups
spanning preparation windows. The global 8,400-token source limit still applies.

The catalog was selected from the 960 source lengths recorded in the B0
benchmark (60 consecutive windows). Boundaries follow this observed distribution
rather than a uniform token grid. The call weight is a scheduling heuristic,
not a measured GPU latency. Catalog selection is heuristic as well; it does not
establish a global optimum or guarantee the fastest GPU execution.

| CPU replay | Padding above real positions | NVARC calls per window |
| --- | ---: | ---: |
| All 60 windows | 21.94% | 4.13 |
| 50 measurement windows | 22.26% | 4.14 |

The measurement-window replay uses 42.84% fewer padded positions than B1;
this is a workload estimate, not a measured speedup. Across all windows, 8 need
three calls, 36 need four, and 16 need five. Eighteen of the 20 regular shapes
occur in the first ten warm-up windows. Selecting catalogs on alternating
30-window subsets and checking the other half gave 25.14% combined padding;
this is an internal sensitivity check, not independent dataset validation.

The replay source is B0 repetition 1 from benchmark `20260903T230458451916Z`;
its ordered source keys and lengths match the other two repetitions. The
`steps.jsonl` SHA-256 is
`1422569fb812adf15c75cfccb7b3fdb074349d1f0647fed5a62ddee8b202f7b5`.

## Preparation order and prefetch

NVARC outputs are restored to original source order before constructing FLUX
inputs. VAE encoding follows that same source-major target order, in chunks of
at most 48 across the whole window, independently of the physical NVARC groups.
The target FIFO, optimizer-batch membership, flow RNG draw order, and per-target
loss weighting stay unchanged. Floating-point results can still differ with
different physical NVARC shapes.

Two CPU DataLoader workers use `spawn`, ordered delivery, and
`prefetch_factor=8`: up to 16 individual source tasks can be outstanding. Workers
load, augment, format, and tokenize sources; models remain in the main process.
The seeded sampler and flow generator remain independent. Prefetch does not
expand the GPU preparation window or count unused samples as prepared work.

## Configuration and validation boundary

[The preparation benchmark config](https://github.com/joshuakauling/diffusion-arc/blob/db6d49f/configs/training/benchmarks/single_gpu_b2_preparation.json)
uses schema version 5. `nvarc_scheduling` declares the catalog and call weight;
it is mutually exclusive with fixed NVARC padding boundaries. Exact-length
singletons are the fixed rule for odd windows and need no separate length
setting. `loop.nvarc_source_batch_size` denotes the preparation-window size
when scheduling is enabled. Schema versions 1–4 retain their existing behavior.

Benchmark rows record actual NVARC call shapes and source indices into that
step's `prepared_sources`, alongside real and padded token counts. Sources are
charged to the step that prepared them; target-prefix metadata follows FIFO
consumption, including across optimizer boundaries.

CPU tests cover scheduling, masking, source/target alignment, optimizer updates,
worker ordering, configuration compatibility, and workload recording. The
[completed combined B2 benchmark](README.md#b2-length-aware-dynamic-batch-single-gpu-pipeline)
measures GPU memory, throughput, and step-loss agreement with dynamic FLUX
batching included. The preparation-only variant was not benchmarked separately;
the combined result does not isolate scheduling, prefetch, or FLUX batching.
