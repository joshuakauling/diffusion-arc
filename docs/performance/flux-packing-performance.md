# FLUX packing performance

This bounded experiment compares existing B2 microbatches with padded cuDNN
and full-model packing through FA2, FA3 and FA4. Target membership, order and
baseline padding boundaries come unchanged from the recorded B2 scheduler.
Every packed variant uses the validated
[FP32 modulation-gradient reduction](flux-modulation-backward.md).

The six groups cover four short targets, four medium targets, two long targets,
the 8,400-token source, two medium targets and a padded long singleton. Each
target also has 900 image tokens. Real NVARC prefixes and VAE latents are prepared
once; each group receives fixed flow draws shared by all four variants.
Each physical mean loss is weighted by its target count divided by 48.

One warmup per group/backend precedes two rotating rounds of three measured
forward/backward calls: 144 timed calls in total. CUDA events separate forward
and backward; synchronized wall time also includes input assembly, layout
construction and processor installation/restoration. Gradient reset, input
preparation, optimizer work, numerical scans and profiler calls are excluded.
These are microbatch measurements, not complete optimizer-step throughput.

Frozen models, FP32 FLUX parameters, persistent FP32 gradients, actual AdamW
moments and the existing preparation reserve remain resident. One LR=0 setup
step initializes optimizer storage; comparison calls do not update weights.
Checkpointing is disabled. The experiment searches neither larger packs nor a
memory limit, and does not change the production scheduler.

Outside timing, each variant must pass small BF16 attention forward/backward
and segment-isolation checks, full-model output/loss screening, finite complete
FP32 gradient coverage and the expected modulation-reduction call counts.
The first group verifies all four forward/backward backend routes using CPU
operator events. CUDA event timing does not depend on CUDA kernel-trace capture.
Full parameter-gradient differences are not recomputed in this performance run.

The launcher reuses image `im-R5CncDboPfMQryWnq0UZn4` and existing checkpoints,
with one H200, eight CPU cores and 64 GiB RAM. The original lifecycle envelope
was 600 seconds, including startup/persistence/cleanup, corresponding to a
list-price resource estimate of $0.904728 using [Modal's published prices](https://modal.com/pricing).
This is an estimate, not an account
billing receipt. There are no automatic retries or image builds. Completed
groups remain in partial artifacts if the budget expires.

The first attempt, `20260909T035828160945Z`, stopped before timing after 134
seconds of app lifecycle (estimated $0.202056). Its generic SDPA profiler gate
incorrectly expected 25 entries. Earlier actual padded-batch profiles contain
50 generic entries under autocast but exactly 25 cuDNN forward and backward
operators. The corrected check counts the specific dispatched backend and
still rejects a different backend or incomplete backward coverage. The first
attempt passed the three kernel preflights and baseline full-gradient health.

The corrected attempt has a 330-second worker deadline and 450-second lifecycle
envelope. Together with the first attempt this is at most 584 lifecycle seconds,
estimated $0.880602, within the original envelope. The two attempts and their
different source/config hashes are retained separately.

Configuration: `configs/diagnostics/flux_packing_performance.json`.
Launcher: `scripts/cloud/probe_flux_packing_performance_modal.py`.
Results namespace: `results/flux-packing-performance/<run-id>`.
Seeds, source/config/data hashes, checkpoint hashes, package versions, physical
shapes, raw timings, numerical screening and backend evidence are saved with
each run.

## Measured results

Run `20260909T040439157616Z` completed all 144 timed calls, 24 warmups and four
full-model backend profiles on H200. The worker took 162.34 seconds and the app
lifecycle 206 seconds. Both apps are stopped with zero tasks. Combined with the
failed first attempt, 340 lifecycle seconds imply an estimated $0.512679 of
resource charges; actual account billing was not queried.

Median assembly + forward + backward time in milliseconds (six samples per
cell; lower is better):

| Fixed B2 group | Actual text lengths | Padded cuDNN | Packed FA2 | Packed FA3 | Packed FA4 |
| --- | --- | ---: | ---: | ---: | ---: |
| Four short targets | 468, 406, 357, 288 | 319.6 | 298.8 | 287.7 | 288.2 |
| Four medium targets | 2077, 2017, 2009, 1956 | 703.2 | 690.8 | 641.8 | 640.3 |
| Two long targets | 4745, 4227 | 794.2 | 732.8 | 644.1 | 642.7 |
| Longest source | 8400 | 733.3 | 754.3 | 621.8 | 620.2 |
| Two medium targets | 2740, 2526 | 469.3 | 452.8 | 413.5 | 414.5 |
| Padded long singleton | 4948, padded to 6100 | 497.9 | 423.6 | 373.1 | 371.1 |

FA3 and FA4 take approximately 9–25% less time than padded cuDNN in these
groups. Their differences are at most 0.52%, with the nominal winner varying
by shape. Two rounds of three calls do not establish a meaningful overall
winner between them. FA2 is slower than FA3/FA4 throughout and is slightly
slower than cuDNN for the 8,400-token singleton. These selected groups are not
a dataset-weighted estimate of training throughput.

Forward/backward medians are available separately in the raw report. For
example, the two-long-target group changes from 275.3/518.4 ms with cuDNN to
213.8/427.0 ms with FA3. Packed metadata assembly adds about 1.6–3.0 ms per
call, compared with about 0.3 ms for baseline text padding.

Peak allocated GPU memory in GiB includes the resident training storage and
reserve. All three FlashAttention variants have the same peaks at this displayed precision:

| Group | Padded cuDNN | Packed FA2/FA3/FA4 |
| --- | ---: | ---: |
| Four short targets | 98.76 | 98.52 |
| Four medium targets | 128.01 | 129.93 |
| Two long targets | 125.25 | 125.69 |
| Longest source | 115.59 | 118.68 |
| Two medium targets | 106.58 | 107.91 |
| Padded long singleton | 105.29 | 102.10 |

Packing does not guarantee lower memory: this implementation materializes
per-token modulation vectors and token-ordering buffers, while B2 already
reduces much of the padding. The measured net memory difference is negative
for two groups and positive for four. Isolating each buffer's contribution
would require memory profiling; these totals alone do not attribute the
entire increase to a single operation.

All groups have finite gradients for all 169 parameter tensors
(3,875,544,576 parameters), persistent FP32 gradient storage and unchanged
weights and inputs. All packed warmups observed the required three FP32
modulation reductions. Output relative L2 against padded cuDNN ranges from
0.606% to 0.891%; relative loss differences remain at or below 0.873%, within
the existing 2%/1% screening limits. This run checks finite gradient coverage,
not a new full-gradient equivalence bound or an optimizer update.

All four requested forward/backward routes were verified. Runtime versions:
PyTorch 2.13.0, CUDA 13.0, cuDNN 92000, Diffusers 0.39.0, FA2 2.8.3.post1,
FA3 3.0.0 and FA4 4.0.0b29. TF32 and checkpointing were disabled. The retrieved
manifest matches all 83 source, three configuration and 965 dataset hashes.

## Implications

FA3 and FA4 are the candidates for the next capacity experiment. FA3 is a
reasonable first choice because its full-model FP32 modulation intervention
has already been compared quantitatively against singleton gradients; this
performance result provides no substantial speed reason to switch to FA4.
FA4 remains a useful cross-check near the eventual operating pack size.

The largest pack measured here contains 11,659 valid text-plus-image tokens,
with a 129.93-GiB peak. It is a successful operating point, not a maximum.
Larger packs must be checked against the full memory footprint and segment
length distribution rather than assumed safe merely because padding vanished.

The current B2 scheduler remains useful as the controlled baseline. Its padded
shape cost catalog should not be reused as an optimum for packed execution.
A later FIFO token-budget packer may replace that catalog if it provides the
same or better complete-optimizer-batch throughput while preserving example
ownership, loss weighting and flow-draw assignment. No production scheduler
or training default has changed in this experiment.

Local validation covered 48 distinct tests across the new coordinator/launcher,
existing packing coordinator and modulation reduction. The added regression
accepts nested generic SDPA events while requiring exact cuDNN kernel dispatch
and backward coverage. Raw evidence lives under the result namespace above;
the earlier failed attempt remains separately identified.
