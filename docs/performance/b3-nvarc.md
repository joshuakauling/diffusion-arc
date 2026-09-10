# B3 NVARC: from padded batches to FA3 packing

B3 runs all 16 NVARC sources in **one unpadded FA3 call**, with separate causal
attention and positions starting at zero for each source. It also stops after
block 27 of 36. This removes the padding catalog and length-based batch selection,
and skips the last quarter of the transformer blocks.

## Alignment: useful for some shapes

We first tried rounding B2's padding lengths to multiples of 8, 32, 64, and
128. The H200 probe covered 72 shapes. The strongest improvement was
**4 × 4,507 → 4 × 4,512 tokens: 631.7 → 534.9 ms (−15.3%)**. Other shapes
were fastest at their original length. There was no generally better rounding
rule, and we did not change B2's catalog.

## Packing: remove the shared padding length

The next test changed the attention backend to FA2, first keeping B2's groups,
then packing the whole window into one call. For the median-length window:

| Execution | NVARC calls | Time per 16-source window |
| --- | ---: | ---: |
| cuDNN, B2 padded groups | 4 | 1,473.2 ms |
| FA2, same padded groups | 4 | 1,084.9 ms |
| FA2, unpadded packing | 1 | 852.4 ms |

Both changes helped: switching the backend, then removing padding and repeated
model calls. These measurements used synthetic ARC token contents at recorded
benchmark lengths, with five measured calls per case.

## Backend choice: FA3 without alignment

We then replayed real benchmark inputs with four packed backends. All used
frozen BF16 NVARC on H200. Times are medians of five calls, in milliseconds
per complete 16-source window:

| Window | FA2 | FA3 | FA4 | cuDNN ragged |
| --- | ---: | ---: | ---: | ---: |
| Short | 644.1 | 616.1 | 615.2 | 621.3 |
| Median | 858.8 | 812.8 | 814.8 | 820.3 |
| With longest source | 1,081.1 | 1,000.7 | 1,012.8 | 1,017.4 |

FA3 took **4.3–7.4% less time than FA2** and was close to FA4 and cuDNN ragged.
We selected FA3 without alignment, which gave no consistent benefit. The table
covers the three complete windows; a fourth was incomplete at the worker time
limit. These few windows do not establish a universal winner.

FA3's selected hidden states differed from the same-input cuDNN/B2 reference
by **0.50–0.55% relative L2**; maximum absolute differences were 8.5–18.5.

## Stop after the last required hidden state

FLUX only needs the raw NVARC states after blocks 9, 18, and 27. B3 therefore
skips blocks 28–36, final normalization, and the language-model head. The full
checkpoint stays loaded; unused weights are still in GPU memory.

Tiny-Qwen CPU tests matched the selected states exactly against a full forward.
The timings above still executed all 36 blocks and the output head; they do
not measure the early-exit gain. The [B3 benchmark](b3.md#benchmark-result)
measures the final combined NVARC and FLUX implementation.

## Evidence

- [Alignment results](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/nvarc-batch-shapes.md): complete shape table and protocol.
- [Packing and backend results](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/nvarc-packing.md): full timing, memory, numerical
  checks, run IDs, and archived diagnostics.
- [NVARC encoder](../training/nvarc-batch-runner.md): implementation and early-exit checks.
