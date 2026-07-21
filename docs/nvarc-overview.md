# NVARC Qwen3-4B: Method, Scaling, Data, and Evaluation Caveats

## Summary

The Qwen component of NVARC is best understood as a substantially scaled and
re-engineered version of the 2024 ARChitects method, rather than a new ARC
reasoning architecture. It keeps the same basic recipe:

1. serialize ARC grids as short token sequences;
2. fine-tune a causal language model on augmented ARC tasks;
3. adapt that model to each test task with LoRA test-time fine-tuning (TTFT);
4. generate candidates with probability-bounded depth-first search (DFS); and
5. use the language model again to score candidates under multiple task
   augmentations.

NVARC changes the scale and implementation of almost every stage. It moves to
Qwen3-4B-Thinking-2507, reduces the vocabulary further, performs full-parameter
supervised fine-tuning on 3.26 million samples, uses a much larger synthetic
task corpus, and accelerates candidate generation with batched DFS. The novel
part of the solution is primarily the synthetic-data pipeline and the resulting
training scale. The NVARC authors explicitly describe the ARChitects method as
the modeling approach they reused because its quality scaled well with added
data. [NVARC paper](https://github.com/1ytic/NVARC/blob/main/nvarc_2025.pdf)

NVARC as a competition entry also explored Tiny Recursive Models (TRM), but the
authors selected the standalone Qwen3-4B submission for their final entry. This
document therefore uses *NVARC model* to mean the Qwen3-4B component unless
stated otherwise.

## Model and training specifications

The released checkpoint belongs to the Qwen3-4B-Thinking-2507 family. The base
architecture has 4.0B nominal parameters, 3.6B non-embedding parameters, 36
layers, a hidden size of 2,560, an intermediate size of 9,728, and grouped-query
attention with 32 query heads and 8 key/value heads. Its native context limit is
262,144 tokens. These are properties of the upstream Qwen checkpoint; NVARC
removes almost all vocabulary rows, so `4B` should be read as the model family
rather than an exact post-conversion parameter count. [Qwen model
card](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)

| Property | NVARC Qwen3-4B |
| --- | --- |
| Upstream model | Qwen3-4B-Thinking-2507 |
| Architecture | Decoder-only causal transformer |
| Layers / hidden width | 36 / 2,560 |
| Attention heads | 32 query heads, 8 key/value heads |
| ARC vocabulary | 16 tokens |
| Offline adaptation | Full-parameter supervised fine-tuning |
| Offline precision | BF16 |
| Offline data | 3,255,481 augmented samples; up to 7 input/output pairs per sample |
| Reported offline compute | 4 nodes x 8 H100 GPUs for 27 hours |
| TTFT adapter | Rank-256 LoRA, alpha 32, one epoch over task augmentations |
| TTFT precision | BF16, without 4-bit quantization or gradient checkpointing |
| Published inference context limit | 8,192 tokens in the Kaggle notebook |
| Candidate generation | Batched probability-bounded DFS |
| Candidate selection | Shared eight-augmentation rescoring plus generation frequency |

The fine-tuned checkpoint is published on Kaggle as
[`qwen3_4b_grids15_sft139`](https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139).
The training configuration names the upstream model, uses BF16 and Megatron,
and lists every dataset in the final training mixture. [NVARC SFT
configuration](https://github.com/1ytic/NVARC/blob/main/ARChitects/sft_mg.yaml)

### ARC representation

ARChitects reduced a normal LLM vocabulary to 64 tokens and serialized each
cell as one token. It also used learned alphabet tokens as a prefix or
"computational buffer." NVARC simplifies this to a 16-token vocabulary covering
the ten colors, newline, user/assistant markers, chat boundary tokens, and
padding. An input/output pair is represented in Qwen's compact dialog form:

```text
<|im_start|>user
123
456<|im_end|>
<|im_start|>assistant
78
90<|im_end|>
```

This preserves one token per grid cell while removing natural-language tokens,
digit merges, and most formatting overhead.

## What changed from ARChitects

The inheritance is direct, but the implementation is not identical.

| Component | 2024 ARChitects | NVARC Qwen component |
| --- | --- | --- |
| Main language model | Mistral-NeMo-Minitron-8B; Llama-3.2-3B was also evaluated | Qwen3-4B-Thinking-2507 |
| Offline training | Rank-256 LoRA with 4-bit base quantization | Full-parameter BF16 fine-tuning |
| Offline task source | Primarily RE-ARC, derived from the 400 ARC training tasks | Real ARC datasets plus about 103k generated tasks |
| Reduced vocabulary | 64 tokens | 16 tokens |
| TTFT | Rank-32 LoRA, alpha 16, 64 steps, 4-bit base | Rank-256 LoRA, alpha 32, one epoch, BF16 base |
| Candidate generation | DFS over each augmentation | Batched DFS, scaling approximately linearly with batch size |
| Rescoring | Product-of-experts likelihood over 16 additional augmentations | Eight augmentations shared across every candidate, combined with DFS occurrence count |

The generator/scorer structure, task augmentations, TTFT, and DFS are therefore
ARChitects ideas. NVARC's main additions are the new data pipeline, the Qwen3
model and representation, larger offline and test-time adaptation, and systems
optimizations needed to fit the Kaggle runtime. The original method and its
ablations are described in the [ARChitects ICML
paper](https://proceedings.mlr.press/v267/franzen25a.html).

The batched DFS implementation introduced numerical nondeterminism. The team
tested batch-invariant operations and obtained better precision and local
validation, but the resulting inference was about 17% slower and was not used
for the final submission. The paper also reports one-to-two leaderboard-point
variation between reruns of the same notebook. These details matter when
interpreting small score differences.

## Synthetic-data pipeline

The data pipeline is the central NVARC contribution. It moves expensive
reasoning and program generation offline, then distills the resulting task
distribution into a model small enough for Kaggle inference.

### 1. Collect descriptions and normalize them into summaries

The seed descriptions came mainly from H-ARC human solution attempts and 160
human-written BARC descriptions. After filtering and merging, the team had
descriptions for 716 ARC-AGI-2 training tasks. Claude Opus 4 and GPT-OSS-120B
were used to create three structured summaries per task.

The summary schema contained input-generation instructions, solution steps, a
rule summary, a key insight, and a list of puzzle concepts. The team then added
29 manually written summaries for ARC-AGI-2 evaluation tasks and generated
summaries for the remaining 91 evaluation tasks and all 1,000 training tasks
using Claude Sonnet 4.5. This produced 3,268 seed summaries.

### 2. Mix concepts

GPT-OSS-120B combined pairs of summaries into more complex task descriptions,
following the general idea of skill mixing. This stage produced 266,593 mixed
summaries.

### 3. Generate and validate input-grid programs

For each summary, the model generated Python code for input-grid construction
and unit tests for that code. A program was retained only if it generated at
least 30 unique input grids and all grids passed the generated tests. The paper
reports 126,901 accepted input programs.

### 4. Generate output transformations by consensus

The model separately generated Python output-transform programs conditioned on
the summary and input program. Instead of trusting a single program, the
pipeline sampled multiple implementations and retained transformations that
agreed on the same 30 generated inputs. The paper gives an example of sampling
20 output programs and retaining the 15 consistent ones. After this stage it
reports 103,253 generated puzzles, each with roughly 30 input/output pairs.

### 5. Build augmented SFT samples

Generated tasks were combined with real and programmatically augmented ARC
sources. The final SFT table contains 104,539 unique tasks in total, including
real datasets, and 3,255,481 augmented samples:

| Source | Unique tasks | Samples per task | Total samples | Share |
| --- | ---: | ---: | ---: | ---: |
| MINI-ARC | 147 | 256 | 37,632 | 1.2% |
| ConceptARC | 160 | 256 | 40,960 | 1.3% |
| RE-ARC | 400 | 256 | 102,392 | 3.2% |
| ARC-AGI-2 | 609 | 256 | 155,904 | 4.8% |
| `nvarc_training` | 47,337 | 24 | 1,132,633 | 34.8% |
| `nvarc_full` | 55,886 | 32 | 1,785,960 | 54.9% |
| **Total** | **104,539** |  | **3,255,481** |  |

The paper distinguishes the two generated subsets explicitly:

- `nvarc_training` is based only on ARC-AGI-2 training-task descriptions.
- `nvarc_full` is based on both ARC-AGI-2 training and evaluation-task
  descriptions.

The reported 103,253 generated puzzles and the 103,223 tasks in the two NVARC
rows are slightly different counts in the paper. They appear to refer to
different pipeline and final-mixture stages; the paper does not reconcile the
30-task difference. The 104,539 total must not be described as entirely
synthetic because it also includes the four real or independently generated
sources above.

## Data leakage and evaluation limits

The released model is not clean with respect to the public ARC-AGI-2 evaluation
set.

First, the team manually summarized 29 public evaluation tasks and generated
summaries for the other 91. Those summaries entered concept mixing and the
program-generation pipeline. The resulting `nvarc_full` subset accounts for
54.9% of the final augmented training samples. This is not necessarily literal
memorization of the original test outputs: the descriptions were transformed
into mixed summaries and newly generated programs. It is nevertheless direct
evaluation-derived training signal, so public-evaluation accuracy is not an
independent generalization estimate.

Second, the released SFT configuration uses `arc2_evaluation6` as its validation
dataset, selects checkpoints by validation loss, and retains the best three.
The paper explicitly acknowledges that evaluation-puzzle descriptions had
"leaked" into the synthetic data while reporting correlation between local
validation loss and the live leaderboard. This adds model-selection leakage on
top of the generated-data path.

The hidden Kaggle sets remain more informative for system-level generalization.
The paper reports 27.64% on the live leaderboard during the competition and
29.72% for a post-deadline variant. The official final private result was
24.03%. These are complete-pipeline results, not frozen-checkpoint results: they
include TTFT, DFS, augmentation rescoring, and the available runtime budget.
[ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis)

## Measured frozen no-TTFT baseline

We evaluated the released Qwen3-4B checkpoint without test-time fine-tuning on
July 21, 2026. The run used the repository's frozen-inference notebook, four
NVIDIA L4 GPUs, seed 42, BF16 weights, a maximum sequence length of 8,192,
sampling temperature 1.0, a DFS token-probability threshold of 0.2, a 540-second
DFS-call limit, and a 1,200-second per-task limit. All four worker manifests
confirmed a 16-row embedding and output head, BF16 parameters, and no trainable
parameters. The Kaggle kernel was `joshuakauling/arc-agi-2-nottt`; the public
evaluation run ID was `20260721-032526-evaluation` and the checkpoint revision
was `qwen3_4b_grids15_sft139/transformers/bfloat16/1`.

The public evaluation set contains 120 tasks and 172 test outputs: 71 tasks have
one test output, 46 have two, and three have three. The percentages below use
the notebook's task-averaged metric: test outputs are averaged within a task and
then all 120 tasks receive equal weight. `Oracle all` asks whether any generated
candidate was correct, before limiting the result to the two submitted outputs.

| Variant | Generation and selection | Pass@1 | Pass@2 | Oracle all | Fully solved tasks at 2 |
| --- | --- | ---: | ---: | ---: | ---: |
| `G0` | Greedy, canonical view, lowest NLL | 6.67% | 6.67% | 6.67% | 7/120 |
| `G16` | Greedy over 16 views, frequency/NLL | 10.69% | 13.75% | 19.86% | 15/120 |
| `S16` | One stochastic sample per view, frequency/NLL | 6.94% | 7.78% | 10.42% | 9/120 |
| `D1` | DFS, canonical view, lowest NLL | 2.50% | 2.50% | 2.50% | 3/120 |
| `D16` | DFS over 16 views, frequency/NLL | 11.25% | 11.67% | 16.25% | 13/120 |
| `D16R` | DFS over 16 views, augmentation rescoring, `kgmon` selector | 12.50% | 14.58% | 16.25% | 17/120 |
| `D16R_probmul` | Same rescored candidates, probability-multiplication selector | 10.83% | **16.25%** | 16.25% | **19/120** |

`D16R` was the primary selector used to produce `submission.json`. It solved 18
of 172 individual outputs at pass@1 and 21 of 172 at pass@2; the second attempt
rescued three outputs. A stricter whole-task count is 15/120 at pass@1 and
17/120 at pass@2, compared with the task-averaged values of 12.50% and 14.58%.
These definitions must not be interchanged when reporting a solution rate.

`D16R_probmul` uses exactly the same generated and rescored candidates as
`D16R`, so its 1.67-point pass@2 gain required no additional model inference in
this run. It reached the candidate oracle on the public evaluation set, but this
is a single contaminated-set result and is not evidence that the selector will
improve a hidden split.

All submitted arrays were syntactically valid ARC grids. Output dimensions were
not supplied or enforced, and wrong-shape candidates were not filtered. Of the
172 test outputs, attempt 1 had the wrong target shape for 105, attempt 2 for
129, and both attempts for 104. This records the consequence of shape-agnostic
decoding without assuming that the target dimensions are known.

The diagnostic variants were not run as equal-compute standalone experiments.
For each task, `D16` and `D16R` ran first; `G0`, `G16`, `S16`, and `D1` then used
the remaining portion of the shared per-task deadline. Sixty of 120 task files
reported reaching that deadline. The table therefore measures the configured
evaluation schedule, not an isolated runtime-matched comparison of every
decoder.

### Public versus semi-private behavior

The same frozen `D16R` system received a 9.58% score on the 2026 Kaggle live
leaderboard. This produces the opposite public/semi-private ordering from the
authors' self-reported complete pipeline:

| System | Public evaluation | Semi-private / live leaderboard | Direction |
| --- | ---: | ---: | --- |
| Frozen checkpoint, no TTFT, `D16R` | 14.58% | 9.58% | Public +5.00 points |
| NVARC complete pipeline with TTFT | 30/120 = 25.00% | 29.72% | Semi-private +4.72 points |

Taken literally, the two rows imply a 10.42-point complete-pipeline increment
on public evaluation and a 20.14-point increment on semi-private. Those
differences are not a controlled estimate of the causal contribution of TTFT.
The authors' result also includes their full inference pipeline and budget, the
public 30/120 value is a strict puzzle count rather than the task-averaged score
used above, and the released checkpoint is contaminated with public-evaluation
signal. The reversal is nevertheless a useful observation: the frozen system
does better on the contaminated public set, while the self-reported adapted
system does better on the hidden live set. Plausible explanations include a
larger benefit from per-task adaptation on novel tasks, split variance, metric
differences, and other complete-pipeline differences. The current evidence does
not distinguish among them.

The 9.58% value is recorded from the observed Kaggle submission, but its exact
submission ID has not yet been preserved. ARC Prize materials describe both the
2025 and 2026 live evaluations as the 120-task ARC-AGI-2 semi-private set, but we
have not found an explicit byte-identity statement for the hidden split across
competition years. Cross-year comparison therefore remains an evidence gap.

A separate submission-mode sanity run processed the 240 visible Kaggle test
placeholder IDs. They exactly matched public training-task IDs and had no
overlap with public evaluation IDs. It obtained 248/259 = 95.75% at output-level
pass@1, 252/259 = 97.30% at output-level pass@2, and 233/240 = 97.08% strict
task-level pass@2. These values only validate the inference and scoring path;
they are not a generalization benchmark.

## Released checkpoints and remaining evidence gaps

Two post-competition checkpoints are public:

- [Qwen3-4B](https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139)
- [Qwen3-VL-2B language model](https://www.kaggle.com/models/sorokin/qwen3_2b_grids15_sft141)

The paper states that both were fine-tuned on exactly the same 3.2-million-sample
dataset. The 2B model therefore does not remove the public-evaluation
contamination. No official checkpoint trained only on `nvarc_training` and the
safe real-data sources is currently documented by the authors.

The NVARC authors did not report a frozen-decoder ablation. The measurements
above now provide that missing baseline for the released Qwen3-4B checkpoint,
including canonical greedy decoding, augmented greedy and stochastic decoding,
and DFS with augmentation rescoring. The official released notebook still
creates a fresh LoRA adapter, trains it on augmented demonstrations for each
task, and only then runs DFS and rescoring.

The remaining high-value ablation is a controlled TTFT on/off comparison using
the same checkpoint, candidate generation, selector, runtime budget, and hidden
task split. Without it, the difference between the measured frozen results and
the authors' complete-pipeline results cannot be attributed to TTFT alone.
Results from the 2024 ARChitects models also cannot be transferred to NVARC
because the model, vocabulary, offline training method, and data distribution
all changed. The official inference implementation is available in the [NVARC
Kaggle notebook](https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue).

## Primary sources

- [NVARC paper](https://github.com/1ytic/NVARC/blob/main/nvarc_2025.pdf)
- [NVARC source repository](https://github.com/1ytic/NVARC)
- [NVARC SFT configuration](https://github.com/1ytic/NVARC/blob/main/ARChitects/sft_mg.yaml)
- [NVARC Qwen3-4B checkpoint](https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139)
- [NVARC submission notebook](https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue)
- [ARChitects paper](https://proceedings.mlr.press/v267/franzen25a.html)
- [Qwen3-4B-Thinking-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
- [ARC Prize 2025 results and analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
