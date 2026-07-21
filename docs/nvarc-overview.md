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

## Released checkpoints and missing evidence

Two post-competition checkpoints are public:

- [Qwen3-4B](https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139)
- [Qwen3-VL-2B language model](https://www.kaggle.com/models/sorokin/qwen3_2b_grids15_sft141)

The paper states that both were fine-tuned on exactly the same 3.2-million-sample
dataset. The 2B model therefore does not remove the public-evaluation
contamination. No official checkpoint trained only on `nvarc_training` and the
safe real-data sources is currently documented by the authors.

The most important missing ablation for a frozen-decoder study is the final
Qwen3-4B checkpoint without test-time fine-tuning. Published values such as
30/120 on local evaluation and 29.72% on the live leaderboard use the authors'
best complete inference settings. The released notebook always creates a fresh
LoRA adapter, trains it on augmented demonstrations for each task, and only then
runs DFS and rescoring. It does not report either of these controlled baselines:

1. frozen Qwen3-4B with simple autoregressive decoding; or
2. frozen Qwen3-4B with the official DFS and rescoring pipeline but no TTFT.

The frozen no-TTFT solution rate is therefore unknown. Results from the 2024
ARChitects models cannot be transferred to NVARC because the model, vocabulary,
offline training method, and data distribution all changed. The official
inference implementation is available in the [NVARC Kaggle
notebook](https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue).

## Primary sources

- [NVARC paper](https://github.com/1ytic/NVARC/blob/main/nvarc_2025.pdf)
- [NVARC source repository](https://github.com/1ytic/NVARC)
- [NVARC SFT configuration](https://github.com/1ytic/NVARC/blob/main/ARChitects/sft_mg.yaml)
- [NVARC Qwen3-4B checkpoint](https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139)
- [NVARC submission notebook](https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue)
- [ARChitects paper](https://proceedings.mlr.press/v267/franzen25a.html)
- [Qwen3-4B-Thinking-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
- [ARC Prize 2025 results and analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
