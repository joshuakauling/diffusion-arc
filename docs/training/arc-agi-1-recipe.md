# First ARC-AGI-1 training run

This is the agreed recipe for the first full epoch of single-GPU training.
It tests whether the lessons from [overfitting](../experiments/overfitting/README.md)
transfer to a broader task distribution. The runner executes the training epoch
with loss validation, fixed-seed sampling, W&B logging and full-state checkpoints.
RunPod Community Cloud is the target platform. Local upload and Pod startup are
implemented; deployment verification and final-test checkpoint selection remain open.

The [run configuration](../../configs/training/arc_agi_1_epoch.json) records
these decisions as `training_run` schema version 1. Its
[loader](../../src/diffusion_arc/training/config.py) validates the complete
recipe and normalizes it into the same `TrainingConfig` used by B3.
The [trainer step](../../src/diffusion_arc/training/trainer.py) executes one
complete optimizer update, including NVARC/VAE encoding. Sampler, loss, optimizer
and scheduler are built outside and supplied to the trainer. The outer loop
owns epoch accounting and checks its expected target/update totals.
The [training entrypoint](../../scripts/train_single_gpu.py) loads this config
before importing the GPU runtime and calls
[`run_training(config)`](../../src/diffusion_arc/training/runner.py). The runner
checks that the prepared view contains the expected number of distinct tasks
with the configured pair count, retains the partial final batch and checks
the observed sequence, target and update totals. Event conditions use the
configured start, periodic and final steps. Validation and sampling share one
W&B record with the current optimizer update, plus an initial record at step zero.
Checkpoints are published after sampling and their metrics join that record.
The separate [profiler](../../scripts/profile_single_gpu.py) and benchmark
entrypoints require baseline configs. Resolve relative paths from the JSON
file's directory. Before the complete run, set the
RunPod output directory and W&B entity/project before execution; credentials
come from `WANDB_API_KEY`. `checkpointing.keep_last: null` retains all three
first-epoch checkpoints. A numeric retention limit is deferred to volume setup,
and final testing stays disabled until its checkpoint policy is agreed.

## Model and execution

Start from the original FLUX.2-klein-base-4B checkpoint with fresh optimizer
state. Train the complete transformer. Keep NVARC/Qwen3-4B and the FLUX VAE
frozen, retaining the established 30×30 canvas, 480×480 rendering and palette.

Use RunPod Community Cloud without a region-bound network volume. Download
data to the pod's local disk at startup. The new runtime image bundles the pinned
initial models under `/opt/models`; older images download them to `/workspace`.
Keep full checkpoints on the Pod;
upload only the trained transformer weights and configuration to Hugging Face.
The external copy supports inference or a fresh optimizer, not exact training
continuation after the pod is deleted. The startup launcher handles data preparation
and weight uploads; writable data, cache and output paths stay on `/workspace`.

Use the [B3 execution path](../performance/b3.md) on one H200: packed FA3 for
NVARC and FLUX, NVARC early exit after block 27 with raw states from 9/18/27,
16-source preparation windows, VAE chunks of 48 and FLUX packs capped at
12,288 joint tokens. NVARC and FLUX compute use BF16; VAE execution, FLUX
parameters, optimizer state and loss reduction use FP32. Keep activation
checkpointing disabled for this run.

### Building the container image

The [Dockerfile](../../Dockerfile) reuses the pinned v1 Python/PyTorch/FA3 runtime
and adds the frozen NVARC checkpoint plus FLUX transformer/VAE weights. A fresh
CUDA final stage copies only the Python runtime and model files. Application
sources and the installed `diffusion-arc` package are absent from the final
image, including its layers. Training code continues to arrive through SSH.
NVARC and FLUX are copied into separate layers to stay below GHCR's
[10 GB per-layer limit](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#troubleshooting).
The [model downloader](../../scripts/cloud/build_image_models.py) records source
versions, file sizes and SHA-256 hashes in `/opt/models/manifest.json`.
The [CPU build launcher](../../scripts/cloud/build_training_image_modal.py) runs
normal Docker inside a temporary [Modal VM Sandbox](https://modal.com/docs/guide/vm-sandboxes).
It requires a local Modal SDK >= 1.4 and an authenticated Modal account, but no
local Docker installation or GitHub Actions workflow. VM Sandboxes are in beta.
On 2026-09-15, the standalone CPU smoke test built and ran a Linux/amd64 Docker
container successfully and terminated the VM afterward. Its logs and run metadata
are in `artifacts/image-builds/20260915T103848369810Z/`.

The full CPU build and upload also succeeded on 2026-09-15, publishing the private
Linux/amd64 image `ghcr.io/joshuakauling/diffusion-arc:b3-20260915-v1`. Its compressed
layers total 6,958,976,445 bytes (6.96 GB), excluding model weights and training
data. Build and upload took about ten minutes. PyTorch/CUDA and FA3 import checks,
`pip check`, and the training CLI help command passed. The source archive,
provenance, immutable RunPod image reference and independently verified registry
manifests are in `artifacts/image-builds/20260915T111356013624Z/`.
On 2026-09-16, the replacement image was published and independently verified:
`ghcr.io/joshuakauling/diffusion-arc:b3-models-20260916-v2`, with immutable reference
`ghcr.io/joshuakauling/diffusion-arc@sha256:8affd0f55f38b26091d32a395174051c7baa729011b3807f7c2b77878b7e2255`.
Its Linux/amd64 layers total **18,862,467,621 compressed bytes (18.86 GB)**;
the largest layer is 6.22 GB. The 16 embedded model assets total 15,186,522,890
uncompressed bytes. Runtime imports, FA3 flags, `pip check`, model manifests and
absence of the application package passed. Registry configuration contains no
credential environment variables. The verified digest, registry responses and
per-file hashes are in `artifacts/image-builds/20260916T110233652959Z/`.
The build VM was terminated after verification. GPU execution of this new image
remains to be checked on RunPod. The launcher follows a remote log file and uses
process status for completion, avoiding dependence on the Modal log stream's EOF.

Start with the standalone Docker test. Adding `--dry-run` inspects the plan
locally without calling Modal:

```bash
.venv/bin/python scripts/cloud/build_training_image_modal.py smoke --dry-run
.venv/bin/python scripts/cloud/build_training_image_modal.py smoke
```

The smoke VM is limited to one physical core, 2 GiB RAM and five minutes. A full
build first repeats that test, then builds and uploads the training image with
four physical cores, 32 GiB RAM and a two-hour VM timeout. These limits cover the
VM lifetime; preparing Modal's small Docker-host image happens before it starts.
CPU and memory limits are explicit, and the VM is terminated in `finally`.

For a full build, create a Modal Secret named `diffusion-arc-registry` containing
`REGISTRY_USERNAME` and `REGISTRY_PASSWORD` in the `diffusion-arc` Modal environment.
Select that environment with `MODAL_ENVIRONMENT=diffusion-arc`; the default `main`
environment has no registry credentials. For GHCR, use a GitHub classic token
with `write:packages` access to the destination namespace. The launcher validates
that the Secret exists before allocating the VM, and supplies it only to the
Docker login command. Choose a fresh image tag for each publication:

```bash
MODAL_ENVIRONMENT=diffusion-arc .venv/bin/python scripts/cloud/build_training_image_modal.py build --image ghcr.io/joshuakauling/diffusion-arc:run-v1 --dry-run
```

Omitting `--dry-run` starts the remote build and upload. The upload context
contains only `Dockerfile`, `.dockerignore` and the model downloader; application
code, data, checkpoints, credentials and local environments are excluded before
transfer. The builder reads the local Hugging Face login and either Kaggle API
environment credentials or `~/.kaggle/kaggle.json`. It supplies them through a
temporary Modal Secret and BuildKit secret mounts only during the model build.
Tokens and download-cache metadata are not copied into the image. Registry login
runs before large transfers; invalid credentials terminate the VM immediately.
Logs, the exact
source archive, file hashes and the published image digest are saved locally
under `artifacts/image-builds/`, alongside the model manifest. The runtime is reused
without recompiling FA3; a registry cache reuses final image layers. New GHCR packages default to private;
RunPod needs separate credentials with `read:packages` access to pull the image.
Use the printed digest reference in its Container Image field. The image keeps
the pod running while data and the resolved run configuration are prepared.
The CPU build checks runtime imports, FA3 build flags, dependency consistency,
model files and absence of the application package. FA3 execution still requires
the launcher's H200 forward/backward preflight before training. Bundling models
moves their transfer into the image pull; a faster fresh-Pod startup has not been
measured and depends on registry throughput and host caching.

### Launching local code on an existing Pod

[`scripts/cloud/launch_runpod.py`](../../scripts/cloud/launch_runpod.py) runs on
the local machine with Python 3.11+, Git and OpenSSH. It uploads the current
working tree and starts [`train_runpod.py`](../../scripts/cloud/train_runpod.py)
inside an existing Pod. Neither script allocates, stops or deletes a Pod.
The published `b3-20260915-v1` image supplies CUDA, PyTorch and compiled FA3.
Python/config edits need no image rebuild or Git commit/push. Rebuild the runtime
when its pinned dependencies or compiled kernels change.

The uploaded source takes precedence over the image's bundled code, including
in preflight and training subprocesses. Startup checks the pinned training
packages and installs the missing `kagglehub==1.0.2` on the v1 image once, with all
installed package versions constrained. Incompatible versions fail explicitly.
The new runtime includes this Kaggle client and carries no application code.
If `/opt/models/manifest.json` exists, the launcher verifies the source pins,
complete inventory, sizes and hashes once per startup, then loads those paths
without model downloads. A corrupt bundle fails explicitly. This check reads
all embedded weights from local disk. The original v1 download path remains
supported. Keep a run on its original model paths when resuming: checkpoint
identity includes those paths, so the new image is intended for fresh runs.

**SSH connection.** Add your local public SSH key to the RunPod account and use
either command shown under **Connect**: basic SSH through `ssh.runpod.io` or
**SSH over exposed TCP**. Use the matching local private key after `-i`.
The launcher detects the proxy and transfers code through its interactive terminal,
then starts the same detached worker and streams its logs. It checks archive and
file hashes exactly as with direct SSH. The proxy needs no public IP, exposed port,
or SSH server inside the image. Source input is transferred with terminal echo
disabled and raw mode enabled to avoid terminal line limits. A disconnected proxy
is reported as an error; uploads and job starts are never retried automatically.
The transport accepts both LF and the CRLF output added by RunPod's proxy, including
during bootstrap. Timeouts identify which initialization phase failed.

**Optional direct SSH setup.** For direct SSH, select a Pod with a public IP and
expose TCP port 22. RunPod's basic proxy does not provide SCP/SFTP; see
[RunPod's SSH guide](https://docs.runpod.io/pods/configuration/use-ssh).
The published custom image has no SSH server. To enable direct SSH in its Pod terminal,
set `PUBLIC_KEY` to your public key if RunPod has not provided it, then run:

```bash
set -e
: "${PUBLIC_KEY:?Set PUBLIC_KEY to your local public SSH key first}"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server
install -d -m 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
install -d -m 755 /run/sshd
ssh-keygen -A
/usr/sbin/sshd -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o PermitRootLogin=prohibit-password
```

Restart the SSH daemon after a Pod restart, or include the setup in its template
start command before `sleep infinity`. An encrypted local key must be unlocked
in `ssh-agent` first; the launcher uses noninteractive key authentication.
It accepts new host keys and rejects changed ones.

Provide these environment secrets in RunPod; registry pull credentials remain
separate in Container Registry Auth:

| Secret | Purpose |
| --- | --- |
| `HF_TOKEN` | Write the output model repository; read FLUX on images without embedded models |
| `WANDB_API_KEY` | Authenticate the configured W&B account |
| `KAGGLE_API_TOKEN` | Read the private `joshuakauling/nvarc-data` dataset; read NVARC on older images |

Missing, blank or unresolved secret references fail together before SDK imports,
workspace writes or downloads. Tokens are not written into configuration or
startup receipts. Subsequent checks verify HF access, W&B authentication, free
workspace disk space and one H200, including the existing small FA3 BF16 varlen
forward/backward preflight. A small dataset-key download checks Kaggle access;
images without embedded weights also check model access before large transfers.
Fresh-run disk headroom is 190 GiB; resume budgets only future
checkpoint saves plus 25 GiB for downloads and preparation.

The upload destination defaults to `<HF username>/diffusion-arc-weights` and
the W&B project to `diffusion-arc`. Override them with `HF_OUTPUT_REPO`,
`WANDB_PROJECT` and `WANDB_ENTITY`, or the corresponding CLI arguments.
The destination HF model repository can be public or private; the launcher
preserves its visibility and creates missing repositories as private. It checks
write access before training. Existing repositories are checked without calling the
creation endpoint. Transient Hub server/network failures get at most three
attempts, with waits of 2 and 4 seconds; access-denied and public-repository
errors fail immediately. Write access is still required before preparation.

Tasks come directly from the private Kaggle dataset
[`joshuakauling/nvarc-data`](https://www.kaggle.com/datasets/joshuakauling/nvarc-data),
pinned to **version 1**. An authenticated metadata check on 2026-09-15 confirmed
that version and its extracted `nvarc-all-classified/` task tree. The launcher
downloads the dataset once into `/workspace/data/nvarc_data/kaggle`; Kaggle's
completion marker allows later starts to reuse it. No HF data repository or
separate data-publication command is needed.

The original 47,337-task index and official ARC-AGI-1 training keys must match
their existing SHA-256 values. The existing subset builder derives the 14,490-task
training-only index and verifies its exact hash. The bundle builder reads only
those indexed files from the extracted tree and checks every task's content hash
and pair count before sampling five pairs with uniform D4/color augmentation.
The historical `data.source_archive` config path accepts either a `.tar.gz`
archive or its extracted directory. Directory manifests identify the input through
the task index's per-file hashes; archive manifests additionally hash the archive.
Both readers produce the same converted tasks and augmented keys in CPU tests.

Validation still freezes 100 tasks from the pinned official ARC archive.
Complete bundles are reused after their checksums pass; mismatching or incomplete
bundles fail instead of being replaced. Source versions and metadata hashes are
recorded in `startup.json` alongside the source-code identity.

From the local repository, copy either of the Pod's SSH commands into `--ssh` and
first prepare assets without starting the epoch:

```bash
python3 scripts/cloud/launch_runpod.py run --ssh "ssh root@POD_IP -p PORT -i ~/.ssh/id_ed25519" --run-name arc1-epoch1 --prepare-only
```

For basic SSH, use `--ssh "ssh POD_USER@ssh.runpod.io -i ~/.ssh/id_rsa"`
with the user shown in RunPod's Connect panel. The Pod must already be running.

Remove `--prepare-only` to start training. Add `--dry-run` to inspect the upload
plan without SSH, downloads or file writes. The first remote check requires the
Pod secrets and mounted workspace before transferring source. Secrets stay on
the Pod; selected environment variables are recovered from the container's
initial environment when the SSH login omits them.

Each invocation captures current `.py` files under `src/` and `scripts/`, JSON
configs, `pyproject.toml` and `README.md`, including modified and untracked files.
Ignored files, `.env`, local environments, datasets and weights are excluded.
File hashes, commit identity and a dirty-working-tree flag accompany the archive.
Local receipts are saved under `artifacts/runpod-launches/<job-id>/`; verified
code copies live under `/workspace/code/<archive-sha256>/`. A later upload leaves
earlier copies intact. The run's `startup.json` records the source manifest.

Training runs in a detached process. The command follows its combined stdout and
stderr and returns its exit code; `--detach` returns immediately after launch.
Closing the terminal leaves the job running. Its printed job ID identifies the
log and status files under `/workspace/jobs/<job-id>/`. Reattach with:

```bash
python3 scripts/cloud/launch_runpod.py logs --ssh "ssh root@POD_IP -p PORT -i ~/.ssh/id_ed25519" --job-id JOB_ID
```

NVARC uses Kaggle model version 1; FLUX uses the pinned Base-4B revision and
downloads only `transformer/` and `vae/`. SDK caches live on `/workspace`.
The launcher freezes resolved paths in `runs/<run-name>/training.json` and
source revisions/hashes in `startup.json`, then invokes the existing training
CLI. Trainer, event cadence and scientific recipe are unchanged.

After successful training, each retained checkpoint contributes only its
`transformer/config.json`, Safetensors and optional shard index to HF, under
`<wandb-run-id>/step-XXXXXX/transformer/`. Full optimizer/RNG state remains on
the Pod. Successful uploads receive local commit receipts under `uploads/`.
An upload failure preserves all local checkpoints. Retry from the local launcher
with `upload` instead of `run`, keeping the same `--ssh` and `--run-name`.
This command requires only `HF_TOKEN` and reuses the saved upload destination.

To continue interrupted training, use the same run name and `--resume` with the
latest complete local checkpoint path. The existing W&B run and its project/entity
must match. A completed epoch requires `upload`, not another training invocation.
Once training has been attempted, a fresh run requires another run name. A workspace
lock prevents concurrent launchers from modifying the same assets or checkpoints.
The Pod remains running after completion or failure; manage its lifecycle in RunPod.

## Data and batching

| Setting | Decision |
| --- | --- |
| Training pool | 14,490 synthetic NVARC tasks whose two parents are in official ARC-AGI-1 training |
| Duration | One complete epoch, one sampled sequence per synthetic task |
| Sequence | Five distinct pairs sampled without replacement, in random order |
| Supervised targets | Outputs of pairs 3, 4 and 5, conditioned on 2, 3 and 4 preceding demonstration pairs |
| Optimizer batch | 16 sequences × 3 targets = 48 targets |
| NVARC call budget | 134,400 real tokens (8,400 × 16); split whole sources for smaller budgets |
| Final batch | 10 sequences = 30 targets; retain this batch |
| Total | 43,470 target exposures and 906 optimizer updates |
| Seed | 167; preserve task order and augmentation/RNG state for resume |

Shuffle tasks reproducibly. For each sequence, sample one permutation of all
10 ARC colors and one of the eight D4 symmetries uniformly, including identity.
Apply the same color mapping and geometry to every input/output pair in that
sequence. Random pair selection already determines pair order. Geometry is
enabled from the first epoch because synthetic tasks can share parent tasks;
it does not multiply the number of training sequences.

Use the `nvarc_train_arc_agi_1_training_only` pool, recorded in
`data/nvarc_data/nvarc_train_arc_agi_1_training_only_tasks.jsonl`.
Validation/test tasks and their derived augmentations must remain outside
training. Do not introduce the VAE-safety filtering used in the tiny overfit
experiment into this full-pool run.

The [epoch preparation command](../../scripts/data/prepare_training_data.py)
reuses the benchmark's archive checks, task conversion and augmentation-key
generation. Run it against the locally available archive and index:

```bash
PYTHONPATH=src .venv/bin/python scripts/data/prepare_training_data.py --config configs/training/arc_agi_1_epoch.json
```

It requires the complete indexed pool, checks both parents against the official
training IDs, verifies source checksums and publishes a new bundle only after
all tasks and keys validate. Existing bundles are never overwritten. The split
manifest records source/config/code hashes, task order, augmentation identities
and expected batch counts. Generated data under `data/training/` is ignored by Git.
Configuration loading and preparation require neither model loading nor W&B
credentials; tokenization still enforces the prefix limit when the dataset is read.

The prepared key order already contains the seeded shuffle, so the shared B3
DataLoader uses `shuffle=False`. Sampled split views explicitly opt into
`geometry_sampling: "uniform"`; historical views without this option retain
their original geometry cycle and random pair/color choices.

## Loss and flow times

For each target, construct `z_t = (1 - t) * z_0 + t * epsilon` with independent
Gaussian noise and velocity target `epsilon - z_0`. Compute squared velocity
error in FP32 and average over latent channels before the spatial reductions:

```text
target_loss = 0.1 * mean_canvas + 0.8 * mean_grid + 0.1 * mean_boundary
```

The canvas term covers all 900 tokens. The grid term covers the true top-left
output grid. The boundary is the one-cell L immediately below/right of that
grid, including its corner and clipped to the canvas. For grid height `h` and
width `w`, it has `min(h + 1, 30) * min(w + 1, 30) - h * w` tokens.
When the boundary is absent at 30×30, omit its term and divide the remaining
sum by 0.9, giving ordinary mean squared error.

Each target receives the same total coefficient mass, regardless of grid size.
Average over the actual optimizer target count, including the final 30-target
batch. Weight physical microbatch contributions by their share of this count.
There is no additional area-based target weight.

In every full optimizer batch, randomly choose **4 of 48 targets for exact
`t=1`** and draw the remaining 44 times independently from `Uniform(0, 1)`.
Assign time/noise in logical target order before packing, independently of grid
size and target position. The final batch uses 3 endpoints and 27 uniform
draws: 3,623 endpoint exposures across the epoch. Endpoint targets receive no
additional loss multiplier.

Training and loss validation share the
[regional reduction](../../src/diffusion_arc/training/step/loss.py), including
the missing-boundary rule. Prepared targets retain grid dimensions only for
this loss; the transformer receives no target-shape hint. The loop assigns
times and noise to the logical optimizer batch before deciding its physical
packs. Each step returns CPU-only target metrics containing regional means,
weighted contributions, times and demonstration counts for later W&B aggregation.
Targets keep their logical order; preparation and packing diagnostics are separate.

## Optimizer and continuation

| Setting | Decision |
| --- | --- |
| Optimizer | AdamW, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01 |
| Learning rate | 1e-5 |
| Warmup | Linear over 50 optimizer updates: 2e-7 on update 1, 1e-5 on update 50 |
| Subsequent schedule | Constant learning rate |
| Gradient clipping | Global norm 1.0 after accumulation, before the optimizer update |

Continuation keeps optimizer, scheduler, RNG, data position and global update
count; it does not restart warmup. A later cosine cooldown can be a separate
experiment, but is not part of this first epoch.

The [warmup scheduler](../../src/diffusion_arc/training/step/lr_schedule.py) advances
after each optimizer update. Step metrics capture the learning rate used before
that advance. A CPU save/reload test verifies exact continuation of random draws,
losses, gradients, learning rates, AdamW moments and weights through HF checkpoints.
GPU numerical agreement remains part of the integration smoke run.

### Third epoch from exported weights

[`arc_agi_1_epoch3_high_noise.json`](../../configs/training/arc_agi_1_epoch3_high_noise.json)
starts from `flipchip167/diffusion-arc-weights`, subfolder
`m7y2mfn2/step-001812/transformer`, pinned to revision
`1126da2fc1a43c210d102f75d385f17279003460`. It creates fresh AdamW state and
repeats the 50-update warmup to `1e-5`. Global progress continues from 1812;
the scheduler counts updates since this optimizer was created. Full checkpoints
preserve both counters for exact continuation within the new stage.

Each full batch randomly assigns 24 targets to `t=1`, 12 to uniform `(0.8,1)`
and 12 to uniform `(0,1)`. The final 30-target batch uses 15/8/7. There is no
importance correction. Seed 169 generates a new view of the same training
pool; validation tasks, noise seeds, loss aggregation and sampling remain fixed.
Training metrics split actual times into `t1`, `high_noise` (`0.8<t<1`) and
`lower_noise` (`t<=0.8`); high-noise counts include draws from the broad uniform
component. Validation losses remain directly comparable to earlier epochs.

[`finetune_runpod.py`](../../scripts/cloud/finetune_runpod.py) accepts `--ssh`
and `--run-name` from a local terminal, plus `--dry-run`, `--prepare-only` and
`--resume` as in the existing launcher. It verifies the completed parent W&B
run and credentials before downloading. By default it resumes that run and
records the new recipe under `epoch_3`. Model weights are uploaded after the
epoch; full checkpoints stay on the Pod. Checkpoints occur at global steps
2112, 2412 and 2718. The launcher leaves the Pod running.

## Validation and final test

Use the existing official ARC-AGI-1 v1.0.2 evaluation tasks in
`data/benchmarks/arc-agi-1/evaluation`. Shuffle the 400 sorted task IDs with
Python `Random(167)`: the first 100 are validation and the remaining 300 are
reserved for the final test. The first 32 validation tasks form the fixed
sampling panel. The frozen identities and payload hashes are in the
[split manifest](../../data/benchmarks/arc_agi_1_validation/manifest.json).
Select recipes and checkpoints without using the reserved test tasks.

| Check | Workload | Update numbers in the first epoch |
| --- | --- | --- |
| Loss | All 100 tasks / 104 queries at t = 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 | 0, 100, 200, …, 800, 906 |
| Direct sampling | Fixed 32 tasks / 33 queries, two independent samples with 40 linear Euler steps | 0, 300, 600, 906 |

The agreed intervals are **100 updates for loss and 300 for sampling**, plus
the initial and final checks. The final check at 906 replaces the scheduled
check at 900. Use all original demonstrations and all query inputs, without
augmentation, voting, target-shape hints or test-time training.

Fix validation noise by query identity and time so checkpoints see the same
probes. Report the weighted loss and its regions separately at every time.
The combined diagnostic is `11/12 * mean_inner_times + 1/12 * loss_at_t1`;
the five inner points approximate the uniform-time expectation. Use separate
fixed seeds for generation. Keep Gaussian initial states and Euler integration
in FP32, then apply VAE decoding and strict PAD-boundary grid decoding.

Exact-grid pass@1 and pass@2 are primary; shape and cell accuracy are diagnostics.
Shape accuracy and invalid-grid rate use all 66 predictions as their denominator.
Cell accuracy is the equally weighted mean of per-prediction cell accuracies
among shape-correct predictions, regardless of grid area. It is omitted when
no predicted shape matches; the shape-correct count remains zero.
For tasks with multiple queries, one shared sample index must solve all queries
for a task-level success. NVARC conditioning never receives the query outputs.
The released NVARC checkpoint has known public-benchmark contamination. This
split isolates FLUX tuning and final comparison; it does not establish unseen
generalization for the entire system. See the
[baseline](../experiments/baselines/nvarc-arc-agi-1-direct-sampling.md).

Cache frozen validation conditioning and VAE target latents once on CPU, staging
one packed group to the GPU at a time. A two-repetition H200 timing check measured
**32.5 s per loss check** and **115.4 s per sampling check**, including transfers,
packing, decoding and metrics. Cache preparation took 7.9 s once and used 2.54 GiB
of CPU memory. At the agreed cadence, validation adds about **13.1 minutes** to
the B3-derived training estimate of 138 minutes: approximately **2 h 31 min**
before model setup, checkpoint writes, image logging and the final test.
These are runtime estimates, not evidence of learning quality. Reproducibility
details remain in the [timing artifact](../../artifacts/benchmarks/validation/20260911T112135652096Z/measurement/report.json).

### Checkpoint comparison: 4530 and 4830 (2026-09-22)

Checkpoint **4830 with four Euler steps** achieved the highest task accuracy
in this comparison: **8/100 pass@1 and 9/100 pass@2**. At 50 steps, checkpoint
4530 performed better than 4830. Checkpoint choice therefore depends on the
sampling budget; these results do not support a blanket conclusion that the
additional training made the model worse.

The [experiment configuration](../../configs/diagnostics/sampling_checkpoints_4530_4830.json)
pins both HF exports and the shared setup. Checkpoint 4530 is the end of epoch
5; 4830 is a saved state partway through epoch 6. Both were evaluated on RunPod
H200 with the same 100 validation tasks, 104 queries, seeds 20167/20168, frozen
NVARC/VAE components and linear Euler schedules. Every trajectory starts at
**t = 1, pure noise**, with no target-derived initialization, augmentation or
voting. The 300 reserved final-test tasks were not used.

Each variant produced 208 query outputs. Pass@1 uses the first fixed seed;
pass@2 accepts either seed, requiring all queries of a task to be correct under
the same seed. Shape and invalid-grid rates use all 208 outputs.

| Checkpoint | Euler steps | Pass@1 | Pass@2 | Correct shape | Invalid grids |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4530 | 50 | 5/100 | 6/100 | 64.9% | 1.0% |
| 4530 | 4 | 3/100 | 6/100 | 65.4% | 13.0% |
| 4830 | 50 | 1/100 | 2/100 | 65.4% | 8.7% |
| 4830 | 4 | 8/100 | 9/100 | 74.5% | 15.4% |

Both runs completed, with 936 unique records each including controls. All 520
control records match across checkpoints, and initial noise matches across
step counts. Total wall time was **24.06 minutes**, including **19.89 minutes**
after model loading. The [comparison report](../../artifacts/diagnostics/checkpoints-4530-4830-v2/report.json)
links the per-checkpoint reports; those contain environment, source identity,
resolved revisions, query identities and timings. Sibling `records.jsonl` files
and `comparisons/` images preserve the individual results. These are local
experiment artifacts, not files intended for Git.

Measured findings and limits:

- **Progress from epoch 4:** at four steps, pass@2 increased from 0/100 at
  checkpoint 3624 to 6/100 at 4530 and 9/100 at 4830. The
  [epoch-4 diagnostic](../../artifacts/diagnostics/arc1-epoch4-steps/report.json)
  used the same queries, seeds and frozen components. The last increase is
  only three tasks, so it does not establish a reliable continuing trend.
- **Sampling sensitivity:** at 4830, four steps solve both tasks solved at 50
  steps plus seven additional tasks. On the 133 query/seed outputs where both
  schedules produce the correct shape, mean cell accuracy improves from
  81.4% to 85.7%. For `e7639916`, query 0, seed 20167, four steps are exact;
  50 steps produce 16 wrong cells. The mechanism behind this step-count
  dependence remains a hypothesis, not a demonstrated integration defect.
- **Padding and content are separate failure modes:** 4830/four steps has 32
  invalid outputs, all classified as `pad_inside_extent`, and 21 valid outputs
  of the wrong shape. Six incorrect outputs match every cell within the true
  target extent but contain additional non-PAD cells outside it. This is a
  reference-based diagnostic only; no cropping or repairs enter the scores.
  Another 138 outputs have the correct shape but wrong content. These counts
  motivate inspecting padding, but do not establish that increasing its loss
  weight would fix the main accuracy limitation.
- **Representation control:** all 104 clean target VAE roundtrips are exact.
  This rules out unavoidable clean-target decoding errors on this set, but
  does not establish that generated latents decode reliably.

The proposed working reference is **4830/four steps**. Before another long
training run, compare **two and eight steps** on the same checkpoint and fixed
inputs, reusing the four-step result. For future training, assess checkpoint
quality through direct sampling at a small step budget as well as validation
loss; occasional longer sampling can remain a control. The existing 40-step
monitor may miss improvements at small budgets, but this experiment did not
test 40 steps directly. These are follow-up proposals: the training recipe and
monitoring cadence have not been changed. Selection on these 100 validation
tasks is not a final-test result, and the NVARC exposure limitation above still
applies.

## Logging and checkpoints

Use Weights & Biases for experiment logging. Associate training metrics,
validation results, generated samples and checkpoints with the global optimizer
update count. Keep individual optimizer-step observations; do not replace them
with 50-step averages in the training process.

### Agreed scalar logging

| Cadence | Measurements |
| --- | --- |
| Every optimizer update | Total weighted loss; unweighted per-target canvas, grid and boundary MSE means; actual weighted contributions of these three terms |
| Every optimizer update | Weighted target-loss means and target counts for exact `t=1`, uniform-time draws, and each of 2/3/4 preceding demonstrations |
| Every optimizer update | Learning rate used for the update, global gradient norm before clipping, clipping indicator |
| Every optimizer update | Actual and cumulative target counts, epoch progress, full training-step duration and targets per second |
| Every optimizer update | NVARC processed tokens, FLUX text/image/joint tokens, and actual FLUX forward-call count; retain per-update and cumulative counts |
| Every loss validation | Weighted loss and regional means at each of the six fixed times, the inner-five-time mean, exact `t=1` loss, the combined diagnostic, and check duration |

Region means average targets equally after reducing channels and the region's
tokens within each target. Exclude targets without a boundary from the raw
boundary mean and record its valid-target count. Their boundary contribution
to the weighted objective is zero; use the actual renormalized coefficients
for the remaining terms. Averaging weighted contributions across all targets
must recover the logged total loss.

Keep subgroup counts so later window averages can be reconstructed as
`sum(step_mean * step_count) / sum(step_count)`, including the smaller final
batch. Dashboard smoothing is a visualization choice, not a substitute for
that target-weighted reduction. The separate subgroup curves do not retain
individual target records or arbitrary intersections of time and context groups.

NVARC tokens count positions actually encoded in each source preparation call.
FLUX text tokens sum the conditioning-prefix lengths of consumed targets,
including overlapping prefixes reused across targets. Image tokens are
`900 * target_count`, and joint tokens are text plus image tokens. These count
forward input positions, not FLOPs or backward work. FLUX calls count executed
packed forwards, not targets or optimizer updates. Keep training, validation
and sampling counters separate; evaluation counts include each repeated time
or Euler-step call. Reuse the existing preparation and executed-pack metadata.

The inner-five-time validation mean is a fixed-probe approximation to the
uniform-time expectation; label it accordingly. It is not a new random-time
validation pass.

### Agreed metric names

The names below are fixed across tasks and runs. Braces abbreviate a finite
set of separate keys; they are not literal characters in a logged name.

| Measurements | Keys |
| --- | --- |
| Training objective and regions | `train/loss`, `train/mse_{canvas,grid,boundary}`, `train/contribution_{canvas,grid,boundary}` |
| Training subgroups | `train/loss_t1`, `train/loss_uniform`, `train/loss_demos_{2,3,4}` |
| Subgroup denominators | `train/targets_t1`, `train/targets_uniform`, `train/targets_demos_{2,3,4}` |
| Optimizer | `train/learning_rate`, `train/grad_norm`, `train/grad_clipped` |
| Training timing | `train/step_seconds`, `train/targets_per_second` |
| Training workload | `train/targets`, `train/nvarc_tokens`, `train/flux_text_tokens`, `train/flux_image_tokens`, `train/flux_joint_tokens`, `train/flux_forward_calls` |
| Progress | `progress/optimizer_step`, `progress/epoch_fraction`, `progress/targets_seen` |
| Validation objective | `val/loss`, `val/loss_uniform_proxy`, `val/loss_t1`, `val/loss_t_{0_1,0_3,0_5,0_7,0_9}`, `val/seconds` |
| Sampling accuracy | `sample/pass_at_1`, `sample/pass_at_2`, `sample/solved_tasks_at_1`, `sample/solved_tasks_at_2`, `sample/shape_accuracy`, `sample/cell_accuracy`, `sample/invalid_grid_rate` |
| Sampling denominators and duration | `sample/task_count`, `sample/shape_correct_count`, `sample/seconds` |
| Checkpoint records | `checkpoint/seconds`, `checkpoint/size_bytes`, `checkpoint/path` |

Append `_total` to cumulative token and forward-call counters. Validation and
sampling reuse the workload names under `val/` and `sample/`. Regional
validation means follow the time-specific names, for example
`val/mse_grid_t_0_5` and `val/mse_grid_t1`. `val/loss` is the combined diagnostic.
Counts and rates retain the reduction rules above; the checkpoint path is a
string record rather than a scalar curve.

### Agreed sample presentation

Log a W&B table under `sample/predictions` at each sampling check: updates
**0, 300, 600 and 906**. Each table contains one row per query, giving 33 rows
for the fixed 32-task validation panel. Associate each table version with the
optimizer update and model state used for sampling.

| Column | Content |
| --- | --- |
| `task_id` | Original ARC task identifier |
| `query_index` | Zero-based query index within the task |
| `comparison` | One labeled comparison image showing input, expected output, sample 1 and sample 2 |
| `sample_1_exact` | Whether sample 1 exactly solves this query |
| `sample_2_exact` | Whether sample 2 exactly solves this query |

Show both classified 30×30 canvases in full using the model's target palette,
so incorrect dimensions and padding remain visible. These are discrete class
predictions from the VAE decode, not its raw RGB output. Do not crop generated
canvases to the expected output shape. Input and reference grids use the same
cell scale and are padded to 30×30 for comparison.
The row flags describe query-level correctness; aggregate task-level pass@1
and pass@2 keep the multi-query rules defined above.

Build the 33 comparison images from the samples already generated for scoring.
This adds image preparation and upload, with no additional FLUX forwards.
Each check creates a fresh immutable table. `sample/seconds` includes generation,
VAE decoding and scoring; it excludes image rendering and W&B serialization/upload.
Sampling reuses the validation conditioning cache, so `sample/nvarc_tokens` is
zero; the one-time encoder work is counted under validation.

### Agreed hardware logging policy

Disable W&B's automatic system-metric collection and collect only the listed
measurements at the agreed intervals below.
The implementation pins W&B 0.30.0 and explicitly sets `x_disable_stats`;
verify the disabled collector in the integrated smoke test. W&B documents it in its
[system-metrics guidance](https://docs.wandb.ai/support/models/articles/how-can-i-disable-logging-of-system-metr).

| Cadence | Measurements | Keys |
| --- | --- | --- |
| Every optimizer update | PyTorch GPU allocated bytes, reserved bytes, and peak allocated bytes during the complete training step | `system/gpu_allocated_bytes`, `system/gpu_reserved_bytes`, `system/gpu_peak_allocated_bytes` |
| Every 30 seconds | Active GPU utilization (%), power draw (W), and temperature (C) | `system/gpu_utilization_percent`, `system/gpu_power_watts`, `system/gpu_temperature_celsius` |
| Every 30 seconds | Training-process CPU utilization and resident RAM bytes | `system/cpu_utilization_percent`, `system/ram_rss_bytes` |

Record system samples with `system/elapsed_seconds` and `system/phase`. Keep
training memory peaks separate from validation, sampling and checkpoint work.
Reset/read the peak counters at explicit phase boundaries. CPU utilization
uses 100% per fully occupied CPU core and can exceed 100%. It describes the
training process since the previous CPU read, which may span phases; GPU
utilization is the device's recent NVML sample, not a 30-second average.

The runner reads training memory immediately after `Trainer.step`, before any
validation or sampling. Other sensors are read at the first training-step
boundary at least 30 seconds after their previous sample; there is no background
hardware thread. Long steps or evaluation blocks can therefore extend the
interval. Due hardware readings join the same W&B call as the update's metrics,
sent after the run-service blocks. Use fixed metric names, without task IDs in keys.
For example, `train/loss` and `train/loss_demos_2` are shared time series; task
identities belong in sample tables or diagnostic records, not separate metric
names such as `train/loss_<task_id>`.
Do not enable parameter/gradient/activation histograms, model graph watching,
profiler traces, or automatic console capture. Keep profiling disabled throughout
this training run. Preserve diagnostic console logs in the run artifacts.
Collect the small loss reductions together
with the existing loss calculation, without additional model forwards or a
separate GPU synchronization per scalar. Verify logging overhead against the
same workload without logging before the full epoch.

### Checkpoint cadence

The prepared Modal continuation uses
`configs/training/arc_agi_1_epoch5_high_noise.json` and
`scripts/cloud/train_modal.py` for epochs 5 and 6. Initialization is pinned to
`flipchip167/diffusion-arc-weights`, revision
`a0080ba91538ab001fe958d2632a2f6a0cf71fb5`, subfolder
`m7y2mfn2/step-003624/transformer`. This weights-only export requires fresh AdamW
and the existing 50-update warmup at epoch 5; epoch 6 restores the full local
checkpoint, including optimizer, scheduler and RNG state. Augmentation seeds
are 171 and 172, with global final steps 4530 and 5436. The high-noise mixture,
loss, learning rate, 40-step training-time sampling and all event intervals stay
unchanged. W&B continues `m7y2mfn2` after checking its existing history.

The Modal launcher requires an explicit profile, expected workspace and
environment. `--dry-run` performs no account lookup or remote work. Actual
execution verifies the account and named Secret before building the shared FA3
runtime. A CPU Sandbox downloads the transformer/VAE from HF and the pinned
NVARC/data from Kaggle before the H200 is allocated. Preparation is bounded to
2 hours; the single H200 Sandbox is bounded to 8 hours including both epochs and
their weights-only HF uploads. Each remote entrypoint exits on completion or
failure, terminating its Sandbox independently of the local terminal. Keep the
terminal connected through CPU preparation so it can launch the GPU stage.

Full checkpoints, caches, logs and status receipts stay on the named v2 Modal
Volume; each stage explicitly syncs it before exiting. Nothing is automatically
deleted. The six saves have a conservative 355 GiB checkpoint budget including
headroom, plus models/data. Modal Volumes grow on demand and report placeholder
free-space values, so the RunPod disk-capacity check does not apply there.
The continuation subsequently ran in the `joshuakauling` Modal workspace.
Epoch 5 completed at step 4530; epoch 6 was stopped with checkpoint 4830 retained.
Both transformer exports are on Hugging Face and were evaluated in the
[checkpoint comparison](#checkpoint-comparison-4530-and-4830-2026-09-22) above.

Save checkpoints every 300 optimizer updates, aligned with direct sampling,
and at the end of the epoch. Following the final-validation convention above,
the first epoch saves at updates **300, 600 and 906**. Sampling results and the
corresponding checkpoint must describe the same model state. Save full-state
checkpoints to the pod's local disk on RunPod, with the output
directory supplied by configuration. Retention must be configurable; the
discussed allowance of approximately 100 checkpoints is provisional. The first
epoch produces only three scheduled checkpoints, all of which are retained.
Disk capacity, startup downloads and external weight uploads are part of the
later RunPod setup.

`training/run/checkpointing.py` writes under
`<output_directory>/checkpoints/<wandb_run_id>/step-000300/`:

- `transformer/`: Diffusers `save_pretrained` configuration and FP32 safetensors,
  split into shards of at most 5 GB except any larger indivisible tensor.
- `training_state.pt`: AdamW, scheduler, dedicated flow generator and Python,
  NumPy, Torch CPU and selected-device CUDA RNG states. Loading uses
  `torch.load(..., weights_only=True, map_location="cpu")`.
- `checkpoint.json`: completed update/source/target counts, workload totals,
  resolved recipe, ordered training-content fingerprint, validation identity,
  W&B run identity, package versions, source hash and file-size inventory.

The frozen NVARC/VAE weights and validation cache are not copied. Saves use a
temporary sibling directory, flush files and publish by directory rename. Failed
writes leave earlier checkpoints intact. Existing checkpoint paths are never
overwritten. `keep_last: null` keeps all; a positive count removes only completed
checkpoints belonging to this W&B run. File sizes detect incomplete/truncated
writes; the inventory is not a checksum of every saved weight byte.

To continue, set `models.resume_from` to the completed checkpoint directory in
the same recipe. The loader resolves it relative to the config file. Resume
requires the same training recipe, ordered data content and validation split;
model/data paths and frozen model files must stay unchanged. Deployment output
paths and retention can change. Ordinary resume stays within the saved epoch.

After a completed epoch, `scripts/cloud/continue_runpod.py` starts one additional
epoch from the last full local checkpoint. With `RUNPOD_SSH` set to the current
Pod's working SSH command, launch from the local repository:

```bash
.venv/bin/python scripts/cloud/continue_runpod.py \
  --ssh "$RUNPOD_SSH" --from-run arc1-epoch1 --run-name arc1-epoch2
```

The new name selects a separate local output directory; W&B retains its existing
run ID. AdamW moments, LR schedule, RNG state and global counters continue from
the checkpoint. The default augmentation seed is the previous seed plus one;
`--augmentation-seed` selects another seed. The same task pool receives newly
sampled pair selection/order, colors and D4 transforms; validation stays fixed.
Epoch two uses global steps 907–1812, with checkpoints at 1206, 1506 and 1812.
Use the same continuation command with `--resume <full-checkpoint-path>` to
resume an interrupted additional epoch. `--prepare-only` checks prerequisites
and prepares the fresh view without training. A failed final HF upload can be
retried with the existing `launch_runpod.py upload` command and the new local
run name.

Continuation requires the original workspace assets and full checkpoint, not
the weights-only HF export. All existing checkpoints are retained. The current
three-save policy requires 190 GiB of free workspace headroom before starting
another complete epoch; a 300 GB volume holding the first epoch's checkpoints
will generally need more space or an explicit cleanup decision. No remote GPU
continuation has been validated yet; local tests cover fresh views, global and
epoch-local cursors, unchanged first-epoch behavior, and real AdamW/RNG/LR
checkpoint roundtrips.

The runner loads FLUX from `transformer/`, reconstructs the frozen CPU cache,
then restores optimizer/scheduler and RNGs before the next update. The data
cursor skips consumed sources before tokenization, independent of prefetched
worker samples. The step-zero checks and warmup do not repeat. Rebuilt NVARC
cache tokens are counted once at the next validation check.

W&B uses the saved entity/project/run ID with `resume="must"`. If its history
extends past the checkpoint, training recomputes those updates and suppresses
their duplicate log records until it catches up; existing history is preserved.
Use the latest completed checkpoint for recovery. A checkpoint and a W&B upload
are separate writes: records absent from the server at interruption are not
reconstructed. W&B server resume and GPU/storage overhead still need the remote
integration test; local tests substitute the resumed SDK state.

At roughly 4 billion parameters, FP32 weights plus two FP32 AdamW moments imply
about **48 GB per full checkpoint**, or **144 GB for three**, before small metadata
overhead. This is a capacity estimate, not a measured file size; a save also needs
space for its complete temporary directory before retention can remove older saves.

## Remaining run setup

Choose the checkpoint-selection rule before using the reserved test tasks.
The proposed first-epoch protocol is to evaluate the final checkpoint at update
906 on all 300 reserved tasks, using two direct samples and 40 Euler steps per
query. Keep final testing a separate explicit action after training; this
proposal still needs agreement.

Implementation work:

1. The validated configuration, deterministic data preparation and recipe core
   are implemented. CPU numerical tests cover regional loss and gradients,
   exact endpoint quotas, unchanged random assignments across pack budgets,
   48-target batches plus a 30-target tail, warmup and gradient clipping.
   Step results contain update scalars and individual CPU-only target metrics;
   logging derives grouped losses outside the trainer. Gradients are released
   before control returns to the outer loop. These tests use small substitute models;
   GPU execution and tuning of the NVARC budget remain unverified.
2. The normal training entrypoint and epoch runner are implemented, reusing B3
   model preparation and packing without starting a profiler. CPU orchestration
   tests cover complete and partial batches, incompatible prepared views,
   early exhaustion and propagation of failed steps. Historical profiling
   remains under `scripts/profile_single_gpu.py` with baseline configs.
   Loss validation now uses the same bundle checks, tokenization and CPU-cache
   preparation as `scripts/benchmark_validation.py`. Preparation respects the
   configured NVARC token budget and VAE image batch size. The runner checks the
   expected 100 tasks / 104 queries before loading models and reuses the cache
   for all six time probes at each scheduled check. Numerical tests verify mode,
   gradient and RNG preservation. Token and forward counters include all six
   probes; NVARC preparation tokens appear only at the first validation event.
   `val/seconds` measures the recurring check, excluding one-time preparation.
3. W&B training scalars and the bounded hardware collector are implemented in
   `training/run/logging.py`. Tests cover target/group reductions, absent boundaries,
   partial batches, selected sensors, intervals, GPU identity, one log call per
   update and failure propagation. The SDK receives the resolved recipe without
   credentials; automatic system/code/console capture is disabled. Before a run,
   supply `logging.project`, `deployment.output_directory` and `WANDB_API_KEY`;
   `logging.entity` can select the owner. Validation losses use the recipe's
   regional weights and endpoint/uniform mixture, with raw regional means for
   each time and separate validation workload totals. Initial validation logs
   at step zero; subsequent checks join that update's single metrics record.
   Fixed sampling now reuses the same cache for 32 tasks / 33 queries and calls
   `inference.generate_packed_arc_samples` through the shared scoring wrapper.
   Generation receives conditioning only; expected outputs enter scoring and
   the comparison table afterward. Tests cover full invalid canvases, task-level
   multi-query success, metric denominators, event cadence and failure propagation.
   A local offline check with W&B 0.30.0 wrote exactly two history records and
   no automatic system-stat records. A further offline media check preserved
   two distinct table versions with 33 query rows each and their comparison
   images. All inputs were synthetic. GPU reads were substituted; real GPU
   measurements and logging overhead still need the integrated smoke run.
4. Full-state save/resume is implemented and tested with real small HF
   safetensors. Tests cover numerical continuation, rejected incompatible or
   incomplete saves, failed writes, retention, data cursors and W&B history
   catch-up. No remote run or full-size checkpoint has been created locally.
5. Verify an integrated GPU smoke run, including interrupted continuation and
   the existing W&B run, and measure logging/checkpoint overhead before the
   full epoch.

Both launchers have local tests, including an actual detached subprocess using
transferred source, failure exit codes, archive integrity and runtime constraints.
Proxy transport tests use real local pseudo-terminals to cover large binary input,
disabled echo, CRLF translation, detached execution, log reconnection, timeouts and
disconnects. On 2026-09-16, a bounded test through the real RunPod proxy verified
Python startup, the presence of required secrets and `/workspace`, and transfer of
a 166,701-byte source archive with a matching SHA-256. It created no remote files or
jobs and downloaded no models or data. Real downloads/uploads and the integrated
H200 run remain deployment checks. The existing image can be used.
