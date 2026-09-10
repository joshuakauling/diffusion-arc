# Single-GPU profiling

The B3 profiling workload runs seven real optimizer updates: four warm-up
updates followed by three captured updates. It uses the existing B0 dataset
and cached checkpoints with NVARC and FLUX Flash Attention 3 packing.

## Nsight Systems

Run from the repository root:

```bash
.venv/bin/python -m modal run --detach --env diffusion-arc \
  scripts/cloud/profile_single_gpu_nsys_modal.py
```

The launcher uses
`configs/training/profiling/single_gpu_b3_torch_profile.json`, shared with
Torch Profiler. It requests one H200, eight CPU cores and 64 GiB RAM.
The function timeout is 15 minutes; the profiling subprocess has a ten-minute
limit and bounded cleanup. There are no automatic retries or model downloads.

CUDA/NVTX tracing records kernels, CUDA API calls, copies, synchronization and
training phases. GPU metrics sample the CUDA-visible device at 10 kHz for SM
activity, Tensor Core activity and DRAM bandwidth, subject to metric-set support.
CPU sampling, context switches, Python sampling/GIL, OS-runtime/file tracing
and CUDA allocation tracing are disabled. PyTorch allocator boundary samples
remain in the training summary.

Capture starts after GPU synchronization at update 4 and stops after GPU
synchronization at update 7. Torch Profiler is inactive in this mode.
NVTX ranges mark source preparation, NVARC, VAE, FLUX forward/backward and AdamW.
These ranges describe CPU submission; asynchronous GPU work can outlast them.

Results are persisted in the `diffusion-arc-models` volume under
`results/nsys-profiler/<run-id>/`: the Nsight report, training output,
preflight diagnostics, submitted config, source/data hashes, runtime metadata
and training summary. Failed runs retain diagnostics where possible.

```bash
mkdir -p artifacts/profiling/nsys/<run-id>
.venv/bin/python -m modal volume get --env diffusion-arc \
  diffusion-arc-models results/nsys-profiler/<run-id> \
  artifacts/profiling/nsys/<run-id>
```

Open `trace.nsys-rep` in a compatible Nsight Systems GUI. Inspect Diagnostics
Summary and verify updates 5–7 and their associated GPU work. Command success
and device discovery do not establish complete collector coverage; the launcher
reports successful captures as `capture_complete_unverified`.

## Observed limitations

The initial broader selections exited with code 139 under Modal/gVisor, both
with and without native CPU sampling. Their exact crash cause remains unknown.
The reduced CUDA/NVTX and GPU-metrics selection subsequently produced a report,
confirmed by the user. Its diagnostics reported 242,093 GPU metric events,
417,822 CUDA events in the main process and a 24.188-second capture.

That report also reported failure to enable Unified Memory counters, fallback
to software-instrumented CUDA tracing and possible incomplete CUDA/NVTX coverage.
Unified Memory counters are distinct from SM/Tensor activity and DRAM bandwidth.
The report supports investigation of visible execution patterns; completeness
must still be checked before relying on aggregate measurements.

## Torch Profiler

```bash
.venv/bin/python -m modal run --detach --env diffusion-arc \
  scripts/cloud/profile_single_gpu_modal.py \
  --config configs/training/profiling/single_gpu_b3_torch_profile.json
```

Torch Profiler uses two wait updates, two warm-up updates and three active
updates. Its Chrome JSON traces are stored under `results/torch-profiler/`.
Use the separate unprofiled benchmark for throughput comparisons; profiler
timings include instrumentation overhead.

## References

- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/2025.6/UserGuide/index.html)
