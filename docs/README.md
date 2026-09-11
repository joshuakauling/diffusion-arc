# Documentation

The root [`README.md`](../README.md) defines the authoritative project scope,
research questions, milestone order, evaluation protocol, and non-goals. The
documents below provide supporting architecture, engineering, and experiment
details.

## Local documentation website

From the repository root, create a separate documentation environment:

```bash
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r requirements-docs.txt
.venv-docs/bin/mkdocs serve
```

Open <http://127.0.0.1:8000>. Markdown and source docstring changes reload
automatically. The [Python API reference](api.md) is generated from the source;
it does not require the training dependencies or model downloads.

To build a static website, run `.venv-docs/bin/mkdocs build`. Generated files go
into the ignored `site/` directory. This command does not publish the website.

Links to repository files outside `docs/` are intended for browsing the repository
and are not bundled into the website.

## Architecture

- [Initial flow-only architecture](architecture/initial-flow-only-architecture.md)
- [FLUX.2 Klein architecture and flow matching](architecture/flux2-klein-architecture.md)
- [NVARC overview and evaluation caveats](architecture/nvarc-overview.md)
- [Architecture diagram](architecture/diagrams/architecture.svg)

## Data

- [ARC data pipeline](data/data-pipeline.md)
- [Training data assembly](data/training-data-assembly.md)

## Training

- [Training system overview](training/README.md)
- [FA3 sequence-packing training](training/flux-sequence-packing.md)
- [Historical batching design](training/batching-and-efficiency-roadmap.md)
- [NVARC batch runner](training/nvarc-batch-runner.md)
- [FLUX text conditioning](training/flux-text-conditioning.md)
- [FLUX target-image encoding](training/flux-image-encoding.md)
- [Flow-matching training state](training/flow-matching.md)

## Performance

- [Performance evaluation and benchmark protocol](performance/README.md)
- [B0: first end-to-end training run](performance/b0.md)
- [B1: fixed buckets and batched preparation](performance/b1.md)
- [B2: scheduled batches and CPU prefetch](performance/b2.md)
- [B3: sequence packing with FlashAttention-3](performance/b3.md)
- [B3 NVARC analysis](performance/b3-nvarc.md)
- [B3 FLUX analysis](performance/b3-flux.md)

## Experiments

- [Experiment documentation index](experiments/README.md)
- [Research ideas](research-ideas.md)
