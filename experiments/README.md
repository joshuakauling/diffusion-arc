# Experiments

Experiment notebooks are grouped by purpose. Durable local result snapshots stay
under [`results/`](results/) and retain their existing run-oriented layout.

The completed overfitting experiments have been removed from the current source
tree. Their [findings and historical sources](../docs/experiments/overfitting/README.md)
remain available, including the original notebooks, runners, and configurations.

## Setup

- [`download_arc_agi_2_to_modal.ipynb`](setup/download_arc_agi_2_to_modal.ipynb)
- [`export_nvarc_classified_kaggle.ipynb`](setup/export_nvarc_classified_kaggle.ipynb)
- [`download_flux2_nvarc_to_drive.ipynb`](setup/download_flux2_nvarc_to_drive.ipynb)
- [`download_flux2_nvarc_to_modal.ipynb`](setup/download_flux2_nvarc_to_modal.ipynb)

## Preflight

- [`nvarc_batch_preflight_colab.ipynb`](preflight/nvarc_batch_preflight_colab.ipynb)
- [`single-gpu-training-preflight-modal.ipynb`](preflight/single-gpu-training-preflight-modal.ipynb)

## Training

- [`single-gpu-memory-footprint-modal.ipynb`](training/single-gpu-memory-footprint-modal.ipynb)
- [`single-gpu-memory-footprint-gradient-checkpointing-modal.ipynb`](training/single-gpu-memory-footprint-gradient-checkpointing-modal.ipynb)

## Inference and evaluation

- [`flux2_klein_base_colab_inference.ipynb`](inference/flux2_klein_base_colab_inference.ipynb)
- [`nvarc_flux_colab_inference.ipynb`](inference/nvarc_flux_colab_inference.ipynb)
- [`nvarc_frozen_inference.ipynb`](inference/nvarc_frozen_inference.ipynb)
- [`nvarc_arc_agi_1_pass32_colab.ipynb`](inference/nvarc_arc_agi_1_pass32_colab.ipynb)
- [`NVARC direct-sampling Kaggle notebook`](inference/kaggle/nvarc_direct_sampling/nvarc_direct_sampling_kaggle.ipynb)
  is a standalone manual-import copy of the frozen-inference notebook. Only cell
  source contents differ; all other fields and metadata are preserved exactly,
  without adding cell IDs or normalizing the notebook. The original remains
  unchanged. Import the `.ipynb`, reuse the same Kaggle
  inputs and runtime setup described in its opening cells, and start with the
  default four-task `S2_A` pass@2 smoke test. No API upload or separate metadata
  file is needed for manual import; the companion API metadata remains optional.
  Decoding uses two canonical direct samples without TTT, augmentation, or voting.

## VAE experiments

- [`vae_palette_experiment.ipynb`](vae/vae_palette_experiment.ipynb)
- [`vae_palette_real_evaluation.ipynb`](vae/vae_palette_real_evaluation.ipynb)
- [`vae-training-output-roundtrip-audit-colab.ipynb`](vae/vae-training-output-roundtrip-audit-colab.ipynb)
