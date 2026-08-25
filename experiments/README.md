# Experiments

Experiment notebooks are grouped by purpose. Durable local result snapshots stay
under [`results/`](results/) and retain their existing run-oriented layout.

## Setup

- [`download_arc_agi_2_to_modal.ipynb`](setup/download_arc_agi_2_to_modal.ipynb)
- [`download_flux2_nvarc_to_drive.ipynb`](setup/download_flux2_nvarc_to_drive.ipynb)
- [`download_flux2_nvarc_to_modal.ipynb`](setup/download_flux2_nvarc_to_modal.ipynb)

## Preflight

- [`nvarc_batch_preflight_colab.ipynb`](preflight/nvarc_batch_preflight_colab.ipynb)
- [`single-gpu-training-preflight-modal.ipynb`](preflight/single-gpu-training-preflight-modal.ipynb)

## Training

- [`single-gpu-memory-footprint-modal.ipynb`](training/single-gpu-memory-footprint-modal.ipynb)
- [`single-gpu-memory-footprint-gradient-checkpointing-modal.ipynb`](training/single-gpu-memory-footprint-gradient-checkpointing-modal.ipynb)
- [`single-gpu-overfitting-modal.ipynb`](training/single-gpu-overfitting-modal.ipynb)
- [`single-gpu-overfitting-v2-modal.ipynb`](training/single-gpu-overfitting-v2-modal.ipynb)
- [`single-task-conditioning-overfitting-ladder-modal.ipynb`](training/single-task-conditioning-overfitting-ladder-modal.ipynb)
- [`single-task-spatial-overfitting-ablation-modal.ipynb`](training/single-task-spatial-overfitting-ablation-modal.ipynb)

## Inference and evaluation

- [`flux2_klein_base_colab_inference.ipynb`](inference/flux2_klein_base_colab_inference.ipynb)
- [`nvarc_flux_colab_inference.ipynb`](inference/nvarc_flux_colab_inference.ipynb)
- [`nvarc_frozen_inference.ipynb`](inference/nvarc_frozen_inference.ipynb)
- [`evaluate-flux-checkpoint-training-view-modal.ipynb`](inference/evaluate-flux-checkpoint-training-view-modal.ipynb)
- [`compare-flux-inference-schedules-modal.ipynb`](inference/compare-flux-inference-schedules-modal.ipynb)
- [`compare-flux-restart-sampling-modal.ipynb`](inference/compare-flux-restart-sampling-modal.ipynb)
- [`Kaggle direct-sampling kernel`](inference/kaggle/nvarc_direct_sampling/)

## VAE experiments

- [`vae_palette_experiment.ipynb`](vae/vae_palette_experiment.ipynb)
- [`vae_palette_real_evaluation.ipynb`](vae/vae_palette_real_evaluation.ipynb)
