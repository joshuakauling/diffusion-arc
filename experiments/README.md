# Experiments

Experiment notebooks are grouped by purpose. Durable local result snapshots stay
under [`results/`](results/) and retain their existing run-oriented layout.

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
- [`single-gpu-overfitting-modal.ipynb`](training/single-gpu-overfitting-modal.ipynb)
- [`single-gpu-overfitting-v2-modal.ipynb`](training/single-gpu-overfitting-v2-modal.ipynb)
- [`single-task-conditioning-overfitting-ladder-modal.ipynb`](training/single-task-conditioning-overfitting-ladder-modal.ipynb)
- [`single-task-conditioning-endpoint-overfitting-modal.ipynb`](training/single-task-conditioning-endpoint-overfitting-modal.ipynb)
- [`single-task-spatial-overfitting-ablation-modal.ipynb`](training/single-task-spatial-overfitting-ablation-modal.ipynb)
- [`single-task-augmentation-generalization-overfitting-modal.ipynb`](training/single-task-augmentation-generalization-overfitting-modal.ipynb)
- [`single-task-augmentation-generalization-continuation-modal.ipynb`](training/single-task-augmentation-generalization-continuation-modal.ipynb)
- [`single-task-region-weighted-loss-overfitting-modal.ipynb`](training/single-task-region-weighted-loss-overfitting-modal.ipynb)
- [`single-task-region-weighted-loss-continuation-modal.ipynb`](training/single-task-region-weighted-loss-continuation-modal.ipynb)
- [`single-task-region-weighted-endpoint-balanced-continuation-modal.ipynb`](training/single-task-region-weighted-endpoint-balanced-continuation-modal.ipynb)
- [`single-task-region-weighted-endpoint-balanced-long-continuation-modal.ipynb`](training/single-task-region-weighted-endpoint-balanced-long-continuation-modal.ipynb)

## Inference and evaluation

- [`flux2_klein_base_colab_inference.ipynb`](inference/flux2_klein_base_colab_inference.ipynb)
- [`nvarc_flux_colab_inference.ipynb`](inference/nvarc_flux_colab_inference.ipynb)
- [`nvarc_frozen_inference.ipynb`](inference/nvarc_frozen_inference.ipynb)
- [`evaluate-flux-checkpoint-training-view-modal.ipynb`](inference/evaluate-flux-checkpoint-training-view-modal.ipynb)
- [`compare-flux-inference-schedules-modal.ipynb`](inference/compare-flux-inference-schedules-modal.ipynb)
- [`compare-flux-restart-sampling-modal.ipynb`](inference/compare-flux-restart-sampling-modal.ipynb)
- [`diagnose-conditioning-ladder-checkpoints-modal.ipynb`](inference/diagnose-conditioning-ladder-checkpoints-modal.ipynb)
- [`evaluate-single-task-checkpoint-cross-task-modal.ipynb`](inference/evaluate-single-task-checkpoint-cross-task-modal.ipynb)
- [`nvarc_arc_agi_1_pass32_colab.ipynb`](inference/nvarc_arc_agi_1_pass32_colab.ipynb)
- [`Kaggle direct-sampling kernel`](inference/kaggle/nvarc_direct_sampling/)

## VAE experiments

- [`vae_palette_experiment.ipynb`](vae/vae_palette_experiment.ipynb)
- [`vae_palette_real_evaluation.ipynb`](vae/vae_palette_real_evaluation.ipynb)
- [`conditioning-ladder-roundtrip-diagnosis-colab.ipynb`](vae/conditioning-ladder-roundtrip-diagnosis-colab.ipynb)
- [`vae-training-output-roundtrip-audit-colab.ipynb`](vae/vae-training-output-roundtrip-audit-colab.ipynb)
