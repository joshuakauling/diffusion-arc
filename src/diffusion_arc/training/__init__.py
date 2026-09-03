"""Training-side assembly and execution utilities."""

from .collation import (
    CollatedTrainingBatch,
    TrainingBatchMetadata,
    TrainingCollator,
)
from .dataset import ArcTrainingDataset
from .flow_probes import (
    FixedFlowProbeResult,
    evaluate_fixed_flow_probe,
    prepare_flow_probe_microbatch,
)
from .flux_input import (
    FluxInput,
    NvarcSequenceState,
    prepare_flux_inputs,
    prepare_flux_inputs_batch,
)
from .flux_loss import compute_flux_microbatch_loss
from .flux_microbatch import FluxMicrobatch, build_flux_microbatch
from .single_gpu_loop import OptimizerStepMetrics, run_single_gpu_training

__all__ = [
    "ArcTrainingDataset",
    "CollatedTrainingBatch",
    "FluxInput",
    "FluxMicrobatch",
    "FixedFlowProbeResult",
    "NvarcSequenceState",
    "OptimizerStepMetrics",
    "TrainingBatchMetadata",
    "TrainingCollator",
    "build_flux_microbatch",
    "compute_flux_microbatch_loss",
    "evaluate_fixed_flow_probe",
    "prepare_flux_inputs",
    "prepare_flux_inputs_batch",
    "prepare_flow_probe_microbatch",
    "run_single_gpu_training",
]
