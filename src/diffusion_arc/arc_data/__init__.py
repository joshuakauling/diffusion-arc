"""Canonical, model-independent ARC data structures."""

from .augmentation import (
    ArcAugmentationKey,
    ArcAugmentationSpec,
    ArcAugmentedTrainingSample,
    ArcGeometry,
    apply_training_augmentation,
    augment_grid,
    format_augmentation_key,
    make_augmentation_key,
    parse_augmentation_key,
    sample_training_augmentation,
    validate_augmentation_for_task,
)
from .canvas import (
    ARC_PALETTE_RGB,
    CANVAS_SIZE,
    CELL_SIZE,
    PAD_CLASS,
    render_grid_canvas,
    render_training_target_canvases,
)
from .loading import (
    load_evaluation_targets,
    load_evaluation_task,
    load_training_task,
)
from .prompt import (
    ArcPromptTarget,
    ArcRenderedTrainingSample,
    render_training_prompt,
)
from .types import (
    ArcEvaluationTargets,
    ArcEvaluationTask,
    ArcPair,
    ArcQuery,
    ArcTarget,
    ArcTrainingTask,
    Grid,
)

__all__ = [
    "ARC_PALETTE_RGB",
    "ArcAugmentationKey",
    "ArcAugmentationSpec",
    "ArcAugmentedTrainingSample",
    "ArcEvaluationTargets",
    "ArcEvaluationTask",
    "ArcGeometry",
    "ArcPair",
    "ArcPromptTarget",
    "ArcQuery",
    "ArcRenderedTrainingSample",
    "ArcTarget",
    "ArcTrainingTask",
    "CANVAS_SIZE",
    "CELL_SIZE",
    "Grid",
    "PAD_CLASS",
    "apply_training_augmentation",
    "augment_grid",
    "format_augmentation_key",
    "load_evaluation_targets",
    "load_evaluation_task",
    "load_training_task",
    "make_augmentation_key",
    "parse_augmentation_key",
    "render_grid_canvas",
    "render_training_prompt",
    "render_training_target_canvases",
    "sample_training_augmentation",
    "validate_augmentation_for_task",
]
