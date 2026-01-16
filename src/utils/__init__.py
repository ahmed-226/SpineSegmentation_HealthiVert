"""Utilities package - Heatmap generation, postprocessing, and cross-validation"""

from .heatmap_utils import (
    generate_heatmap_target,
    generate_single_heatmap,
    LearnableSigmas,
    extract_landmark_from_heatmap,
    extract_all_landmarks,
    gaussian_smooth_3d,
    find_local_maxima_3d,
    compute_landmark_distance_error,
    interpolate_missing_landmarks
)

from .postprocessing import (
    HeatmapPostprocessor,
    SpineGraphOptimizer,
    LandmarkInterpolator,
    TopBottomFilter,
    VertebraePostprocessor,
    compute_identification_rate,
    compute_segmentation_metrics,
    VERTEBRAE_NAMES,
    VERTEBRAE_DISTANCES_MEAN,
    VERTEBRAE_DISTANCES_STD
)

from .kfold import (
    create_kfold_splits,
    save_fold_splits,
    load_fold_split,
    get_fold_summary
)
