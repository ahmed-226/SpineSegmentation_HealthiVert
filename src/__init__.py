"""
VerSe2019 Vertebrae Segmentation Pipeline - PyTorch Implementation

A 3-stage deep learning pipeline for automatic vertebrae localization and 
segmentation in CT scans:
1. Spine Localization
2. Vertebrae Localization (25 landmarks)
3. Vertebrae Segmentation
"""

__version__ = '1.0.0'
__author__ = 'HealthiVert Team'

from .config import (
    PipelineConfig,
    SpineLocalizationConfig,
    VertebraeLocalizationConfig,
    VertebraeSegmentationConfig,
    AugmentationConfig,
    DataConfig,
    VERTEBRAE_LABELS,
    LABEL_TO_INDEX
)
