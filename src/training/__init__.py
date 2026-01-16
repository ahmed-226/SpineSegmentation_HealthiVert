"""Training package - Training pipelines for all stages"""

from .trainer import (
    BaseTrainer,
    SpineLocalizationTrainer,
    VertebraeLocalizationTrainer,
    VertebraeSegmentationTrainer,
    train_pipeline
)
