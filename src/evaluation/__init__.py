"""
Evaluation module for VerSe2019 Pipeline
"""
from .evaluator import (
    MetricsLogger,
    LocalizationEvaluator,
    SegmentationEvaluator,
    ResultsVisualizer,
    generate_results_report,
    generate_all_plots,
    VERTEBRAE_NAMES,
    REGION_INDICES
)

__all__ = [
    'MetricsLogger',
    'LocalizationEvaluator',
    'SegmentationEvaluator',
    'ResultsVisualizer',
    'generate_results_report',
    'generate_all_plots',
    'VERTEBRAE_NAMES',
    'REGION_INDICES'
]
