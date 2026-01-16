"""
Configuration file for the VerSe2019 Vertebrae Segmentation Pipeline
PyTorch Implementation
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import os


@dataclass
class SpineLocalizationConfig:
    """Configuration for Stage 1: Spine Localization"""
    # Image configuration
    image_size: Tuple[int, int, int] = (64, 64, 128)  # (D, H, W)
    image_spacing: Tuple[float, float, float] = (8.0, 8.0, 8.0)  # mm
    
    # Network configuration
    num_labels: int = 1  # Single spine heatmap
    num_filters_base: int = 64
    num_levels: int = 4
    dropout_ratio: float = 0.0
    heatmap_initialization: bool = True
    
    # Training configuration
    batch_size: int = 1
    learning_rate: float = 0.0001
    max_iterations: int = 50000
    test_interval: int = 5000
    snapshot_interval: int = 5000
    
    # Heatmap generation
    heatmap_sigma: float = 4.0


@dataclass
class VertebraeLocalizationConfig:
    """Configuration for Stage 2: Vertebrae Localization"""
    # Image configuration
    image_size: Tuple[int, int, int] = (96, 96, 128)  # (D, H, W)
    image_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # mm
    
    # Network configuration
    num_landmarks: int = 25  # C1-C7 (7) + T1-T12 (12) + L1-L6 (6)
    num_filters_base: int = 96
    num_levels: int = 4
    dropout_ratio: float = 0.25
    spatial_downsample: int = 4
    activation: str = 'leaky_relu'
    local_activation: str = 'tanh'
    spatial_activation: str = 'tanh'
    
    # Training configuration
    batch_size: int = 1
    learning_rate: float = 0.0001
    max_iterations: int = 50000
    test_interval: int = 5000
    snapshot_interval: int = 5000
    gradient_clip_norm: float = 10000.0
    
    # Heatmap configuration
    heatmap_sigma: float = 4.0
    learnable_sigma: bool = True
    sigma_regularization: float = 0.00001


@dataclass
class VertebraeSegmentationConfig:
    """Configuration for Stage 3: Vertebrae Segmentation"""
    # Image configuration
    image_size: Tuple[int, int, int] = (96, 96, 128)  # (D, H, W)
    image_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # mm (finer)
    
    # Network configuration
    num_labels: int = 26  # Background + 25 vertebrae
    num_filters_base: int = 64
    num_levels: int = 4
    dropout_ratio: float = 0.0
    
    # Training configuration
    batch_size: int = 1
    learning_rate: float = 0.0001
    max_iterations: int = 50000
    test_interval: int = 5000
    snapshot_interval: int = 5000


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    # Spatial augmentations
    random_rotation: Tuple[float, float, float] = (0.25, 0.25, 0.25)  # radians
    random_scale: float = 0.15  # ±15%
    random_translation: Tuple[float, float, float] = (30.0, 30.0, 30.0)  # mm
    flip_probability: float = 0.5
    
    # Elastic deformation
    elastic_deformation: bool = True
    elastic_grid_nodes: Tuple[int, int, int] = (7, 7, 7)  # More control points
    elastic_max_deformation: Tuple[float, float, float] = (8.0, 8.0, 8.0)  # Reduced to avoid folding
    
    # Intensity augmentations
    intensity_shift_range: float = 0.25
    intensity_scale_range: float = 0.25
    gamma_range: Tuple[float, float] = (0.9, 1.1)


@dataclass
class DataConfig:
    """Data paths and configuration"""
    # Base paths
    data_folder: str = ""
    output_folder: str = ""
    
    # Dataset split files
    train_list: str = "train.txt"
    val_list: str = "val.txt"
    test_list: str = "test.txt"
    
    # Landmarks file
    landmarks_file: str = "landmarks.csv"
    
    # Cross-validation fold
    cv_fold: int = 0
    
    # Data format
    data_format: str = 'channels_first'
    
    # Intensity normalization
    hu_min: float = -1024.0
    hu_max: float = 3000.0
    intensity_min: float = -1.0
    intensity_max: float = 1.0


@dataclass
class PipelineConfig:
    """Master configuration for the entire pipeline"""
    # Stage configurations
    spine_localization: SpineLocalizationConfig = field(
        default_factory=SpineLocalizationConfig
    )
    vertebrae_localization: VertebraeLocalizationConfig = field(
        default_factory=VertebraeLocalizationConfig
    )
    vertebrae_segmentation: VertebraeSegmentationConfig = field(
        default_factory=VertebraeSegmentationConfig
    )
    augmentation: AugmentationConfig = field(
        default_factory=AugmentationConfig
    )
    data: DataConfig = field(
        default_factory=DataConfig
    )
    
    # Device configuration
    device: str = 'cuda'
    num_workers: int = 2  # Default to 2 for Colab compatibility
    pin_memory: bool = True
    
    # Random seed
    seed: int = 42
    
    def update_paths(self, data_folder: str, output_folder: str):
        """Update data paths"""
        self.data.data_folder = data_folder
        self.data.output_folder = output_folder
        
        # Create output folder if not exists
        os.makedirs(output_folder, exist_ok=True)


# Vertebrae labels mapping
VERTEBRAE_LABELS = {
    # Cervical vertebrae
    0: 'C1', 1: 'C2', 2: 'C3', 3: 'C4', 4: 'C5', 5: 'C6', 6: 'C7',
    # Thoracic vertebrae
    7: 'T1', 8: 'T2', 9: 'T3', 10: 'T4', 11: 'T5', 12: 'T6',
    13: 'T7', 14: 'T8', 15: 'T9', 16: 'T10', 17: 'T11', 18: 'T12',
    # Lumbar vertebrae
    19: 'L1', 20: 'L2', 21: 'L3', 22: 'L4', 23: 'L5', 24: 'L6'
}

LABEL_TO_INDEX = {v: k for k, v in VERTEBRAE_LABELS.items()}
