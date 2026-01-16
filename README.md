# VerSe2019 Vertebrae Segmentation Pipeline - PyTorch Implementation

A complete PyTorch implementation of a 3-stage deep learning pipeline for automatic vertebrae localization and segmentation in CT scans.

## Overview

This pipeline processes CT scans through three stages:

1. **Stage 1 - Spine Localization**: Identifies the overall spine region in the CT scan
2. **Stage 2 - Vertebrae Localization**: Locates 25 individual vertebrae centroids (C1-C7, T1-T12, L1-L6)
3. **Stage 3 - Vertebrae Segmentation**: Performs voxel-wise segmentation of each vertebra

## Architecture

### Stage 1: Spine Localization
- Simple 3D U-Net for spine heatmap prediction
- Input: 64×64×128 @ 8mm isotropic spacing
- Output: Single-channel heatmap

### Stage 2: Vertebrae Localization
- **SpatialConfigurationNet**: Dual-pathway architecture
  - Local Appearance Pathway: Processes image features
  - Spatial Configuration Pathway: Encodes spatial relationships
  - Element-wise multiplication of heatmaps
- Input: 96×96×128 @ 2mm isotropic spacing
- Output: 25-channel heatmap (one per vertebra)

### Stage 3: Vertebrae Segmentation
- 3D U-Net with distance transform inputs
- Input: Image + distance transforms from landmarks, 96×96×128 @ 1mm spacing
- Output: 26-class segmentation (25 vertebrae + background)

## Project Structure

```
verse_pytorch/
├── __init__.py
├── config.py                    # Pipeline configuration
├── models/
│   ├── __init__.py
│   └── networks.py              # Neural network architectures
├── data/
│   ├── __init__.py
│   └── dataset.py               # Dataset classes and data loading
├── utils/
│   ├── __init__.py
│   ├── heatmap_utils.py         # Heatmap generation and extraction
│   └── postprocessing.py        # Landmark postprocessing
├── training/
│   ├── __init__.py
│   └── trainer.py               # Training pipelines
└── inference/
    ├── __init__.py
    └── inference.py             # Inference pipelines

train.py                         # Main training script
run_inference.py                 # Main inference script
requirements.txt                 # Python dependencies
```

## Installation

```bash
# Clone or navigate to the project directory
cd verse_pytorch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Expected Data Directory Structure

```
data/
├── images/
│   ├── subject001.nii.gz
│   ├── subject002.nii.gz
│   └── ...
├── landmarks/
│   ├── subject001.csv          # Centroid coordinates
│   ├── subject002.csv
│   └── ...
├── segmentations/              # For Stage 3 training
│   ├── subject001_seg.nii.gz
│   ├── subject002_seg.nii.gz
│   └── ...
├── train.txt                   # List of training subject IDs
└── val.txt                     # List of validation subject IDs
```

### Landmark CSV Format

```csv
label,X,Y,Z
C1,123.5,234.2,345.1
C2,124.1,233.8,338.5
...
L6,130.2,240.1,180.3
```

### Subject ID List Format

```
subject001
subject002
subject003
...
```

## Training

### Train All Stages

```bash
python train.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --stage all \
    --batch_size 1 \
    --learning_rate 1e-4
```

### Train Individual Stage

```bash
# Stage 1: Spine Localization
python train.py --stage 1 --data_dir /path/to/data --output_dir /path/to/output

# Stage 2: Vertebrae Localization
python train.py --stage 2 --data_dir /path/to/data --output_dir /path/to/output

# Stage 3: Vertebrae Segmentation
python train.py --stage 3 --data_dir /path/to/data --output_dir /path/to/output
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_dir` | Path to dataset directory | Required |
| `--output_dir` | Path to output directory | Required |
| `--stage` | Stage(s) to train: 1, 2, 3, or all | all |
| `--epochs` | Number of training epochs | Stage-dependent |
| `--batch_size` | Training batch size | 1 |
| `--learning_rate` | Initial learning rate | 1e-4 |
| `--num_workers` | Data loading workers | 4 |
| `--gpu` | GPU device ID | 0 |
| `--seed` | Random seed | 42 |

## Inference

### Run Full Pipeline

```bash
python run_inference.py \
    --input /path/to/ct_scan.nii.gz \
    --output /path/to/output \
    --stage1_ckpt /path/to/checkpoints/stage1_best.pth \
    --stage2_ckpt /path/to/checkpoints/stage2_best.pth \
    --stage3_ckpt /path/to/checkpoints/stage3_best.pth
```

### Process Multiple Files

```bash
python run_inference.py \
    --input /path/to/input_folder \
    --output /path/to/output \
    --stage1_ckpt /path/to/stage1.pth \
    --stage2_ckpt /path/to/stage2.pth \
    --stage3_ckpt /path/to/stage3.pth
```

### Output Files

For each input file, the pipeline generates:
- `{name}_segmentation.nii.gz`: Vertebrae segmentation mask
- `{name}_landmarks.json`: Detected landmarks in JSON format
- `{name}_landmarks.csv`: Detected landmarks in CSV format

## Python API Usage

### Training Example

```python
from verse_pytorch.config import PipelineConfig
from verse_pytorch.models import SpatialConfigurationNet
from verse_pytorch.data import VertebraeLocalizationDataset
from verse_pytorch.training import VertebraeLocalizationTrainer

# Configuration
config = PipelineConfig()

# Create model
model = SpatialConfigurationNet(
    in_channels=1,
    num_landmarks=25,
    num_filters=64
)

# Create datasets
train_dataset = VertebraeLocalizationDataset(
    data_dir='./data',
    subject_ids=['sub001', 'sub002'],
    config=config.vertebrae_localization,
    is_training=True
)

# Create trainer
trainer = VertebraeLocalizationTrainer(
    model=model,
    train_dataset=train_dataset,
    config=config.vertebrae_localization,
    output_dir='./output'
)

# Train
trainer.train(num_epochs=100)
```

### Inference Example

```python
from verse_pytorch.inference import FullPipelineInference
from verse_pytorch.data import load_nifti

# Load models (see run_inference.py for full example)
pipeline = FullPipelineInference(
    stage1_model=stage1_model,
    stage2_model=stage2_model,
    stage3_model=stage3_model,
    config=config
)

# Load image
image, spacing, affine = load_nifti('ct_scan.nii.gz')

# Run inference
results = pipeline.run(image, spacing)

# Access results
segmentation = results['segmentation']  # (D, H, W) array
landmarks = results['landmarks']        # Dict[str, np.ndarray]
```

## Model Architecture Details

### 3D U-Net

```
Encoder:
  Conv3D(in, 64) -> BN -> ReLU -> Conv3D(64, 64) -> BN -> ReLU -> AvgPool3D
  Conv3D(64, 128) -> BN -> ReLU -> Conv3D(128, 128) -> BN -> ReLU -> AvgPool3D
  Conv3D(128, 256) -> BN -> ReLU -> Conv3D(256, 256) -> BN -> ReLU -> AvgPool3D
  Conv3D(256, 512) -> BN -> ReLU -> Conv3D(512, 512) -> BN -> ReLU

Decoder:
  Upsample -> Concat(skip) -> Conv3D -> BN -> ReLU -> Conv3D -> BN -> ReLU
  (repeated for each level)

Output:
  Conv3D(64, out_channels) -> Sigmoid/Softmax
```

### SpatialConfigurationNet

```
Local Appearance U-Net:
  Input: CT Image (1 channel)
  Output: 25-channel local heatmaps

Spatial Configuration U-Net:
  Input: CT Image (1 channel)
  Output: 25-channel spatial heatmaps

Final:
  Element-wise multiplication of local and spatial heatmaps
```

## Training Recommendations

### GPU Memory
- Stage 1: ~4GB VRAM (small input size)
- Stage 2: ~8GB VRAM
- Stage 3: ~10GB VRAM

### Training Time (Approximate)
- Stage 1: ~2-4 hours
- Stage 2: ~8-12 hours
- Stage 3: ~6-10 hours

### Hyperparameters
- Optimizer: Adam with exponential LR decay (γ=0.99)
- Loss: Soft Dice Loss + MSE for heatmaps, Cross-entropy + Dice for segmentation
- Augmentations: Random flips, rotation, elastic deformation, intensity shifts

## Vertebrae Labels

| Index | Label | Description |
|-------|-------|-------------|
| 1-7 | C1-C7 | Cervical vertebrae |
| 8-19 | T1-T12 | Thoracic vertebrae |
| 20-25 | L1-L6 | Lumbar vertebrae |

## License

This project is provided for research and educational purposes.

## Citation

If you use this code, please cite the original VerSe challenge papers:

```bibtex
@article{verse2021,
    title={VerSe: A Vertebrae Labelling and Segmentation Benchmark for Multi-detector CT Images},
    author={...},
    journal={Medical Image Analysis},
    year={2021}
}
```

## Acknowledgments

- Based on the VerSe2019 challenge methodology
- Reimplemented from TensorFlow to PyTorch
- Uses TorchIO for medical image augmentation
