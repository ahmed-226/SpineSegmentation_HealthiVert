#!/usr/bin/env python
"""
Main training script for VerSe2019 Vertebrae Segmentation Pipeline

This script trains all three stages of the pipeline:
1. Spine Localization
2. Vertebrae Localization  
3. Vertebrae Segmentation

Usage:
    python train.py --stage all --data_dir /path/to/data --output_dir /path/to/output
    python train.py --stage 1 --data_dir /path/to/data --output_dir /path/to/output
    python train.py --stage 2 --checkpoint_stage1 /path/to/stage1.pth
"""

import argparse
import os
import sys
import torch
import logging
from pathlib import Path
from datetime import datetime

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    PipelineConfig,
    SpineLocalizationConfig,
    VertebraeLocalizationConfig,
    VertebraeSegmentationConfig,
    DataConfig,
    AugmentationConfig
)
from src.models import UNet3D, SpatialConfigurationNet, SimpleUNet, SegmentationUNet
from src.data import (
    SpineLocalizationDataset,
    VertebraeLocalizationDataset,
    VertebraeSegmentationDataset,
    load_id_list,
    load_all_verse_landmarks,
    create_data_loaders
)
from src.training import (
    SpineLocalizationTrainer,
    VertebraeLocalizationTrainer,
    VertebraeSegmentationTrainer,
    train_pipeline
)


def setup_logging(output_dir: str, stage: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_stage{stage}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_stage1_model(config: SpineLocalizationConfig, device: torch.device) -> torch.nn.Module:
    """Create spine localization model"""
    model = SimpleUNet(
        in_channels=1,
        num_labels=1,
        num_filters_base=config.num_filters_base
    ).to(device)
    return model


def create_stage2_model(config: VertebraeLocalizationConfig, device: torch.device) -> torch.nn.Module:
    """Create vertebrae localization model"""
    model = SpatialConfigurationNet(
        in_channels=1,
        num_labels=config.num_landmarks,
        num_filters_base=config.num_filters_base
    ).to(device)
    return model


def create_stage3_model(config: VertebraeSegmentationConfig, device: torch.device) -> torch.nn.Module:
    """Create vertebrae segmentation model"""
    # Input: image (1) + distance transforms (num_landmarks)
    model = SegmentationUNet(
        in_channels=1 + config.num_labels,
        num_classes=config.num_labels + 1,  # +1 for background
        num_filters_base=config.num_filters_base
    ).to(device)
    return model


def train_stage1(args, config: PipelineConfig, logger: logging.Logger):
    """Train Stage 1: Spine Localization"""
    logger.info("=" * 60)
    logger.info("Starting Stage 1: Spine Localization Training")
    logger.info("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_stage1_model(config.spine_localization, device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load data
    logger.info(f"Loading training data from: {args.data_dir}")
    train_ids = load_id_list(Path(args.data_dir) / 'train.txt')
    val_ids = load_id_list(Path(args.data_dir) / 'val.txt')
    
    # Load landmarks from VerSe19 structure
    logger.info("Loading landmarks...")
    all_ids = train_ids + val_ids
    landmarks_dict = load_all_verse_landmarks(args.data_dir, all_ids)
    logger.info(f"Loaded landmarks for {len(landmarks_dict)} subjects")
    
    train_dataset = SpineLocalizationDataset(
        data_folder=args.data_dir,
        id_list=train_ids,
        landmarks_dict=landmarks_dict,
        image_size=config.spine_localization.image_size,
        image_spacing=config.spine_localization.image_spacing,
        heatmap_sigma=config.spine_localization.heatmap_sigma,
        is_training=True,
        augmentation_config=None
    )
    
    val_dataset = SpineLocalizationDataset(
        data_folder=args.data_dir,
        id_list=val_ids,
        landmarks_dict=landmarks_dict,
        image_size=config.spine_localization.image_size,
        image_spacing=config.spine_localization.image_spacing,
        heatmap_sigma=config.spine_localization.heatmap_sigma,
        is_training=False
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create trainer
    trainer = SpineLocalizationTrainer(
        config=config,
        output_dir=str(Path(args.output_dir) / 'stage1'),
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device
    )
    
    # Train
    trainer.train()
    
    logger.info("Stage 1 training completed!")
    return str(trainer.output_dir / 'checkpoints' / 'fold_0' / 'best_model.pth')


def train_stage2(args, config: PipelineConfig, logger: logging.Logger, stage1_checkpoint: str = None):
    """Train Stage 2: Vertebrae Localization"""
    logger.info("=" * 60)
    logger.info("Starting Stage 2: Vertebrae Localization Training")
    logger.info("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_stage2_model(config.vertebrae_localization, device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load data
    logger.info(f"Loading training data from: {args.data_dir}")
    train_ids = load_id_list(Path(args.data_dir) / 'train.txt')
    val_ids = load_id_list(Path(args.data_dir) / 'val.txt')
    
    # Load landmarks from VerSe19 structure
    logger.info("Loading landmarks...")
    all_ids = train_ids + val_ids
    landmarks_dict = load_all_verse_landmarks(args.data_dir, all_ids)
    logger.info(f"Loaded landmarks for {len(landmarks_dict)} subjects")
    
    train_dataset = VertebraeLocalizationDataset(
        data_folder=args.data_dir,
        id_list=train_ids,
        landmarks_dict=landmarks_dict,
        image_size=config.vertebrae_localization.image_size,
        image_spacing=config.vertebrae_localization.image_spacing,
        num_landmarks=config.vertebrae_localization.num_landmarks,
        heatmap_sigma=config.vertebrae_localization.heatmap_sigma,
        is_training=True,
        augmentation_config=None
    )
    
    val_dataset = VertebraeLocalizationDataset(
        data_folder=args.data_dir,
        id_list=val_ids,
        landmarks_dict=landmarks_dict,
        image_size=config.vertebrae_localization.image_size,
        image_spacing=config.vertebrae_localization.image_spacing,
        num_landmarks=config.vertebrae_localization.num_landmarks,
        heatmap_sigma=config.vertebrae_localization.heatmap_sigma,
        is_training=False
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create trainer
    trainer = VertebraeLocalizationTrainer(
        config=config,
        output_dir=str(Path(args.output_dir) / 'stage2'),
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device
    )
    
    # Train
    trainer.train()
    
    logger.info("Stage 2 training completed!")
    return str(trainer.output_dir / 'checkpoints' / 'fold_0' / 'best_model.pth')


def train_stage3(args, config: PipelineConfig, logger: logging.Logger, stage2_checkpoint: str = None):
    """Train Stage 3: Vertebrae Segmentation"""
    logger.info("=" * 60)
    logger.info("Starting Stage 3: Vertebrae Segmentation Training")
    logger.info("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_stage3_model(config.vertebrae_segmentation, device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load data
    logger.info(f"Loading training data from: {args.data_dir}")
    train_ids = load_id_list(Path(args.data_dir) / 'train.txt')
    val_ids = load_id_list(Path(args.data_dir) / 'val.txt')
    
    # Load landmarks from VerSe19 structure
    logger.info("Loading landmarks...")
    all_ids = train_ids + val_ids
    landmarks_dict = load_all_verse_landmarks(args.data_dir, all_ids)
    logger.info(f"Loaded landmarks for {len(landmarks_dict)} subjects")
    
    train_dataset = VertebraeSegmentationDataset(
        data_folder=args.data_dir,
        id_list=train_ids,
        landmarks_dict=landmarks_dict,
        labels_folder=args.data_dir,  # Will search for masks in VerSe structure
        image_size=config.vertebrae_segmentation.image_size,
        image_spacing=config.vertebrae_segmentation.image_spacing,
        num_classes=config.vertebrae_segmentation.num_labels,
        is_training=True,
        augmentation_config=None
    )
    
    val_dataset = VertebraeSegmentationDataset(
        data_folder=args.data_dir,
        id_list=val_ids,
        landmarks_dict=landmarks_dict,
        labels_folder=args.data_dir,
        image_size=config.vertebrae_segmentation.image_size,
        image_spacing=config.vertebrae_segmentation.image_spacing,
        num_classes=config.vertebrae_segmentation.num_labels,
        is_training=False
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create trainer
    trainer = VertebraeSegmentationTrainer(
        config=config,
        output_dir=str(Path(args.output_dir) / 'stage3'),
        labels_folder=args.data_dir,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device
    )
    
    # Train
    trainer.train()
    
    logger.info("Stage 3 training completed!")
    return str(trainer.output_dir / 'checkpoints' / 'fold_0' / 'best_model.pth')


def main():
    parser = argparse.ArgumentParser(
        description='Train VerSe2019 Vertebrae Segmentation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory containing images, landmarks, and id lists')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for checkpoints and logs')
    
    # Stage selection
    parser.add_argument('--stage', type=str, default='all',
                        choices=['1', '2', '3', 'all'],
                        help='Which stage(s) to train: 1, 2, 3, or all')
    
    # Optional checkpoints for continuing training
    parser.add_argument('--checkpoint_stage1', type=str, default=None,
                        help='Path to Stage 1 checkpoint (required for Stage 2 if training from scratch)')
    parser.add_argument('--checkpoint_stage2', type=str, default=None,
                        help='Path to Stage 2 checkpoint (required for Stage 3 if training from scratch)')
    
    # Training parameters
    parser.add_argument('--iterations', type=int, default=None,
                        help='Max training iterations (default: 50000)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (alternative to --iterations, calculates based on dataset size)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers (default: 2 for Colab)')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.stage)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("VerSe2019 Vertebrae Segmentation Pipeline - Training")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Stage: {args.stage}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    if args.iterations:
        logger.info(f"Max iterations: {args.iterations}")
    if args.epochs:
        logger.info(f"Epochs: {args.epochs}")
    
    # Create configuration
    config = PipelineConfig()
    
    # Update config with command line arguments
    config.data.data_root = args.data_dir
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    
    # Update learning rate for all stages
    config.spine_localization.learning_rate = args.learning_rate
    config.vertebrae_localization.learning_rate = args.learning_rate
    config.vertebrae_segmentation.learning_rate = args.learning_rate
    
    # Update batch size for all stages
    config.spine_localization.batch_size = args.batch_size
    config.vertebrae_localization.batch_size = args.batch_size
    config.vertebrae_segmentation.batch_size = args.batch_size
    
    # Update max iterations if specified
    if args.iterations:
        config.spine_localization.max_iterations = args.iterations
        config.vertebrae_localization.max_iterations = args.iterations
        config.vertebrae_segmentation.max_iterations = args.iterations
    elif args.epochs:
        # Estimate iterations from epochs (will be recalculated per stage based on dataset size)
        # For now, use epochs * estimated_samples_per_epoch
        estimated_iters = args.epochs * 100  # rough estimate, actual calc happens per stage
        config.spine_localization.max_iterations = estimated_iters
        config.vertebrae_localization.max_iterations = estimated_iters
        config.vertebrae_segmentation.max_iterations = estimated_iters
        logger.info(f"Estimated max iterations from epochs: {estimated_iters} (will be adjusted per stage)")
    
    try:
        if args.stage == '1' or args.stage == 'all':
            stage1_ckpt = train_stage1(args, config, logger)
            if args.stage == '1':
                logger.info(f"Stage 1 complete. Checkpoint: {stage1_ckpt}")
                return
        else:
            stage1_ckpt = args.checkpoint_stage1
        
        if args.stage == '2' or args.stage == 'all':
            stage2_ckpt = train_stage2(args, config, logger, stage1_ckpt)
            if args.stage == '2':
                logger.info(f"Stage 2 complete. Checkpoint: {stage2_ckpt}")
                return
        else:
            stage2_ckpt = args.checkpoint_stage2
        
        if args.stage == '3' or args.stage == 'all':
            stage3_ckpt = train_stage3(args, config, logger, stage2_ckpt)
            logger.info(f"Stage 3 complete. Checkpoint: {stage3_ckpt}")
        
        if args.stage == 'all':
            logger.info("=" * 60)
            logger.info("All stages completed successfully!")
            logger.info("=" * 60)
            logger.info(f"Stage 1 checkpoint: {stage1_ckpt}")
            logger.info(f"Stage 2 checkpoint: {stage2_ckpt}")
            logger.info(f"Stage 3 checkpoint: {stage3_ckpt}")
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
