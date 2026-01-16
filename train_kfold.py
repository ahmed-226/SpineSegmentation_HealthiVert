#!/usr/bin/env python
"""
K-Fold Cross-Validation Training Script for VerSe2019 Pipeline

Supports 5-fold cross-validation with comprehensive evaluation and visualization.

Usage:
    python train_kfold.py --data_dir /path/to/data --output_dir /path/to/output --stage all --n_folds 5
"""

import argparse
import os
import sys
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    PipelineConfig,
    SpineLocalizationConfig,
    VertebraeLocalizationConfig,
    VertebraeSegmentationConfig
)
from src.models import SimpleUNet, SpatialConfigurationNet, SegmentationUNet
from src.data import (
    SpineLocalizationDataset,
    VertebraeLocalizationDataset,
    VertebraeSegmentationDataset,
    load_id_list,
    load_all_verse_landmarks
)
from src.training import (
    SpineLocalizationTrainer,
    VertebraeLocalizationTrainer,
    VertebraeSegmentationTrainer
)
from src.utils.kfold import create_kfold_splits, save_fold_splits, get_fold_summary
from src.evaluation import (
    MetricsLogger,
    LocalizationEvaluator,
    SegmentationEvaluator,
    ResultsVisualizer,
    generate_results_report,
    generate_all_plots
)


def setup_logging(output_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{experiment_name}_{timestamp}.log'
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_models(config: PipelineConfig, device: torch.device):
    """Create all stage models"""
    models = {
        'stage1': SimpleUNet(
            in_channels=1,
            num_labels=1,
            num_filters_base=config.spine_localization.num_filters_base
        ).to(device),
        'stage2': SpatialConfigurationNet(
            in_channels=1,
            num_labels=config.vertebrae_localization.num_landmarks,
            num_filters_base=config.vertebrae_localization.num_filters_base
        ).to(device),
        'stage3': SegmentationUNet(
            in_channels=1,
            num_classes=config.vertebrae_segmentation.num_labels + 1,
            num_filters_base=config.vertebrae_segmentation.num_filters_base
        ).to(device)
    }
    return models


def train_fold_stage1(
    args,
    config: PipelineConfig,
    fold_idx: int,
    train_ids: List[str],
    val_ids: List[str],
    landmarks_dict: Dict,
    logger: logging.Logger
) -> str:
    """Train Stage 1 for a single fold"""
    logger.info(f"  Training Stage 1 (Fold {fold_idx})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNet(
        in_channels=1,
        num_labels=1,
        num_filters_base=config.spine_localization.num_filters_base
    )  # Don't move to device here - trainer will handle it
    
    train_dataset = SpineLocalizationDataset(
        data_folder=args.data_dir,
        id_list=train_ids,
        landmarks_dict=landmarks_dict,
        image_size=config.spine_localization.image_size,
        image_spacing=config.spine_localization.image_spacing,
        heatmap_sigma=config.spine_localization.heatmap_sigma,
        is_training=True
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
    
    output_dir = Path(args.output_dir) / 'stage1' / f'fold_{fold_idx}'
    
    trainer = SpineLocalizationTrainer(
        config=config,
        output_dir=str(output_dir),
        cv_fold=fold_idx,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        use_multi_gpu=args.multi_gpu
    )
    
    trainer.train()
    
    checkpoint_path = output_dir / 'checkpoints' / f'fold_{fold_idx}'
    best_model = checkpoint_path / 'best_model.pth'
    final_model = checkpoint_path / 'final_model.pth'
    
    return str(best_model if best_model.exists() else final_model)


def train_fold_stage2(
    args,
    config: PipelineConfig,
    fold_idx: int,
    train_ids: List[str],
    val_ids: List[str],
    landmarks_dict: Dict,
    logger: logging.Logger
) -> str:
    """Train Stage 2 for a single fold"""
    logger.info(f"  Training Stage 2 (Fold {fold_idx})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpatialConfigurationNet(
        in_channels=1,
        num_labels=config.vertebrae_localization.num_landmarks,
        num_filters_base=config.vertebrae_localization.num_filters_base
    )  # Don't move to device here - trainer will handle it
    
    train_dataset = VertebraeLocalizationDataset(
        data_folder=args.data_dir,
        id_list=train_ids,
        landmarks_dict=landmarks_dict,
        image_size=config.vertebrae_localization.image_size,
        image_spacing=config.vertebrae_localization.image_spacing,
        num_landmarks=config.vertebrae_localization.num_landmarks,
        heatmap_sigma=config.vertebrae_localization.heatmap_sigma,
        is_training=True
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
    
    output_dir = Path(args.output_dir) / 'stage2' / f'fold_{fold_idx}'
    
    trainer = VertebraeLocalizationTrainer(
        config=config,
        output_dir=str(output_dir),
        cv_fold=fold_idx,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        use_multi_gpu=args.multi_gpu
    )
    
    trainer.train()
    
    checkpoint_path = output_dir / 'checkpoints' / f'fold_{fold_idx}'
    best_model = checkpoint_path / 'best_model.pth'
    final_model = checkpoint_path / 'final_model.pth'
    
    return str(best_model if best_model.exists() else final_model)


def train_fold_stage3(
    args,
    config: PipelineConfig,
    fold_idx: int,
    train_ids: List[str],
    val_ids: List[str],
    landmarks_dict: Dict,
    logger: logging.Logger
) -> str:
    """Train Stage 3 for a single fold"""
    logger.info(f"  Training Stage 3 (Fold {fold_idx})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegmentationUNet(
        in_channels=1,
        num_classes=config.vertebrae_segmentation.num_labels + 1,
        num_filters_base=config.vertebrae_segmentation.num_filters_base
    )  # Don't move to device here - trainer will handle it
    
    train_dataset = VertebraeSegmentationDataset(
        data_folder=args.data_dir,
        id_list=train_ids,
        landmarks_dict=landmarks_dict,
        labels_folder=args.data_dir,
        image_size=config.vertebrae_segmentation.image_size,
        image_spacing=config.vertebrae_segmentation.image_spacing,
        num_classes=config.vertebrae_segmentation.num_labels,
        is_training=True
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
    
    # Validate datasets have samples
    if len(train_dataset) == 0:
        logger.warning(f"  Stage 3 SKIPPED: No training samples found (0 valid landmarks)")
        logger.warning(f"    This typically means landmark JSON files are missing or malformed.")
        logger.warning(f"    Ensure derivatives/*/{'{'}subject_id{'}'}_seg-subreg_ctd.json files exist.")
        return None
    
    if len(val_dataset) == 0:
        logger.warning(f"  Stage 3: No validation samples found. Training will proceed without validation.")
    
    logger.info(f"  Stage 3 datasets - Train: {len(train_dataset)} vertebra samples, Val: {len(val_dataset)} vertebra samples")
    
    output_dir = Path(args.output_dir) / 'stage3' / f'fold_{fold_idx}'
    
    trainer = VertebraeSegmentationTrainer(
        config=config,
        output_dir=str(output_dir),
        cv_fold=fold_idx,
        labels_folder=args.data_dir,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        use_multi_gpu=args.multi_gpu
    )
    
    trainer.train()
    
    checkpoint_path = output_dir / 'checkpoints' / f'fold_{fold_idx}'
    best_model = checkpoint_path / 'best_model.pth'
    final_model = checkpoint_path / 'final_model.pth'
    
    return str(best_model if best_model.exists() else final_model)


def main():
    parser = argparse.ArgumentParser(
        description='K-Fold Cross-Validation Training for VerSe2019 Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory')
    
    # Stage selection
    parser.add_argument('--stage', type=str, default='all',
                        choices=['1', '2', '3', 'all'],
                        help='Which stage(s) to train')
    
    # K-Fold parameters
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--fold', type=int, default=None,
                        help='Train only specific fold (0 to n_folds-1). If None, trains all folds.')
    
    # Training parameters
    parser.add_argument('--iterations', type=int, default=None,
                        help='Max training iterations per fold (default: 50000)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--test_interval', type=int, default=None,
                        help='Validation interval (default: 500)')
    parser.add_argument('--snapshot_interval', type=int, default=None,
                        help='Checkpoint save interval (default: 1000)')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (ignored if --multi_gpu is set)')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Use all available GPUs with DataParallel')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output options
    parser.add_argument('--generate_plots', action='store_true',
                        help='Generate visualization plots after training')
    parser.add_argument('--experiment_name', type=str, default='verse2019_kfold',
                        help='Name for the experiment')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        if not args.multi_gpu:
            torch.cuda.set_device(args.gpu)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.experiment_name)
    
    # Log configuration
    logger.info("=" * 70)
    logger.info("VerSe2019 Pipeline - K-Fold Cross-Validation Training")
    logger.info("=" * 70)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Number of folds: {args.n_folds}")
    logger.info(f"Specific fold: {args.fold if args.fold is not None else 'All'}")
    
    # Log GPU information
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if args.multi_gpu and num_gpus > 1:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
            logger.info(f"Multi-GPU mode: Using {num_gpus} GPUs: {gpu_names}")
        else:
            logger.info(f"GPU: {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
            if num_gpus > 1:
                logger.info(f"  Note: {num_gpus} GPUs available. Use --multi_gpu to enable multi-GPU training.")
    else:
        logger.info("GPU: None (using CPU)")
    
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    if args.iterations:
        logger.info(f"Max iterations: {args.iterations}")
    logger.info("=" * 70)
    
    # Create configuration
    config = PipelineConfig()
    
    # Update config with command line arguments
    config.spine_localization.batch_size = args.batch_size
    config.vertebrae_localization.batch_size = args.batch_size
    config.vertebrae_segmentation.batch_size = args.batch_size
    
    config.spine_localization.learning_rate = args.learning_rate
    config.vertebrae_localization.learning_rate = args.learning_rate
    config.vertebrae_segmentation.learning_rate = args.learning_rate
    
    if args.iterations:
        config.spine_localization.max_iterations = args.iterations
        config.vertebrae_localization.max_iterations = args.iterations
        config.vertebrae_segmentation.max_iterations = args.iterations
    
    if args.test_interval:
        config.spine_localization.test_interval = args.test_interval
        config.vertebrae_localization.test_interval = args.test_interval
        config.vertebrae_segmentation.test_interval = args.test_interval
    
    if args.snapshot_interval:
        config.spine_localization.snapshot_interval = args.snapshot_interval
        config.vertebrae_localization.snapshot_interval = args.snapshot_interval
        config.vertebrae_segmentation.snapshot_interval = args.snapshot_interval
    
    # Load all subject IDs
    train_file = Path(args.data_dir) / 'train.txt'
    val_file = Path(args.data_dir) / 'val.txt'
    
    all_ids = []
    if train_file.exists():
        all_ids.extend(load_id_list(str(train_file)))
    if val_file.exists():
        all_ids.extend(load_id_list(str(val_file)))

    # Fallback: scan directories if no IDs loaded
    if not all_ids:
        logger.info("No train.txt or val.txt found or they were empty. Scanning for subjects...")
        data_path = Path(args.data_dir)
        
        # Method 1: Recursive search for rawdata/sub-*
        # This handles structures like:
        # /content/verse19/dataset-verse19training/rawdata/sub-verse004
        # /content/verse19/dataset-verse19test/rawdata/sub-verse012
        rawdata_dirs = list(data_path.rglob('rawdata'))
        if rawdata_dirs:
            for rd in rawdata_dirs:
                # Find all sub-* directories inside rawdata
                subs = [p.name for p in rd.glob('sub-*') if p.is_dir()]
                all_ids.extend(subs)
                if subs:
                    logger.info(f"Found {len(subs)} subjects in {rd}")
        
        # Method 2: Recursive search for derivatives/sub-*
        # This handles /content/verse19/dataset-verse19training/derivatives/sub-*
        if not all_ids:
            deriv_dirs = list(data_path.rglob('derivatives'))
            for dd in deriv_dirs:
                 subs = [p.name for p in dd.glob('sub-*') if p.is_dir()]
                 all_ids.extend(subs)
                 if subs:
                    logger.info(f"Found {len(subs)} subjects in {dd}")

        # Method 3: Direct sub-* children search (recursive)
        if not all_ids:
             # Look for any directory starting with sub- anywhere in the tree
             # limiting depth to avoid traversing too deep if unnecessary
             for p in data_path.rglob('sub-*/'):
                 if p.is_dir() and 'sub-' in p.name:
                     all_ids.append(p.name)

    # Remove duplicates while preserving order
    all_ids = sorted(list(set(all_ids)))
    
    logger.info(f"Total subjects for cross-validation: {len(all_ids)}")
    
    if len(all_ids) == 0:
        logger.error(f"No subjects found in {args.data_dir}. Exiting.")
        logger.error("Please ensure the data directory contains 'sub-*' folders or 'dataset-verse*/rawdata/sub-*' structures.")
        return

    if len(all_ids) < args.n_folds:
        logger.warning(f"Not enough subjects ({len(all_ids)}) for {args.n_folds}-fold CV. "
                      f"Reducing to {len(all_ids)} folds.")
        args.n_folds = len(all_ids)
    
    if args.n_folds < 1:
        logger.error("Invalid number of folds (0). Requires at least 1 subject.")
        return
    
    # Create k-fold splits
    folds = create_kfold_splits(all_ids, n_folds=args.n_folds, seed=args.seed)
    fold_summary = get_fold_summary(folds)
    
    logger.info(f"Fold summary: {fold_summary}")
    
    # Save fold splits
    save_fold_splits(folds, Path(args.output_dir) / 'fold_splits')
    
    # Load landmarks for all subjects
    logger.info("Loading landmarks...")
    landmarks_dict = load_all_verse_landmarks(args.data_dir, all_ids)
    logger.info(f"Loaded landmarks for {len(landmarks_dict)} subjects")
    
    # Determine which folds to train
    if args.fold is not None:
        folds_to_train = [args.fold]
    else:
        folds_to_train = list(range(args.n_folds))
    
    # Storage for results
    fold_results = {
        'stage1': {},
        'stage2': {},
        'stage3': {}
    }
    
    # Train each fold
    for fold_idx in folds_to_train:
        train_ids, val_ids = folds[fold_idx]
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"FOLD {fold_idx + 1}/{args.n_folds}")
        logger.info(f"  Train: {len(train_ids)} samples, Val: {len(val_ids)} samples")
        logger.info("=" * 70)
        
        try:
            # Stage 1
            if args.stage in ['1', 'all']:
                ckpt1 = train_fold_stage1(
                    args, config, fold_idx, train_ids, val_ids, landmarks_dict, logger
                )
                fold_results['stage1'][fold_idx] = {'checkpoint': ckpt1}
                logger.info(f"  Stage 1 checkpoint: {ckpt1}")
            
            # Stage 2
            if args.stage in ['2', 'all']:
                ckpt2 = train_fold_stage2(
                    args, config, fold_idx, train_ids, val_ids, landmarks_dict, logger
                )
                fold_results['stage2'][fold_idx] = {'checkpoint': ckpt2}
                logger.info(f"  Stage 2 checkpoint: {ckpt2}")
            
            # Stage 3
            if args.stage in ['3', 'all']:
                ckpt3 = train_fold_stage3(
                    args, config, fold_idx, train_ids, val_ids, landmarks_dict, logger
                )
                fold_results['stage3'][fold_idx] = {'checkpoint': ckpt3}
                logger.info(f"  Stage 3 checkpoint: {ckpt3}")
            
            logger.info(f"Fold {fold_idx} completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in fold {fold_idx}: {e}", exc_info=True)
            continue
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    
    for stage, results in fold_results.items():
        if results:
            logger.info(f"\n{stage.upper()} Results:")
            for fold_idx, data in sorted(results.items()):
                logger.info(f"  Fold {fold_idx}: {data.get('checkpoint', 'N/A')}")
    
    # Generate report file
    report_path = Path(args.output_dir) / f'{args.experiment_name}_summary.txt'
    with open(report_path, 'w') as f:
        f.write(f"VerSe2019 K-Fold Cross-Validation Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data directory: {args.data_dir}\n")
        f.write(f"Number of folds: {args.n_folds}\n")
        f.write(f"Total subjects: {len(all_ids)}\n")
        f.write(f"Iterations per fold: {args.iterations or config.spine_localization.max_iterations}\n\n")
        
        for stage, results in fold_results.items():
            if results:
                f.write(f"\n{stage.upper()}:\n")
                f.write("-" * 30 + "\n")
                for fold_idx, data in sorted(results.items()):
                    f.write(f"  Fold {fold_idx}: {data.get('checkpoint', 'N/A')}\n")
    
    logger.info(f"\nSummary saved to: {report_path}")
    
    # Generate plots if requested
    if args.generate_plots:
        logger.info("\nGenerating visualization plots...")
        # Note: This would require running evaluation on checkpoints
        # For now, just create placeholder
        plots_dir = Path(args.output_dir) / 'plots'
        plots_dir.mkdir(exist_ok=True)
        logger.info(f"Plots directory: {plots_dir}")
        logger.info("Note: Run evaluate.py to generate comprehensive plots from trained models")
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()
