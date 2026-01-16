#!/usr/bin/env python
"""
Main inference script for VerSe2019 Vertebrae Segmentation Pipeline

This script runs inference on new CT scans using trained models.

Usage:
    # Run full pipeline
    python run_inference.py --input /path/to/ct_scan.nii.gz --output /path/to/output \
        --stage1_ckpt /path/to/stage1.pth \
        --stage2_ckpt /path/to/stage2.pth \
        --stage3_ckpt /path/to/stage3.pth
    
    # Run specific stage only
    python run_inference.py --input /path/to/ct_scan.nii.gz --output /path/to/output \
        --stage 1 --stage1_ckpt /path/to/stage1.pth
"""

import argparse
import os
import sys
import torch
import numpy as np
import nibabel as nib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    PipelineConfig,
    SpineLocalizationConfig,
    VertebraeLocalizationConfig,
    VertebraeSegmentationConfig,
    VERTEBRAE_LABELS
)
from src.models import UNet3D, SpatialConfigurationNet, SimpleUNet, SegmentationUNet
from src.data import load_nifti
from src.inference import (
    SpineLocalizationInference,
    VertebraeLocalizationInference,
    VertebraeSegmentationInference,
    FullPipelineInference,
    run_inference
)
from src.utils import VertebraePostprocessor


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'inference_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_model(checkpoint_path: str, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def save_landmarks_json(landmarks: Dict[str, np.ndarray], output_path: str):
    """Save landmarks to JSON file"""
    landmarks_list = []
    for label, coords in landmarks.items():
        if coords is not None:
            landmarks_list.append({
                'label': label,
                'X': float(coords[0]),
                'Y': float(coords[1]),
                'Z': float(coords[2])
            })
    
    with open(output_path, 'w') as f:
        json.dump({'landmarks': landmarks_list}, f, indent=2)


def save_landmarks_csv(landmarks: Dict[str, np.ndarray], output_path: str):
    """Save landmarks to CSV file"""
    with open(output_path, 'w') as f:
        f.write('label,X,Y,Z\n')
        for label, coords in landmarks.items():
            if coords is not None:
                f.write(f'{label},{coords[0]:.2f},{coords[1]:.2f},{coords[2]:.2f}\n')


def run_full_pipeline(args, config: PipelineConfig, logger: logging.Logger):
    """Run full 3-stage pipeline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create models
    logger.info("Creating models...")
    
    stage1_model = SimpleUNet(
        in_channels=1,
        num_labels=1,
        num_filters_base=config.spine_localization.num_filters_base
    ).to(device)
    
    stage2_model = SpatialConfigurationNet(
        in_channels=1,
        num_labels=config.vertebrae_localization.num_landmarks,
        num_filters_base=config.vertebrae_localization.num_filters_base
    ).to(device)
    
    stage3_model = SegmentationUNet(
        in_channels=1 + config.vertebrae_segmentation.num_labels,
        num_classes=config.vertebrae_segmentation.num_labels + 1,
        num_filters_base=config.vertebrae_segmentation.num_filters_base
    ).to(device)
    
    # Load checkpoints
    logger.info("Loading checkpoints...")
    stage1_model = load_model(args.stage1_ckpt, stage1_model, device)
    stage2_model = load_model(args.stage2_ckpt, stage2_model, device)
    stage3_model = load_model(args.stage3_ckpt, stage3_model, device)
    
    logger.info("Models loaded successfully!")
    
    # Create inference pipeline
    pipeline = FullPipelineInference(
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        stage3_model=stage3_model,
        config=config,
        device=device
    )
    
    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        input_files = [input_path]
    else:
        # Process all NIfTI files in directory
        input_files = list(input_path.glob('*.nii.gz')) + list(input_path.glob('*.nii'))
    
    logger.info(f"Found {len(input_files)} input files")
    
    # Process each file
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for input_file in input_files:
        logger.info(f"Processing: {input_file.name}")
        
        try:
            # Load image
            image, spacing, affine = load_nifti(str(input_file))
            
            # Run pipeline
            results = pipeline.run(image, spacing)
            
            # Get outputs
            segmentation = results['segmentation']
            landmarks = results['landmarks']
            
            # Create output filename base
            base_name = input_file.stem.replace('.nii', '')
            
            # Save segmentation
            seg_nifti = nib.Nifti1Image(segmentation.astype(np.int16), affine)
            seg_path = output_dir / f'{base_name}_segmentation.nii.gz'
            nib.save(seg_nifti, str(seg_path))
            logger.info(f"  Saved segmentation: {seg_path.name}")
            
            # Save landmarks
            landmarks_json_path = output_dir / f'{base_name}_landmarks.json'
            landmarks_csv_path = output_dir / f'{base_name}_landmarks.csv'
            save_landmarks_json(landmarks, str(landmarks_json_path))
            save_landmarks_csv(landmarks, str(landmarks_csv_path))
            logger.info(f"  Saved landmarks: {landmarks_json_path.name}, {landmarks_csv_path.name}")
            
            # Count detected landmarks
            n_detected = sum(1 for v in landmarks.values() if v is not None)
            logger.info(f"  Detected {n_detected}/25 vertebrae")
            
        except Exception as e:
            logger.error(f"  Error processing {input_file.name}: {e}", exc_info=True)
    
    logger.info("Inference complete!")


def run_stage1_only(args, config: PipelineConfig, logger: logging.Logger):
    """Run Stage 1 only: Spine Localization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create and load model
    model = SimpleUNet(
        in_channels=1,
        num_labels=1,
        num_filters_base=config.spine_localization.num_filters_base
    ).to(device)
    model = load_model(args.stage1_ckpt, model, device)
    
    # Create inference
    inference = SpineLocalizationInference(
        model=model,
        config=config.spine_localization,
        device=device
    )
    
    # Process files
    input_path = Path(args.input)
    if input_path.is_file():
        input_files = [input_path]
    else:
        input_files = list(input_path.glob('*.nii.gz')) + list(input_path.glob('*.nii'))
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for input_file in input_files:
        logger.info(f"Processing: {input_file.name}")
        
        try:
            image, spacing, affine = load_nifti(str(input_file))
            spine_heatmap, spine_center = inference.run(image, spacing)
            
            base_name = input_file.stem.replace('.nii', '')
            
            # Save heatmap
            heatmap_nifti = nib.Nifti1Image(spine_heatmap.astype(np.float32), affine)
            heatmap_path = output_dir / f'{base_name}_spine_heatmap.nii.gz'
            nib.save(heatmap_nifti, str(heatmap_path))
            
            # Save spine center
            center_path = output_dir / f'{base_name}_spine_center.json'
            with open(center_path, 'w') as f:
                json.dump({
                    'spine_center': {
                        'X': float(spine_center[0]),
                        'Y': float(spine_center[1]),
                        'Z': float(spine_center[2])
                    }
                }, f, indent=2)
            
            logger.info(f"  Spine center: {spine_center}")
            
        except Exception as e:
            logger.error(f"  Error: {e}", exc_info=True)


def run_stage2_only(args, config: PipelineConfig, logger: logging.Logger):
    """Run Stage 2 only: Vertebrae Localization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create and load model
    model = SpatialConfigurationNet(
        in_channels=1,
        num_labels=config.vertebrae_localization.num_landmarks,
        num_filters_base=config.vertebrae_localization.num_filters_base
    ).to(device)
    model = load_model(args.stage2_ckpt, model, device)
    
    # Create inference
    inference = VertebraeLocalizationInference(
        model=model,
        config=config.vertebrae_localization,
        device=device
    )
    
    # Process files
    input_path = Path(args.input)
    if input_path.is_file():
        input_files = [input_path]
    else:
        input_files = list(input_path.glob('*.nii.gz')) + list(input_path.glob('*.nii'))
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for input_file in input_files:
        logger.info(f"Processing: {input_file.name}")
        
        try:
            image, spacing, affine = load_nifti(str(input_file))
            
            # Run inference (uses image center as default spine center)
            spine_center = np.array(image.shape) // 2
            landmarks, heatmaps = inference.run(image, spacing, spine_center)
            
            base_name = input_file.stem.replace('.nii', '')
            
            # Save landmarks
            landmarks_path = output_dir / f'{base_name}_landmarks.json'
            save_landmarks_json(landmarks, str(landmarks_path))
            
            n_detected = sum(1 for v in landmarks.values() if v is not None)
            logger.info(f"  Detected {n_detected}/25 vertebrae")
            
        except Exception as e:
            logger.error(f"  Error: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with VerSe2019 Vertebrae Segmentation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CT scan (NIfTI) or directory of scans')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output directory')
    
    # Stage selection
    parser.add_argument('--stage', type=str, default='all',
                        choices=['1', '2', '3', 'all'],
                        help='Which stage(s) to run')
    
    # Model checkpoints
    parser.add_argument('--stage1_ckpt', type=str,
                        help='Path to Stage 1 (Spine Localization) checkpoint')
    parser.add_argument('--stage2_ckpt', type=str,
                        help='Path to Stage 2 (Vertebrae Localization) checkpoint')
    parser.add_argument('--stage3_ckpt', type=str,
                        help='Path to Stage 3 (Vertebrae Segmentation) checkpoint')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    
    # Postprocessing options
    parser.add_argument('--no_postprocess', action='store_true',
                        help='Disable postprocessing of landmarks')
    parser.add_argument('--save_heatmaps', action='store_true',
                        help='Save intermediate heatmaps')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.stage == 'all':
        if not all([args.stage1_ckpt, args.stage2_ckpt, args.stage3_ckpt]):
            parser.error("--stage all requires all three checkpoint paths")
    elif args.stage == '1':
        if not args.stage1_ckpt:
            parser.error("--stage 1 requires --stage1_ckpt")
    elif args.stage == '2':
        if not args.stage2_ckpt:
            parser.error("--stage 2 requires --stage2_ckpt")
    elif args.stage == '3':
        if not all([args.stage2_ckpt, args.stage3_ckpt]):
            parser.error("--stage 3 requires --stage2_ckpt and --stage3_ckpt")
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output)
    
    logger.info("=" * 60)
    logger.info("VerSe2019 Vertebrae Segmentation Pipeline - Inference")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Stage: {args.stage}")
    
    # Create configuration
    config = PipelineConfig()
    
    try:
        if args.stage == 'all':
            run_full_pipeline(args, config, logger)
        elif args.stage == '1':
            run_stage1_only(args, config, logger)
        elif args.stage == '2':
            run_stage2_only(args, config, logger)
        elif args.stage == '3':
            # Stage 3 requires landmarks from Stage 2
            logger.warning("Stage 3 alone requires pre-computed landmarks. Running Stage 2+3.")
            run_full_pipeline(args, config, logger)
            
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
