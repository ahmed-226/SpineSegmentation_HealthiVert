#!/usr/bin/env python
"""
Evaluation Script for VerSe2019 Pipeline

Generates comprehensive evaluation metrics, results files, and visualization plots.

Usage:
    python evaluate.py --data_dir /path/to/data --output_dir /path/to/output --checkpoint_dir /path/to/checkpoints
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig
from src.models import SimpleUNet, SpatialConfigurationNet, SegmentationUNet
from src.data import (
    SpineLocalizationDataset,
    VertebraeLocalizationDataset,
    VertebraeSegmentationDataset,
    load_id_list,
    load_all_verse_landmarks
)
from src.evaluation import (
    MetricsLogger,
    LocalizationEvaluator,
    SegmentationEvaluator,
    ResultsVisualizer,
    generate_results_report,
    generate_all_plots
)


# Vertebrae names for visualization
VERTEBRAE_NAMES = [
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
    'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
    'L1', 'L2', 'L3', 'L4', 'L5', 'L6'
]

REGION_RANGES = {
    'Cervical': (0, 7),    # C1-C7
    'Thoracic': (7, 19),   # T1-T12
    'Lumbar': (19, 25)     # L1-L6
}


def compute_distance(pred_coords: np.ndarray, gt_coords: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    """Compute Euclidean distance between predicted and ground truth coordinates in mm"""
    if pred_coords is None or gt_coords is None:
        return np.nan
    spacing = np.array(spacing)
    diff = (pred_coords - gt_coords) * spacing
    return np.sqrt(np.sum(diff ** 2))


def extract_landmarks_from_heatmap(heatmap: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Extract landmark coordinates from heatmap using argmax"""
    coords = []
    for i in range(heatmap.shape[0]):
        hm = heatmap[i]
        if hm.max() > threshold:
            idx = np.unravel_index(np.argmax(hm), hm.shape)
            coords.append(np.array(idx))
        else:
            coords.append(None)
    return coords


def evaluate_localization(
    model,
    dataset,
    device: torch.device,
    spacing: Tuple[float, float, float],
    num_landmarks: int = 25
) -> Dict:
    """Evaluate localization model on dataset"""
    model.eval()
    
    all_distances = {i: [] for i in range(num_landmarks)}
    identification_counts = {
        2: {i: 0 for i in range(num_landmarks)},
        4: {i: 0 for i in range(num_landmarks)},
        10: {i: 0 for i in range(num_landmarks)},
        20: {i: 0 for i in range(num_landmarks)}
    }
    total_per_vertebra = {i: 0 for i in range(num_landmarks)}
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            
            # Forward pass
            pred_heatmap = model(image)
            if isinstance(pred_heatmap, tuple):
                pred_heatmap = pred_heatmap[0]
            pred_heatmap = pred_heatmap.cpu().numpy()[0]
            
            # Extract landmarks
            pred_coords = extract_landmarks_from_heatmap(pred_heatmap)
            
            # Get ground truth coordinates
            if 'heatmap' in sample:
                # If dataset returns heatmap (Stage 1)
                gt_heatmap = sample['heatmap'].numpy()
                gt_coords = extract_landmarks_from_heatmap(gt_heatmap)
            elif 'landmarks' in sample:
                # If dataset returns coordinates directly (Stage 2)
                landmarks = sample['landmarks'].numpy()
                gt_coords = []
                for i in range(num_landmarks):
                    if i < len(landmarks) and landmarks[i, 0] > 0.5:  # Check validity flag
                        gt_coords.append(landmarks[i, 1:4])
                    else:
                        gt_coords.append(None)
            else:
                print(f"Warning: Sample keys: {sample.keys()}")
                continue  # Skip if no ground truth available
            
            # Compute distances
            for i in range(num_landmarks):
                if gt_coords[i] is not None:
                    total_per_vertebra[i] += 1
                    
                    if pred_coords[i] is not None:
                        dist = compute_distance(pred_coords[i], gt_coords[i], spacing)
                        all_distances[i].append(dist)
                        
                        # Count identifications at different thresholds
                        for thresh in [2, 4, 10, 20]:
                            if dist <= thresh:
                                identification_counts[thresh][i] += 1
    
    # Compute per-vertebra statistics
    vertebra_stats = {}
    for i in range(num_landmarks):
        distances = all_distances[i]
        if distances:
            vertebra_stats[i] = {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'median': float(np.median(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'count': len(distances),
                'id_rate_2mm': identification_counts[2][i] / total_per_vertebra[i] * 100 if total_per_vertebra[i] > 0 else 0,
                'id_rate_4mm': identification_counts[4][i] / total_per_vertebra[i] * 100 if total_per_vertebra[i] > 0 else 0,
                'id_rate_10mm': identification_counts[10][i] / total_per_vertebra[i] * 100 if total_per_vertebra[i] > 0 else 0,
                'id_rate_20mm': identification_counts[20][i] / total_per_vertebra[i] * 100 if total_per_vertebra[i] > 0 else 0
            }
    
    # Compute overall statistics
    all_dists = [d for dists in all_distances.values() for d in dists]
    
    return {
        'per_vertebra': vertebra_stats,
        'overall': {
            'mean': float(np.mean(all_dists)) if all_dists else 0,
            'std': float(np.std(all_dists)) if all_dists else 0,
            'median': float(np.median(all_dists)) if all_dists else 0
        }
    }


def compute_dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice score between prediction and ground truth"""
    intersection = np.sum(pred * gt)
    if pred.sum() + gt.sum() == 0:
        return 1.0  # Both empty
    return 2 * intersection / (pred.sum() + gt.sum())


def evaluate_segmentation(
    model,
    dataset,
    device: torch.device,
    num_classes: int = 26
) -> Dict:
    """Evaluate segmentation model on dataset"""
    model.eval()
    
    dice_scores = {i: [] for i in range(num_classes)}  # 0 = background
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            gt_mask = sample['label'].numpy()
            
            # Forward pass
            pred_logits = model(image)
            if isinstance(pred_logits, tuple):
                pred_logits = pred_logits[0]
            pred_mask = torch.argmax(pred_logits, dim=1).cpu().numpy()[0]
            
            # Compute Dice for each class
            for c in range(num_classes):
                pred_c = (pred_mask == c).astype(float)
                gt_c = (gt_mask == c).astype(float)
                
                # Only compute if ground truth has this class
                if gt_c.sum() > 0:
                    dice = compute_dice_score(pred_c, gt_c)
                    dice_scores[c].append(dice)
    
    # Compute per-class statistics
    class_stats = {}
    for c in range(num_classes):
        scores = dice_scores[c]
        if scores:
            class_stats[c] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'count': len(scores)
            }
    
    # Overall (excluding background)
    all_scores = [s for c, scores in dice_scores.items() if c > 0 for s in scores]
    
    return {
        'per_class': class_stats,
        'overall': {
            'mean': float(np.mean(all_scores)) if all_scores else 0,
            'std': float(np.std(all_scores)) if all_scores else 0,
            'min': float(np.min(all_scores)) if all_scores else 0,
            'max': float(np.max(all_scores)) if all_scores else 0
        }
    }


def plot_localization_boxplot(results: Dict, output_dir: Path, fold_idx: Optional[int] = None):
    """Generate boxplot for localization distances per vertebra"""
    vertebra_stats = results.get('per_vertebra', {})
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    data = []
    labels = []
    for i in range(25):
        if i in vertebra_stats and vertebra_stats[i]['count'] > 0:
            # Create pseudo-distribution from statistics
            mean = vertebra_stats[i]['mean']
            std = vertebra_stats[i]['std']
            n = min(vertebra_stats[i]['count'], 50)
            samples = np.random.normal(mean, std, n)
            samples = np.clip(samples, 0, vertebra_stats[i]['max'])
            data.append(samples)
            labels.append(VERTEBRAE_NAMES[i])
    
    if data:
        bp = ax.boxplot(data, patch_artist=True, labels=labels)
        
        # Color by region
        colors = []
        for label in labels:
            if label.startswith('C'):
                colors.append('#FF6B6B')  # Cervical - red
            elif label.startswith('T'):
                colors.append('#4ECDC4')  # Thoracic - teal
            else:
                colors.append('#45B7D1')  # Lumbar - blue
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Vertebra', fontsize=12)
        ax.set_ylabel('Distance Error (mm)', fontsize=12)
        title = 'Localization Error per Vertebra'
        if fold_idx is not None:
            title += f' (Fold {fold_idx})'
        ax.set_title(title, fontsize=14)
        ax.axhline(y=4, color='r', linestyle='--', alpha=0.5, label='4mm threshold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        suffix = f'_fold{fold_idx}' if fold_idx is not None else ''
        plt.savefig(output_dir / f'localization_boxplot{suffix}.png', dpi=150)
        plt.close()


def plot_identification_rates(results: Dict, output_dir: Path, fold_idx: Optional[int] = None):
    """Generate bar chart for identification rates at different thresholds"""
    vertebra_stats = results.get('per_vertebra', {})
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = []
    rates_2mm = []
    rates_4mm = []
    rates_10mm = []
    rates_20mm = []
    
    for i in range(25):
        if i in vertebra_stats:
            x.append(VERTEBRAE_NAMES[i])
            rates_2mm.append(vertebra_stats[i]['id_rate_2mm'])
            rates_4mm.append(vertebra_stats[i]['id_rate_4mm'])
            rates_10mm.append(vertebra_stats[i]['id_rate_10mm'])
            rates_20mm.append(vertebra_stats[i]['id_rate_20mm'])
    
    if x:
        x_pos = np.arange(len(x))
        width = 0.2
        
        ax.bar(x_pos - 1.5*width, rates_2mm, width, label='≤2mm', color='#FF6B6B', alpha=0.8)
        ax.bar(x_pos - 0.5*width, rates_4mm, width, label='≤4mm', color='#4ECDC4', alpha=0.8)
        ax.bar(x_pos + 0.5*width, rates_10mm, width, label='≤10mm', color='#45B7D1', alpha=0.8)
        ax.bar(x_pos + 1.5*width, rates_20mm, width, label='≤20mm', color='#96CEB4', alpha=0.8)
        
        ax.set_xlabel('Vertebra', fontsize=12)
        ax.set_ylabel('Identification Rate (%)', fontsize=12)
        title = 'Vertebra Identification Rates at Different Thresholds'
        if fold_idx is not None:
            title += f' (Fold {fold_idx})'
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        suffix = f'_fold{fold_idx}' if fold_idx is not None else ''
        plt.savefig(output_dir / f'identification_rates{suffix}.png', dpi=150)
        plt.close()


def plot_segmentation_dice(results: Dict, output_dir: Path, fold_idx: Optional[int] = None):
    """Generate bar chart for Dice scores per vertebra"""
    class_stats = results.get('per_class', {})
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = []
    dice_means = []
    dice_stds = []
    
    for i in range(1, 26):  # Skip background
        if i in class_stats:
            x.append(VERTEBRAE_NAMES[i-1])
            dice_means.append(class_stats[i]['mean'] * 100)  # Convert to percentage
            dice_stds.append(class_stats[i]['std'] * 100)
    
    if x:
        x_pos = np.arange(len(x))
        
        # Color by region
        colors = []
        for label in x:
            if label.startswith('C'):
                colors.append('#FF6B6B')  # Cervical
            elif label.startswith('T'):
                colors.append('#4ECDC4')  # Thoracic
            else:
                colors.append('#45B7D1')  # Lumbar
        
        bars = ax.bar(x_pos, dice_means, yerr=dice_stds, capsize=3, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Vertebra', fontsize=12)
        ax.set_ylabel('Dice Score (%)', fontsize=12)
        title = 'Segmentation Dice Score per Vertebra'
        if fold_idx is not None:
            title += f' (Fold {fold_idx})'
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='80% target')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        suffix = f'_fold{fold_idx}' if fold_idx is not None else ''
        plt.savefig(output_dir / f'dice_scores{suffix}.png', dpi=150)
        plt.close()


def plot_region_comparison(loc_results: Dict, seg_results: Dict, output_dir: Path, fold_idx: Optional[int] = None):
    """Generate region-wise comparison plot"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Localization - Mean distance per region
    ax1 = axes[0]
    regions = ['Cervical', 'Thoracic', 'Lumbar']
    loc_means = []
    loc_stds = []
    
    vertebra_stats = loc_results.get('per_vertebra', {})
    for region, (start, end) in REGION_RANGES.items():
        distances = []
        for i in range(start, end):
            if i in vertebra_stats:
                distances.append(vertebra_stats[i]['mean'])
        if distances:
            loc_means.append(np.mean(distances))
            loc_stds.append(np.std(distances))
        else:
            loc_means.append(0)
            loc_stds.append(0)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax1.bar(regions, loc_means, yerr=loc_stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Mean Distance Error (mm)', fontsize=12)
    ax1.set_title('Localization Error by Region', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # Segmentation - Mean Dice per region
    ax2 = axes[1]
    seg_means = []
    seg_stds = []
    
    class_stats = seg_results.get('per_class', {})
    for region, (start, end) in REGION_RANGES.items():
        scores = []
        for i in range(start + 1, end + 1):  # +1 because class 0 is background
            if i in class_stats:
                scores.append(class_stats[i]['mean'] * 100)
        if scores:
            seg_means.append(np.mean(scores))
            seg_stds.append(np.std(scores))
        else:
            seg_means.append(0)
            seg_stds.append(0)
    
    ax2.bar(regions, seg_means, yerr=seg_stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Mean Dice Score (%)', fontsize=12)
    ax2.set_title('Segmentation Dice by Region', fontsize=14)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    suffix = f'_fold{fold_idx}' if fold_idx is not None else ''
    plt.savefig(output_dir / f'region_comparison{suffix}.png', dpi=150)
    plt.close()


def plot_cross_validation_summary(all_fold_results: Dict, output_dir: Path):
    """Generate cross-validation summary plots"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Localization across folds
    ax1 = axes[0]
    folds = sorted(all_fold_results.keys())
    loc_means = []
    loc_stds = []
    
    for fold in folds:
        if 'localization' in all_fold_results[fold]:
            overall = all_fold_results[fold]['localization']['overall']
            loc_means.append(overall['mean'])
            loc_stds.append(overall['std'])
    
    if loc_means:
        x_pos = np.arange(len(folds))
        ax1.bar(x_pos, loc_means, yerr=loc_stds, capsize=5, color='#4ECDC4', alpha=0.8)
        ax1.axhline(y=np.mean(loc_means), color='r', linestyle='--', label=f'Mean: {np.mean(loc_means):.2f}mm')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'Fold {f}' for f in folds])
        ax1.set_ylabel('Mean Distance Error (mm)', fontsize=12)
        ax1.set_title('Localization Error Across Folds', fontsize=14)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # Segmentation across folds
    ax2 = axes[1]
    seg_means = []
    seg_stds = []
    
    for fold in folds:
        if 'segmentation' in all_fold_results[fold]:
            overall = all_fold_results[fold]['segmentation']['overall']
            seg_means.append(overall['mean'] * 100)
            seg_stds.append(overall['std'] * 100)
    
    if seg_means:
        x_pos = np.arange(len(folds))
        ax2.bar(x_pos, seg_means, yerr=seg_stds, capsize=5, color='#45B7D1', alpha=0.8)
        ax2.axhline(y=np.mean(seg_means), color='r', linestyle='--', label=f'Mean: {np.mean(seg_means):.1f}%')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'Fold {f}' for f in folds])
        ax2.set_ylabel('Mean Dice Score (%)', fontsize=12)
        ax2.set_title('Segmentation Dice Across Folds', fontsize=14)
        ax2.set_ylim(0, 105)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_validation_summary.png', dpi=150)
    plt.close()


def generate_text_report(results: Dict, output_path: Path, experiment_name: str):
    """Generate comprehensive text report"""
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"VerSe2019 Pipeline Evaluation Report\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # Localization results
        if 'localization' in results:
            loc = results['localization']
            f.write("LOCALIZATION RESULTS\n")
            f.write("-" * 50 + "\n\n")
            
            # Overall metrics
            overall = loc.get('overall', {})
            f.write("Overall Metrics:\n")
            f.write(f"  Mean Distance Error: {overall.get('mean', 0):.2f} ± {overall.get('std', 0):.2f} mm\n")
            f.write(f"  Median Distance Error: {overall.get('median', 0):.2f} mm\n\n")
            
            # Per-vertebra table
            f.write("Per-Vertebra Metrics:\n")
            f.write(f"{'Vertebra':<10} {'Mean (mm)':<12} {'Std (mm)':<12} {'ID@2mm':<10} {'ID@4mm':<10} {'ID@20mm':<10}\n")
            f.write("-" * 64 + "\n")
            
            vertebra_stats = loc.get('per_vertebra', {})
            for i in range(25):
                if i in vertebra_stats:
                    stats = vertebra_stats[i]
                    f.write(f"{VERTEBRAE_NAMES[i]:<10} {stats['mean']:>10.2f} {stats['std']:>10.2f} "
                           f"{stats['id_rate_2mm']:>8.1f}% {stats['id_rate_4mm']:>8.1f}% {stats['id_rate_20mm']:>8.1f}%\n")
            
            # Region summary
            f.write("\nRegion Summary:\n")
            for region, (start, end) in REGION_RANGES.items():
                distances = [vertebra_stats[i]['mean'] for i in range(start, end) if i in vertebra_stats]
                if distances:
                    f.write(f"  {region}: Mean = {np.mean(distances):.2f} mm, Std = {np.std(distances):.2f} mm\n")
            
            f.write("\n")
        
        # Segmentation results
        if 'segmentation' in results:
            seg = results['segmentation']
            f.write("\nSEGMENTATION RESULTS\n")
            f.write("-" * 50 + "\n\n")
            
            # Overall metrics
            overall = seg.get('overall', {})
            f.write("Overall Metrics:\n")
            f.write(f"  Mean Dice Score: {overall.get('mean', 0)*100:.2f} ± {overall.get('std', 0)*100:.2f}%\n")
            f.write(f"  Min Dice Score: {overall.get('min', 0)*100:.2f}%\n")
            f.write(f"  Max Dice Score: {overall.get('max', 0)*100:.2f}%\n\n")
            
            # Per-class table
            f.write("Per-Vertebra Dice Scores:\n")
            f.write(f"{'Vertebra':<10} {'Mean (%)':<12} {'Std (%)':<12} {'Min (%)':<12} {'Max (%)':<12}\n")
            f.write("-" * 58 + "\n")
            
            class_stats = seg.get('per_class', {})
            for i in range(1, 26):  # Skip background
                if i in class_stats:
                    stats = class_stats[i]
                    f.write(f"{VERTEBRAE_NAMES[i-1]:<10} {stats['mean']*100:>10.2f} {stats['std']*100:>10.2f} "
                           f"{stats['min']*100:>10.2f} {stats['max']*100:>10.2f}\n")
            
            # Region summary
            f.write("\nRegion Summary:\n")
            for region, (start, end) in REGION_RANGES.items():
                scores = [class_stats[i]['mean']*100 for i in range(start+1, end+1) if i in class_stats]
                if scores:
                    f.write(f"  {region}: Mean Dice = {np.mean(scores):.2f}%, Std = {np.std(scores):.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("End of Report\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VerSe2019 Pipeline Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory (contains trained models)')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['2', '3', 'all'],
                        help='Which stage(s) to evaluate')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds trained')
    parser.add_argument('--fold', type=int, default=None,
                        help='Evaluate specific fold only')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--experiment_name', type=str, default='verse2019',
                        help='Experiment name for reports')
    parser.add_argument('--splits_dir', type=str, default=None,
                        help='Directory containing train.txt/val.txt')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / 'plots'
    reports_dir = output_dir / 'reports'
    
    # Create parent directories if they don't exist
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = PipelineConfig()
    
    # Load subject IDs for evaluation
    if args.splits_dir:
        val_file = Path(args.splits_dir) / 'val.txt'
    else:
        # Check current directory/output directory first
        val_file = Path('.') / 'val.txt'
        if not val_file.exists():
            val_file = Path(args.data_dir) / 'val.txt'

    if val_file.exists():
        val_ids = load_id_list(str(val_file))
    else:
        print(f"Warning: No val.txt found at {val_file}. Please specify validation IDs.")
        # If no validation file, maybe we try test.txt?
        test_file = Path('.') / 'test.txt'
        if not test_file.exists() and args.splits_dir:
             test_file = Path(args.splits_dir) / 'test.txt'
        
        if test_file.exists():
            print(f"Loading test IDs from {test_file}")
            val_ids = load_id_list(str(test_file))
        else:
            return
    
    print(f"Evaluating on {len(val_ids)} subjects")
    
    # Load landmarks
    landmarks_dict = load_all_verse_landmarks(args.data_dir, val_ids)
    
    # Determine folds to evaluate
    folds_to_eval = [args.fold] if args.fold is not None else list(range(args.n_folds))
    
    all_fold_results = {}
    
    for fold_idx in folds_to_eval:
        print(f"\n{'='*50}")
        print(f"Evaluating Fold {fold_idx}")
        print(f"{'='*50}")
        
        fold_results = {}
        
        # Stage 2: Vertebrae Localization
        if args.stage in ['2', 'all']:
            # Check standard path
            # Structure: stage2/fold_X/checkpoints/fold_X/model.pth
            checkpoint_dir = output_dir / 'stage2' / f'fold_{fold_idx}' / 'checkpoints' / f'fold_{fold_idx}'
            
            print(f"DEBUG: Looking for Stage 2 checkpoint in {checkpoint_dir}")
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            
            # Fallback to final model if best doesn't exist
            if not checkpoint_path.exists():
                print("DEBUG: best_model.pth not found, checking final_model.pth")
                checkpoint_path = checkpoint_dir / 'final_model.pth'
            
            if checkpoint_path.exists():
                print(f"Loading Stage 2 model from {checkpoint_path}")
                
                model = SpatialConfigurationNet(
                    in_channels=1,
                    num_labels=config.vertebrae_localization.num_landmarks,
                    num_filters_base=config.vertebrae_localization.num_filters_base
                ).to(device)
                
                # Load checkpoint which contains more than just weights
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                dataset = VertebraeLocalizationDataset(
                    data_folder=args.data_dir,
                    id_list=val_ids,
                    landmarks_dict=landmarks_dict,
                    image_size=config.vertebrae_localization.image_size,
                    image_spacing=config.vertebrae_localization.image_spacing,
                    num_landmarks=config.vertebrae_localization.num_landmarks,
                    heatmap_sigma=config.vertebrae_localization.heatmap_sigma,
                    is_training=False
                )
                
                print("Evaluating localization...")
                loc_results = evaluate_localization(
                    model, dataset, device, 
                    config.vertebrae_localization.image_spacing,
                    config.vertebrae_localization.num_landmarks
                )
                fold_results['localization'] = loc_results
                
                print(f"  Mean distance error: {loc_results['overall']['mean']:.2f} mm")
                
                # Generate plots
                plot_localization_boxplot(loc_results, plots_dir, fold_idx)
                plot_identification_rates(loc_results, plots_dir, fold_idx)
            else:
                print(f"Stage 2 checkpoint not found: {checkpoint_path}")
        
        # Stage 3: Segmentation
        if args.stage in ['3', 'all']:
            # Check standard path
            # Structure: stage3/fold_X/checkpoints/fold_X/model.pth
            checkpoint_dir = output_dir / 'stage3' / f'fold_{fold_idx}' / 'checkpoints' / f'fold_{fold_idx}'
            
            print(f"DEBUG: Looking for Stage 3 checkpoint in {checkpoint_dir}")
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            
            # Fallback to final model if best doesn't exist
            if not checkpoint_path.exists():
                print("DEBUG: best_model.pth not found, checking final_model.pth")
                checkpoint_path = checkpoint_dir / 'final_model.pth'
            
            if checkpoint_path.exists():
                print(f"Loading Stage 3 model from {checkpoint_path}")
                
                model = SegmentationUNet(
                    in_channels=1,
                    num_classes=config.vertebrae_segmentation.num_labels + 1,
                    num_filters_base=config.vertebrae_segmentation.num_filters_base
                ).to(device)
                
                # Load checkpoint which contains more than just weights
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                dataset = VertebraeSegmentationDataset(
                    data_folder=args.data_dir,
                    id_list=val_ids,
                    landmarks_dict=landmarks_dict,
                    labels_folder=args.data_dir,
                    image_size=config.vertebrae_segmentation.image_size,
                    image_spacing=config.vertebrae_segmentation.image_spacing,
                    num_classes=config.vertebrae_segmentation.num_labels,
                    is_training=False
                )
                
                print("Evaluating segmentation...")
                seg_results = evaluate_segmentation(
                    model, dataset, device,
                    config.vertebrae_segmentation.num_labels + 1
                )
                fold_results['segmentation'] = seg_results
                
                print(f"  Mean Dice score: {seg_results['overall']['mean']*100:.2f}%")
                
                # Generate plots
                plot_segmentation_dice(seg_results, plots_dir, fold_idx)
            else:
                print(f"Stage 3 checkpoint not found: {checkpoint_path}")
        
        # Region comparison if both available
        if 'localization' in fold_results and 'segmentation' in fold_results:
            plot_region_comparison(fold_results['localization'], fold_results['segmentation'], plots_dir, fold_idx)
        
        # Generate text report for fold
        if fold_results:
            report_path = reports_dir / f'results_fold{fold_idx}.txt'
            generate_text_report(fold_results, report_path, f"{args.experiment_name}_fold{fold_idx}")
            print(f"Report saved to: {report_path}")
        
        all_fold_results[fold_idx] = fold_results
    
    # Cross-validation summary
    if len(all_fold_results) > 1:
        print(f"\n{'='*50}")
        print("Cross-Validation Summary")
        print(f"{'='*50}")
        
        plot_cross_validation_summary(all_fold_results, plots_dir)
        
        # Overall summary report
        summary_path = reports_dir / f'{args.experiment_name}_cv_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Cross-Validation Summary - {args.experiment_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Number of folds: {len(all_fold_results)}\n\n")
            
            # Localization summary
            loc_means = []
            for fold, results in all_fold_results.items():
                if 'localization' in results:
                    loc_means.append(results['localization']['overall']['mean'])
            
            if loc_means:
                f.write("Localization:\n")
                f.write(f"  Mean ± Std across folds: {np.mean(loc_means):.2f} ± {np.std(loc_means):.2f} mm\n\n")
            
            # Segmentation summary
            seg_means = []
            for fold, results in all_fold_results.items():
                if 'segmentation' in results:
                    seg_means.append(results['segmentation']['overall']['mean'] * 100)
            
            if seg_means:
                f.write("Segmentation:\n")
                f.write(f"  Mean Dice ± Std across folds: {np.mean(seg_means):.2f} ± {np.std(seg_means):.2f}%\n")
        
        print(f"CV Summary saved to: {summary_path}")
    
    # Save all results as JSON
    json_path = reports_dir / f'{args.experiment_name}_results.json'
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    with open(json_path, 'w') as f:
        json.dump(convert_numpy(all_fold_results), f, indent=2)
    
    print(f"\nAll results saved to: {json_path}")
    print(f"Plots saved to: {plots_dir}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
