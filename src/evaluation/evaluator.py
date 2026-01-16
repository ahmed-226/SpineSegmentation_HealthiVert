"""
Evaluation and Visualization utilities for VerSe2019 Pipeline
Generates metrics, plots, and reports for localization and segmentation results.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime


# Vertebrae names for labeling
VERTEBRAE_NAMES = [
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
    'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
    'L1', 'L2', 'L3', 'L4', 'L5', 'L6'
]

REGION_INDICES = {
    'Cervical': list(range(0, 7)),      # C1-C7
    'Thoracic': list(range(7, 19)),     # T1-T12
    'Lumbar': list(range(19, 25))       # L1-L6
}


class MetricsLogger:
    """Logger for tracking training and validation metrics across folds."""
    
    def __init__(self, output_dir: str, experiment_name: str = "experiment"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Storage for metrics
        self.fold_metrics = {}  # {fold_idx: {stage: {metric_name: [values]}}}
        self.final_metrics = {}  # {fold_idx: {stage: {metric_name: value}}}
        
    def log_iteration(self, fold: int, stage: str, iteration: int, metrics: Dict[str, float]):
        """Log metrics for a training iteration."""
        if fold not in self.fold_metrics:
            self.fold_metrics[fold] = {}
        if stage not in self.fold_metrics[fold]:
            self.fold_metrics[fold][stage] = {'iterations': [], 'metrics': {}}
        
        self.fold_metrics[fold][stage]['iterations'].append(iteration)
        for name, value in metrics.items():
            if name not in self.fold_metrics[fold][stage]['metrics']:
                self.fold_metrics[fold][stage]['metrics'][name] = []
            self.fold_metrics[fold][stage]['metrics'][name].append(value)
    
    def log_final(self, fold: int, stage: str, metrics: Dict[str, float]):
        """Log final metrics for a fold."""
        if fold not in self.final_metrics:
            self.final_metrics[fold] = {}
        self.final_metrics[fold][stage] = metrics
    
    def save_to_file(self, filename: str = None):
        """Save all metrics to JSON file."""
        if filename is None:
            filename = f"{self.experiment_name}_metrics.json"
        
        output_path = self.output_dir / filename
        
        # Convert to serializable format
        data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'fold_metrics': self.fold_metrics,
            'final_metrics': self.final_metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        print(f"Saved metrics to {output_path}")
        return output_path


class LocalizationEvaluator:
    """Evaluator for vertebrae localization results."""
    
    def __init__(
        self,
        spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        distance_thresholds: List[float] = [2.0, 4.0, 10.0, 20.0]
    ):
        self.spacing = np.array(spacing)
        self.distance_thresholds = distance_thresholds
    
    def compute_metrics(
        self,
        predicted_landmarks: np.ndarray,
        target_landmarks: np.ndarray,
        valid_mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute localization metrics.
        
        Args:
            predicted_landmarks: [num_landmarks, 3] predicted coordinates
            target_landmarks: [num_landmarks, 3] target coordinates
            valid_mask: [num_landmarks] boolean mask for valid landmarks
            
        Returns:
            Dictionary of metrics
        """
        if valid_mask is None:
            valid_mask = np.ones(len(predicted_landmarks), dtype=bool)
        
        # Compute distances in mm
        diff = (predicted_landmarks - target_landmarks) * self.spacing
        distances = np.linalg.norm(diff, axis=1)
        valid_distances = distances[valid_mask]
        
        metrics = {
            'mean_distance_mm': float(np.mean(valid_distances)),
            'median_distance_mm': float(np.median(valid_distances)),
            'std_distance_mm': float(np.std(valid_distances)),
            'max_distance_mm': float(np.max(valid_distances)),
            'min_distance_mm': float(np.min(valid_distances)),
            'num_valid': int(valid_mask.sum())
        }
        
        # Identification rates at different thresholds
        for thresh in self.distance_thresholds:
            correct = valid_distances < thresh
            metrics[f'id_rate_{thresh}mm'] = float(np.mean(correct))
        
        # Per-region metrics
        for region_name, indices in REGION_INDICES.items():
            region_mask = np.zeros(len(distances), dtype=bool)
            for idx in indices:
                if idx < len(distances):
                    region_mask[idx] = True
            region_valid = valid_mask & region_mask
            
            if region_valid.sum() > 0:
                region_distances = distances[region_valid]
                metrics[f'{region_name.lower()}_mean_mm'] = float(np.mean(region_distances))
                metrics[f'{region_name.lower()}_id_rate_20mm'] = float(np.mean(region_distances < 20.0))
        
        # Per-vertebra distances
        per_vertebra = {}
        for i, name in enumerate(VERTEBRAE_NAMES):
            if i < len(distances) and valid_mask[i]:
                per_vertebra[name] = float(distances[i])
        metrics['per_vertebra_distances'] = per_vertebra
        
        return metrics
    
    def aggregate_fold_metrics(
        self,
        fold_metrics: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across folds.
        
        Args:
            fold_metrics: List of metric dictionaries, one per fold
            
        Returns:
            Dictionary with mean, std for each metric
        """
        # Collect scalar metrics
        scalar_metrics = {}
        for fold_metric in fold_metrics:
            for key, value in fold_metric.items():
                if isinstance(value, (int, float)) and not key.startswith('per_'):
                    if key not in scalar_metrics:
                        scalar_metrics[key] = []
                    scalar_metrics[key].append(value)
        
        # Compute statistics
        aggregated = {}
        for key, values in scalar_metrics.items():
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': [float(v) for v in values]
            }
        
        return aggregated


class SegmentationEvaluator:
    """Evaluator for vertebrae segmentation results."""
    
    def __init__(self, num_classes: int = 26):
        self.num_classes = num_classes
    
    def compute_dice(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        smooth: float = 1e-7
    ) -> Dict[str, float]:
        """
        Compute Dice scores for segmentation.
        
        Args:
            prediction: 3D array of predicted labels
            target: 3D array of target labels
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dictionary of Dice scores
        """
        dice_scores = {}
        all_dice = []
        
        for c in range(1, self.num_classes):  # Skip background
            pred_c = (prediction == c)
            target_c = (target == c)
            
            intersection = np.sum(pred_c & target_c)
            union = np.sum(pred_c) + np.sum(target_c)
            
            if union > 0:
                dice = (2 * intersection + smooth) / (union + smooth)
                dice_scores[VERTEBRAE_NAMES[c-1] if c <= len(VERTEBRAE_NAMES) else f'class_{c}'] = float(dice)
                all_dice.append(dice)
        
        # Aggregate metrics
        if all_dice:
            dice_scores['mean_dice'] = float(np.mean(all_dice))
            dice_scores['std_dice'] = float(np.std(all_dice))
            dice_scores['min_dice'] = float(np.min(all_dice))
            dice_scores['max_dice'] = float(np.max(all_dice))
        
        # Per-region Dice
        for region_name, indices in REGION_INDICES.items():
            region_dice = []
            for idx in indices:
                c = idx + 1  # Class label
                if c < self.num_classes:
                    pred_c = (prediction == c)
                    target_c = (target == c)
                    
                    intersection = np.sum(pred_c & target_c)
                    union = np.sum(pred_c) + np.sum(target_c)
                    
                    if union > 0:
                        dice = (2 * intersection + smooth) / (union + smooth)
                        region_dice.append(dice)
            
            if region_dice:
                dice_scores[f'{region_name.lower()}_mean_dice'] = float(np.mean(region_dice))
        
        return dice_scores
    
    def aggregate_fold_metrics(
        self,
        fold_metrics: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate Dice metrics across folds."""
        scalar_metrics = {}
        for fold_metric in fold_metrics:
            for key, value in fold_metric.items():
                if isinstance(value, (int, float)):
                    if key not in scalar_metrics:
                        scalar_metrics[key] = []
                    scalar_metrics[key].append(value)
        
        aggregated = {}
        for key, values in scalar_metrics.items():
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values]
            }
        
        return aggregated


class ResultsVisualizer:
    """Generate plots and visualizations for results."""
    
    def __init__(self, output_dir: str, dpi: int = 150):
        self.output_dir = Path(output_dir) / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_training_curves(
        self,
        metrics_history: Dict[str, List[float]],
        iterations: List[int],
        title: str = "Training Curves",
        filename: str = "training_curves.png"
    ):
        """Plot training loss and other metrics over iterations."""
        fig, axes = plt.subplots(1, len(metrics_history), figsize=(5*len(metrics_history), 4))
        if len(metrics_history) == 1:
            axes = [axes]
        
        for ax, (name, values) in zip(axes, metrics_history.items()):
            ax.plot(iterations, values, 'b-', linewidth=1.5)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(name.replace('_', ' ').title())
            ax.set_title(f'{name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_fold_comparison_bars(
        self,
        fold_metrics: Dict[int, Dict[str, float]],
        metric_names: List[str],
        title: str = "Fold Comparison",
        filename: str = "fold_comparison.png"
    ):
        """Create bar chart comparing metrics across folds."""
        n_folds = len(fold_metrics)
        n_metrics = len(metric_names)
        
        fig, ax = plt.subplots(figsize=(max(8, n_folds * 2), 6))
        
        x = np.arange(n_folds)
        width = 0.8 / n_metrics
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
        
        for i, metric in enumerate(metric_names):
            values = [fold_metrics[fold].get(metric, 0) for fold in sorted(fold_metrics.keys())]
            offset = (i - n_metrics/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title(), color=colors[i])
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i}' for i in sorted(fold_metrics.keys())])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_localization_boxplot(
        self,
        per_vertebra_distances: Dict[str, List[float]],
        title: str = "Localization Error per Vertebra",
        filename: str = "localization_boxplot.png"
    ):
        """Create boxplot of localization errors per vertebra."""
        # Organize data by vertebra
        vertebra_names = []
        data = []
        
        for name in VERTEBRAE_NAMES:
            if name in per_vertebra_distances and per_vertebra_distances[name]:
                vertebra_names.append(name)
                data.append(per_vertebra_distances[name])
        
        if not data:
            print("No localization data available for boxplot")
            return
        
        fig, ax = plt.subplots(figsize=(max(12, len(vertebra_names) * 0.5), 6))
        
        # Create boxplot with colors by region
        bp = ax.boxplot(data, patch_artist=True, labels=vertebra_names)
        
        # Color by region
        colors = {'C': '#FF6B6B', 'T': '#4ECDC4', 'L': '#45B7D1'}
        for i, (box, name) in enumerate(zip(bp['boxes'], vertebra_names)):
            region = name[0]
            box.set_facecolor(colors.get(region, 'gray'))
            box.set_alpha(0.7)
        
        ax.set_xlabel('Vertebra')
        ax.set_ylabel('Distance Error (mm)')
        ax.set_title(title)
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20mm threshold')
        ax.legend()
        
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_segmentation_dice_bars(
        self,
        dice_scores: Dict[str, float],
        title: str = "Dice Scores per Vertebra",
        filename: str = "segmentation_dice_bars.png"
    ):
        """Create bar chart of Dice scores per vertebra."""
        # Filter to per-vertebra scores
        vertebra_dice = {k: v for k, v in dice_scores.items() if k in VERTEBRAE_NAMES}
        
        if not vertebra_dice:
            print("No Dice data available for bar chart")
            return
        
        # Sort by vertebra order
        sorted_names = [n for n in VERTEBRAE_NAMES if n in vertebra_dice]
        sorted_values = [vertebra_dice[n] for n in sorted_names]
        
        fig, ax = plt.subplots(figsize=(max(12, len(sorted_names) * 0.5), 6))
        
        # Color by region
        colors = []
        color_map = {'C': '#FF6B6B', 'T': '#4ECDC4', 'L': '#45B7D1'}
        for name in sorted_names:
            colors.append(color_map.get(name[0], 'gray'))
        
        bars = ax.bar(sorted_names, sorted_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, sorted_values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7, rotation=90)
        
        ax.set_xlabel('Vertebra')
        ax.set_ylabel('Dice Score')
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=dice_scores.get('mean_dice', 0), color='red', linestyle='--', 
                   alpha=0.7, label=f"Mean: {dice_scores.get('mean_dice', 0):.3f}")
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_region_comparison(
        self,
        metrics: Dict[str, float],
        metric_prefix: str = 'mean',
        metric_suffix: str = 'mm',
        title: str = "Region Comparison",
        filename: str = "region_comparison.png"
    ):
        """Create bar chart comparing metrics across spine regions."""
        regions = ['Cervical', 'Thoracic', 'Lumbar']
        
        values = []
        for region in regions:
            key = f'{region.lower()}_{metric_prefix}_{metric_suffix}'
            values.append(metrics.get(key, 0))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.bar(regions, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12)
        
        ax.set_xlabel('Spine Region')
        ax.set_ylabel(f'{metric_prefix.title()} {metric_suffix.upper()}')
        ax.set_title(title)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_identification_rates(
        self,
        metrics: Dict[str, float],
        title: str = "Identification Rates at Different Thresholds",
        filename: str = "identification_rates.png"
    ):
        """Plot identification rates at different distance thresholds."""
        thresholds = []
        rates = []
        
        for key, value in sorted(metrics.items()):
            if key.startswith('id_rate_') and key.endswith('mm'):
                thresh = float(key.replace('id_rate_', '').replace('mm', ''))
                thresholds.append(thresh)
                rates.append(value * 100)  # Convert to percentage
        
        if not thresholds:
            print("No identification rate data available")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(thresholds, rates, 'o-', linewidth=2, markersize=10, color='#2E86AB')
        ax.fill_between(thresholds, rates, alpha=0.3)
        
        ax.set_xlabel('Distance Threshold (mm)')
        ax.set_ylabel('Identification Rate (%)')
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for t, r in zip(thresholds, rates):
            ax.annotate(f'{r:.1f}%', xy=(t, r), xytext=(0, 10),
                       textcoords='offset points', ha='center', fontsize=10)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def generate_results_report(
    output_dir: str,
    fold_localization_metrics: Dict[int, Dict],
    fold_segmentation_metrics: Dict[int, Dict],
    experiment_name: str = "VerSe2019_Pipeline"
) -> str:
    """
    Generate a comprehensive text report of results.
    
    Args:
        output_dir: Directory to save report
        fold_localization_metrics: Dict of fold -> localization metrics
        fold_segmentation_metrics: Dict of fold -> segmentation metrics
        experiment_name: Name of the experiment
        
    Returns:
        Path to the saved report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / f'{experiment_name}_results.txt'
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"RESULTS REPORT: {experiment_name}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    
    # Localization Results
    if fold_localization_metrics:
        lines.append("-" * 40)
        lines.append("LOCALIZATION RESULTS")
        lines.append("-" * 40)
        
        loc_evaluator = LocalizationEvaluator()
        aggregated = loc_evaluator.aggregate_fold_metrics(list(fold_localization_metrics.values()))
        
        lines.append(f"\nNumber of Folds: {len(fold_localization_metrics)}")
        lines.append("")
        
        # Summary table
        lines.append("Metric                          Mean ± Std          Min - Max")
        lines.append("-" * 65)
        
        for metric, stats in sorted(aggregated.items()):
            if not metric.startswith('per_'):
                mean = stats['mean']
                std = stats['std']
                min_val = stats.get('min', min(stats['values']))
                max_val = stats.get('max', max(stats['values']))
                lines.append(f"{metric:<30} {mean:>7.3f} ± {std:<7.3f} {min_val:>7.3f} - {max_val:<7.3f}")
        
        lines.append("")
        
        # Per-fold results
        lines.append("Per-Fold Results:")
        for fold, metrics in sorted(fold_localization_metrics.items()):
            lines.append(f"\n  Fold {fold}:")
            lines.append(f"    Mean Distance: {metrics.get('mean_distance_mm', 0):.2f} mm")
            lines.append(f"    ID Rate (20mm): {metrics.get('id_rate_20mm', 0)*100:.1f}%")
    
    lines.append("")
    
    # Segmentation Results
    if fold_segmentation_metrics:
        lines.append("-" * 40)
        lines.append("SEGMENTATION RESULTS")
        lines.append("-" * 40)
        
        seg_evaluator = SegmentationEvaluator()
        aggregated = seg_evaluator.aggregate_fold_metrics(list(fold_segmentation_metrics.values()))
        
        lines.append(f"\nNumber of Folds: {len(fold_segmentation_metrics)}")
        lines.append("")
        
        lines.append("Metric                          Mean ± Std")
        lines.append("-" * 50)
        
        for metric, stats in sorted(aggregated.items()):
            mean = stats['mean']
            std = stats['std']
            lines.append(f"{metric:<30} {mean:>7.4f} ± {std:<7.4f}")
        
        lines.append("")
        
        # Per-fold results
        lines.append("Per-Fold Results:")
        for fold, metrics in sorted(fold_segmentation_metrics.items()):
            lines.append(f"\n  Fold {fold}:")
            lines.append(f"    Mean Dice: {metrics.get('mean_dice', 0):.4f}")
            lines.append(f"    Cervical Dice: {metrics.get('cervical_mean_dice', 0):.4f}")
            lines.append(f"    Thoracic Dice: {metrics.get('thoracic_mean_dice', 0):.4f}")
            lines.append(f"    Lumbar Dice: {metrics.get('lumbar_mean_dice', 0):.4f}")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    # Write to file
    with open(report_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved report to: {report_file}")
    return str(report_file)


def generate_all_plots(
    output_dir: str,
    fold_localization_metrics: Dict[int, Dict],
    fold_segmentation_metrics: Dict[int, Dict],
):
    """Generate all visualization plots."""
    visualizer = ResultsVisualizer(output_dir)
    
    # Aggregate per-vertebra distances across folds for boxplot
    if fold_localization_metrics:
        per_vertebra_all = {}
        for fold, metrics in fold_localization_metrics.items():
            if 'per_vertebra_distances' in metrics:
                for name, dist in metrics['per_vertebra_distances'].items():
                    if name not in per_vertebra_all:
                        per_vertebra_all[name] = []
                    per_vertebra_all[name].append(dist)
        
        if per_vertebra_all:
            visualizer.plot_localization_boxplot(
                per_vertebra_all,
                title="Localization Error Distribution per Vertebra (All Folds)",
                filename="localization_boxplot_all_folds.png"
            )
        
        # Fold comparison for localization
        visualizer.plot_fold_comparison_bars(
            fold_localization_metrics,
            metric_names=['mean_distance_mm', 'id_rate_20mm'],
            title="Localization Metrics Across Folds",
            filename="localization_fold_comparison.png"
        )
        
        # Aggregate metrics for identification rate plot
        avg_metrics = {}
        for fold, metrics in fold_localization_metrics.items():
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    if key not in avg_metrics:
                        avg_metrics[key] = []
                    avg_metrics[key].append(val)
        
        avg_metrics = {k: np.mean(v) for k, v in avg_metrics.items()}
        visualizer.plot_identification_rates(avg_metrics)
        visualizer.plot_region_comparison(avg_metrics, 'mean', 'mm', 
                                          "Mean Localization Error by Region")
    
    # Segmentation plots
    if fold_segmentation_metrics:
        # Aggregate Dice scores
        avg_dice = {}
        for fold, metrics in fold_segmentation_metrics.items():
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    if key not in avg_dice:
                        avg_dice[key] = []
                    avg_dice[key].append(val)
        
        avg_dice = {k: np.mean(v) for k, v in avg_dice.items()}
        
        visualizer.plot_segmentation_dice_bars(
            avg_dice,
            title="Mean Dice Score per Vertebra (Across Folds)",
            filename="segmentation_dice_bars.png"
        )
        
        visualizer.plot_fold_comparison_bars(
            fold_segmentation_metrics,
            metric_names=['mean_dice', 'cervical_mean_dice', 'thoracic_mean_dice', 'lumbar_mean_dice'],
            title="Segmentation Dice Scores Across Folds",
            filename="segmentation_fold_comparison.png"
        )
    
    print(f"\nAll plots saved to: {visualizer.output_dir}")
