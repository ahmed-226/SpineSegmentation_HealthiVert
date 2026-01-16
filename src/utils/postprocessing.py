"""
Postprocessing utilities for vertebrae localization and segmentation
PyTorch Implementation
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass
from scipy.spatial.distance import cdist
import networkx as nx


# Prior statistics for vertebrae spacing (in mm)
# These are approximate anatomical distances between adjacent vertebrae
VERTEBRAE_DISTANCES_MEAN = {
    # Cervical
    ('C1', 'C2'): 25.0, ('C2', 'C3'): 18.0, ('C3', 'C4'): 16.0,
    ('C4', 'C5'): 16.0, ('C5', 'C6'): 16.0, ('C6', 'C7'): 18.0,
    # Cervical-Thoracic transition
    ('C7', 'T1'): 20.0,
    # Thoracic
    ('T1', 'T2'): 22.0, ('T2', 'T3'): 23.0, ('T3', 'T4'): 24.0,
    ('T4', 'T5'): 25.0, ('T5', 'T6'): 26.0, ('T6', 'T7'): 27.0,
    ('T7', 'T8'): 28.0, ('T8', 'T9'): 29.0, ('T9', 'T10'): 30.0,
    ('T10', 'T11'): 31.0, ('T11', 'T12'): 32.0,
    # Thoracic-Lumbar transition
    ('T12', 'L1'): 30.0,
    # Lumbar
    ('L1', 'L2'): 32.0, ('L2', 'L3'): 33.0, ('L3', 'L4'): 34.0,
    ('L4', 'L5'): 35.0, ('L5', 'L6'): 35.0,
}

VERTEBRAE_DISTANCES_STD = {key: val * 0.15 for key, val in VERTEBRAE_DISTANCES_MEAN.items()}

# Vertebrae names
VERTEBRAE_NAMES = [
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
    'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
    'L1', 'L2', 'L3', 'L4', 'L5', 'L6'
]


class HeatmapPostprocessor:
    """
    Postprocess heatmaps to extract landmark coordinates.
    Includes smoothing, local maxima detection, and filtering.
    """
    
    def __init__(
        self,
        sigma: float = 2.0,
        threshold: float = 0.05,
        min_distance: int = 5,
        return_multiple: bool = False
    ):
        """
        Args:
            sigma: Gaussian smoothing sigma
            threshold: Minimum heatmap value threshold
            min_distance: Minimum distance between detected maxima
            return_multiple: Whether to return multiple candidates per channel
        """
        self.sigma = sigma
        self.threshold = threshold
        self.min_distance = min_distance
        self.return_multiple = return_multiple
    
    def __call__(
        self, 
        heatmaps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract landmarks from heatmaps.
        
        Args:
            heatmaps: [num_landmarks, D, H, W] array
            
        Returns:
            landmarks: [num_landmarks, 3] coordinates
            confidences: [num_landmarks] confidence values
        """
        num_landmarks = heatmaps.shape[0]
        landmarks = np.zeros((num_landmarks, 3))
        confidences = np.zeros(num_landmarks)
        
        for i in range(num_landmarks):
            heatmap = heatmaps[i]
            
            # Smooth
            if self.sigma > 0:
                heatmap = gaussian_filter(heatmap, sigma=self.sigma)
            
            # Find local maxima
            maxima = self._find_local_maxima(heatmap)
            
            if len(maxima) > 0:
                # Take the maximum with highest value
                best_idx = np.argmax([m[1] for m in maxima])
                landmarks[i] = maxima[best_idx][0]
                confidences[i] = maxima[best_idx][1]
        
        return landmarks, confidences
    
    def _find_local_maxima(
        self, 
        heatmap: np.ndarray
    ) -> List[Tuple[np.ndarray, float]]:
        """Find local maxima in heatmap"""
        # Apply maximum filter
        max_filtered = maximum_filter(heatmap, size=self.min_distance * 2 + 1)
        
        # Find local maxima above threshold
        maxima_mask = (heatmap == max_filtered) & (heatmap > self.threshold)
        
        # Get coordinates and values
        coords = np.argwhere(maxima_mask)
        values = heatmap[maxima_mask]
        
        # Sort by value descending
        sorted_idx = np.argsort(-values)
        
        results = []
        for idx in sorted_idx:
            results.append((coords[idx].astype(float), float(values[idx])))
        
        return results


class SpineGraphOptimizer:
    """
    Graph-based optimization for vertebrae localization.
    Uses anatomical priors (expected distances) to optimize landmark selection.
    """
    
    def __init__(
        self,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        distance_weight: float = 1.0,
        confidence_weight: float = 1.0
    ):
        """
        Args:
            spacing: Voxel spacing in mm
            distance_weight: Weight for distance-based cost
            confidence_weight: Weight for confidence-based cost
        """
        self.spacing = np.array(spacing)
        self.distance_weight = distance_weight
        self.confidence_weight = confidence_weight
    
    def optimize(
        self,
        landmarks: np.ndarray,
        confidences: np.ndarray,
        candidates: Optional[Dict[int, List[Tuple[np.ndarray, float]]]] = None
    ) -> np.ndarray:
        """
        Optimize landmark positions using graph-based method.
        
        Args:
            landmarks: [num_landmarks, 3] initial landmarks
            confidences: [num_landmarks] confidence values
            candidates: Optional dict of additional candidates per landmark
            
        Returns:
            optimized: [num_landmarks, 3] optimized landmarks
        """
        num_landmarks = len(landmarks)
        optimized = landmarks.copy()
        
        # Build graph
        G = nx.DiGraph()
        
        # Add source and sink
        G.add_node('source')
        G.add_node('sink')
        
        # Add nodes for each landmark candidate
        for i in range(num_landmarks):
            if confidences[i] > 0:
                node_id = f'L{i}_0'
                G.add_node(node_id, pos=landmarks[i], conf=confidences[i])
                
                # Add candidates if provided
                if candidates and i in candidates:
                    for j, (pos, conf) in enumerate(candidates[i][1:], 1):
                        node_id = f'L{i}_{j}'
                        G.add_node(node_id, pos=pos, conf=conf)
        
        # Add edges from source to first valid landmark
        for i in range(num_landmarks):
            if confidences[i] > 0:
                for j in range(self._get_num_candidates(i, candidates)):
                    G.add_edge('source', f'L{i}_{j}', weight=0)
                break
        
        # Add edges between consecutive landmarks
        for i in range(num_landmarks - 1):
            if confidences[i] <= 0:
                continue
            
            # Find next valid landmark
            for next_i in range(i + 1, num_landmarks):
                if confidences[next_i] > 0:
                    break
            else:
                continue
            
            # Get expected distance
            exp_distance = self._get_expected_distance(i, next_i)
            exp_std = exp_distance * 0.15
            
            # Add edges between all candidate combinations
            for j in range(self._get_num_candidates(i, candidates)):
                for k in range(self._get_num_candidates(next_i, candidates)):
                    src_node = f'L{i}_{j}'
                    dst_node = f'L{next_i}_{k}'
                    
                    if src_node in G.nodes and dst_node in G.nodes:
                        src_pos = G.nodes[src_node]['pos']
                        dst_pos = G.nodes[dst_node]['pos']
                        
                        # Compute distance
                        dist = np.linalg.norm((src_pos - dst_pos) * self.spacing)
                        
                        # Distance cost (normalized Gaussian)
                        dist_cost = ((dist - exp_distance) / exp_std) ** 2
                        
                        # Confidence cost
                        conf_cost = -np.log(G.nodes[dst_node]['conf'] + 1e-8)
                        
                        # Total weight
                        weight = (self.distance_weight * dist_cost + 
                                 self.confidence_weight * conf_cost)
                        
                        G.add_edge(src_node, dst_node, weight=weight)
        
        # Add edges to sink from last valid landmark
        for i in range(num_landmarks - 1, -1, -1):
            if confidences[i] > 0:
                for j in range(self._get_num_candidates(i, candidates)):
                    G.add_edge(f'L{i}_{j}', 'sink', weight=0)
                break
        
        # Find shortest path
        try:
            path = nx.shortest_path(G, 'source', 'sink', weight='weight')
            
            # Extract optimized positions
            for node in path[1:-1]:  # Skip source and sink
                landmark_idx = int(node.split('_')[0][1:])
                optimized[landmark_idx] = G.nodes[node]['pos']
        except nx.NetworkXNoPath:
            pass
        
        return optimized
    
    def _get_num_candidates(
        self, 
        idx: int, 
        candidates: Optional[Dict]
    ) -> int:
        """Get number of candidates for a landmark"""
        if candidates and idx in candidates:
            return len(candidates[idx])
        return 1
    
    def _get_expected_distance(self, idx1: int, idx2: int) -> float:
        """Get expected distance between two vertebrae"""
        total_distance = 0
        
        for i in range(idx1, idx2):
            if i < len(VERTEBRAE_NAMES) - 1:
                key = (VERTEBRAE_NAMES[i], VERTEBRAE_NAMES[i + 1])
                if key in VERTEBRAE_DISTANCES_MEAN:
                    total_distance += VERTEBRAE_DISTANCES_MEAN[key]
                else:
                    total_distance += 25.0  # Default
        
        return total_distance


class LandmarkInterpolator:
    """
    Interpolate missing landmarks from neighbors.
    """
    
    def __init__(self, max_gap: int = 2):
        """
        Args:
            max_gap: Maximum gap size to interpolate
        """
        self.max_gap = max_gap
    
    def interpolate(
        self,
        landmarks: np.ndarray,
        confidences: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate missing landmarks.
        
        Args:
            landmarks: [num_landmarks, 3] coordinates
            confidences: [num_landmarks] confidence values
            
        Returns:
            landmarks: [num_landmarks, 3] interpolated landmarks
            confidences: [num_landmarks] updated confidences
        """
        result_landmarks = landmarks.copy()
        result_confidences = confidences.copy()
        
        num_landmarks = len(landmarks)
        valid = confidences > 0
        
        for i in range(num_landmarks):
            if valid[i]:
                continue
            
            # Find nearest valid neighbors
            prev_idx = None
            next_idx = None
            
            for j in range(i - 1, -1, -1):
                if valid[j]:
                    prev_idx = j
                    break
            
            for j in range(i + 1, num_landmarks):
                if valid[j]:
                    next_idx = j
                    break
            
            # Check if we can interpolate
            if prev_idx is not None and next_idx is not None:
                gap = next_idx - prev_idx - 1
                if gap <= self.max_gap:
                    # Linear interpolation
                    t = (i - prev_idx) / (next_idx - prev_idx)
                    
                    result_landmarks[i] = (
                        landmarks[prev_idx] * (1 - t) + 
                        landmarks[next_idx] * t
                    )
                    result_confidences[i] = 0.5  # Mark as interpolated
        
        return result_landmarks, result_confidences


class TopBottomFilter:
    """
    Filter landmarks at the top and bottom of the spine.
    Removes spurious detections outside the valid range.
    """
    
    def __init__(
        self,
        min_valid_landmarks: int = 5,
        max_cervical_z: Optional[float] = None,
        min_lumbar_z: Optional[float] = None
    ):
        self.min_valid_landmarks = min_valid_landmarks
        self.max_cervical_z = max_cervical_z
        self.min_lumbar_z = min_lumbar_z
    
    def filter(
        self,
        landmarks: np.ndarray,
        confidences: np.ndarray,
        image_shape: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter top and bottom landmarks.
        
        Args:
            landmarks: [num_landmarks, 3] coordinates
            confidences: [num_landmarks] confidence values
            image_shape: (D, H, W) image dimensions
            
        Returns:
            filtered_landmarks: [num_landmarks, 3]
            filtered_confidences: [num_landmarks]
        """
        result_landmarks = landmarks.copy()
        result_confidences = confidences.copy()
        
        valid = confidences > 0
        valid_indices = np.where(valid)[0]
        
        if len(valid_indices) < self.min_valid_landmarks:
            return result_landmarks, result_confidences
        
        # Get z-coordinates of valid landmarks
        z_coords = landmarks[valid, 0]
        
        # Check for outliers at top (cervical)
        if self.max_cervical_z is not None:
            for i in valid_indices[:7]:  # C1-C7
                if landmarks[i, 0] > self.max_cervical_z:
                    result_confidences[i] = 0
        
        # Check for outliers at bottom (lumbar)
        if self.min_lumbar_z is not None:
            for i in valid_indices[-6:]:  # L1-L6
                if landmarks[i, 0] < self.min_lumbar_z:
                    result_confidences[i] = 0
        
        # Filter landmarks outside image bounds
        for i in range(len(landmarks)):
            if result_confidences[i] > 0:
                pos = landmarks[i]
                if (pos[0] < 0 or pos[0] >= image_shape[0] or
                    pos[1] < 0 or pos[1] >= image_shape[1] or
                    pos[2] < 0 or pos[2] >= image_shape[2]):
                    result_confidences[i] = 0
        
        return result_landmarks, result_confidences


class VertebraePostprocessor:
    """
    Complete postprocessing pipeline for vertebrae localization.
    """
    
    def __init__(
        self,
        spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        heatmap_sigma: float = 2.0,
        heatmap_threshold: float = 0.05,
        use_graph_optimization: bool = True,
        interpolate_missing: bool = True,
        max_interpolation_gap: int = 2
    ):
        self.heatmap_processor = HeatmapPostprocessor(
            sigma=heatmap_sigma,
            threshold=heatmap_threshold,
            return_multiple=use_graph_optimization
        )
        
        self.graph_optimizer = SpineGraphOptimizer(
            spacing=spacing
        ) if use_graph_optimization else None
        
        self.interpolator = LandmarkInterpolator(
            max_gap=max_interpolation_gap
        ) if interpolate_missing else None
        
        self.top_bottom_filter = TopBottomFilter()
    
    def __call__(
        self,
        heatmaps: np.ndarray,
        image_shape: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run full postprocessing pipeline.
        
        Args:
            heatmaps: [num_landmarks, D, H, W] heatmaps
            image_shape: Original image shape
            
        Returns:
            landmarks: [num_landmarks, 3] final coordinates
            confidences: [num_landmarks] confidence values
        """
        # Extract landmarks from heatmaps
        landmarks, confidences = self.heatmap_processor(heatmaps)
        
        # Graph-based optimization
        if self.graph_optimizer is not None:
            landmarks = self.graph_optimizer.optimize(landmarks, confidences)
        
        # Interpolate missing landmarks
        if self.interpolator is not None:
            landmarks, confidences = self.interpolator.interpolate(
                landmarks, confidences
            )
        
        # Filter top/bottom outliers
        landmarks, confidences = self.top_bottom_filter.filter(
            landmarks, confidences, image_shape
        )
        
        return landmarks, confidences


def compute_identification_rate(
    predicted: np.ndarray,
    target: np.ndarray,
    distance_threshold: float = 20.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, float]:
    """
    Compute vertebrae identification rate metrics.
    
    Args:
        predicted: [num_landmarks, 3] predicted coordinates
        target: [num_landmarks, 3] ground truth coordinates
        distance_threshold: Threshold for correct identification (mm)
        spacing: Voxel spacing in mm
        
    Returns:
        Dictionary with identification metrics
    """
    spacing = np.array(spacing)
    num_landmarks = len(predicted)
    
    # Compute distances
    distances = np.linalg.norm((predicted - target) * spacing, axis=1)
    
    # Correct identifications
    correct = distances < distance_threshold
    
    # Compute metrics
    id_rate = np.mean(correct)
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    
    # Per-region metrics
    cervical_correct = np.mean(correct[:7]) if num_landmarks >= 7 else 0
    thoracic_correct = np.mean(correct[7:19]) if num_landmarks >= 19 else 0
    lumbar_correct = np.mean(correct[19:25]) if num_landmarks >= 25 else 0
    
    return {
        'identification_rate': id_rate,
        'mean_distance_mm': mean_distance,
        'median_distance_mm': median_distance,
        'cervical_id_rate': cervical_correct,
        'thoracic_id_rate': thoracic_correct,
        'lumbar_id_rate': lumbar_correct,
        'num_correct': int(correct.sum()),
        'num_total': num_landmarks
    }


def compute_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int = 26
) -> Dict[str, float]:
    """
    Compute segmentation metrics.
    
    Args:
        prediction: 3D array with predicted labels
        target: 3D array with ground truth labels
        num_classes: Number of classes including background
        
    Returns:
        Dictionary with segmentation metrics
    """
    dice_scores = []
    
    for c in range(1, num_classes):  # Skip background
        pred_c = (prediction == c)
        target_c = (target == c)
        
        intersection = np.logical_and(pred_c, target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = 2 * intersection / union
        else:
            dice = 1.0 if intersection == 0 else 0.0
        
        dice_scores.append(dice)
    
    dice_scores = np.array(dice_scores)
    
    return {
        'mean_dice': float(np.mean(dice_scores)),
        'median_dice': float(np.median(dice_scores)),
        'min_dice': float(np.min(dice_scores)),
        'max_dice': float(np.max(dice_scores)),
        'dice_per_class': dice_scores.tolist()
    }
