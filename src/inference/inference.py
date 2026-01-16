"""
Inference pipeline for VerSe2019 Vertebrae Segmentation
PyTorch Implementation
"""
import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from ..models.networks import SimpleUNet, SpatialConfigurationNet, SegmentationUNet
from ..data.dataset import load_nifti, IntensityNormalization, world_to_voxel, voxel_to_world
from ..utils.heatmap_utils import (
    extract_all_landmarks, gaussian_smooth_3d, find_local_maxima_3d,
    interpolate_missing_landmarks
)
from ..config import PipelineConfig, VERTEBRAE_LABELS

try:
    import torchio as tio
    HAS_TORCHIO = True
except ImportError:
    HAS_TORCHIO = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocess images for inference"""
    
    def __init__(
        self,
        target_spacing: Tuple[float, float, float],
        target_size: Tuple[int, int, int],
        normalize: bool = True
    ):
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.normalize = normalize
        
        if normalize:
            self.intensity_norm = IntensityNormalization(
                scale=1.0 / 2048.0,
                pre_clamp_min=-1024.0
            )
    
    def __call__(
        self, 
        image: np.ndarray,
        original_spacing: Tuple[float, float, float],
        center: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Preprocess image for inference.
        
        Args:
            image: 3D numpy array [D, H, W]
            original_spacing: Original voxel spacing
            center: Optional center point for cropping
            
        Returns:
            preprocessed: [1, 1, D, H, W] tensor
            metadata: Dictionary with preprocessing info
        """
        from scipy.ndimage import zoom
        
        # Resample to target spacing
        zoom_factors = np.array(original_spacing) / np.array(self.target_spacing)
        resampled = zoom(image.astype(np.float32), zoom_factors, order=1)
        
        # Store original resampled shape
        resampled_shape = resampled.shape
        
        # Crop or pad to target size
        if center is not None:
            # Crop around center
            center_resampled = center * zoom_factors
            processed = self._crop_around_center(resampled, center_resampled, self.target_size)
            crop_offset = center_resampled - np.array(self.target_size) // 2
        else:
            # Center crop/pad
            processed = self._center_crop_or_pad(resampled, self.target_size)
            crop_offset = (np.array(resampled_shape) - np.array(self.target_size)) // 2
        
        # Normalize intensity
        if self.normalize:
            processed = self.intensity_norm(processed, is_training=False)
        
        # Convert to tensor
        tensor = torch.from_numpy(processed).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        
        metadata = {
            'original_spacing': original_spacing,
            'target_spacing': self.target_spacing,
            'zoom_factors': zoom_factors,
            'resampled_shape': resampled_shape,
            'crop_offset': crop_offset,
            'target_size': self.target_size
        }
        
        return tensor, metadata
    
    def _crop_around_center(
        self, 
        image: np.ndarray, 
        center: np.ndarray,
        size: Tuple[int, int, int]
    ) -> np.ndarray:
        """Crop image around center point"""
        result = np.full(size, -1024.0, dtype=image.dtype)
        
        half_size = np.array(size) // 2
        start = (center - half_size).astype(int)
        end = start + np.array(size)
        
        src_start = np.maximum(start, 0)
        src_end = np.minimum(end, image.shape)
        
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        
        result[
            dst_start[0]:dst_end[0],
            dst_start[1]:dst_end[1],
            dst_start[2]:dst_end[2]
        ] = image[
            src_start[0]:src_end[0],
            src_start[1]:src_end[1],
            src_start[2]:src_end[2]
        ]
        
        return result
    
    def _center_crop_or_pad(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int, int]
    ) -> np.ndarray:
        """Center crop or pad to target size"""
        result = np.full(target_size, -1024.0, dtype=np.float32)
        
        for dim in range(3):
            if image.shape[dim] >= target_size[dim]:
                start = (image.shape[dim] - target_size[dim]) // 2
                slc = slice(start, start + target_size[dim])
                if dim == 0:
                    image = image[slc, :, :]
                elif dim == 1:
                    image = image[:, slc, :]
                else:
                    image = image[:, :, slc]
        
        # Copy to result
        for dim in range(3):
            if image.shape[dim] < target_size[dim]:
                pad_before = (target_size[dim] - image.shape[dim]) // 2
                if dim == 0:
                    result[pad_before:pad_before+image.shape[0], :image.shape[1], :image.shape[2]] = image
                    return result
        
        result[:image.shape[0], :image.shape[1], :image.shape[2]] = image
        return result


class TiledInference:
    """
    Handle inference on large volumes using overlapping tiles.
    """
    
    def __init__(
        self,
        tile_size: Tuple[int, int, int],
        overlap: Tuple[int, int, int] = (16, 16, 16),
        merge_mode: str = 'max'
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.merge_mode = merge_mode
    
    def __call__(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Run tiled inference on a large volume.
        
        Args:
            model: Network model
            image: [1, C, D, H, W] input tensor
            device: Computation device
            
        Returns:
            output: [1, num_labels, D, H, W] heatmaps
        """
        _, C, D, H, W = image.shape
        td, th, tw = self.tile_size
        od, oh, ow = self.overlap
        
        # Calculate tile positions
        d_positions = self._get_tile_positions(D, td, od)
        h_positions = self._get_tile_positions(H, th, oh)
        w_positions = self._get_tile_positions(W, tw, ow)
        
        # Get number of output channels from a test forward pass
        with torch.no_grad():
            test_tile = image[:, :, :td, :th, :tw].to(device)
            test_out = model(test_tile)
            if isinstance(test_out, tuple):
                test_out = test_out[0]
            num_labels = test_out.shape[1]
        
        # Initialize output
        if self.merge_mode == 'max':
            output = torch.full((1, num_labels, D, H, W), -float('inf'), device='cpu')
        else:
            output = torch.zeros((1, num_labels, D, H, W), device='cpu')
            count = torch.zeros((1, 1, D, H, W), device='cpu')
        
        model.eval()
        with torch.no_grad():
            for d in d_positions:
                for h in h_positions:
                    for w in w_positions:
                        # Extract tile
                        tile = image[:, :, d:d+td, h:h+th, w:w+tw].to(device)
                        
                        # Pad if necessary
                        pad_d = td - tile.shape[2]
                        pad_h = th - tile.shape[3]
                        pad_w = tw - tile.shape[4]
                        
                        if pad_d > 0 or pad_h > 0 or pad_w > 0:
                            tile = F.pad(tile, (0, pad_w, 0, pad_h, 0, pad_d), value=-1)
                        
                        # Run inference
                        pred = model(tile)
                        if isinstance(pred, tuple):
                            pred = pred[0]
                        
                        # Remove padding
                        if pad_d > 0:
                            pred = pred[:, :, :-pad_d, :, :]
                        if pad_h > 0:
                            pred = pred[:, :, :, :-pad_h, :]
                        if pad_w > 0:
                            pred = pred[:, :, :, :, :-pad_w]
                        
                        pred = pred.cpu()
                        
                        # Merge
                        actual_d = pred.shape[2]
                        actual_h = pred.shape[3]
                        actual_w = pred.shape[4]
                        
                        if self.merge_mode == 'max':
                            output[:, :, d:d+actual_d, h:h+actual_h, w:w+actual_w] = torch.maximum(
                                output[:, :, d:d+actual_d, h:h+actual_h, w:w+actual_w],
                                pred
                            )
                        else:
                            output[:, :, d:d+actual_d, h:h+actual_h, w:w+actual_w] += pred
                            count[:, :, d:d+actual_d, h:h+actual_h, w:w+actual_w] += 1
        
        if self.merge_mode == 'mean':
            output = output / (count + 1e-8)
        
        return output
    
    def _get_tile_positions(self, size: int, tile_size: int, overlap: int) -> List[int]:
        """Calculate tile starting positions"""
        stride = tile_size - overlap
        positions = list(range(0, size - tile_size + 1, stride))
        
        if len(positions) == 0 or positions[-1] + tile_size < size:
            positions.append(max(0, size - tile_size))
        
        return positions


class SpineLocalizationInference:
    """Inference for Stage 1: Spine Localization"""
    
    def __init__(
        self,
        config: PipelineConfig,
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.stage_config = config.spine_localization
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = SimpleUNet(
            in_channels=1,
            num_labels=self.stage_config.num_labels,
            num_filters_base=self.stage_config.num_filters_base,
            num_levels=self.stage_config.num_levels,
            heatmap_initialization=True
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Preprocessor
        self.preprocessor = ImagePreprocessor(
            target_spacing=self.stage_config.image_spacing,
            target_size=self.stage_config.image_size
        )
        
        logger.info(f"Loaded spine localization model from {checkpoint_path}")
    
    def __call__(self, image_path: str) -> Dict:
        """
        Run spine localization inference.
        
        Args:
            image_path: Path to NIfTI image
            
        Returns:
            Dictionary with spine center coordinates
        """
        # Load image
        image_data, affine, spacing, origin = load_nifti(image_path)
        
        # Preprocess
        input_tensor, metadata = self.preprocessor(
            image_data.astype(np.float32),
            original_spacing=spacing
        )
        
        # Run inference
        with torch.no_grad():
            heatmap = self.model(input_tensor.to(self.device))
            heatmap = heatmap.cpu()
        
        # Find spine center (weighted centroid)
        heatmap_np = heatmap[0, 0].numpy()
        heatmap_np = np.maximum(heatmap_np, 0)  # Ensure non-negative
        
        # Compute centroid
        total_weight = heatmap_np.sum()
        if total_weight > 0:
            z_coords, y_coords, x_coords = np.indices(heatmap_np.shape)
            cz = (heatmap_np * z_coords).sum() / total_weight
            cy = (heatmap_np * y_coords).sum() / total_weight
            cx = (heatmap_np * x_coords).sum() / total_weight
            
            center_voxel = np.array([cz, cy, cx])
        else:
            center_voxel = np.array(heatmap_np.shape) / 2
        
        # Transform back to original space
        center_resampled = center_voxel + metadata['crop_offset']
        center_original = center_resampled / metadata['zoom_factors']
        center_world = voxel_to_world(center_original, affine)
        
        return {
            'spine_center_voxel': center_original,
            'spine_center_world': center_world,
            'heatmap': heatmap_np,
            'confidence': float(heatmap_np.max())
        }


class VertebraeLocalizationInference:
    """Inference for Stage 2: Vertebrae Localization"""
    
    def __init__(
        self,
        config: PipelineConfig,
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.stage_config = config.vertebrae_localization
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = SpatialConfigurationNet(
            in_channels=1,
            num_labels=self.stage_config.num_landmarks,
            num_filters_base=self.stage_config.num_filters_base,
            num_levels=self.stage_config.num_levels,
            spatial_downsample=self.stage_config.spatial_downsample,
            dropout_ratio=0.0  # No dropout at inference
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Preprocessor
        self.preprocessor = ImagePreprocessor(
            target_spacing=self.stage_config.image_spacing,
            target_size=self.stage_config.image_size
        )
        
        # Tiled inference for large volumes
        self.tiled_inference = TiledInference(
            tile_size=self.stage_config.image_size,
            overlap=(16, 16, 16)
        )
        
        logger.info(f"Loaded vertebrae localization model from {checkpoint_path}")
    
    def __call__(
        self, 
        image_path: str,
        spine_center: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run vertebrae localization inference.
        
        Args:
            image_path: Path to NIfTI image
            spine_center: Optional spine center from Stage 1
            
        Returns:
            Dictionary with vertebrae landmarks
        """
        # Load image
        image_data, affine, spacing, origin = load_nifti(image_path)
        
        # Preprocess
        input_tensor, metadata = self.preprocessor(
            image_data.astype(np.float32),
            original_spacing=spacing,
            center=spine_center
        )
        
        # Run inference
        with torch.no_grad():
            # Check if we need tiled inference
            if (np.array(image_data.shape) / np.array(metadata['zoom_factors']) > 
                np.array(self.stage_config.image_size) * 1.5).any():
                # Use tiled inference
                heatmaps = self.tiled_inference(
                    self.model, input_tensor, self.device
                )
            else:
                heatmaps, _, _ = self.model(input_tensor.to(self.device))
                heatmaps = heatmaps.cpu()
        
        # Smooth heatmaps
        heatmaps_smooth = gaussian_smooth_3d(heatmaps[0], sigma=2.0)
        
        # Extract landmarks
        landmarks, confidences = extract_all_landmarks(
            heatmaps_smooth, method='argmax', threshold=0.05
        )
        
        # Interpolate missing landmarks
        landmarks = interpolate_missing_landmarks(landmarks)
        
        # Transform to original space
        landmarks_world = self._transform_landmarks_to_world(
            landmarks, metadata, affine
        )
        
        return {
            'landmarks': landmarks,
            'landmarks_world': landmarks_world,
            'confidences': confidences,
            'heatmaps': heatmaps_smooth.numpy(),
            'vertebrae_names': [
                VERTEBRAE_LABELS.get(i, f'V{i}') for i in range(self.stage_config.num_landmarks)
            ]
        }
    
    def _transform_landmarks_to_world(
        self,
        landmarks: torch.Tensor,
        metadata: dict,
        affine: np.ndarray
    ) -> np.ndarray:
        """Transform landmarks from network output space to world coordinates"""
        landmarks_world = np.zeros((landmarks.shape[0], 4))
        
        for i in range(landmarks.shape[0]):
            if landmarks[i, 0] > 0.5:  # is_valid
                # Get voxel coordinates in network space
                voxel_net = landmarks[i, 1:4].numpy()
                
                # Transform to resampled space
                voxel_resampled = voxel_net + metadata['crop_offset']
                
                # Transform to original voxel space
                voxel_original = voxel_resampled / metadata['zoom_factors']
                
                # Transform to world coordinates
                world_coords = voxel_to_world(voxel_original, affine)
                
                landmarks_world[i, 0] = 1.0
                landmarks_world[i, 1:4] = world_coords
        
        return landmarks_world


class VertebraeSegmentationInference:
    """Inference for Stage 3: Vertebrae Segmentation"""
    
    def __init__(
        self,
        config: PipelineConfig,
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.stage_config = config.vertebrae_segmentation
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = SegmentationUNet(
            in_channels=1,
            num_classes=self.stage_config.num_labels + 1,
            num_filters_base=self.stage_config.num_filters_base,
            num_levels=self.stage_config.num_levels
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Preprocessor
        self.preprocessor = ImagePreprocessor(
            target_spacing=self.stage_config.image_spacing,
            target_size=self.stage_config.image_size
        )
        
        logger.info(f"Loaded vertebrae segmentation model from {checkpoint_path}")
    
    def __call__(
        self,
        image_path: str,
        landmarks: np.ndarray
    ) -> Dict:
        """
        Run vertebrae segmentation inference.
        
        Args:
            image_path: Path to NIfTI image
            landmarks: [num_landmarks, 4] array from Stage 2
            
        Returns:
            Dictionary with segmentation results
        """
        # Load image
        image_data, affine, spacing, origin = load_nifti(image_path)
        
        # Initialize output segmentation
        segmentation = np.zeros(image_data.shape, dtype=np.int32)
        
        # Process each vertebra
        for i in range(landmarks.shape[0]):
            if landmarks[i, 0] < 0.5:  # Not valid
                continue
            
            # Get vertebra center in voxel coordinates
            center_world = landmarks[i, 1:4]
            center_voxel = world_to_voxel(center_world, affine)
            
            # Preprocess patch around vertebra
            input_tensor, metadata = self.preprocessor(
                image_data.astype(np.float32),
                original_spacing=spacing,
                center=center_voxel
            )
            
            # Run inference
            with torch.no_grad():
                logits = self.model(input_tensor.to(self.device))
                probs = F.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)
                pred = pred.cpu().numpy()[0]
            
            # Map prediction to original space
            self._merge_segmentation(
                segmentation, pred, i + 1,  # Label starts from 1
                center_voxel, metadata
            )
        
        return {
            'segmentation': segmentation,
            'affine': affine,
            'spacing': spacing
        }
    
    def _merge_segmentation(
        self,
        output: np.ndarray,
        patch: np.ndarray,
        label: int,
        center: np.ndarray,
        metadata: dict
    ):
        """Merge patch segmentation into output volume"""
        from scipy.ndimage import zoom
        
        # Find voxels belonging to this vertebra
        mask = (patch > 0)
        
        # Get bounds in patch
        if not mask.any():
            return
        
        # Transform mask to original resolution
        inv_zoom = 1.0 / metadata['zoom_factors']
        mask_original = zoom(mask.astype(float), inv_zoom, order=0) > 0.5
        
        # Calculate position in output volume
        half_size = np.array(metadata['target_size']) // 2
        half_size_original = (half_size / metadata['zoom_factors']).astype(int)
        
        start = (center - half_size_original).astype(int)
        end = start + np.array(mask_original.shape)
        
        # Clip to volume bounds
        start_clip = np.maximum(start, 0)
        end_clip = np.minimum(end, output.shape)
        
        # Calculate corresponding patch bounds
        patch_start = start_clip - start
        patch_end = patch_start + (end_clip - start_clip)
        
        # Merge
        mask_region = mask_original[
            patch_start[0]:patch_end[0],
            patch_start[1]:patch_end[1],
            patch_start[2]:patch_end[2]
        ]
        
        output_region = output[
            start_clip[0]:end_clip[0],
            start_clip[1]:end_clip[1],
            start_clip[2]:end_clip[2]
        ]
        
        # Only update where no label exists or where this patch predicts foreground
        update_mask = mask_region & (output_region == 0)
        output_region[update_mask] = label


class FullPipelineInference:
    """
    Complete inference pipeline: Spine -> Vertebrae -> Segmentation
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        spine_checkpoint: str,
        vertebrae_checkpoint: str,
        segmentation_checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize stage models
        self.spine_inference = SpineLocalizationInference(
            config, spine_checkpoint, self.device
        )
        
        self.vertebrae_inference = VertebraeLocalizationInference(
            config, vertebrae_checkpoint, self.device
        )
        
        if segmentation_checkpoint:
            self.segmentation_inference = VertebraeSegmentationInference(
                config, segmentation_checkpoint, self.device
            )
        else:
            self.segmentation_inference = None
        
        logger.info("Full pipeline initialized")
    
    def __call__(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        run_segmentation: bool = True
    ) -> Dict:
        """
        Run full inference pipeline.
        
        Args:
            image_path: Path to input NIfTI image
            output_dir: Optional directory to save outputs
            run_segmentation: Whether to run segmentation stage
            
        Returns:
            Dictionary with all results
        """
        results = {}
        
        # Stage 1: Spine Localization
        logger.info("Stage 1: Spine Localization...")
        spine_result = self.spine_inference(image_path)
        results['spine'] = spine_result
        
        # Stage 2: Vertebrae Localization
        logger.info("Stage 2: Vertebrae Localization...")
        vertebrae_result = self.vertebrae_inference(
            image_path,
            spine_center=spine_result['spine_center_voxel']
        )
        results['vertebrae'] = vertebrae_result
        
        # Stage 3: Segmentation (optional)
        if run_segmentation and self.segmentation_inference is not None:
            logger.info("Stage 3: Vertebrae Segmentation...")
            seg_result = self.segmentation_inference(
                image_path,
                landmarks=vertebrae_result['landmarks_world']
            )
            results['segmentation'] = seg_result
        
        # Save outputs
        if output_dir:
            self._save_outputs(results, image_path, output_dir)
        
        logger.info("Pipeline completed!")
        return results
    
    def _save_outputs(self, results: Dict, image_path: str, output_dir: str):
        """Save inference outputs"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_id = Path(image_path).stem.replace('.nii', '')
        
        # Save landmarks as CSV
        landmarks = results['vertebrae']['landmarks_world']
        names = results['vertebrae']['vertebrae_names']
        
        csv_path = output_dir / f'{image_id}_landmarks.csv'
        with open(csv_path, 'w') as f:
            f.write('vertebra,is_valid,x,y,z,confidence\n')
            for i, name in enumerate(names):
                is_valid = int(landmarks[i, 0])
                x, y, z = landmarks[i, 1:4]
                conf = float(results['vertebrae']['confidences'][i])
                f.write(f'{name},{is_valid},{x:.2f},{y:.2f},{z:.2f},{conf:.4f}\n')
        
        logger.info(f"Saved landmarks: {csv_path}")
        
        # Save segmentation as NIfTI
        if 'segmentation' in results:
            seg = results['segmentation']['segmentation']
            affine = results['segmentation']['affine']
            
            nii = nib.Nifti1Image(seg.astype(np.int16), affine)
            seg_path = output_dir / f'{image_id}_segmentation.nii.gz'
            nib.save(nii, str(seg_path))
            
            logger.info(f"Saved segmentation: {seg_path}")


def run_inference(
    image_path: str,
    config: PipelineConfig,
    spine_checkpoint: str,
    vertebrae_checkpoint: str,
    segmentation_checkpoint: Optional[str] = None,
    output_dir: Optional[str] = None,
    run_segmentation: bool = True
) -> Dict:
    """
    Convenience function to run full pipeline inference.
    
    Args:
        image_path: Path to input image
        config: Pipeline configuration
        spine_checkpoint: Path to spine localization checkpoint
        vertebrae_checkpoint: Path to vertebrae localization checkpoint
        segmentation_checkpoint: Optional path to segmentation checkpoint
        output_dir: Optional output directory
        run_segmentation: Whether to run segmentation
        
    Returns:
        Dictionary with all results
    """
    pipeline = FullPipelineInference(
        config=config,
        spine_checkpoint=spine_checkpoint,
        vertebrae_checkpoint=vertebrae_checkpoint,
        segmentation_checkpoint=segmentation_checkpoint
    )
    
    return pipeline(image_path, output_dir, run_segmentation)
