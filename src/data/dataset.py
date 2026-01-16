"""
Data loading and dataset classes for VerSe2019 Pipeline
PyTorch Implementation with TorchIO for medical image processing
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from functools import lru_cache
import nibabel as nib
import SimpleITK as sitk

try:
    import torchio as tio
    HAS_TORCHIO = True
except ImportError:
    HAS_TORCHIO = False
    print("Warning: TorchIO not installed. Some augmentations will be limited.")


@dataclass
class Landmark:
    """Landmark data structure"""
    coords: Tuple[float, float, float]  # (x, y, z) in world coordinates
    is_valid: bool


class ImageCache:
    """LRU cache for medical images"""
    
    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self._cache = {}
        self._order = []
    
    def get(self, path: str) -> Optional[Tuple[np.ndarray, Tuple, Tuple]]:
        """Get cached image data"""
        if path in self._cache:
            # Move to end (most recently used)
            self._order.remove(path)
            self._order.append(path)
            return self._cache[path]
        return None
    
    def put(self, path: str, data: Tuple[np.ndarray, Tuple, Tuple]):
        """Cache image data"""
        if path in self._cache:
            return
        
        # Evict oldest if full
        while len(self._cache) >= self.maxsize:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        
        self._cache[path] = data
        self._order.append(path)
    
    def clear(self):
        """Clear the cache"""
        self._cache.clear()
        self._order.clear()


# Global image cache
_image_cache = ImageCache(maxsize=100)


def load_nifti(path: str, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, Tuple, Tuple]:
    """
    Load NIfTI image with optional caching.
    
    Args:
        path: Path to NIfTI file
        use_cache: Whether to use caching
        
    Returns:
        Tuple of (image_data, affine, spacing, origin)
    """
    if use_cache:
        cached = _image_cache.get(path)
        if cached is not None:
            return cached
    
    # Load with SimpleITK for better metadata handling
    sitk_image = sitk.ReadImage(path)
    image_data = sitk.GetArrayFromImage(sitk_image)  # Returns (D, H, W)
    
    # Get metadata
    spacing = sitk_image.GetSpacing()[::-1]  # Convert to (D, H, W) order
    origin = sitk_image.GetOrigin()[::-1]
    direction = sitk_image.GetDirection()
    
    # Build affine matrix
    nib_img = nib.load(path)
    affine = nib_img.affine
    
    result = (image_data, affine, spacing, origin)
    
    if use_cache:
        _image_cache.put(path, result)
    
    return result


def load_landmarks_csv(csv_path: str, num_landmarks: int = 25) -> Dict[str, List[Landmark]]:
    """
    Load landmarks from CSV file.
    
    Expected CSV format:
        image_id, landmark_id, x, y, z, [is_valid]
    
    Args:
        csv_path: Path to landmarks CSV
        num_landmarks: Number of landmarks per image
        
    Returns:
        Dictionary mapping image_id to list of Landmarks
    """
    df = pd.read_csv(csv_path)
    landmarks = {}
    
    for image_id, group in df.groupby('image_id'):
        lm_list = [Landmark((0.0, 0.0, 0.0), False) for _ in range(num_landmarks)]
        
        for _, row in group.iterrows():
            idx = int(row['landmark_id'])
            if 0 <= idx < num_landmarks:
                is_valid = bool(row.get('is_valid', 1))
                coords = (float(row['x']), float(row['y']), float(row['z']))
                lm_list[idx] = Landmark(coords, is_valid)
        
        landmarks[str(image_id)] = lm_list
    
    return landmarks


def load_verse_landmarks_json(json_path: str, num_landmarks: int = 25) -> List[Landmark]:
    """
    Load landmarks from VerSe19 JSON file (seg-subreg_ctd.json format).
    
    VerSe19 JSON format:
        [
            {"direction": [...], "origin": [...], ...},  # metadata (index 0)
            {"label": 1, "X": 123.4, "Y": 234.5, "Z": 345.6},  # vertebra 1
            {"label": 2, "X": 124.5, "Y": 235.6, "Z": 346.7},  # vertebra 2
            ...
        ]
    
    Label mapping (VerSe uses vertebra labels 1-25):
        1-7: C1-C7 (Cervical)
        8-19: T1-T12 (Thoracic) 
        20-25: L1-L6 (Lumbar)
    
    Args:
        json_path: Path to VerSe landmarks JSON
        num_landmarks: Number of landmarks
        
    Returns:
        List of Landmarks
    """
    import json
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Initialize empty landmarks
    lm_list = [Landmark((0.0, 0.0, 0.0), False) for _ in range(num_landmarks)]
    
    for item in data:
        # Skip metadata entry (has 'direction' key)
        if 'direction' in item or 'label' not in item:
            continue
        
        label = int(item['label'])
        # VerSe uses 1-indexed labels, convert to 0-indexed
        idx = label - 1
        
        if 0 <= idx < num_landmarks:
            # VerSe uses X, Y, Z in world coordinates
            coords = (float(item['X']), float(item['Y']), float(item['Z']))
            lm_list[idx] = Landmark(coords, True)
    
    return lm_list


def load_all_verse_landmarks(data_folder: str, id_list: List[str], num_landmarks: int = 25) -> Dict[str, List[Landmark]]:
    """
    Load landmarks for all subjects from VerSe19 directory structure.
    
    Args:
        data_folder: Root data folder
        id_list: List of subject IDs
        num_landmarks: Number of landmarks per subject
        
    Returns:
        Dictionary mapping subject_id to list of Landmarks
    """
    landmarks = {}
    
    for subject_id in id_list:
        # Try different VerSe19 paths
        verse_patterns = [
            f"dataset-verse19training/derivatives/{subject_id}/{subject_id}_seg-subreg_ctd.json",
            f"dataset-verse19validation/derivatives/{subject_id}/{subject_id}_seg-subreg_ctd.json",
            f"dataset-verse19test/derivatives/{subject_id}/{subject_id}_seg-subreg_ctd.json",
            f"derivatives/{subject_id}/{subject_id}_seg-subreg_ctd.json",
        ]
        
        json_path = None
        for pattern in verse_patterns:
            path = os.path.join(data_folder, pattern)
            if os.path.exists(path):
                json_path = path
                break
        
        if json_path is not None:
            try:
                landmarks[subject_id] = load_verse_landmarks_json(json_path, num_landmarks)
            except Exception as e:
                print(f"Warning: Failed to load landmarks for {subject_id}: {e}")
                landmarks[subject_id] = [Landmark((0.0, 0.0, 0.0), False) for _ in range(num_landmarks)]
        else:
            # No landmarks found - create empty list
            landmarks[subject_id] = [Landmark((0.0, 0.0, 0.0), False) for _ in range(num_landmarks)]
    
    return landmarks


def load_id_list(list_path: str) -> List[str]:
    """Load list of image IDs from text file"""
    with open(list_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def world_to_voxel(
    world_coords: np.ndarray, 
    affine: np.ndarray
) -> np.ndarray:
    """
    Convert world coordinates to voxel coordinates.
    
    Args:
        world_coords: (3,) array of world coordinates
        affine: (4, 4) affine matrix
        
    Returns:
        (3,) array of voxel coordinates
    """
    inv_affine = np.linalg.inv(affine)
    coords_homog = np.append(world_coords, 1.0)
    voxel_coords = inv_affine @ coords_homog
    return voxel_coords[:3]


def voxel_to_world(
    voxel_coords: np.ndarray,
    affine: np.ndarray
) -> np.ndarray:
    """
    Convert voxel coordinates to world coordinates.
    
    Args:
        voxel_coords: (3,) array of voxel coordinates
        affine: (4, 4) affine matrix
        
    Returns:
        (3,) array of world coordinates
    """
    coords_homog = np.append(voxel_coords, 1.0)
    world_coords = affine @ coords_homog
    return world_coords[:3]


class IntensityNormalization:
    """
    Intensity normalization with shift, scale, and clamping.
    Replicates ShiftScaleClamp from MedicalDataAugmentationTool.
    """
    
    def __init__(
        self,
        shift: float = 0.0,
        scale: float = 1.0 / 2048.0,
        random_shift: float = 0.0,
        random_scale: float = 0.0,
        clamp_min: float = -1.0,
        clamp_max: float = 1.0,
        pre_clamp_min: Optional[float] = -1024.0,
        pre_clamp_max: Optional[float] = None
    ):
        self.shift = shift
        self.scale = scale
        self.random_shift = random_shift
        self.random_scale = random_scale
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.pre_clamp_min = pre_clamp_min
        self.pre_clamp_max = pre_clamp_max
    
    def __call__(
        self, 
        image: Union[np.ndarray, torch.Tensor], 
        is_training: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply normalization"""
        # Pre-clamp (clip HU values)
        if self.pre_clamp_min is not None or self.pre_clamp_max is not None:
            if isinstance(image, torch.Tensor):
                image = torch.clamp(
                    image,
                    min=self.pre_clamp_min if self.pre_clamp_min is not None else float('-inf'),
                    max=self.pre_clamp_max if self.pre_clamp_max is not None else float('inf')
                )
            else:
                image = np.clip(
                    image,
                    a_min=self.pre_clamp_min,
                    a_max=self.pre_clamp_max
                )
        
        shift = self.shift
        scale = self.scale
        
        # Add randomness during training
        if is_training:
            if isinstance(image, torch.Tensor):
                shift += (torch.rand(1).item() * 2 - 1) * self.random_shift
                scale *= 1 + (torch.rand(1).item() * 2 - 1) * self.random_scale
            else:
                shift += (np.random.rand() * 2 - 1) * self.random_shift
                scale *= 1 + (np.random.rand() * 2 - 1) * self.random_scale
        
        # Apply shift and scale
        image = (image + shift) * scale
        
        # Final clamp
        if isinstance(image, torch.Tensor):
            image = torch.clamp(image, self.clamp_min, self.clamp_max)
        else:
            image = np.clip(image, self.clamp_min, self.clamp_max)
        
        return image


class RandomGamma:
    """Random gamma augmentation"""
    
    def __init__(self, gamma_range: Tuple[float, float] = (0.9, 1.1)):
        self.gamma_range = gamma_range
    
    def __call__(
        self, 
        image: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        gamma = np.random.uniform(*self.gamma_range)
        
        # Shift to positive range
        if isinstance(image, torch.Tensor):
            min_val = image.min()
            image_shifted = image - min_val
            max_val = image_shifted.max()
            
            if max_val > 0:
                image_norm = image_shifted / max_val
                image_gamma = torch.pow(image_norm, gamma)
                image = image_gamma * max_val + min_val
        else:
            min_val = image.min()
            image_shifted = image - min_val
            max_val = image_shifted.max()
            
            if max_val > 0:
                image_norm = image_shifted / max_val
                image_gamma = np.power(image_norm, gamma)
                image = image_gamma * max_val + min_val
        
        return image


class SpineLocalizationDataset(Dataset):
    """
    Dataset for Stage 1: Spine Localization
    
    Loads full CT volumes resampled to coarse spacing (8mm)
    Generates single-channel spine heatmap as target
    """
    
    def __init__(
        self,
        data_folder: str,
        id_list: List[str],
        landmarks_dict: Dict[str, List[Landmark]],
        image_size: Tuple[int, int, int] = (64, 64, 128),
        image_spacing: Tuple[float, float, float] = (8.0, 8.0, 8.0),
        heatmap_sigma: float = 4.0,
        is_training: bool = True,
        augmentation_config: Optional[dict] = None
    ):
        self.data_folder = data_folder
        self.id_list = id_list
        self.landmarks_dict = landmarks_dict
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.heatmap_sigma = heatmap_sigma
        self.is_training = is_training
        
        # Intensity normalization
        self.normalize = IntensityNormalization(
            scale=1.0 / 2048.0,
            random_shift=0.25 if is_training else 0.0,
            random_scale=0.25 if is_training else 0.0
        )
        
        # Setup augmentations
        self._setup_augmentations(augmentation_config)
    
    def _setup_augmentations(self, config: Optional[dict]):
        """Setup data augmentation transforms"""
        if not HAS_TORCHIO or not self.is_training:
            self.augment = None
            return
        
        if config is None:
            config = {}
        
        transforms = []
        
        # Spatial augmentations
        transforms.append(tio.RandomAffine(
            scales=config.get('random_scale', 0.15),
            degrees=config.get('random_rotation', 15),
            translation=config.get('random_translation', 30),
            default_pad_value=-1024
        ))
        
        if config.get('flip_probability', 0.5) > 0:
            transforms.append(tio.RandomFlip(
                axes=(0,),
                flip_probability=config.get('flip_probability', 0.5)
            ))
        
        # Intensity augmentations
        transforms.append(tio.RandomGamma(log_gamma=(-0.1, 0.1)))
        
        self.augment = tio.Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.id_list)
    
    def _get_image_path(self, image_id: str) -> str:
        """Get image file path from ID for VerSe19 structure"""
        # Try common naming patterns in root
        patterns = [
            f"{image_id}.nii.gz",
            f"{image_id}.nii",
            f"{image_id}_ct.nii.gz",
        ]
        
        for pattern in patterns:
            path = os.path.join(self.data_folder, pattern)
            if os.path.exists(path):
                return path
        
        # VerSe19 structure: dataset-*/rawdata/sub-*/sub-*_ct.nii.gz
        verse_patterns = [
            f"dataset-verse19training/rawdata/{image_id}/{image_id}_ct.nii.gz",
            f"dataset-verse19validation/rawdata/{image_id}/{image_id}_ct.nii.gz",
            f"dataset-verse19test/rawdata/{image_id}/{image_id}_ct.nii.gz",
            f"rawdata/{image_id}/{image_id}_ct.nii.gz",
        ]
        
        for pattern in verse_patterns:
            path = os.path.join(self.data_folder, pattern)
            if os.path.exists(path):
                return path
        
        # Search in subdirectories (fallback)
        for root, _, files in os.walk(self.data_folder):
            for f in files:
                # Match image_id in filename and ensure it's a CT image (not mask)
                if image_id in f and f.endswith('_ct.nii.gz'):
                    return os.path.join(root, f)
                # Also try exact match without _ct suffix
                if f == f"{image_id}.nii.gz" or f == f"{image_id}.nii":
                    return os.path.join(root, f)
        
        raise FileNotFoundError(f"Image not found for ID: {image_id}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_id = self.id_list[idx]
        
        # Load image
        image_path = self._get_image_path(image_id)
        image_data, affine, original_spacing, _ = load_nifti(image_path)
        
        # Convert to float32
        image_data = image_data.astype(np.float32)
        
        # Create TorchIO subject for preprocessing
        if HAS_TORCHIO:
            subject = tio.Subject(
                image=tio.ScalarImage(
                    tensor=image_data[np.newaxis],
                    affine=affine
                )
            )
            
            # Resample to target spacing and size
            resample = tio.Resample(target=self.image_spacing)
            crop_or_pad = tio.CropOrPad(target_shape=self.image_size)
            
            subject = resample(subject)
            subject = crop_or_pad(subject)
            
            # Apply augmentations
            if self.augment is not None:
                subject = self.augment(subject)
            
            image_tensor = subject.image.data[0]  # [D, H, W]
            current_affine = subject.image.affine
        else:
            # Manual resampling using scipy
            from scipy.ndimage import zoom
            
            zoom_factors = np.array(original_spacing) / np.array(self.image_spacing)
            image_resampled = zoom(image_data, zoom_factors, order=1)
            
            # Crop or pad to target size
            image_tensor = self._crop_or_pad(image_resampled, self.image_size)
            image_tensor = torch.from_numpy(image_tensor)
            current_affine = affine
        
        # Apply intensity normalization
        image_tensor = self.normalize(image_tensor, self.is_training)
        
        # Generate spine center heatmap
        landmarks = self.landmarks_dict.get(image_id, [])
        spine_heatmap = self._generate_spine_heatmap(
            landmarks, current_affine
        )
        
        return {
            'image': image_tensor.unsqueeze(0).float(),  # [1, D, H, W]
            'target': spine_heatmap.unsqueeze(0).float(),  # [1, D, H, W]
            'image_id': image_id
        }
    
    def _crop_or_pad(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int, int]
    ) -> np.ndarray:
        """Crop or pad image to target size"""
        result = np.zeros(target_size, dtype=image.dtype)
        
        # Calculate crop/pad for each dimension
        for i in range(3):
            if image.shape[i] >= target_size[i]:
                # Crop (center crop)
                start = (image.shape[i] - target_size[i]) // 2
                slc = slice(start, start + target_size[i])
                if i == 0:
                    image = image[slc, :, :]
                elif i == 1:
                    image = image[:, slc, :]
                else:
                    image = image[:, :, slc]
            else:
                # Pad (center pad)
                pad_before = (target_size[i] - image.shape[i]) // 2
                pad_after = target_size[i] - image.shape[i] - pad_before
                pad_width = [(0, 0)] * 3
                pad_width[i] = (pad_before, pad_after)
                image = np.pad(image, pad_width, mode='constant', constant_values=-1024)
        
        return image[:target_size[0], :target_size[1], :target_size[2]]
    
    def _generate_spine_heatmap(
        self,
        landmarks: List[Landmark],
        affine: np.ndarray
    ) -> torch.Tensor:
        """Generate spine center heatmap from all vertebrae landmarks"""
        D, H, W = self.image_size
        heatmap = torch.zeros(D, H, W)
        
        # Calculate spine center from valid landmarks
        valid_coords = []
        for lm in landmarks:
            if lm.is_valid:
                voxel = world_to_voxel(np.array(lm.coords), affine)
                valid_coords.append(voxel)
        
        if len(valid_coords) == 0:
            return heatmap
        
        # Compute centroid
        centroid = np.mean(valid_coords, axis=0)
        
        # Generate Gaussian heatmap
        z = torch.arange(D, dtype=torch.float32)
        y = torch.arange(H, dtype=torch.float32)
        x = torch.arange(W, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        
        cz, cy, cx = centroid
        dist_sq = (zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2
        heatmap = torch.exp(-dist_sq / (2 * self.heatmap_sigma**2))
        
        return heatmap


class VertebraeLocalizationDataset(Dataset):
    """
    Dataset for Stage 2: Vertebrae Localization
    
    Loads CT volumes cropped around spine center
    Generates 25-channel heatmap as target (one per vertebra)
    """
    
    def __init__(
        self,
        data_folder: str,
        id_list: List[str],
        landmarks_dict: Dict[str, List[Landmark]],
        spine_landmarks_dict: Optional[Dict[str, np.ndarray]] = None,
        image_size: Tuple[int, int, int] = (96, 96, 128),
        image_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        num_landmarks: int = 25,
        heatmap_sigma: float = 4.0,
        is_training: bool = True,
        augmentation_config: Optional[dict] = None
    ):
        self.data_folder = data_folder
        self.id_list = id_list
        self.landmarks_dict = landmarks_dict
        self.spine_landmarks_dict = spine_landmarks_dict or {}
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.num_landmarks = num_landmarks
        self.heatmap_sigma = heatmap_sigma
        self.is_training = is_training
        
        # Intensity normalization
        self.normalize = IntensityNormalization(
            scale=1.0 / 2048.0,
            random_shift=0.25 if is_training else 0.0,
            random_scale=0.25 if is_training else 0.0
        )
        
        # Setup augmentations
        self._setup_augmentations(augmentation_config)
    
    def _setup_augmentations(self, config: Optional[dict]):
        """Setup data augmentation transforms"""
        if not HAS_TORCHIO or not self.is_training:
            self.augment = None
            return
        
        if config is None:
            config = {}
        
        transforms = []
        
        # Spatial augmentations
        transforms.append(tio.RandomAffine(
            scales=config.get('random_scale', 0.15),
            degrees=config.get('random_rotation', 15),
            translation=config.get('random_translation', 30),
            default_pad_value=-1024
        ))
        
        if config.get('flip_probability', 0.5) > 0:
            transforms.append(tio.RandomFlip(
                axes=(0,),
                flip_probability=config.get('flip_probability', 0.5)
            ))
        
        # Elastic deformation - calculate safe max_displacement to avoid folding
        # Grid spacing = image_size / (num_control_points - 1)
        # max_displacement must be < grid_spacing to avoid folding
        if config.get('elastic_deformation', True):
            num_control_points = 5  # Reduced from 7 for larger grid spacing
            # Calculate safe displacement: grid_spacing = min(image_size) / (num_control_points - 1)
            # For image_size (96, 96, 128) with 5 control points: 96/4 = 24, so max_disp < 24
            # Use conservative value of 5 to be safe across all image sizes
            safe_max_displacement = 5.0
            print(f"[DEBUG] Elastic deformation: num_control_points={num_control_points}, max_displacement={safe_max_displacement}")
            transforms.append(tio.RandomElasticDeformation(
                num_control_points=num_control_points,
                max_displacement=safe_max_displacement
            ))
        
        # Intensity augmentations
        transforms.append(tio.RandomGamma(log_gamma=(-0.1, 0.1)))
        
        self.augment = tio.Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.id_list)
    
    def _get_image_path(self, image_id: str) -> str:
        """Get image file path from ID for VerSe19 structure"""
        patterns = [
            f"{image_id}.nii.gz",
            f"{image_id}.nii",
            f"{image_id}_ct.nii.gz",
        ]
        
        for pattern in patterns:
            path = os.path.join(self.data_folder, pattern)
            if os.path.exists(path):
                return path
        
        # VerSe19 structure
        verse_patterns = [
            f"dataset-verse19training/rawdata/{image_id}/{image_id}_ct.nii.gz",
            f"dataset-verse19validation/rawdata/{image_id}/{image_id}_ct.nii.gz",
            f"dataset-verse19test/rawdata/{image_id}/{image_id}_ct.nii.gz",
            f"rawdata/{image_id}/{image_id}_ct.nii.gz",
        ]
        
        for pattern in verse_patterns:
            path = os.path.join(self.data_folder, pattern)
            if os.path.exists(path):
                return path
        
        for root, _, files in os.walk(self.data_folder):
            for f in files:
                if image_id in f and f.endswith('_ct.nii.gz'):
                    return os.path.join(root, f)
        
        raise FileNotFoundError(f"Image not found for ID: {image_id}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_id = self.id_list[idx]
        
        # Load image
        image_path = self._get_image_path(image_id)
        image_data, affine, original_spacing, _ = load_nifti(image_path)
        image_data = image_data.astype(np.float32)
        
        # Get landmarks
        landmarks = self.landmarks_dict.get(image_id, [])
        
        # Get spine center for cropping
        spine_center = self.spine_landmarks_dict.get(image_id)
        if spine_center is None:
            # Compute from landmarks
            valid_coords = [world_to_voxel(np.array(lm.coords), affine) 
                          for lm in landmarks if lm.is_valid]
            if valid_coords:
                spine_center = np.mean(valid_coords, axis=0)
            else:
                spine_center = np.array(image_data.shape) / 2
        
        # Process with TorchIO
        if HAS_TORCHIO:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image_data[np.newaxis], affine=affine)
            )
            
            # Resample and crop around spine center
            resample = tio.Resample(target=self.image_spacing)
            subject = resample(subject)
            
            # Center crop around spine
            crop_or_pad = tio.CropOrPad(target_shape=self.image_size)
            subject = crop_or_pad(subject)
            
            # Apply augmentations
            if self.augment is not None:
                subject = self.augment(subject)
            
            image_tensor = subject.image.data[0]
            current_affine = subject.image.affine
        else:
            from scipy.ndimage import zoom
            
            zoom_factors = np.array(original_spacing) / np.array(self.image_spacing)
            image_resampled = zoom(image_data, zoom_factors, order=1)
            image_tensor = torch.from_numpy(
                self._crop_or_pad(image_resampled, self.image_size)
            )
            current_affine = affine
        
        # Apply intensity normalization
        image_tensor = self.normalize(image_tensor, self.is_training)
        
        # Prepare landmarks tensor
        landmarks_tensor = self._prepare_landmarks(landmarks, current_affine)
        
        return {
            'image': image_tensor.unsqueeze(0).float(),  # [1, D, H, W]
            'landmarks': landmarks_tensor.float(),  # [num_landmarks, 4]
            'image_id': image_id
        }
    
    def _crop_or_pad(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int, int]
    ) -> np.ndarray:
        """Crop or pad image to target size"""
        result = np.zeros(target_size, dtype=image.dtype)
        
        for i in range(3):
            if image.shape[i] >= target_size[i]:
                start = (image.shape[i] - target_size[i]) // 2
                slc = slice(start, start + target_size[i])
                if i == 0:
                    image = image[slc, :, :]
                elif i == 1:
                    image = image[:, slc, :]
                else:
                    image = image[:, :, slc]
            else:
                pad_before = (target_size[i] - image.shape[i]) // 2
                pad_after = target_size[i] - image.shape[i] - pad_before
                pad_width = [(0, 0)] * 3
                pad_width[i] = (pad_before, pad_after)
                image = np.pad(image, pad_width, mode='constant', constant_values=-1024)
        
        return image[:target_size[0], :target_size[1], :target_size[2]]
    
    def _prepare_landmarks(
        self,
        landmarks: List[Landmark],
        affine: np.ndarray
    ) -> torch.Tensor:
        """
        Prepare landmarks tensor.
        
        Format: [num_landmarks, 4]
            [..., 0] = is_valid (0 or 1)
            [..., 1:4] = z, y, x in voxel coordinates
        """
        landmarks_tensor = torch.zeros(self.num_landmarks, 4)
        
        for i, lm in enumerate(landmarks):
            if i >= self.num_landmarks:
                break
            if lm.is_valid:
                voxel_coords = world_to_voxel(np.array(lm.coords), affine)
                landmarks_tensor[i, 0] = 1.0
                landmarks_tensor[i, 1:4] = torch.from_numpy(voxel_coords)
        
        return landmarks_tensor


class VertebraeSegmentationDataset(Dataset):
    """
    Dataset for Stage 3: Vertebrae Segmentation
    
    Loads CT volumes cropped around individual vertebrae
    Loads corresponding segmentation masks as targets
    """
    
    def __init__(
        self,
        data_folder: str,
        id_list: List[str],
        landmarks_dict: Dict[str, List[Landmark]],
        labels_folder: str,
        image_size: Tuple[int, int, int] = (96, 96, 128),
        image_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        num_classes: int = 26,
        is_training: bool = True,
        augmentation_config: Optional[dict] = None
    ):
        self.data_folder = data_folder
        self.labels_folder = labels_folder
        self.id_list = id_list
        self.landmarks_dict = landmarks_dict
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.num_classes = num_classes
        self.is_training = is_training
        
        # Build sample list (image_id, vertebra_idx)
        self.samples = self._build_sample_list()
        
        # Intensity normalization
        self.normalize = IntensityNormalization(
            scale=1.0 / 2048.0,
            random_shift=0.25 if is_training else 0.0,
            random_scale=0.25 if is_training else 0.0
        )
        
        # Setup augmentations
        self._setup_augmentations(augmentation_config)
    
    def _build_sample_list(self) -> List[Tuple[str, int]]:
        """Build list of (image_id, vertebra_idx) samples"""
        samples = []
        for image_id in self.id_list:
            landmarks = self.landmarks_dict.get(image_id, [])
            for i, lm in enumerate(landmarks):
                if lm.is_valid:
                    samples.append((image_id, i))
        return samples
    
    def _setup_augmentations(self, config: Optional[dict]):
        """Setup data augmentation transforms"""
        if not HAS_TORCHIO or not self.is_training:
            self.augment = None
            return
        
        if config is None:
            config = {}
        
        transforms = []
        
        transforms.append(tio.RandomAffine(
            scales=config.get('random_scale', 0.15),
            degrees=config.get('random_rotation', 15),
            translation=config.get('random_translation', 10),
            default_pad_value=-1024
        ))
        
        if config.get('flip_probability', 0.5) > 0:
            transforms.append(tio.RandomFlip(
                axes=(0,),
                flip_probability=config.get('flip_probability', 0.5)
            ))
        
        self.augment = tio.Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _get_paths(self, image_id: str) -> Tuple[str, str]:
        """Get image and label paths for VerSe19 structure"""
        image_path = None
        label_path = None
        
        # Standard patterns in root
        image_patterns = [f"{image_id}.nii.gz", f"{image_id}_ct.nii.gz"]
        label_patterns = [f"{image_id}_seg.nii.gz", f"{image_id}_mask.nii.gz"]
        
        for pattern in image_patterns:
            path = os.path.join(self.data_folder, pattern)
            if os.path.exists(path):
                image_path = path
                break
        
        for pattern in label_patterns:
            path = os.path.join(self.labels_folder, pattern)
            if os.path.exists(path):
                label_path = path
                break
        
        # VerSe19 structure
        if image_path is None:
            verse_img_patterns = [
                f"dataset-verse19training/rawdata/{image_id}/{image_id}_ct.nii.gz",
                f"dataset-verse19validation/rawdata/{image_id}/{image_id}_ct.nii.gz",
                f"dataset-verse19test/rawdata/{image_id}/{image_id}_ct.nii.gz",
            ]
            for pattern in verse_img_patterns:
                path = os.path.join(self.data_folder, pattern)
                if os.path.exists(path):
                    image_path = path
                    break
        
        if label_path is None:
            verse_lbl_patterns = [
                f"dataset-verse19training/derivatives/{image_id}/{image_id}_seg-vert_msk.nii.gz",
                f"dataset-verse19validation/derivatives/{image_id}/{image_id}_seg-vert_msk.nii.gz",
                f"dataset-verse19test/derivatives/{image_id}/{image_id}_seg-vert_msk.nii.gz",
            ]
            for pattern in verse_lbl_patterns:
                path = os.path.join(self.labels_folder, pattern)
                if os.path.exists(path):
                    label_path = path
                    break
        
        # Fallback: search
        if image_path is None:
            for root, _, files in os.walk(self.data_folder):
                for f in files:
                    if image_id in f and f.endswith('_ct.nii.gz'):
                        image_path = os.path.join(root, f)
                        break
                if image_path:
                    break
        
        if label_path is None:
            for root, _, files in os.walk(self.labels_folder):
                for f in files:
                    if image_id in f and 'seg-vert_msk' in f and f.endswith('.nii.gz'):
                        label_path = os.path.join(root, f)
                        break
                if label_path:
                    break
        
        if image_path is None:
            raise FileNotFoundError(f"Image not found for ID: {image_id}")
        if label_path is None:
            raise FileNotFoundError(f"Label not found for ID: {image_id}")
        
        return image_path, label_path
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_id, vertebra_idx = self.samples[idx]
        
        # Load image and label
        image_path, label_path = self._get_paths(image_id)
        image_data, affine, _, _ = load_nifti(image_path)
        label_data, _, _, _ = load_nifti(label_path)
        
        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.int64)
        
        # Get vertebra center for cropping
        landmarks = self.landmarks_dict.get(image_id, [])
        vertebra_center = world_to_voxel(
            np.array(landmarks[vertebra_idx].coords), affine
        )
        
        # Crop around vertebra
        image_crop = self._crop_around_center(
            image_data, vertebra_center, self.image_size, pad_value=-1024
        )
        label_crop = self._crop_around_center(
            label_data, vertebra_center, self.image_size, pad_value=0
        )
        
        # Apply intensity normalization
        image_tensor = self.normalize(
            torch.from_numpy(image_crop), self.is_training
        )
        label_tensor = torch.from_numpy(label_crop)
        
        return {
            'image': image_tensor.unsqueeze(0).float(),  # [1, D, H, W]
            'label': label_tensor.long(),  # [D, H, W]
            'image_id': image_id,
            'vertebra_idx': vertebra_idx
        }
    
    def _crop_around_center(
        self,
        image: np.ndarray,
        center: np.ndarray,
        size: Tuple[int, int, int],
        pad_value: float = 0
    ) -> np.ndarray:
        """Crop image around center point with proper boundary handling"""
        result = np.full(size, pad_value, dtype=image.dtype)
        
        # Ensure center is within image bounds (clamp to valid range)
        center = np.clip(center, [0, 0, 0], np.array(image.shape) - 1)
        
        # Calculate crop bounds
        half_size = np.array(size) // 2
        start = (center - half_size).astype(int)
        end = start + np.array(size)
        
        # Clip to image bounds
        src_start = np.maximum(start, 0)
        src_end = np.minimum(end, image.shape)
        
        # Check if we have a valid region to copy
        if np.any(src_start >= src_end):
            # No valid region, return padded result
            return result
        
        # Calculate destination positions
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        
        # Ensure destination slices are valid
        if np.any(dst_start >= dst_end):
            return result
        
        # Copy data
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


def create_data_loaders(
    config,
    stage: str = 'vertebrae_localization'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders for a specific stage.
    
    Args:
        config: PipelineConfig object
        stage: 'spine_localization', 'vertebrae_localization', or 'vertebrae_segmentation'
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load ID lists
    train_ids = load_id_list(
        os.path.join(config.data.data_folder, config.data.train_list)
    )
    val_ids = load_id_list(
        os.path.join(config.data.data_folder, config.data.val_list)
    )
    
    # Load landmarks
    landmarks_dict = load_landmarks_csv(
        os.path.join(config.data.data_folder, config.data.landmarks_file)
    )
    
    # Create appropriate dataset
    if stage == 'spine_localization':
        stage_config = config.spine_localization
        train_dataset = SpineLocalizationDataset(
            data_folder=config.data.data_folder,
            id_list=train_ids,
            landmarks_dict=landmarks_dict,
            image_size=stage_config.image_size,
            image_spacing=stage_config.image_spacing,
            heatmap_sigma=stage_config.heatmap_sigma,
            is_training=True
        )
        val_dataset = SpineLocalizationDataset(
            data_folder=config.data.data_folder,
            id_list=val_ids,
            landmarks_dict=landmarks_dict,
            image_size=stage_config.image_size,
            image_spacing=stage_config.image_spacing,
            heatmap_sigma=stage_config.heatmap_sigma,
            is_training=False
        )
    elif stage == 'vertebrae_localization':
        stage_config = config.vertebrae_localization
        train_dataset = VertebraeLocalizationDataset(
            data_folder=config.data.data_folder,
            id_list=train_ids,
            landmarks_dict=landmarks_dict,
            image_size=stage_config.image_size,
            image_spacing=stage_config.image_spacing,
            num_landmarks=stage_config.num_landmarks,
            heatmap_sigma=stage_config.heatmap_sigma,
            is_training=True
        )
        val_dataset = VertebraeLocalizationDataset(
            data_folder=config.data.data_folder,
            id_list=val_ids,
            landmarks_dict=landmarks_dict,
            image_size=stage_config.image_size,
            image_spacing=stage_config.image_spacing,
            num_landmarks=stage_config.num_landmarks,
            heatmap_sigma=stage_config.heatmap_sigma,
            is_training=False
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=stage_config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader
