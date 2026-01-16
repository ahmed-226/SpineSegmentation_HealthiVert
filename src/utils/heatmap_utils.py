"""
Utility functions for heatmap generation and landmark processing
PyTorch Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


def generate_heatmap_target(
    shape: Tuple[int, int, int],
    landmarks: torch.Tensor,
    sigmas: torch.Tensor,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate Gaussian heatmaps from landmark coordinates.
    
    Args:
        shape: (D, H, W) output heatmap shape
        landmarks: [B, num_landmarks, 4] tensor
                   [..., 0] = is_valid (0 or 1)
                   [..., 1:4] = z, y, x coordinates in voxel space
        sigmas: [num_landmarks] sigmas per landmark (or scalar)
        device: Target device
        
    Returns:
        heatmaps: [B, num_landmarks, D, H, W] tensor
    """
    if device is None:
        device = landmarks.device
    
    B, L, _ = landmarks.shape
    D, H, W = shape
    
    # Ensure sigmas has correct shape
    if sigmas.dim() == 0:
        sigmas = sigmas.expand(L)
    
    # Create coordinate grids: [D, H, W, 3]
    coords = torch.stack(torch.meshgrid(
        torch.arange(D, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    ), dim=-1)  # [D, H, W, 3]
    
    # Reshape for broadcasting
    coords = coords.view(1, 1, D, H, W, 3)  # [1, 1, D, H, W, 3]
    
    # Landmark centers: [B, L, 1, 1, 1, 3]
    centers = landmarks[:, :, 1:4].view(B, L, 1, 1, 1, 3)
    
    # Compute squared distances: [B, L, D, H, W]
    dist_sq = ((coords - centers) ** 2).sum(dim=-1)
    
    # Apply Gaussian: [B, L, D, H, W]
    sigmas_sq = (sigmas.view(1, L, 1, 1, 1) ** 2).to(device)
    heatmaps = torch.exp(-dist_sq / (2 * sigmas_sq + 1e-8))
    
    # Mask invalid landmarks
    valid = landmarks[:, :, 0].view(B, L, 1, 1, 1)
    heatmaps = heatmaps * valid
    
    return heatmaps


def generate_single_heatmap(
    shape: Tuple[int, int, int],
    center: torch.Tensor,
    sigma: float,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate a single Gaussian heatmap.
    
    Args:
        shape: (D, H, W) output shape
        center: (3,) tensor with z, y, x coordinates
        sigma: Gaussian sigma
        device: Target device
        
    Returns:
        heatmap: [D, H, W] tensor
    """
    if device is None:
        device = center.device
    
    D, H, W = shape
    
    # Create coordinate grids
    z = torch.arange(D, device=device, dtype=torch.float32)
    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    
    # Compute distance
    cz, cy, cx = center[0], center[1], center[2]
    dist_sq = (zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2
    
    # Gaussian
    heatmap = torch.exp(-dist_sq / (2 * sigma**2))
    
    return heatmap


class LearnableSigmas(nn.Module):
    """
    Learnable sigma parameters for heatmap generation.
    One sigma per landmark, with optional regularization.
    """
    
    def __init__(
        self,
        num_landmarks: int = 25,
        initial_sigma: float = 4.0,
        min_sigma: float = 1.0,
        max_sigma: float = 10.0
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        
        # Learnable raw sigmas (unconstrained)
        self.raw_sigmas = nn.Parameter(
            torch.full((num_landmarks,), initial_sigma)
        )
    
    @property
    def sigmas(self) -> torch.Tensor:
        """Get constrained sigmas"""
        # Clamp to valid range
        return torch.clamp(self.raw_sigmas, self.min_sigma, self.max_sigma)
    
    def forward(self) -> torch.Tensor:
        return self.sigmas
    
    def regularization_loss(
        self, 
        valid_mask: Optional[torch.Tensor] = None,
        weight: float = 0.00001
    ) -> torch.Tensor:
        """
        Compute regularization loss for sigmas.
        
        Args:
            valid_mask: [B, num_landmarks] or [num_landmarks] mask of valid landmarks
            weight: Regularization weight
            
        Returns:
            Scalar loss
        """
        sigmas = self.sigmas
        
        if valid_mask is not None:
            if valid_mask.dim() == 2:
                # Sum over batch
                valid_count = valid_mask.sum(dim=0)
                loss = weight * torch.sum(sigmas ** 2 * valid_count)
            else:
                loss = weight * torch.sum(sigmas ** 2 * valid_mask)
        else:
            loss = weight * torch.sum(sigmas ** 2)
        
        return loss


def extract_landmark_from_heatmap(
    heatmap: torch.Tensor,
    method: str = 'argmax'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract landmark coordinates from heatmap.
    
    Args:
        heatmap: [D, H, W] heatmap tensor
        method: 'argmax' or 'weighted_mean'
        
    Returns:
        coords: (3,) tensor with z, y, x coordinates
        confidence: Scalar confidence value
    """
    D, H, W = heatmap.shape
    
    if method == 'argmax':
        # Find maximum
        flat_idx = torch.argmax(heatmap)
        z = flat_idx // (H * W)
        y = (flat_idx % (H * W)) // W
        x = flat_idx % W
        
        coords = torch.stack([z.float(), y.float(), x.float()])
        confidence = heatmap[z, y, x]
    
    elif method == 'weighted_mean':
        # Weighted centroid
        z_coords = torch.arange(D, device=heatmap.device, dtype=torch.float32)
        y_coords = torch.arange(H, device=heatmap.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=heatmap.device, dtype=torch.float32)
        
        # Normalize heatmap
        heatmap_sum = heatmap.sum() + 1e-8
        heatmap_norm = heatmap / heatmap_sum
        
        # Compute weighted average
        z = (heatmap_norm.sum(dim=(1, 2)) * z_coords).sum()
        y = (heatmap_norm.sum(dim=(0, 2)) * y_coords).sum()
        x = (heatmap_norm.sum(dim=(0, 1)) * x_coords).sum()
        
        coords = torch.stack([z, y, x])
        confidence = heatmap.max()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return coords, confidence


def extract_all_landmarks(
    heatmaps: torch.Tensor,
    method: str = 'argmax',
    threshold: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract landmark coordinates from all heatmap channels.
    
    Args:
        heatmaps: [num_landmarks, D, H, W] or [B, num_landmarks, D, H, W]
        method: 'argmax' or 'weighted_mean'
        threshold: Minimum confidence threshold
        
    Returns:
        landmarks: [num_landmarks, 4] or [B, num_landmarks, 4]
                   [..., 0] = is_valid
                   [..., 1:4] = z, y, x coordinates
        confidences: [num_landmarks] or [B, num_landmarks]
    """
    if heatmaps.dim() == 4:
        L, D, H, W = heatmaps.shape
        
        landmarks = torch.zeros(L, 4, device=heatmaps.device)
        confidences = torch.zeros(L, device=heatmaps.device)
        
        for i in range(L):
            coords, conf = extract_landmark_from_heatmap(heatmaps[i], method)
            landmarks[i, 1:4] = coords
            landmarks[i, 0] = 1.0 if conf > threshold else 0.0
            confidences[i] = conf
        
        return landmarks, confidences
    
    elif heatmaps.dim() == 5:
        B, L, D, H, W = heatmaps.shape
        
        landmarks = torch.zeros(B, L, 4, device=heatmaps.device)
        confidences = torch.zeros(B, L, device=heatmaps.device)
        
        for b in range(B):
            for i in range(L):
                coords, conf = extract_landmark_from_heatmap(heatmaps[b, i], method)
                landmarks[b, i, 1:4] = coords
                landmarks[b, i, 0] = 1.0 if conf > threshold else 0.0
                confidences[b, i] = conf
        
        return landmarks, confidences
    
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {heatmaps.dim()}D")


def gaussian_smooth_3d(
    image: torch.Tensor,
    sigma: float = 2.0,
    kernel_size: Optional[int] = None
) -> torch.Tensor:
    """
    Apply 3D Gaussian smoothing to a tensor.
    
    Args:
        image: [B, C, D, H, W] or [C, D, H, W] or [D, H, W] tensor
        sigma: Gaussian sigma
        kernel_size: Kernel size (auto-computed if None)
        
    Returns:
        Smoothed tensor with same shape
    """
    original_dim = image.dim()
    
    # Add batch and channel dimensions if needed
    if original_dim == 3:
        image = image.unsqueeze(0).unsqueeze(0)
    elif original_dim == 4:
        image = image.unsqueeze(0)
    
    B, C, D, H, W = image.shape
    
    # Compute kernel size
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - kernel_size // 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    # Create separable 3D kernel (apply 1D convolution in each direction)
    padding = kernel_size // 2
    
    # Reshape for group convolution
    image_flat = image.view(B * C, 1, D, H, W)
    
    # Apply separable convolution
    # Z direction
    kernel_z = gauss_1d.view(1, 1, -1, 1, 1)
    out = F.conv3d(image_flat, kernel_z, padding=(padding, 0, 0))
    
    # Y direction
    kernel_y = gauss_1d.view(1, 1, 1, -1, 1)
    out = F.conv3d(out, kernel_y, padding=(0, padding, 0))
    
    # X direction
    kernel_x = gauss_1d.view(1, 1, 1, 1, -1)
    out = F.conv3d(out, kernel_x, padding=(0, 0, padding))
    
    out = out.view(B, C, D, H, W)
    
    # Remove added dimensions
    if original_dim == 3:
        out = out.squeeze(0).squeeze(0)
    elif original_dim == 4:
        out = out.squeeze(0)
    
    return out


def find_local_maxima_3d(
    heatmap: torch.Tensor,
    min_distance: int = 3,
    threshold: float = 0.05
) -> List[Tuple[torch.Tensor, float]]:
    """
    Find local maxima in a 3D heatmap.
    
    Args:
        heatmap: [D, H, W] tensor
        min_distance: Minimum distance between maxima
        threshold: Minimum value threshold
        
    Returns:
        List of (coords, value) tuples
    """
    # Convert to numpy for scipy operations
    heatmap_np = heatmap.detach().cpu().numpy()
    
    from scipy.ndimage import maximum_filter, label
    
    # Apply maximum filter
    max_filtered = maximum_filter(heatmap_np, size=min_distance * 2 + 1)
    
    # Find local maxima
    maxima_mask = (heatmap_np == max_filtered) & (heatmap_np > threshold)
    
    # Get coordinates
    coords = np.argwhere(maxima_mask)
    values = heatmap_np[maxima_mask]
    
    # Sort by value (descending)
    sorted_idx = np.argsort(-values)
    
    results = []
    for idx in sorted_idx:
        coord = torch.tensor(coords[idx], dtype=torch.float32, device=heatmap.device)
        value = float(values[idx])
        results.append((coord, value))
    
    return results


def compute_landmark_distance_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> torch.Tensor:
    """
    Compute Euclidean distance error between predicted and target landmarks.
    
    Args:
        predicted: [B, L, 4] or [L, 4] predicted landmarks
        target: [B, L, 4] or [L, 4] target landmarks
        spacing: (sz, sy, sx) voxel spacing in mm
        
    Returns:
        distances: [B, L] or [L] Euclidean distances in mm
    """
    spacing = torch.tensor(spacing, device=predicted.device)
    
    # Get coordinates
    pred_coords = predicted[..., 1:4]  # z, y, x
    target_coords = target[..., 1:4]
    
    # Get valid masks
    pred_valid = predicted[..., 0]
    target_valid = target[..., 0]
    both_valid = (pred_valid > 0.5) & (target_valid > 0.5)
    
    # Compute distances in mm
    diff = (pred_coords - target_coords) * spacing
    distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    
    # Mask invalid
    distances = distances * both_valid.float()
    
    return distances


def interpolate_missing_landmarks(
    landmarks: torch.Tensor,
    max_gap: int = 2
) -> torch.Tensor:
    """
    Interpolate missing landmarks from neighbors.
    
    Args:
        landmarks: [L, 4] landmarks tensor
        max_gap: Maximum gap size to interpolate
        
    Returns:
        interpolated: [L, 4] landmarks with missing values filled
    """
    result = landmarks.clone()
    L = landmarks.shape[0]
    
    valid = landmarks[:, 0] > 0.5
    
    for i in range(L):
        if valid[i]:
            continue
        
        # Find nearest valid neighbors
        prev_valid = None
        next_valid = None
        
        for j in range(i - 1, -1, -1):
            if valid[j]:
                prev_valid = j
                break
        
        for j in range(i + 1, L):
            if valid[j]:
                next_valid = j
                break
        
        # Check if we can interpolate
        if prev_valid is not None and next_valid is not None:
            gap = next_valid - prev_valid - 1
            if gap <= max_gap:
                # Linear interpolation
                t = (i - prev_valid) / (next_valid - prev_valid)
                
                prev_coords = landmarks[prev_valid, 1:4]
                next_coords = landmarks[next_valid, 1:4]
                
                interp_coords = prev_coords + t * (next_coords - prev_coords)
                
                result[i, 0] = 1.0
                result[i, 1:4] = interp_coords
    
    return result


if __name__ == "__main__":
    # Test heatmap generation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Test generate_heatmap_target
    print("\n=== Testing generate_heatmap_target ===")
    shape = (96, 96, 128)
    B, L = 2, 25
    
    landmarks = torch.zeros(B, L, 4, device=device)
    # Set some valid landmarks
    for b in range(B):
        for i in range(0, L, 2):
            landmarks[b, i, 0] = 1.0
            landmarks[b, i, 1] = 48 + i  # z
            landmarks[b, i, 2] = 48      # y
            landmarks[b, i, 3] = 64      # x
    
    sigmas = torch.full((L,), 4.0, device=device)
    
    heatmaps = generate_heatmap_target(shape, landmarks, sigmas, device)
    print(f"Landmarks shape: {landmarks.shape}")
    print(f"Heatmaps shape: {heatmaps.shape}")
    print(f"Heatmap range: [{heatmaps.min():.4f}, {heatmaps.max():.4f}]")
    
    # Test LearnableSigmas
    print("\n=== Testing LearnableSigmas ===")
    learnable_sigmas = LearnableSigmas(num_landmarks=25, initial_sigma=4.0)
    sigmas = learnable_sigmas()
    reg_loss = learnable_sigmas.regularization_loss()
    print(f"Sigmas shape: {sigmas.shape}")
    print(f"Sigmas range: [{sigmas.min():.2f}, {sigmas.max():.2f}]")
    print(f"Regularization loss: {reg_loss:.6f}")
    
    # Test extract_all_landmarks
    print("\n=== Testing extract_all_landmarks ===")
    extracted, confidences = extract_all_landmarks(heatmaps[0], method='argmax')
    print(f"Extracted landmarks shape: {extracted.shape}")
    print(f"Confidences shape: {confidences.shape}")
    
    # Test gaussian_smooth_3d
    print("\n=== Testing gaussian_smooth_3d ===")
    test_heatmap = heatmaps[0, 0]  # [D, H, W]
    smoothed = gaussian_smooth_3d(test_heatmap, sigma=2.0)
    print(f"Input shape: {test_heatmap.shape}")
    print(f"Output shape: {smoothed.shape}")
    
    # Test compute_landmark_distance_error
    print("\n=== Testing compute_landmark_distance_error ===")
    noisy_landmarks = landmarks.clone()
    noisy_landmarks[:, :, 1:4] += torch.randn_like(landmarks[:, :, 1:4]) * 2
    
    distances = compute_landmark_distance_error(
        noisy_landmarks, landmarks, spacing=(2.0, 2.0, 2.0)
    )
    print(f"Distance errors shape: {distances.shape}")
    print(f"Mean distance error: {distances[distances > 0].mean():.2f} mm")
    
    print("\n=== All tests passed! ===")
