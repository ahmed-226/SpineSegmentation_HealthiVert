"""
3D U-Net and SpatialConfigurationNet architectures for VerSe2019 Pipeline
PyTorch Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class ConvBlock3D(nn.Module):
    """
    Basic 3D convolutional block with batch normalization and activation.
    
    Structure: Conv3D -> BatchNorm -> Activation -> Conv3D -> BatchNorm -> Activation -> Dropout
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        num_convs: int = 2,
        activation: str = 'leaky_relu',
        dropout_ratio: float = 0.0,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_convs):
            # Convolution
            layers.append(nn.Conv3d(
                current_channels, out_channels, kernel_size, 
                padding=padding, bias=not use_batch_norm
            ))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm3d(out_channels))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == 'selu':
                layers.append(nn.SELU(inplace=True))
            elif activation == 'elu':
                layers.append(nn.ELU(inplace=True))
            
            current_channels = out_channels
        
        # Dropout (only at the end of the block)
        if dropout_ratio > 0:
            if activation == 'selu':
                layers.append(nn.AlphaDropout(dropout_ratio))
            else:
                layers.append(nn.Dropout3d(dropout_ratio))
        
        self.block = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights(activation)
    
    def _init_weights(self, activation: str):
        """Initialize weights using He/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if activation in ['relu', 'leaky_relu']:
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', 
                        nonlinearity='leaky_relu' if activation == 'leaky_relu' else 'relu'
                    )
                elif activation == 'selu':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    """
    3D U-Net with average pooling downsampling and trilinear upsampling.
    
    Architecture follows the VerSe2019 implementation:
    - 4 encoder levels with doubling filters
    - Average pooling for downsampling
    - Trilinear interpolation for upsampling
    - Skip connections via concatenation
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: Optional[int] = None,
        num_filters_base: int = 64,
        num_levels: int = 4,
        num_convs_per_level: int = 2,
        kernel_size: int = 3,
        activation: str = 'leaky_relu',
        dropout_ratio: float = 0.0,
        use_batch_norm: bool = True,
        return_features: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (None = num_filters_base)
            num_filters_base: Base number of filters (doubled at each level)
            num_levels: Number of encoder/decoder levels
            num_convs_per_level: Convolutions per level
            kernel_size: Convolution kernel size
            activation: Activation function ('relu', 'leaky_relu', 'selu')
            dropout_ratio: Dropout probability
            use_batch_norm: Whether to use batch normalization
            return_features: If True, return features without final conv
        """
        super().__init__()
        
        self.num_levels = num_levels
        self.return_features = return_features
        
        if out_channels is None:
            out_channels = num_filters_base
        
        # Calculate filter sizes for each level
        self.filters = [num_filters_base * (2 ** i) for i in range(num_levels)]
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        current_channels = in_channels
        
        for level in range(num_levels):
            block = ConvBlock3D(
                in_channels=current_channels,
                out_channels=self.filters[level],
                kernel_size=kernel_size,
                num_convs=num_convs_per_level,
                activation=activation,
                dropout_ratio=dropout_ratio if level == num_levels - 1 else 0.0,
                use_batch_norm=use_batch_norm
            )
            self.encoder_blocks.append(block)
            current_channels = self.filters[level]
        
        # Downsampling: Average pooling
        self.downsample = nn.AvgPool3d(kernel_size=2, stride=2)
        
        # Decoder path
        self.decoder_blocks = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()
        
        for level in range(num_levels - 2, -1, -1):
            # Upsampling is done with trilinear interpolation
            # Then we need a conv to match channels before concat
            upsample_conv = nn.Conv3d(
                self.filters[level + 1], self.filters[level],
                kernel_size=1, bias=False
            )
            self.upsample_convs.append(upsample_conv)
            
            # Decoder block: receives concatenated features
            block = ConvBlock3D(
                in_channels=self.filters[level] * 2,  # Skip + upsampled
                out_channels=self.filters[level],
                kernel_size=kernel_size,
                num_convs=num_convs_per_level,
                activation=activation,
                dropout_ratio=0.0,
                use_batch_norm=use_batch_norm
            )
            self.decoder_blocks.append(block)
        
        # Output convolution (if not returning features)
        if not return_features:
            self.output_conv = nn.Conv3d(self.filters[0], out_channels, kernel_size=1)
        
        self.out_channels = self.filters[0] if return_features else out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape [B, C, D, H, W]
            
        Returns:
            Output tensor of shape [B, out_channels, D, H, W]
        """
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder path
        for level in range(self.num_levels):
            x = self.encoder_blocks[level](x)
            
            if level < self.num_levels - 1:
                encoder_outputs.append(x)
                x = self.downsample(x)
        
        # Decoder path
        for i, level in enumerate(range(self.num_levels - 2, -1, -1)):
            # Upsample
            x = F.interpolate(
                x, scale_factor=2, mode='trilinear', align_corners=True
            )
            x = self.upsample_convs[i](x)
            
            # Get skip connection
            skip = encoder_outputs[level]
            
            # Handle size mismatch (can happen due to pooling)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
            
            # Concatenate and apply decoder block
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[i](x)
        
        # Output
        if not self.return_features:
            x = self.output_conv(x)
        
        return x


class SpatialConfigurationNet(nn.Module):
    """
    Spatial Configuration Network for vertebrae localization.
    
    Dual-pathway architecture:
    1. Local Appearance Network: U-Net for local features
    2. Spatial Configuration Network: Downsampled U-Net for global context
    
    Final output is the element-wise product of local and spatial heatmaps.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_labels: int = 25,
        num_filters_base: int = 96,
        num_levels: int = 4,
        spatial_downsample: int = 4,
        dropout_ratio: float = 0.25,
        activation: str = 'leaky_relu',
        local_activation: str = 'tanh',
        spatial_activation: str = 'tanh',
        use_batch_norm: bool = True
    ):
        """
        Args:
            in_channels: Input channels (1 for CT)
            num_labels: Number of landmarks (25 vertebrae)
            num_filters_base: Base filters for U-Net
            num_levels: Number of U-Net levels
            spatial_downsample: Downsampling factor for spatial path
            dropout_ratio: Dropout probability
            activation: Internal activation function
            local_activation: Output activation for local heatmaps
            spatial_activation: Output activation for spatial heatmaps
            use_batch_norm: Use batch normalization
        """
        super().__init__()
        
        self.spatial_downsample = spatial_downsample
        self.num_labels = num_labels
        
        # Local appearance pathway
        self.local_unet = UNet3D(
            in_channels=in_channels,
            num_filters_base=num_filters_base,
            num_levels=num_levels,
            dropout_ratio=dropout_ratio,
            activation=activation,
            use_batch_norm=use_batch_norm,
            return_features=True
        )
        
        self.local_conv = nn.Conv3d(
            self.local_unet.out_channels, num_labels, kernel_size=1
        )
        
        # Local activation
        if local_activation == 'tanh':
            self.local_activation = nn.Tanh()
        elif local_activation == 'sigmoid':
            self.local_activation = nn.Sigmoid()
        else:
            self.local_activation = nn.Identity()
        
        # Spatial configuration pathway
        # Uses fewer levels since input is downsampled
        spatial_levels = max(2, num_levels - 1)
        spatial_filters = num_filters_base // 2
        
        self.spatial_unet = UNet3D(
            in_channels=num_labels,  # Takes local heatmaps as input
            num_filters_base=spatial_filters,
            num_levels=spatial_levels,
            dropout_ratio=dropout_ratio,
            activation=activation,
            use_batch_norm=use_batch_norm,
            return_features=True
        )
        
        self.spatial_conv = nn.Conv3d(
            self.spatial_unet.out_channels, num_labels, kernel_size=1
        )
        
        # Spatial activation
        if spatial_activation == 'tanh':
            self.spatial_activation = nn.Tanh()
        elif spatial_activation == 'sigmoid':
            self.spatial_activation = nn.Sigmoid()
        else:
            self.spatial_activation = nn.Identity()
        
        # Initialize output convolutions
        self._init_output_convs()
    
    def _init_output_convs(self):
        """Initialize output convolution layers"""
        for conv in [self.local_conv, self.spatial_conv]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='linear')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Spatial Configuration Network.
        
        Args:
            x: Input tensor of shape [B, 1, D, H, W]
            
        Returns:
            Tuple of:
                - heatmaps: Final combined heatmaps [B, num_labels, D, H, W]
                - local_heatmaps: Local appearance heatmaps
                - spatial_heatmaps: Spatial configuration heatmaps
        """
        # Local appearance pathway
        local_features = self.local_unet(x)
        local_heatmaps = self.local_activation(self.local_conv(local_features))
        
        # Downsample for spatial context
        spatial_input = F.avg_pool3d(
            local_heatmaps, 
            kernel_size=self.spatial_downsample,
            stride=self.spatial_downsample
        )
        
        # Spatial configuration pathway
        spatial_features = self.spatial_unet(spatial_input)
        spatial_heatmaps_low = self.spatial_activation(
            self.spatial_conv(spatial_features)
        )
        
        # Upsample spatial heatmaps to original resolution
        target_size = x.shape[2:]  # (D, H, W)
        spatial_heatmaps = F.interpolate(
            spatial_heatmaps_low,
            size=target_size,
            mode='trilinear',
            align_corners=True
        )
        
        # Combine: element-wise product
        heatmaps = local_heatmaps * spatial_heatmaps
        
        return heatmaps, local_heatmaps, spatial_heatmaps


class SimpleUNet(nn.Module):
    """
    Simple U-Net wrapper for spine localization and segmentation.
    
    Adds output convolution with optional heatmap initialization.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_labels: int = 1,
        num_filters_base: int = 64,
        num_levels: int = 4,
        dropout_ratio: float = 0.0,
        activation: str = 'leaky_relu',
        heatmap_initialization: bool = False,
        use_batch_norm: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels
            num_labels: Number of output labels
            num_filters_base: Base number of filters
            num_levels: Number of U-Net levels
            dropout_ratio: Dropout probability
            activation: Activation function
            heatmap_initialization: Use special init for heatmap output
            use_batch_norm: Use batch normalization
        """
        super().__init__()
        
        self.unet = UNet3D(
            in_channels=in_channels,
            num_filters_base=num_filters_base,
            num_levels=num_levels,
            dropout_ratio=dropout_ratio,
            activation=activation,
            use_batch_norm=use_batch_norm,
            return_features=True
        )
        
        self.output_conv = nn.Conv3d(
            self.unet.out_channels, num_labels, kernel_size=1
        )
        
        # Initialize output conv
        if heatmap_initialization:
            # Special initialization for heatmap prediction
            # Start with low activation (bias = -5)
            nn.init.trunc_normal_(self.output_conv.weight, std=0.0001)
            nn.init.constant_(self.output_conv.bias, -5.0)
        else:
            nn.init.kaiming_normal_(
                self.output_conv.weight, mode='fan_in', nonlinearity='linear'
            )
            nn.init.zeros_(self.output_conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            
        Returns:
            Output tensor [B, num_labels, D, H, W]
        """
        features = self.unet(x)
        return self.output_conv(features)


class SegmentationUNet(nn.Module):
    """
    U-Net for vertebrae segmentation with softmax output.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 26,  # Background + 25 vertebrae
        num_filters_base: int = 64,
        num_levels: int = 4,
        dropout_ratio: float = 0.0,
        activation: str = 'leaky_relu',
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.unet = UNet3D(
            in_channels=in_channels,
            num_filters_base=num_filters_base,
            num_levels=num_levels,
            dropout_ratio=dropout_ratio,
            activation=activation,
            use_batch_norm=use_batch_norm,
            return_features=True
        )
        
        self.output_conv = nn.Conv3d(
            self.unet.out_channels, num_classes, kernel_size=1
        )
        
        # Initialize
        nn.init.kaiming_normal_(
            self.output_conv.weight, mode='fan_in', nonlinearity='linear'
        )
        nn.init.zeros_(self.output_conv.bias)
    
    def forward(
        self, x: torch.Tensor, return_logits: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            return_logits: If True, return logits; else return softmax
            
        Returns:
            Output tensor [B, num_classes, D, H, W]
        """
        features = self.unet(x)
        logits = self.output_conv(features)
        
        if return_logits:
            return logits
        else:
            return F.softmax(logits, dim=1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the networks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Test input
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 96, 96, 128).to(device)
    
    # Test UNet3D
    print("\n=== Testing UNet3D ===")
    unet = UNet3D(
        in_channels=1,
        num_filters_base=64,
        num_levels=4,
        return_features=True
    ).to(device)
    output = unet(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {count_parameters(unet):,}")
    
    # Test SpatialConfigurationNet
    print("\n=== Testing SpatialConfigurationNet ===")
    scn = SpatialConfigurationNet(
        in_channels=1,
        num_labels=25,
        num_filters_base=96,
        num_levels=4,
        spatial_downsample=4,
        dropout_ratio=0.25
    ).to(device)
    heatmaps, local, spatial = scn(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Heatmaps shape: {heatmaps.shape}")
    print(f"Local heatmaps shape: {local.shape}")
    print(f"Spatial heatmaps shape: {spatial.shape}")
    print(f"Parameters: {count_parameters(scn):,}")
    
    # Test SimpleUNet for spine localization
    print("\n=== Testing SimpleUNet (Spine Localization) ===")
    spine_net = SimpleUNet(
        in_channels=1,
        num_labels=1,
        num_filters_base=64,
        heatmap_initialization=True
    ).to(device)
    output = spine_net(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {count_parameters(spine_net):,}")
    
    # Test SegmentationUNet
    print("\n=== Testing SegmentationUNet ===")
    seg_net = SegmentationUNet(
        in_channels=1,
        num_classes=26,
        num_filters_base=64
    ).to(device)
    output = seg_net(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {count_parameters(seg_net):,}")
    
    print("\n=== All tests passed! ===")
