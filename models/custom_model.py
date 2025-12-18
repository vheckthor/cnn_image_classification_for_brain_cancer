"""
Custom U-Net inspired architecture for brain tumor segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block (encoder path)."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.double_conv(x)
        x = self.pool(skip)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block (decoder path)."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2, bilinear: bool = True):
        super(UpBlock, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatch
        diff_h = skip.size()[2] - x.size()[2]
        diff_w = skip.size()[3] - x.size()[3]
        
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])
        
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class CustomUNet(nn.Module):
    """
    Custom U-Net architecture for brain tumor segmentation.
    
    Architecture:
    - Encoder: 4 downsampling blocks with increasing filters
    - Bottleneck: Deepest layer
    - Decoder: 4 upsampling blocks with skip connections
    - Output: 1-channel binary segmentation mask
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        filters: list = [64, 128, 256, 512, 1024],
        dropout: float = 0.2,
        bilinear: bool = True
    ):
        """
        Initialize custom U-Net.
        
        Args:
            in_channels: Number of input channels (1 for grayscale)
            filters: List of filter sizes for each level
            dropout: Dropout rate
            bilinear: Use bilinear upsampling instead of transpose convolution
        """
        super(CustomUNet, self).__init__()
        
        self.filters = filters
        self.bilinear = bilinear
        
        # Encoder (downsampling path)
        self.down1 = DownBlock(in_channels, filters[0], dropout)
        self.down2 = DownBlock(filters[0], filters[1], dropout)
        self.down3 = DownBlock(filters[1], filters[2], dropout)
        self.down4 = DownBlock(filters[2], filters[3], dropout)
        
        # Bottleneck
        self.bottleneck = DoubleConv(filters[3], filters[4], dropout)
        
        # Decoder (upsampling path)
        self.up1 = UpBlock(filters[4], filters[3], dropout, bilinear)
        self.up2 = UpBlock(filters[3], filters[2], dropout, bilinear)
        self.up3 = UpBlock(filters[2], filters[1], dropout, bilinear)
        self.up4 = UpBlock(filters[1], filters[0], dropout, bilinear)
        
        # Output layer
        self.out_conv = nn.Conv2d(filters[0], 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        
        # Output
        x = self.out_conv(x)
        x = self.sigmoid(x)
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

