"""
Pre-trained model architectures for brain tumor segmentation.
Uses transfer learning with ResNet, VGG, or U-Net backbones.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetUNet(nn.Module):
    """
    U-Net with ResNet encoder for brain tumor segmentation.
    Uses pre-trained ResNet as encoder backbone.
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize ResNet-U-Net.
        
        Args:
            encoder_name: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            pretrained: Use ImageNet pre-trained weights
            num_classes: Number of output classes (1 for binary segmentation)
            dropout: Dropout rate
        """
        super(ResNetUNet, self).__init__()
        
        # Load pre-trained ResNet
        resnet_models = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101
        }
        
        if encoder_name not in resnet_models:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        
        encoder = resnet_models[encoder_name](pretrained=pretrained)
        
        # Encoder layers
        self.encoder0 = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu
        )
        self.encoder1 = nn.Sequential(
            encoder.maxpool,
            encoder.layer1
        )
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4
        
        # Get channel sizes
        if encoder_name in ['resnet18', 'resnet34']:
            channels = [64, 64, 128, 256, 512]
        else:  # resnet50, resnet101
            channels = [64, 256, 512, 1024, 2048]
        
        # Decoder
        self.decoder4 = self._make_decoder_block(channels[4], channels[3], dropout)
        self.decoder3 = self._make_decoder_block(channels[3], channels[2], dropout)
        self.decoder2 = self._make_decoder_block(channels[2], channels[1], dropout)
        self.decoder1 = self._make_decoder_block(channels[1], channels[0], dropout)
        self.decoder0 = self._make_decoder_block(channels[0], 64, dropout)
        
        # Output
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int, dropout: float):
        """Create decoder block with upsampling and skip connections."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
    
    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)  # 64 channels
        e1 = self.encoder1(e0)  # 64/256 channels
        e2 = self.encoder2(e1)  # 128/512 channels
        e3 = self.encoder3(e2)  # 256/1024 channels
        e4 = self.encoder4(e3)  # 512/2048 channels
        
        # Decoder with skip connections
        d4 = self.decoder4(e4)
        d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = self.decoder3(d4)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.decoder2(d3)
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e1], dim=1)
        
        d1 = self.decoder1(d2)
        d1 = F.interpolate(d1, size=e0.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e0], dim=1)
        
        d0 = self.decoder0(d1)
        
        # Output
        out = self.final_conv(d0)
        
        return out


class VGGUNet(nn.Module):
    """
    U-Net with VGG encoder for brain tumor segmentation.
    Uses pre-trained VGG as encoder backbone.
    """
    
    def __init__(
        self,
        encoder_name: str = "vgg16",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize VGG-U-Net.
        
        Args:
            encoder_name: VGG variant ('vgg11', 'vgg13', 'vgg16', 'vgg19')
            pretrained: Use ImageNet pre-trained weights
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(VGGUNet, self).__init__()
        
        # Load pre-trained VGG
        vgg_models = {
            'vgg11': models.vgg11,
            'vgg13': models.vgg13,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19
        }
        
        if encoder_name not in vgg_models:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        
        vgg = vgg_models[encoder_name](pretrained=pretrained)
        features = list(vgg.features.children())
        
        # Encoder blocks
        self.encoder1 = nn.Sequential(*features[0:4])   # 64 channels
        self.encoder2 = nn.Sequential(*features[4:9])   # 128 channels
        self.encoder3 = nn.Sequential(*features[9:16])  # 256 channels
        self.encoder4 = nn.Sequential(*features[16:23]) # 512 channels
        self.encoder5 = nn.Sequential(*features[23:30]) # 512 channels
        
        # Decoder
        self.decoder5 = self._make_decoder_block(512, 512, dropout)
        self.decoder4 = self._make_decoder_block(512, 256, dropout)
        self.decoder3 = self._make_decoder_block(256, 128, dropout)
        self.decoder2 = self._make_decoder_block(128, 64, dropout)
        self.decoder1 = self._make_decoder_block(64, 64, dropout)
        
        # Output
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int, dropout: float):
        """Create decoder block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # 64
        e2 = self.encoder2(e1)  # 128
        e3 = self.encoder3(e2)  # 256
        e4 = self.encoder4(e3)  # 512
        e5 = self.encoder5(e4)  # 512
        
        # Decoder with skip connections
        d5 = self.decoder5(e5)
        d5 = F.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d5 = torch.cat([d5, e4], dim=1)
        
        d4 = self.decoder4(d5)
        d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = self.decoder3(d4)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.decoder2(d3)
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e1], dim=1)
        
        d1 = self.decoder1(d2)
        
        # Output
        out = self.final_conv(d1)
        
        return out


def get_pretrained_model(
    model_type: str = "resnet",
    encoder_name: str = "resnet50",
    pretrained: bool = True,
    num_classes: int = 1,
    dropout: float = 0.2
):
    """
    Factory function to get pre-trained model.
    
    Args:
        model_type: Model type ('resnet', 'vgg')
        encoder_name: Encoder name (e.g., 'resnet50', 'vgg16')
        pretrained: Use pre-trained weights
        num_classes: Number of output classes
        dropout: Dropout rate
    
    Returns:
        Model instance
    """
    if model_type.lower() == "resnet":
        return ResNetUNet(encoder_name, pretrained, num_classes, dropout)
    elif model_type.lower() == "vgg":
        return VGGUNet(encoder_name, pretrained, num_classes, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

