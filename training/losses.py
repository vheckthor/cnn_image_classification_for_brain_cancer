"""
Loss functions for brain tumor segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Directly optimizes Dice Similarity Coefficient.
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.
        
        Args:
            predictions: Predicted masks (B, 1, H, W) with values in [0, 1]
            targets: Ground truth masks (B, 1, H, W) with values in [0, 1]
        
        Returns:
            Dice loss value
        """
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute intersection and union
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        # Return 1 - dice (loss to minimize)
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross-Entropy Loss.
    Often provides better training stability.
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        """
        Initialize Combined Loss.
        
        Args:
            dice_weight: Weight for Dice Loss
            bce_weight: Weight for BCE Loss
            smooth: Smoothing factor for Dice Loss
        """
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            predictions: Predicted masks
            targets: Ground truth masks
        
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        
        return self.dice_weight * dice + self.bce_weight * bce


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Useful when tumor regions are small compared to background.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            predictions: Predicted masks
            targets: Ground truth masks
        
        Returns:
            Focal loss value
        """
        # Compute BCE
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Compute p_t
        p_t = predictions * targets + (1 - predictions) * (1 - targets)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()

