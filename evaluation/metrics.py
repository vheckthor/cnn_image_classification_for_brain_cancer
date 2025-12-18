"""
Evaluation metrics for brain tumor segmentation.
Primary metric: Dice Similarity Coefficient (DSC)
"""
import torch
import numpy as np
from typing import Dict, List, Optional


def dice_coefficient(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Compute Dice Similarity Coefficient (DSC).
    
    Args:
        predictions: Predicted masks (B, 1, H, W) with values in [0, 1]
        targets: Ground truth masks (B, 1, H, W) with values in [0, 1]
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient (0-1, higher is better)
    """
    # Binarize predictions
    predictions_binary = (predictions > threshold).float()
    targets_binary = targets.float()
    
    # Flatten tensors
    predictions_flat = predictions_binary.view(-1)
    targets_flat = targets_binary.view(-1)
    
    # Compute intersection and union
    intersection = (predictions_flat * targets_flat).sum()
    dice = (2.0 * intersection + smooth) / (
        predictions_flat.sum() + targets_flat.sum() + smooth
    )
    
    return dice.item()


def iou_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Compute Intersection over Union (IoU).
    
    Args:
        predictions: Predicted masks
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
    
    Returns:
        IoU score (0-1, higher is better)
    """
    # Binarize predictions
    predictions_binary = (predictions > threshold).float()
    targets_binary = targets.float()
    
    # Flatten tensors
    predictions_flat = predictions_binary.view(-1)
    targets_flat = targets_binary.view(-1)
    
    # Compute intersection and union
    intersection = (predictions_flat * targets_flat).sum()
    union = predictions_flat.sum() + targets_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def pixel_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute pixel accuracy.
    
    Args:
        predictions: Predicted masks
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
    
    Returns:
        Pixel accuracy (0-1, higher is better)
    """
    predictions_binary = (predictions > threshold).float()
    targets_binary = targets.float()
    
    correct = (predictions_binary == targets_binary).float()
    accuracy = correct.sum() / correct.numel()
    
    return accuracy.item()


def sensitivity(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Compute sensitivity (recall, true positive rate).
    
    Args:
        predictions: Predicted masks
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
    
    Returns:
        Sensitivity (0-1, higher is better)
    """
    predictions_binary = (predictions > threshold).float()
    targets_binary = targets.float()
    
    # True positives and false negatives
    tp = ((predictions_binary == 1) & (targets_binary == 1)).float().sum()
    fn = ((predictions_binary == 0) & (targets_binary == 1)).float().sum()
    
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    
    return sensitivity.item()


def specificity(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Compute specificity (true negative rate).
    
    Args:
        predictions: Predicted masks
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
    
    Returns:
        Specificity (0-1, higher is better)
    """
    predictions_binary = (predictions > threshold).float()
    targets_binary = targets.float()
    
    # True negatives and false positives
    tn = ((predictions_binary == 0) & (targets_binary == 0)).float().sum()
    fp = ((predictions_binary == 1) & (targets_binary == 0)).float().sum()
    
    specificity = (tn + smooth) / (tn + fp + smooth)
    
    return specificity.item()


def precision_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Compute precision (positive predictive value).
    
    Args:
        predictions: Predicted masks
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
    
    Returns:
        Precision (0-1, higher is better)
    """
    predictions_binary = (predictions > threshold).float()
    targets_binary = targets.float()
    
    # True positives and false positives
    tp = ((predictions_binary == 1) & (targets_binary == 1)).float().sum()
    fp = ((predictions_binary == 1) & (targets_binary == 0)).float().sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    
    return precision.item()


def f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Compute F1 score (harmonic mean of precision and recall).
    
    Args:
        predictions: Predicted masks
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
    
    Returns:
        F1 score (0-1, higher is better)
    """
    prec = precision_score(predictions, targets, threshold, smooth)
    sens = sensitivity(predictions, targets, threshold, smooth)
    
    f1 = (2 * prec * sens + smooth) / (prec + sens + smooth)
    
    return f1


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Predicted masks
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
    
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'dice': dice_coefficient(predictions, targets, threshold),
        'iou': iou_score(predictions, targets, threshold),
        'accuracy': pixel_accuracy(predictions, targets, threshold),
        'sensitivity': sensitivity(predictions, targets, threshold),
        'specificity': specificity(predictions, targets, threshold),
        'precision': precision_score(predictions, targets, threshold),
        'f1': f1_score(predictions, targets, threshold)
    }
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        threshold: Threshold for binarizing predictions
    
    Returns:
        Dictionary of average metrics
    """
    model.eval()
    
    all_metrics = {
        'dice': [],
        'iou': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f1': []
    }
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute metrics for this batch
            batch_metrics = compute_all_metrics(predictions, masks, threshold)
            
            # Accumulate
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])
    
    # Compute averages
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    return avg_metrics

