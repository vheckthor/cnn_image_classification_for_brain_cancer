"""
Visualization utilities for model predictions and results.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import seaborn as sns


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 8,
    threshold: float = 0.5,
    save_path: Optional[Path] = None
):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: Trained model
        dataloader: DataLoader for samples
        device: Device to run inference on
        num_samples: Number of samples to visualize
        threshold: Threshold for binarizing predictions
        save_path: Path to save visualization
    """
    model.eval()
    
    # Get a batch
    batch = next(iter(dataloader))
    images = batch['image'].to(device)
    masks = batch['mask'].to(device)
    
    # Limit to num_samples
    images = images[:num_samples]
    masks = masks[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        predictions = model(images)
        predictions_binary = (predictions > threshold).float()
    
    # Move to CPU and convert to numpy
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    predictions = predictions.cpu().numpy()
    predictions_binary = predictions_binary.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(images[i, 0], cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(masks[i, 0], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted mask (probability)
        axes[i, 2].imshow(predictions[i, 0], cmap='hot')
        axes[i, 2].set_title('Prediction (Probability)')
        axes[i, 2].axis('off')
        
        # Predicted mask (binary)
        axes[i, 3].imshow(predictions_binary[i, 0], cmap='gray')
        axes[i, 3].set_title('Prediction (Binary)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_dice_scores: List[float],
    save_path: Optional[Path] = None
):
    """
    Plot training curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_dice_scores: List of validation DSC scores per epoch
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation DSC
    ax2.plot(val_dice_scores, label='Validation DSC', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Similarity Coefficient')
    ax2.set_title('Validation DSC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(
    custom_metrics: dict,
    pretrained_metrics: dict,
    save_path: Optional[Path] = None
):
    """
    Plot comparison of metrics between models.
    
    Args:
        custom_metrics: Metrics dictionary for custom model
        pretrained_metrics: Metrics dictionary for pre-trained model
        save_path: Path to save plot
    """
    metrics_names = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1']
    
    custom_values = [custom_metrics.get(m, 0) for m in metrics_names]
    pretrained_values = [pretrained_metrics.get(m, 0) for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, custom_values, width, label='Custom Model', alpha=0.8)
    bars2 = ax.bar(x + width/2, pretrained_values, width, label='Pre-trained Model', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - Evaluation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_confusion_matrix_plot(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    save_path: Optional[Path] = None
):
    """
    Create and plot confusion matrix.
    
    Args:
        predictions: Predicted masks
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
        save_path: Path to save plot
    """
    predictions_binary = (predictions > threshold).float()
    targets_binary = targets.float()
    
    # Flatten
    pred_flat = predictions_binary.view(-1).cpu().numpy()
    target_flat = targets_binary.view(-1).cpu().numpy()
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(target_flat, pred_flat, labels=[0, 1])
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Background', 'Tumor'],
                yticklabels=['Background', 'Tumor'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

