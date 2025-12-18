"""
Evaluation script for comparing custom and pre-trained models.
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config
from utils.data_loader import load_data_splits, create_dataloaders
from preprocessing.augmentation import PreprocessingTransform
from models.custom_model import CustomUNet
from models.pretrained_model import get_pretrained_model
from evaluation.metrics import evaluate_model, compute_all_metrics


def load_model(model_path: Path, model_type: str, device: torch.device, config):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    if model_type == "custom":
        model = CustomUNet(
            in_channels=1,
            filters=config.CUSTOM_MODEL_FILTERS,
            dropout=config.DROPOUT_RATE,
            bilinear=True
        ).to(device)
    else:  # pretrained
        if config.PRETRAINED_MODEL_TYPE.lower() == "resnet":
            model_type_name = "resnet"
            encoder_name = config.PRETRAINED_ENCODER
        else:
            model_type_name = "vgg"
            encoder_name = "vgg16"
        
        model = get_pretrained_model(
            model_type=model_type_name,
            encoder_name=encoder_name,
            pretrained=False,  # Don't load ImageNet weights, we have trained weights
            num_classes=1,
            dropout=config.DROPOUT_RATE
        ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def evaluate_single_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_name: str
) -> dict:
    """Evaluate a single model on test set."""
    print(f"\nEvaluating {model_name}...")
    
    metrics = evaluate_model(model, test_loader, device)
    
    print(f"\n{model_name} Results:")
    print("-" * 60)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")
    print("-" * 60)
    
    return metrics


def main():
    """Main evaluation function."""
    # Initialize configuration
    config = Config()
    config.create_directories()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data splits
    splits = load_data_splits(config.SPLITS_DIR)
    
    # Create preprocessing transform
    preprocessing = PreprocessingTransform(
        target_size=config.IMAGE_SIZE,
        denoising_method=config.DENOISING_METHOD,
        normalization_method=config.NORMALIZATION_METHOD
    )
    
    # Create test data loader
    from utils.data_loader import create_dataloaders
    dataloaders = create_dataloaders(
        config.HDF5_DATASET_PATH,
        splits,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_transform=preprocessing,
        val_transform=preprocessing,
        return_label=False
    )
    
    test_loader = dataloaders['test']
    
    results = {}
    
    # Evaluate custom model
    custom_model_path = config.SAVED_MODELS_DIR / "custom_unet_best.pth"
    if custom_model_path.exists():
        custom_model, custom_checkpoint = load_model(
            custom_model_path, "custom", device, config
        )
        custom_metrics = evaluate_single_model(
            custom_model, test_loader, device, "Custom Model"
        )
        results['custom'] = {
            'metrics': custom_metrics,
            'epoch': custom_checkpoint.get('epoch', 'unknown'),
            'best_val_dice': custom_checkpoint.get('best_val_dice', 'unknown')
        }
    else:
        print(f"Custom model not found at {custom_model_path}")
    
    # Evaluate pre-trained model
    pretrained_model_path = config.SAVED_MODELS_DIR / f"pretrained_{config.PRETRAINED_MODEL_TYPE}_{config.PRETRAINED_ENCODER}_best.pth"
    if pretrained_model_path.exists():
        pretrained_model, pretrained_checkpoint = load_model(
            pretrained_model_path, "pretrained", device, config
        )
        pretrained_metrics = evaluate_single_model(
            pretrained_model, test_loader, device, "Pre-trained Model"
        )
        results['pretrained'] = {
            'metrics': pretrained_metrics,
            'epoch': pretrained_checkpoint.get('epoch', 'unknown'),
            'best_val_dice': pretrained_checkpoint.get('best_val_dice', 'unknown')
        }
    else:
        print(f"Pre-trained model not found at {pretrained_model_path}")
    
    # Comparison
    if 'custom' in results and 'pretrained' in results:
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        
        comparison = {}
        for metric in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1']:
            custom_val = results['custom']['metrics'][metric]
            pretrained_val = results['pretrained']['metrics'][metric]
            diff = pretrained_val - custom_val
            comparison[metric] = {
                'custom': custom_val,
                'pretrained': pretrained_val,
                'difference': diff,
                'winner': 'pretrained' if diff > 0 else 'custom'
            }
        
        print("\nMetric Comparison:")
        print("-" * 60)
        print(f"{'Metric':<15} {'Custom':<12} {'Pre-trained':<12} {'Difference':<12} {'Winner':<12}")
        print("-" * 60)
        for metric, comp in comparison.items():
            print(f"{metric.upper():<15} {comp['custom']:<12.4f} {comp['pretrained']:<12.4f} "
                  f"{comp['difference']:+.4f}        {comp['winner']:<12}")
        print("-" * 60)
        
        # Determine overall winner
        dice_diff = comparison['dice']['difference']
        if abs(dice_diff) < 0.01:
            winner = "Tie (difference < 0.01)"
        else:
            winner = "Pre-trained Model" if dice_diff > 0 else "Custom Model"
        
        print(f"\nOverall Winner (based on DSC): {winner}")
        print(f"DSC Difference: {dice_diff:+.4f}")
        
        results['comparison'] = comparison
        results['overall_winner'] = winner
    
    # Save results
    results_path = config.RESULTS_DIR / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    main()

