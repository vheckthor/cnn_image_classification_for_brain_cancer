"""
Training script for pre-trained brain tumor segmentation model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config
from utils.data_loader import (
    HDF5DatasetExplorer,
    create_data_splits,
    load_data_splits,
    create_dataloaders
)
from preprocessing.augmentation import AugmentationTransform, PreprocessingTransform
from models.pretrained_model import get_pretrained_model
from training.losses import DiceLoss, CombinedLoss
from training.trainer import Trainer


def main():
    """Main training function."""
    # Initialize configuration
    config = Config()
    config.create_directories()
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed_all(config.RANDOM_SEED)
    import numpy as np
    np.random.seed(config.RANDOM_SEED)
    import random
    random.seed(config.RANDOM_SEED)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Explore dataset
    print("\n" + "="*60)
    print("Exploring Dataset")
    print("="*60)
    explorer = HDF5DatasetExplorer(config.HDF5_DATASET_PATH)
    explorer.print_summary()
    
    # Create or load data splits
    splits_dir = config.SPLITS_DIR
    if not (splits_dir / "train_ids.json").exists():
        print("\n" + "="*60)
        print("Creating Data Splits")
        print("="*60)
        splits = create_data_splits(
            config.HDF5_DATASET_PATH,
            splits_dir,
            train_ratio=config.TRAIN_SPLIT,
            val_ratio=config.VAL_SPLIT,
            test_ratio=config.TEST_SPLIT,
            random_seed=config.RANDOM_SEED,
            stratify=True
        )
    else:
        print("\n" + "="*60)
        print("Loading Existing Data Splits")
        print("="*60)
        splits = load_data_splits(splits_dir)
        for split_name, ids in splits.items():
            print(f"{split_name.upper()}: {len(ids)} patients")
    
    # Create transforms
    print("\n" + "="*60)
    print("Setting up Preprocessing and Augmentation")
    print("="*60)
    
    preprocessing = PreprocessingTransform(
        target_size=config.IMAGE_SIZE,
        denoising_method=config.DENOISING_METHOD,
        normalization_method=config.NORMALIZATION_METHOD
    )
    
    augmentation = AugmentationTransform(
        rotation_range=config.ROTATION_RANGE,
        translation_range=config.TRANSLATION_RANGE,
        scale_range=config.SCALE_RANGE,
        brightness_range=config.BRIGHTNESS_RANGE,
        contrast_range=config.CONTRAST_RANGE,
        flip_probability=config.FLIP_PROBABILITY
    ) if config.AUGMENTATION_ENABLED else None
    
    # Compose transforms
    def train_transform(sample):
        sample = preprocessing(sample)
        if augmentation:
            sample = augmentation(sample)
        return sample
    
    # Create data loaders
    print("\n" + "="*60)
    print("Creating Data Loaders")
    print("="*60)
    dataloaders = create_dataloaders(
        config.HDF5_DATASET_PATH,
        splits,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_transform=train_transform,
        val_transform=preprocessing,
        return_label=False
    )
    
    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches: {len(dataloaders['val'])}")
    print(f"Test batches: {len(dataloaders['test'])}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating Pre-trained Model")
    print("="*60)
    
    # Determine model type and encoder
    if config.PRETRAINED_MODEL_TYPE.lower() == "resnet":
        model_type = "resnet"
        encoder_name = config.PRETRAINED_ENCODER
    elif config.PRETRAINED_MODEL_TYPE.lower() == "vgg":
        model_type = "vgg"
        encoder_name = "vgg16"  # Default VGG
    else:
        model_type = "resnet"
        encoder_name = "resnet50"
    
    model = get_pretrained_model(
        model_type=model_type,
        encoder_name=encoder_name,
        pretrained=True,
        num_classes=1,
        dropout=config.DROPOUT_RATE
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = DiceLoss(smooth=1e-6)
    # Alternative: CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    
    # Optimizer with differential learning rates
    # Lower LR for encoder (pre-trained), higher for decoder (new)
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': config.LEARNING_RATE * 0.1},  # 10x lower LR for encoder
        {'params': decoder_params, 'lr': config.LEARNING_RATE}
    ], weight_decay=config.WEIGHT_DECAY)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
        model_name=f"pretrained_{model_type}_{encoder_name}"
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    trainer.train(num_epochs=config.NUM_EPOCHS)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation DSC: {trainer.best_val_dice:.4f}")
    print(f"Model saved to: {config.SAVED_MODELS_DIR}")


if __name__ == "__main__":
    main()

