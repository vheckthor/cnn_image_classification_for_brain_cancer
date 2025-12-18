"""
Configuration management for brain tumor segmentation project.
"""
import os
from pathlib import Path

class Config:
    """Project configuration settings."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"
    SPLITS_DIR = DATA_DIR / "splits"
    
    MODELS_DIR = PROJECT_ROOT / "models"
    CUSTOM_MODELS_DIR = MODELS_DIR / "custom"
    PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
    SAVED_MODELS_DIR = MODELS_DIR / "saved"
    
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Dataset paths
    HDF5_DATASET_PATH = PROJECT_ROOT / "dataset" / "brain_tumor_dataset.h5"
    
    # Data settings
    IMAGE_SIZE = 256  # Standardize to 256x256
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    RANDOM_SEED = 42
    
    # Preprocessing settings
    DENOISING_METHOD = "bilateral"  # Options: gaussian, median, bilateral, nlm, wavelet
    NORMALIZATION_METHOD = "z_score"  # Options: z_score, min_max, histogram_eq, clahe
    
    # Data augmentation settings
    AUGMENTATION_ENABLED = True
    ROTATION_RANGE = 15  # degrees
    TRANSLATION_RANGE = 0.1  # 10% of image size
    SCALE_RANGE = (0.9, 1.1)
    BRIGHTNESS_RANGE = 0.1  # ±10%
    CONTRAST_RANGE = 0.1  # ±10%
    FLIP_PROBABILITY = 0.5
    
    # Training settings
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 10
    REDUCE_LR_FACTOR = 0.5
    MIN_LEARNING_RATE = 1e-7
    
    # Model settings
    NUM_CLASSES = 1  # Binary segmentation
    DROPOUT_RATE = 0.2
    
    # Custom model settings
    CUSTOM_MODEL_FILTERS = [64, 128, 256, 512, 1024]
    
    # Pre-trained model settings
    PRETRAINED_MODEL_TYPE = "unet"  # Options: unet, resnet, vgg
    PRETRAINED_ENCODER = "resnet50"  # For ResNet-based models
    FREEZE_ENCODER_EPOCHS = 5  # Freeze encoder for first N epochs
    
    # Evaluation settings
    EVAL_METRICS = ["dice", "iou", "accuracy", "sensitivity", "specificity", "precision", "f1"]
    
    # Hardware settings
    NUM_WORKERS = 4
    PIN_MEMORY = True
    USE_MIXED_PRECISION = True
    
    # Logging settings
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    VISUALIZE_INTERVAL = 50  # Visualize predictions every N batches
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories."""
        directories = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
            cls.EXTERNAL_DATA_DIR, cls.SPLITS_DIR,
            cls.MODELS_DIR, cls.CUSTOM_MODELS_DIR, cls.PRETRAINED_MODELS_DIR,
            cls.SAVED_MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name: str, model_type: str = "custom"):
        """Get path for saving/loading model."""
        if model_type == "custom":
            return cls.SAVED_MODELS_DIR / f"{model_name}_custom.pth"
        else:
            return cls.SAVED_MODELS_DIR / f"{model_name}_pretrained.pth"

