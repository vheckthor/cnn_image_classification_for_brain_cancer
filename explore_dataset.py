"""
Script to explore the brain tumor dataset.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config
from utils.data_loader import HDF5DatasetExplorer, create_data_splits
import h5py


def visualize_samples(hdf5_path: Path, num_samples: int = 6):
    """Visualize sample images and masks from the dataset."""
    explorer = HDF5DatasetExplorer(hdf5_path)
    stats = explorer.explore()
    
    patient_ids = stats['patient_ids']
    
    # Get random samples
    np.random.seed(42)
    sample_ids = np.random.choice(patient_ids, min(num_samples, len(patient_ids)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    label_names = {1: 'Meningioma', 2: 'Glioma', 3: 'Pituitary'}
    
    with h5py.File(hdf5_path, 'r') as f:
        for idx, patient_id in enumerate(sample_ids):
            group = f[patient_id]
            image = np.array(group['image'])
            mask = np.array(group['tumor_mask'])
            label = int(group['label'][()])
            
            # Original image
            axes[idx, 0].imshow(image, cmap='gray')
            axes[idx, 0].set_title(f'Patient: {patient_id}\nImage Shape: {image.shape}')
            axes[idx, 0].axis('off')
            
            # Mask
            axes[idx, 1].imshow(mask, cmap='gray')
            axes[idx, 1].set_title('Tumor Mask')
            axes[idx, 1].axis('off')
            
            # Overlay
            overlay = image.copy()
            if image.max() > 1.0:
                overlay = overlay / 255.0
            overlay_colored = plt.cm.gray(overlay)
            mask_binary = (mask > 0.5).astype(float)
            overlay_colored[:, :, 0] = np.maximum(overlay_colored[:, :, 0], mask_binary * 0.5)
            
            axes[idx, 2].imshow(overlay_colored)
            axes[idx, 2].set_title(f'Overlay\nLabel: {label_names[label]}')
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    output_path = Config.RESULTS_DIR / "dataset_samples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSample visualizations saved to: {output_path}")
    plt.close()


def main():
    """Main exploration function."""
    config = Config()
    config.create_directories()
    
    print("=" * 60)
    print("Brain Tumor Dataset Exploration")
    print("=" * 60)
    
    # Explore dataset
    explorer = HDF5DatasetExplorer(config.HDF5_DATASET_PATH)
    explorer.print_summary()
    
    # Visualize samples
    print("\n" + "=" * 60)
    print("Visualizing Sample Images")
    print("=" * 60)
    visualize_samples(config.HDF5_DATASET_PATH, num_samples=6)
    
    # Create splits if they don't exist
    if not (config.SPLITS_DIR / "train_ids.json").exists():
        print("\n" + "=" * 60)
        print("Creating Data Splits")
        print("=" * 60)
        create_data_splits(
            config.HDF5_DATASET_PATH,
            config.SPLITS_DIR,
            train_ratio=config.TRAIN_SPLIT,
            val_ratio=config.VAL_SPLIT,
            test_ratio=config.TEST_SPLIT,
            random_seed=config.RANDOM_SEED,
            stratify=True
        )
    else:
        print("\nData splits already exist. Skipping split creation.")
    
    print("\n" + "=" * 60)
    print("Exploration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

