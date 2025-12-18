"""
HDF5 dataset loader for brain tumor MRI images.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

from utils.config import Config


class BrainTumorDataset(Dataset):
    """PyTorch Dataset for brain tumor HDF5 data."""
    
    def __init__(
        self,
        hdf5_path: Path,
        patient_ids: List[str],
        transform=None,
        return_label: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            hdf5_path: Path to HDF5 file
            patient_ids: List of patient group IDs to use
            transform: Optional transform to apply to image and mask
            return_label: Whether to return tumor type label
        """
        self.hdf5_path = hdf5_path
        self.patient_ids = patient_ids
        self.transform = transform
        self.return_label = return_label
        
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        patient_id = self.patient_ids[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[patient_id]
            image = np.array(group['image'], dtype=np.float32)
            mask = np.array(group['tumor_mask'], dtype=np.float32)
            
            if self.return_label:
                try:
                    label_data = group['label']
                    # Handle different HDF5 dataset structures
                    if hasattr(label_data, 'dtype') and label_data.dtype.names:
                        # Compound type - access first field
                        label = int(label_data[0])
                    elif label_data.shape == ():
                        # Scalar dataset
                        label = int(label_data[()])
                    else:
                        # Array dataset - take first element
                        label = int(label_data[0])
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Warning: Could not read label for {patient_id}: {e}")
                    label = 0
            else:
                label = None
        
        # Normalize image to [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0
        
        # Ensure mask is binary [0, 1]
        mask = (mask > 0.5).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform({'image': image, 'mask': mask})
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        result = {
            'image': image,
            'mask': mask,
            'patient_id': patient_id
        }
        
        if self.return_label:
            result['label'] = label
        
        return result


class HDF5DatasetExplorer:
    """Utility class to explore HDF5 dataset structure."""
    
    def __init__(self, hdf5_path: Path):
        self.hdf5_path = hdf5_path
    
    def explore(self) -> Dict:
        """Explore dataset and return statistics."""
        stats = {
            'total_patients': 0,
            'patient_ids': [],
            'image_shapes': [],
            'mask_shapes': [],
            'labels': [],
            'label_distribution': {}
        }
        
        with h5py.File(self.hdf5_path, 'r') as f:
            patient_ids = list(f.keys())
            stats['total_patients'] = len(patient_ids)
            stats['patient_ids'] = patient_ids
            
            for patient_id in patient_ids:
                group = f[patient_id]
                
                # Get image shape
                image = group['image']
                stats['image_shapes'].append(image.shape)
                
                # Get mask shape
                mask = group['tumor_mask']
                stats['mask_shapes'].append(mask.shape)
                
                # Get label
                try:
                    label_data = group['label']
                    # Handle different HDF5 dataset structures
                    if hasattr(label_data, 'dtype') and label_data.dtype.names:
                        # Compound type - access first field
                        label = int(label_data[0])
                    elif label_data.shape == ():
                        # Scalar dataset
                        label = int(label_data[()])
                    else:
                        # Array dataset - take first element
                        label = int(label_data[0])
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Warning: Could not read label for {patient_id}: {e}")
                    label = 0
                stats['labels'].append(label)
                
                # Update label distribution
                label_name = {1: 'Meningioma', 2: 'Glioma', 3: 'Pituitary'}[label]
                stats['label_distribution'][label_name] = \
                    stats['label_distribution'].get(label_name, 0) + 1
        
        return stats
    
    def print_summary(self):
        """Print dataset summary."""
        stats = self.explore()
        
        print("=" * 60)
        print("Dataset Summary")
        print("=" * 60)
        print(f"Total Patients: {stats['total_patients']}")
        print(f"\nLabel Distribution:")
        for label, count in stats['label_distribution'].items():
            percentage = (count / stats['total_patients']) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        print(f"\nImage Shapes:")
        unique_shapes = set(stats['image_shapes'])
        for shape in unique_shapes:
            count = stats['image_shapes'].count(shape)
            print(f"  {shape}: {count} images")
        
        print(f"\nMask Shapes:")
        unique_shapes = set(stats['mask_shapes'])
        for shape in unique_shapes:
            count = stats['mask_shapes'].count(shape)
            print(f"  {shape}: {count} masks")
        print("=" * 60)


def create_data_splits(
    hdf5_path: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True
) -> Dict[str, List[str]]:
    """
    Create train/validation/test splits with optional stratification.
    
    Args:
        hdf5_path: Path to HDF5 file
        output_dir: Directory to save split files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        stratify: Whether to stratify by tumor type label
    
    Returns:
        Dictionary with 'train', 'val', 'test' patient ID lists
    """
    # Load all patient IDs and labels
    patient_ids = []
    labels = []
    
    with h5py.File(hdf5_path, 'r') as f:
        for patient_id in f.keys():
            patient_ids.append(patient_id)
            try:
                label_data = f[patient_id]['label']
                # Handle different HDF5 dataset structures
                if hasattr(label_data, 'dtype') and label_data.dtype.names:
                    # Compound type - access first field
                    label = int(label_data[0])
                elif label_data.shape == ():
                    # Scalar dataset
                    label = int(label_data[()])
                else:
                    # Array dataset - take first element
                    label = int(label_data[0])
            except (KeyError, ValueError, TypeError) as e:
                print(f"Warning: Could not read label for {patient_id}: {e}")
                label = 0
            labels.append(label)
    
    patient_ids = np.array(patient_ids)
    labels = np.array(labels)
    
    # Create splits
    if stratify:
        # First split: train vs (val + test)
        train_ids, temp_ids, train_labels, temp_labels = train_test_split(
            patient_ids, labels,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
            stratify=labels
        )
        
        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_ids, test_ids, val_labels, test_labels = train_test_split(
            temp_ids, temp_labels,
            test_size=(1 - val_ratio_adjusted),
            random_state=random_seed,
            stratify=temp_labels
        )
    else:
        # Random split without stratification
        train_ids, temp_ids = train_test_split(
            patient_ids,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed
        )
        
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1 - val_ratio_adjusted),
            random_state=random_seed
        )
    
    splits = {
        'train': train_ids.tolist(),
        'val': val_ids.tolist(),
        'test': test_ids.tolist()
    }
    
    # Save splits to JSON files
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ids in splits.items():
        output_path = output_dir / f"{split_name}_ids.json"
        with open(output_path, 'w') as f:
            json.dump(ids, f, indent=2)
    
    # Print split statistics
    print("=" * 60)
    print("Data Split Summary")
    print("=" * 60)
    for split_name, ids in splits.items():
        print(f"{split_name.upper()}: {len(ids)} patients ({len(ids)/len(patient_ids)*100:.1f}%)")
    print("=" * 60)
    
    return splits


def load_data_splits(splits_dir: Path) -> Dict[str, List[str]]:
    """Load previously created data splits."""
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_path = splits_dir / f"{split_name}_ids.json"
        if split_path.exists():
            with open(split_path, 'r') as f:
                splits[split_name] = json.load(f)
        else:
            raise FileNotFoundError(f"Split file not found: {split_path}")
    return splits


def create_dataloaders(
    hdf5_path: Path,
    splits: Dict[str, List[str]],
    batch_size: int = 16,
    num_workers: int = 4,
    train_transform=None,
    val_transform=None,
    return_label: bool = False
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    datasets = {
        'train': BrainTumorDataset(
            hdf5_path, splits['train'],
            transform=train_transform,
            return_label=return_label
        ),
        'val': BrainTumorDataset(
            hdf5_path, splits['val'],
            transform=val_transform,
            return_label=return_label
        ),
        'test': BrainTumorDataset(
            hdf5_path, splits['test'],
            transform=val_transform,
            return_label=return_label
        )
    }
    
    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return dataloaders

