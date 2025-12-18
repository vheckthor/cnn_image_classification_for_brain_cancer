"""
Image normalization methods for MRI brain images.
"""
import numpy as np
import cv2
from typing import Tuple


def z_score_normalize(image: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Z-score normalization: (pixel - mean) / std
    
    Args:
        image: Input image
    
    Returns:
        Normalized image, mean, std
    """
    mean = np.mean(image)
    std = np.std(image)
    
    if std == 0:
        return image, mean, std
    
    normalized = (image - mean) / std
    return normalized, mean, std


def min_max_normalize(image: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Min-max normalization: (pixel - min) / (max - min)
    
    Args:
        image: Input image
    
    Returns:
        Normalized image (0-1 range), min, max
    """
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val == min_val:
        return image, min_val, max_val
    
    normalized = (image - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    """
    Histogram equalization for contrast enhancement.
    
    Args:
        image: Input image (0-255 range expected)
    
    Returns:
        Equalized image
    """
    # Convert to uint8 if needed
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    
    equalized = cv2.equalizeHist(image_uint8)
    
    # Convert back to float
    return equalized.astype(np.float32) / 255.0 if image.max() <= 1.0 else equalized.astype(np.float32)


def clahe_normalize(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image: Input image (0-255 range expected)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        CLAHE normalized image
    """
    # Convert to uint8 if needed
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    normalized = clahe.apply(image_uint8)
    
    # Convert back to float
    return normalized.astype(np.float32) / 255.0 if image.max() <= 1.0 else normalized.astype(np.float32)


def apply_normalization(
    image: np.ndarray,
    method: str = "z_score",
    **kwargs
) -> np.ndarray:
    """
    Apply normalization using specified method.
    
    Args:
        image: Input image
        method: Normalization method ('z_score', 'min_max', 'histogram_eq', 'clahe')
        **kwargs: Additional arguments for specific normalization method
    
    Returns:
        Normalized image
    """
    methods = {
        'z_score': lambda img: z_score_normalize(img)[0],
        'min_max': lambda img: min_max_normalize(img)[0],
        'histogram_eq': histogram_equalize,
        'clahe': clahe_normalize
    }
    
    if method not in methods:
        raise ValueError(f"Unknown normalization method: {method}. Choose from {list(methods.keys())}")
    
    return methods[method](image, **kwargs)

