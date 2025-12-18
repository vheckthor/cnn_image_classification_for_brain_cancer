"""
Image denoising methods for MRI brain images.
"""
import numpy as np
import cv2
from typing import Optional
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import gaussian, median
from scipy import ndimage


def gaussian_denoise(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian filtering for denoising.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        Denoised image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return gaussian(image, sigma=sigma, mode='reflect')


def median_denoise(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Apply median filtering for denoising.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of median filter kernel (must be odd)
    
    Returns:
        Denoised image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return median(image, selem=np.ones((kernel_size, kernel_size)))


def bilateral_denoise(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Apply bilateral filtering for edge-preserving denoising.
    
    Args:
        image: Input grayscale image (0-255 range)
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
    
    Returns:
        Denoised image
    """
    # Convert to uint8 if needed
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    
    denoised = cv2.bilateralFilter(
        image_uint8, d, sigma_color, sigma_space
    )
    
    # Convert back to float
    return denoised.astype(np.float32) / 255.0 if image.max() <= 1.0 else denoised.astype(np.float32)


def nlm_denoise(
    image: np.ndarray,
    patch_size: int = 5,
    patch_distance: int = 6,
    h: Optional[float] = None
) -> np.ndarray:
    """
    Apply Non-Local Means denoising.
    
    Args:
        image: Input grayscale image
        patch_size: Size of patches used for denoising
        patch_distance: Max distance to search for patches
        h: Cut-off distance for the exponential function (auto-estimated if None)
    
    Returns:
        Denoised image
    """
    # Estimate noise if h not provided
    if h is None:
        sigma_est = estimate_sigma(image, multichannel=False)
        h = 0.8 * sigma_est
    
    denoised = denoise_nl_means(
        image,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        multichannel=False,
        fast_mode=True
    )
    
    return denoised


def wavelet_denoise(
    image: np.ndarray,
    wavelet: str = 'db4',
    mode: str = 'soft',
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Apply wavelet denoising.
    
    Args:
        image: Input grayscale image
        wavelet: Wavelet type (e.g., 'db4', 'haar', 'bior2.2')
        mode: Thresholding mode ('soft' or 'hard')
        threshold: Threshold value (auto-estimated if None)
    
    Returns:
        Denoised image
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("pywt (PyWavelets) is required for wavelet denoising")
    
    # Decompose
    coeffs = pywt.wavedec2(image, wavelet, mode='symmetric')
    
    # Estimate threshold if not provided
    if threshold is None:
        # Use universal threshold
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(image.size))
    
    # Threshold coefficients
    coeffs_thresh = list(coeffs)
    coeffs_thresh[0] = pywt.threshold(coeffs[0], threshold, mode=mode)
    for i in range(1, len(coeffs)):
        coeffs_thresh[i] = tuple(
            pywt.threshold(detail, threshold, mode=mode)
            for detail in coeffs[i]
        )
    
    # Reconstruct
    denoised = pywt.waverec2(coeffs_thresh, wavelet, mode='symmetric')
    
    return denoised


def apply_denoising(
    image: np.ndarray,
    method: str = "bilateral",
    **kwargs
) -> np.ndarray:
    """
    Apply denoising using specified method.
    
    Args:
        image: Input grayscale image
        method: Denoising method ('gaussian', 'median', 'bilateral', 'nlm', 'wavelet')
        **kwargs: Additional arguments for specific denoising method
    
    Returns:
        Denoised image
    """
    methods = {
        'gaussian': gaussian_denoise,
        'median': median_denoise,
        'bilateral': bilateral_denoise,
        'nlm': nlm_denoise,
        'wavelet': wavelet_denoise
    }
    
    if method not in methods:
        raise ValueError(f"Unknown denoising method: {method}. Choose from {list(methods.keys())}")
    
    return methods[method](image, **kwargs)

