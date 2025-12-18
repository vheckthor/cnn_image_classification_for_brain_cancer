"""
Data augmentation transforms for training.
"""
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import random
from scipy.ndimage import rotate, shift, zoom


class AugmentationTransform:
    """Apply augmentation to both image and mask."""
    
    def __init__(
        self,
        rotation_range: float = 15,
        translation_range: float = 0.1,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        brightness_range: float = 0.1,
        contrast_range: float = 0.1,
        flip_probability: float = 0.5,
        elastic_deformation: bool = False
    ):
        """
        Initialize augmentation parameters.
        
        Args:
            rotation_range: Maximum rotation in degrees
            translation_range: Maximum translation as fraction of image size
            scale_range: (min, max) scaling factors
            brightness_range: Maximum brightness adjustment (±)
            contrast_range: Maximum contrast adjustment (±)
            flip_probability: Probability of horizontal/vertical flip
            elastic_deformation: Whether to apply elastic deformation
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.flip_probability = flip_probability
        self.elastic_deformation = elastic_deformation
    
    def __call__(self, sample: Dict) -> Dict:
        """
        Apply augmentation to sample.
        
        Args:
            sample: Dictionary with 'image' and 'mask' keys
        
        Returns:
            Augmented sample
        """
        image = sample['image'].copy()
        mask = sample['mask'].copy()
        
        # Random rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = self._rotate(image, angle)
            mask = self._rotate(mask, angle)
        
        # Random translation
        if self.translation_range > 0:
            h, w = image.shape[:2]
            tx = random.uniform(-self.translation_range, self.translation_range) * w
            ty = random.uniform(-self.translation_range, self.translation_range) * h
            image = self._translate(image, tx, ty)
            mask = self._translate(mask, tx, ty)
        
        # Random scaling
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            image = self._scale(image, scale)
            mask = self._scale(mask, scale)
        
        # Random flips
        if random.random() < self.flip_probability:
            if random.random() < 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)
            else:
                image = np.flipud(image)
                mask = np.flipud(mask)
        
        # Intensity augmentations (only for image)
        if self.brightness_range > 0:
            brightness = random.uniform(-self.brightness_range, self.brightness_range)
            image = self._adjust_brightness(image, brightness)
        
        if self.contrast_range > 0:
            contrast = random.uniform(-self.contrast_range, self.contrast_range)
            image = self._adjust_contrast(image, contrast)
        
        # Elastic deformation
        if self.elastic_deformation:
            image, mask = self._elastic_deform(image, mask)
        
        return {'image': image, 'mask': mask}
    
    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        rotated = cv2.warpAffine(
            image_uint8, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return rotated.astype(np.float32) / 255.0 if image.max() <= 1.0 else rotated.astype(np.float32)
    
    def _translate(self, image: np.ndarray, tx: float, ty: float) -> np.ndarray:
        """Translate image."""
        h, w = image.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        translated = cv2.warpAffine(
            image_uint8, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return translated.astype(np.float32) / 255.0 if image.max() <= 1.0 else translated.astype(np.float32)
    
    def _scale(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Scale image."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        scaled = cv2.resize(image_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if scale > 1.0:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            scaled = scaled[start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            scaled = np.pad(scaled, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='reflect')
        
        return scaled.astype(np.float32) / 255.0 if image.max() <= 1.0 else scaled.astype(np.float32)
    
    def _adjust_brightness(self, image: np.ndarray, brightness: float) -> np.ndarray:
        """Adjust brightness."""
        return np.clip(image + brightness, 0, 1)
    
    def _adjust_contrast(self, image: np.ndarray, contrast: float) -> np.ndarray:
        """Adjust contrast."""
        mean = np.mean(image)
        return np.clip((image - mean) * (1 + contrast) + mean, 0, 1)
    
    def _elastic_deform(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply elastic deformation."""
        # Simplified elastic deformation
        alpha = random.uniform(50, 150)
        sigma = random.uniform(5, 10)
        
        h, w = image.shape[:2]
        dx = np.random.randn(h, w) * alpha
        dy = np.random.randn(h, w) * alpha
        
        # Smooth the displacement fields
        from scipy.ndimage import gaussian_filter
        dx = gaussian_filter(dx, sigma)
        dy = gaussian_filter(dy, sigma)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)
        
        # Apply deformation
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        image_deformed = cv2.remap(
            image_uint8, x_new, y_new,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        if mask.max() <= 1.0:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.astype(np.uint8)
        
        mask_deformed = cv2.remap(
            mask_uint8, x_new, y_new,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return (
            image_deformed.astype(np.float32) / 255.0 if image.max() <= 1.0 else image_deformed.astype(np.float32),
            mask_deformed.astype(np.float32) / 255.0 if mask.max() <= 1.0 else mask_deformed.astype(np.float32)
        )


class PreprocessingTransform:
    """Apply preprocessing (denoising, normalization, resizing) to images."""
    
    def __init__(
        self,
        target_size: int = 256,
        denoising_method: Optional[str] = None,
        normalization_method: str = "z_score"
    ):
        """
        Initialize preprocessing transform.
        
        Args:
            target_size: Target image size (square)
            denoising_method: Denoising method (None to skip)
            normalization_method: Normalization method
        """
        self.target_size = target_size
        self.denoising_method = denoising_method
        self.normalization_method = normalization_method
    
    def __call__(self, sample: Dict) -> Dict:
        """
        Apply preprocessing to sample.
        
        Args:
            sample: Dictionary with 'image' and 'mask' keys
        
        Returns:
            Preprocessed sample
        """
        image = sample['image'].copy()
        mask = sample['mask'].copy()
        
        # Resize if needed
        if image.shape[0] != self.target_size or image.shape[1] != self.target_size:
            image = self._resize(image, self.target_size)
            mask = self._resize(mask, self.target_size, is_mask=True)
        
        # Denoising (only for image)
        if self.denoising_method:
            from preprocessing.denoising import apply_denoising
            image = apply_denoising(image, method=self.denoising_method)
        
        # Normalization (only for image)
        if self.normalization_method:
            from preprocessing.normalization import apply_normalization
            image = apply_normalization(image, method=self.normalization_method)
        
        # Ensure mask is binary
        mask = (mask > 0.5).astype(np.float32)
        
        return {'image': image, 'mask': mask}
    
    def _resize(self, image: np.ndarray, target_size: int, is_mask: bool = False) -> np.ndarray:
        """Resize image to target size."""
        if image.max() <= 1.0 and not is_mask:
            image_uint8 = (image * 255).astype(np.uint8)
        elif is_mask:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        resized = cv2.resize(image_uint8, (target_size, target_size), interpolation=interpolation)
        
        if image.max() <= 1.0 or is_mask:
            return resized.astype(np.float32) / 255.0
        else:
            return resized.astype(np.float32)

