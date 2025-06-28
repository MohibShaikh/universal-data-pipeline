"""
Image Data Processor - Handles image preprocessing
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from .base_processor import BaseProcessor


class ImageProcessor(BaseProcessor):
    """Processor for image data"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.torch_transform = None
        
    def _setup_default_config(self):
        """Setup default configuration"""
        default_config = {
            'preserve_original_size': True,  # Keep original image dimensions
            'target_size': None,  # Only resize if explicitly requested
            'normalize_pixel_values': True,  # 0-255 normalization, not ImageNet
            'output_format': 'numpy',  # 'numpy', 'pil', 'tensor'
            'ensure_rgb': True,  # Convert to RGB but keep as images
            'remove_noise': True,  # Basic image cleanup
            'enhance_contrast': False  # Optional enhancement
        }
        self.config.update({k: v for k, v in default_config.items() if k not in self.config})
    
    def fit(self, data: Union[str, Path, List[str], np.ndarray, None]) -> 'ImageProcessor':
        """Fit the processor"""
        self._setup_transforms()
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[str, Path, List[str], np.ndarray]) -> torch.Tensor:
        """Transform the image data"""
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        if isinstance(data, (str, Path)):
            return self._process_single_image(data)
        elif isinstance(data, list):
            return self._process_multiple_images(data)
        elif isinstance(data, np.ndarray):
            return self._process_numpy_array(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _setup_transforms(self):
        """Setup image transformation pipeline for domain-appropriate output"""
        transforms_list = []
        
        # Only resize if explicitly requested
        if self.config['target_size'] is not None:
            transforms_list.append(transforms.Resize(self.config['target_size']))
        
        # Convert to tensor only if requested
        if self.config['output_format'] == 'tensor':
            transforms_list.append(transforms.ToTensor())
        
        self.torch_transform = transforms.Compose(transforms_list) if transforms_list else None
    
    def _process_single_image(self, image_path: Union[str, Path]) -> Union[np.ndarray, Image.Image, torch.Tensor]:
        """Process a single image file"""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if requested
            if self.config['ensure_rgb']:
                image = image.convert('RGB')
            
            # Apply basic cleanup
            if self.config['remove_noise']:
                # Basic noise reduction (you can enhance this)
                image = image.filter(ImageFilter.SMOOTH_MORE)
            
            # Apply transforms if any
            if self.torch_transform is not None:
                image = self.torch_transform(image)
            
            # Return in requested format
            if self.config['output_format'] == 'numpy':
                if isinstance(image, torch.Tensor):
                    return image.numpy()
                else:
                    return np.array(image)
            elif self.config['output_format'] == 'pil':
                return image
            elif self.config['output_format'] == 'tensor':
                if not isinstance(image, torch.Tensor):
                    return transforms.ToTensor()(image)
                return image
            else:
                return np.array(image)  # Default to numpy
                
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {str(e)}")
    
    def _process_multiple_images(self, image_paths: List[str]) -> torch.Tensor:
        """Process multiple image files"""
        processed_images = []
        for image_path in image_paths:
            processed_image = self._process_single_image(image_path)
            processed_images.append(processed_image)
        return torch.stack(processed_images)
    
    def _process_numpy_array(self, image_array: np.ndarray) -> Union[np.ndarray, Image.Image, torch.Tensor]:
        """Process numpy array"""
        if image_array.ndim == 2:
            # Grayscale image (H, W)
            # Ensure uint8 format for PIL
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            # Convert grayscale to RGB if requested
            if self.config['ensure_rgb']:
                image = Image.fromarray(image_array, 'L').convert('RGB')
            else:
                image = Image.fromarray(image_array, 'L')
            
        elif image_array.ndim == 3:
            # Single image
            if image_array.shape[0] in [1, 3, 4]:  # (C, H, W)
                image_array = np.transpose(image_array, (1, 2, 0))
            
            # Ensure uint8 format for PIL
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            # Convert to RGB if needed
            if image_array.shape[2] == 3:
                image = Image.fromarray(image_array, 'RGB')
            elif image_array.shape[2] == 1:
                image = Image.fromarray(image_array.squeeze(-1), 'L')
                if self.config['ensure_rgb']:
                    image = image.convert('RGB')
            else:
                # Take first 3 channels if more than 3
                image = Image.fromarray(image_array[:, :, :3], 'RGB')
        else:
            raise ValueError(f"Unsupported array dimensions: {image_array.ndim}")
        
        # Apply basic cleanup
        if self.config['remove_noise']:
            image = image.filter(ImageFilter.SMOOTH_MORE)
        
        # Apply transforms if any
        if self.torch_transform is not None:
            image = self.torch_transform(image)
        
        # Return in requested format
        if self.config['output_format'] == 'numpy':
            if isinstance(image, torch.Tensor):
                return image.numpy()
            else:
                return np.array(image)
        elif self.config['output_format'] == 'pil':
            return image
        elif self.config['output_format'] == 'tensor':
            if not isinstance(image, torch.Tensor):
                return transforms.ToTensor()(image)
            return image
        else:
            return np.array(image)  # Default to numpy
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get the output shape after transformation"""
        return (self.config['channels'], *self.config['image_size']) 