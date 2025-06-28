"""
Image Data Processor - Handles image preprocessing with flexible sizing and label transformation
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import cv2

from .base_processor import BaseProcessor


class ImageProcessor(BaseProcessor):
    """Advanced processor for image data with flexible sizing and label handling"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.torch_transform = None
        self.resize_info = {}  # Store resize information for label transformation
        
    def _setup_default_config(self):
        """Setup default configuration"""
        default_config = {
            # Size and orientation
            'target_size': None,  # (width, height) or single int for square
            'preserve_aspect_ratio': True,  # Maintain aspect ratio with padding
            'auto_orient': True,  # Auto-rotate based on EXIF data
            'resize_strategy': 'letterbox',  # 'letterbox', 'stretch', 'crop', 'fit'
            'padding_color': (114, 114, 114),  # Gray padding for letterbox
            
            # Output format
            'output_format': 'numpy',  # 'numpy', 'pil', 'tensor'
            'normalize_pixel_values': True,  # Keep 0-255 range for object detection
            'ensure_rgb': True,  # Convert to RGB
            'channel_order': 'RGB',  # 'RGB' or 'BGR' 
            
            # Enhancement
            'remove_noise': False,  # Keep original for detection
            'enhance_contrast': False,  # Optional enhancement
            
            # Label handling
            'transform_labels': False,  # Whether to transform bounding box labels
            'label_format': 'yolo',  # 'yolo', 'pascal_voc', 'coco'
            'return_resize_info': False  # Return transformation info for labels
        }
        self.config.update({k: v for k, v in default_config.items() if k not in self.config})
    
    def fit(self, data: Union[str, Path, List[str], np.ndarray, None]) -> 'ImageProcessor':
        """Fit the processor"""
        self._setup_transforms()
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[str, Path, List[str], np.ndarray], labels: Optional[Union[str, List, np.ndarray]] = None) -> Union[torch.Tensor, Tuple]:
        """Transform the image data and optionally labels"""
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        if isinstance(data, (str, Path)):
            result = self._process_single_image(data)
        elif isinstance(data, list):
            result = self._process_multiple_images(data)
        elif isinstance(data, np.ndarray):
            result = self._process_numpy_array(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Handle label transformation if requested
        if labels is not None and self.config['transform_labels']:
            transformed_labels = self._transform_labels(labels, data)
            if self.config['return_resize_info']:
                return result, transformed_labels, self.resize_info
            return result, transformed_labels
        
        if self.config['return_resize_info']:
            return result, self.resize_info
        
        return result
    
    def _setup_transforms(self):
        """Setup image transformation pipeline"""
        # Basic transforms will be handled in processing methods
        # for more control over aspect ratio and padding
        pass
    
    def _process_single_image(self, image_path: Union[str, Path]) -> Union[np.ndarray, Image.Image, torch.Tensor]:
        """Process a single image file with smart resizing"""
        try:
            image = Image.open(image_path)
            
            # Auto-orient based on EXIF data
            if self.config['auto_orient']:
                image = self._auto_orient_image(image)
            
            # Convert to RGB if requested
            if self.config['ensure_rgb']:
                image = image.convert('RGB')
            
            # Store original size for label transformation
            original_size = image.size  # (width, height)
            
            # Apply smart resizing
            if self.config['target_size'] is not None:
                image, resize_info = self._smart_resize(image)
                self.resize_info[str(image_path)] = resize_info
            
            # Apply enhancement if requested
            if self.config['remove_noise']:
                image = image.filter(ImageFilter.SMOOTH_MORE)
            
            # Convert to requested format
            return self._convert_to_output_format(image)
                
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {str(e)}")
    
    def _auto_orient_image(self, image: Image.Image) -> Image.Image:
        """Auto-orient image based on EXIF data"""
        try:
            # Get EXIF orientation tag
            for orientation in range(1, 9):
                try:
                    exif = image._getexif()
                    if exif is not None:
                        orientation = exif.get(274, 1)  # 274 is orientation tag
                        break
                except:
                    orientation = 1
                    break
            
            # Apply rotation based on orientation
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
            
            return image
        except:
            return image  # Return original if EXIF processing fails
    
    def _smart_resize(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Smart resize with multiple strategies"""
        target_size = self._normalize_target_size(self.config['target_size'])
        target_w, target_h = target_size
        original_w, original_h = image.size
        
        resize_info = {
            'original_size': (original_w, original_h),
            'target_size': (target_w, target_h),
            'strategy': self.config['resize_strategy']
        }
        
        if self.config['resize_strategy'] == 'letterbox':
            image, letterbox_info = self._letterbox_resize(image, target_size)
            resize_info.update(letterbox_info)
        
        elif self.config['resize_strategy'] == 'stretch':
            image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            resize_info.update({
                'scale_x': target_w / original_w,
                'scale_y': target_h / original_h,
                'offset_x': 0,
                'offset_y': 0
            })
        
        elif self.config['resize_strategy'] == 'crop':
            image = self._center_crop_resize(image, target_size)
            resize_info.update({
                'scale_x': target_w / original_w,
                'scale_y': target_h / original_h,
                'offset_x': 0,
                'offset_y': 0
            })
        
        elif self.config['resize_strategy'] == 'fit':
            image = self._fit_resize(image, target_size)
            # Calculate actual scales after fit
            new_w, new_h = image.size
            resize_info.update({
                'scale_x': new_w / original_w,
                'scale_y': new_h / original_h,
                'offset_x': 0,
                'offset_y': 0
            })
        
        return image, resize_info
    
    def _letterbox_resize(self, image: Image.Image, target_size: Tuple[int, int]) -> Tuple[Image.Image, Dict]:
        """Letterbox resize - maintain aspect ratio with padding"""
        target_w, target_h = target_size
        original_w, original_h = image.size
        
        # Calculate scale to fit image in target size
        scale = min(target_w / original_w, target_h / original_h)
        
        # Calculate new size
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # Resize image
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create new image with target size and padding
        new_image = Image.new('RGB', target_size, self.config['padding_color'])
        
        # Calculate padding offsets to center the image
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        
        # Paste resized image onto padded background
        new_image.paste(image, (offset_x, offset_y))
        
        letterbox_info = {
            'scale': scale,
            'scale_x': scale,
            'scale_y': scale,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'new_size': (new_w, new_h)
        }
        
        return new_image, letterbox_info
    
    def _center_crop_resize(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Center crop and resize"""
        target_w, target_h = target_size
        original_w, original_h = image.size
        
        # Calculate crop box to maintain aspect ratio
        target_aspect = target_w / target_h
        original_aspect = original_w / original_h
        
        if original_aspect > target_aspect:
            # Image is wider, crop width
            new_w = int(original_h * target_aspect)
            left = (original_w - new_w) // 2
            crop_box = (left, 0, left + new_w, original_h)
        else:
            # Image is taller, crop height
            new_h = int(original_w / target_aspect)
            top = (original_h - new_h) // 2
            crop_box = (0, top, original_w, top + new_h)
        
        # Crop and resize
        image = image.crop(crop_box)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image
    
    def _fit_resize(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Fit resize - maintain aspect ratio, may not fill target size"""
        target_w, target_h = target_size
        original_w, original_h = image.size
        
        # Calculate scale to fit within target size
        scale = min(target_w / original_w, target_h / original_h)
        
        # Calculate new size
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    def _normalize_target_size(self, target_size) -> Tuple[int, int]:
        """Normalize target size to (width, height)"""
        if isinstance(target_size, int):
            return (target_size, target_size)
        elif isinstance(target_size, (list, tuple)) and len(target_size) == 2:
            return tuple(target_size)
        else:
            raise ValueError(f"Invalid target_size: {target_size}")
    
    def _transform_labels(self, labels: Union[str, List, np.ndarray], image_path: Union[str, Path]) -> Union[List, np.ndarray]:
        """Transform bounding box labels based on image transformation"""
        if str(image_path) not in self.resize_info:
            return labels  # No transformation info available
        
        resize_info = self.resize_info[str(image_path)]
        
        if isinstance(labels, str):
            # Assume it's a label file path
            labels = self._load_label_file(labels)
        
        if isinstance(labels, list):
            labels = np.array(labels)
        
        # Transform based on label format
        if self.config['label_format'] == 'yolo':
            return self._transform_yolo_labels(labels, resize_info)
        elif self.config['label_format'] == 'pascal_voc':
            return self._transform_pascal_labels(labels, resize_info)
        elif self.config['label_format'] == 'coco':
            return self._transform_coco_labels(labels, resize_info)
        
        return labels
    
    def _transform_yolo_labels(self, labels: np.ndarray, resize_info: Dict) -> np.ndarray:
        """Transform YOLO format labels (normalized coordinates)"""
        if len(labels) == 0:
            return labels
        
        # For letterbox, need to adjust for padding
        if resize_info['strategy'] == 'letterbox':
            transformed_labels = labels.copy()
            scale = resize_info['scale']
            offset_x = resize_info['offset_x']
            offset_y = resize_info['offset_y']
            target_w, target_h = resize_info['target_size']
            original_w, original_h = resize_info['original_size']
            
            # Transform center coordinates
            for i in range(len(transformed_labels)):
                if len(transformed_labels[i]) >= 5:  # class, x, y, w, h
                    # Convert from normalized to absolute coordinates in original image
                    x_abs = transformed_labels[i][1] * original_w
                    y_abs = transformed_labels[i][2] * original_h
                    w_abs = transformed_labels[i][3] * original_w
                    h_abs = transformed_labels[i][4] * original_h
                    
                    # Apply scale and offset
                    x_new = (x_abs * scale + offset_x) / target_w
                    y_new = (y_abs * scale + offset_y) / target_h
                    w_new = (w_abs * scale) / target_w
                    h_new = (h_abs * scale) / target_h
                    
                    # Update labels
                    transformed_labels[i][1] = x_new
                    transformed_labels[i][2] = y_new
                    transformed_labels[i][3] = w_new
                    transformed_labels[i][4] = h_new
            
            return transformed_labels
        
        # For other strategies, labels remain the same (normalized coordinates)
        return labels
    
    def _load_label_file(self, label_path: str) -> np.ndarray:
        """Load label file (assuming YOLO format)"""
        try:
            return np.loadtxt(label_path).reshape(-1, 5)  # Ensure 2D array
        except:
            return np.array([])  # Return empty array if file doesn't exist or is empty
    
    def _process_multiple_images(self, image_paths: List[str]) -> torch.Tensor:
        """Process multiple image files"""
        processed_images = []
        for image_path in image_paths:
            processed_image = self._process_single_image(image_path)
            processed_images.append(processed_image)
        return torch.stack([torch.from_numpy(img) if isinstance(img, np.ndarray) else img for img in processed_images])
    
    def _process_numpy_array(self, image_array: np.ndarray) -> Union[np.ndarray, Image.Image, torch.Tensor]:
        """Process numpy array"""
        if image_array.ndim == 2:
            # Grayscale image (H, W)
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            image = Image.fromarray(image_array, 'L')
            if self.config['ensure_rgb']:
                image = image.convert('RGB')
            
        elif image_array.ndim == 3:
            # Single image
            if image_array.shape[0] in [1, 3, 4]:  # (C, H, W)
                image_array = np.transpose(image_array, (1, 2, 0))
            
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            if image_array.shape[2] == 3:
                image = Image.fromarray(image_array, 'RGB')
            elif image_array.shape[2] == 1:
                image = Image.fromarray(image_array.squeeze(-1), 'L')
                if self.config['ensure_rgb']:
                    image = image.convert('RGB')
            else:
                image = Image.fromarray(image_array[:, :, :3], 'RGB')
        else:
            raise ValueError(f"Unsupported array dimensions: {image_array.ndim}")
        
        # Apply smart resizing if configured
        if self.config['target_size'] is not None:
            image, resize_info = self._smart_resize(image)
            self.resize_info['numpy_array'] = resize_info
        
        return self._convert_to_output_format(image)
    
    def _convert_to_output_format(self, image: Image.Image) -> Union[np.ndarray, Image.Image, torch.Tensor]:
        """Convert image to requested output format"""
        if self.config['output_format'] == 'numpy':
            array = np.array(image)
            # Handle channel order
            if self.config['channel_order'] == 'BGR' and len(array.shape) == 3 and array.shape[2] == 3:
                array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            return array
        elif self.config['output_format'] == 'pil':
            return image
        elif self.config['output_format'] == 'tensor':
            tensor = transforms.ToTensor()(image)
            # Handle channel order
            if self.config['channel_order'] == 'BGR':
                tensor = tensor[[2, 1, 0], :, :]  # RGB to BGR
            return tensor
        else:
            return np.array(image)
    
    def get_resize_info(self, image_path: str = None) -> Dict:
        """Get resize transformation info for label transformation"""
        if image_path:
            return self.resize_info.get(str(image_path), {})
        return self.resize_info
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get the output shape after transformation"""
        if self.config['target_size'] is not None:
            target_size = self._normalize_target_size(self.config['target_size'])
            if self.config['output_format'] == 'tensor':
                return (3, target_size[1], target_size[0])  # (C, H, W)
            else:
                return (target_size[1], target_size[0], 3)  # (H, W, C)
        return None  # Variable size 