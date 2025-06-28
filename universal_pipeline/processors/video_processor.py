"""
Video Data Processor - Handles video preprocessing
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from .base_processor import BaseProcessor


class VideoProcessor(BaseProcessor):
    """Processor for video data"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
    def _setup_default_config(self):
        """Setup default configuration"""
        default_config = {
            'frame_size': (224, 224),
            'num_frames': 16,
            'fps': None,  # Use original fps if None
            'normalize': True,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'channels': 3
        }
        self.config.update({k: v for k, v in default_config.items() if k not in self.config})
    
    def fit(self, data: Union[str, Path, List[str], np.ndarray]) -> 'VideoProcessor':
        """Fit the processor"""
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[str, Path, List[str], np.ndarray]) -> torch.Tensor:
        """Transform video data"""
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        if isinstance(data, (str, Path)):
            return self._process_single_video(data)
        elif isinstance(data, list):
            return self._process_multiple_videos(data)
        elif isinstance(data, np.ndarray):
            return self._process_numpy_video(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _process_single_video(self, video_path: Union[str, Path]) -> torch.Tensor:
        """Process a single video file"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame indices to sample
            if total_frames >= self.config['num_frames']:
                indices = np.linspace(0, total_frames - 1, self.config['num_frames'], dtype=int)
            else:
                indices = list(range(total_frames))
                # Pad with last frame if needed
                while len(indices) < self.config['num_frames']:
                    indices.append(indices[-1])
            
            # Extract frames
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize frame
                    frame = cv2.resize(frame, self.config['frame_size'])
                    frames.append(frame)
            
            cap.release()
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # Convert to tensor
            video_tensor = torch.tensor(np.stack(frames), dtype=torch.float32)
            # Normalize to [0, 1]
            video_tensor = video_tensor / 255.0
            
            # Normalize with ImageNet stats if requested
            if self.config['normalize']:
                mean = torch.tensor(self.config['mean']).view(1, 1, 1, 3)
                std = torch.tensor(self.config['std']).view(1, 1, 1, 3)
                video_tensor = (video_tensor - mean) / std
            
            # Rearrange to (channels, frames, height, width)
            video_tensor = video_tensor.permute(3, 0, 1, 2)
            
            return video_tensor
            
        except Exception as e:
            raise ValueError(f"Error processing video {video_path}: {str(e)}")
    
    def _process_multiple_videos(self, video_paths: List[str]) -> torch.Tensor:
        """Process multiple video files"""
        processed_videos = []
        for video_path in video_paths:
            processed_video = self._process_single_video(video_path)
            processed_videos.append(processed_video)
        return torch.stack(processed_videos)
    
    def _process_numpy_video(self, video_array: np.ndarray) -> torch.Tensor:
        """Process numpy video array"""
        # Assume input is (frames, height, width, channels)
        if video_array.ndim != 4:
            raise ValueError("Video array must be 4D (frames, height, width, channels)")
        
        # Resize frames if needed
        if video_array.shape[1:3] != self.config['frame_size']:
            resized_frames = []
            for frame in video_array:
                resized_frame = cv2.resize(frame, self.config['frame_size'])
                resized_frames.append(resized_frame)
            video_array = np.stack(resized_frames)
        
        # Sample frames if needed
        if video_array.shape[0] != self.config['num_frames']:
            if video_array.shape[0] >= self.config['num_frames']:
                indices = np.linspace(0, video_array.shape[0] - 1, self.config['num_frames'], dtype=int)
                video_array = video_array[indices]
            else:
                # Pad with last frame
                padding_needed = self.config['num_frames'] - video_array.shape[0]
                last_frame = video_array[-1:]
                padding = np.repeat(last_frame, padding_needed, axis=0)
                video_array = np.concatenate([video_array, padding], axis=0)
        
        # Convert to tensor
        video_tensor = torch.tensor(video_array, dtype=torch.float32)
        
        # Normalize to [0, 1] if needed
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0
        
        # Normalize with ImageNet stats if requested
        if self.config['normalize']:
            mean = torch.tensor(self.config['mean']).view(1, 1, 1, 3)
            std = torch.tensor(self.config['std']).view(1, 1, 1, 3)
            video_tensor = (video_tensor - mean) / std
        
        # Rearrange to (channels, frames, height, width)
        video_tensor = video_tensor.permute(3, 0, 1, 2)
        
        return video_tensor
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get the output shape after transformation"""
        return (self.config['channels'], self.config['num_frames'], *self.config['frame_size']) 