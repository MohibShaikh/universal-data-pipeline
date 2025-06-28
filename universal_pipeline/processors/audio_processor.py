"""
Audio Data Processor - Handles audio preprocessing
"""

import librosa
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from .base_processor import BaseProcessor


class AudioProcessor(BaseProcessor):
    """Processor for audio data"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
    def _setup_default_config(self):
        """Setup default configuration"""
        default_config = {
            'sample_rate': None,  # Preserve original sample rate
            'preserve_duration': True,  # Keep original audio length
            'output_format': 'raw_audio',  # 'raw_audio', 'mfcc', 'melspectrogram'
            'normalize_amplitude': True,  # Basic audio normalization
            'remove_silence': False,  # Optional silence removal
            'mono_conversion': True,  # Convert to mono if needed
            'noise_reduction': False  # Optional noise reduction
        }
        self.config.update({k: v for k, v in default_config.items() if k not in self.config})
    
    def fit(self, data: Union[str, Path, List[str], np.ndarray]) -> 'AudioProcessor':
        """Fit the processor"""
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[str, Path, List[str], np.ndarray]) -> np.ndarray:
        """Transform audio data"""
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        if isinstance(data, (str, Path)):
            return self._process_single_audio(data)
        elif isinstance(data, list):
            return self._process_multiple_audio(data)
        elif isinstance(data, np.ndarray):
            return self._process_numpy_audio(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _process_single_audio(self, audio_path: Union[str, Path]) -> Union[np.ndarray, Dict]:
        """Process a single audio file"""
        try:
            # Load audio - preserve sample rate if requested
            y, sr = librosa.load(audio_path, sr=self.config['sample_rate'])
            
            # Convert to mono if requested
            if self.config['mono_conversion'] and y.ndim > 1:
                y = librosa.to_mono(y)
            
            # Remove silence if requested
            if self.config['remove_silence']:
                y, _ = librosa.effects.trim(y)
            
            # Normalize amplitude if requested
            if self.config['normalize_amplitude']:
                y = librosa.util.normalize(y)
            
            # Return based on output format
            if self.config['output_format'] == 'raw_audio':
                return {
                    'audio': y,
                    'sample_rate': sr,
                    'duration': len(y) / sr,
                    'shape': y.shape
                }
            elif self.config['output_format'] == 'mfcc':
                features = librosa.feature.mfcc(y=y, sr=sr)
                return features
            elif self.config['output_format'] == 'melspectrogram':
                features = librosa.feature.melspectrogram(y=y, sr=sr)
                return features
            else:
                # Default: return clean raw audio
                return {
                    'audio': y,
                    'sample_rate': sr,
                    'duration': len(y) / sr,
                    'shape': y.shape
                }
            
        except Exception as e:
            raise ValueError(f"Error processing audio {audio_path}: {str(e)}")
    
    def _process_multiple_audio(self, audio_paths: List[str]) -> np.ndarray:
        """Process multiple audio files"""
        processed_audio = []
        for audio_path in audio_paths:
            processed = self._process_single_audio(audio_path)
            processed_audio.append(processed)
        return np.stack(processed_audio)
    
    def _process_numpy_audio(self, audio_array: np.ndarray) -> Union[np.ndarray, Dict]:
        """Process numpy audio array"""
        # Convert to mono if requested
        if self.config['mono_conversion'] and audio_array.ndim > 1:
            audio_array = librosa.to_mono(audio_array)
        
        # Normalize amplitude if requested
        if self.config['normalize_amplitude']:
            audio_array = librosa.util.normalize(audio_array)
        
        # Return based on output format
        if self.config['output_format'] == 'raw_audio':
            return {
                'audio': audio_array,
                'sample_rate': self.config.get('sample_rate', 22050),
                'duration': len(audio_array) / (self.config.get('sample_rate') or 22050),
                'shape': audio_array.shape
            }
        elif self.config['output_format'] == 'mfcc':
            features = librosa.feature.mfcc(y=audio_array, sr=self.config.get('sample_rate', 22050))
            return features
        elif self.config['output_format'] == 'melspectrogram':
            features = librosa.feature.melspectrogram(y=audio_array, sr=self.config.get('sample_rate', 22050))
            return features
        else:
            # Default: return clean raw audio
            return {
                'audio': audio_array,
                'sample_rate': self.config.get('sample_rate', 22050),
                'duration': len(audio_array) / (self.config.get('sample_rate') or 22050),
                'shape': audio_array.shape
            } 