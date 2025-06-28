"""
Data Type Detection Module
Automatically detects the type of input data (images, videos, audio, tabular, time series)
"""

import os
import mimetypes
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional
from PIL import Image

# Try to import magic, make it optional
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False


class DataTypeDetector:
    """Detects data type automatically from file path, content, or data structure"""
    
    # File extensions for different data types
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', '.svg'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.wma', '.m4a'}
    TABULAR_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.parquet', '.tsv', '.json'}
    
    def __init__(self):
        """Initialize the data type detector"""
        self.mime_types = {
            'image': ['image/'],
            'video': ['video/'],
            'audio': ['audio/'],
            'text': ['text/', 'application/json', 'application/csv']
        }
    
    def detect_from_path(self, data_path: Union[str, Path]) -> str:
        """
        Detect data type from file path/extension
        
        Args:
            data_path: Path to the data file
            
        Returns:
            str: Detected data type ('image', 'video', 'audio', 'tabular', 'timeseries', 'unknown')
        """
        path = Path(data_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {data_path}")
        
        # Check if it's a directory (batch processing)
        if path.is_dir():
            return self._detect_from_directory(path)
        
        # Check if this is a YOLO annotation file (should not be processed as tabular)
        if self._is_yolo_annotation_file(path):
            return 'annotation'  # Special type for annotation files
        
        # Get file extension
        extension = path.suffix.lower()
        
        # Check by extension first
        if extension in self.IMAGE_EXTENSIONS:
            return 'image'
        elif extension in self.VIDEO_EXTENSIONS:
            return 'video'
        elif extension in self.AUDIO_EXTENSIONS:
            return 'audio'
        elif extension in self.TABULAR_EXTENSIONS:
            # Need to check if it's time series
            return self._check_tabular_or_timeseries(data_path)
        
        # Fallback to MIME type detection
        if HAS_MAGIC:
            try:
                mime_type = magic.from_file(str(path), mime=True)
                return self._detect_from_mime(mime_type)
            except:
                pass
        
        # Final fallback to mimetypes library
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type:
            return self._detect_from_mime(mime_type)
        
        return 'unknown'
    
    def detect_from_data(self, data) -> str:
        """
        Detect data type from loaded data object
        
        Args:
            data: The data object (DataFrame, array, etc.)
            
        Returns:
            str: Detected data type
        """
        if isinstance(data, pd.DataFrame):
            return self._analyze_dataframe(data)
        elif isinstance(data, np.ndarray):
            return self._analyze_numpy_array(data)
        elif isinstance(data, (list, tuple)):
            return self._analyze_list_data(data)
        else:
            return 'unknown'
    
    def _detect_from_directory(self, dir_path: Path) -> str:
        """Detect data type from directory contents with enhanced dataset structure recognition"""
        files = list(dir_path.glob('*'))
        if not files:
            return 'unknown'
        
        # First, check for structured image dataset patterns (YOLO, COCO, etc.)
        dataset_type = self._detect_image_dataset_structure(dir_path)
        if dataset_type:
            return dataset_type
        
        # Check for class-based image structure (subdirectories with images)
        subdirs = [f for f in files if f.is_dir()]
        if subdirs:
            # Check if subdirectories contain image files
            image_subdir_count = 0
            total_image_files = 0
            
            for subdir in subdirs:
                subdir_files = list(subdir.glob('*'))
                image_files = [f for f in subdir_files if f.suffix.lower() in self.IMAGE_EXTENSIONS]
                if len(image_files) > 0:
                    image_subdir_count += 1
                    total_image_files += len(image_files)
            
            # If multiple subdirs contain many images, it's likely an image collection
            if image_subdir_count >= 2 and total_image_files > 50:
                return 'image'
            elif image_subdir_count == 1 and total_image_files > 20:
                # Single subdir with many images might still be image collection
                return 'image'
        
        # Check file extensions directly
        all_files = []
        for item in files:
            if item.is_file():
                all_files.append(item)
            elif item.is_dir():
                # Also check files in subdirectories (but limit depth)
                subfiles = list(item.glob('*'))[:100]  # Limit to avoid performance issues
                all_files.extend([f for f in subfiles if f.is_file()])
        
        # Count file types by extension
        type_counts = {'image': 0, 'video': 0, 'audio': 0, 'tabular': 0}
        
        for file_path in all_files:
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in self.IMAGE_EXTENSIONS:
                    type_counts['image'] += 1
                elif ext in self.VIDEO_EXTENSIONS:
                    type_counts['video'] += 1
                elif ext in self.AUDIO_EXTENSIONS:
                    type_counts['audio'] += 1
                elif ext in self.TABULAR_EXTENSIONS:
                    type_counts['tabular'] += 1
        
        # Enhanced decision logic
        total_files = sum(type_counts.values())
        if total_files == 0:
            return 'unknown'
        
        # If 70%+ images and substantial count, it's an image collection
        if type_counts['image'] > 0 and type_counts['image'] / total_files >= 0.7 and type_counts['image'] > 10:
            return 'image'
        
        # Otherwise return the most common type
        return max(type_counts, key=type_counts.get)
    
    def _detect_image_dataset_structure(self, dir_path: Path) -> str:
        """Detect if directory follows a structured ML image dataset pattern"""
        files = list(dir_path.glob('*'))
        subdirs = [f for f in files if f.is_dir()]
        
        # Check for YOLO/ML dataset structure: train/val/test with images/labels
        ml_splits = {'train', 'val', 'test', 'validation'}
        found_splits = []
        
        for subdir in subdirs:
            if subdir.name.lower() in ml_splits:
                found_splits.append(subdir.name.lower())
                
                # Check if this split has images/ and labels/ subdirs
                split_contents = list(subdir.glob('*'))
                split_subdirs = [f.name.lower() for f in split_contents if f.is_dir()]
                
                if 'images' in split_subdirs and 'labels' in split_subdirs:
                    # This looks like a proper ML dataset structure
                    return 'image_dataset'
        
        # Check for dataset metadata files that indicate ML dataset
        metadata_files = ['dataset_info.json', 'data.yaml', 'dataset.yaml', 'config.yaml']
        has_metadata = any((dir_path / filename).exists() for filename in metadata_files)
        
        # If we have 2+ ML splits or 1+ split with metadata, it's likely an ML dataset
        if len(found_splits) >= 2 or (len(found_splits) >= 1 and has_metadata):
            return 'image_dataset'
        
        # Check for class-based structure (subdirs named as classes with images)
        if len(subdirs) >= 3:  # At least 3 class directories
            class_dirs_with_images = 0
            for subdir in subdirs[:10]:  # Check first 10 to avoid performance issues
                image_files = list(subdir.glob('*'))
                image_count = sum(1 for f in image_files if f.suffix.lower() in self.IMAGE_EXTENSIONS)
                if image_count > 5:  # At least 5 images per class
                    class_dirs_with_images += 1
            
            if class_dirs_with_images >= 3:
                return 'image_dataset'
        
        return None  # Not a structured dataset
    
    def _detect_from_mime(self, mime_type: str) -> str:
        """Detect data type from MIME type"""
        if not mime_type:
            return 'unknown'
        
        mime_type = mime_type.lower()
        
        if mime_type.startswith('image/'):
            return 'image'
        elif mime_type.startswith('video/'):
            return 'video'
        elif mime_type.startswith('audio/'):
            return 'audio'
        elif mime_type in ['text/csv', 'application/json', 'text/plain']:
            return 'tabular'
        
        return 'unknown'
    
    def _check_tabular_or_timeseries(self, file_path: str) -> str:
        """Check if tabular data is actually time series"""
        try:
            # Try to load a sample of the data
            path = Path(file_path)
            extension = path.suffix.lower()
            
            if extension == '.csv':
                df = pd.read_csv(file_path, nrows=100)  # Sample first 100 rows
            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=100)
            elif extension == '.parquet':
                df = pd.read_parquet(file_path)
                df = df.head(100)
            elif extension == '.json':
                # First check if this is a complex JSON that shouldn't be treated as tabular
                if not self._is_tabular_json(file_path):
                    return 'unknown'  # Don't process complex JSON as tabular
                df = pd.read_json(file_path, lines=True, nrows=100)
            else:
                return 'tabular'
            
            return self._analyze_dataframe(df)
            
        except Exception:
            return 'tabular'  # Default to tabular if can't analyze
    
    def _is_tabular_json(self, file_path: str) -> bool:
        """Check if JSON file contains tabular data or complex nested structures"""
        try:
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check for common metadata/config file patterns
            metadata_indicators = [
                'processing_summary', 'config', 'metadata', 'pipeline', 
                'summary', 'info', 'settings', 'status', 'results'
            ]
            
            # If JSON contains these keys at root level, it's likely metadata
            if isinstance(data, dict):
                for key in data.keys():
                    if any(indicator in key.lower() for indicator in metadata_indicators):
                        return False
                
                # Check for nested dictionary structures (not tabular)
                nested_dict_count = sum(1 for v in data.values() if isinstance(v, dict))
                if nested_dict_count > len(data) * 0.5:  # More than 50% nested dicts
                    return False
            
            # If it's a list of simple objects, it might be tabular
            elif isinstance(data, list):
                if not data:
                    return False
                
                # Check first few items
                sample_size = min(5, len(data))
                for item in data[:sample_size]:
                    if not isinstance(item, dict):
                        return False
                    # Check if all values are simple types (not nested)
                    for value in item.values():
                        if isinstance(value, (dict, list)):
                            return False
                
                return True  # List of simple objects = tabular
            
            else:
                return False  # Not dict or list = not tabular
            
            return True  # Passed all checks
            
        except Exception:
            return False  # If can't parse, assume it's not tabular
    
    def _analyze_dataframe(self, df: pd.DataFrame) -> str:
        """Analyze DataFrame to determine if it's tabular or time series"""
        if df.empty:
            return 'tabular'
        
        # Check for time-related columns
        time_indicators = ['time', 'date', 'timestamp', 'datetime', 'day', 'month', 'year']
        
        # Check column names
        column_names = [col.lower() for col in df.columns]
        has_time_column = any(indicator in ' '.join(column_names) for indicator in time_indicators)
        
        # Check for datetime columns
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        has_datetime_column = len(datetime_columns) > 0
        
        # Check if index looks like time series
        index_is_datetime = pd.api.types.is_datetime64_any_dtype(df.index)
        
        # Heuristic: if more than 30% of columns are numeric and we have time indicators
        numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
        
        if (has_time_column or has_datetime_column or index_is_datetime) and numeric_ratio > 0.3:
            return 'timeseries'
        
        return 'tabular'
    
    def _analyze_numpy_array(self, arr: np.ndarray) -> str:
        """Analyze numpy array to determine data type"""
        if arr.ndim == 1:
            return 'tabular'  # 1D array, likely tabular
        elif arr.ndim == 2:
            # Could be image (if small) or tabular data
            if arr.shape[0] < 1000 and arr.shape[1] < 1000:
                return 'image'  # Likely an image
            else:
                return 'tabular'  # Likely tabular data
        elif arr.ndim == 3:
            # Could be image (H, W, C) or time series (samples, timesteps, features)
            if arr.shape[2] in [1, 3, 4]:  # Likely channels
                return 'image'
            else:
                return 'timeseries'
        elif arr.ndim == 4:
            return 'image'  # Likely batch of images (N, H, W, C)
        
        return 'unknown'
    
    def _analyze_list_data(self, data: Union[list, tuple]) -> str:
        """Analyze list/tuple data"""
        if not data:
            return 'unknown'
        
        first_item = data[0]
        
        if isinstance(first_item, str):
            # Check if it's file paths
            if os.path.exists(first_item):
                return self.detect_from_path(first_item)
            return 'text'
        elif isinstance(first_item, (int, float)):
            return 'tabular'
        elif isinstance(first_item, np.ndarray):
            return self._analyze_numpy_array(first_item)
        
        return 'unknown'
    
    def get_detailed_info(self, data_path: Union[str, Path]) -> dict:
        """
        Get detailed information about the data
        
        Args:
            data_path: Path to the data
            
        Returns:
            dict: Detailed information about the data
        """
        path = Path(data_path)
        info = {
            'path': str(path),
            'exists': path.exists(),
            'is_file': path.is_file(),
            'is_directory': path.is_dir(),
            'extension': path.suffix.lower(),
            'size_bytes': path.stat().st_size if path.exists() else 0,
            'detected_type': 'unknown'
        }
        
        if path.exists():
            info['detected_type'] = self.detect_from_path(data_path)
            
            # Add MIME type info
            try:
                info['mime_type'] = magic.from_file(str(path), mime=True)
            except:
                info['mime_type'] = mimetypes.guess_type(str(path))[0]
        
        return info

    def _is_yolo_annotation_file(self, path: Path) -> bool:
        """Check if the file is a YOLO annotation file"""
        # Check if it's a .txt file in a labels directory
        if path.suffix.lower() != '.txt':
            return False
        
        # Check if parent directory is named 'labels'
        if path.parent.name.lower() == 'labels':
            # Check if there's a corresponding images directory
            labels_parent = path.parent.parent
            images_dir = labels_parent / 'images'
            if images_dir.exists() and images_dir.is_dir():
                return True
        
        # Additional check: try to read the file and see if it matches YOLO format
        try:
            with open(path, 'r') as f:
                lines = f.read().strip().split('\n')
                if not lines or lines == ['']:
                    return True  # Empty annotation file (valid in YOLO)
                
                # Check first few lines for YOLO format: class_id x_center y_center width height
                for line in lines[:3]:  # Check first 3 lines
                    if line.strip():  # Skip empty lines
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                # Try to parse as YOLO format
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # YOLO coordinates should be normalized (0.0 to 1.0)
                                if (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and
                                    0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
                                    return True
                            except ValueError:
                                continue
                        
        except Exception:
            pass
        
        return False


# Convenience function
def detect_data_type(data_input: Union[str, Path, pd.DataFrame, np.ndarray, list]) -> str:
    """
    Convenience function to detect data type
    
    Args:
        data_input: File path, data object, or list of data
        
    Returns:
        str: Detected data type
    """
    detector = DataTypeDetector()
    
    if isinstance(data_input, (str, Path)):
        return detector.detect_from_path(data_input)
    else:
        return detector.detect_from_data(data_input) 