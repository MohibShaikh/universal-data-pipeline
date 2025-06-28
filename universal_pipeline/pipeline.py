"""
Universal Pipeline - Main pipeline that automatically detects and processes any data type
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings

from .data_detector import DataTypeDetector, detect_data_type
from .processors import (
    TabularProcessor,
    ImageProcessor,
    VideoProcessor,
    AudioProcessor,
    TimeSeriesProcessor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalPipeline:
    """
    Universal data pipeline that automatically detects data type and applies appropriate preprocessing
    
    Usage:
        pipeline = UniversalPipeline()
        processed_data = pipeline.fit_transform("path/to/data.csv")
        
        # Or with custom configuration
        config = {'image_size': (256, 256), 'scaling_method': 'minmax'}
        pipeline = UniversalPipeline(config)
        processed_data = pipeline.fit_transform("path/to/images/")
        
        # For datasets, automatic 70-15-15 split by default
        train, val, test = pipeline.fit_transform("path/to/dataset/")
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the universal pipeline
        
        Args:
            config: Configuration dictionary for processors
        """
        self.config = config or {}
        
        # Set default dataset splitting configuration
        self.default_dataset_config = {
            'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
            'stratify': True,
            'random_seed': 42,
            'shuffle': True,
            'return_format': 'tuple',  # Returns (train, val, test) tuple
            'respect_existing_splits': True  # NEW: Respect existing train/test/val splits
        }
        
        # Merge default dataset config with user config
        if 'dataset' not in self.config:
            self.config['dataset'] = {}
        
        for key, value in self.default_dataset_config.items():
            if key not in self.config['dataset']:
                self.config['dataset'][key] = value
        
        self.detector = DataTypeDetector()
        self.processor = None
        self.detected_type = None
        self.is_fitted = False
        self.dataset_structure = None  # NEW: Store detected dataset structure
        
        # Processor mapping
        self.processor_classes = {
            'tabular': TabularProcessor,
            'image': ImageProcessor,
            'image_dataset': ImageProcessor,  # Use ImageProcessor for structured datasets
            'video': VideoProcessor,
            'audio': AudioProcessor,
            'timeseries': TimeSeriesProcessor
        }
    
    def detect_data_type(self, data: Union[str, Path, Any]) -> str:
        """
        Detect the type of input data
        
        Args:
            data: Input data (file path, directory, or data object)
            
        Returns:
            str: Detected data type
        """
        return self.detector.detect_from_path(data) if isinstance(data, (str, Path)) else self.detector.detect_from_data(data)
    
    def fit(self, data: Union[str, Path, Any], data_type: Optional[str] = None) -> 'UniversalPipeline':
        """
        Fit the pipeline to the data
        
        Args:
            data: Input data
            data_type: Force a specific data type (optional)
            
        Returns:
            self: Returns self for method chaining
        """
        # Detect data type if not provided
        if data_type is None:
            self.detected_type = self.detect_data_type(data)
        else:
            self.detected_type = data_type
        
        logger.info(f"Detected data type: {self.detected_type}")
        
        # Validate detected type
        if self.detected_type not in self.processor_classes:
            raise ValueError(f"Unsupported data type: {self.detected_type}. Supported types: {list(self.processor_classes.keys())}")
        
        # Get processor-specific config
        processor_config = self.config.get(self.detected_type, {})
        
        # Initialize the appropriate processor
        processor_class = self.processor_classes[self.detected_type]
        self.processor = processor_class(processor_config)
        
        logger.info(f"Initializing {processor_class.__name__} with config: {processor_config}")
        
        # Fit the processor
        try:
            self.processor.fit(data)
            self.is_fitted = True
            logger.info("Pipeline fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting processor: {str(e)}")
            raise
        
        return self
    
    def transform(self, data: Union[str, Path, Any]) -> Any:
        """
        Transform the data using the fitted processor
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data ready for model training
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform. Call fit() first.")
        
        logger.info(f"Transforming data using {self.processor.__class__.__name__}")
        
        try:
            result = self.processor.transform(data)
            logger.info(f"Data transformed successfully. Output shape: {getattr(result, 'shape', 'N/A')}")
            return result
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def fit_transform(self, data: Union[str, Path, Any], data_type: Optional[str] = None) -> Any:
        """
        Fit the pipeline and transform the data in one step
        
        Args:
            data: Input data
            data_type: Force a specific data type (optional)
            
        Returns:
            Transformed data ready for model training.
            For datasets (directories), returns (train_data, val_data, test_data) tuple by default.
        """
        # Check if this is a dataset directory that should be split
        if isinstance(data, (str, Path)) and self._is_dataset_directory(data):
            # Detect data type first
            if data_type is None:
                self.detected_type = self.detect_data_type(data)
            else:
                self.detected_type = data_type
                
            logger.info(f"Detected dataset directory with data type: {self.detected_type}")
            
            # Get processor-specific config and initialize processor
            processor_config = self.config.get(self.detected_type, {})
            processor_class = self.processor_classes[self.detected_type]
            self.processor = processor_class(processor_config)
            
            # Split and process the dataset
            train_data, val_data, test_data = self._split_dataset(data, self.detected_type)
            self.is_fitted = True
            
            logger.info("Dataset processing complete - returning (train, val, test) tuple")
            return train_data, val_data, test_data
        
        else:
            # Regular single file/data processing
            return self.fit(data, data_type).transform(data)
    
    def get_processor_info(self) -> Dict:
        """
        Get information about the current processor
        
        Returns:
            dict: Processor information
        """
        if not self.processor:
            return {"status": "No processor initialized"}
        
        info = {
            "detected_type": self.detected_type,
            "processor_class": self.processor.__class__.__name__,
            "is_fitted": self.is_fitted,
            "config": self.processor.get_config()
        }
        
        # Add processor-specific info
        if hasattr(self.processor, 'get_feature_names'):
            info["feature_names"] = self.processor.get_feature_names()
        
        if hasattr(self.processor, 'get_output_shape'):
            try:
                info["output_shape"] = self.processor.get_output_shape()
            except:
                info["output_shape"] = "N/A"
        
        return info
    
    def get_data_info(self, data: Union[str, Path, Any]) -> Dict:
        """
        Get detailed information about the input data
        
        Args:
            data: Input data
            
        Returns:
            dict: Detailed data information
        """
        if isinstance(data, (str, Path)):
            return self.detector.get_detailed_info(data)
        else:
            return {
                "data_type": type(data).__name__,
                "detected_type": self.detect_data_type(data),
                "shape": getattr(data, 'shape', 'N/A'),
                "size": getattr(data, 'size', 'N/A')
            }
    
    def save_pipeline(self, filepath: Union[str, Path]) -> None:
        """
        Save the fitted pipeline to disk
        
        Args:
            filepath: Path to save the pipeline
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted pipeline")
        
        import joblib
        
        pipeline_state = {
            'config': self.config,
            'detected_type': self.detected_type,
            'is_fitted': self.is_fitted,
            'processor_class': self.processor.__class__.__name__
        }
        
        # Save processor separately
        processor_path = str(filepath).replace('.pkl', '_processor.pkl')
        self.processor.save_state(processor_path)
        pipeline_state['processor_path'] = processor_path
        
        joblib.dump(pipeline_state, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: Union[str, Path]) -> 'UniversalPipeline':
        """
        Load a fitted pipeline from disk
        
        Args:
            filepath: Path to load the pipeline from
            
        Returns:
            self: Returns self for method chaining
        """
        import joblib
        
        pipeline_state = joblib.load(filepath)
        
        self.config = pipeline_state['config']
        self.detected_type = pipeline_state['detected_type']
        self.is_fitted = pipeline_state['is_fitted']
        
        # Load processor
        processor_class = self.processor_classes[self.detected_type]
        self.processor = processor_class()
        self.processor.load_state(pipeline_state['processor_path'])
        
        logger.info(f"Pipeline loaded from {filepath}")
        return self
    
    def update_config(self, config: Dict, data_type: Optional[str] = None) -> 'UniversalPipeline':
        """
        Update configuration for specific data type or all types
        
        Args:
            config: Configuration to update
            data_type: Specific data type to update, or None for global config
            
        Returns:
            self: Returns self for method chaining
        """
        if data_type:
            if data_type not in self.config:
                self.config[data_type] = {}
            self.config[data_type].update(config)
        else:
            self.config.update(config)
        
        # Update current processor config if it exists and matches
        if self.processor and self.detected_type == data_type:
            self.processor.set_config(config)
        
        return self
    
    def process_batch(self, data_list: List[Union[str, Path, Any]], 
                     data_types: Optional[List[str]] = None) -> List[Any]:
        """
        Process a batch of data items
        
        Args:
            data_list: List of data items to process
            data_types: Optional list of data types (one per item)
            
        Returns:
            List of processed data
        """
        if data_types and len(data_types) != len(data_list):
            raise ValueError("Length of data_types must match length of data_list")
        
        results = []
        for i, data_item in enumerate(data_list):
            data_type = data_types[i] if data_types else None
            
            # Create a new pipeline for each item (in case of different types)
            item_pipeline = UniversalPipeline(self.config)
            result = item_pipeline.fit_transform(data_item, data_type)
            results.append(result)
        
        return results
    
    def __repr__(self) -> str:
        """String representation of the pipeline"""
        if self.is_fitted:
            return f"UniversalPipeline(fitted=True, type={self.detected_type}, processor={self.processor.__class__.__name__})"
        else:
            return f"UniversalPipeline(fitted=False)"

    def _is_dataset_directory(self, data_path: Union[str, Path]) -> bool:
        """
        Check if the input is a dataset directory that should be split
        
        Args:
            data_path: Path to check
            
        Returns:
            bool: True if it's a dataset directory
        """
        path = Path(data_path)
        if not path.is_dir():
            return False
        
        # Check if it contains many files or subdirectories
        files = list(path.glob('*'))
        if len(files) < 10:  # Small directories, might not need splitting
            return False
            
        # Check for common dataset patterns
        has_images = any(f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'} for f in files if f.is_file())
        has_subdirs = any(f.is_dir() for f in files)
        
        return has_images or has_subdirs

    def _detect_existing_splits(self, data_path: Union[str, Path]) -> Dict:
        """
        Detect if dataset already has train/test/val splits
        
        Args:
            data_path: Path to dataset directory
            
        Returns:
            Dict: Information about detected splits
        """
        path = Path(data_path)
        
        # Check for common split directory names
        split_patterns = {
            'train_test_val': ['train', 'test', 'val'],
            'train_test_valid': ['train', 'test', 'valid'], 
            'training_testing_validation': ['training', 'testing', 'validation'],
            'train_dev_test': ['train', 'dev', 'test'],
            'tr_te_va': ['tr', 'te', 'va']
        }
        
        structure = {
            'has_splits': False,
            'split_type': None,
            'splits': {},
            'total_files': 0
        }
        
        # Check if any split pattern exists
        for pattern_name, dirs in split_patterns.items():
            if all((path / d).exists() and (path / d).is_dir() for d in dirs):
                structure['has_splits'] = True
                structure['split_type'] = pattern_name
                
                logger.info(f"🎉 EXISTING SPLITS DETECTED: {dirs}")
                
                # Count files in each split
                for split_dir in dirs:
                    split_path = path / split_dir
                    
                    # Count data files (look in 'images' subdirectory if it exists)
                    if (split_path / 'images').exists():
                        file_count = len(list((split_path / 'images').glob('*')))
                    else:
                        file_count = len([f for f in split_path.iterdir() if f.is_file()])
                    
                    structure['splits'][split_dir] = {
                        'path': str(split_path),
                        'file_count': file_count
                    }
                    structure['total_files'] += file_count
                
                # Log the split distribution
                total = structure['total_files']
                logger.info("📊 SPLIT DISTRIBUTION:")
                for split_name, split_info in structure['splits'].items():
                    count = split_info['file_count']
                    pct = (count / total * 100) if total > 0 else 0
                    logger.info(f"   {split_name.upper()}: {count} files ({pct:.1f}%)")
                
                break
        
        if not structure['has_splits']:
            logger.info("❌ No existing splits found - will create new splits")
        
        self.dataset_structure = structure
        return structure

    def _split_dataset(self, data_path: Union[str, Path], data_type: str) -> Tuple:
        """
        Split dataset into train/val/test according to configuration
        
        Args:
            data_path: Path to dataset directory
            data_type: Detected data type
            
        Returns:
            Tuple: (train_data, val_data, test_data)
        """
        import random
        import shutil
        from sklearn.model_selection import train_test_split
        
        path = Path(data_path)
        split_config = self.config['dataset']
        
        logger.info(f"Splitting dataset with ratios: {split_config['split_ratios']}")
        
        # Set random seed for reproducibility
        if split_config['random_seed']:
            random.seed(split_config['random_seed'])
            import numpy as np
            np.random.seed(split_config['random_seed'])
        
        if data_type == 'image':
            return self._split_image_dataset(path, split_config)
        elif data_type == 'tabular':
            return self._split_tabular_dataset(path, split_config)
        elif data_type == 'timeseries':
            return self._split_timeseries_dataset(path, split_config)
        else:
            # Generic file-based splitting
            return self._split_generic_dataset(path, split_config)

    def _split_image_dataset(self, path: Path, config: Dict) -> Tuple:
        """Split image dataset"""
        from sklearn.model_selection import train_test_split
        
        # Collect all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        if any((path / subdir).is_dir() for subdir in ['train', 'val', 'test']):
            # Dataset already has splits, respect them
            train_data = self.processor.fit_transform(path / 'train') if (path / 'train').exists() else None
            val_data = self.processor.transform(path / 'val') if (path / 'val').exists() else None
            test_data = self.processor.transform(path / 'test') if (path / 'test').exists() else None
            return train_data, val_data, test_data
        
        # Check if it's a class-based structure (subdirectories are classes)
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if subdirs:
            # Class-based dataset
            all_files = []
            labels = []
            
            for class_dir in subdirs:
                class_files = [f for f in class_dir.iterdir() 
                              if f.suffix.lower() in image_extensions]
                all_files.extend(class_files)
                labels.extend([class_dir.name] * len(class_files))
            
            # Split with stratification to maintain class balance
            train_files, temp_files, train_labels, temp_labels = train_test_split(
                all_files, labels, 
                test_size=(1 - config['split_ratios']['train']),
                stratify=labels if config['stratify'] else None,
                random_state=config['random_seed']
            )
            
            val_size = config['split_ratios']['val'] / (config['split_ratios']['val'] + config['split_ratios']['test'])
            val_files, test_files, val_labels, test_labels = train_test_split(
                temp_files, temp_labels,
                test_size=(1 - val_size),
                stratify=temp_labels if config['stratify'] else None,
                random_state=config['random_seed']
            )
            
        else:
            # Flat directory with images
            all_files = [f for f in path.iterdir() if f.suffix.lower() in image_extensions]
            
            train_files, temp_files = train_test_split(
                all_files, test_size=(1 - config['split_ratios']['train']),
                random_state=config['random_seed']
            )
            
            val_size = config['split_ratios']['val'] / (config['split_ratios']['val'] + config['split_ratios']['test'])
            val_files, test_files = train_test_split(
                temp_files, test_size=(1 - val_size),
                random_state=config['random_seed']
            )
            
            train_labels = val_labels = test_labels = None
        
        # Process each split
        train_data = self.processor.fit_transform(train_files)
        val_data = self.processor.transform(val_files)
        test_data = self.processor.transform(test_files)
        
        logger.info(f"Dataset split complete: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        return train_data, val_data, test_data

    def _split_tabular_dataset(self, path: Path, config: Dict) -> Tuple:
        """Split tabular dataset files"""
        from sklearn.model_selection import train_test_split
        
        csv_files = list(path.glob('*.csv'))
        if not csv_files:
            raise ValueError(f"No CSV files found in {path}")
        
        if len(csv_files) == 1:
            # Single large file, split by rows
            import pandas as pd
            df = pd.read_csv(csv_files[0])
            
            train_df, temp_df = train_test_split(
                df, test_size=(1 - config['split_ratios']['train']),
                random_state=config['random_seed']
            )
            
            val_size = config['split_ratios']['val'] / (config['split_ratios']['val'] + config['split_ratios']['test'])
            val_df, test_df = train_test_split(
                temp_df, test_size=(1 - val_size),
                random_state=config['random_seed']
            )
            
            # Process each split
            train_data = self.processor.fit_transform(train_df)
            val_data = self.processor.transform(val_df)
            test_data = self.processor.transform(test_df)
            
        else:
            # Multiple files, split by files
            train_files, temp_files = train_test_split(
                csv_files, test_size=(1 - config['split_ratios']['train']),
                random_state=config['random_seed']
            )
            
            val_size = config['split_ratios']['val'] / (config['split_ratios']['val'] + config['split_ratios']['test'])
            val_files, test_files = train_test_split(
                temp_files, test_size=(1 - val_size),
                random_state=config['random_seed']
            )
            
            # Process each split
            train_data = self.processor.fit_transform(train_files)
            val_data = self.processor.transform(val_files)
            test_data = self.processor.transform(test_files)
        
        return train_data, val_data, test_data

    def _split_generic_dataset(self, path: Path, config: Dict) -> Tuple:
        """Generic file-based splitting"""
        from sklearn.model_selection import train_test_split
        
        all_files = [f for f in path.iterdir() if f.is_file()]
        
        train_files, temp_files = train_test_split(
            all_files, test_size=(1 - config['split_ratios']['train']),
            random_state=config['random_seed']
        )
        
        val_size = config['split_ratios']['val'] / (config['split_ratios']['val'] + config['split_ratios']['test'])
        val_files, test_files = train_test_split(
            temp_files, test_size=(1 - val_size),
            random_state=config['random_seed']
        )
        
        # Process each split
        train_data = self.processor.fit_transform(train_files)
        val_data = self.processor.transform(val_files)
        test_data = self.processor.transform(test_files)
        
        return train_data, val_data, test_data

    def _split_timeseries_dataset(self, path: Path, config: Dict) -> Tuple:
        """Split time series dataset"""
        # For time series, we typically want temporal splits (not random)
        csv_files = list(path.glob('*.csv'))
        
        if len(csv_files) == 1:
            # Single file - split by time
            import pandas as pd
            df = pd.read_csv(csv_files[0])
            
            # Sort by time column if available
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_cols:
                df = df.sort_values(time_cols[0])
            
            # Temporal split (no shuffling for time series)
            n = len(df)
            train_end = int(n * config['split_ratios']['train'])
            val_end = train_end + int(n * config['split_ratios']['val'])
            
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
            
            train_data = self.processor.fit_transform(train_df)
            val_data = self.processor.transform(val_df)
            test_data = self.processor.transform(test_df)
            
        else:
            # Multiple files - split by files
            from sklearn.model_selection import train_test_split
            
            train_files, temp_files = train_test_split(
                csv_files, test_size=(1 - config['split_ratios']['train']),
                random_state=config['random_seed']
            )
            
            val_size = config['split_ratios']['val'] / (config['split_ratios']['val'] + config['split_ratios']['test'])
            val_files, test_files = train_test_split(
                temp_files, test_size=(1 - val_size),
                random_state=config['random_seed']
            )
            
            train_data = self.processor.fit_transform(train_files)
            val_data = self.processor.transform(val_files)
            test_data = self.processor.transform(test_files)
        
        return train_data, val_data, test_data


# Convenience function for quick processing
def process_data(data: Union[str, Path, Any], config: Optional[Dict] = None, 
                data_type: Optional[str] = None) -> Any:
    """
    Convenience function to quickly process data
    
    Args:
        data: Input data
        config: Configuration dictionary
        data_type: Force specific data type
        
    Returns:
        Processed data ready for model training
    """
    pipeline = UniversalPipeline(config)
    return pipeline.fit_transform(data, data_type) 