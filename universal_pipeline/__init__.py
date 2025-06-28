"""
Universal Data Pipeline
A comprehensive pipeline system that automatically detects data types and applies appropriate preprocessing.
"""

from .pipeline import UniversalPipeline
from .data_detector import DataTypeDetector
from .processors import (
    TabularProcessor,
    ImageProcessor,
    VideoProcessor,
    AudioProcessor,
    TimeSeriesProcessor
)

# Convenience function for quick processing
def process_data(data, config=None):
    """
    Convenience function for quick data processing
    
    Args:
        data: Input data (file path, DataFrame, array, etc.)
        config: Optional configuration dictionary
        
    Returns:
        Processed data
    """
    pipeline = UniversalPipeline(config)
    return pipeline.fit_transform(data)

__version__ = "1.0.0"
__author__ = "Universal Pipeline Team"

__all__ = [
    "UniversalPipeline",
    "DataTypeDetector", 
    "TabularProcessor",
    "ImageProcessor",
    "VideoProcessor",
    "AudioProcessor",
    "TimeSeriesProcessor",
    "process_data"
] 