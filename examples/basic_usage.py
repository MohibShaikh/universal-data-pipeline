"""
Basic Usage Examples for Universal Data Pipeline
This demonstrates how to use the pipeline for different data types
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universal_pipeline import UniversalPipeline, process_data
import numpy as np
import pandas as pd


def example_tabular_data():
    """Example: Processing tabular data (CSV)"""
    print("=== Tabular Data Example ===")
    
    # Create sample tabular data
    data = {
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.randint(0, 2, 1000)
    }
    df = pd.DataFrame(data)
    df.to_csv('sample_data.csv', index=False)
    
    # Create pipeline
    pipeline = UniversalPipeline({
        'tabular': {
            'target_column': 'target',
            'scaling_method': 'standard'
        }
    })
    
    # Process data
    X, y = pipeline.fit_transform('sample_data.csv')
    
    print(f"Input shape: {df.shape}")
    print(f"Output X shape: {X.shape}")
    print(f"Output y shape: {y.shape}")
    print(f"Detected type: {pipeline.detected_type}")
    print(f"Feature names: {pipeline.processor.get_feature_names()}")
    
    # Clean up
    os.remove('sample_data.csv')
    print()


def example_image_data():
    """Example: Processing image data"""
    print("=== Image Data Example ===")
    
    # Create sample image data as numpy array
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Create pipeline with custom config
    config = {
        'image': {
            'image_size': (256, 256),
            'normalize': True
        }
    }
    pipeline = UniversalPipeline(config)
    
    # Process image
    processed_image = pipeline.fit_transform(sample_image)
    
    print(f"Input shape: {sample_image.shape}")
    print(f"Output shape: {processed_image.shape}")
    print(f"Detected type: {pipeline.detected_type}")
    print(f"Output type: {type(processed_image)}")
    print()


def example_timeseries_data():
    """Example: Processing time series data"""
    print("=== Time Series Data Example ===")
    
    # Create sample time series data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    ts_data = {
        'date': dates,
        'value1': np.cumsum(np.random.randn(1000)) + 100,
        'value2': np.sin(np.arange(1000) * 2 * np.pi / 365) + np.random.randn(1000) * 0.1,
        'value3': np.random.exponential(2, 1000)
    }
    ts_df = pd.DataFrame(ts_data)
    ts_df.to_csv('sample_timeseries.csv', index=False)
    
    # Create pipeline
    pipeline = UniversalPipeline({
        'timeseries': {
            'time_column': 'date',
            'sequence_length': 30,
            'scaling_method': 'minmax'
        }
    })
    
    # Process data
    processed_ts = pipeline.fit_transform('sample_timeseries.csv')
    
    print(f"Input shape: {ts_df.shape}")
    print(f"Output shape: {processed_ts.shape}")
    print(f"Detected type: {pipeline.detected_type}")
    print(f"Feature columns: {pipeline.processor.get_feature_names()}")
    
    # Clean up
    os.remove('sample_timeseries.csv')
    print()


def example_audio_data():
    """Example: Processing audio data (simulated)"""
    print("=== Audio Data Example ===")
    
    # Create sample audio data as numpy array
    # Simulating 3 seconds of audio at 22050 Hz
    sample_rate = 22050
    duration = 3
    audio_data = np.random.randn(sample_rate * duration)
    
    # Create pipeline
    pipeline = UniversalPipeline({
        'audio': {
            'feature_type': 'mfcc',
            'n_mels': 64,
            'duration': duration
        }
    })
    
    # Process audio (force type since we're using numpy array)
    processed_audio = pipeline.fit_transform(audio_data, data_type='audio')
    
    print(f"Input shape: {audio_data.shape}")
    print(f"Output shape: {processed_audio.shape}")
    print(f"Detected type: {pipeline.detected_type}")
    print()


def example_automatic_detection():
    """Example: Automatic data type detection"""
    print("=== Automatic Detection Example ===")
    
    # Create different types of sample data
    
    # 1. Tabular data
    df = pd.DataFrame({
        'x': range(100),
        'y': np.random.randn(100),
        'z': np.random.choice(['A', 'B'], 100)
    })
    df.to_csv('auto_tabular.csv', index=False)
    
    # 2. Image-like data
    image_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # 3. Time series data with datetime
    ts_data = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=200),
        'sensor1': np.cumsum(np.random.randn(200)),
        'sensor2': np.random.randn(200)
    })
    ts_data.to_csv('auto_timeseries.csv', index=False)
    
    # Test automatic detection
    pipeline = UniversalPipeline()
    
    # Test each data type
    data_samples = [
        ('auto_tabular.csv', 'CSV file'),
        (image_data, 'Numpy image array'),
        ('auto_timeseries.csv', 'Time series CSV')
    ]
    
    for data, description in data_samples:
        detected_type = pipeline.detect_data_type(data)
        print(f"{description}: Detected as '{detected_type}'")
    
    # Clean up
    os.remove('auto_tabular.csv')
    os.remove('auto_timeseries.csv')
    print()


def example_batch_processing():
    """Example: Processing multiple files at once"""
    print("=== Batch Processing Example ===")
    
    # Create multiple sample files
    for i in range(3):
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        df.to_csv(f'batch_file_{i}.csv', index=False)
    
    # Process batch
    pipeline = UniversalPipeline({
        'tabular': {'target_column': 'target'}
    })
    
    file_list = [f'batch_file_{i}.csv' for i in range(3)]
    results = pipeline.process_batch(file_list)
    
    print(f"Processed {len(results)} files")
    for i, (X, y) in enumerate(results):
        print(f"File {i}: X shape {X.shape}, y shape {y.shape}")
    
    # Clean up
    for i in range(3):
        os.remove(f'batch_file_{i}.csv')
    print()


def example_pipeline_info():
    """Example: Getting pipeline and data information"""
    print("=== Pipeline Information Example ===")
    
    # Create sample data
    df = pd.DataFrame({
        'numeric1': np.random.randn(100),
        'numeric2': np.random.randn(100),
        'category': np.random.choice(['X', 'Y', 'Z'], 100)
    })
    
    # Create and fit pipeline
    pipeline = UniversalPipeline()
    processed_data = pipeline.fit_transform(df)
    
    # Get information
    print("Data Info:")
    data_info = pipeline.get_data_info(df)
    for key, value in data_info.items():
        print(f"  {key}: {value}")
    
    print("\nProcessor Info:")
    processor_info = pipeline.get_processor_info()
    for key, value in processor_info.items():
        print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    print("Universal Data Pipeline - Basic Usage Examples\n")
    
    # Run all examples
    example_tabular_data()
    example_image_data()
    example_timeseries_data()
    example_audio_data()
    example_automatic_detection()
    example_batch_processing()
    example_pipeline_info()
    
    print("All examples completed successfully!") 