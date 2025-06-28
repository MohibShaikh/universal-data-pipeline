"""
Universal Data Pipeline Demo
Quick demonstration of the pipeline's capabilities
"""

from universal_pipeline import UniversalPipeline, process_data
import numpy as np
import pandas as pd
import os


def demo():
    """Run a quick demo of the universal pipeline"""
    print("🚀 Universal Data Pipeline Demo")
    print("=" * 50)
    
    # 1. Tabular Data Demo
    print("\n📊 1. Tabular Data Processing")
    print("-" * 30)
    
    # Create sample data
    data = {
        'age': np.random.randint(18, 80, 500),
        'income': np.random.normal(50000, 15000, 500),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 500),
        'purchased': np.random.choice([0, 1], 500)
    }
    df = pd.DataFrame(data)
    df.to_csv('demo_data.csv', index=False)
    
    # Process automatically
    pipeline = UniversalPipeline({
        'tabular': {'target_column': 'purchased'}
    })
    X, y = pipeline.fit_transform('demo_data.csv')
    
    print(f"✓ Detected type: {pipeline.detected_type}")
    print(f"✓ Input shape: {df.shape}")
    print(f"✓ Processed X shape: {X.shape}")
    print(f"✓ Processed y shape: {y.shape}")
    print(f"✓ Features: {pipeline.processor.get_feature_names()[:3]}... ({len(pipeline.processor.get_feature_names())} total)")
    
    # Clean up
    os.remove('demo_data.csv')
    
    # 2. Image Data Demo
    print("\n🖼️  2. Image Data Processing")
    print("-" * 30)
    
    # Create sample image
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Process automatically
    processed_image = process_data(sample_image, {
        'image': {'image_size': (128, 128)}
    })
    
    print(f"✓ Input shape: {sample_image.shape}")
    print(f"✓ Output shape: {processed_image.shape}")
    print(f"✓ Data type: {type(processed_image)}")
    print(f"✓ Value range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    
    # 3. Time Series Demo
    print("\n📈 3. Time Series Data Processing")
    print("-" * 30)
    
    # Create sample time series
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    ts_data = pd.DataFrame({
        'date': dates,
        'temperature': 20 + 10 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.randn(365) * 2,
        'humidity': 50 + 20 * np.cos(np.arange(365) * 2 * np.pi / 365) + np.random.randn(365) * 5,
        'pressure': 1013 + np.random.randn(365) * 10
    })
    ts_data.to_csv('demo_timeseries.csv', index=False)
    
    # Process automatically
    processed_ts = process_data('demo_timeseries.csv', {
        'timeseries': {
            'time_column': 'date',
            'sequence_length': 7  # Weekly sequences
        }
    })
    
    print(f"✓ Input shape: {ts_data.shape}")
    print(f"✓ Output shape: {processed_ts.shape}")
    print(f"✓ Sequence length: 7 days")
    print(f"✓ Features: temperature, humidity, pressure")
    
    # Clean up
    os.remove('demo_timeseries.csv')
    
    # 4. Automatic Detection Demo
    print("\n🔍 4. Automatic Data Type Detection")
    print("-" * 30)
    
    detector_pipeline = UniversalPipeline()
    
    test_cases = [
        (np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), "RGB Image Array"),
        (pd.DataFrame({'x': range(10), 'y': range(10)}), "DataFrame"),
        (np.random.randn(1000), "1D Array")
    ]
    
    for data, description in test_cases:
        detected = detector_pipeline.detect_data_type(data)
        print(f"✓ {description}: Detected as '{detected}'")
    
    # 5. Performance Summary
    print("\n⚡ 5. Performance Summary")
    print("-" * 30)
    print("✓ Supports: Tabular, Image, Video, Audio, Time Series")
    print("✓ Auto-detection: File extension + content analysis")
    print("✓ Auto-preprocessing: Scaling, encoding, normalization")
    print("✓ Ready for ML: Output directly usable in models")
    print("✓ Flexible: Configurable for custom requirements")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Check examples/basic_usage.py for more detailed examples")
    print("2. Customize configurations for your specific needs")
    print("3. Use pipeline.fit_transform(your_data) for any data type")


if __name__ == "__main__":
    demo() 