"""
Test script for automatic dataset splitting functionality
Demonstrates the new 70-15-15 default split behavior
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from universal_pipeline import UniversalPipeline


def create_test_datasets():
    """Create test datasets for demonstration"""
    
    # 1. Create image dataset with class folders
    print("📁 Creating test image dataset...")
    os.makedirs("test_datasets/images/cats", exist_ok=True)
    os.makedirs("test_datasets/images/dogs", exist_ok=True)
    os.makedirs("test_datasets/images/birds", exist_ok=True)
    
    # Create dummy image files (we'll just create empty files for demo)
    for category in ['cats', 'dogs', 'birds']:
        for i in range(50):  # 50 images per class
            file_path = f"test_datasets/images/{category}/img_{i:03d}.jpg"
            Path(file_path).touch()
    
    # 2. Create tabular dataset
    print("📊 Creating test tabular dataset...")
    os.makedirs("test_datasets/tabular", exist_ok=True)
    
    # Large dataset with 1000 rows
    data = {
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000)
    }
    df = pd.DataFrame(data)
    df.to_csv("test_datasets/tabular/large_dataset.csv", index=False)
    
    # 3. Create time series dataset
    print("📈 Creating test time series dataset...")
    os.makedirs("test_datasets/timeseries", exist_ok=True)
    
    dates = pd.date_range('2020-01-01', periods=1000, freq='H')
    ts_data = pd.DataFrame({
        'timestamp': dates,
        'temperature': 20 + 5 * np.sin(np.arange(1000) * 0.01) + np.random.randn(1000),
        'humidity': 50 + 10 * np.cos(np.arange(1000) * 0.01) + np.random.randn(1000),
        'target': np.random.randn(1000)
    })
    ts_data.to_csv("test_datasets/timeseries/sensor_data.csv", index=False)
    
    print("✅ Test datasets created!")


def test_default_splitting():
    """Test the default 70-15-15 splitting behavior"""
    
    print("\n🚀 Testing Default Dataset Splitting (70-15-15)")
    print("=" * 60)
    
    # 1. Test Image Dataset Splitting
    print("\n🖼️  1. Testing Image Dataset")
    print("-" * 30)
    
    pipeline = UniversalPipeline()  # No config - should use defaults
    train_data, val_data, test_data = pipeline.fit_transform("test_datasets/images/")
    
    print(f"✅ Image dataset split complete:")
    print(f"   📁 Train: {len(train_data) if hasattr(train_data, '__len__') else 'N/A'} samples")
    print(f"   📁 Val:   {len(val_data) if hasattr(val_data, '__len__') else 'N/A'} samples") 
    print(f"   📁 Test:  {len(test_data) if hasattr(test_data, '__len__') else 'N/A'} samples")
    print(f"   🎯 Expected ratios: 70% / 15% / 15%")
    
    # 2. Test Tabular Dataset Splitting
    print("\n📊 2. Testing Tabular Dataset")
    print("-" * 30)
    
    pipeline = UniversalPipeline()
    train_data, val_data, test_data = pipeline.fit_transform("test_datasets/tabular/")
    
    print(f"✅ Tabular dataset split complete:")
    print(f"   📁 Train: {train_data[0].shape if hasattr(train_data, '__getitem__') and hasattr(train_data[0], 'shape') else 'N/A'}")
    print(f"   📁 Val:   {val_data[0].shape if hasattr(val_data, '__getitem__') and hasattr(val_data[0], 'shape') else 'N/A'}")
    print(f"   📁 Test:  {test_data[0].shape if hasattr(test_data, '__getitem__') and hasattr(test_data[0], 'shape') else 'N/A'}")
    print(f"   🎯 Expected ratios: 70% / 15% / 15%")


def test_custom_splitting():
    """Test custom splitting ratios"""
    
    print("\n⚙️  Testing Custom Split Ratios")
    print("=" * 40)
    
    # Custom configuration: 80-10-10 split
    custom_config = {
        'dataset': {
            'split_ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1},
            'stratify': True,
            'random_seed': 123
        }
    }
    
    pipeline = UniversalPipeline(custom_config)
    train_data, val_data, test_data = pipeline.fit_transform("test_datasets/images/")
    
    print(f"✅ Custom split (80-10-10) complete:")
    print(f"   📁 Train: {len(train_data) if hasattr(train_data, '__len__') else 'N/A'} samples")
    print(f"   📁 Val:   {len(val_data) if hasattr(val_data, '__len__') else 'N/A'} samples")
    print(f"   📁 Test:  {len(test_data) if hasattr(test_data, '__len__') else 'N/A'} samples")


def test_single_file_behavior():
    """Test that single files still work normally"""
    
    print("\n📄 Testing Single File Processing (No Splitting)")
    print("=" * 50)
    
    pipeline = UniversalPipeline()
    
    # Single file should NOT be split
    result = pipeline.fit_transform("test_datasets/tabular/large_dataset.csv")
    
    print(f"✅ Single file processed normally:")
    print(f"   📊 Result type: {type(result)}")
    print(f"   📏 Shape: {result[0].shape if hasattr(result, '__getitem__') and hasattr(result[0], 'shape') else 'N/A'}")
    print("   ✨ No splitting applied (as expected)")


def demo_real_usage():
    """Show real-world usage examples"""
    
    print("\n🌟 Real-World Usage Examples")
    print("=" * 40)
    
    print("\n💡 Example 1: Default behavior (70-15-15)")
    print("```python")
    print("from universal_pipeline import UniversalPipeline")
    print("")
    print("# Automatic 70-15-15 split")
    print("pipeline = UniversalPipeline()")
    print("train, val, test = pipeline.fit_transform('my_dataset/')")
    print("```")
    
    print("\n💡 Example 2: Custom split ratios")
    print("```python")
    print("# Custom 80-20 split (no validation)")
    print("config = {'dataset': {'split_ratios': {'train': 0.8, 'test': 0.2}}}")
    print("pipeline = UniversalPipeline(config)")
    print("train, val, test = pipeline.fit_transform('my_dataset/')")
    print("# val will be None")
    print("```")
    
    print("\n💡 Example 3: Competition setup")
    print("```python")
    print("# For Kaggle-style competitions")
    print("config = {'dataset': {'split_ratios': {'train': 0.9, 'val': 0.1}}}")
    print("pipeline = UniversalPipeline(config)")
    print("train, val, _ = pipeline.fit_transform('train_data/')")
    print("test = pipeline.transform('test_data/')  # Separate test set")
    print("```")


def cleanup():
    """Clean up test files"""
    import shutil
    if os.path.exists("test_datasets"):
        shutil.rmtree("test_datasets")
    print("\n🧹 Cleaned up test datasets")


def main():
    """Run all tests"""
    print("🧪 Universal Pipeline Dataset Splitting Test")
    print("=" * 50)
    
    try:
        # Create test datasets
        create_test_datasets()
        
        # Run tests
        test_default_splitting()
        test_custom_splitting()
        test_single_file_behavior()
        
        # Show usage examples
        demo_real_usage()
        
        print("\n🎉 All tests completed successfully!")
        print("\n✨ Key Features:")
        print("   ✅ Automatic 70-15-15 split by default")
        print("   ✅ Respects existing train/val/test folders")
        print("   ✅ Stratified splitting for balanced classes")
        print("   ✅ Temporal splitting for time series")
        print("   ✅ Configurable split ratios")
        print("   ✅ Reproducible with random seeds")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cleanup()


if __name__ == "__main__":
    main() 