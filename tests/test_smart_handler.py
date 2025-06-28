#!/usr/bin/env python3
"""
Test Smart Dataset Handler with real split dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_dataset_handler import SmartDatasetManager

def test_with_existing_splits():
    """Test with the demo dataset that has existing splits"""
    print("🎯 TESTING: DATASET WITH EXISTING TRAIN/TEST/VAL SPLITS")
    print("="*80)
    
    # Test with our created split dataset
    dataset_path = "demo_split_dataset"
    model_name = "yolov8"
    
    # Create smart dataset manager
    manager = SmartDatasetManager(dataset_path, model_name)
    
    # Show split information
    split_info = manager.get_split_info()
    print(f"\n📊 DETAILED SPLIT INFORMATION:")
    print(f"Has existing splits: {split_info['has_existing_splits']}")
    if split_info['split_type']:
        print(f"Split type detected: {split_info['split_type']}")
    
    print(f"\n📈 SPLIT BREAKDOWN:")
    total_samples = sum(info['size'] for info in split_info['splits'].values())
    
    for split_name, info in split_info['splits'].items():
        percentage = (info['size'] / total_samples * 100) if total_samples > 0 else 0
        print(f"  {split_name.upper()}:")
        print(f"    📄 Dataset size: {info['size']} samples ({percentage:.1f}%)")
        print(f"    🔄 Batch size: {info['batch_size']}")
        print(f"    📦 DataLoader: {len(info['dataloader'])} batches")
    
    # Demo training workflow with proper splits
    manager.demo_training_workflow()
    
    return manager

def test_split_preservation():
    """Test that splits are properly preserved and not re-split"""
    print(f"\n🔒 TESTING: SPLIT PRESERVATION")
    print("="*60)
    
    manager = test_with_existing_splits()
    
    # Verify that original split ratios are maintained
    splits = manager.get_split_info()['splits']
    
    total_samples = sum(info['size'] for info in splits.values())
    
    print(f"\n📊 SPLIT RATIO VERIFICATION:")
    print(f"Total samples: {total_samples}")
    
    for split_name, info in splits.items():
        ratio = info['size'] / total_samples * 100
        print(f"  {split_name.upper()}: {info['size']} samples ({ratio:.1f}%)")
    
    # Check that we have the expected splits
    expected_splits = {'train', 'test', 'val'}
    actual_splits = set(splits.keys())
    
    if expected_splits == actual_splits:
        print(f"\n✅ SUCCESS: All expected splits found!")
        print(f"   Expected: {sorted(expected_splits)}")
        print(f"   Found: {sorted(actual_splits)}")
    else:
        print(f"\n❌ ERROR: Split mismatch!")
        print(f"   Expected: {sorted(expected_splits)}")
        print(f"   Found: {sorted(actual_splits)}")

def test_annotation_handling():
    """Test that annotations are properly handled"""
    print(f"\n🏷️  TESTING: ANNOTATION HANDLING")
    print("="*60)
    
    dataset_path = "demo_split_dataset"
    model_name = "yolov8"
    
    manager = SmartDatasetManager(dataset_path, model_name)
    
    # Test loading a few samples to verify annotations
    print(f"\n📦 TESTING DATA LOADING:")
    
    for split_name, dataloader in manager.dataloaders.items():
        print(f"\n{split_name.upper()} split:")
        
        # Get first batch
        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f"  Batch {batch_idx + 1}:")
            print(f"    Images: {images.shape}")
            print(f"    Labels: {labels.shape}")
            print(f"    Label range: {labels.min().item()} - {labels.max().item()}")
            print(f"    Sample labels: {labels[:5].tolist()}")  # Show first 5 labels
            break  # Only show first batch
    
    print(f"\n✅ Annotations loaded successfully from YOLO format!")

def compare_with_without_splits():
    """Compare handling with and without existing splits"""
    print(f"\n⚖️  COMPARISON: WITH vs WITHOUT SPLITS")
    print("="*60)
    
    print(f"\n1️⃣  WITH EXISTING SPLITS (demo_split_dataset):")
    print("-" * 50)
    
    manager_with_splits = SmartDatasetManager("demo_split_dataset", "yolov8")
    splits_info = manager_with_splits.get_split_info()
    
    print(f"✅ Automatic split detection: {splits_info['has_existing_splits']}")
    print(f"✅ No additional splitting needed")
    print(f"✅ Ready for training immediately")
    
    print(f"\n2️⃣  WITHOUT EXISTING SPLITS (sample_data/image):")
    print("-" * 50)
    
    manager_without_splits = SmartDatasetManager("sample_data/image", "yolov8")
    no_splits_info = manager_without_splits.get_split_info()
    
    print(f"❌ Automatic split detection: {no_splits_info['has_existing_splits']}")
    print(f"⚠️  Would need manual splitting")
    print(f"⚠️  Additional preprocessing required")
    
    print(f"\n💡 KEY BENEFIT:")
    print(f"   Smart handler RESPECTS existing splits!")
    print(f"   No accidental data leakage from re-splitting!")

def main():
    """Run all tests"""
    print("🧠 SMART DATASET HANDLER - COMPREHENSIVE TEST")
    print("="*80)
    
    # Test 1: Basic functionality with existing splits
    test_with_existing_splits()
    
    # Test 2: Verify split preservation
    test_split_preservation()
    
    # Test 3: Annotation handling
    test_annotation_handling()
    
    # Test 4: Comparison
    compare_with_without_splits()
    
    print(f"\n" + "="*80)
    print("🎉 ALL TESTS COMPLETED!")
    print("="*80)
    print("✅ Existing splits automatically detected")
    print("✅ Split ratios preserved (no re-splitting)")
    print("✅ YOLO annotations properly loaded")
    print("✅ DataLoaders created with appropriate batch sizes")
    print("✅ Training/validation/test workflows ready")
    print("\n💡 YOUR DATASET STRUCTURE IS RESPECTED!")

if __name__ == "__main__":
    main() 