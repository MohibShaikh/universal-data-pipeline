#!/usr/bin/env python3
"""
Test Smart Dataset Handler on your test directory
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_dataset_handler import SmartDatasetManager
from pathlib import Path

def test_your_directory():
    """Test the smart dataset handler on your test directory"""
    print("🎯 TESTING YOUR TEST DIRECTORY")
    print("="*80)
    
    test_dir = "test"
    
    # Check if directory exists
    if not Path(test_dir).exists():
        print(f"❌ Directory '{test_dir}' not found!")
        return
    
    print(f"📁 Analyzing directory: {test_dir}")
    print(f"🔍 Looking for existing train/test/val splits...")
    
    # Test with YOLOv8 (good for object detection)
    model_name = "yolov8"
    
    try:
        # Create smart dataset manager
        manager = SmartDatasetManager(test_dir, model_name)
        
        # Show split information
        split_info = manager.get_split_info()
        
        print(f"\n📊 YOUR DATASET ANALYSIS:")
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
        
        # Test a training workflow if splits are detected
        if split_info['has_existing_splits']:
            print(f"\n🚀 TRAINING WORKFLOW TEST:")
            manager.demo_training_workflow()
        else:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   - Your test directory contains {total_samples} images")
            print(f"   - Consider organizing into train/test/val subdirectories")
            print(f"   - Typical split: 70% train, 15% validation, 15% test")
            print(f"   - Or use sklearn.model_selection.train_test_split")
        
        # Test with a few different models
        print(f"\n🌐 TESTING WITH DIFFERENT MODELS:")
        models_to_test = ['resnet50', 'efficientnet_b0', 'facenet']
        
        for model in models_to_test:
            try:
                test_manager = SmartDatasetManager(test_dir, model)
                test_split_info = test_manager.get_split_info()
                total = sum(info['size'] for info in test_split_info['splits'].values())
                print(f"  ✅ {model:<15} → Ready for {total} images")
            except Exception as e:
                print(f"  ❌ {model:<15} → Error: {str(e)[:50]}...")
        
        return manager
        
    except Exception as e:
        print(f"❌ Error analyzing directory: {e}")
        return None

def show_directory_structure():
    """Show the structure of the test directory"""
    print(f"\n📂 TEST DIRECTORY STRUCTURE:")
    print("="*50)
    
    test_dir = Path("test")
    if not test_dir.exists():
        print("❌ Test directory not found")
        return
    
    def print_tree(directory, prefix="", max_files=10):
        try:
            items = list(directory.iterdir())
            dirs = [item for item in items if item.is_dir()]
            files = [item for item in items if item.is_file()]
            
            # Show directories first
            for i, dir_item in enumerate(dirs):
                is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                print(f"{prefix}{'└── ' if is_last_dir else '├── '}{dir_item.name}/")
                
                extension = "    " if is_last_dir else "│   "
                print_tree(dir_item, prefix + extension, max_files=5)
            
            # Show files (limited)
            for i, file_item in enumerate(files[:max_files]):
                is_last = i == len(files[:max_files]) - 1
                print(f"{prefix}{'└── ' if is_last else '├── '}{file_item.name}")
            
            if len(files) > max_files:
                print(f"{prefix}└── ... and {len(files) - max_files} more files")
                
        except PermissionError:
            print(f"{prefix}└── [Permission Denied]")
    
    print(f"test/")
    print_tree(test_dir)

def main():
    """Main test function"""
    print("🧠 SMART DATASET HANDLER - TESTING YOUR DIRECTORY")
    print("="*80)
    
    # Show directory structure first
    show_directory_structure()
    
    # Test the smart handler
    manager = test_your_directory()
    
    if manager:
        print(f"\n" + "="*80)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*80)
        print("✅ Your test directory has been analyzed")
        print("✅ Pipeline created and ready for training")
        print("✅ DataLoaders configured with appropriate settings")
        
        if manager.detector.structure['has_splits']:
            print("✅ Existing splits detected and preserved")
            print("💡 Ready to train immediately - no splitting needed!")
        else:
            print("💡 Consider organizing images into train/test/val folders")
            print("💡 This will prevent accidental data leakage")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. If splits detected: Start training with the DataLoaders")
    print(f"   2. If no splits: Organize into train/test/val directories")
    print(f"   3. Use: manager.dataloaders['train'] for training")
    print(f"   4. Use: manager.dataloaders['val'] for validation")

if __name__ == "__main__":
    main() 