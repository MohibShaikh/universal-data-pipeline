#!/usr/bin/env python3
"""
Simple Dataset Analyzer
Usage: python analyze_dataset.py <directory_name>
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_dataset_handler import SmartDatasetManager

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_dataset.py <directory_name>")
        print("Example: python analyze_dataset.py test")
        print("Example: python analyze_dataset.py demo_split_dataset")
        sys.exit(1)
    
    directory_name = sys.argv[1]
    
    if not os.path.exists(directory_name):
        print(f"❌ Directory '{directory_name}' not found!")
        sys.exit(1)
    
    print(f"🔍 ANALYZING DATASET: {directory_name}")
    print("="*60)
    
    # Analyze with YOLOv8 pipeline (most common for object detection)
    manager = SmartDatasetManager(directory_name, 'yolov8')
    
    # Show split information
    split_info = manager.get_split_info()
    
    if split_info['has_existing_splits']:
        print(f"\n🎉 RESULT: EXISTING SPLITS DETECTED!")
        print(f"✅ Ready for training immediately")
        print(f"✅ No additional splitting needed")
        
        # Show quick summary
        total_samples = sum(info['size'] for info in split_info['splits'].values())
        print(f"\n📊 QUICK SUMMARY:")
        for split_name, info in split_info['splits'].items():
            percentage = (info['size'] / total_samples * 100) if total_samples > 0 else 0
            print(f"   {split_name.upper()}: {info['size']} images ({percentage:.1f}%)")
        
    else:
        print(f"\n⚠️  RESULT: NO SPLITS DETECTED")
        print(f"💡 You need to split this dataset into train/test/val")
        print(f"📄 Total images found: {sum(info['size'] for info in split_info['splits'].values())}")

if __name__ == "__main__":
    main() 