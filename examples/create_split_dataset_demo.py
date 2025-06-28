#!/usr/bin/env python3
"""
Create a demo dataset with existing train/test/valid splits
"""

import os
import numpy as np
import cv2
from pathlib import Path
import shutil

def create_split_dataset():
    """Create a dataset with proper train/test/valid splits"""
    
    # Create directory structure
    base_dir = Path("demo_split_dataset")
    
    # Remove existing if present
    if base_dir.exists():
        shutil.rmtree(base_dir)
    
    # Create split directories
    splits = ['train', 'test', 'val']
    
    for split in splits:
        # Create images directory
        images_dir = base_dir / split / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create labels directory (YOLO format)
        labels_dir = base_dir / split / 'labels'
        labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created directory structure:")
    print(f"   {base_dir}/")
    print(f"   ├── train/")
    print(f"   │   ├── images/")
    print(f"   │   └── labels/")
    print(f"   ├── test/")
    print(f"   │   ├── images/")
    print(f"   │   └── labels/")
    print(f"   └── val/")
    print(f"       ├── images/")
    print(f"       └── labels/")
    
    # Create sample images and labels for each split
    split_sizes = {'train': 20, 'test': 8, 'val': 6}
    
    for split, num_files in split_sizes.items():
        print(f"\n📸 Creating {num_files} images for {split} split...")
        
        images_dir = base_dir / split / 'images'
        labels_dir = base_dir / split / 'labels'
        
        for i in range(num_files):
            # Create random image
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some patterns to make it look more realistic
            cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), 2)
            cv2.circle(img, (400, 300), 50, (0, 255, 0), -1)
            
            # Save image
            img_name = f"{split}_image_{i:04d}.jpg"
            img_path = images_dir / img_name
            cv2.imwrite(str(img_path), img)
            
            # Create YOLO format label
            # Format: class_id center_x center_y width height (normalized)
            label_content = []
            
            # Add 1-3 random bounding boxes
            num_boxes = np.random.randint(1, 4)
            for box_idx in range(num_boxes):
                class_id = np.random.randint(0, 80)  # COCO has 80 classes
                center_x = np.random.uniform(0.1, 0.9)
                center_y = np.random.uniform(0.1, 0.9)
                width = np.random.uniform(0.05, 0.3)
                height = np.random.uniform(0.05, 0.3)
                
                label_content.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            
            # Save label file
            label_name = f"{split}_image_{i:04d}.txt"
            label_path = labels_dir / label_name
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_content))
        
        print(f"   ✅ Created {num_files} images and labels")
    
    # Create dataset info file
    info = {
        "dataset_name": "Demo Split Dataset",
        "splits": {
            "train": split_sizes['train'],
            "test": split_sizes['test'], 
            "val": split_sizes['val']
        },
        "total_images": sum(split_sizes.values()),
        "annotation_format": "yolo",
        "num_classes": 80,
        "class_names": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light"]  # First 10 COCO classes
    }
    
    import json
    with open(base_dir / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✅ Dataset created successfully!")
    print(f"📊 Total: {sum(split_sizes.values())} images")
    print(f"   - Train: {split_sizes['train']} images")
    print(f"   - Test: {split_sizes['test']} images") 
    print(f"   - Val: {split_sizes['val']} images")
    print(f"🏷️  Annotation format: YOLO")
    
    return base_dir

if __name__ == "__main__":
    create_split_dataset() 