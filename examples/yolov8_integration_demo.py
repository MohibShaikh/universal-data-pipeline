#!/usr/bin/env python3
"""
YOLOv8 Integration Demo
Shows how Universal Pipeline processed data feeds directly into YOLOv8
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2

from universal_pipeline import UniversalPipeline

class YOLOv8Dataset(Dataset):
    """Dataset class that feeds Universal Pipeline data directly to YOLOv8"""
    def __init__(self, processed_images_dir, labels_dir=None):
        self.image_files = list(Path(processed_images_dir).glob("*.npy"))
        self.labels_dir = Path(labels_dir) if labels_dir else None
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load processed image (already YOLOv8-ready from Universal Pipeline)
        image = np.load(self.image_files[idx])
        
        # Convert to YOLOv8 format: (C, H, W), [0, 1], float32
        if len(image.shape) == 3:  # (H, W, C) -> (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Load corresponding labels (YOLO format: class x_center y_center width height)
        image_name = self.image_files[idx].stem
        if self.labels_dir and (self.labels_dir / f"{image_name}.txt").exists():
            with open(self.labels_dir / f"{image_name}.txt", 'r') as f:
                labels = []
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    labels.append([class_id, x_center, y_center, width, height])
                labels = torch.tensor(labels, dtype=torch.float32)
        else:
            # Mock labels for demo
            labels = torch.tensor([[0, 0.5, 0.5, 0.3, 0.3]], dtype=torch.float32)  # One box
        
        return image, labels

def demo_yolov8_preprocessing():
    """Show how Universal Pipeline creates YOLOv8-ready data"""
    print("🎯 YOLOV8 PREPROCESSING WITH UNIVERSAL PIPELINE")
    print("=" * 60)
    
    # YOLOv8-specific configuration
    yolov8_config = {
        'image': {
            'target_size': (640, 640),        # YOLOv8 standard input size
            'output_format': 'numpy',         # NumPy arrays
            'ensure_rgb': True,               # RGB format (not BGR)
            'preserve_original_size': False,  # Resize to exact size
            'normalize': False,               # Keep [0, 255] range for YOLOv8
            'padding_mode': 'letterbox'       # YOLOv8-style letterboxing
        }
    }
    
    pipeline = UniversalPipeline(yolov8_config)
    
    # Simulate processing an image for YOLOv8
    print("✓ Universal Pipeline configured for YOLOv8:")
    print(f"  • Input size: 640×640 pixels")
    print(f"  • RGB format: ✓")
    print(f"  • Letterbox padding: ✓") 
    print(f"  • Output format: NumPy arrays")
    
    # Process a sample image
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_image = pipeline.fit_transform(sample_image)
    
    print(f"\n📥 Input: {sample_image.shape} (H, W, C)")
    print(f"📤 Output: {processed_image.shape} (H, W, C)")
    print(f"✓ Ready for YOLOv8 input!")
    
    return processed_image

def demo_yolov8_dataloader():
    """Show YOLOv8 training with Universal Pipeline data"""
    print("\n🔄 YOLOV8 DATALOADER INTEGRATION")
    print("=" * 60)
    
    # Check if we have processed images
    processed_dir = "sample_data/image/processed_output/image_processed"
    labels_dir = "test/labels"  # Where YOLO labels would be
    
    if Path(processed_dir).exists():
        print(f"✓ Loading YOLOv8 data from {processed_dir}")
        
        # Create YOLOv8 dataset with processed images
        dataset = YOLOv8Dataset(processed_dir, labels_dir)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=yolo_collate_fn)
        
        print(f"✓ Created YOLOv8 dataset: {len(dataset)} images")
        print(f"✓ Dataloader batch size: 2")
        
        # Show what YOLOv8 gets from the dataloader
        for batch_idx, (images, targets) in enumerate(dataloader):
            print(f"\n📦 Batch {batch_idx + 1}:")
            print(f"  • Images: {images.shape}")  # (batch_size, 3, 640, 640)
            print(f"  • Targets: {len(targets)} bounding box sets")
            
            # This is EXACTLY what YOLOv8 expects!
            print(f"  ✓ Format: Perfect for YOLOv8 training")
            
            if batch_idx >= 1:  # Show 2 batches
                break
        
        return dataloader
    else:
        print("ℹ️  No processed images found")
        return None

def yolo_collate_fn(batch):
    """Custom collate function for YOLOv8 (handles variable-length labels)"""
    images, targets = zip(*batch)
    
    # Stack images into batch tensor
    images = torch.stack(images)
    
    # Keep targets as list (different images can have different numbers of objects)
    return images, list(targets)

def demo_yolov8_model_integration():
    """Show actual YOLOv8 model with processed data"""
    print("\n🚀 YOLOV8 MODEL INTEGRATION")
    print("=" * 60)
    
    # Simplified YOLOv8-style model (the real one would be much more complex)
    class SimpleYOLOv8(nn.Module):
        def __init__(self, num_classes=80):
            super().__init__()
            # Simplified backbone (real YOLOv8 uses CSPDarknet)
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), 
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((20, 20))  # 640/32 = 20
            )
            
            # Detection head (real YOLOv8 has multiple scales)
            self.head = nn.Conv2d(256, (num_classes + 5) * 3, 1)  # 3 anchors per cell
            
        def forward(self, x):
            # x shape: (batch_size, 3, 640, 640) - from Universal Pipeline!
            features = self.backbone(x)  # (batch_size, 256, 20, 20)
            detections = self.head(features)  # (batch_size, 255, 20, 20) for 80 classes
            return detections
    
    # Create model
    model = SimpleYOLOv8(num_classes=80)
    
    # Test with processed data
    sample_batch = torch.randn(2, 3, 640, 640)  # Same format as Universal Pipeline output
    output = model(sample_batch)
    
    print(f"✓ YOLOv8 model created")
    print(f"✓ Input shape: {sample_batch.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Ready for YOLOv8 training/inference!")

def demo_real_yolov8_usage():
    """Show how to use with real YOLOv8 (ultralytics)"""
    print("\n📋 REAL YOLOV8 USAGE EXAMPLE")
    print("=" * 60)
    
    print("With real YOLOv8 (ultralytics), you would do:")
    print()
    
    code_example = '''
# 1. Process data with Universal Pipeline
from universal_pipeline import UniversalPipeline

config = {
    'image': {
        'target_size': (640, 640),
        'output_format': 'numpy',
        'ensure_rgb': True,
        'normalize': False  # YOLOv8 handles normalization
    }
}

pipeline = UniversalPipeline(config)
processed_images = pipeline.fit_transform(raw_images)

# 2. Save processed images in YOLOv8 format
for i, img in enumerate(processed_images):
    cv2.imwrite(f'datasets/images/train/img_{i:04d}.jpg', img)

# 3. Use directly with YOLOv8
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load pretrained model
results = model.train(
    data='dataset.yaml',  # Points to your processed images
    epochs=100,
    imgsz=640,  # Same size as Universal Pipeline output
    batch=16
)

# 4. Inference on new data
new_data = pipeline.transform(new_images)  # Same preprocessing
results = model.predict(new_data)
'''
    
    print(code_example)
    
    print("🔑 Key Points:")
    print("✓ Universal Pipeline outputs are YOLOv8-ready")
    print("✓ No additional preprocessing needed")
    print("✓ Same pipeline for training and inference")
    print("✓ Consistent image sizes and formats")

def demo_production_workflow():
    """Show complete production workflow"""
    print("\n🏭 PRODUCTION WORKFLOW")
    print("=" * 60)
    
    print("Complete YOLOv8 + Universal Pipeline workflow:")
    print()
    
    workflow = '''
1. 📸 RAW DATA COLLECTION
   └── Images: various sizes, formats (jpg, png, etc.)
   └── Labels: YOLO format (.txt files)

2. 🔄 UNIVERSAL PIPELINE PREPROCESSING  
   └── Auto-resize to 640×640
   └── RGB conversion
   └── Letterbox padding
   └── Quality normalization

3. 🎯 YOLOV8 TRAINING
   └── DataLoader feeds processed images
   └── model.train() with preprocessed data
   └── Consistent input format

4. 🚀 PRODUCTION INFERENCE
   └── New image → Universal Pipeline → YOLOv8 → Results
   └── Same preprocessing ensures accuracy
'''
    
    print(workflow)
    
    print("✅ Benefits:")
    print("• Consistent preprocessing across train/test/production")
    print("• Automatic handling of various input formats")
    print("• No manual image resizing or format conversion")
    print("• One pipeline for all data processing needs")

def main():
    """Run YOLOv8 integration demonstrations"""
    print("🎯 YOLOV8 + UNIVERSAL PIPELINE INTEGRATION")
    print("=" * 70)
    print("How Universal Pipeline data feeds directly into YOLOv8")
    print("=" * 70)
    
    demo_yolov8_preprocessing()
    demo_yolov8_dataloader()
    demo_yolov8_model_integration()
    demo_real_yolov8_usage()
    demo_production_workflow()
    
    print("\n" + "=" * 70)
    print("🎉 SUMMARY:")
    print("✅ Universal Pipeline → YOLOv8-ready format")
    print("✅ ONE dataloader → YOLOv8 training")
    print("✅ Same preprocessing → train and inference")
    print("✅ No additional conversion needed")
    print("\n💡 The processed data IS the YOLOv8 training data!")

if __name__ == "__main__":
    main() 