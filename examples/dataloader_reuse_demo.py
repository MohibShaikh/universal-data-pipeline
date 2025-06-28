#!/usr/bin/env python3
"""
DataLoader Reuse Demo
Shows how ONE dataloader can feed multiple different models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from universal_pipeline import UniversalPipeline

class UniversalImageDataset(Dataset):
    """ONE dataset class that works with ALL image models"""
    def __init__(self, processed_images_dir, labels=None):
        self.image_files = list(Path(processed_images_dir).glob("*.npy"))
        self.labels = labels if labels is not None else [0] * len(self.image_files)  # Mock labels
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load processed image (already standardized by Universal Pipeline)
        image = np.load(self.image_files[idx])
        
        # Convert to PyTorch tensor - THIS SAME FORMAT WORKS FOR ALL MODELS
        if len(image.shape) == 3:  # (H, W, C)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (C, H, W)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Multiple different models that ALL use the SAME dataloader
class YOLOStyleModel(nn.Module):
    """YOLO-style object detection model"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.head = nn.Linear(64 * 7 * 7, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

class ResNetStyleModel(nn.Module):
    """ResNet-style classification model"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class VisionTransformerStyle(nn.Module):
    """Vision Transformer-style model"""
    def __init__(self, num_classes=10):
        super().__init__()
        # Simplified ViT-like model
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)  # 16x16 patches
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True),
            num_layers=2
        )
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        # Convert to patches
        x = self.patch_embed(x)  # (B, 768, H/16, W/16)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).transpose(1, 2)  # (B, patches, 768)
        
        # Add simple positional encoding
        x = x + torch.randn_like(x) * 0.01
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling and classify
        x = x.mean(dim=1)  # (B, 768)
        return self.classifier(x)

def demo_one_dataloader_multiple_models():
    """Show how ONE dataloader feeds multiple different models"""
    print("🔄 ONE DATALOADER → MULTIPLE MODELS")
    print("=" * 60)
    
    # 1. Create ONE dataset from processed images
    processed_dir = "sample_data/image/processed_output/image_processed"
    
    if Path(processed_dir).exists():
        print(f"✓ Loading processed images from {processed_dir}")
        
        # Create ONE dataset that works for ALL models
        dataset = UniversalImageDataset(processed_dir)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        print(f"✓ Created ONE dataloader with {len(dataset)} images")
        
        # 2. Create multiple different models
        yolo_model = YOLOStyleModel(num_classes=10)
        resnet_model = ResNetStyleModel(num_classes=10)
        vit_model = VisionTransformerStyle(num_classes=10)
        
        print("✓ Created 3 different model architectures:")
        print("  • YOLO-style object detection")
        print("  • ResNet-style CNN")
        print("  • Vision Transformer")
        
        # 3. The SAME dataloader feeds ALL models!
        print("\n🚀 Testing with the SAME batch from the SAME dataloader:")
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f"\n📦 Batch {batch_idx + 1}: {images.shape}")
            
            # SAME data goes into ALL models
            yolo_output = yolo_model(images)
            resnet_output = resnet_model(images)
            vit_output = vit_model(images)
            
            print(f"  ✓ YOLO output: {yolo_output.shape}")
            print(f"  ✓ ResNet output: {resnet_output.shape}")
            print(f"  ✓ ViT output: {vit_output.shape}")
            
            if batch_idx >= 1:  # Just show 2 batches
                break
        
        print("\n🎯 KEY INSIGHT:")
        print("✓ Universal Pipeline → Standardized format → ONE dataloader")
        print("✓ ONE dataloader → Multiple different models")
        print("✓ No need to create separate dataloaders per model!")
        
    else:
        print("ℹ️  No processed images found - run the CLI processor first")

def demo_different_model_requirements():
    """Show how to handle models with different input requirements"""
    print("\n🔧 HANDLING DIFFERENT MODEL REQUIREMENTS")
    print("=" * 60)
    
    print("Sometimes models need SLIGHTLY different preprocessing:")
    print()
    
    # Different configs for different model families
    configs = {
        'yolo': {
            'image': {
                'target_size': (640, 640),  # YOLO prefers 640x640
                'output_format': 'numpy',
                'normalize': True
            }
        },
        'vit': {
            'image': {
                'target_size': (224, 224),  # ViT prefers 224x224
                'output_format': 'numpy', 
                'normalize': True
            }
        },
        'custom': {
            'image': {
                'target_size': (512, 512),  # Custom size
                'output_format': 'numpy',
                'normalize': True
            }
        }
    }
    
    print("📋 Different configs for different model families:")
    for model_type, config in configs.items():
        size = config['image']['target_size']
        print(f"  • {model_type.upper()}: {size[0]}x{size[1]} images")
    
    print("\n💡 Solution: Use Universal Pipeline with model-specific configs")
    print("✓ Still ONE pipeline system")
    print("✓ Just different config parameters")
    print("✓ Same processing logic, different output sizes")

def demo_training_multiple_models():
    """Show training multiple models with the same processed data"""
    print("\n🏋️ TRAINING MULTIPLE MODELS")
    print("=" * 60)
    
    processed_dir = "sample_data/image/processed_output/image_processed"
    
    if Path(processed_dir).exists():
        # Create dataset and dataloader ONCE
        dataset = UniversalImageDataset(processed_dir)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        print(f"✓ Created training dataloader: {len(dataset)} images")
        
        # Create multiple models
        models = {
            'YOLO': YOLOStyleModel(),
            'ResNet': ResNetStyleModel(),
            'ViT': VisionTransformerStyle()
        }
        
        optimizers = {
            name: torch.optim.Adam(model.parameters(), lr=0.001)
            for name, model in models.items()
        }
        
        criterion = nn.CrossEntropyLoss()
        
        print("\n🔄 Training ALL models with the SAME dataloader:")
        
        # Train all models for 1 epoch
        for name, model in models.items():
            model.train()
            total_loss = 0
            optimizer = optimizers[name]
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                if batch_idx >= 2:  # Just train on a few batches for demo
                    break
            
            avg_loss = total_loss / min(3, len(train_loader))
            print(f"  ✓ {name} trained - Avg Loss: {avg_loss:.4f}")
        
        print("\n🎉 All models trained with the SAME processed data!")
    else:
        print("ℹ️  No processed images found")

def main():
    """Run dataloader reuse demonstrations"""
    print("🚀 DATALOADER REUSE WITH UNIVERSAL PIPELINE")
    print("=" * 70)
    print("Answer: NO, you don't need separate dataloaders per model!")
    print("=" * 70)
    
    demo_one_dataloader_multiple_models()
    demo_different_model_requirements()
    demo_training_multiple_models()
    
    print("\n" + "=" * 70)
    print("🎯 SUMMARY:")
    print("✅ Universal Pipeline → Standardized format")
    print("✅ ONE dataset class → Multiple models")
    print("✅ ONE dataloader → All model architectures")
    print("✅ Same preprocessing → Different model families")
    print("\n💡 Only create new dataloaders for:")
    print("  • Different batch sizes")
    print("  • Different data splits (train/val/test)")
    print("  • Different augmentation strategies")
    print("  • NOT for different model architectures!")

if __name__ == "__main__":
    main() 