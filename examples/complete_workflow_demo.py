#!/usr/bin/env python3
"""
Complete Workflow Demo
Shows the COMPLETE process after UniversalModelFactory.create_pipeline('model_name')
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
import cv2

from comprehensive_model_configs import UniversalModelFactory

def step1_create_pipeline():
    """STEP 1: Create pipeline for your model"""
    print("🔧 STEP 1: CREATE PIPELINE")
    print("=" * 60)
    
    # Choose your model
    model_name = 'yolov8'  # Could be any of the 100+ supported models
    
    print(f"Creating pipeline for: {model_name}")
    
    # THIS IS WHERE YOU LEFT OFF!
    pipeline = UniversalModelFactory.create_pipeline(model_name)
    
    print(f"✅ Pipeline created for {model_name}")
    print(f"📋 Pipeline configured for: {UniversalModelFactory.get_model_info(model_name)}")
    
    return pipeline

def step2_process_single_image(pipeline):
    """STEP 2: Process a single image"""
    print("\n📸 STEP 2: PROCESS SINGLE IMAGE")
    print("=" * 60)
    
    # Load or create an image
    raw_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"Raw image shape: {raw_image.shape}")
    
    # THIS IS WHAT HAPPENS NEXT!
    processed_image = pipeline.fit_transform(raw_image)
    
    print(f"✅ Processed image shape: {processed_image.shape}")
    print(f"✅ Ready for model input!")
    
    return processed_image

def step3_create_dataset(pipeline):
    """STEP 3: Create dataset for training"""
    print("\n📦 STEP 3: CREATE DATASET")
    print("=" * 60)
    
    # Simulate multiple images
    raw_images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
        for _ in range(10)
    ]
    
    print(f"Processing {len(raw_images)} images...")
    
    # Process all images through the pipeline
    processed_images = []
    for img in raw_images:
        processed = pipeline.transform(img)  # Use transform for subsequent images
        processed_images.append(processed)
    
    # Convert to tensors for PyTorch
    tensors = []
    for img in processed_images:
        if len(img.shape) == 3:  # (H, W, C)
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C, H, W)
            tensors.append(tensor)
    
    # Stack into batch
    batch_tensor = torch.stack(tensors)
    
    print(f"✅ Dataset created: {batch_tensor.shape}")
    print(f"✅ Ready for DataLoader!")
    
    return batch_tensor

def step4_create_dataloader(batch_tensor):
    """STEP 4: Create DataLoader"""
    print("\n🔄 STEP 4: CREATE DATALOADER")
    print("=" * 60)
    
    # Create labels (mock for demo)
    labels = torch.randint(0, 80, (batch_tensor.shape[0],))  # 80 classes for COCO
    
    # Create dataset
    dataset = TensorDataset(batch_tensor, labels)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"✅ DataLoader created with batch_size=4")
    print(f"✅ Dataset size: {len(dataset)} images")
    
    # Test the dataloader
    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"   Batch {batch_idx + 1}: Images {images.shape}, Labels {targets.shape}")
        if batch_idx >= 2:  # Show first 3 batches
            break
    
    return dataloader

def step5_train_model(dataloader):
    """STEP 5: Train your model"""
    print("\n🏋️ STEP 5: TRAIN MODEL")
    print("=" * 60)
    
    # Create a simple model (this could be YOLOv8, ResNet, etc.)
    class SimpleYOLO(nn.Module):
        def __init__(self, num_classes=80):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(128, num_classes)
            
        def forward(self, x):
            x = self.backbone(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    # Initialize model
    model = SimpleYOLO(num_classes=80)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("🔧 Model initialized")
    print("🔧 Loss function: CrossEntropyLoss")
    print("🔧 Optimizer: Adam")
    
    # Training loop
    model.train()
    for epoch in range(3):  # Train for 3 epochs
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch + 1}/3 - Average Loss: {avg_loss:.4f}")
    
    print("🎉 Model training completed!")
    return model

def step6_inference(model, pipeline):
    """STEP 6: Use trained model for inference"""
    print("\n🔮 STEP 6: INFERENCE")
    print("=" * 60)
    
    # New image for inference
    new_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    print(f"New image shape: {new_image.shape}")
    
    # Process through the SAME pipeline
    processed = pipeline.transform(new_image)
    
    # Convert to tensor
    tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Run inference
    model.eval()
    with torch.no_grad():
        prediction = model(tensor)
        predicted_class = torch.argmax(prediction, dim=1)
    
    print(f"✅ Processed for inference: {tensor.shape}")
    print(f"✅ Prediction shape: {prediction.shape}")
    print(f"✅ Predicted class: {predicted_class.item()}")
    
    return prediction

def step7_production_deployment():
    """STEP 7: Production deployment"""
    print("\n🚀 STEP 7: PRODUCTION DEPLOYMENT")
    print("=" * 60)
    
    production_code = '''
# Save your trained model and pipeline
torch.save(model.state_dict(), 'trained_yolov8.pth')
pickle.dump(pipeline, open('yolov8_pipeline.pkl', 'wb'))

# Production inference service
class InferenceService:
    def __init__(self, model_path, pipeline_path):
        # Load trained model
        self.model = SimpleYOLO(num_classes=80)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Load preprocessing pipeline
        self.pipeline = pickle.load(open(pipeline_path, 'rb'))
    
    def predict(self, raw_image):
        # Preprocess with SAME pipeline used in training
        processed = self.pipeline.transform(raw_image)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(tensor)
            return prediction
    
    def batch_predict(self, raw_images):
        results = []
        for img in raw_images:
            result = self.predict(img)
            results.append(result)
        return results

# Usage in production
service = InferenceService('trained_yolov8.pth', 'yolov8_pipeline.pkl')

# Single image prediction
result = service.predict(new_image)

# Batch prediction
batch_results = service.batch_predict([img1, img2, img3])
'''
    
    print(production_code)

def demo_multiple_models():
    """Demo using multiple models simultaneously"""
    print("\n🌐 BONUS: MULTIPLE MODELS")
    print("=" * 60)
    
    models_to_test = ['yolov8', 'resnet50', 'efficientnet_b4', 'facenet']
    
    # Create pipelines for different models
    pipelines = {}
    for model_name in models_to_test:
        pipelines[model_name] = UniversalModelFactory.create_pipeline(model_name)
    
    print(f"✅ Created {len(pipelines)} different pipelines")
    
    # Process same image with all pipelines
    raw_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    results = {}
    for model_name, pipeline in pipelines.items():
        processed = pipeline.fit_transform(raw_image)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        results[model_name] = tensor.shape
        print(f"  ✅ {model_name:<15} → {tensor.shape}")
    
    print("🎉 Same image processed for multiple models!")

def complete_workflow_summary():
    """Summary of the complete workflow"""
    print("\n📋 COMPLETE WORKFLOW SUMMARY")
    print("=" * 60)
    
    workflow = '''
STEP 1: Create Pipeline
    pipeline = UniversalModelFactory.create_pipeline('yolov8')

STEP 2: Process Images
    processed_image = pipeline.fit_transform(raw_image)      # First image
    processed_image = pipeline.transform(new_raw_image)     # Subsequent images

STEP 3: Create Dataset
    tensors = [torch.from_numpy(img).permute(2,0,1) for img in processed_images]
    batch_tensor = torch.stack(tensors)

STEP 4: Create DataLoader
    dataset = TensorDataset(batch_tensor, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

STEP 5: Train Model
    for images, labels in dataloader:
        outputs = model(images)        # Images are already processed!
        loss = criterion(outputs, labels)
        # ... training loop

STEP 6: Inference
    new_processed = pipeline.transform(new_raw_image)
    tensor = torch.from_numpy(new_processed).permute(2,0,1).unsqueeze(0)
    prediction = model(tensor)

STEP 7: Production
    # Save model + pipeline
    # Load in production service
    # Use SAME pipeline for consistent preprocessing
'''
    
    print(workflow)

def main():
    """Run complete workflow demonstration"""
    print("🎯 COMPLETE WORKFLOW: FROM PIPELINE TO PRODUCTION")
    print("=" * 80)
    print("What happens AFTER UniversalModelFactory.create_pipeline('model_name')")
    print("=" * 80)
    
    # Run the complete workflow
    pipeline = step1_create_pipeline()
    processed = step2_process_single_image(pipeline)
    batch_tensor = step3_create_dataset(pipeline)
    dataloader = step4_create_dataloader(batch_tensor)
    model = step5_train_model(dataloader)
    prediction = step6_inference(model, pipeline)
    step7_production_deployment()
    demo_multiple_models()
    complete_workflow_summary()
    
    print("\n" + "=" * 80)
    print("🎉 COMPLETE WORKFLOW DEMONSTRATED!")
    print("=" * 80)
    print("✅ Pipeline Creation → Data Processing → Dataset → DataLoader")
    print("✅ Model Training → Inference → Production Deployment")
    print("✅ Same pipeline for training AND inference")
    print("✅ Works with ANY of the 100+ supported models")
    print("\n💡 The pipeline handles ALL the preprocessing - you just train and infer!")

if __name__ == "__main__":
    main() 