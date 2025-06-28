#!/usr/bin/env python3
"""
Comprehensive Usage Demo
Practical examples of using Universal Pipeline with all supported models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Import our comprehensive configs
from comprehensive_model_configs import UniversalModelFactory, ALL_MODEL_CONFIGS

def demo_multi_task_pipeline():
    """Demo processing same data for multiple different tasks"""
    print("🔄 MULTI-TASK PROCESSING")
    print("=" * 70)
    print("Same image → Multiple AI tasks simultaneously")
    print()
    
    # Simulate processing the same image for different tasks
    sample_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    tasks = {
        'Object Detection': 'yolov8',
        'Classification': 'efficientnet_b4', 
        'OCR': 'paddleocr',
        'Face Recognition': 'facenet',
        'Segmentation': 'unet',
        'Pose Estimation': 'openpose'
    }
    
    print("📸 Processing same image for multiple tasks:")
    print(f"Original image: {sample_image.shape}")
    print()
    
    results = {}
    for task_name, model_name in tasks.items():
        try:
            pipeline = UniversalModelFactory.create_pipeline(model_name)
            processed = pipeline.fit_transform(sample_image)
            
            # Convert to tensor format
            if len(processed.shape) == 3:
                tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            results[task_name] = tensor.shape
            print(f"✅ {task_name:<20} ({model_name:<15}) → {tensor.shape}")
            
        except Exception as e:
            print(f"❌ {task_name:<20} → Error: {e}")
    
    print(f"\n🎉 Processed 1 image for {len(results)} different AI tasks!")
    print("💡 Each task gets optimally formatted data")

def demo_model_comparison():
    """Demo comparing different models for same task"""
    print("\n🔍 MODEL COMPARISON")
    print("=" * 70)
    print("Comparing different models for same task")
    print()
    
    # Compare different classification models
    classification_models = [
        'resnet50', 'efficientnet_b4', 'vit_base_patch16_224', 
        'densenet121', 'mobilenet_v3_large'
    ]
    
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("📊 Classification Model Comparison:")
    print(f"Input image: {sample_image.shape}")
    print()
    
    for model in classification_models:
        pipeline = UniversalModelFactory.create_pipeline(model)
        processed = pipeline.fit_transform(sample_image)
        
        info = UniversalModelFactory.get_model_info(model)
        input_size = info['input_size']
        
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        print(f"  ✅ {model:<25} → {input_size} → {tensor.shape}")
    
    print("\n💡 All models get their optimal input format!")

def demo_ocr_pipeline():
    """Demo comprehensive OCR processing"""
    print("\n📝 COMPREHENSIVE OCR PIPELINE")
    print("=" * 70)
    
    ocr_models = {
        'Text Detection': ['east', 'craft', 'db_resnet50'],
        'Text Recognition': ['crnn', 'rare', 'star_net'],
        'End-to-End OCR': ['paddleocr', 'easyocr', 'trocr']
    }
    
    # Simulate document image
    document_image = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
    
    print("📄 Processing document image with multiple OCR approaches:")
    print(f"Document size: {document_image.shape}")
    print()
    
    for category, models in ocr_models.items():
        print(f"📂 {category}:")
        
        for model in models:
            try:
                pipeline = UniversalModelFactory.create_pipeline(model)
                processed = pipeline.fit_transform(document_image)
                
                info = UniversalModelFactory.get_model_info(model)
                
                if len(processed.shape) == 3:
                    tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                else:
                    tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float() / 255.0
                
                print(f"  ✅ {model:<15} → {info['input_size']} → {tensor.shape}")
                
            except Exception as e:
                print(f"  ❌ {model:<15} → Error: {str(e)[:50]}...")
        print()

def demo_batch_processing():
    """Demo batch processing with different models"""
    print("📦 BATCH PROCESSING")
    print("=" * 70)
    
    # Simulate batch of images
    batch_images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)
    ]
    
    models_to_test = ['yolov8', 'resnet50', 'facenet', 'unet']
    
    print(f"🔄 Processing batch of {len(batch_images)} images:")
    print()
    
    for model_name in models_to_test:
        pipeline = UniversalModelFactory.create_pipeline(model_name)
        
        batch_tensors = []
        for img in batch_images:
            processed = pipeline.fit_transform(img)
            if len(processed.shape) == 3:
                tensor = torch.from_numpy(processed).permute(2, 0, 1).float() / 255.0
                batch_tensors.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(batch_tensors)
        
        info = UniversalModelFactory.get_model_info(model_name)
        print(f"  ✅ {model_name:<15} → Batch: {batch_tensor.shape}")
    
    print("\n💡 Same batch processing pipeline for all models!")

def demo_real_world_scenarios():
    """Demo real-world usage scenarios"""
    print("\n🌍 REAL-WORLD SCENARIOS")
    print("=" * 70)
    
    scenarios = {
        "🚗 Autonomous Driving": {
            'description': 'Multi-model processing for self-driving cars',
            'models': {
                'Object Detection': 'yolov8',      # Detect cars, pedestrians
                'Segmentation': 'deeplabv3_plus',  # Road segmentation  
                'Classification': 'efficientnet_b5' # Traffic sign classification
            }
        },
        "🏥 Medical Imaging": {
            'description': 'Medical image analysis pipeline',
            'models': {
                'Segmentation': 'unet',           # Organ segmentation
                'Classification': 'densenet201',  # Disease classification
                'Detection': 'faster_rcnn'        # Lesion detection
            }
        },
        "📱 Mobile App": {
            'description': 'Lightweight models for mobile deployment',
            'models': {
                'Object Detection': 'yolov5',
                'Classification': 'mobilenet_v3_small',
                'Face Recognition': 'blazeface'
            }
        },
        "🏭 Industrial Inspection": {
            'description': 'Quality control and defect detection',
            'models': {
                'Defect Detection': 'yolov8',
                'Surface Classification': 'efficientnet_b3',
                'Segmentation': 'unet'
            }
        }
    }
    
    for scenario_name, scenario_info in scenarios.items():
        print(f"\n{scenario_name}")
        print(f"Purpose: {scenario_info['description']}")
        print("Models required:")
        
        for task, model in scenario_info['models'].items():
            try:
                info = UniversalModelFactory.get_model_info(model)
                if info:
                    size = info['input_size']
                    print(f"  ✅ {task:<20} → {model:<20} ({size})")
                else:
                    print(f"  ⚠️  {task:<20} → {model:<20} (config needed)")
            except:
                print(f"  ⚠️  {task:<20} → {model:<20} (config needed)")

def demo_production_code():
    """Show production-ready code examples"""
    print("\n🚀 PRODUCTION-READY CODE")
    print("=" * 70)
    
    production_code = '''
# Production Computer Vision Service
class CVService:
    def __init__(self):
        self.pipelines = {}
        
    def setup_model(self, task_name, model_name):
        """Setup pipeline for specific model"""
        self.pipelines[task_name] = UniversalModelFactory.create_pipeline(model_name)
        
    def process_image(self, image, task_name):
        """Process image for specific task"""
        if task_name not in self.pipelines:
            raise ValueError(f"Task {task_name} not configured")
            
        return self.pipelines[task_name].transform(image)
    
    def batch_process(self, images, task_name, batch_size=32):
        """Batch process images"""
        pipeline = self.pipelines[task_name]
        
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            processed_batch = [pipeline.transform(img) for img in batch]
            results.extend(processed_batch)
            
        return results

# Usage Example
cv_service = CVService()

# Setup different models
cv_service.setup_model('detection', 'yolov8')
cv_service.setup_model('classification', 'efficientnet_b4') 
cv_service.setup_model('ocr', 'paddleocr')
cv_service.setup_model('faces', 'facenet')

# Process images for different tasks
detection_result = cv_service.process_image(image, 'detection')
classification_result = cv_service.process_image(image, 'classification')
ocr_result = cv_service.process_image(image, 'ocr')
face_result = cv_service.process_image(image, 'faces')

# Batch processing
batch_results = cv_service.batch_process(image_list, 'detection', batch_size=16)
'''
    
    print(production_code)

def main():
    """Run comprehensive usage demonstrations"""
    print("🎯 COMPREHENSIVE MODEL USAGE DEMONSTRATIONS")
    print("=" * 70)
    print("Real-world examples with Universal Pipeline")
    print("=" * 70)
    
    demo_multi_task_pipeline()
    demo_model_comparison()
    demo_ocr_pipeline() 
    demo_batch_processing()
    demo_real_world_scenarios()
    demo_production_code()
    
    print("\n" + "=" * 70)
    print("🎉 COMPREHENSIVE COMPUTER VISION PIPELINE")
    print("=" * 70)
    print(f"✅ {len(ALL_MODEL_CONFIGS)} Models Supported")
    print("✅ 8 Major Computer Vision Categories")
    print("✅ Production-Ready Architecture")
    print("✅ Batch Processing Support")
    print("✅ Multi-Task Processing")
    print("✅ Real-World Scenarios Covered")
    print("\n💡 ONE Universal Pipeline → EVERY Computer Vision Need!")

if __name__ == "__main__":
    main() 