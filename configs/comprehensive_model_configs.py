#!/usr/bin/env python3
"""
Comprehensive Model Configurations
ALL popular image models, OCR, segmentation, detection, and more
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from pathlib import Path
from universal_pipeline import UniversalPipeline

# ================================================================================================
# OBJECT DETECTION MODELS
# ================================================================================================
OBJECT_DETECTION_CONFIGS = {
    # YOLO Family
    'yolov1': {'image': {'target_size': (448, 448), 'output_format': 'numpy', 'ensure_rgb': True}},
    'yolov2': {'image': {'target_size': (416, 416), 'output_format': 'numpy', 'ensure_rgb': True, 'padding_mode': 'letterbox'}},
    'yolov3': {'image': {'target_size': (416, 416), 'output_format': 'numpy', 'ensure_rgb': True, 'padding_mode': 'letterbox'}},
    'yolov4': {'image': {'target_size': (608, 608), 'output_format': 'numpy', 'ensure_rgb': True, 'padding_mode': 'letterbox'}},
    'yolov5': {'image': {'target_size': (640, 640), 'output_format': 'numpy', 'ensure_rgb': True, 'padding_mode': 'letterbox'}},
    'yolov6': {'image': {'target_size': (640, 640), 'output_format': 'numpy', 'ensure_rgb': True, 'padding_mode': 'letterbox'}},
    'yolov7': {'image': {'target_size': (640, 640), 'output_format': 'numpy', 'ensure_rgb': True, 'padding_mode': 'letterbox'}},
    'yolov8': {'image': {'target_size': (640, 640), 'output_format': 'numpy', 'ensure_rgb': True, 'padding_mode': 'letterbox'}},
    'yolov9': {'image': {'target_size': (640, 640), 'output_format': 'numpy', 'ensure_rgb': True, 'padding_mode': 'letterbox'}},
    'yolov10': {'image': {'target_size': (640, 640), 'output_format': 'numpy', 'ensure_rgb': True, 'padding_mode': 'letterbox'}},
    'yolov11': {'image': {'target_size': (640, 640), 'output_format': 'numpy', 'ensure_rgb': True, 'padding_mode': 'letterbox'}},
    
    # R-CNN Family
    'rcnn': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'fast_rcnn': {'image': {'target_size': (600, 1000), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'faster_rcnn': {'image': {'target_size': (800, 1333), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'mask_rcnn': {'image': {'target_size': (800, 1333), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # Single Shot Detectors
    'ssd300': {'image': {'target_size': (300, 300), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'ssd512': {'image': {'target_size': (512, 512), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'retinanet': {'image': {'target_size': (800, 1333), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # Modern Detectors
    'detr': {'image': {'target_size': (800, 1333), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'fcos': {'image': {'target_size': (800, 1333), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'centernet': {'image': {'target_size': (512, 512), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
}

# ================================================================================================
# CLASSIFICATION MODELS
# ================================================================================================
CLASSIFICATION_CONFIGS = {
    # ResNet Family
    'resnet18': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
    'resnet34': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
    'resnet50': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
    'resnet101': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
    'resnet152': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
    'resnext50': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
    'resnext101': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}},
    
    # EfficientNet Family
    'efficientnet_b0': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'efficientnet_b1': {'image': {'target_size': (240, 240), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'efficientnet_b2': {'image': {'target_size': (260, 260), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'efficientnet_b3': {'image': {'target_size': (300, 300), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'efficientnet_b4': {'image': {'target_size': (380, 380), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'efficientnet_b5': {'image': {'target_size': (456, 456), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'efficientnet_b6': {'image': {'target_size': (528, 528), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'efficientnet_b7': {'image': {'target_size': (600, 600), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # DenseNet Family
    'densenet121': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'densenet169': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'densenet201': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # MobileNet Family
    'mobilenet_v1': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'mobilenet_v2': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'mobilenet_v3_small': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'mobilenet_v3_large': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # Inception Family
    'inception_v3': {'image': {'target_size': (299, 299), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'inception_v4': {'image': {'target_size': (299, 299), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'inception_resnet_v2': {'image': {'target_size': (299, 299), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # VGG Family
    'vgg16': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'vgg19': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
}

# ================================================================================================
# VISION TRANSFORMERS
# ================================================================================================
VISION_TRANSFORMER_CONFIGS = {
    # Vision Transformer Family
    'vit_tiny_patch16_224': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'vit_small_patch16_224': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'vit_base_patch16_224': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'vit_large_patch16_224': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'vit_huge_patch14_224': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # SWIN Transformer Family
    'swin_tiny_patch4_window7_224': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'swin_small_patch4_window7_224': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'swin_base_patch4_window7_224': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'swin_large_patch4_window7_224': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # DEIT Family
    'deit_tiny_patch16_224': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'deit_small_patch16_224': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'deit_base_patch16_224': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
}

# ================================================================================================
# OCR MODELS
# ================================================================================================
OCR_CONFIGS = {
    # Text Detection
    'east': {'image': {'target_size': (512, 512), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'craft': {'image': {'target_size': (1280, 1280), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'textboxes': {'image': {'target_size': (512, 512), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'db_resnet50': {'image': {'target_size': (736, 1280), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # Text Recognition
    'crnn': {'image': {'target_size': (32, 128), 'output_format': 'numpy', 'ensure_rgb': False, 'normalize': True}},  # Grayscale
    'rare': {'image': {'target_size': (32, 100), 'output_format': 'numpy', 'ensure_rgb': False, 'normalize': True}},
    'star_net': {'image': {'target_size': (32, 128), 'output_format': 'numpy', 'ensure_rgb': False, 'normalize': True}},
    'rosetta': {'image': {'target_size': (32, 128), 'output_format': 'numpy', 'ensure_rgb': False, 'normalize': True}},
    
    # End-to-End OCR
    'paddleocr': {'image': {'target_size': (960, 960), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'easyocr': {'image': {'target_size': (640, 640), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'tesseract': {'image': {'target_size': (None, None), 'output_format': 'numpy', 'ensure_rgb': False, 'preserve_original_size': True}},
    'trocr': {'image': {'target_size': (384, 384), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
}

# ================================================================================================
# SEGMENTATION MODELS
# ================================================================================================
SEGMENTATION_CONFIGS = {
    # U-Net Family
    'unet': {'image': {'target_size': (256, 256), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'unet_plus_plus': {'image': {'target_size': (256, 256), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'attention_unet': {'image': {'target_size': (256, 256), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # DeepLab Family
    'deeplabv3': {'image': {'target_size': (513, 513), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'deeplabv3_plus': {'image': {'target_size': (513, 513), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # PSP Family
    'pspnet': {'image': {'target_size': (473, 473), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # Segformer
    'segformer_b0': {'image': {'target_size': (512, 512), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'segformer_b5': {'image': {'target_size': (640, 640), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
}

# ================================================================================================
# FACE RECOGNITION & ANALYSIS
# ================================================================================================
FACE_CONFIGS = {
    # Face Detection
    'mtcnn': {'image': {'target_size': (160, 160), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'retinaface': {'image': {'target_size': (640, 640), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'blazeface': {'image': {'target_size': (128, 128), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # Face Recognition
    'facenet': {'image': {'target_size': (160, 160), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'arcface': {'image': {'target_size': (112, 112), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'sphereface': {'image': {'target_size': (112, 112), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'cosface': {'image': {'target_size': (112, 112), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # Face Analysis
    'age_gender_prediction': {'image': {'target_size': (224, 224), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'emotion_recognition': {'image': {'target_size': (48, 48), 'output_format': 'numpy', 'ensure_rgb': False, 'normalize': True}},
}

# ================================================================================================
# POSE ESTIMATION
# ================================================================================================
POSE_CONFIGS = {
    'openpose': {'image': {'target_size': (368, 368), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'posenet': {'image': {'target_size': (257, 257), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'hrnet': {'image': {'target_size': (256, 192), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'alphapose': {'image': {'target_size': (256, 192), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
}

# ================================================================================================
# GENERATIVE MODELS
# ================================================================================================
GENERATIVE_CONFIGS = {
    # GANs
    'stylegan2': {'image': {'target_size': (1024, 1024), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'stylegan3': {'image': {'target_size': (1024, 1024), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'cyclegan': {'image': {'target_size': (256, 256), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'pix2pix': {'image': {'target_size': (256, 256), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    
    # Diffusion Models
    'stable_diffusion': {'image': {'target_size': (512, 512), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'dalle2': {'image': {'target_size': (1024, 1024), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
    'midjourney': {'image': {'target_size': (1024, 1024), 'output_format': 'numpy', 'ensure_rgb': True, 'normalize': True}},
}

# ================================================================================================
# MASTER CONFIGURATION DICTIONARY
# ================================================================================================
ALL_MODEL_CONFIGS = {
    **OBJECT_DETECTION_CONFIGS,
    **CLASSIFICATION_CONFIGS,
    **VISION_TRANSFORMER_CONFIGS,
    **OCR_CONFIGS,
    **SEGMENTATION_CONFIGS,
    **FACE_CONFIGS,
    **POSE_CONFIGS,
    **GENERATIVE_CONFIGS
}

# ================================================================================================
# UNIVERSAL MODEL FACTORY
# ================================================================================================
class UniversalModelFactory:
    """Factory for creating dataloaders for ANY computer vision model"""
    
    @staticmethod
    def get_all_supported_models():
        """Get list of all supported models"""
        return list(ALL_MODEL_CONFIGS.keys())
    
    @staticmethod
    def get_models_by_category():
        """Get models organized by category"""
        return {
            'Object Detection': list(OBJECT_DETECTION_CONFIGS.keys()),
            'Classification': list(CLASSIFICATION_CONFIGS.keys()),
            'Vision Transformers': list(VISION_TRANSFORMER_CONFIGS.keys()),
            'OCR': list(OCR_CONFIGS.keys()),
            'Segmentation': list(SEGMENTATION_CONFIGS.keys()),
            'Face Recognition': list(FACE_CONFIGS.keys()),
            'Pose Estimation': list(POSE_CONFIGS.keys()),
            'Generative Models': list(GENERATIVE_CONFIGS.keys())
        }
    
    @staticmethod
    def get_config(model_name):
        """Get configuration for specific model"""
        return ALL_MODEL_CONFIGS.get(model_name.lower())
    
    @staticmethod
    def create_pipeline(model_name):
        """Create Universal Pipeline for specific model"""
        config = UniversalModelFactory.get_config(model_name)
        if config is None:
            raise ValueError(f"Model '{model_name}' not supported. Use get_all_supported_models() to see available models.")
        
        return UniversalPipeline(config)
    
    @staticmethod
    def get_model_info(model_name):
        """Get detailed info about a model"""
        config = UniversalModelFactory.get_config(model_name)
        if config is None:
            return None
            
        img_config = config['image']
        return {
            'model_name': model_name,
            'input_size': img_config.get('target_size'),
            'channels': 3 if img_config.get('ensure_rgb', True) else 1,
            'normalization': img_config.get('normalize', False),
            'output_format': img_config.get('output_format', 'numpy')
        }

def demo_comprehensive_support():
    """Demonstrate comprehensive model support"""
    print("🌍 COMPREHENSIVE MODEL SUPPORT")
    print("=" * 80)
    
    categories = UniversalModelFactory.get_models_by_category()
    
    total_models = 0
    for category, models in categories.items():
        print(f"\n📂 {category} ({len(models)} models):")
        for i, model in enumerate(models[:5], 1):  # Show first 5 of each category
            info = UniversalModelFactory.get_model_info(model)
            size = info['input_size']
            channels = info['channels']
            print(f"  {i:2d}. ✅ {model:<25} → {size} ({channels}ch)")
        
        if len(models) > 5:
            print(f"      ... and {len(models) - 5} more models")
        
        total_models += len(models)
    
    print(f"\n🎉 TOTAL SUPPORTED MODELS: {total_models}")
    print("💡 ONE Universal Pipeline → ALL these models!")

def demo_model_usage_examples():
    """Show usage examples for different model types"""
    print("\n🚀 USAGE EXAMPLES")
    print("=" * 80)
    
    examples = [
        ('yolov8', 'Object Detection'),
        ('efficientnet_b4', 'Image Classification'),
        ('vit_base_patch16_224', 'Vision Transformer'),
        ('paddleocr', 'OCR Text Detection'),
        ('unet', 'Semantic Segmentation'),
        ('facenet', 'Face Recognition'),
        ('openpose', 'Pose Estimation'),
        ('stable_diffusion', 'Generative AI')
    ]
    
    for model_name, category in examples:
        print(f"\n📝 {category} - {model_name.upper()}:")
        
        try:
            pipeline = UniversalModelFactory.create_pipeline(model_name)
            info = UniversalModelFactory.get_model_info(model_name)
            
            print(f"  ✓ Input size: {info['input_size']}")
            print(f"  ✓ Channels: {info['channels']}")
            print(f"  ✓ Ready for {category} tasks")
            
            # Example usage code
            print(f"  💻 Usage:")
            print(f"     pipeline = UniversalModelFactory.create_pipeline('{model_name}')")
            print(f"     processed_data = pipeline.fit_transform(images)")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def main():
    """Main demonstration"""
    print("🎯 COMPREHENSIVE COMPUTER VISION MODEL CONFIGURATIONS")
    print("=" * 80)
    print("Supporting ALL popular CV models with Universal Pipeline")
    print("=" * 80)
    
    demo_comprehensive_support()
    demo_model_usage_examples()
    
    print("\n" + "=" * 80)
    print("🎉 UNIVERSAL COMPUTER VISION PIPELINE")
    print("=" * 80)
    print("✅ Object Detection: YOLO v1-v11, R-CNN, SSD, RetinaNet, DETR")
    print("✅ Classification: ResNet, EfficientNet, DenseNet, MobileNet, VGG")
    print("✅ Vision Transformers: ViT, SWIN, DEIT")
    print("✅ OCR: PaddleOCR, EasyOCR, CRNN, EAST, CRAFT, TrOCR")
    print("✅ Segmentation: U-Net, DeepLab, PSPNet, SegFormer")
    print("✅ Face Recognition: FaceNet, ArcFace, MTCNN, RetinaFace")
    print("✅ Pose Estimation: OpenPose, PoseNet, HRNet, AlphaPose")
    print("✅ Generative AI: StyleGAN, Stable Diffusion, DALL-E")
    print("\n💡 ONE Pipeline → EVERY Computer Vision Task!")

if __name__ == "__main__":
    main() 