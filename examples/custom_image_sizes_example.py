"""
Custom Image Sizes Example
==========================

This example shows how to configure the Universal Data Pipeline
to output custom image sizes for different ML use cases, instead
of the default auto-sizing.

Perfect for:
- YOLO object detection (any input size)
- Custom CNN architectures  
- Specific model requirements
- Mobile/edge deployment optimization
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universal_pipeline import UniversalPipeline


def yolo_training_example():
    """Example: YOLOv8 training with 640x640 input"""
    print("🎯 YOLOv8 Training Example (640x640)")
    print("=" * 45)
    
    # YOLOv8 configuration - letterbox resize to 640x640
    yolo_config = {
        'image': {
            'target_size': 640,  # Square input
            'resize_strategy': 'letterbox',  # Preserve aspect ratio
            'output_format': 'tensor',  # PyTorch tensor
            'channel_order': 'RGB',
            'padding_color': (114, 114, 114),  # YOLO standard gray
            'auto_orient': True  # Handle rotated images
        }
    }
    
    pipeline = UniversalPipeline(config=yolo_config)
    
    # Process sample images
    sample_images = [
        "data/sample_data/image/sample_rgb.jpg",
        "data/sample_data/image/pattern.bmp"
    ]
    
    for image_path in sample_images:
        if Path(image_path).exists():
            result = pipeline.fit_transform(image_path)
            print(f"✅ {Path(image_path).name}")
            print(f"   Output shape: {result.shape}")  # Should be [3, 640, 640]
            print(f"   Data type: {result.dtype}")
            print(f"   Ready for YOLO training: ✅")
            break
    else:
        print("❌ No sample images found")


def mobile_detection_example():
    """Example: Mobile deployment with 416x416 input"""
    print("\n📱 Mobile Detection Example (416x416)")
    print("=" * 42)
    
    # Mobile-optimized configuration
    mobile_config = {
        'image': {
            'target_size': (416, 416),  # Smaller for mobile
            'resize_strategy': 'letterbox',
            'output_format': 'numpy',  # For TensorFlow Lite
            'channel_order': 'RGB'
        }
    }
    
    pipeline = UniversalPipeline(config=mobile_config)
    
    sample_images = [
        "data/sample_data/image/sample_rgb.jpg",
        "data/sample_data/image/pattern.bmp"
    ]
    
    for image_path in sample_images:
        if Path(image_path).exists():
            result = pipeline.fit_transform(image_path)
            print(f"✅ {Path(image_path).name}")
            print(f"   Output shape: {result.shape}")  # Should be (416, 416, 3)
            print(f"   Memory usage: {result.nbytes / 1024:.1f} KB")
            print(f"   Mobile-optimized: ✅")
            break
    else:
        print("❌ No sample images found")


def hd_detection_example():
    """Example: HD detection with 1280x720 input"""
    print("\n🖥️  HD Detection Example (1280x720)")
    print("=" * 40)
    
    # HD configuration for high-resolution detection
    hd_config = {
        'image': {
            'target_size': (1280, 720),  # HD 16:9 aspect ratio
            'resize_strategy': 'letterbox',
            'output_format': 'numpy',
            'channel_order': 'BGR',  # For OpenCV integration
            'padding_color': (114, 114, 114)
        }
    }
    
    pipeline = UniversalPipeline(config=hd_config)
    
    sample_images = [
        "data/sample_data/image/sample_rgb.jpg",
        "data/sample_data/image/pattern.bmp"
    ]
    
    for image_path in sample_images:
        if Path(image_path).exists():
            result = pipeline.fit_transform(image_path)
            print(f"✅ {Path(image_path).name}")
            print(f"   Output shape: {result.shape}")  # Should be (720, 1280, 3)
            print(f"   Memory usage: {result.nbytes / (1024*1024):.2f} MB")
            print(f"   Color order: BGR (OpenCV ready)")
            print(f"   HD resolution: ✅")
            break
    else:
        print("❌ No sample images found")


def custom_cnn_example():
    """Example: Custom CNN with 224x224 input"""
    print("\n🧠 Custom CNN Example (224x224)")
    print("=" * 35)
    
    # Classic CNN configuration (ImageNet size)
    cnn_config = {
        'image': {
            'target_size': (224, 224),  # ImageNet standard
            'resize_strategy': 'crop',  # Center crop for classification
            'output_format': 'tensor',
            'channel_order': 'RGB'
        }
    }
    
    pipeline = UniversalPipeline(config=cnn_config)
    
    sample_images = [
        "data/sample_data/image/sample_rgb.jpg",
        "data/sample_data/image/pattern.bmp"
    ]
    
    for image_path in sample_images:
        if Path(image_path).exists():
            result = pipeline.fit_transform(image_path)
            print(f"✅ {Path(image_path).name}")
            print(f"   Output shape: {result.shape}")  # Should be [3, 224, 224]
            print(f"   Strategy: Center crop")
            print(f"   Ready for CNN: ✅")
            break
    else:
        print("❌ No sample images found")


def batch_processing_example():
    """Example: Batch processing with different sizes"""
    print("\n📦 Batch Processing Example")
    print("=" * 30)
    
    # Different configurations for different use cases
    configs = [
        {'name': 'YOLO', 'size': 640, 'format': 'tensor'},
        {'name': 'Mobile', 'size': 416, 'format': 'numpy'},
        {'name': 'CNN', 'size': 224, 'format': 'tensor'}
    ]
    
    for config in configs:
        print(f"\n🔧 {config['name']} Configuration:")
        
        pipeline_config = {
            'image': {
                'target_size': config['size'],
                'resize_strategy': 'letterbox',
                'output_format': config['format']
            }
        }
        
        pipeline = UniversalPipeline(config=pipeline_config)
        
        # Process a sample image
        sample_path = "data/sample_data/image/sample_rgb.jpg"
        if Path(sample_path).exists():
            result = pipeline.fit_transform(sample_path)
            print(f"   Size: {config['size']}x{config['size']}")
            print(f"   Format: {config['format']}")
            print(f"   Output shape: {result.shape}")
            print(f"   ✅ Ready for {config['name']} models")
        else:
            print(f"   ❌ Sample image not found")


def cli_usage_examples():
    """Show CLI usage examples for custom sizes"""
    print("\n💻 CLI Usage Examples")
    print("=" * 25)
    
    print("📝 To process images with custom sizes using the CLI:")
    print()
    
    print("# YOLO training (640x640):")
    print("python scripts/cli_processor.py your_images/ \\")
    print("  --config '{\"image\": {\"target_size\": 640, \"resize_strategy\": \"letterbox\"}}'")
    print()
    
    print("# Mobile deployment (416x416):")
    print("python scripts/cli_processor.py your_images/ \\")
    print("  --config '{\"image\": {\"target_size\": 416, \"output_format\": \"numpy\"}}'")
    print()
    
    print("# HD detection (1280x720):")
    print("python scripts/cli_processor.py your_images/ \\")
    print("  --config '{\"image\": {\"target_size\": [1280, 720], \"channel_order\": \"BGR\"}}'")
    print()
    
    print("# Custom CNN (224x224 with center crop):")
    print("python scripts/cli_processor.py your_images/ \\")
    print("  --config '{\"image\": {\"target_size\": 224, \"resize_strategy\": \"crop\"}}'")


def configuration_reference():
    """Show complete configuration reference"""
    print("\n📚 Configuration Reference")
    print("=" * 30)
    
    print("""
🔧 Available Configuration Options:

image:
  target_size: int or [width, height]  # Custom output size
    - 640 (square)
    - [1280, 720] (HD)
    - [416, 416] (mobile)
    - null (preserve original)
  
  resize_strategy: str  # How to handle aspect ratio
    - "letterbox" (preserve aspect, add padding) ⭐ Recommended
    - "stretch" (ignore aspect ratio)
    - "crop" (center crop to fit)
    - "fit" (fit within bounds)
  
  output_format: str  # Output data format
    - "numpy" (NumPy array) ⭐ Default
    - "tensor" (PyTorch tensor)
    - "pil" (PIL Image)
  
  channel_order: str  # Color channel order
    - "RGB" ⭐ Default
    - "BGR" (for OpenCV)
  
  padding_color: [R, G, B]  # Letterbox padding color
    - [114, 114, 114] ⭐ YOLO standard
    - [0, 0, 0] (black)
    - [255, 255, 255] (white)
  
  auto_orient: bool  # Auto-rotate based on EXIF
    - true ⭐ Recommended
    - false

🎯 Common Use Cases:

YOLO Object Detection:
  target_size: 640
  resize_strategy: "letterbox"
  output_format: "tensor"
  padding_color: [114, 114, 114]

Mobile Deployment:
  target_size: 416
  resize_strategy: "letterbox"
  output_format: "numpy"

Image Classification:
  target_size: 224
  resize_strategy: "crop"
  output_format: "tensor"

High Resolution Detection:
  target_size: [1280, 720]
  resize_strategy: "letterbox"
  channel_order: "BGR"
""")


def main():
    """Run all examples"""
    print("🚀 Custom Image Sizes Example")
    print("=" * 50)
    print("Learn how to configure custom image sizes for any ML use case!")
    
    try:
        # Run all examples
        yolo_training_example()
        mobile_detection_example()
        hd_detection_example()
        custom_cnn_example()
        batch_processing_example()
        cli_usage_examples()
        configuration_reference()
        
        print("\n✅ All examples completed successfully!")
        print("\n🎉 Key Takeaways:")
        print("• ✅ ANY image size supported - no more forced 640x640!")
        print("• ✅ Aspect ratio preservation with letterbox padding")
        print("• ✅ Multiple output formats: numpy, tensor, PIL")
        print("• ✅ Color order control: RGB or BGR") 
        print("• ✅ Auto-orientation from EXIF data")
        print("• ✅ Works with CLI processor and Python API")
        
        print(f"\n📖 Usage:")
        print("# Python API:")
        print("config = {'image': {'target_size': 640, 'resize_strategy': 'letterbox'}}")
        print("pipeline = UniversalPipeline(config=config)")
        print("result = pipeline.fit_transform('your_image.jpg')")
        print()
        print("# CLI:")
        print("python scripts/cli_processor.py images/ --config '{\"image\": {\"target_size\": 640}}'")
        
    except Exception as e:
        print(f"\n❌ Error during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 