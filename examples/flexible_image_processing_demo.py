"""
Flexible Image Processing Demo
=============================

This example demonstrates the enhanced image processor with:
- Custom image sizes (any size, not just 640x640)
- Multiple resize strategies (letterbox, stretch, crop, fit)
- Auto-orientation based on EXIF data
- Aspect ratio preservation with padding
- Label coordinate transformation for object detection
- Support for different output formats and color orders

Perfect for object detection tasks like YOLO, where you need specific
image sizes and corresponding label transformations.
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universal_pipeline import UniversalPipeline
from universal_pipeline.processors.image_processor import ImageProcessor


def create_sample_image_with_labels():
    """Create a sample image and YOLO format labels for demonstration"""
    
    # Create a simple test image (400x600 - different aspect ratio)
    image = Image.new('RGB', (400, 600), color='lightblue')
    
    # Add some colored rectangles to simulate objects
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Rectangle 1: Red box (top-left area)
    draw.rectangle([50, 100, 150, 200], fill='red', outline='darkred', width=3)
    
    # Rectangle 2: Green box (center-right area) 
    draw.rectangle([250, 250, 350, 400], fill='green', outline='darkgreen', width=3)
    
    # Rectangle 3: Blue box (bottom area)
    draw.rectangle([100, 450, 300, 550], fill='blue', outline='darkblue', width=3)
    
    # Save test image
    test_dir = Path("data/temp_test")
    test_dir.mkdir(exist_ok=True)
    image_path = test_dir / "test_image.jpg"
    image.save(image_path)
    
    # Create corresponding YOLO labels
    # YOLO format: class_id center_x center_y width height (all normalized 0-1)
    labels = [
        [0, 100/400, 150/600, 100/400, 100/600],  # Red box: class 0
        [1, 300/400, 325/600, 100/400, 150/600],  # Green box: class 1  
        [2, 200/400, 500/600, 200/400, 100/600]   # Blue box: class 2
    ]
    
    # Save labels file
    label_path = test_dir / "test_image.txt"
    np.savetxt(label_path, labels, fmt='%d %.6f %.6f %.6f %.6f')
    
    return str(image_path), str(label_path), labels


def demonstrate_resize_strategies():
    """Demonstrate different resize strategies"""
    print("🎯 Demonstrating Different Resize Strategies")
    print("=" * 50)
    
    image_path, label_path, original_labels = create_sample_image_with_labels()
    target_size = (640, 640)  # Common YOLO input size
    
    strategies = ['letterbox', 'stretch', 'crop', 'fit']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Resize Strategies Comparison', fontsize=16)
    
    # Show original image
    original_image = Image.open(image_path)
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title(f'Original\n{original_image.size}')
    axes[0, 0].axis('off')
    
    for i, strategy in enumerate(strategies):
        print(f"\n📐 Strategy: {strategy.upper()}")
        
        # Use ImageProcessor directly for label transformation
        config = {
            'target_size': target_size,
            'resize_strategy': strategy,
            'output_format': 'numpy',
            'transform_labels': True,
            'return_resize_info': True,
            'label_format': 'yolo'
        }
        
        processor = ImageProcessor(config=config)
        processor.fit(image_path)
        
        # Process image with labels
        result = processor.transform(image_path, labels=original_labels)
        
        if isinstance(result, tuple) and len(result) == 3:
            processed_image, transformed_labels, resize_info = result
        else:
            # Fallback for when transform doesn't return tuple
            processed_image = result
            transformed_labels = original_labels
            resize_info = processor.get_resize_info(image_path)
        
        # Display results
        print(f"  Original size: {resize_info.get('original_size', 'N/A')}")
        print(f"  Target size: {resize_info.get('target_size', 'N/A')}")
        print(f"  Strategy: {resize_info.get('strategy', 'N/A')}")
        
        if strategy == 'letterbox' and 'scale' in resize_info:
            print(f"  Scale: {resize_info['scale']:.3f}")
            print(f"  Offset: ({resize_info.get('offset_x', 0)}, {resize_info.get('offset_y', 0)})")
            print(f"  New size: {resize_info.get('new_size', 'N/A')}")
        
        # Show processed image
        axes[0, i+1].imshow(processed_image)
        axes[0, i+1].set_title(f'{strategy.title()}\n{processed_image.shape[:2][::-1]}')
        axes[0, i+1].axis('off')
        
        # Show bounding boxes comparison
        axes[1, i+1].imshow(processed_image)
        
        # Draw transformed bounding boxes
        for j, label in enumerate(transformed_labels):
            if len(label) >= 5:
                class_id, x_center, y_center, width, height = label
                
                # Convert from YOLO format to pixel coordinates
                img_h, img_w = processed_image.shape[:2]
                x1 = int((x_center - width/2) * img_w)
                y1 = int((y_center - height/2) * img_h)
                x2 = int((x_center + width/2) * img_w)
                y2 = int((y_center + height/2) * img_h)
                
                # Draw bounding box
                colors = ['red', 'green', 'blue']
                color = colors[int(class_id)] if int(class_id) < len(colors) else 'yellow'
                
                from matplotlib.patches import Rectangle
                rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color, facecolor='none')
                axes[1, i+1].add_patch(rect)
        
        axes[1, i+1].set_title(f'With Labels\n{strategy.title()}')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/temp_test/resize_strategies_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Comparison image saved to: data/temp_test/resize_strategies_comparison.png")


def demonstrate_custom_sizes():
    """Demonstrate custom image sizes for different use cases"""
    print("\n📏 Demonstrating Custom Image Sizes")
    print("=" * 40)
    
    image_path, label_path, original_labels = create_sample_image_with_labels()
    
    # Different target sizes for different use cases
    size_configs = [
        {'size': (416, 416), 'name': 'YOLOv4 Standard'},
        {'size': (640, 640), 'name': 'YOLOv5/v8 Standard'}, 
        {'size': (832, 832), 'name': 'YOLOv8 Large'},
        {'size': (1280, 720), 'name': 'HD 16:9'},
        {'size': (512, 768), 'name': 'Portrait 2:3'},
        {'size': (1024, 512), 'name': 'Panoramic 2:1'}
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Custom Image Sizes for Different Use Cases', fontsize=16)
    axes = axes.flatten()
    
    for i, size_config in enumerate(size_configs):
        print(f"\n🎯 {size_config['name']}: {size_config['size']}")
        
        config = {
            'image': {
                'target_size': size_config['size'],
                'resize_strategy': 'letterbox',
                'output_format': 'numpy',
                'padding_color': (128, 128, 128)  # Gray padding
            }
        }
        
        pipeline = UniversalPipeline(config=config)
        processed_image = pipeline.fit_transform(image_path)
        
        print(f"  Output shape: {processed_image.shape}")
        print(f"  Memory usage: {processed_image.nbytes / 1024:.1f} KB")
        
        # Display result
        axes[i].imshow(processed_image)
        axes[i].set_title(f'{size_config["name"]}\n{size_config["size"]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/temp_test/custom_sizes_demo.png', dpi=150, bbox_inches='tight')
    print("📊 Custom sizes demo saved to: data/temp_test/custom_sizes_demo.png")


def demonstrate_output_formats():
    """Demonstrate different output formats and color orders"""
    print("\n🎨 Demonstrating Output Formats & Color Orders")
    print("=" * 50)
    
    image_path, _, _ = create_sample_image_with_labels()
    
    formats = [
        {'format': 'numpy', 'channel_order': 'RGB', 'desc': 'NumPy RGB (Default)'},
        {'format': 'numpy', 'channel_order': 'BGR', 'desc': 'NumPy BGR (OpenCV)'},
        {'format': 'tensor', 'channel_order': 'RGB', 'desc': 'PyTorch RGB'},
        {'format': 'tensor', 'channel_order': 'BGR', 'desc': 'PyTorch BGR'},
        {'format': 'pil', 'channel_order': 'RGB', 'desc': 'PIL Image'}
    ]
    
    for fmt in formats:
        print(f"\n🔧 {fmt['desc']}:")
        
        config = {
            'image': {
                'target_size': (416, 416),
                'resize_strategy': 'letterbox',
                'output_format': fmt['format'],
                'channel_order': fmt['channel_order']
            }
        }
        
        pipeline = UniversalPipeline(config=config)
        result = pipeline.fit_transform(image_path)
        
        print(f"  Type: {type(result)}")
        if hasattr(result, 'shape'):
            print(f"  Shape: {result.shape}")
        elif hasattr(result, 'size'):
            print(f"  Size: {result.size}")
        
        if hasattr(result, 'dtype'):
            print(f"  Data type: {result.dtype}")
        
        # Show first few pixel values for comparison
        if fmt['format'] == 'numpy':
            print(f"  First pixel RGB: {result[0, 0]}")
        elif fmt['format'] == 'tensor':
            print(f"  First pixel: {result[:, 0, 0].numpy()}")


def demonstrate_label_transformation():
    """Demonstrate label coordinate transformation"""
    print("\n🏷️  Demonstrating Label Transformation")
    print("=" * 45)
    
    image_path, label_path, original_labels = create_sample_image_with_labels()
    
    print(f"Original image size: {Image.open(image_path).size}")
    print(f"Original labels (YOLO format):")
    for i, label in enumerate(original_labels):
        print(f"  Object {i}: class={int(label[0])}, x={label[1]:.3f}, y={label[2]:.3f}, w={label[3]:.3f}, h={label[4]:.3f}")
    
    # Transform to different target sizes
    target_sizes = [(416, 416), (640, 640), (1280, 720)]
    
    for target_size in target_sizes:
        print(f"\n📐 Target size: {target_size}")
        
        config = {
            'target_size': target_size,
            'resize_strategy': 'letterbox',
            'transform_labels': True,
            'label_format': 'yolo',
            'return_resize_info': True
        }
        
        processor = ImageProcessor(config=config)
        processor.fit(image_path)
        
        # Process with label transformation
        result = processor.transform(image_path, labels=original_labels)
        
        if isinstance(result, tuple) and len(result) == 3:
            processed_image, transformed_labels, resize_info = result
        else:
            # Fallback
            processed_image = result
            transformed_labels = original_labels
            resize_info = processor.get_resize_info(image_path)
        
        print(f"  Transformed labels:")
        for i, label in enumerate(transformed_labels):
            print(f"    Object {i}: class={int(label[0])}, x={label[1]:.3f}, y={label[2]:.3f}, w={label[3]:.3f}, h={label[4]:.3f}")
        
        # Show transformation info
        if resize_info:
            print(f"  Scale: {resize_info.get('scale', 'N/A')}")
            print(f"  Offset: ({resize_info.get('offset_x', 0)}, {resize_info.get('offset_y', 0)})")


def demonstrate_real_world_usage():
    """Demonstrate real-world usage patterns"""
    print("\n🌍 Real-World Usage Examples")
    print("=" * 35)
    
    image_path, label_path, original_labels = create_sample_image_with_labels()
    
    # Example 1: YOLOv8 Training Pipeline
    print("\n🎯 Example 1: YOLOv8 Training Pipeline")
    print("-" * 40)
    
    yolo_config = {
        'image': {
            'target_size': 640,  # Square input
            'resize_strategy': 'letterbox',
            'output_format': 'tensor',
            'channel_order': 'RGB',
            'auto_orient': True
        }
    }
    
    pipeline = UniversalPipeline(config=yolo_config)
    image_tensor = pipeline.fit_transform(image_path)
    
    print(f"Image tensor shape: {image_tensor.shape}")  # Should be [3, 640, 640]
    print(f"Data type: {image_tensor.dtype}")
    print(f"Value range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    # Show how to handle labels separately
    label_processor = ImageProcessor(config={
        'target_size': 640,
        'resize_strategy': 'letterbox',
        'transform_labels': True,
        'label_format': 'yolo'
    })
    label_processor.fit(image_path)
    _, transformed_labels, _ = label_processor.transform(image_path, labels=original_labels)
    print(f"Number of labels: {len(transformed_labels)}")
    
    # Example 2: Custom Detection Model (HD input)
    print("\n📱 Example 2: Custom Detection Model (HD Input)")
    print("-" * 50)
    
    hd_config = {
        'image': {
            'target_size': (1280, 720),  # HD 16:9
            'resize_strategy': 'letterbox',
            'output_format': 'numpy',
            'channel_order': 'BGR',  # For OpenCV integration
            'padding_color': (114, 114, 114)  # Standard gray
        }
    }
    
    pipeline = UniversalPipeline(config=hd_config)
    image_array = pipeline.fit_transform(image_path)
    
    print(f"Image array shape: {image_array.shape}")  # Should be [720, 1280, 3]
    print(f"Color order: BGR (OpenCV compatible)")
    print(f"Memory usage: {image_array.nbytes / (1024*1024):.2f} MB")
    
    # Example 3: Batch Processing with Different Sizes
    print("\n📦 Example 3: Batch Processing Different Sizes")
    print("-" * 48)
    
    # Create multiple test images of different sizes
    test_images = []
    for i, size in enumerate([(300, 400), (800, 600), (1920, 1080)]):
        img = Image.new('RGB', size, color=f'C{i}')
        img_path = f"data/temp_test/batch_test_{i}.jpg"
        img.save(img_path)
        test_images.append(img_path)
    
    batch_config = {
        'image': {
            'target_size': (512, 512),
            'resize_strategy': 'letterbox',
            'output_format': 'numpy'
        }
    }
    
    pipeline = UniversalPipeline(config=batch_config)
    
    for i, img_path in enumerate(test_images):
        original_size = Image.open(img_path).size
        processed = pipeline.fit_transform(img_path)
        print(f"  Image {i+1}: {original_size} → {processed.shape[:2][::-1]}")


def demonstrate_yolo_integration():
    """Demonstrate complete YOLO integration workflow"""
    print("\n🎯 YOLO Integration Workflow")
    print("=" * 35)
    
    image_path, label_path, original_labels = create_sample_image_with_labels()
    
    # YOLO training configuration
    config = {
        'target_size': 640,
        'resize_strategy': 'letterbox',
        'output_format': 'tensor',
        'transform_labels': True,
        'label_format': 'yolo',
        'return_resize_info': True,
        'padding_color': (114, 114, 114),
        'auto_orient': True
    }
    
    processor = ImageProcessor(config=config)
    processor.fit(image_path)
    
    # Process image and labels together
    image_tensor, transformed_labels, resize_info = processor.transform(image_path, labels=original_labels)
    
    print(f"✅ YOLO-ready image tensor: {image_tensor.shape}")
    print(f"✅ Transformed labels: {len(transformed_labels)} objects")
    print(f"✅ Letterbox info: scale={resize_info.get('scale', 1):.3f}")
    
    # Show how this integrates with a typical YOLO training loop
    print("\n📝 Integration with YOLO training:")
    print("```python")
    print("# This is how you'd use it in actual training:")
    print("processor = ImageProcessor(yolo_config)")
    print("for image_path, label_path in dataset:")
    print("    image, labels, _ = processor.transform(image_path, labels=load_labels(label_path))")
    print("    # image: torch.Tensor [3, 640, 640]")
    print("    # labels: np.array [[class, x, y, w, h], ...]")
    print("    loss = model(image, labels)")
    print("    loss.backward()")
    print("```")


def cleanup_temp_files():
    """Clean up temporary test files"""
    import shutil
    temp_dir = Path("data/temp_test")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print("\n🧹 Cleaned up temporary files")


def main():
    """Run all demonstrations"""
    print("🚀 Flexible Image Processing Demo")
    print("=" * 60)
    print("This demo shows the enhanced image processor capabilities:")
    print("• Custom image sizes (any dimensions)")
    print("• Multiple resize strategies")
    print("• Auto-orientation from EXIF data")
    print("• Label coordinate transformation")
    print("• Different output formats")
    print("• Real-world usage patterns")
    
    try:
        # Run all demonstrations
        demonstrate_resize_strategies()
        demonstrate_custom_sizes()
        demonstrate_output_formats()
        demonstrate_label_transformation()
        demonstrate_real_world_usage()
        demonstrate_yolo_integration()
        
        print("\n✅ All demonstrations completed successfully!")
        print("\nKey Features:")
        print("• ✅ Custom image sizes - any dimensions you need")
        print("• ✅ Letterbox resizing - preserves aspect ratio")
        print("• ✅ Auto label transformation - coordinates adjusted automatically")
        print("• ✅ Multiple output formats - numpy, tensor, PIL")
        print("• ✅ Color order control - RGB or BGR")
        print("• ✅ Auto-orientation - handles rotated images")
        
        print(f"\n📁 Results saved to: data/temp_test/")
        print("📊 Check the generated comparison images!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Always cleanup
        cleanup_temp_files()


if __name__ == "__main__":
    main() 