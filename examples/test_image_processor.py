"""
Simple test for enhanced ImageProcessor
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universal_pipeline.processors.image_processor import ImageProcessor


def test_basic_functionality():
    """Test basic ImageProcessor functionality"""
    print("🧪 Testing Enhanced ImageProcessor")
    print("=" * 40)
    
    # Create a test image
    test_dir = Path("data/temp_test")
    test_dir.mkdir(exist_ok=True)
    
    # Simple test image
    image = Image.new('RGB', (400, 300), color='red')
    image_path = test_dir / "test.jpg"
    image.save(image_path)
    
    print(f"Created test image: {image.size}")
    
    # Test 1: Basic resize with letterbox
    print("\n📐 Test 1: Letterbox resize to 640x640")
    config = {
        'target_size': (640, 640),
        'resize_strategy': 'letterbox',
        'output_format': 'numpy'
    }
    
    processor = ImageProcessor(config=config)
    processor.fit(str(image_path))
    result = processor.transform(str(image_path))
    
    print(f"  Input: {image.size}")
    print(f"  Output: {result.shape}")
    print(f"  Expected: (640, 640, 3)")
    print(f"  ✅ Success: {result.shape == (640, 640, 3)}")
    
    # Test 2: Different output formats
    print("\n🎨 Test 2: Different output formats")
    
    formats = ['numpy', 'tensor', 'pil']
    for fmt in formats:
        config = {
            'target_size': (416, 416),
            'output_format': fmt
        }
        
        processor = ImageProcessor(config=config)
        processor.fit(str(image_path))
        result = processor.transform(str(image_path))
        
        print(f"  Format {fmt}: {type(result)}")
        if hasattr(result, 'shape'):
            print(f"    Shape: {result.shape}")
        elif hasattr(result, 'size'):
            print(f"    Size: {result.size}")
    
    # Test 3: Label transformation
    print("\n🏷️  Test 3: Label transformation")
    
    # Simple YOLO labels for testing
    labels = np.array([
        [0, 0.5, 0.5, 0.3, 0.4],  # Center box
        [1, 0.2, 0.3, 0.1, 0.2]   # Top-left box
    ])
    
    config = {
        'target_size': (640, 640),
        'resize_strategy': 'letterbox',
        'transform_labels': True,
        'return_resize_info': True,
        'label_format': 'yolo'
    }
    
    processor = ImageProcessor(config=config)
    processor.fit(str(image_path))
    
    try:
        result = processor.transform(str(image_path), labels=labels)
        if isinstance(result, tuple):
            image_result, transformed_labels, resize_info = result
            print(f"  ✅ Label transformation successful")
            print(f"  Original labels: {len(labels)}")
            print(f"  Transformed labels: {len(transformed_labels)}")
            print(f"  Resize info keys: {list(resize_info.keys())}")
        else:
            print(f"  ⚠️  Unexpected result type: {type(result)}")
    except Exception as e:
        print(f"  ❌ Label transformation failed: {e}")
    
    # Cleanup
    if image_path.exists():
        image_path.unlink()
    if test_dir.exists():
        test_dir.rmdir()
    
    print("\n✅ Basic tests completed!")


if __name__ == "__main__":
    test_basic_functionality() 