#!/usr/bin/env python3
"""
Simple Universal Data Processor
Just run: python process.py <directory_name>
"""

import sys
import time
from pathlib import Path

# Add the parent directory to access universal_pipeline
sys.path.append(str(Path(__file__).parent.parent))

def main():
    if len(sys.argv) != 2:
        print("Usage: python process.py <directory>")
        print("Example: python process.py ./my_data")
        sys.exit(1)
    
    directory = sys.argv[1]
    
    if not Path(directory).exists():
        print(f"❌ Directory '{directory}' does not exist")
        sys.exit(1)
    
    print("🚀 Processing your data...")
    start_time = time.time()
    
    # Use the quick processor
    from quick_process import QuickProcessor
    processor = QuickProcessor(directory)
    results = processor.scan_and_process()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Done! Processed in {duration:.1f} seconds")
    print(f"📁 Results saved to: {processor.output_dir}")

if __name__ == "__main__":
    main() 