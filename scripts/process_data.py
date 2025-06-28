#!/usr/bin/env python3
"""
Universal Data Processing CLI
A comprehensive command-line tool for automatic data preprocessing.

Usage:
    python process_data.py <directory> [options]

Examples:
    python process_data.py ./my_data
    python process_data.py ./my_data --output ./results
    python process_data.py ./my_data --advanced --config config.json
    python process_data.py ./my_data --preview
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add the parent directory to access universal_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_banner():
    """Print a nice banner"""
    print("="*60)
    print("🚀 UNIVERSAL DATA PROCESSING PIPELINE")
    print("   Automatic preprocessing for any data type")
    print("="*60)

def preview_directory(directory: Path):
    """Preview what files would be processed"""
    from universal_pipeline.data_detector import DataTypeDetector
    
    detector = DataTypeDetector()
    files_by_type = {}
    
    print(f"\n📁 Scanning directory: {directory}")
    print("-" * 40)
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            try:
                data_type = detector.detect_from_path(str(file_path))
                if data_type not in files_by_type:
                    files_by_type[data_type] = []
                files_by_type[data_type].append(file_path.name)
            except Exception:
                continue
    
    total_files = sum(len(files) for files in files_by_type.values())
    
    print(f"📊 Found {total_files} files:")
    for data_type, files in files_by_type.items():
        print(f"   {data_type}: {len(files)} files")
        if len(files) <= 5:
            for file in files[:3]:
                print(f"     • {file}")
        else:
            for file in files[:3]:
                print(f"     • {file}")
            print(f"     ... and {len(files)-3} more")
    
    print("\n💡 Use without --preview to process these files")
    return files_by_type

def basic_processing(directory: str, output_dir: str = None):
    """Simple processing using the quick processor"""
    from quick_process import QuickProcessor
    
    print("\n🔄 Starting basic processing...")
    start_time = time.time()
    
    processor = QuickProcessor(directory)
    if output_dir:
        processor.output_dir = Path(output_dir)
    
    results = processor.scan_and_process()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n✅ Processing completed in {duration:.2f} seconds")
    print(f"📁 Results saved to: {processor.output_dir}")
    
    return results

def advanced_processing(directory: str, output_dir: str = None, config_file: str = None):
    """Advanced processing using the full CLI processor"""
    from cli_processor import ThreadSafeDirectoryProcessor
    
    print("\n🔄 Starting advanced processing...")
    start_time = time.time()
    
    # Load config if provided
    config = {}
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"📋 Loaded configuration from {config_file}")
    
    processor = ThreadSafeDirectoryProcessor(directory, output_dir, config)
    results = processor.process_directory()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n✅ Processing completed in {duration:.2f} seconds")
    print(f"📁 Results saved to: {processor.output_dir}")
    
    return results

def create_sample_config():
    """Create a sample configuration file"""
    config_path = "sample_config.json"
    
    sample_config = {
        "image": {
            "resize": [224, 224],
            "normalize": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "tabular": {
            "scaling_method": "standard",
            "encoding_method": "onehot",
            "handle_missing": "impute"
        },
        "audio": {
            "features": ["mfcc", "spectral_centroid"],
            "n_mfcc": 13
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"📄 Sample configuration created: {config_path}")
    print("   Edit this file to customize processing parameters")

def main():
    parser = argparse.ArgumentParser(
        description="Universal Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_data.py ./data                          # Basic processing
  python process_data.py ./data --output ./results       # Custom output
  python process_data.py ./data --preview                # Preview files
  python process_data.py ./data --advanced               # Advanced mode
  python process_data.py --create-config                 # Create config file
        """
    )
    
    parser.add_argument('directory', nargs='?', help='Directory to process')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--preview', '-p', action='store_true', 
                       help='Preview files without processing')
    parser.add_argument('--advanced', '-a', action='store_true',
                       help='Use advanced processing with full pipeline')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--create-config', action='store_true',
                       help='Create a sample configuration file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Handle special actions
    if args.create_config:
        create_sample_config()
        return
    
    if not args.directory:
        parser.print_help()
        print("\n❌ Error: Directory path is required")
        return
    
    directory = Path(args.directory)
    if not directory.exists():
        print(f"❌ Error: Directory '{directory}' does not exist")
        return
    
    if not directory.is_dir():
        print(f"❌ Error: '{directory}' is not a directory")
        return
    
    if not args.quiet:
        print_banner()
    
    # Preview mode
    if args.preview:
        preview_directory(directory)
        return
    
    # Processing mode
    try:
        if args.advanced:
            results = advanced_processing(
                str(directory), 
                args.output, 
                args.config
            )
        else:
            results = basic_processing(str(directory), args.output)
        
        if not args.quiet:
            print("\n🎉 All done! Your data is ready for ML training.")
            
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        if not args.quiet:
            print("💡 Try using --preview to check your files first")
        sys.exit(1)

if __name__ == "__main__":
    main() 