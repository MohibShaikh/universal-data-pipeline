#!/usr/bin/env python3
"""
Quick Universal Data Processor
A simplified version for quick testing without heavy dependencies.

Usage:
    python quick_process.py <directory_path>
"""

import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List

# Add the parent directory to access universal_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universal_pipeline.data_detector import DataTypeDetector

class QuickProcessor:
    """Quick and simple directory processor."""
    
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = self.input_dir / "processed_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.detector = DataTypeDetector()
    
    def scan_and_process(self):
        """Scan directory and process files."""
        print(f"🔍 Scanning directory: {self.input_dir}")
        
        if not self.input_dir.exists():
            print(f"❌ Directory not found: {self.input_dir}")
            return
        
        files_processed = 0
        files_by_type = {}
        
        # Scan all files
        for file_path in self.input_dir.rglob("*"):
            if file_path.is_file():
                try:
                    # Detect data type
                    data_type = self.detector.detect_from_path(str(file_path))
                    
                    if data_type not in files_by_type:
                        files_by_type[data_type] = []
                    files_by_type[data_type].append(file_path)
                    
                except Exception as e:
                    print(f"⚠️  Could not process {file_path.name}: {e}")
                    continue
        
        # Print findings
        print(f"\n📊 Found files by type:")
        for data_type, files in files_by_type.items():
            print(f"  {data_type}: {len(files)} files")
        
        # Process each type
        for data_type, files in files_by_type.items():
            print(f"\n🔄 Processing {data_type} files...")
            
            output_type_dir = self.output_dir / f"{data_type}_processed"
            output_type_dir.mkdir(exist_ok=True)
            
            for file_path in files:
                try:
                    processed = self.process_file(file_path, data_type)
                    if processed is not None:
                        # Save processed data
                        output_file = output_type_dir / f"{file_path.stem}_processed.npy"
                        np.save(output_file, processed)
                        print(f"  ✅ {file_path.name} → {output_file.name}")
                        files_processed += 1
                    
                except Exception as e:
                    print(f"  ❌ Failed to process {file_path.name}: {e}")
        
        # Summary
        print(f"\n🎉 Processing complete!")
        print(f"   Files processed: {files_processed}")
        print(f"   Output directory: {self.output_dir}")
        
        # Save summary
        summary = {
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "files_processed": files_processed,
            "files_by_type": {k: len(v) for k, v in files_by_type.items()}
        }
        
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   Summary saved: {summary_file}")
    
    def process_file(self, file_path: Path, data_type: str):
        """Process a single file based on its type."""
        try:
            if data_type == "tabular":
                # Load and basic processing
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                else:
                    return None
                
                # Simple preprocessing: convert to numeric where possible
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                
                # Fill missing values with mean for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                
                # Convert categorical to numeric codes
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    df[col] = pd.Categorical(df[col]).codes
                
                return df.values
            
            elif data_type == "image":
                # Basic image processing
                try:
                    from PIL import Image
                    img = Image.open(file_path)
                    img_array = np.array(img)
                    
                    # Resize to standard size if too large
                    if img_array.shape[0] > 256 or img_array.shape[1] > 256:
                        img = img.resize((256, 256))
                        img_array = np.array(img)
                    
                    # Normalize to 0-1
                    img_array = img_array.astype(np.float32) / 255.0
                    
                    return img_array
                except ImportError:
                    print("  ⚠️  PIL not available for image processing")
                    return None
            
            elif data_type in ["audio", "video"]:
                # For now, just return file path as string
                return np.array([str(file_path)])
            
            else:
                return None
                
        except Exception as e:
            print(f"  ❌ Error processing {file_path.name}: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(
        description="Quick Universal Data Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python quick_process.py ./test
    
This will:
1. Scan the directory for all files
2. Detect data types automatically
3. Apply basic preprocessing
4. Save results to processed_output/
        """
    )
    
    parser.add_argument(
        "directory",
        help="Directory path to process"
    )
    
    args = parser.parse_args()
    
    try:
        processor = QuickProcessor(args.directory)
        processor.scan_and_process()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 