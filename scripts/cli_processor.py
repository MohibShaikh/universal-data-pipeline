#!/usr/bin/env python3
"""
Universal Data Pipeline CLI
Automatically processes all files in a directory using appropriate data processors.

Usage:
    python cli_processor.py <directory_path> [--output <output_dir>] [--config <config_file>]

Examples:
    python cli_processor.py ./test
    python cli_processor.py ./data --output ./processed_data
    python cli_processor.py ./my_data --config config.json
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from multiprocessing import cpu_count
import threading  # Add threading for locks
from concurrent.futures import as_completed

# Add the parent directory to access universal_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universal_pipeline.data_detector import DataTypeDetector
from universal_pipeline.pipeline import UniversalPipeline

# Configure logging with proper setup
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)  # Ensure logs directory exists
log_file = os.path.join(log_dir, 'pipeline_processing.log')

# Clear any existing handlers to avoid conflicts
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Setup root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add file handler
file_handler = logging.FileHandler(log_file, mode='a')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers to root logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class ThreadSafeDirectoryProcessor:
    """Thread-safe processor for all files in a directory using the Universal Data Pipeline."""
    
    def __init__(self, input_dir: str, output_dir: str = None, config: Dict = None, 
                 parallel: bool = True, max_workers: int = None, batch_size: int = 10, verbose: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / "processed_output"
        self.config = config or {}
        
        # Performance settings
        self.parallel = parallel
        self.max_workers = max_workers or min(cpu_count(), 8)  # Reasonable default
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Thread safety
        self.pipeline_locks = {}  # Lock per data type
        self.fitted_pipelines = {}  # Pre-fitted pipelines
        
        # Initialize components
        self.detector = DataTypeDetector()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipelines for different data types
        self.pipelines = {}
        
    def scan_directory(self) -> Dict[str, List[Path]]:
        """Scan directory and group files by detected data type."""
        logger.info(f"Scanning directory: {self.input_dir}")
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.input_dir}")
        
        files_by_type = {}
        total_files = 0
        
        # Get all files recursively
        for file_path in self.input_dir.rglob("*"):
            if file_path.is_file():
                try:
                    # Detect data type
                    data_type = self.detector.detect_from_path(str(file_path))
                    
                    if data_type not in files_by_type:
                        files_by_type[data_type] = []
                    
                    files_by_type[data_type].append(file_path)
                    total_files += 1
                    
                except Exception as e:
                    logger.warning(f"Could not process file {file_path}: {e}")
        
        logger.info(f"Found {total_files} files across {len(files_by_type)} data types")
        for data_type, files in files_by_type.items():
            logger.info(f"  {data_type}: {len(files)} files")
        
        return files_by_type
    
    def load_file_data(self, file_path: Path, data_type: str) -> Any:
        """Load data from file based on its type."""
        try:
            if data_type == "tabular":
                if file_path.suffix.lower() == '.csv':
                    return pd.read_csv(file_path)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    return pd.read_excel(file_path)
                elif file_path.suffix.lower() == '.parquet':
                    return pd.read_parquet(file_path)
                elif file_path.suffix.lower() == '.json':
                    import json
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                        if isinstance(json_data, list):
                            return pd.DataFrame(json_data)
                        else:
                            return pd.json_normalize(json_data)
                else:
                    # Try CSV as fallback
                    return pd.read_csv(file_path)
            
            elif data_type == "image":
                from PIL import Image
                return np.array(Image.open(file_path))
            
            elif data_type == "audio":
                import librosa
                audio_data, sr = librosa.load(str(file_path))
                return {"audio": audio_data, "sample_rate": sr}
            
            elif data_type == "video":
                import cv2
                return str(file_path)  # Pass path for video processing
            
            elif data_type == "timeseries":
                # Assume CSV format for time series
                return pd.read_csv(file_path, parse_dates=[0] if file_path.suffix.lower() == '.csv' else None)
            
            else:
                logger.warning(f"Unknown data type {data_type} for file {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None

    def prepare_pipeline(self, schema: str, sample_files: List[Path]) -> bool:
        """Pre-fit pipeline with a sample file to ensure thread safety."""
        try:
            # Extract data type from schema
            data_type = schema.split('_')[0] if schema != "unknown" else "unknown"
            
            # Initialize pipeline for this schema if not exists
            if schema not in self.pipelines:
                processor_config = self.config.get(data_type, {})
                full_config = {data_type: processor_config} if processor_config else {}
                self.pipelines[schema] = UniversalPipeline(config=full_config)
                self.pipeline_locks[schema] = threading.Lock()
            
            pipeline = self.pipelines[schema]
            
            # Use first file as sample to fit the pipeline
            sample_file = sample_files[0]
            sample_data = self.load_file_data(sample_file, data_type)
            
            if sample_data is None:
                return False
            
            # Fit the pipeline with sample data
            if data_type == "audio" and isinstance(sample_data, dict):
                pipeline.fit(sample_data["audio"], data_type="audio")
            else:
                pipeline.fit(sample_data, data_type=data_type)
            
            # Store the fitted pipeline
            self.fitted_pipelines[schema] = pipeline
            logger.info(f"Pipeline fitted for {schema} using sample: {sample_file.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fitting pipeline for {schema}: {e}")
            return False

    def process_single_file_safe(self, file_path: Path, schema: str, retry_count: int = 2) -> Dict[str, Any]:
        """Thread-safe processing of a single file with retry mechanism."""
        last_error = None
        data_type = schema.split('_')[0] if schema != "unknown" else "unknown"
        
        for attempt in range(retry_count + 1):
            try:
                # Load data
                data = self.load_file_data(file_path, data_type)
                if data is None:
                    return {
                        "original_file": str(file_path),
                        "error": f"Could not load data from {file_path}",
                        "status": "failed"
                    }
                
                # Get the pre-fitted pipeline
                pipeline = self.fitted_pipelines[schema]
                
                # Thread-safe transform (pipeline is already fitted)
                with self.pipeline_locks[schema]:
                    if data_type == "audio" and isinstance(data, dict):
                        processed_data = pipeline.transform(data["audio"])
                    else:
                        processed_data = pipeline.transform(data)
                
                # Save processed data
                output_file = self.output_dir / f"{data_type}_processed" / f"{file_path.stem}_processed.npy"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_file, processed_data)
                
                return {
                    "original_file": str(file_path),
                    "output_file": str(output_file),
                    "shape": processed_data.shape if hasattr(processed_data, 'shape') else len(processed_data),
                    "data_type": data_type,
                    "schema": schema,
                    "status": "success",
                    "attempts": attempt + 1
                }
                
            except Exception as e:
                last_error = str(e)
                if attempt < retry_count:
                    logger.debug(f"Attempt {attempt + 1} failed for {file_path}, retrying: {e}")
                    time.sleep(0.1)  # Brief pause before retry
                else:
                    logger.error(f"All {retry_count + 1} attempts failed for {file_path}: {e}")
        
        return {
            "original_file": str(file_path),
            "error": last_error,
            "status": "failed",
            "data_type": data_type,
            "schema": schema,
            "attempts": retry_count + 1
        }

    def validate_file_integrity(self, file_path: Path, data_type: str) -> bool:
        """Validate if a file can be processed before attempting."""
        try:
            if data_type == "image":
                from PIL import Image
                with Image.open(file_path) as img:
                    # Try to load the image to check if it's corrupted
                    img.verify()
                return True
            elif data_type == "audio":
                import librosa
                # Try to load just a small sample to check file integrity
                try:
                    librosa.load(str(file_path), duration=0.1)
                    return True
                except:
                    return False
            elif data_type in ["tabular", "timeseries"]:
                # Check if file is readable and not empty
                return file_path.stat().st_size > 0
            else:
                return True  # Assume valid for other types
        except Exception:
            return False

    def process_files_batch_safe(self, files: List[Path], schema: str) -> List[Dict[str, Any]]:
        """Process a batch of files with thread safety and improved error handling."""
        if not self.parallel or len(files) <= 1:
            # Sequential processing
            return [self.process_single_file_safe(f, schema) for f in files]
        
        # Parallel processing using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file_safe, file_path, schema): file_path
                for file_path in files
            }
            
            results = []
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Unexpected error processing {file_path}: {e}")
                    results.append({
                        "original_file": str(file_path),
                        "error": f"Unexpected error: {e}",
                        "status": "failed",
                        "schema": schema
                    })
            
            return results

    def process_files_by_type(self, files_by_type: Dict[str, List[Path]]) -> Dict[str, Any]:
        """Process files grouped by data type and schema."""
        results = {}
        
        # Group files by schema
        files_by_schema = self.group_files_by_schema(files_by_type)
        
        # Create a mapping from schema back to data type
        schema_to_datatype = {}
        for schema, files in files_by_schema.items():
            if schema == "unknown":
                schema_to_datatype[schema] = "unknown"
            else:
                # Extract data type from schema (e.g., "timeseries_12345" -> "timeseries")
                data_type = schema.split('_')[0]
                schema_to_datatype[schema] = data_type
        
        # Process each schema group
        for schema, files in files_by_schema.items():
            data_type = schema_to_datatype[schema]
            
            if data_type == "unknown":
                logger.info(f"Skipping {len(files)} files: unknown data type")
                for file in files[:3]:  # Show first 3
                    logger.info(f"  Skipped: {file}")
                if len(files) > 3:
                    logger.info(f"  ... and {len(files) - 3} more")
                continue
            
            logger.info(f"Processing {len(files)} {data_type} files (schema: {schema})...")
            
            # Prepare pipeline for this schema
            if not self.prepare_pipeline(schema, files):
                logger.error(f"Failed to prepare pipeline for schema {schema}")
                continue
            
            start_time = time.time()
            
            if self.parallel and len(files) > 1:
                # Parallel processing
                successful_results = []
                failed_files = []
                
                # Process files in batches
                num_batches = (len(files) + self.batch_size - 1) // self.batch_size
                
                for batch_num in range(num_batches):
                    start_idx = batch_num * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(files))
                    batch_files = files[start_idx:end_idx]
                    
                    logger.info(f"  Processing batch {batch_num + 1}/{num_batches} ({len(batch_files)} files)")
                    
                    batch_results = self.process_files_batch_safe(batch_files, schema)
                    
                    for result in batch_results:
                        if result["status"] == "success":
                            successful_results.append(result)
                        else:
                            failed_files.append(result)
            else:
                # Sequential processing
                successful_results = []
                failed_files = []
                
                for file_path in files:
                    result = self.process_single_file_safe(file_path, schema)
                    if result["status"] == "success":
                        successful_results.append(result)
                    else:
                        failed_files.append(result)
            
            processing_time = time.time() - start_time
            
            # Report failed files
            if failed_files:
                logger.warning(f"Failed to process {len(failed_files)} {data_type} files:")
                for failed in failed_files[:3]:  # Show first 3 failures
                    logger.warning(f"  {failed['original_file']}: {failed.get('error', 'Unknown error')}")
                if len(failed_files) > 3:
                    logger.warning(f"  ... and {len(failed_files) - 3} more failures")
            
            logger.info(f"  Completed {data_type}: {len(successful_results)}/{len(files)} files ({len(successful_results)/len(files)*100:.1f}%) in {processing_time:.2f}s")
            
            # Store results by data type (combining all schemas of same type)
            if data_type not in results:
                results[data_type] = {
                    "total_files": 0,
                    "processed_files": 0,
                    "failed_files": 0,
                    "results": []
                }
            
            results[data_type]["total_files"] += len(files)
            results[data_type]["processed_files"] += len(successful_results)
            results[data_type]["failed_files"] += len(failed_files)
            results[data_type]["results"].extend(successful_results)
            
            # Save pipeline for this schema
            if self.fitted_pipelines.get(schema):
                try:
                    pipeline_file = self.output_dir / f"{schema}_pipeline.pkl"
                    pipeline = self.fitted_pipelines[schema]
                    pipeline.save_pipeline(str(pipeline_file))
                    logger.info(f"Saved {schema} pipeline to {pipeline_file}")
                except Exception as e:
                    logger.error(f"Could not save pipeline for {schema}: {e}")
        
        return results
    
    def generate_summary_report(self, results: Dict[str, Any]):
        """Generate a comprehensive summary report."""
        summary = {
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "processing_summary": {},
            "total_files_processed": 0,
            "data_types_found": list(results.keys())
        }
        
        for data_type, result in results.items():
            summary["processing_summary"][data_type] = {
                "total_files": result["total_files"],
                "successfully_processed": result["processed_files"],
                "success_rate": f"{(result['processed_files'] / result['total_files'] * 100):.1f}%" if result["total_files"] > 0 else "0%"
            }
            summary["total_files_processed"] += result["processed_files"]
        
        # Save summary report
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary to console
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Input Directory: {summary['input_directory']}")
        print(f"Output Directory: {summary['output_directory']}")
        print(f"Total Files Processed: {summary['total_files_processed']}")
        print(f"Data Types Found: {', '.join(summary['data_types_found'])}")
        print("\nDetailed Results:")
        
        for data_type, stats in summary["processing_summary"].items():
            print(f"  {data_type.capitalize()}:")
            print(f"    Files: {stats['successfully_processed']}/{stats['total_files']} ({stats['success_rate']})")
        
        print(f"\nFull report saved to: {summary_file}")
        print("="*60)
        
        return summary
    
    def process_directory(self) -> Dict[str, Any]:
        """Main processing function."""
        logger.info("Starting directory processing...")
        
        try:
            # Scan directory
            files_by_type = self.scan_directory()
            
            if not files_by_type:
                logger.warning("No processable files found in directory")
                return {}
            
            # Process files
            results = self.process_files_by_type(files_by_type)
            
            # Generate summary
            summary = self.generate_summary_report(results)
            
            logger.info("Directory processing completed successfully!")
            return summary
            
        except Exception as e:
            logger.error(f"Error during directory processing: {e}")
            raise

    def get_file_schema(self, file_path: Path, data_type: str) -> str:
        """Get a schema signature for a file based on its columns."""
        try:
            data = self.load_file_data(file_path, data_type)
            if data is None:
                return "unknown"
            
            if data_type == "tabular" and hasattr(data, 'columns'):
                # Sort columns for consistent schema detection
                columns = sorted(data.columns.tolist())
                return f"tabular_{hash(tuple(columns))}"
            
            elif data_type == "timeseries" and hasattr(data, 'columns'):
                # For timeseries, use column names to create schema
                columns = sorted(data.columns.tolist())
                return f"timeseries_{hash(tuple(columns))}"
            
            elif data_type == "image":
                if hasattr(data, 'shape'):
                    # Group images by dimensions
                    return f"image_{len(data.shape)}d_{data.shape}"
                return f"image_default"
            
            elif data_type == "audio":
                if isinstance(data, dict) and 'audio' in data:
                    return f"audio_dict"
                return f"audio_array"
            
            else:
                return f"{data_type}_default"
                
        except Exception as e:
            logger.warning(f"Could not determine schema for {file_path}: {e}")
            return f"{data_type}_unknown"

    def group_files_by_schema(self, files_by_type: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
        """Group files by both data type and schema."""
        schema_groups = {}
        
        for data_type, files in files_by_type.items():
            # Skip unknown files
            if data_type == "unknown":
                schema_groups[data_type] = files
                continue
            
            # Group files by schema within data type
            type_schema_groups = {}
            for file_path in files:
                schema = self.get_file_schema(file_path, data_type)
                if schema not in type_schema_groups:
                    type_schema_groups[schema] = []
                type_schema_groups[schema].append(file_path)
            
            # Add schema groups to main groups
            for schema, schema_files in type_schema_groups.items():
                schema_groups[schema] = schema_files
        
        return schema_groups

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(
        description="Universal Data Pipeline CLI - Process all files in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli_processor.py ./test
    python cli_processor.py ./data --output ./processed_data
    python cli_processor.py ./my_data --config config.json
    python cli_processor.py ./data --parallel --workers 4 --batch-size 5
    python cli_processor.py ./large_dataset --no-parallel --verbose
    
The script will:
1. Scan the directory for all files
2. Automatically detect data types (tabular, image, audio, video, timeseries)
3. Process each file with appropriate preprocessing
4. Save results in organized output directory
5. Generate a comprehensive summary report

Performance Options:
- Use --parallel for faster processing of multiple files
- Adjust --workers to control CPU usage
- Increase --batch-size for better throughput on large datasets
        """
    )
    
    parser.add_argument(
        "directory",
        help="Input directory path to process"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output directory for processed files (default: <input_dir>/processed_output)"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Configuration file path (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--parallel", 
        action="store_true",
        default=True,
        help="Enable parallel processing (default: True)"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true", 
        help="Disable parallel processing"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        help=f"Number of parallel workers (default: min(CPU cores, 8) = {min(cpu_count(), 8)})"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of files to process in each batch (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Determine parallel processing setting
    parallel = args.parallel and not args.no_parallel
    
    try:
        # Initialize processor
        processor = ThreadSafeDirectoryProcessor(
            input_dir=args.directory,
            output_dir=args.output,
            config=config,
            parallel=parallel,
            max_workers=args.workers,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        
        logger.info(f"Performance settings: parallel={parallel}, max_workers={processor.max_workers}, batch_size={args.batch_size}")
        
        # Process directory
        summary = processor.process_directory()
        
        if summary.get("total_files_processed", 0) > 0:
            print(f"\n✅ Successfully processed {summary['total_files_processed']} files!")
        else:
            print("\n⚠️  No files were processed.")
            
    except Exception as e:
        logger.error(f"Failed to process directory: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 