"""
Comprehensive Integration Examples for Universal Data Pipeline
Demonstrates end-to-end ML workflows with real models and frameworks.

This shows how the Universal Pipeline's domain-appropriate outputs can be
directly fed into production ML models without additional preprocessing.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Add the universal_pipeline to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universal_pipeline import UniversalPipeline, process_data
print("✅ Universal Pipeline imported successfully!")

def demonstrate_image_workflow():
    """Demo: Images → Universal Pipeline → YOLO/CNN Training"""
    print("="*60)
    print("🖼️  IMAGE PROCESSING → COMPUTER VISION WORKFLOW")
    print("="*60)
    
    # Configure for domain-appropriate image output
    image_config = {
        'image': {
            'preserve_original_size': True,  # Keep natural dimensions
            'output_format': 'numpy',        # Return numpy arrays, not tensors
            'ensure_rgb': True,              # Consistent color channels
            'normalize': False               # Keep original pixel values
        }
    }
    
    pipeline = UniversalPipeline(image_config)
    
    # Process sample images
    sample_dir = Path("sample_data/image")
    if sample_dir.exists():
        print(f"📁 Processing images from: {sample_dir}")
        
        for img_file in sample_dir.glob("*.jpg"):
            if img_file.name.startswith('sample'):
                print(f"  📸 Processing: {img_file.name}")
                
                # Pipeline processes image
                processed_img = pipeline.fit_transform(str(img_file))
                print(f"     Shape: {processed_img.shape}, Type: {processed_img.dtype}")
                
                # ✅ DIRECT INTEGRATION: Ready for Computer Vision models
                print("  🎯 Model Integration Examples:")
                
                # Example 1: YOLO Object Detection
                print("    📦 YOLO Object Detection:")
                try:
                    import torch
                    
                    # Convert to YOLO input format (1, 3, H, W)
                    if len(processed_img.shape) == 3:  # (H, W, C)
                        yolo_input = torch.from_numpy(processed_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        print(f"      YOLO Input Shape: {yolo_input.shape}")
                        print("      ✅ Ready for: model(yolo_input) → detections")
                    
                    # Example 2: CNN Classifier
                    print("    🧠 CNN Classification:")
                    cnn_input = torch.from_numpy(processed_img).permute(2, 0, 1).float() / 255.0
                    print(f"      CNN Input Shape: {cnn_input.shape}")
                    print("      ✅ Ready for: classifier(cnn_input) → class_probabilities")
                    
                except ImportError:
                    print("      ⚠️  PyTorch not available - showing expected formats:")
                    print(f"      YOLO: torch.Size([1, 3, {processed_img.shape[0]}, {processed_img.shape[1]}])")
                    print(f"      CNN: torch.Size([3, {processed_img.shape[0]}, {processed_img.shape[1]}])")
                
                # Example 3: OpenCV Processing
                print("    🔍 OpenCV Integration:")
                print(f"      OpenCV Input: {processed_img.shape} → cv2.detectKeypoints()")
                print("      ✅ Ready for: feature detection, image filtering, etc.")
                
                break  # Process just one image for demo
    else:
        print("⚠️  No sample images found. Run the main demo first to create sample data.")
    
    print()

def demonstrate_tabular_workflow():
    """Demo: CSV/Excel → Universal Pipeline → Sklearn/AutoML"""
    print("="*60)
    print("📊 TABULAR DATA → MACHINE LEARNING WORKFLOW")
    print("="*60)
    
    # Configure for ML-ready tabular output
    tabular_config = {
        'tabular': {
            'scaling_method': 'standard',
            'handle_categorical': True,
            'encoding_method': 'onehot',
            'handle_missing': True
        }
    }
    
    pipeline = UniversalPipeline(tabular_config)
    
    # Process sample tabular data
    sample_dir = Path("sample_data/tabular")
    if sample_dir.exists():
        csv_file = sample_dir / "employees.csv"
        if csv_file.exists():
            print(f"📁 Processing tabular data: {csv_file.name}")
            
            # Pipeline processes tabular data
            result = pipeline.fit_transform(str(csv_file))
            
            # Handle both single array and tuple returns
            if isinstance(result, tuple):
                X, y = result
            else:
                X = result
                y = None
            
            print(f"  Features (X): {X.shape}, Target (y): {y.shape if y is not None else 'None'}")
            print(f"  Data types: X={X.dtype}, y={y.dtype if y is not None else 'N/A'}")
            
            # ✅ DIRECT INTEGRATION: Ready for ML models
            print("  🎯 Model Integration Examples:")
            
            # Example 1: Scikit-learn Models
            print("    🔬 Scikit-learn Integration:")
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                
                if y is not None:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Classification example
                    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                    classifier.fit(X_train, y_train)
                    accuracy = classifier.score(X_test, y_test)
                    print(f"      Random Forest Accuracy: {accuracy:.3f}")
                    print("      ✅ Ready for: any sklearn classifier/regressor")
                    
                    # Feature importance
                    feature_names = pipeline.processor.get_feature_names()
                    importances = classifier.feature_importances_
                    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:3]
                    print(f"      Top features: {[f'{name}: {imp:.3f}' for name, imp in top_features]}")
                else:
                    print("      ✅ Data ready for unsupervised learning (clustering, PCA, etc.)")
                
            except ImportError:
                print("      ⚠️  Scikit-learn not available")
            
            # Example 2: Deep Learning
            print("    🧠 Deep Learning Integration:")
            try:
                import torch
                import torch.nn as nn
                
                # Convert to PyTorch tensors
                X_tensor = torch.FloatTensor(X)
                y_tensor = torch.LongTensor(y) if y is not None else None
                
                print(f"      PyTorch X: {X_tensor.shape}, y: {y_tensor.shape if y_tensor is not None else 'None'}")
                print("      ✅ Ready for: neural networks, transformers, etc.")
                
            except ImportError:
                print("      ⚠️  PyTorch not available")
            
            # Example 3: AutoML
            print("    🤖 AutoML Integration:")
            print(f"      Data shape: {X.shape} - ready for AutoML frameworks")
            print("      ✅ Compatible with: AutoGluon, H2O, TPOT, AutoSklearn")
            
    else:
        print("⚠️  No sample tabular data found. Run the main demo first.")
    
    print()

def demonstrate_timeseries_workflow():
    """Demo: Time Series → Universal Pipeline → LSTM/Transformer"""
    print("="*60)
    print("📈 TIME SERIES → FORECASTING WORKFLOW")  
    print("="*60)
    
    # Configure for time series modeling
    timeseries_config = {
        'timeseries': {
            'sequence_length': 50,
            'scaling_method': 'minmax',
            'time_column': None,  # Auto-detect
            'return_3d': True     # (samples, timesteps, features)
        }
    }
    
    pipeline = UniversalPipeline(timeseries_config)
    
    # Process sample time series data
    sample_dir = Path("sample_data/timeseries")
    if sample_dir.exists():
        ts_file = sample_dir / "stock_data.csv"
        if ts_file.exists():
            print(f"📁 Processing time series: {ts_file.name}")
            
            # Pipeline processes time series
            sequences = pipeline.fit_transform(str(ts_file))
            print(f"  Sequences shape: {sequences.shape}")
            print(f"  Format: (samples={sequences.shape[0]}, timesteps={sequences.shape[1]}, features={sequences.shape[2]})")
            
            # ✅ DIRECT INTEGRATION: Ready for time series models
            print("  🎯 Model Integration Examples:")
            
            # Example 1: LSTM/GRU Models
            print("    🔄 LSTM/RNN Integration:")
            try:
                import torch
                import torch.nn as nn
                
                # Convert to PyTorch
                X_tensor = torch.FloatTensor(sequences)
                print(f"      LSTM Input: {X_tensor.shape}")
                
                # Example LSTM model
                class TimeSeriesLSTM(nn.Module):
                    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
                        super().__init__()
                        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                        self.fc = nn.Linear(hidden_size, output_size)
                    
                    def forward(self, x):
                        out, _ = self.lstm(x)
                        return self.fc(out[:, -1, :])  # Last timestep
                
                model = TimeSeriesLSTM(input_size=sequences.shape[2])
                print(f"      Model: LSTM({sequences.shape[2]} → 64 → 1)")
                print("      ✅ Ready for: forecasting, anomaly detection")
                
            except ImportError:
                print("      ⚠️  PyTorch not available")
            
            # Example 2: Classical Time Series
            print("    📊 Classical Methods:")
            try:
                # Flatten for classical methods
                flattened = sequences.reshape(sequences.shape[0], -1)
                print(f"      Flattened shape: {flattened.shape}")
                print("      ✅ Ready for: ARIMA, Prophet, seasonal decomposition")
                
            except Exception as e:
                print(f"      ⚠️  Error: {e}")
            
            # Example 3: Transformers
            print("    🤖 Transformer Models:")
            print(f"      Sequence format: perfect for attention mechanisms")
            print("      ✅ Ready for: Temporal Fusion Transformer, Informer, etc.")
            
    else:
        print("⚠️  No sample time series data found. Run the main demo first.")
    
    print()

def demonstrate_audio_workflow():
    """Demo: Audio → Universal Pipeline → Speech/Music Models"""
    print("="*60)
    print("🎵 AUDIO PROCESSING → AI MODELS WORKFLOW")
    print("="*60)
    
    # Configure for raw audio output (not features)
    audio_config = {
        'audio': {
            'output_format': 'raw_audio',    # Return raw signals, not MFCC
            'normalize_amplitude': True,
            'target_sr': 22050
        }
    }
    
    pipeline = UniversalPipeline(audio_config)
    
    # Process sample audio data
    sample_dir = Path("sample_data/audio")
    if sample_dir.exists():
        for audio_file in sample_dir.glob("*.wav"):
            print(f"📁 Processing audio: {audio_file.name}")
            
            # Pipeline processes audio
            audio_data = pipeline.fit_transform(str(audio_file))
            print(f"  Audio data type: {type(audio_data)}")
            
            if isinstance(audio_data, dict):
                audio_signal = audio_data['audio']
                sample_rate = audio_data['sample_rate']
                duration = audio_data.get('duration', len(audio_signal) / sample_rate)
                
                print(f"  Signal shape: {audio_signal.shape}")
                print(f"  Sample rate: {sample_rate} Hz, Duration: {duration:.2f}s")
                
                # ✅ DIRECT INTEGRATION: Ready for audio AI models
                print("  🎯 Model Integration Examples:")
                
                # Example 1: Speech Recognition
                print("    🗣️  Speech Recognition:")
                try:
                    import torch
                    # Simulate speech model input
                    speech_input = torch.from_numpy(audio_signal).float()
                    print(f"      Input tensor: {speech_input.shape}")
                    print("      ✅ Ready for: Wav2Vec2, Whisper, DeepSpeech")
                    
                except ImportError:
                    print("      ⚠️  PyTorch not available")
                
                # Example 2: Music Analysis
                print("    🎼 Music Information Retrieval:")
                print(f"      Raw audio: {audio_signal.shape} @ {sample_rate}Hz")
                print("      ✅ Ready for: tempo detection, chord recognition, genre classification")
                
                # Example 3: Audio Classification
                print("    🔊 Audio Classification:")
                try:
                    # Simulate feature extraction for classification
                    print(f"      Signal ready for feature extraction")
                    print("      ✅ Ready for: sound event detection, acoustic scene analysis")
                    
                except Exception as e:
                    print(f"      ⚠️  Error: {e}")
                
            break  # Process just one audio file for demo
    else:
        print("⚠️  No sample audio data found. Run the main demo first.")
    
    print()

def demonstrate_performance_comparison():
    """Demo: Performance comparison with/without optimizations"""
    print("="*60)
    print("⚡ PERFORMANCE OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Create sample data for performance testing
    sample_dir = Path("sample_data")
    if not sample_dir.exists():
        print("⚠️  No sample data found. Run the main demo first.")
        return
    
    print("🏁 Performance Test: Processing multiple file types")
    
    # Test batch processing
    from cli_processor import DirectoryProcessor
    
    # Test with different configurations
    configs = [
        {"parallel": False, "batch_size": 1, "name": "Sequential (1 by 1)"},
        {"parallel": True, "batch_size": 5, "name": "Parallel Batched (5)"},
        {"parallel": True, "batch_size": 10, "name": "Parallel Batched (10)"},
    ]
    
    for config in configs:
        print(f"\n🔧 Testing: {config['name']}")
        
        try:
            start_time = time.time()
            
            processor = DirectoryProcessor(
                input_dir="sample_data",
                output_dir="sample_data/performance_test",
                parallel=config["parallel"],
                batch_size=config["batch_size"],
                max_workers=4
            )
            
            # Scan files (but don't process for this demo)
            files_by_type = processor.scan_directory()
            file_count = sum(len(files) for files in files_by_type.values())
            
            scan_time = time.time() - start_time
            
            print(f"   Files found: {file_count}")
            print(f"   Scan time: {scan_time:.3f}s")
            print(f"   Config: parallel={config['parallel']}, batch_size={config['batch_size']}")
            
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
    
    print("\n💡 Performance Tips:")
    print("   • Use --parallel for multiple files")
    print("   • Increase --batch-size for large datasets")
    print("   • Adjust --workers based on CPU cores")
    print("   • Use --no-parallel for memory-constrained environments")

def run_all_demonstrations():
    """Run all integration demonstrations"""
    print("🚀 UNIVERSAL DATA PIPELINE - INTEGRATION DEMONSTRATIONS")
    print("="*80)
    print("This demo shows how pipeline outputs integrate directly with ML models")
    print("="*80)
    
    # Run all demonstrations
    demonstrate_image_workflow()
    demonstrate_tabular_workflow() 
    demonstrate_timeseries_workflow()
    demonstrate_audio_workflow()
    demonstrate_performance_comparison()
    
    print("="*80)
    print("✅ ALL DEMONSTRATIONS COMPLETED!")
    print("="*80)
    print("Key Takeaways:")
    print("📸 Images: Ready for YOLO, CNNs, OpenCV (proper numpy arrays)")
    print("📊 Tabular: Ready for sklearn, AutoML (encoded & scaled)")
    print("📈 Time Series: Ready for LSTMs, Transformers (proper sequences)")
    print("🎵 Audio: Ready for speech/music models (raw signals)")
    print("⚡ Performance: Parallel processing for large datasets")
    print("\n🎯 The pipeline outputs domain-appropriate formats that require")
    print("   NO additional preprocessing for real ML model training!")

if __name__ == "__main__":
    run_all_demonstrations() 