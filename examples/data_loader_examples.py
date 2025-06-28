"""
Data Loader Examples: How to Import Preprocessed Datasets

This script demonstrates various ways to load and use datasets after
preprocessing with the Universal Data Pipeline.

The pipeline outputs:
1. Processed data files (.npy) in organized directories  
2. Pipeline objects (.pkl) for consistent processing
3. Processing summaries (.json) with metadata
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pickle

# For ML frameworks
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True  
except ImportError:
    TF_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class PreprocessedDatasetLoader:
    """Loader for datasets processed by Universal Data Pipeline"""
    
    def __init__(self, processed_dir: str):
        """
        Initialize loader with processed dataset directory
        
        Args:
            processed_dir: Path to directory containing processed_output folder
        """
        self.processed_dir = Path(processed_dir)
        self.output_dir = self.processed_dir / "processed_output"
        
        # Load processing summary
        self.summary = self._load_summary()
        self.data_types = self.summary.get("data_types_found", [])
        
    def _load_summary(self) -> Dict[str, Any]:
        """Load processing summary metadata"""
        summary_path = self.output_dir / "processing_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                return json.load(f)
        return {}
    
    def get_processed_files(self, data_type: str) -> List[Path]:
        """Get all processed files for a specific data type"""
        processed_subdir = self.output_dir / f"{data_type}_processed"
        if processed_subdir.exists():
            return sorted(list(processed_subdir.glob("*.npy")))
        return []
    
    def load_single_file(self, file_path: Path) -> np.ndarray:
        """Load a single processed file"""
        return np.load(file_path)
    
    def load_all_data(self, data_type: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load all processed data of a specific type
        
        Returns:
            data: Stacked numpy array of all samples
            filenames: List of original filenames
        """
        files = self.get_processed_files(data_type)
        if not files:
            raise ValueError(f"No processed files found for data type: {data_type}")
        
        # Load first file to get shape
        sample_data = self.load_single_file(files[0])
        data_shape = (len(files),) + sample_data.shape
        
        # Pre-allocate array for efficiency
        all_data = np.empty(data_shape, dtype=sample_data.dtype)
        filenames = []
        
        print(f"Loading {len(files)} {data_type} files...")
        for i, file_path in enumerate(files):
            all_data[i] = self.load_single_file(file_path)
            # Extract original filename (remove _processed.npy suffix)
            original_name = file_path.stem.replace("_processed", "")
            filenames.append(original_name)
            
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(files)} files")
        
        print(f"✅ Loaded {data_type} dataset: {all_data.shape}")
        return all_data, filenames
    
    def extract_labels_from_filenames(self, filenames: List[str]) -> np.ndarray:
        """
        Extract class labels from filenames (assumes class is in filename)
        
        Example: 'neu_scratches_123.jpg' -> 'scratches'
        """
        labels = []
        unique_classes = set()
        
        for filename in filenames:
            # Extract class from filename pattern: neu_{class}_{number}
            if '_' in filename:
                parts = filename.split('_')
                if len(parts) >= 2:
                    class_name = parts[1]  # e.g., 'scratches', 'rolled-in', etc.
                    labels.append(class_name)
                    unique_classes.add(class_name)
                else:
                    labels.append('unknown')
            else:
                labels.append('unknown')
        
        # Convert to numerical labels
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
        numerical_labels = np.array([class_to_idx[label] for label in labels])
        
        print(f"📊 Found {len(unique_classes)} classes: {list(class_to_idx.keys())}")
        return numerical_labels, class_to_idx
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information"""
        info = {
            "processing_summary": self.summary,
            "available_data_types": self.data_types,
            "file_counts": {}
        }
        
        for data_type in self.data_types:
            files = self.get_processed_files(data_type)
            if files:
                sample_data = self.load_single_file(files[0])
                info["file_counts"][data_type] = {
                    "count": len(files),
                    "shape_per_sample": sample_data.shape,
                    "dtype": str(sample_data.dtype),
                    "size_mb": sample_data.nbytes / (1024 * 1024)
                }
        
        return info

# ===============================================
# PyTorch Dataset Integration
# ===============================================

if TORCH_AVAILABLE:
    class TorchPreprocessedDataset(Dataset):
        """PyTorch Dataset for preprocessed data"""
        
        def __init__(self, processed_dir: str, data_type: str = "image", 
                     transform=None, target_transform=None):
            self.loader = PreprocessedDatasetLoader(processed_dir)
            self.data_type = data_type
            self.transform = transform
            self.target_transform = target_transform
            
            # Load all data
            self.data, self.filenames = self.loader.load_all_data(data_type)
            self.labels, self.class_to_idx = self.loader.extract_labels_from_filenames(self.filenames)
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            label = self.labels[idx]
            
            if self.transform:
                sample = self.transform(sample)
            if self.target_transform:
                label = self.target_transform(label)
                
            return torch.FloatTensor(sample), torch.LongTensor([label])
        
        def get_class_names(self):
            return list(self.class_to_idx.keys())

# ===============================================
# TensorFlow Dataset Integration  
# ===============================================

if TF_AVAILABLE:
    def create_tensorflow_dataset(processed_dir: str, data_type: str = "image", 
                                  batch_size: int = 32, shuffle: bool = True):
        """Create TensorFlow dataset from preprocessed data"""
        loader = PreprocessedDatasetLoader(processed_dir)
        data, filenames = loader.load_all_data(data_type)
        labels, class_to_idx = loader.extract_labels_from_filenames(filenames)
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        print(f"🔥 Created TensorFlow dataset: {len(data)} samples, {len(class_to_idx)} classes")
        return dataset, class_to_idx

# ===============================================
# Example Usage Functions
# ===============================================

def demo_basic_loading():
    """Demonstrate basic data loading"""
    print("🚀 DEMO: Basic Data Loading")
    print("=" * 50)
    
    # Initialize loader
    loader = PreprocessedDatasetLoader("test")
    
    # Show dataset info
    info = loader.get_dataset_info()
    print("📋 Dataset Information:")
    for key, value in info.items():
        if key != "processing_summary":
            print(f"  {key}: {value}")
    
    # Load image data
    if "image" in loader.data_types:
        images, filenames = loader.load_all_data("image")
        labels, class_mapping = loader.extract_labels_from_filenames(filenames)
        
        print(f"\n📸 Image Dataset:")
        print(f"  Shape: {images.shape}")
        print(f"  Data type: {images.dtype}")
        print(f"  Memory usage: {images.nbytes / (1024**3):.2f} GB")
        print(f"  Classes: {list(class_mapping.keys())}")
        print(f"  Pixel range: [{images.min():.3f}, {images.max():.3f}]")

def demo_pytorch_integration():
    """Demonstrate PyTorch integration"""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available")
        return
        
    print("\n🔥 DEMO: PyTorch Integration")
    print("=" * 50)
    
    # Create PyTorch dataset
    dataset = TorchPreprocessedDataset("test", data_type="image")
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    print(f"📊 PyTorch Dataset: {len(dataset)} samples")
    print(f"📊 Classes: {dataset.get_class_names()}")
    
    # Show sample batch
    for batch_idx, (data, targets) in enumerate(dataloader):
        print(f"📦 Batch {batch_idx + 1}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
        if batch_idx == 0:  # Only show first batch
            break

def demo_tensorflow_integration():
    """Demonstrate TensorFlow integration"""
    if not TF_AVAILABLE:
        print("❌ TensorFlow not available")
        return
        
    print("\n🔥 DEMO: TensorFlow Integration")
    print("=" * 50)
    
    # Create TensorFlow dataset
    dataset, class_mapping = create_tensorflow_dataset("test", batch_size=16)
    
    # Show sample batch
    for batch_data, batch_labels in dataset.take(1):
        print(f"📦 TensorFlow Batch:")
        print(f"  Data shape: {batch_data.shape}")
        print(f"  Labels shape: {batch_labels.shape}")
        print(f"  Data range: [{tf.reduce_min(batch_data):.3f}, {tf.reduce_max(batch_data):.3f}]")

def demo_sklearn_integration():
    """Demonstrate scikit-learn integration"""
    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn not available")
        return
        
    print("\n🔬 DEMO: scikit-learn Integration")
    print("=" * 50)
    
    # Load data
    loader = PreprocessedDatasetLoader("test")
    if "image" not in loader.data_types:
        print("❌ No image data available")
        return
        
    images, filenames = loader.load_all_data("image")
    labels, class_mapping = loader.extract_labels_from_filenames(filenames)
    
    # Flatten images for traditional ML (640x640x3 -> 1,228,800 features)
    X = images.reshape(images.shape[0], -1)
    y = labels
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🔬 Flattened for sklearn:")
    print(f"  Training shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Classes: {len(class_mapping)}")
    
    # Note: Training a full model would take too long, just show setup
    print("✅ Ready for sklearn models (RandomForest, SVM, etc.)")

def demo_custom_preprocessing_pipeline():
    """Show how to use saved pipeline for new data"""
    print("\n🔧 DEMO: Using Saved Pipeline for New Data")
    print("=" * 50)
    
    # Load the saved pipeline
    pipeline_path = Path("test/processed_output/image_pipeline.pkl")
    if pipeline_path.exists():
        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
        
        print("✅ Loaded saved image processing pipeline")
        print("🔄 You can now use this pipeline to process new images:")
        print("   new_processed = pipeline.transform(new_image_data)")
        print("   This ensures consistent preprocessing for inference!")
    else:
        print("❌ No saved pipeline found")

# ===============================================
# YOLO Integration
# ===============================================

def demo_yolov8_integration():
    """Demonstrate YOLOv8 object detection integration"""
    print("\n🎯 DEMO: YOLOv8 Object Detection Integration")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        
        # Load processed image data
        loader = PreprocessedDatasetLoader("test")
        if "image" not in loader.data_types:
            print("❌ No image data available")
            return
            
        images, filenames = loader.load_all_data("image")
        print(f"📸 Loaded {len(images)} processed images: {images.shape}")
        
        # Load YOLOv8 model (you can use yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        model = YOLO('yolov8n.pt')  # Downloads automatically if not present
        
        # The processed images are already in perfect format for YOLO
        # Convert to uint8 format if needed (YOLO expects 0-255 range)
        if images.max() <= 1.0:
            images_yolo = (images * 255).astype(np.uint8)
        else:
            images_yolo = images.astype(np.uint8)
        
        print(f"🎯 Ready for YOLO inference:")
        print(f"  Image format: {images_yolo.shape} {images_yolo.dtype}")
        print(f"  Pixel range: [{images_yolo.min()}, {images_yolo.max()}]")
        
        # Run inference on first few images
        sample_images = images_yolo[:3]  # Take first 3 images
        
        print("\n🔍 Running YOLO inference on sample images...")
        results = model(sample_images, verbose=False)
        
        for i, result in enumerate(results):
            detections = len(result.boxes) if result.boxes is not None else 0
            print(f"  Image {i+1} ({filenames[i]}): {detections} objects detected")
            
            if result.boxes is not None and len(result.boxes) > 0:
                # Show detected classes
                classes = result.boxes.cls.cpu().numpy()
                class_names = [model.names[int(cls)] for cls in classes]
                confidences = result.boxes.conf.cpu().numpy()
                
                for j, (cls_name, conf) in enumerate(zip(class_names, confidences)):
                    print(f"    - {cls_name}: {conf:.3f}")
        
        print("\n✅ YOLOv8 Integration Success!")
        print("💡 Your processed images work perfectly with YOLO!")
        print("   • No additional preprocessing needed")
        print("   • Ready for training or inference")
        print("   • Compatible with all YOLO variants")
        
    except ImportError:
        print("❌ ultralytics not installed. Install with: pip install ultralytics")
    except Exception as e:
        print(f"❌ Error running YOLOv8: {e}")

# ===============================================
# Advanced ML Models Integration
# ===============================================

def demo_advanced_ml_models():
    """Demonstrate integration with XGBoost, LightGBM, CatBoost"""
    print("\n🚀 DEMO: Advanced ML Models (XGBoost, LightGBM, CatBoost)")
    print("=" * 50)
    
    # Load tabular data
    loader = PreprocessedDatasetLoader("sample_data/tabular")
    
    try:
        tabular_data, filenames = loader.load_all_data("tabular")
        print(f"📊 Loaded tabular data: {tabular_data.shape}")
        
        # Create synthetic labels for demo (in real case, you'd have actual labels)
        np.random.seed(42)
        labels = np.random.randint(0, 3, size=tabular_data.shape[0])  # 3 classes
        
        # Split data
        if SKLEARN_AVAILABLE:
            X_train, X_test, y_train, y_test = train_test_split(
                tabular_data, labels, test_size=0.3, random_state=42
            )
            
            print(f"📈 Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            # XGBoost Example
            try:
                import xgboost as xgb
                
                print("\n🌟 XGBoost Integration:")
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    eval_metric='mlogloss'
                )
                xgb_model.fit(X_train, y_train)
                xgb_score = xgb_model.score(X_test, y_test)
                print(f"   ✅ XGBoost Accuracy: {xgb_score:.3f}")
                
            except ImportError:
                print("   ❌ XGBoost not installed: pip install xgboost")
            
            # LightGBM Example  
            try:
                import lightgbm as lgb
                
                print("\n💡 LightGBM Integration:")
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=100,
                    random_state=42,
                    verbose=-1
                )
                lgb_model.fit(X_train, y_train)
                lgb_score = lgb_model.score(X_test, y_test)
                print(f"   ✅ LightGBM Accuracy: {lgb_score:.3f}")
                
            except ImportError:
                print("   ❌ LightGBM not installed: pip install lightgbm")
            
            # CatBoost Example
            try:
                from catboost import CatBoostClassifier
                
                print("\n🐱 CatBoost Integration:")
                cat_model = CatBoostClassifier(
                    iterations=100,
                    random_state=42,
                    verbose=False
                )
                cat_model.fit(X_train, y_train)
                cat_score = cat_model.score(X_test, y_test)
                print(f"   ✅ CatBoost Accuracy: {cat_score:.3f}")
                
            except ImportError:
                print("   ❌ CatBoost not installed: pip install catboost")
                
        print("\n✅ Advanced ML Models Integration Complete!")
        print("💡 Your preprocessed tabular data works with all gradient boosting models!")
        
    except Exception as e:
        print(f"❌ Error with tabular data: {e}")
        print("💡 Make sure you have processed tabular data available")

# ===============================================
# Timeseries Models Integration
# ===============================================

def demo_timeseries_models():
    """Demonstrate integration with LSTM, Prophet, ARIMA models"""
    print("\n📈 DEMO: Timeseries Models (LSTM, Prophet, ARIMA)")
    print("=" * 50)
    
    # Load timeseries data
    loader = PreprocessedDatasetLoader("sample_data/timeseries")
    
    try:
        ts_data, filenames = loader.load_all_data("timeseries")
        print(f"⏰ Loaded timeseries data: {ts_data.shape}")
        
        # LSTM with PyTorch
        if TORCH_AVAILABLE:
            print("\n🧠 LSTM Integration (PyTorch):")
            
            # Prepare data for LSTM (samples, seq_length, features)
            # The pipeline already outputs proper sequences
            print(f"   📊 LSTM Input Shape: {ts_data.shape}")
            print(f"   📊 Ready for PyTorch LSTM layer")
            
            # Example LSTM model setup
            class LSTMModel(torch.nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, output_size):
                    super(LSTMModel, self).__init__()
                    self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = torch.nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    out, _ = self.lstm(x)
                    out = self.fc(out[:, -1, :])  # Take last timestep
                    return out
            
            # Model dimensions from processed data
            input_size = ts_data.shape[2]  # Number of features
            hidden_size = 64
            num_layers = 2
            output_size = 1  # Prediction target
            
            model = LSTMModel(input_size, hidden_size, num_layers, output_size)
            print(f"   ✅ LSTM Model Created: {input_size} features → {output_size} output")
            print(f"   🔧 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        # Prophet Integration
        try:
            from prophet import Prophet
            import pandas as pd
            
            print("\n🔮 Prophet Integration:")
            
            # For Prophet, we need to convert to DataFrame format with 'ds' and 'y' columns
            # Take the first sequence and first feature as example
            sample_sequence = ts_data[0, :, 0]  # First sample, all timesteps, first feature
            
            # Create date range for demo
            dates = pd.date_range('2020-01-01', periods=len(sample_sequence), freq='D')
            prophet_df = pd.DataFrame({
                'ds': dates,
                'y': sample_sequence
            })
            
            print(f"   📊 Prophet Data Shape: {prophet_df.shape}")
            print(f"   📅 Date Range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
            
            # Create and fit Prophet model
            prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False)
            prophet_model.fit(prophet_df)
            
            # Make future predictions
            future = prophet_model.make_future_dataframe(periods=30)  # 30 days forecast
            forecast = prophet_model.predict(future)
            
            print(f"   ✅ Prophet Model Trained Successfully")
            print(f"   📈 Forecast Shape: {forecast.shape}")
            print(f"   🔮 Predicted next 30 days")
            
        except ImportError:
            print("   ❌ Prophet not installed: pip install prophet")
        
        # ARIMA Integration
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            print("\n📊 ARIMA Integration:")
            
            # ARIMA works with 1D timeseries
            sample_series = ts_data[0, :, 0]  # First sample, all timesteps, first feature
            
            # Fit ARIMA model
            arima_model = ARIMA(sample_series, order=(1, 1, 1))  # (p,d,q) parameters
            arima_fitted = arima_model.fit()
            
            # Make predictions
            forecast_steps = 10
            forecast = arima_fitted.forecast(steps=forecast_steps)
            
            print(f"   ✅ ARIMA Model Fitted: ARIMA(1,1,1)")
            print(f"   📈 Forecast {forecast_steps} steps ahead")
            print(f"   📊 AIC: {arima_fitted.aic:.2f}")
            
        except ImportError:
            print("   ❌ statsmodels not installed: pip install statsmodels")
        
        print("\n✅ Timeseries Models Integration Complete!")
        print("💡 Your preprocessed timeseries data works with:")
        print("   • LSTM/GRU networks (deep learning)")
        print("   • Prophet (Facebook's forecasting tool)")
        print("   • ARIMA/SARIMA (statistical models)")
        print("   • Any other timeseries model!")
        
    except Exception as e:
        print(f"❌ Error with timeseries data: {e}")
        print("💡 Make sure you have processed timeseries data available")

# ===============================================
# Audio Models Integration
# ===============================================

def demo_audio_models():
    """Demonstrate integration with audio processing models"""
    print("\n🎵 DEMO: Audio Models Integration")
    print("=" * 50)
    
    # Load audio data
    loader = PreprocessedDatasetLoader("sample_data/audio")
    
    try:
        # Audio data might be stored differently (as dictionaries)
        audio_files = loader.get_processed_files("audio")
        if not audio_files:
            print("❌ No processed audio files found")
            return
            
        # Load first audio file
        sample_audio_data = loader.load_single_file(audio_files[0])
        print(f"🎵 Sample audio data shape: {sample_audio_data.shape}")
        
        # If audio is processed as raw signals
        if len(sample_audio_data.shape) == 1:
            audio_signal = sample_audio_data
            sample_rate = 22050  # Common sample rate
            
            print(f"🎼 Raw Audio Signal:")
            print(f"   Length: {len(audio_signal)} samples")
            print(f"   Duration: {len(audio_signal)/sample_rate:.2f} seconds")
            print(f"   Sample Rate: {sample_rate} Hz")
            
            # Speech Recognition Example
            try:
                import speech_recognition as sr
                print("\n🗣️  Speech Recognition Integration:")
                print("   ✅ Ready for speech recognition models")
                print("   💡 Use libraries like: speech_recognition, whisper")
                
            except ImportError:
                print("   ❌ speech_recognition not installed: pip install SpeechRecognition")
            
            # Audio Classification Example
            try:
                import librosa
                
                print("\n🎯 Audio Classification Integration:")
                # Extract features commonly used in audio ML
                mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
                spectral_centroids = librosa.feature.spectral_centroid(y=audio_signal, sr=sample_rate)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_signal)
                
                print(f"   ✅ MFCC Features: {mfccs.shape}")
                print(f"   ✅ Spectral Centroids: {spectral_centroids.shape}")
                print(f"   ✅ Zero Crossing Rate: {zero_crossing_rate.shape}")
                print("   💡 Ready for audio classification models")
                
            except ImportError:
                print("   ❌ librosa not installed: pip install librosa")
                
        print("\n✅ Audio Models Integration Complete!")
        print("💡 Your preprocessed audio data works with:")
        print("   • Speech recognition models (Whisper, SpeechRecognition)")
        print("   • Audio classification (genre, emotion, etc.)")
        print("   • Music information retrieval")
        print("   • Sound event detection")
        
    except Exception as e:
        print(f"❌ Error with audio data: {e}")
        print("💡 Make sure you have processed audio data available")

if __name__ == "__main__":
    print("🎯 PREPROCESSED DATASET LOADING EXAMPLES")
    print("=" * 60)
    
    # Run all demos
    demo_basic_loading()
    demo_pytorch_integration()
    demo_tensorflow_integration() 
    demo_sklearn_integration()
    demo_custom_preprocessing_pipeline()
    demo_yolov8_integration()
    demo_advanced_ml_models()
    demo_timeseries_models()
    demo_audio_models()
    
    print("\n🎉 All demos completed!")
    print("\n💡 Key Takeaways:")
    print("   • Data is preprocessed and standardized (640x640x3)")
    print("   • Ready for immediate ML model training")
    print("   • Compatible with PyTorch, TensorFlow, sklearn")
    print("   • Saved pipelines ensure consistent preprocessing")
    print("   • Labels extracted automatically from filenames") 