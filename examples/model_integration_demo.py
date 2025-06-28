#!/usr/bin/env python3
"""
Model Integration Demo
Shows how Universal Pipeline outputs feed into real ML/CV models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
from pathlib import Path

from universal_pipeline import UniversalPipeline

class ProcessedImageDataset(Dataset):
    """PyTorch Dataset for processed images - feeds directly into YOLO/CNN"""
    def __init__(self, processed_images_dir, labels=None):
        self.image_files = list(Path(processed_images_dir).glob("*.npy"))
        self.labels = labels
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load processed image (already cleaned by pipeline)
        image = np.load(self.image_files[idx])
        
        # Convert to tensor for PyTorch models
        if len(image.shape) == 3:  # (H, W, C)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (C, H, W)
        
        if self.labels is not None:
            return image, torch.tensor(self.labels[idx])
        return image

class ProcessedAudioDataset(Dataset):
    """PyTorch Dataset for processed audio - feeds into speech/audio models"""
    def __init__(self, processed_audio_dir):
        self.audio_files = list(Path(processed_audio_dir).glob("*.npy"))
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load processed audio data (raw audio from pipeline)
        audio_data = np.load(self.audio_files[idx], allow_pickle=True).item()
        
        # Extract audio array and metadata
        audio = torch.from_numpy(audio_data['audio']).float()
        sample_rate = audio_data['sample_rate']
        
        return audio, sample_rate

def demo_yolo_integration():
    """Demo: Using processed images with YOLO-style object detection"""
    print("🎯 YOLO OBJECT DETECTION - Using Processed Images")
    print("=" * 60)
    
    # 1. Process images with Universal Pipeline
    config = {
        'image': {
            'preserve_original_size': False,  # YOLO needs specific size
            'target_size': (640, 640),        # YOLO input size
            'output_format': 'numpy',
            'ensure_rgb': True
        }
    }
    
    pipeline = UniversalPipeline(config)
    
    # Simulate processing an image
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_image = pipeline.fit_transform(sample_image)
    
    print(f"✓ Image processed: {processed_image.shape}")
    print(f"✓ Ready for YOLO input: (batch_size, 3, 640, 640)")
    
    # 2. Create YOLO-compatible tensor
    yolo_input = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    print(f"✓ YOLO tensor shape: {yolo_input.shape}")
    
    # 3. This would go into a YOLO model like:
    print("✓ Usage: model(yolo_input) → detections")
    print()

def demo_cnn_training():
    """Demo: Training a CNN with processed images"""
    print("🖼️  CNN TRAINING - Using Processed Images")
    print("=" * 60)
    
    # Simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((7, 7))
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
            
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # Create model and show how processed data feeds in
    model = SimpleCNN()
    
    # Simulate processed image dataset
    processed_images_dir = "sample_data/image/processed_output/image_processed"
    if Path(processed_images_dir).exists():
        dataset = ProcessedImageDataset(processed_images_dir)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        print(f"✓ Dataset created from processed images")
        print(f"✓ Ready for training: for batch in dataloader: output = model(batch)")
    else:
        print(f"✓ Would create dataset from: {processed_images_dir}")
    print()

def demo_sklearn_integration():
    """Demo: Using processed tabular data with sklearn"""
    print("📊 SKLEARN ML - Using Processed Tabular Data")
    print("=" * 60)
    
    # 1. Process tabular data
    config = {
        'tabular': {
            'target_column': 'target',
            'scaling_method': 'standard',
            'return_dataframe': False  # Get arrays for sklearn
        }
    }
    
    # Create sample data
    data = {
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.randint(0, 2, 1000)
    }
    df = pd.DataFrame(data)
    
    pipeline = UniversalPipeline(config)
    X, y = pipeline.fit_transform(df)
    
    print(f"✓ Processed data: X{X.shape}, y{y.shape}")
    
    # 2. Train sklearn model directly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Model trained on processed data")
    print(f"✓ Accuracy: {accuracy:.3f}")
    print(f"✓ Ready for production: model.predict(new_processed_data)")
    print()

def demo_audio_model_integration():
    """Demo: Using processed audio with speech/audio models"""
    print("🎵 AUDIO MODEL - Using Processed Audio Data")
    print("=" * 60)
    
    # 1. Process audio with Universal Pipeline
    config = {
        'audio': {
            'output_format': 'raw_audio',  # Get clean raw audio
            'normalize_amplitude': True,
            'mono_conversion': True
        }
    }
    
    pipeline = UniversalPipeline(config)
    
    # Simulate audio processing
    print("✓ Audio processed by pipeline: clean signals with metadata")
    print("✓ Output: {'audio': array, 'sample_rate': int, 'duration': float}")
    
    # 2. This feeds into audio models like:
    print("✓ Speech Recognition: whisper_model(audio_array)")
    print("✓ Audio Classification: audio_classifier(audio_features)")
    print("✓ Music Analysis: librosa.feature.* on clean audio")
    print()

def demo_timeseries_lstm():
    """Demo: Using processed time series with LSTM"""
    print("📈 LSTM TRAINING - Using Processed Time Series")
    print("=" * 60)
    
    # Simple LSTM model
    class TimeSeriesLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])  # Use last timestep
            return output
    
    # Create time series data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    ts_data = pd.DataFrame({
        'date': dates,
        'price': np.cumsum(np.random.randn(1000)) + 100,
        'volume': np.random.poisson(1000, 1000),
        'target': np.random.randn(1000)
    })
    
    # Process with Universal Pipeline
    config = {
        'timeseries': {
            'time_column': 'date',
            'sequence_length': 30,  # 30-day sequences
            'scaling_method': 'standard'
        }
    }
    
    pipeline = UniversalPipeline(config)
    sequences = pipeline.fit_transform(ts_data)
    targets = np.random.randn(sequences.shape[0])  # Mock targets for demo
    
    print(f"✓ Time series processed: {sequences.shape}")
    print(f"✓ Ready for LSTM: model(torch.tensor(sequences))")
    
    # Convert to tensors and show model usage
    X_tensor = torch.from_numpy(sequences).float()
    y_tensor = torch.from_numpy(targets).float()
    
    model = TimeSeriesLSTM(input_size=sequences.shape[2])
    print(f"✓ LSTM model created for {sequences.shape[2]} features")
    print(f"✓ Training: loss = criterion(model(X_tensor), y_tensor)")
    print()

def demo_complete_pipeline():
    """Demo: Complete pipeline from raw data to model training"""
    print("🔄 COMPLETE PIPELINE - Raw Data → Processing → Model Training")
    print("=" * 70)
    
    print("1️⃣  RAW DATA:")
    print("   • Images: jpg/png files")
    print("   • Audio: wav/mp3 files") 
    print("   • Tabular: csv/excel files")
    print("   • Time Series: temporal csv data")
    print()
    
    print("2️⃣  UNIVERSAL PIPELINE PROCESSING:")
    print("   • Auto-detect data type")
    print("   • Apply domain-appropriate preprocessing")
    print("   • Output in model-ready formats")
    print()
    
    print("3️⃣  MODEL INTEGRATION:")
    print("   • Images → PyTorch Dataset → YOLO/CNN training")
    print("   • Audio → Raw signals → Speech/Audio models")
    print("   • Tabular → sklearn arrays → ML algorithms")
    print("   • Time Series → Sequences → LSTM/Transformer")
    print()
    
    print("4️⃣  PRODUCTION DEPLOYMENT:")
    print("   • Save trained models")
    print("   • Save preprocessing pipelines")
    print("   • New data: pipeline.transform() → model.predict()")
    print()

def main():
    """Run all integration demos"""
    print("🚀 UNIVERSAL PIPELINE → MODEL INTEGRATION DEMOS")
    print("=" * 70)
    print("How processed data feeds into real ML/CV models")
    print("=" * 70)
    print()
    
    demo_yolo_integration()
    demo_cnn_training()
    demo_sklearn_integration()
    demo_audio_model_integration()
    demo_timeseries_lstm()
    demo_complete_pipeline()
    
    print("🎉 SUMMARY:")
    print("✓ Pipeline outputs are directly compatible with:")
    print("  • PyTorch models (images, audio, time series)")
    print("  • sklearn algorithms (tabular data)")
    print("  • YOLO/object detection (processed images)")
    print("  • Speech recognition (clean audio)")
    print("  • Time series forecasting (LSTM/Transformer)")

if __name__ == "__main__":
    main() 