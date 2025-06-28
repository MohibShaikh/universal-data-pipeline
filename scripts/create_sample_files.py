#!/usr/bin/env python3
"""
Script to create sample files for testing the universal data pipeline
"""

import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path
from PIL import Image
import json
import librosa
from datetime import datetime, timedelta

def create_sample_tabular():
    """Create sample tabular data (CSV, Excel, JSON)"""
    print("Creating sample tabular files...")
    
    # Sample CSV data
    data = {
        'id': range(1, 101),
        'name': [f'Person_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'salary': np.random.normal(50000, 15000, 100),
        'department': np.random.choice(['HR', 'Engineering', 'Sales', 'Marketing'], 100),
        'rating': np.random.uniform(1, 5, 100)
    }
    df = pd.DataFrame(data)
    
    # Save as CSV
    df.to_csv('../data/sample_data/tabular/employees.csv', index=False)
    
    # Save as Excel
    df.to_excel('../data/sample_data/tabular/employees.xlsx', index=False)
    
    # Save as Parquet
    df.to_parquet('../data/sample_data/tabular/employees.parquet', index=False)
    
    # Create a JSON file
    json_data = df.to_dict('records')
    with open('../data/sample_data/tabular/employees.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print("✅ Tabular files created: employees.csv, employees.xlsx, employees.parquet, employees.json")

def create_sample_images():
    """Create sample image files"""
    print("Creating sample image files...")
    
    # Create synthetic images
    np.random.seed(42)
    
    # RGB image
    rgb_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(rgb_image).save('../data/sample_data/image/sample_rgb.jpg')
    Image.fromarray(rgb_image).save('../data/sample_data/image/sample_rgb.png')
    
    # Grayscale image
    gray_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    Image.fromarray(gray_image, mode='L').save('../data/sample_data/image/sample_gray.jpg')
    
    # Pattern image
    pattern = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            pattern[i, j] = [i % 256, j % 256, (i + j) % 256]
    Image.fromarray(pattern).save('../data/sample_data/image/pattern.bmp')
    
    print("✅ Image files created: sample_rgb.jpg, sample_rgb.png, sample_gray.jpg, pattern.bmp")

def create_sample_video():
    """Create sample video files"""
    print("Creating sample video files...")
    
    # Create a simple video with moving rectangle
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('../data/sample_data/video/sample_video.mp4', fourcc, 30.0, (640, 480))
    
    for frame_num in range(90):  # 3 seconds at 30fps
        # Create frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Moving rectangle
        x = int((frame_num / 90) * 500) + 50
        y = 200
        cv2.rectangle(frame, (x, y), (x + 50, y + 50), (0, 255, 0), -1)
        
        # Add frame number
        cv2.putText(frame, f'Frame {frame_num}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    
    print("✅ Video file created: sample_video.mp4")

def create_sample_audio():
    """Create sample audio files"""
    print("Creating sample audio files...")
    
    # Create synthetic audio signals
    duration = 5  # seconds
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Sine wave
    frequency = 440  # A4 note
    sine_wave = np.sin(2 * np.pi * frequency * t)
    
    # Add some harmonics for richer sound
    sine_wave += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
    sine_wave += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)
    
    # Normalize
    sine_wave = sine_wave / np.max(np.abs(sine_wave))
    
    # Save as WAV using soundfile (librosa dependency)
    import soundfile as sf
    sf.write('../data/sample_data/audio/sine_wave.wav', sine_wave, sample_rate)
    
    # Create white noise
    noise = np.random.normal(0, 0.1, len(t))
    sf.write('../data/sample_data/audio/white_noise.wav', noise, sample_rate)
    
    # Create chirp signal (frequency sweep)
    f0, f1 = 100, 2000  # Start and end frequencies
    chirp = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)
    sf.write('../data/sample_data/audio/chirp.wav', chirp, sample_rate)
    
    print("✅ Audio files created: sine_wave.wav, white_noise.wav, chirp.wav")

def create_sample_timeseries():
    """Create sample time series data"""
    print("Creating sample timeseries files...")
    
    # Create time series data
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365 * 3)]  # 3 years of daily data
    
    # Generate synthetic stock price data
    np.random.seed(42)
    price = 100  # Starting price
    prices = [price]
    
    for i in range(1, len(dates)):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
        price *= (1 + change)
        prices.append(price)
    
    # Add some seasonality
    seasonal = [10 * np.sin(2 * np.pi * i / 365) for i in range(len(dates))]
    prices = np.array(prices) + seasonal
    
    # Create DataFrame
    ts_data = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.poisson(1000000, len(dates)),
        'high': prices * (1 + np.random.uniform(0, 0.05, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.05, len(dates))),
        'returns': np.concatenate([[0], np.diff(np.log(prices))])
    })
    
    ts_data['high'] = np.maximum(ts_data['price'], ts_data['high'])
    ts_data['low'] = np.minimum(ts_data['price'], ts_data['low'])
    
    # Save time series data
    ts_data.to_csv('../data/sample_data/timeseries/stock_data.csv', index=False)
    
    # Create sensor data time series
    timestamps = pd.date_range(start='2023-01-01', periods=10000, freq='1T')  # 1 minute intervals
    sensor_data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': 20 + 5 * np.sin(2 * np.pi * np.arange(len(timestamps)) / (24 * 60)) + np.random.normal(0, 1, len(timestamps)),
        'humidity': 50 + 10 * np.cos(2 * np.pi * np.arange(len(timestamps)) / (12 * 60)) + np.random.normal(0, 2, len(timestamps)),
        'pressure': 1013 + np.random.normal(0, 5, len(timestamps))
    })
    
    sensor_data.to_csv('../data/sample_data/timeseries/sensor_data.csv', index=False)
    
    print("✅ Timeseries files created: stock_data.csv, sensor_data.csv")

def main():
    """Create all sample files"""
    print("Creating sample files for Universal Data Pipeline testing...")
    print("=" * 60)
    
    # Ensure directories exist
    os.makedirs('../data/sample_data/tabular', exist_ok=True)
    os.makedirs('../data/sample_data/image', exist_ok=True)
    os.makedirs('../data/sample_data/video', exist_ok=True)
    os.makedirs('../data/sample_data/audio', exist_ok=True)
    os.makedirs('../data/sample_data/timeseries', exist_ok=True)
    
    # Create sample files for each data type
    create_sample_tabular()
    create_sample_images()
    create_sample_video()
    create_sample_audio()
    create_sample_timeseries()
    
    print("=" * 60)
    print("✅ All sample files created successfully!")
    print("\nSample data structure:")
    for root, dirs, files in os.walk('sample_data'):
        level = root.replace('sample_data', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')

if __name__ == "__main__":
    main() 