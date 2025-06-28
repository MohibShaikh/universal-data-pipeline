# Universal Data Pipeline

A comprehensive, automated data preprocessing pipeline that detects data types and applies appropriate preprocessing for machine learning models.

## 🚀 Features

- **Automatic Data Type Detection**: Automatically identifies whether your data is tabular, images, videos, audio, or time series
- **Smart Preprocessing**: Applies data-type-specific preprocessing automatically
- **Ready for ML**: Output is immediately usable for model training (YOLO, PyTorch, sklearn, etc.)
- **Highly Configurable**: Customize preprocessing for your specific needs
- **Multiple Data Formats**: Supports CSV, Excel, Parquet, JSON, images, videos, audio files
- **Batch Processing**: Process multiple files at once
- **Pipeline Persistence**: Save and load fitted pipelines
- **Model Integration**: Direct integration with YOLOv8, XGBoost, LSTM, Prophet, and more

## 📁 Project Structure

```
universal-data-pipeline/
├── universal_pipeline/          # Core package
│   ├── __init__.py
│   ├── pipeline.py             # Main pipeline class
│   ├── data_detector.py        # Data type detection
│   └── processors/             # Data type processors
│       ├── base_processor.py
│       ├── tabular_processor.py
│       ├── image_processor.py
│       ├── audio_processor.py
│       ├── video_processor.py
│       └── timeseries_processor.py
├── examples/                    # Usage examples and demos
│   ├── basic_usage.py
│   ├── data_loader_examples.py
│   ├── yolov8_integration_demo.py
│   ├── model_integration_demo.py
│   └── complete_workflow_demo.py
├── scripts/                     # Utility scripts
│   ├── cli_processor.py
│   ├── create_sample_files.py
│   ├── quick_process.py
│   └── analyze_dataset.py
├── tests/                       # Test files
├── configs/                     # Configuration files
├── models/                      # Trained model files
├── data/                        # Sample and test data
│   ├── sample_data/
│   ├── test_images/
│   └── demo_split_dataset/
├── logs/                        # Log files
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 📋 Supported Data Types

| Data Type | File Formats | Preprocessing Features |
|-----------|-------------|----------------------|
| **Tabular** | CSV, Excel, Parquet, JSON | Scaling, encoding, imputation, feature selection |
| **Images** | JPG, PNG, BMP, TIFF, WebP | Resizing, normalization, augmentation |
| **Videos** | MP4, AVI, MOV, MKV | Frame extraction, resizing, normalization |
| **Audio** | WAV, MP3, FLAC, AAC | Feature extraction (MFCC, spectrograms) |
| **Time Series** | CSV, Excel with datetime | Sequence creation, scaling, temporal features |

## 🛠️ Installation

### Basic Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/universal-data-pipeline](https://github.com/MohibShaikh/universal-data-pipeline.git)
cd universal-data-pipeline

# Install basic dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Install development dependencies
pip install -e ".[dev]"
```

### Using Package Manager

```bash
# Install as package (when published)
pip install universal-data-pipeline

# With all optional dependencies
pip install universal-data-pipeline[all]
```

## 🎯 Quick Start

### Basic Usage - Automatic Detection

```python
from universal_pipeline import UniversalPipeline

# Create pipeline
pipeline = UniversalPipeline()

# Process any data type automatically
processed_data = pipeline.fit_transform("path/to/your/data.csv")
print(f"Detected type: {pipeline.detected_type}")
```

### Tabular Data Example

```python
from universal_pipeline import UniversalPipeline

# Configure for tabular data
config = {
    'tabular': {
        'target_column': 'target',
        'scaling_method': 'standard',
        'categorical_encoding': 'onehot'
    }
}

pipeline = UniversalPipeline(config)
X, y = pipeline.fit_transform("data.csv")

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

### Image Data Example

```python
import numpy as np
from universal_pipeline import UniversalPipeline

# Configure for images
config = {
    'image': {
        'image_size': (224, 224),
        'normalize': True,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}

pipeline = UniversalPipeline(config)

# Process single image
processed = pipeline.fit_transform("image.jpg")

# Process image array
image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
processed = pipeline.fit_transform(image_array)
```

### Time Series Example

```python
from universal_pipeline import UniversalPipeline

config = {
    'timeseries': {
        'time_column': 'timestamp',
        'sequence_length': 30,
        'scaling_method': 'minmax'
    }
}

pipeline = UniversalPipeline(config)
sequences = pipeline.fit_transform("timeseries_data.csv")
print(f"Sequences shape: {sequences.shape}")  # (samples, timesteps, features)
```

### Audio Data Example

```python
from universal_pipeline import UniversalPipeline

config = {
    'audio': {
        'feature_type': 'mfcc',
        'n_mels': 128,
        'sample_rate': 22050
    }
}

pipeline = UniversalPipeline(config)
features = pipeline.fit_transform("audio_file.wav")
```

## 🔧 Configuration Options

### Tabular Data Configuration

```python
tabular_config = {
    'scaling_method': 'standard',  # standard, minmax, none
    'categorical_encoding': 'onehot',  # onehot, label
    'handle_missing': True,
    'target_column': 'target',
    'return_dataframe': False
}
```

### Image Data Configuration

```python
image_config = {
    'image_size': (224, 224),
    'channels': 3,
    'normalize': True,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'return_tensor': True
}
```

### Time Series Configuration

```python
timeseries_config = {
    'sequence_length': 100,
    'scaling_method': 'standard',
    'time_column': 'timestamp',
    'target_column': None,
    'step_size': 1,
    'return_3d': True
}
```

### Audio Configuration

```python
audio_config = {
    'sample_rate': 22050,
    'duration': 30,
    'feature_type': 'mfcc',  # mfcc, melspectrogram, stft
    'n_mels': 128,
    'normalize': True
}
```

### Video Configuration

```python
video_config = {
    'frame_size': (224, 224),
    'num_frames': 16,
    'channels': 3,
    'normalize': True
}
```

## 🔍 Advanced Features

### Data Type Detection

```python
from universal_pipeline import UniversalPipeline

pipeline = UniversalPipeline()

# Detect data type
data_type = pipeline.detect_data_type("your_data.csv")
print(f"Detected: {data_type}")

# Get detailed information
info = pipeline.get_data_info("your_data.csv")
print(info)
```

### Batch Processing

```python
from universal_pipeline import UniversalPipeline

pipeline = UniversalPipeline()

# Process multiple files
file_list = ["data1.csv", "data2.csv", "data3.csv"]
results = pipeline.process_batch(file_list)
```

### Pipeline Persistence

```python
from universal_pipeline import UniversalPipeline

# Fit and save pipeline
pipeline = UniversalPipeline()
pipeline.fit_transform("training_data.csv")
pipeline.save_pipeline("my_pipeline.pkl")

# Load and use pipeline
new_pipeline = UniversalPipeline()
new_pipeline.load_pipeline("my_pipeline.pkl")
processed = new_pipeline.transform("new_data.csv")
```

### Custom Configuration per Data Type

```python
config = {
    'tabular': {
        'scaling_method': 'minmax',
        'target_column': 'label'
    },
    'image': {
        'image_size': (256, 256),
        'normalize': True
    },
    'timeseries': {
        'sequence_length': 50,
        'scaling_method': 'standard'
    }
}

pipeline = UniversalPipeline(config)
```

## 🎮 Running Examples and Scripts

### Core Examples
```bash
# Basic pipeline usage
python examples/basic_usage.py

# Comprehensive integration examples  
python examples/integration_examples.py

# Complete workflow demonstration
python examples/complete_workflow_demo.py

# Advanced data loader patterns
python examples/data_loader_examples.py
```

### Model Integration Examples
```bash
# YOLOv8 object detection integration
python examples/yolov8_integration_demo.py

# ML model training with processed data
python examples/real_model_training_example.py

# Model compatibility demonstrations
python examples/universal_model_compatibility.py

# Complete model integration showcase
python examples/model_integration_demo.py
```

### Utility Scripts
```bash
# Process data via command line
python scripts/cli_processor.py path/to/data/

# Quick data processing
python scripts/quick_process.py

# Analyze dataset characteristics
python scripts/analyze_dataset.py data/sample_data/

# Create sample datasets for testing
python scripts/create_sample_files.py

# Smart dataset handling and splitting
python scripts/smart_dataset_handler.py
```

### Console Commands (after installation)
```bash
# Process data directories
universal-pipeline path/to/data/

# Quick processing shortcut
up-process data/

# Analyze datasets
up-analyze data/sample_data/
```

## 📊 Example Output

### Tabular Data
```
Input: CSV file (1000 rows, 4 columns)
Output: 
- X: (1000, 6) - scaled numerical + encoded categorical features
- y: (1000,) - target column
```

### Image Data
```
Input: Image file or array (224, 224, 3)
Output: Tensor (3, 224, 224) - normalized, ready for CNN
```

### Time Series
```
Input: Time series CSV (365 days, 3 features)
Output: (359, 7, 3) - sequences of 7 days, 3 features each
```

## 🏗️ Architecture

```
Universal Pipeline
├── DataTypeDetector     # Automatic data type detection
├── Processors/
│   ├── TabularProcessor    # CSV, Excel, Parquet
│   ├── ImageProcessor      # Images with torchvision
│   ├── VideoProcessor      # Video frame extraction
│   ├── AudioProcessor      # Audio feature extraction
│   └── TimeSeriesProcessor # Temporal sequence creation
└── Pipeline               # Main orchestrator
```

## 🤝 Contributing

The pipeline is designed to be extensible:

1. **Add new data types**: Implement new processors inheriting from `BaseProcessor`
2. **Extend detection**: Add patterns to `DataTypeDetector`
3. **Custom preprocessing**: Modify processor configurations

## 📝 Requirements

- Python 3.7+
- pandas, numpy, scikit-learn
- torch, torchvision (for images/videos)
- librosa (for audio)
- opencv-python (for video processing)
- See `requirements.txt` for complete list

## 🔄 Workflow

1. **Detection**: Analyze file extension, content, and structure
2. **Selection**: Choose appropriate processor based on detected type
3. **Preprocessing**: Apply data-type-specific transformations
4. **Output**: Return ML-ready data in standard format

## 💡 Use Cases

- **Rapid Prototyping**: Quickly preprocess any dataset for ML experiments
- **Data Science Pipelines**: Standardized preprocessing across projects
- **AutoML Systems**: Automated feature engineering component
- **Research**: Focus on modeling rather than data preprocessing
- **Production**: Consistent data processing across environments

## 🚨 Error Handling

The pipeline includes comprehensive error handling:
- Invalid file formats
- Missing dependencies
- Corrupted data files
- Configuration errors
- Memory constraints

## 📈 Performance

- **Lazy Loading**: Processes data only when needed
- **Memory Efficient**: Handles large datasets through chunking
- **Parallelizable**: Batch processing for multiple files
- **Caching**: Avoids recomputation of fitted transformers

---

**Ready to process any data type with just one line of code!** 🎉

```python
processed_data = UniversalPipeline().fit_transform("your_data.*")
``` 
