"""
Time Series Data Processor - Handles temporal data preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from .base_processor import BaseProcessor


class TimeSeriesProcessor(BaseProcessor):
    """Processor for time series data"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.scaler = None
        self.imputer = None
        self.feature_columns = []
        
    def _setup_default_config(self):
        """Setup default configuration"""
        default_config = {
            'sequence_length': 100,
            'scaling_method': 'standard',  # standard, minmax, none
            'handle_missing': True,
            'impute_strategy': 'forward_fill',  # forward_fill, backward_fill, mean, median
            'time_column': None,
            'target_column': None,
            'step_size': 1,  # For sliding window
            'normalize_features': True,
            'return_3d': True  # Return (samples, timesteps, features)
        }
        self.config.update({k: v for k, v in default_config.items() if k not in self.config})
    
    def fit(self, data: Union[str, Path, pd.DataFrame, np.ndarray]) -> 'TimeSeriesProcessor':
        """Fit the processor to the data"""
        if isinstance(data, (str, Path)):
            df = self._load_data(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Prepare data
        df = self._prepare_dataframe(df)
        
        # Fit scalers and imputers
        if self.config['handle_missing']:
            if self.config['impute_strategy'] in ['mean', 'median']:
                self.imputer = SimpleImputer(strategy=self.config['impute_strategy'])
                self.imputer.fit(df[self.feature_columns])
        
        if self.config['scaling_method'] == 'standard':
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.feature_columns])
        elif self.config['scaling_method'] == 'minmax':
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[self.feature_columns])
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[str, Path, pd.DataFrame, np.ndarray]) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Transform the time series data"""
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        if isinstance(data, (str, Path)):
            df = self._load_data(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Prepare data
        df = self._prepare_dataframe(df)
        
        # Handle missing values
        if self.config['handle_missing']:
            if self.config['impute_strategy'] == 'forward_fill':
                df[self.feature_columns] = df[self.feature_columns].ffill()
            elif self.config['impute_strategy'] == 'backward_fill':
                df[self.feature_columns] = df[self.feature_columns].bfill()
            elif self.imputer:
                df[self.feature_columns] = self.imputer.transform(df[self.feature_columns])
        
        # Scale features
        if self.scaler:
            df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        # Create sequences
        X, y = self._create_sequences(df)
        
        if y is not None:
            return X, y
        else:
            return X
    
    def _load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load time series data from file"""
        filepath = Path(filepath)
        extension = filepath.suffix.lower()
        
        if extension == '.csv':
            df = pd.read_csv(filepath)
        elif extension in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif extension == '.parquet':
            df = pd.read_parquet(filepath)
        elif extension == '.json':
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        return df
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the dataframe for processing"""
        # Handle time column
        if self.config['time_column'] and self.config['time_column'] in df.columns:
            df[self.config['time_column']] = pd.to_datetime(df[self.config['time_column']])
            df = df.set_index(self.config['time_column']).sort_index()
        
        # Identify feature columns
        exclude_columns = []
        if self.config['time_column']:
            exclude_columns.append(self.config['time_column'])
        if self.config['target_column']:
            exclude_columns.append(self.config['target_column'])
        
        self.feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Keep only numeric columns for features
        numeric_features = df[self.feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = numeric_features
        
        return df
    
    def _create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences from the time series data"""
        feature_data = df[self.feature_columns].values
        
        # Handle target data
        if self.config['target_column'] and self.config['target_column'] in df.columns:
            target_data = df[self.config['target_column']].values
            has_target = True
        else:
            target_data = None
            has_target = False
        
        sequences = []
        targets = [] if has_target else None
        
        for i in range(0, len(feature_data) - self.config['sequence_length'] + 1, self.config['step_size']):
            # Extract sequence
            seq = feature_data[i:i + self.config['sequence_length']]
            sequences.append(seq)
            
            # Extract target (next value or sequence)
            if has_target:
                target_idx = i + self.config['sequence_length'] - 1
                if target_idx < len(target_data):
                    targets.append(target_data[target_idx])
        
        X = np.array(sequences)
        y = np.array(targets) if targets else None
        
        # Ensure 3D output if requested
        if self.config['return_3d'] and X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_columns
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get the output shape after transformation"""
        if self.config['return_3d']:
            return (None, self.config['sequence_length'], len(self.feature_columns))
        else:
            return (None, self.config['sequence_length'] * len(self.feature_columns))
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform the scaled data"""
        if not self.is_fitted or not self.scaler:
            return data
        
        # Handle different shapes
        original_shape = data.shape
        if data.ndim == 3:
            # Reshape from (samples, timesteps, features) to (samples*timesteps, features)
            data_2d = data.reshape(-1, data.shape[-1])
            inverse_data = self.scaler.inverse_transform(data_2d)
            return inverse_data.reshape(original_shape)
        elif data.ndim == 2:
            return self.scaler.inverse_transform(data)
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}")
    
    def _get_save_state(self) -> Dict:
        """Get processor-specific state for saving"""
        return {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns
        }
    
    def _load_save_state(self, state: Dict):
        """Load processor-specific state"""
        self.scaler = state.get('scaler')
        self.imputer = state.get('imputer')
        self.feature_columns = state.get('feature_columns', []) 