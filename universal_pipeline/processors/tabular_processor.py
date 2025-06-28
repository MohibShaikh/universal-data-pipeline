"""
Tabular Data Processor - Handles CSV, Excel and structured data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

from .base_processor import BaseProcessor


class TabularProcessor(BaseProcessor):
    """Processor for tabular data with comprehensive preprocessing"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.column_transformer = None
        self.feature_names = []
        self.target_encoder = None
        
    def _setup_default_config(self):
        """Setup default configuration"""
        default_config = {
            'scaling_method': 'standard',
            'categorical_encoding': 'onehot',
            'handle_missing': True,
            'target_column': None,
            'return_dataframe': False
        }
        self.config.update({k: v for k, v in default_config.items() if k not in self.config})
    
    def fit(self, data: Union[str, Path, pd.DataFrame]) -> 'TabularProcessor':
        """Fit the processor to the data"""
        df = self._load_data(data) if isinstance(data, (str, Path)) else data.copy()
        
        # Separate features and target
        if self.config['target_column'] and self.config['target_column'] in df.columns:
            X = df.drop(columns=[self.config['target_column']])
            y = df[self.config['target_column']]
            if y.dtype == 'object':
                self.target_encoder = LabelEncoder()
                self.target_encoder.fit(y)
        else:
            X = df
        
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipeline
        transformers = []
        
        if numeric_features:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler() if self.config['scaling_method'] == 'standard' else 'passthrough')
            ])
            transformers.append(('num', numeric_pipeline, numeric_features))
        
        if categorical_features:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_pipeline, categorical_features))
        
        self.column_transformer = ColumnTransformer(transformers=transformers)
        self.column_transformer.fit(X)
        
        # Update feature names
        self._update_feature_names(numeric_features, categorical_features)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[str, Path, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame, Tuple]:
        """Transform the data"""
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        df = self._load_data(data) if isinstance(data, (str, Path)) else data.copy()
        
        # Handle target
        if self.config['target_column'] and self.config['target_column'] in df.columns:
            X = df.drop(columns=[self.config['target_column']])
            y = df[self.config['target_column']]
            if self.target_encoder:
                y = self.target_encoder.transform(y)
            return_target = True
        else:
            X = df
            y = None
            return_target = False
        
        # Transform features
        X_transformed = self.column_transformer.transform(X)
        
        if self.config['return_dataframe']:
            X_transformed = pd.DataFrame(X_transformed, columns=self.feature_names)
        
        return (X_transformed, y) if return_target else X_transformed
    
    def _load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load data from file"""
        filepath = Path(filepath)
        extension = filepath.suffix.lower()
        
        if extension == '.csv':
            return pd.read_csv(filepath)
        elif extension in ['.xlsx', '.xls']:
            return pd.read_excel(filepath)
        elif extension == '.parquet':
            return pd.read_parquet(filepath)
        elif extension == '.json':
            return pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _update_feature_names(self, numeric_features: List[str], categorical_features: List[str]):
        """Update feature names after transformation"""
        feature_names = []
        
        if numeric_features:
            feature_names.extend(numeric_features)
        
        if categorical_features:
            # Get categorical feature names from encoder
            cat_transformer = None
            for name, transformer, columns in self.column_transformer.transformers_:
                if name == 'cat':
                    cat_transformer = transformer
                    break
            
            if cat_transformer and hasattr(cat_transformer.named_steps['encoder'], 'get_feature_names_out'):
                cat_names = cat_transformer.named_steps['encoder'].get_feature_names_out(categorical_features)
                feature_names.extend(cat_names)
            else:
                feature_names.extend([f"cat_{i}" for i in range(len(categorical_features))])
        
        self.feature_names = feature_names
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names 