"""
Base Processor Class
Defines the interface and common functionality for all data processors
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
from pathlib import Path


class BaseProcessor(ABC):
    """Base class for all data processors"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the processor
        
        Args:
            config: Configuration dictionary for the processor
        """
        self.config = config or {}
        self.is_fitted = False
        self._setup_default_config()
    
    def _setup_default_config(self):
        """Setup default configuration. Override in subclasses."""
        pass
    
    @abstractmethod
    def fit(self, data: Any) -> 'BaseProcessor':
        """
        Fit the processor to the data (e.g., compute statistics for normalization)
        
        Args:
            data: Input data to fit on
            
        Returns:
            self: Returns self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """
        Transform the data
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data ready for model training
        """
        pass
    
    def fit_transform(self, data: Any) -> Any:
        """
        Fit the processor and transform the data in one step
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        return self.fit(data).transform(data)
    
    def get_config(self) -> Dict:
        """Get the current configuration"""
        return self.config.copy()
    
    def set_config(self, config: Dict) -> 'BaseProcessor':
        """
        Update the configuration
        
        Args:
            config: New configuration to merge
            
        Returns:
            self: Returns self for method chaining
        """
        self.config.update(config)
        return self
    
    def save_state(self, filepath: Union[str, Path]) -> None:
        """
        Save the processor state (for fitted processors)
        
        Args:
            filepath: Path to save the state
        """
        import joblib
        state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'processor_class': self.__class__.__name__
        }
        # Add any processor-specific state
        state.update(self._get_save_state())
        joblib.dump(state, filepath)
    
    def load_state(self, filepath: Union[str, Path]) -> 'BaseProcessor':
        """
        Load the processor state
        
        Args:
            filepath: Path to load the state from
            
        Returns:
            self: Returns self for method chaining
        """
        import joblib
        state = joblib.load(filepath)
        self.config = state['config']
        self.is_fitted = state['is_fitted']
        self._load_save_state(state)
        return self
    
    def _get_save_state(self) -> Dict:
        """Get processor-specific state for saving. Override in subclasses."""
        return {}
    
    def _load_save_state(self, state: Dict) -> None:
        """Load processor-specific state. Override in subclasses."""
        pass
    
    def get_feature_names(self) -> Optional[list]:
        """Get feature names if applicable. Override in subclasses."""
        return None
    
    def get_info(self) -> Dict:
        """Get information about the processor"""
        return {
            'processor_type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'config': self.config,
            'feature_names': self.get_feature_names()
        } 