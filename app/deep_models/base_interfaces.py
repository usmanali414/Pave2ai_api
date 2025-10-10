"""
Base interfaces for data parsers and models in the AMCNN training pipeline.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class DataParser(ABC):
    """Base interface for data parsers."""
    
    @abstractmethod
    def load_data(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Load data from the specified paths.
        
        Args:
            data_paths: Dictionary containing S3 paths for different data types
            
        Returns:
            Dictionary containing loaded data
        """
        pass
    
    @abstractmethod
    def preprocess(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the loaded data for training.
        
        Args:
            data: Raw data dictionary from load_data()
            
        Returns:
            Tuple of (X, y) for training
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate the loaded data.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass


class Model(ABC):
    """Base interface for ML models."""
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the model with given data.
        
        Args:
            X: Training features
            y: Training labels
            config: Training configuration
            
        Returns:
            Training metrics and results
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def save_weights(self, weights_path: str) -> bool:
        """
        Save model weights to specified path.
        
        Args:
            weights_path: S3 path to save weights
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_weights(self, weights_path: str) -> bool:
        """
        Load model weights from specified path.
        
        Args:
            weights_path: S3 path to load weights from
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and configuration.
        
        Returns:
            Dictionary containing model information
        """
        pass
