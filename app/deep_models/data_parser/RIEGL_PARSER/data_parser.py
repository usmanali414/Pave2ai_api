"""
RIEGL_PARSER implementation of DataParser interface.
"""
import numpy as np
from typing import Dict, Any, Tuple, List
from app.deep_models.base_interfaces import DataParser
from app.deep_models.data_parser.RIEGL_PARSER.data_loader import RieglDataLoader
from app.deep_models.data_parser.RIEGL_PARSER.preprocessor import RieglPreprocessor
from app.deep_models.data_parser.RIEGL_PARSER.config import RIEGL_PARSER_CONFIG
from app.utils.logger_utils import logger


class RieglParser(DataParser):
    """RIEGL point cloud data parser implementation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or RIEGL_PARSER_CONFIG
        self.data_loader = RieglDataLoader(self.config)
        self.preprocessor = RieglPreprocessor(self.config)
        self.raw_data = None
    
    def load_data(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Load RIEGL point cloud data from S3 paths.
        
        Args:
            data_paths: Dictionary containing S3 paths:
                - "point_clouds": List of S3 URLs or single URL
                - "labels": S3 URL to labels file
                - "metadata": Optional metadata file URL
        
        Returns:
            Dictionary containing loaded data
        """
        try:
            logger.info("Starting RIEGL data loading...")
            
            # Load point cloud data
            point_cloud_urls = data_paths.get("point_clouds", [])
            if isinstance(point_cloud_urls, str):
                point_cloud_urls = [point_cloud_urls]
            
            if not point_cloud_urls:
                raise ValueError("No point cloud URLs provided")
            
            point_clouds = self.data_loader.load_point_cloud_data(point_cloud_urls)
            
            if not point_clouds:
                raise ValueError("No point clouds loaded successfully")
            
            # Load labels
            labels_url = data_paths.get("labels", "")
            if not labels_url:
                raise ValueError("No labels URL provided")
            
            labels = self.data_loader.load_labels(labels_url)
            
            if len(labels) == 0:
                raise ValueError("No labels loaded successfully")
            
            # Ensure we have matching number of samples
            min_samples = min(len(point_clouds), len(labels))
            point_clouds = point_clouds[:min_samples]
            labels = labels[:min_samples]
            
            # Store raw data
            self.raw_data = {
                "point_clouds": point_clouds,
                "labels": labels,
                "metadata": data_paths.get("metadata", {}),
                "num_samples": min_samples
            }
            
            logger.info(f"Successfully loaded {min_samples} RIEGL samples")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading RIEGL data: {str(e)}")
            raise
    
    def preprocess(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess RIEGL point cloud data for training.
        
        Args:
            data: Raw data dictionary from load_data()
            
        Returns:
            Tuple of (X, y) for training
        """
        try:
            logger.info("Starting RIEGL data preprocessing...")
            
            if data is None:
                data = self.raw_data
            
            if data is None:
                raise ValueError("No data available for preprocessing. Call load_data() first.")
            
            point_clouds = data["point_clouds"]
            labels = data["labels"]
            
            # Preprocess point clouds
            X, y = self.preprocessor.preprocess_point_clouds(point_clouds, labels)
            
            logger.info(f"Preprocessing completed. Features shape: {X.shape}, Labels shape: {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preprocessing RIEGL data: {str(e)}")
            raise
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate RIEGL point cloud data.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            if not data:
                logger.error("No data provided for validation")
                return False
            
            # Check required keys
            required_keys = ["point_clouds", "labels"]
            for key in required_keys:
                if key not in data:
                    logger.error(f"Missing required key: {key}")
                    return False
            
            point_clouds = data["point_clouds"]
            labels = data["labels"]
            
            # Validate point clouds
            if not isinstance(point_clouds, list) or len(point_clouds) == 0:
                logger.error("Invalid point clouds data")
                return False
            
            # Validate labels
            if not isinstance(labels, np.ndarray) or len(labels) == 0:
                logger.error("Invalid labels data")
                return False
            
            # Check sample count consistency
            if len(point_clouds) != len(labels):
                logger.error(f"Sample count mismatch: {len(point_clouds)} clouds vs {len(labels)} labels")
                return False
            
            # Validate individual point clouds
            for i, cloud in enumerate(point_clouds):
                if not isinstance(cloud, np.ndarray):
                    logger.error(f"Point cloud {i} is not a numpy array")
                    return False
                
                if cloud.shape[1] != 3:  # Should have x, y, z coordinates
                    logger.error(f"Point cloud {i} has invalid shape: {cloud.shape}")
                    return False
            
            logger.info("RIEGL data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating RIEGL data: {str(e)}")
            return False
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about loaded data.
        
        Returns:
            Dictionary containing data information
        """
        if self.raw_data is None:
            return {"status": "no_data_loaded"}
        
        return {
            "num_samples": self.raw_data["num_samples"],
            "point_cloud_shapes": [cloud.shape for cloud in self.raw_data["point_clouds"]],
            "labels_shape": self.raw_data["labels"].shape,
            "labels_unique": np.unique(self.raw_data["labels"]).tolist(),
            "parser_config": self.config
        }
