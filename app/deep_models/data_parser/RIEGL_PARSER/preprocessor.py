"""
Preprocessor for RIEGL point cloud data.
"""
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from app.utils.logger_utils import logger


class RieglPreprocessor:
    """Preprocessor for RIEGL point cloud data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def preprocess_point_clouds(self, point_clouds: list, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess point clouds and labels for training.
        
        Args:
            point_clouds: List of point cloud arrays
            labels: Corresponding labels
            
        Returns:
            Tuple of (processed_features, processed_labels)
        """
        try:
            # Normalize point clouds
            normalized_clouds = []
            for cloud in point_clouds:
                if cloud is not None and len(cloud) > 0:
                    normalized_cloud = self._normalize_point_cloud(cloud)
                    normalized_clouds.append(normalized_cloud)
            
            if not normalized_clouds:
                raise ValueError("No valid point clouds found after preprocessing")
            
            # Pad/truncate to uniform size
            max_points = self.config["data_processing"]["max_points_per_cloud"]
            uniform_clouds = self._make_uniform_size(normalized_clouds, max_points)
            
            # Convert to training format
            X = np.array(uniform_clouds)
            y = labels[:len(X)]  # Ensure labels match number of samples
            
            # Apply additional preprocessing
            if self.config["data_processing"]["remove_outliers"]:
                X, y = self._remove_outliers(X, y)
            
            logger.info(f"Preprocessed {len(X)} samples with {X.shape[1]} points each")
            return X, y
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/test sets.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        train_ratio = self.config["preprocessing"]["train_test_split"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=train_ratio,
            random_state=42,
            stratify=y
        )
        
        logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
    
    def _normalize_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Normalize point cloud coordinates.
        
        Args:
            point_cloud: Raw point cloud
            
        Returns:
            Normalized point cloud
        """
        if not self.config["data_processing"]["normalize"]:
            return point_cloud
        
        # Center the point cloud
        centered = point_cloud - np.mean(point_cloud, axis=0)
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        if max_dist > 0:
            normalized = centered / max_dist
        else:
            normalized = centered
        
        return normalized.astype(np.float32)
    
    def _make_uniform_size(self, point_clouds: list, max_points: int) -> list:
        """
        Make all point clouds the same size.
        
        Args:
            point_clouds: List of point clouds
            max_points: Maximum number of points per cloud
            
        Returns:
            List of uniformly sized point clouds
        """
        uniform_clouds = []
        
        for cloud in point_clouds:
            if len(cloud) > max_points:
                # Randomly sample points
                indices = np.random.choice(len(cloud), max_points, replace=False)
                uniform_cloud = cloud[indices]
            elif len(cloud) < max_points:
                # Pad with zeros
                padding = np.zeros((max_points - len(cloud), cloud.shape[1]))
                uniform_cloud = np.vstack([cloud, padding])
            else:
                uniform_cloud = cloud
            
            uniform_clouds.append(uniform_cloud)
        
        return uniform_clouds
    
    def _remove_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outlier samples.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (X_clean, y_clean)
        """
        threshold = self.config["data_processing"]["outlier_threshold"]
        
        # Calculate distances from mean
        mean_distances = np.mean(np.linalg.norm(X, axis=(1, 2)), axis=1)
        
        # Find inliers
        inlier_mask = mean_distances < (np.mean(mean_distances) + threshold * np.std(mean_distances))
        
        X_clean = X[inlier_mask]
        y_clean = y[inlier_mask]
        
        logger.info(f"Removed {len(X) - len(X_clean)} outlier samples")
        return X_clean, y_clean
