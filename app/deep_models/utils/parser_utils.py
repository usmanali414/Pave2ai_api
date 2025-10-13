"""
Generic utilities for data parsers.
"""
import os
from pathlib import Path
from typing import Dict, Any, List
from app.database.conn import mongo_client
from config import database_config
from app.utils.logger_utils import logger


class ParserUtils:
    """Generic utilities for data parsers."""
    
    @staticmethod
    def load_bucket_config(project_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        Load bucket configuration from database.
        
        Args:
            project_id: Project ID
            tenant_id: Tenant ID
            
        Returns:
            Dictionary containing bucket configuration
        """
        try:
            db = mongo_client.database
            bucket_configs = db[database_config["BUCKET_CONFIG_COLLECTION"]]
            
            # Find bucket config by project_id
            config = bucket_configs.find_one({"project_id": project_id})
            
            if not config:
                raise ValueError(f"bucket_config not found for project: {project_id}")
            
            logger.info(f"Loaded bucket config for project: {project_id}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading bucket config: {str(e)}")
            raise
    
    @staticmethod
    def get_s3_urls_from_bucket_config(bucket_config: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract S3 URLs from bucket configuration.
        
        Args:
            bucket_config: Bucket configuration dictionary
            
        Returns:
            Dictionary containing S3 URLs
        """
        try:
            folder_structure = bucket_config.get("folder_structure", {})
            
            if not folder_structure:
                raise ValueError("folder_structure not found in bucket_config")
            
            logger.info(f"Retrieved S3 URLs: {list(folder_structure.keys())}")
            return folder_structure
            
        except Exception as e:
            logger.error(f"Error getting S3 URLs from bucket config: {str(e)}")
            raise
    
    @staticmethod
    def setup_local_directories(base_config: Dict[str, Any]) -> Dict[str, Path]:
        """
        Setup local directories based on configuration.
        
        Args:
            base_config: Base configuration containing local_storage settings
            
        Returns:
            Dictionary containing Path objects for directories
        """
        try:
            # Get the project root directory (go up from current file to project root)
            current_file = Path(__file__)
            project_root = current_file.parents[3]  # Go up 3 levels to reach project root
            
            # Setup paths using relative paths from project root
            base_path = project_root / base_config["local_storage"]["base_path"]
            
            # Create directories based on configuration
            directories = {}
            
            for key, relative_path in base_config["local_storage"].items():
                if key != "base_path":  # Skip base_path itself
                    full_path = base_path / relative_path
                    full_path.mkdir(parents=True, exist_ok=True)
                    directories[key] = full_path
                    logger.info(f"Setup directory: {key} -> {full_path}")
            
            return directories
            
        except Exception as e:
            logger.error(f"Error setting up local directories: {str(e)}")
            raise
    
    @staticmethod
    def validate_transferred_data(data: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate transferred data structure.
        
        Args:
            data: Data dictionary to validate
            required_keys: List of required keys
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check required keys
            for key in required_keys:
                if key not in data:
                    logger.error(f"Missing required key: {key}")
                    return False
            
            # Check data types and lengths for path lists
            for key in required_keys:
                if key.endswith("_paths"):
                    paths = data[key]
                    
                    if not isinstance(paths, list):
                        logger.error(f"{key} must be a list")
                        return False
                    
                    if len(paths) == 0:
                        logger.error(f"Empty {key} list")
                        return False
            
            # Validate that files exist
            for key in required_keys:
                if key.endswith("_paths"):
                    paths = data[key]
                    
                    for i, path in enumerate(paths):
                        if not Path(path).exists():
                            logger.error(f"File does not exist: {path}")
                            return False
            
            logger.info(f"Data validation passed for keys: {required_keys}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False
    
    @staticmethod
    def create_image_annotation_pairs(image_paths: List[str], annotation_paths: List[str]) -> List[tuple]:
        """
        Create image-annotation pairs from file paths.
        
        Args:
            image_paths: List of image file paths
            annotation_paths: List of annotation file paths
            
        Returns:
            List of (image_path, annotation_path) tuples
        """
        try:
            if len(image_paths) != len(annotation_paths):
                raise ValueError(f"Mismatch between images ({len(image_paths)}) and annotations ({len(annotation_paths)})")
            
            pairs = list(zip(image_paths, annotation_paths))
            logger.info(f"Created {len(pairs)} image-annotation pairs")
            return pairs
            
        except Exception as e:
            logger.error(f"Error creating image-annotation pairs: {str(e)}")
            raise
