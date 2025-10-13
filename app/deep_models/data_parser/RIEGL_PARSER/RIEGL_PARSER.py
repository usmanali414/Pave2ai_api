"""
RIEGL Parser for transferring data from S3 to local directories.
"""
from pathlib import Path
from typing import Dict, Any, List
from app.services.s3.s3_operations import S3Operations
from app.deep_models.base_interfaces import DataParser
from app.deep_models.data_parser.RIEGL_PARSER.config import RIEGL_PARSER_CONFIG
from app.deep_models.utils.parser_utils import ParserUtils
from app.utils.logger_utils import logger


class RIEGL_PARSER(DataParser):
    """RIEGL Parser for transferring images and annotations from S3 to local directories."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3_operations = S3Operations()
        self.parser_config = RIEGL_PARSER_CONFIG
        self.s3_urls = config.get("s3_urls", {})
        self.train_config = config.get("train_config", {})
        self._setup_local_directories()
    
    def _setup_local_directories(self):
        """Setup local directories using generic parser utils."""
        # Use generic parser utils to setup directories
        directories = ParserUtils.setup_local_directories(self.parser_config)
        
        # Assign specific directories for RIEGL parser
        self.images_dir = directories["images_dir"]
        self.jsons_dir = directories["jsons_dir"]
    
    def load_data(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Transfer data from S3 to local directories.
        
        Args:
            data_paths: Dictionary containing S3 paths for different data types
            
        Returns:
            Dictionary containing transfer results and local paths
        """
        logger.info("RIEGL_PARSER.load_data() method called - transferring data from S3 to local directories")
        
        try:
            # Get project info from train config
            project_id = self.train_config.get("project_id")
            tenant_id = self.train_config.get("tenant_id")
            
            # Use S3 URLs from bucket config - transfer images from preprocessed_data and annotations from annotate_label
            preprocessed_data_url = self.s3_urls.get("preprocessed_data", data_paths.get("preprocessed_data", ""))
            annotate_label_url = self.s3_urls.get("annotate_label", data_paths.get("annotate_label", ""))
            
            if not preprocessed_data_url or not annotate_label_url:
                raise ValueError(f"Required S3 URLs not found in bucket config. Missing: preprocessed_data={preprocessed_data_url}, annotate_label={annotate_label_url}")
            
            logger.info(f"Transferring images from: {preprocessed_data_url}")
            logger.info(f"Transferring annotations from: {annotate_label_url}")
            
            # Transfer images and annotations to local directories
            image_paths = self._transfer_images_from_s3(preprocessed_data_url, project_id)
            annotation_paths = self._transfer_annotations_from_s3(annotate_label_url, project_id)
            
            if not image_paths or not annotation_paths:
                raise ValueError("No images or annotations transferred")
            
            logger.info(f"Transferred {len(image_paths)} images and {len(annotation_paths)} annotations")
            
            return {
                "image_paths": image_paths,
                "annotation_paths": annotation_paths,
                "project_id": project_id,
                "tenant_id": tenant_id,
                "s3_urls": {
                    "preprocessed_data": preprocessed_data_url,
                    "annotate_label": annotate_label_url
                },
                "local_directories": {
                    "images_dir": str(self.images_dir),
                    "jsons_dir": str(self.jsons_dir)
                }
            }
            
        except Exception as e:
            logger.error(f"Error transferring RIEGL data: {str(e)}")
            raise
    
    
    def preprocess(self, data: Dict[str, Any]) -> tuple:
        """
        Preprocess method stub - preprocessing is handled by AMCNN preprocessor.
        
        Args:
            data: Data dictionary from load_data()
            
        Returns:
            Tuple of (X, y) for training (not used since preprocessing is done elsewhere)
        """
        logger.info("RIEGL_PARSER.preprocess() method called - preprocessing handled by AMCNN preprocessor")
        # Return empty arrays since preprocessing is handled by AMCNN preprocessor
        import numpy as np
        return np.array([]), np.array([])
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate the transferred data using generic parser utils.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Use generic parser utils for validation
            required_keys = ["image_paths", "annotation_paths"]
            return ParserUtils.validate_transferred_data(data, required_keys)
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False
    
    def _transfer_images_from_s3(self, s3_base_url: str, project_id: str) -> List[str]:
        """
        Transfer images from S3 to local directory.
        
        Args:
            s3_base_url: Base S3 URL containing images
            project_id: Project ID for organizing files
            
        Returns:
            List of local file paths
        """
        local_paths = []
        
        try:
            # List files in S3 directory
            list_result = self.s3_operations.list_files(s3_base_url)
            
            if not list_result["success"]:
                raise ValueError(f"Failed to list files in {s3_base_url}: {list_result['error']}")
            
            # Filter for image files
            supported_formats = self.parser_config["supported_formats"]
            image_files = [
                f for f in list_result["files"] 
                if any(f["key"].lower().endswith(ext.lower()) for ext in supported_formats)
                and not f["key"].lower().endswith('.json')
            ]
            
            if not image_files:
                logger.warning(f"No image files found in {s3_base_url}")
                return local_paths
            
            # Use the configured images directory
            images_dir = self.images_dir
            
            # Download images to local directory
            for file_info in image_files:
                try:
                    s3_url = file_info["s3_url"]
                    filename = file_info["key"].split('/')[-1]
                    local_path = images_dir / filename
                    
                    # Download file from S3
                    result = self.s3_operations.download_file(
                        s3_url=s3_url,
                        local_path=str(local_path)
                    )
                    
                    if not result["success"]:
                        logger.error(f"Failed to download image from {s3_url}: {result['error']}")
                        continue
                    
                    local_paths.append(str(local_path))
                    logger.info(f"Successfully transferred image: {filename}")
                    
                except Exception as e:
                    logger.error(f"Error transferring image from {file_info['s3_url']}: {str(e)}")
                    continue
            
            logger.info(f"Successfully transferred {len(local_paths)} images")
            return local_paths
            
        except Exception as e:
            logger.error(f"Error transferring images from S3: {str(e)}")
            return local_paths
    
    def _transfer_annotations_from_s3(self, s3_base_url: str, project_id: str) -> List[str]:
        """
        Transfer JSON annotations from S3 to local directory.
        
        Args:
            s3_base_url: Base S3 URL containing annotations
            project_id: Project ID for organizing files
            
        Returns:
            List of local file paths
        """
        local_paths = []
        
        try:
            # List files in S3 directory
            list_result = self.s3_operations.list_files(s3_base_url)
            
            if not list_result["success"]:
                raise ValueError(f"Failed to list files in {s3_base_url}: {list_result['error']}")
            
            # Filter for JSON files
            json_files = [
                f for f in list_result["files"] 
                if f["key"].lower().endswith('.json')
            ]
            
            if not json_files:
                logger.warning(f"No JSON annotation files found in {s3_base_url}")
                return local_paths
            
            # Use the configured JSONs directory
            jsons_dir = self.jsons_dir
            
            # Download annotations to local directory
            for file_info in json_files:
                try:
                    s3_url = file_info["s3_url"]
                    filename = file_info["key"].split('/')[-1]
                    local_path = jsons_dir / filename
                    
                    # Download file from S3
                    result = self.s3_operations.download_file(
                        s3_url=s3_url,
                        local_path=str(local_path)
                    )
                    
                    if not result["success"]:
                        logger.error(f"Failed to download annotation from {s3_url}: {result['error']}")
                        continue
                    
                    local_paths.append(str(local_path))
                    logger.info(f"Successfully transferred annotation: {filename}")
                    
                except Exception as e:
                    logger.error(f"Error transferring annotation from {file_info['s3_url']}: {str(e)}")
                    continue
            
            logger.info(f"Successfully transferred {len(local_paths)} annotations")
            return local_paths
            
        except Exception as e:
            logger.error(f"Error transferring annotations from S3: {str(e)}")
            return local_paths