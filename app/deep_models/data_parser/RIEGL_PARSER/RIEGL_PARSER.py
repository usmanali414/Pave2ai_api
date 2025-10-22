"""
RIEGL Parser for transferring data from S3 to local directories.
"""
from pathlib import Path
import os
from typing import Dict, Any, List
from tqdm import tqdm as tq
from app.services.s3.s3_operations import S3Operations
from app.database.conn import mongo_client
from config import database_config
from app.deep_models.data_parser.RIEGL_PARSER.config import RIEGL_PARSER_CONFIG
from app.utils.logger_utils import logger

class RIEGL_PARSER():
    """RIEGL Parser for transferring images and annotations from S3 to local directories."""
    
    def __init__(self):
        self.s3_operations = S3Operations()
        self.parser_config = RIEGL_PARSER_CONFIG
        self.directories = self._setup_local_directories()
    
    def _setup_local_directories(self) -> Dict[str, Path]:        
        # Get the project root directory (go up from current file to project root)
        current_file = Path(__file__)
        project_root = current_file.parents[4]  # Go up 3 levels to reach project root
        
        # Setup paths using relative paths from project root
        base_path = project_root / self.parser_config["local_storage"]["base_path"]
        
        # Create directories based on configuration
        directories = {}
        
        for key, relative_path in self.parser_config["local_storage"].items():
            if key != "base_path":  # Skip base_path itself
                full_path = base_path / relative_path
                full_path.mkdir(parents=True, exist_ok=True)
                directories[key] = full_path
                # logger.info(f"Setup directory: {key} -> {full_path}")
            
        return directories
    
    async def load_data(self, train_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transfer data from S3 to local directories.
        
        Args:
            train_config: Dictionary containing train configuration
            
        Returns:
            Dictionary containing transfer results and local paths
        """
        # logger.info("RIEGL_PARSER.load_data() method called - transferring data from S3 to local directories")
        
        try:
            # Get project info from train config
            project_id = train_config.get("project_id")
            tenant_id = train_config.get("tenant_id")
            db = mongo_client.database
            bucket_configs = db[database_config["BUCKET_CONFIG_COLLECTION"]]
            bucket_config = await bucket_configs.find_one({"project_id": project_id, "tenant_id": tenant_id})
            if not bucket_config:
                raise ValueError(f"Bucket config not found for project: {project_id} and tenant: {tenant_id}")

            folder_structure = bucket_config.get("folder_structure", {})
            preprocessed_data_url = folder_structure.get("preprocessed_data", "")
            annotate_label_url = folder_structure.get("annotate_label", "")
            
            if not preprocessed_data_url or not annotate_label_url:
                raise ValueError(f"Required S3 URLs not found in bucket config. Missing: preprocessed_data={preprocessed_data_url}, annotate_label={annotate_label_url}")
            
            logger.info(f"Transferring images from: {preprocessed_data_url}")
            logger.info(f"Transferring annotations from: {annotate_label_url}")
            
            # Transfer images and annotations to local directories
            image_paths = self._transfer_images_from_s3(preprocessed_data_url)
            annotation_paths = self._transfer_annotations_from_s3(annotate_label_url)
            
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
                    "images_dir": str(self.directories["images_dir"]),
                    "jsons_dir": str(self.directories["jsons_dir"])
                }
            }
            
        except Exception as e:
            logger.error(f"Error transferring RIEGL data: {str(e)}")
            raise
    
    def validate_transferred_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate transferred data structure.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_keys = ["image_paths", "annotation_paths"]
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
    
    def _transfer_images_from_s3(self, s3_base_url: str) -> List[str]:
        """
        Transfer images from S3 to local directory.
        
        Args:
            s3_base_url: Base S3 URL containing images
            
        Returns:
            List of local file paths
        """
        local_paths = []
        
        try:
            # List files in S3 directory
            list_result = self.s3_operations.list_files(s3_base_url)
            
            if not list_result["success"]:
                raise ValueError(f"Failed to list files in {s3_base_url}: {list_result['error']}")
            
            # Use the configured images directory
            images_dir = self.directories["images_dir"]
            
            # Download images to local directory with progress bar
            files = list_result["files"]
            
            for file_info in tq(files, desc="Downloading images"):
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
                    
                except Exception as e:
                    logger.error(f"Error transferring image from {file_info['s3_url']}: {str(e)}")
                    continue
            
            logger.info(f"Successfully transferred {len(local_paths)} images")
            return local_paths
            
        except Exception as e:
            logger.error(f"Error transferring images from S3: {str(e)}")
            return local_paths
    
    def _transfer_annotations_from_s3(self, s3_base_url: str) -> List[str]:
        """
        Transfer JSON annotations from S3 to local directory.
        
        Args:
            s3_base_url: Base S3 URL containing annotations
            
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
            jsons_dir = self.directories["jsons_dir"]
            
            # Download annotations to local directory with progress bar
            for file_info in tq(json_files, desc="Downloading annotations"):
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
                    
                except Exception as e:
                    logger.error(f"Error transferring annotation from {file_info['s3_url']}: {str(e)}")
                    continue
            
            logger.info(f"Successfully transferred {len(local_paths)} annotations")
            return local_paths
            
        except Exception as e:
            logger.error(f"Error transferring annotations from S3: {str(e)}")
            return local_paths