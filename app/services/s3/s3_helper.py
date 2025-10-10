import os
from typing import List, Dict, Optional
from botocore.exceptions import ClientError
from app.utils.logger_utils import logger
from app.services.s3.s3_client import s3_client
from config import S3_DATA_BUCKET


class S3Helper:
    """Helper class for S3 operations"""
    
    @staticmethod
    def generate_project_s3_path(tenant_id: str, project_id: str) -> str:
        """
        Generate the base S3 path for a project
        Format: s3:accounts/{tenant_id}/project/{project_id}/
        """
        return f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/"
    
    @staticmethod
    def generate_raw_data_url(tenant_id: str, project_id: str) -> str:
        """
        Generate the raw data S3 URL for a project
        Format: s3:accounts/{tenant_id}/project/{project_id}/data/raw/
        """
        return f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/data/raw/"
    
    @staticmethod
    def create_project_folder_structure(tenant_id: str, project_id: str) -> Dict[str, str]:
        """
        Create the complete folder structure for a project in S3
        
        Structure:
        accounts/
            {tenant_id}/
            └── project/
                └── {project_id}/
                    ├── user/
                    │   └── logs/
                    ├── data/
                    │   ├── raw/
                    │   └── preprocessed/
                    ├── annotate/
                    │   ├── tool/
                    │   └── label/
                    ├── train/
                    │   ├── input_model/
                    │   ├── output_model/
                    │   └── output_metadata/
                    └── inference/
                        ├── input_model/
                        ├── output_labels/
                        └── output_metadata/
        
        Returns:
            Dict with all created folder URLs
        """
        if not s3_client.is_initialized():
            logger.warning("S3 client not initialized. Skipping folder creation.")
            return {"raw_data_url": S3Helper.generate_raw_data_url(tenant_id, project_id)}
        
        base_path = f"accounts/{tenant_id}/project/{project_id}"
        
        # Define all folder paths (annotate/train/inference are siblings of data)
        folders = [
            # User logs
            f"{base_path}/user/logs/",
            
            # Data folders
            f"{base_path}/data/raw/",
            f"{base_path}/data/preprocessed/",
            
            # Annotation folders
            f"{base_path}/annotate/tool/",
            f"{base_path}/annotate/label/",
            
            # Training folders
            f"{base_path}/train/input_model/",
            f"{base_path}/train/output_model/",
            f"{base_path}/train/output_metadata/",
            
            # Inference folders
            f"{base_path}/inference/input_model/",
            f"{base_path}/inference/output_labels/",
            f"{base_path}/inference/output_metadata/",
        ]
        
        created_folders = []
        
        try:
            bucket = s3_client.resource.Bucket(S3_DATA_BUCKET)
            
            # Create folders by uploading empty objects with trailing slash
            for folder_path in folders:
                try:
                    # S3 doesn't have actual folders, we create a 0-byte object with trailing /
                    bucket.put_object(Key=folder_path, Body=b'')
                    created_folders.append(folder_path)
                    logger.info(f"Created S3 folder: s3://{S3_DATA_BUCKET}/{folder_path}")
                except ClientError as e:
                    logger.error(f"Failed to create folder {folder_path}: {str(e)}")
            
            logger.info(f"Successfully created {len(created_folders)} folders for project {project_id}")
            
            # Build complete folder structure dict
            folder_structure = {
                "user_logs": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/user/logs/",
                "raw_data": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/data/raw/",
                "preprocessed_data": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/data/preprocessed/",
                "annotate_tool": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/annotate/tool/",
                "annotate_label": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/annotate/label/",
                "train_input_model": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/train/input_model/",
                "train_output_model": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/train/output_model/",
                "train_output_metadata": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/train/output_metadata/",
                "inference_input_model": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/inference/input_model/",
                "inference_output_labels": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/inference/output_labels/",
                "inference_output_metadata": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/inference/output_metadata/",
            }
            
            return {
                "raw_data_url": S3Helper.generate_raw_data_url(tenant_id, project_id),
                "folder_structure": folder_structure,
                "created_folders": created_folders,
                "total_folders": len(folders),
                "success": len(created_folders) == len(folders)
            }
            
        except Exception as e:
            logger.error(f"Error creating project folder structure: {str(e)}")
            # Return the URLs anyway, even if folder creation fails
            folder_structure = {
                "user_logs": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/user/logs/",
                "raw_data": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/data/raw/",
                "preprocessed_data": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/data/preprocessed/",
                "annotate_tool": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/annotate/tool/",
                "annotate_label": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/annotate/label/",
                "train_input_model": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/train/input_model/",
                "train_output_model": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/train/output_model/",
                "train_output_metadata": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/train/output_metadata/",
                "inference_input_model": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/inference/input_model/",
                "inference_output_labels": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/inference/output_labels/",
                "inference_output_metadata": f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/inference/output_metadata/",
            }
            return {
                "raw_data_url": S3Helper.generate_raw_data_url(tenant_id, project_id),
                "folder_structure": folder_structure,
                "error": str(e),
                "success": False
            }
    
    @staticmethod
    def check_folder_exists(tenant_id: str, project_id: str, folder_path: str = "data/raw/") -> bool:
        """
        Check if a specific folder exists in the project structure
        
        Args:
            tenant_id: Tenant ID
            project_id: Project ID
            folder_path: Relative folder path (e.g., "data/raw/")
        
        Returns:
            True if folder exists, False otherwise
        """
        if not s3_client.is_initialized():
            logger.warning("S3 client not initialized. Cannot check folder existence.")
            return False
        
        try:
            full_path = f"accounts/{tenant_id}/project/{project_id}/{folder_path}"
            
            # Check if object exists
            s3_client.client.head_object(Bucket=S3_DATA_BUCKET, Key=full_path)
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"Error checking folder existence: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking folder: {str(e)}")
            return False
    
    @staticmethod
    def list_folder_contents(tenant_id: str, project_id: str, folder_path: str = "data/raw/") -> List[str]:
        """
        List contents of a specific folder in the project
        
        Args:
            tenant_id: Tenant ID
            project_id: Project ID
            folder_path: Relative folder path
        
        Returns:
            List of object keys in the folder
        """
        if not s3_client.is_initialized():
            logger.warning("S3 client not initialized. Cannot list folder contents.")
            return []
        
        try:
            full_path = f"accounts/{tenant_id}/project/{project_id}/{folder_path}"
            
            response = s3_client.client.list_objects_v2(
                Bucket=S3_DATA_BUCKET,
                Prefix=full_path,
                Delimiter='/'
            )
            
            contents = []
            
            # Get objects (files)
            if 'Contents' in response:
                contents.extend([obj['Key'] for obj in response['Contents']])
            
            # Get common prefixes (subdirectories)
            if 'CommonPrefixes' in response:
                contents.extend([prefix['Prefix'] for prefix in response['CommonPrefixes']])
            
            return contents
            
        except Exception as e:
            logger.error(f"Error listing folder contents: {str(e)}")
            return []
    
    @staticmethod
    def delete_project_folders(tenant_id: str, project_id: str) -> bool:
        """
        Delete all folders and objects for a project
        
        Args:
            tenant_id: Tenant ID
            project_id: Project ID
        
        Returns:
            True if successful, False otherwise
        """
        if not s3_client.is_initialized():
            logger.warning("S3 client not initialized. Cannot delete folders.")
            return False
        
        try:
            base_path = f"accounts/{tenant_id}/project/{project_id}/"
            bucket = s3_client.resource.Bucket(S3_DATA_BUCKET)
            
            # Delete all objects with this prefix
            objects_to_delete = list(bucket.objects.filter(Prefix=base_path))
            
            if objects_to_delete:
                bucket.delete_objects(
                    Delete={
                        'Objects': [{'Key': obj.key} for obj in objects_to_delete]
                    }
                )
                logger.info(f"Deleted {len(objects_to_delete)} objects for project {project_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting project folders: {str(e)}")
            return False
    
    @staticmethod
    def generate_presigned_upload_url(
        tenant_id: str,
        project_id: str,
        file_key: str,
        expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate a presigned URL for uploading a file directly to S3
        
        Args:
            tenant_id: Tenant ID
            project_id: Project ID
            file_key: File key/path within the project (e.g., "data/raw/image.jpg")
            expiration: URL expiration time in seconds (default: 1 hour)
        
        Returns:
            Presigned URL string or None if failed
        """
        if not s3_client.is_initialized():
            logger.warning("S3 client not initialized. Cannot generate presigned URL.")
            return None
        
        try:
            full_key = f"accounts/{tenant_id}/project/{project_id}/{file_key}"
            
            url = s3_client.client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': S3_DATA_BUCKET,
                    'Key': full_key
                },
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned upload URL for: {full_key}")
            return url
            
        except Exception as e:
            logger.error(f"Error generating presigned URL: {str(e)}")
            return None


# Singleton instance
s3_helper = S3Helper()

