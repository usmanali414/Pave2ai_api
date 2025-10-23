import os
import io
import mimetypes
from typing import List, Dict, Optional, Union, BinaryIO, Tuple
from pathlib import Path
from botocore.exceptions import ClientError
from app.utils.logger_utils import logger
from app.services.s3.s3_client import s3_client
from config import S3_DATA_BUCKET


class S3Operations:
    """Comprehensive S3 operations for file upload, download, and management"""
    
    @staticmethod
    def parse_s3_url(s3_url: str) -> Tuple[str, str]:
        """
        Parse S3 URL to extract bucket and key
        
        Args:
            s3_url: S3 URL (e.g., "s3://bucket-name/path/to/file")
        
        Returns:
            Tuple of (bucket_name, key)
        
        Raises:
            ValueError: If URL format is invalid
        """
        if not s3_url.startswith('s3://'):
            raise ValueError(f"Invalid S3 URL format: {s3_url}")
        
        # Remove s3:// prefix and split
        path = s3_url[5:]  # Remove 's3://'
        parts = path.split('/', 1)
        
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URL format: {s3_url}")
        
        bucket_name, key = parts
        return bucket_name, key
    
    @staticmethod
    def build_s3_url(bucket: str, key: str) -> str:
        """
        Build S3 URL from bucket and key
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
        
        Returns:
            Complete S3 URL
        """
        return f"s3://{bucket}/{key}"
    
    @staticmethod
    def upload_file(
        file_data: Union[bytes, BinaryIO, str],
        s3_url: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = True
    ) -> Dict[str, Union[bool, str, Dict]]:
        """
        Upload a file to S3
        
        Args:
            file_data: File content as bytes, file-like object, or file path
            s3_url: Target S3 URL (e.g., "s3://bucket/path/to/file")
            content_type: MIME type of the file (auto-detected if not provided)
            metadata: Additional metadata to store with the file
            overwrite: Whether to overwrite existing file
        
        Returns:
            Dict with upload result information
        """
        if not s3_client.is_initialized():
            logger.error("S3 client not initialized")
            return {
                "success": False,
                "error": "S3 client not initialized",
                "s3_url": s3_url
            }
        
        try:
            # Parse S3 URL
            bucket_name, key = S3Operations.parse_s3_url(s3_url)
            
            # Handle different input types
            if isinstance(file_data, str):
                # File path
                if not os.path.exists(file_data):
                    return {
                        "success": False,
                        "error": f"File not found: {file_data}",
                        "s3_url": s3_url
                    }
                
                with open(file_data, 'rb') as f:
                    file_content = f.read()
                
                # Auto-detect content type from file extension
                if not content_type:
                    content_type, _ = mimetypes.guess_type(file_data)
                    
            elif hasattr(file_data, 'read'):
                # File-like object
                file_content = file_data.read()
                
                # Try to get filename for content type detection
                filename = getattr(file_data, 'name', None)
                if filename and not content_type:
                    content_type, _ = mimetypes.guess_type(filename)
                    
            elif isinstance(file_data, bytes):
                # Raw bytes
                file_content = file_data
                
            else:
                return {
                    "success": False,
                    "error": "Invalid file_data type. Expected bytes, file-like object, or file path",
                    "s3_url": s3_url
                }
            
            # Check if file already exists (if overwrite is False)
            if not overwrite:
                try:
                    s3_client.client.head_object(Bucket=bucket_name, Key=key)
                    return {
                        "success": False,
                        "error": f"File already exists: {s3_url}",
                        "s3_url": s3_url
                    }
                except ClientError as e:
                    if e.response['Error']['Code'] != '404':
                        raise
            
            # Prepare upload parameters
            upload_params = {
                'Bucket': bucket_name,
                'Key': key,
                'Body': file_content
            }
            
            if content_type:
                upload_params['ContentType'] = content_type
            
            if metadata:
                upload_params['Metadata'] = metadata
            
            # Upload file
            response = s3_client.client.put_object(**upload_params)
            
            # logger.info(f"Successfully uploaded file to {s3_url}")
            
            return {
                "success": True,
                "s3_url": s3_url,
                "bucket": bucket_name,
                "key": key,
                "content_type": content_type,
                "size": len(file_content),
                "etag": response.get('ETag', '').strip('"'),
                "metadata": metadata or {}
            }
            
        except Exception as e:
            logger.error(f"Error uploading file to {s3_url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "s3_url": s3_url
            }
    
    @staticmethod
    def upload_file_to_project_path(
        file_data: Union[bytes, BinaryIO, str],
        tenant_id: str,
        project_id: str,
        file_path: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = True
    ) -> Dict[str, Union[bool, str, Dict]]:
        """
        Upload a file to a specific project path in S3
        
        Args:
            file_data: File content as bytes, file-like object, or file path
            tenant_id: Tenant ID
            project_id: Project ID
            file_path: Relative path within the project (e.g., "data/raw/image.jpg")
            content_type: MIME type of the file
            metadata: Additional metadata
            overwrite: Whether to overwrite existing file
        
        Returns:
            Dict with upload result information
        """
        # Build S3 URL
        s3_url = f"s3://{S3_DATA_BUCKET}/{tenant_id}/project/{project_id}/{file_path}"
        
        return S3Operations.upload_file(
            file_data=file_data,
            s3_url=s3_url,
            content_type=content_type,
            metadata=metadata,
            overwrite=overwrite
        )
    
    @staticmethod
    def download_file(
        s3_url: str,
        local_path: Optional[str] = None,
        return_content: bool = False
    ) -> Dict[str, Union[bool, str, bytes, Dict]]:
        """
        Download a file from S3
        
        Args:
            s3_url: S3 URL of the file to download
            local_path: Local file path to save the file (optional if return_content=True)
            return_content: Whether to return file content as bytes
        
        Returns:
            Dict with download result information
        """
        if not s3_client.is_initialized():
            logger.error("S3 client not initialized")
            return {
                "success": False,
                "error": "S3 client not initialized",
                "s3_url": s3_url
            }
        
        if not local_path and not return_content:
            return {
                "success": False,
                "error": "Either local_path or return_content must be specified",
                "s3_url": s3_url
            }
        
        try:
            # Parse S3 URL
            bucket_name, key = S3Operations.parse_s3_url(s3_url)
            
            # Get object metadata first
            try:
                head_response = s3_client.client.head_object(Bucket=bucket_name, Key=key)
                file_size = head_response.get('ContentLength', 0)
                content_type = head_response.get('ContentType', 'application/octet-stream')
                metadata = head_response.get('Metadata', {})
                last_modified = head_response.get('LastModified')
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    return {
                        "success": False,
                        "error": f"File not found: {s3_url}",
                        "s3_url": s3_url
                    }
                raise
            
            # Download file content
            response = s3_client.client.get_object(Bucket=bucket_name, Key=key)
            file_content = response['Body'].read()
            
            result = {
                "success": True,
                "s3_url": s3_url,
                "bucket": bucket_name,
                "key": key,
                "size": file_size,
                "content_type": content_type,
                "metadata": metadata,
                "last_modified": last_modified
            }
            
            # Save to local path if specified
            if local_path:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'wb') as f:
                    f.write(file_content)
                result["local_path"] = local_path
            
            # Return content if requested
            if return_content:
                result["content"] = file_content
            
            return result
            
        except Exception as e:
            logger.error(f"Error downloading file from {s3_url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "s3_url": s3_url
            }
    
    @staticmethod
    def read_file_content(s3_url: str) -> Dict[str, Union[bool, str, bytes, Dict]]:
        """
        Read file content from S3 without saving to disk
        
        Args:
            s3_url: S3 URL of the file to read
        
        Returns:
            Dict with file content and metadata
        """
        return S3Operations.download_file(
            s3_url=s3_url,
            return_content=True
        )
    
    @staticmethod
    def list_files(
        s3_url_prefix: str,
        max_keys: int = None,
        delimiter: Optional[str] = None,
        fetch_all: bool = True
    ) -> Dict[str, Union[bool, List, str]]:
        """
        List files in an S3 location
        
        Args:
            s3_url_prefix: S3 URL prefix to search (e.g., "s3://bucket/path/")
            max_keys: Maximum number of keys to return (None = no limit, fetch all with pagination)
            delimiter: Character to use to group keys
            fetch_all: If True, automatically paginate through all results (default: True)
        
        Returns:
            Dict with list of files and metadata
        """
        if not s3_client.is_initialized():
            logger.error("S3 client not initialized")
            return {
                "success": False,
                "error": "S3 client not initialized"
            }
        
        try:
            # Parse S3 URL
            bucket_name, key_prefix = S3Operations.parse_s3_url(s3_url_prefix)
            
            files = []
            folders = []
            continuation_token = None
            total_iterations = 0
            
            # Keep fetching until we have all files or reach max_keys
            while True:
                # List objects
                list_params = {
                    'Bucket': bucket_name,
                    'MaxKeys': 1000  # S3 API max per request
                }
                
                if key_prefix:
                    list_params['Prefix'] = key_prefix
                
                if delimiter:
                    list_params['Delimiter'] = delimiter
                
                if continuation_token:
                    list_params['ContinuationToken'] = continuation_token
                
                response = s3_client.client.list_objects_v2(**list_params)
                total_iterations += 1
                
                # Process files
                if 'Contents' in response:
                    for obj in response['Contents']:
                        files.append({
                            "key": obj['Key'],
                            "s3_url": S3Operations.build_s3_url(bucket_name, obj['Key']),
                            "size": obj['Size'],
                            "last_modified": obj['LastModified'],
                            "etag": obj['ETag'].strip('"'),
                            "storage_class": obj.get('StorageClass', 'STANDARD')
                        })
                
                # Process common prefixes (folders)
                if 'CommonPrefixes' in response:
                    for prefix in response['CommonPrefixes']:
                        folders.append({
                            "prefix": prefix['Prefix'],
                            "s3_url": S3Operations.build_s3_url(bucket_name, prefix['Prefix'])
                        })
                
                # Check if we should continue pagination
                is_truncated = response.get('IsTruncated', False)
                continuation_token = response.get('NextContinuationToken')
                
                # Stop if: not truncated, fetch_all is False, or reached max_keys
                if not is_truncated:
                    break
                
                if not fetch_all:
                    break
                
                if max_keys is not None and len(files) >= max_keys:
                    files = files[:max_keys]  # Trim to max_keys
                    break
            
            # logger.info(f"Listed {len(files)} files and {len(folders)} folders from {s3_url_prefix} (iterations: {total_iterations})")
            
            return {
                "success": True,
                "files": files,
                "folders": folders,
                "total_files": len(files),
                "total_folders": len(folders),
                "is_truncated": False,  # Always False since we fetched all
                "next_continuation_token": None
            }
            
        except Exception as e:
            logger.error(f"Error listing files from {s3_url_prefix}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "files": [],
                "folders": []
            }
    
    @staticmethod
    def delete_file(s3_url: str) -> Dict[str, Union[bool, str]]:
        """
        Delete a file from S3
        
        Args:
            s3_url: S3 URL of the file to delete
        
        Returns:
            Dict with deletion result
        """
        if not s3_client.is_initialized():
            logger.error("S3 client not initialized")
            return {
                "success": False,
                "error": "S3 client not initialized"
            }
        
        try:
            # Parse S3 URL
            bucket_name, key = S3Operations.parse_s3_url(s3_url)
            
            # Delete object
            s3_client.client.delete_object(Bucket=bucket_name, Key=key)
            
            logger.info(f"Successfully deleted file: {s3_url}")
            
            return {
                "success": True,
                "s3_url": s3_url,
                "message": "File deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deleting file {s3_url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "s3_url": s3_url
            }
    
    @staticmethod
    def batch_upload_files(
        files: List[Dict[str, Union[str, bytes, BinaryIO]]],
        base_s3_url: str,
        overwrite: bool = True
    ) -> Dict[str, Union[bool, List, str]]:
        """
        Upload multiple files to S3 in batch
        
        Args:
            files: List of file dictionaries with 'file_data', 'file_path', and optional 'content_type', 'metadata'
            base_s3_url: Base S3 URL (e.g., "s3://bucket/path/")
            overwrite: Whether to overwrite existing files
        
        Returns:
            Dict with batch upload results
        """
        results = []
        successful_uploads = 0
        failed_uploads = 0
        
        for file_info in files:
            file_data = file_info['file_data']
            file_path = file_info['file_path']
            content_type = file_info.get('content_type')
            metadata = file_info.get('metadata')
            
            # Build full S3 URL
            full_s3_url = f"{base_s3_url.rstrip('/')}/{file_path.lstrip('/')}"
            
            # Upload file
            result = S3Operations.upload_file(
                file_data=file_data,
                s3_url=full_s3_url,
                content_type=content_type,
                metadata=metadata,
                overwrite=overwrite
            )
            
            results.append({
                "file_path": file_path,
                "s3_url": full_s3_url,
                **result
            })
            
            if result['success']:
                successful_uploads += 1
            else:
                failed_uploads += 1
        
        logger.info(f"Batch upload completed: {successful_uploads} successful, {failed_uploads} failed")
        
        return {
            "success": failed_uploads == 0,
            "total_files": len(files),
            "successful_uploads": successful_uploads,
            "failed_uploads": failed_uploads,
            "results": results
        }
    
    @staticmethod
    def copy_file(
        source_s3_url: str,
        destination_s3_url: str,
        overwrite: bool = True
    ) -> Dict[str, Union[bool, str]]:
        """
        Copy a file from one S3 location to another
        
        Args:
            source_s3_url: Source S3 URL
            destination_s3_url: Destination S3 URL
            overwrite: Whether to overwrite existing destination file
        
        Returns:
            Dict with copy result
        """
        if not s3_client.is_initialized():
            logger.error("S3 client not initialized")
            return {
                "success": False,
                "error": "S3 client not initialized"
            }
        
        try:
            # Parse URLs
            source_bucket, source_key = S3Operations.parse_s3_url(source_s3_url)
            dest_bucket, dest_key = S3Operations.parse_s3_url(destination_s3_url)
            
            # Check if destination exists (if overwrite is False)
            if not overwrite:
                try:
                    s3_client.client.head_object(Bucket=dest_bucket, Key=dest_key)
                    return {
                        "success": False,
                        "error": f"Destination file already exists: {destination_s3_url}",
                        "source_s3_url": source_s3_url,
                        "destination_s3_url": destination_s3_url
                    }
                except ClientError as e:
                    if e.response['Error']['Code'] != '404':
                        raise
            
            # Copy object
            copy_source = {'Bucket': source_bucket, 'Key': source_key}
            s3_client.client.copy_object(
                CopySource=copy_source,
                Bucket=dest_bucket,
                Key=dest_key
            )
            
            logger.info(f"Successfully copied file from {source_s3_url} to {destination_s3_url}")
            
            return {
                "success": True,
                "source_s3_url": source_s3_url,
                "destination_s3_url": destination_s3_url,
                "message": "File copied successfully"
            }
            
        except Exception as e:
            logger.error(f"Error copying file from {source_s3_url} to {destination_s3_url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "source_s3_url": source_s3_url,
                "destination_s3_url": destination_s3_url
            }
    
    @staticmethod
    def get_file_info(s3_url: str) -> Dict[str, Union[bool, str, Dict]]:
        """
        Get metadata information about a file in S3
        
        Args:
            s3_url: S3 URL of the file
        
        Returns:
            Dict with file metadata
        """
        if not s3_client.is_initialized():
            logger.error("S3 client not initialized")
            return {
                "success": False,
                "error": "S3 client not initialized"
            }
        
        try:
            # Parse S3 URL
            bucket_name, key = S3Operations.parse_s3_url(s3_url)
            
            # Get object metadata
            response = s3_client.client.head_object(Bucket=bucket_name, Key=key)
            
            return {
                "success": True,
                "s3_url": s3_url,
                "bucket": bucket_name,
                "key": key,
                "size": response.get('ContentLength', 0),
                "content_type": response.get('ContentType', 'application/octet-stream'),
                "last_modified": response.get('LastModified'),
                "etag": response.get('ETag', '').strip('"'),
                "storage_class": response.get('StorageClass', 'STANDARD'),
                "metadata": response.get('Metadata', {}),
                "server_side_encryption": response.get('ServerSideEncryption'),
                "version_id": response.get('VersionId')
            }
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return {
                    "success": False,
                    "error": f"File not found: {s3_url}",
                    "s3_url": s3_url
                }
            logger.error(f"Error getting file info for {s3_url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "s3_url": s3_url
            }
        except Exception as e:
            logger.error(f"Unexpected error getting file info for {s3_url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "s3_url": s3_url
            }

    @staticmethod
    def delete_folder_contents(
        s3_url_prefix: str,
        max_keys: int = None
    ) -> Dict[str, Union[bool, str, int]]:
        """
        Delete all files in an S3 folder/prefix.
        
        Args:
            s3_url_prefix: S3 URL prefix to delete (e.g., "s3://bucket/path/")
            max_keys: Maximum number of keys to process (None = no limit, delete all files)
        
        Returns:
            Dict with deletion results and statistics
        """
        if not s3_client.is_initialized():
            logger.error("S3 client not initialized")
            return {
                "success": False,
                "error": "S3 client not initialized"
            }
        
        try:
            # Parse S3 URL
            bucket_name, key_prefix = S3Operations.parse_s3_url(s3_url_prefix)
            
            # List all objects with the prefix (fetch_all=True by default)
            list_result = S3Operations.list_files(s3_url_prefix, max_keys=max_keys)
            
            if not list_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to list files: {list_result['error']}"
                }
            
            files = list_result["files"]
            
            if not files:
                logger.info(f"No files found to delete in {s3_url_prefix}")
                return {
                    "success": True,
                    "deleted_count": 0,
                    "message": "No files found to delete"
                }
            
            # Delete files in batches (S3 supports up to 1000 objects per batch)
            deleted_count = 0
            failed_count = 0
            batch_size = 1000
            
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                
                # Prepare delete objects list
                delete_objects = []
                for file_info in batch:
                    delete_objects.append({
                        'Key': file_info['key']
                    })
                
                try:
                    # Delete batch
                    response = s3_client.client.delete_objects(
                        Bucket=bucket_name,
                        Delete={
                            'Objects': delete_objects
                        }
                    )
                    
                    # Count successful deletions
                    if 'Deleted' in response:
                        deleted_count += len(response['Deleted'])
                    
                    # Count failed deletions
                    if 'Errors' in response:
                        failed_count += len(response['Errors'])
                        for error in response['Errors']:
                            logger.error(f"Failed to delete {error['Key']}: {error['Message']}")
                
                except Exception as e:
                    logger.error(f"Error deleting batch: {str(e)}")
                    failed_count += len(batch)
            
            logger.info(f"Deleted {deleted_count} files, {failed_count} failed from {s3_url_prefix}")
            
            return {
                "success": failed_count == 0,
                "deleted_count": deleted_count,
                "failed_count": failed_count,
                "total_files": len(files),
                "message": f"Deleted {deleted_count} files, {failed_count} failed"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error deleting folder contents for {s3_url_prefix}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Singleton instance
s3_operations = S3Operations()
