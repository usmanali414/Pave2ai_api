"""
Project deletion service for comprehensive cleanup.
Handles both S3 folder deletion and MongoDB document cleanup.
"""
from typing import Dict, Any
from fastapi import HTTPException, status
from bson import ObjectId
from app.database.conn import mongo_client
from app.database.schema import database_config
from app.services.s3.s3_operations import s3_operations
from app.services.s3.s3_helper import s3_helper
from app.utils.logger_utils import logger
from config import S3_DATA_BUCKET


async def delete_project_completely(tenant_id: str, project_id: str) -> Dict[str, Any]:
    """
    Delete a project completely - both S3 folder structure and MongoDB documents.
    
    Args:
        tenant_id: The tenant ID
        project_id: The project ID to delete
        
    Returns:
        Dictionary with deletion results
        
    Raises:
        HTTPException: If project not found or deletion fails
    """
    try:
        # Step 1: Verify project exists and belongs to tenant
        project = await _get_project_details(project_id)
        if project["tenant_id"] != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Project {project_id} does not belong to tenant {tenant_id}"
            )
        
        logger.info(f"Starting complete deletion for project {project_id} (tenant: {tenant_id})")
        
        # Step 2: Delete S3 folder structure
        s3_deletion_result = await _delete_project_s3_folders(tenant_id, project_id)
        
        # Step 3: Delete bucket_config documents
        bucket_config_deletion_result = await _delete_project_bucket_configs(project_id)
        
        # Step 4: Delete project document
        project_deletion_result = await _delete_project_document(project_id)
        
        # Prepare response
        deletion_summary = {
            "project_id": project_id,
            "tenant_id": tenant_id,
            "deletion_results": {
                "s3_folders": s3_deletion_result,
                "bucket_configs": bucket_config_deletion_result,
                "project_document": project_deletion_result
            },
            "overall_success": (
                s3_deletion_result["success"] and 
                bucket_config_deletion_result["success"] and 
                project_deletion_result["success"]
            )
        }
        
        if deletion_summary["overall_success"]:
            logger.info(f"Successfully deleted project {project_id} completely")
            deletion_summary["message"] = "Project deleted successfully"
        else:
            logger.warning(f"Partial deletion completed for project {project_id}")
            deletion_summary["message"] = "Project deletion completed with some warnings"
        
        return deletion_summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in complete project deletion for {project_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during project deletion: {str(e)}"
        )


async def _get_project_details(project_id: str) -> Dict[str, Any]:
    """Get project details including tenant_id."""
    try:
        project = await mongo_client.database[database_config["PROJECT_COLLECTION"]].find_one({
            "_id": ObjectId(project_id)
        })
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project not found: {project_id}"
            )
        
        return {
            "id": str(project["_id"]),
            "name": project["name"],
            "tenant_id": project["tenant_id"],
            "user_id": project["user_id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project details for {project_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving project details: {str(e)}"
        )


async def _delete_project_s3_folders(tenant_id: str, project_id: str) -> Dict[str, Any]:
    """Delete all S3 folders and files for a project."""
    try:
        # Use the existing S3Helper function
        success = s3_helper.delete_project_folders(tenant_id, project_id)
        
        if success:
            logger.info(f"Successfully deleted S3 folders for project {project_id}")
            return {
                "success": True,
                "message": f"S3 folders deleted for project {project_id}"
            }
        else:
            logger.error(f"Failed to delete S3 folders for project {project_id}")
            return {
                "success": False,
                "message": f"Failed to delete S3 folders for project {project_id}"
            }
            
    except Exception as e:
        logger.error(f"Error deleting S3 folders for project {project_id}: {str(e)}")
        return {
            "success": False,
            "message": f"Error deleting S3 folders: {str(e)}"
        }


async def _delete_project_bucket_configs(project_id: str) -> Dict[str, Any]:
    """Delete all bucket_config documents for a project."""
    try:
        # Find all bucket configs for this project
        bucket_configs = await mongo_client.database[database_config["BUCKET_CONFIG_COLLECTION"]].find({
            "project_id": project_id
        }).to_list(length=None)
        
        if not bucket_configs:
            logger.info(f"No bucket configs found for project {project_id}")
            return {
                "success": True,
                "deleted_count": 0,
                "message": f"No bucket configs found for project {project_id}"
            }
        
        # Delete all bucket configs
        deleted_count = 0
        for bucket_config in bucket_configs:
            try:
                result = await mongo_client.database[database_config["BUCKET_CONFIG_COLLECTION"]].delete_one({
                    "_id": bucket_config["_id"]
                })
                if result.deleted_count > 0:
                    deleted_count += 1
                    logger.info(f"Deleted bucket config {bucket_config['_id']} for project {project_id}")
            except Exception as e:
                logger.error(f"Error deleting bucket config {bucket_config['_id']}: {str(e)}")
        
        logger.info(f"Deleted {deleted_count} bucket configs for project {project_id}")
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} bucket configs for project {project_id}"
        }
        
    except Exception as e:
        logger.error(f"Error deleting bucket configs for project {project_id}: {str(e)}")
        return {
            "success": False,
            "message": f"Error deleting bucket configs: {str(e)}"
        }


async def _delete_project_document(project_id: str) -> Dict[str, Any]:
    """Delete the project document itself."""
    try:
        result = await mongo_client.database[database_config["PROJECT_COLLECTION"]].delete_one({
            "_id": ObjectId(project_id)
        })
        
        if result.deleted_count == 0:
            return {
                "success": False,
                "message": f"Project document not found: {project_id}"
            }
        
        logger.info(f"Successfully deleted project document {project_id}")
        return {
            "success": True,
            "message": f"Project document deleted: {project_id}"
        }
        
    except Exception as e:
        logger.error(f"Error deleting project document {project_id}: {str(e)}")
        return {
            "success": False,
            "message": f"Error deleting project document: {str(e)}"
        }
