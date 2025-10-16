"""
Project deletion controller endpoints.
"""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Query
from app.services.project_deletion.project_deletion import delete_project_completely
from app.utils.logger_utils import logger

router = APIRouter()


@router.delete("/project/{tenant_id}/{project_id}", response_model=Dict[str, Any])
async def delete_project_complete(tenant_id: str, project_id: str):
    """
    Delete a project completely - both S3 folder structure and MongoDB documents.
    
    This endpoint will:
    1. Delete all S3 folders and files for the project
    2. Delete all bucket_config documents associated with the project
    3. Delete the project document itself
    
    Args:
        tenant_id: The tenant ID
        project_id: The project ID to delete
        
    Returns:
        Dictionary with deletion results and statistics
        
    Raises:
        HTTPException: If project not found or deletion fails
    """
    try:
        logger.info(f"Received request to delete project {project_id} for tenant {tenant_id}")
        
        result = await delete_project_completely(tenant_id, project_id)
        
        if result["overall_success"]:
            return result
        else:
            # Return partial success with warning
            raise HTTPException(
                status_code=status.HTTP_207_MULTI_STATUS,
                detail=result
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in delete_project_complete endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
