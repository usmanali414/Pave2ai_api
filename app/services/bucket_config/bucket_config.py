from app.database.conn import mongo_client
from app.models.bucket_config.bucket_config import BucketConfig, BucketConfigOut
from app.utils.logger_utils import logger
from app.services.s3.s3_helper import s3_helper
from fastapi import HTTPException, status
from bson import ObjectId
from datetime import datetime   
from config import database_config
from app.utils.pass_hashing import hash_password

async def create_bucket_config_helper(payload: BucketConfig) -> BucketConfig:
    logger.info("Creating bucket config")
    
    try:
        # 0) Guard: If a config already exists in DB for this project, return it
        existing_in_db = await mongo_client.database[database_config["BUCKET_CONFIG_COLLECTION"]].find_one({
            "project_id": payload.project_id
        })
        if existing_in_db:
            logger.info(f"Bucket config already exists in DB for project {payload.project_id}")
            existing_in_db["_id"] = str(existing_in_db["_id"])  # normalize id
            return BucketConfigOut(**existing_in_db)

        # 1) Check S3: if the structure (at least raw folder) already exists, abort with conflict
        try:
            raw_exists = s3_helper.check_folder_exists(
                tenant_id=payload.tenant_id,
                project_id=payload.project_id,
                folder_path="data/raw/",
            )
        except Exception:
            raw_exists = False

        if raw_exists:
            logger.warning(
                f"S3 folder structure already present for tenant {payload.tenant_id}, project {payload.project_id}. Aborting create."
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Project is already configured in S3. Please select another project.",
            )

        # Create S3 folder structure for the project
        logger.info(f"Creating S3 folder structure for tenant {payload.tenant_id}, project {payload.project_id}")
        s3_result = s3_helper.create_project_folder_structure(
            tenant_id=payload.tenant_id,
            project_id=payload.project_id
        )
        
        # Log S3 operation result
        if s3_result.get("success"):
            logger.info(f"S3 folders created successfully: {s3_result.get('total_folders')} folders")
        else:
            logger.warning(f"S3 folder creation incomplete: {s3_result.get('error', 'Unknown error')}")
        
        # Convert Pydantic model to dict
        bucket_doc = payload.dict()
        
        # Get folder structure from S3 result and save it
        bucket_doc["folder_structure"] = s3_result.get("folder_structure", {})
        
        # Add metadata
        bucket_doc["created_at"] = datetime.utcnow()
        bucket_doc["updated_at"] = datetime.utcnow()
        bucket_doc["deleted_at"] = None
        
        # Insert into database
        result = await mongo_client.database[database_config["BUCKET_CONFIG_COLLECTION"]].insert_one(bucket_doc)
        
        if result.inserted_id:
            logger.info(f"Bucket config created successfully with ID: {result.inserted_id}")
            bucket_doc["_id"] = str(result.inserted_id)
            return BucketConfigOut(**bucket_doc)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create bucket configuration"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating bucket config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

async def get_bucket_configs_helper(tenant_id: str | None = None, user_id: str | None = None, project_id: str | None = None, bucket_id: str | None = None):
    try:
        query = {}
        # Build query from available filters
        if tenant_id:
            query["tenant_id"] = tenant_id
        # user_id removed from schema; keep param for backward-compat but ignore
        if project_id:
            query["project_id"] = project_id
        if bucket_id:
            query["_id"] = ObjectId(bucket_id)

        docs = await mongo_client.database[database_config["BUCKET_CONFIG_COLLECTION"]].find(query).to_list(length=None)
        items: list[BucketConfigOut] = []
        for doc in docs:
            doc["_id"] = str(doc["_id"]) 
            items.append(BucketConfigOut(**doc))
        return items
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting bucket configs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    
async def update_bucket_config_helper(bucket_id: str, payload: BucketConfig) -> BucketConfig:
    logger.info(f"Updating bucket config: {bucket_id}")
    
    try:
        # Check if bucket exists
        existing_bucket = await mongo_client.database[database_config["BUCKET_CONFIG_COLLECTION"]].find_one({"_id": ObjectId(bucket_id)})
        if not existing_bucket:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Bucket configuration not found"
            )
        
        # Prepare update data
        update_data = {k: v for k, v in payload.dict().items() if v is not None}
        
        # Encrypt sensitive data if being updated
        if "access_key_id" in update_data:
            update_data["access_key_id"] = hash_password(update_data["access_key_id"])
        if "secret_access_key" in update_data:
            update_data["secret_access_key"] = hash_password(update_data["secret_access_key"])
        
        update_data["updated_at"] = datetime.utcnow()
        
        # Update bucket config
        result = await mongo_client.database[database_config["BUCKET_CONFIG_COLLECTION"]].update_one(
            {"_id": ObjectId(bucket_id)},
            {"$set": update_data}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update bucket configuration"
            )
        
        # Return updated bucket
        updated_bucket = await mongo_client.database[database_config["BUCKET_CONFIG_COLLECTION"]].find_one({"_id": ObjectId(bucket_id)})
        updated_bucket["_id"] = str(updated_bucket["_id"]) 
        return BucketConfigOut(**updated_bucket)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating bucket config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    
async def delete_bucket_config_helper(bucket_id: str) -> dict:
    logger.info(f"Deleting bucket config: {bucket_id}")
    
    try:
        # Check if bucket exists
        existing_bucket = await mongo_client.database[database_config["BUCKET_CONFIG_COLLECTION"]].find_one({"_id": ObjectId(bucket_id)})
        if not existing_bucket:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Bucket configuration not found"
            )
        
        result = await mongo_client.database[database_config["BUCKET_CONFIG_COLLECTION"]].delete_one({"_id": ObjectId(bucket_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete bucket configuration"
            )
        
        logger.info(f"Bucket config {bucket_id} deleted successfully")
        return {"message": "Bucket configuration deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting bucket config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

