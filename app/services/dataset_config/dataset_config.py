from app.database.conn import mongo_client
from app.utils.logger_utils import logger
from app.models.dataset_config.dataset_config import DatasetConfig, DatasetConfigOut, DatasetConfigUpdate
from fastapi import HTTPException, status
from bson import ObjectId
from datetime import datetime
from config import database_config

def _db():
    return mongo_client.database

async def create_dataset_config_helper(payload: DatasetConfig) -> DatasetConfigOut:
    try:
        doc = payload.dict()
        doc["created_at"] = datetime.utcnow()
        doc["updated_at"] = datetime.utcnow()
        doc["deleted_at"] = None
        result = await _db()[database_config["DATASET_CONFIG_COLLECTION"]].insert_one(doc)
        if not result.inserted_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create dataset config")
        doc["_id"] = str(result.inserted_id)
        return DatasetConfigOut(**doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"create_dataset_config_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def get_dataset_configs_helper(project_id: str | None = None, dataset_id: str | None = None):
    try:
        query = {}
        # Optimized query construction
        if dataset_id:
            query["_id"] = ObjectId(dataset_id)
        if project_id:
            query["project_id"] = project_id

        docs = await _db()[database_config["DATASET_CONFIG_COLLECTION"]].find(query).to_list(length=None)
        if not docs:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset configs not found")

        items: list[DatasetConfigOut] = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            items.append(DatasetConfigOut(**doc))
        return items
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_dataset_configs_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

async def update_dataset_config_helper(config_id: str, payload: DatasetConfigUpdate) -> DatasetConfigOut:
    try:
        update_data = {k: v for k, v in payload.dict().items() if v is not None}
        update_data["updated_at"] = datetime.utcnow()
        res = await _db()[database_config["DATASET_CONFIG_COLLECTION"]].update_one({"_id": ObjectId(config_id)}, {"$set": update_data})
        if res.matched_count == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset config not found")
        if res.modified_count == 0:
            # No changes, still return current doc
            pass
        doc = await _db()[database_config["DATASET_CONFIG_COLLECTION"]].find_one({"_id": ObjectId(config_id)})
        doc["_id"] = str(doc["_id"])
        return DatasetConfigOut(**doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"update_dataset_config_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def delete_dataset_config_helper(config_id: str):
    try:
        res = await _db()[database_config["DATASET_CONFIG_COLLECTION"]].delete_one({"_id": ObjectId(config_id)})
        if res.deleted_count == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset config not found")
        return {"detail": "Dataset config deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete_dataset_config_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

