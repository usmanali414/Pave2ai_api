from app.database.conn import mongo_client
from app.utils.logger_utils import logger
from app.models.train_config.train_config import TrainConfig, TrainConfigOut, TrainConfigUpdate
from fastapi import HTTPException, status
from bson import ObjectId
from datetime import datetime
from config import database_config


def _db():
    return mongo_client.database

def _train_config_col():
    return _db()[database_config["TRAIN_CONFIG_COLLECTION"]]

async def create_train_config_helper(payload: TrainConfig) -> TrainConfigOut:
    try:
        # Enforce single config per project (pre-check to provide friendly error)
        existing = await _train_config_col().find_one({"project_id": payload.project_id})
        if existing:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A train configuration already exists for this project")
        doc = payload.dict()
        doc["created_at"] = datetime.utcnow()
        doc["updated_at"] = datetime.utcnow()
        doc["deleted_at"] = None
        result = await _train_config_col().insert_one(doc)
        if not result.inserted_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create train config")
        doc["_id"] = str(result.inserted_id)
        return TrainConfigOut(**doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"create_train_config_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def get_train_configs_helper(project_id: str | None = None, train_config_id: str | None = None):
    try:
        query = {}
        if train_config_id:
            query["_id"] = ObjectId(train_config_id)
        if project_id:
            query["project_id"] = project_id

        docs = await _train_config_col().find(query).to_list(length=None)
        if not docs:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Train configs not found")

        items: list[TrainConfigOut] = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            items.append(TrainConfigOut(**doc))
        return items
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_train_configs_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def update_train_config_helper(config_id: str, payload: TrainConfigUpdate) -> TrainConfigOut:
    try:
        update_data = {k: v for k, v in payload.dict().items() if v is not None}
        update_data["updated_at"] = datetime.utcnow()
        res = await _train_config_col().update_one({"_id": ObjectId(config_id)}, {"$set": update_data})
        if res.matched_count == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Train config not found")
        doc = await _train_config_col().find_one({"_id": ObjectId(config_id)})
        doc["_id"] = str(doc["_id"])
        return TrainConfigOut(**doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"update_train_config_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def delete_train_config_helper(config_id: str):
    try:
        res = await _train_config_col().delete_one({"_id": ObjectId(config_id)})
        if res.deleted_count == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Train config not found")
        return {"detail": "Train config deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete_train_config_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


