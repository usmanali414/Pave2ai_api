from app.database.conn import mongo_client
from app.utils.logger_utils import logger
from app.models.project.project import Project, ProjectOut, ProjectUpdate
from fastapi import HTTPException, status
from bson import ObjectId
from datetime import datetime
from config import database_config
from typing import Optional, List


def _db():
    return mongo_client.database


async def create_project_helper(payload: Project) -> ProjectOut:
    try:
        doc = payload.dict()
        doc["created_at"] = datetime.utcnow()
        doc["updated_at"] = datetime.utcnow()
        doc["deleted_at"] = None
        result = await _db()[database_config["PROJECT_COLLECTION"]].insert_one(doc)
        if not result.inserted_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create project")
        doc["_id"] = str(result.inserted_id)
        return ProjectOut(**doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"create_project_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def get_projects_helper(project_id: Optional[str] | None = None, tenant_id: Optional[str] | None = None, user_id: Optional[str] | None = None) -> List[ProjectOut]:
    try:
        query = {}
        if project_id:
            query["_id"] = ObjectId(project_id)
        if tenant_id:
            query["tenant_id"] = tenant_id
        if user_id:
            query["user_id"] = user_id

        cursor = _db()[database_config["PROJECT_COLLECTION"]].find(query)
        items: list[ProjectOut] = []
        async for doc in cursor:
            doc["_id"] = str(ObjectId(doc["_id"]))
            items.append(ProjectOut(**doc))
        return items
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_projects_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def update_project_helper(project_id: str, payload: ProjectUpdate) -> ProjectOut:
    try:
        update_data = {k: v for k, v in payload.dict().items() if v is not None}
        update_data["updated_at"] = datetime.utcnow()
        res = await _db()[database_config["PROJECT_COLLECTION"]].update_one({"_id": ObjectId(project_id)}, {"$set": update_data})
        if res.matched_count == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        doc = await _db()[database_config["PROJECT_COLLECTION"]].find_one({"_id": ObjectId(project_id)})
        doc["_id"] = str(ObjectId(doc["_id"]))
        return ProjectOut(**doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"update_project_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def delete_project_helper(project_id: str) -> dict:
    try:
        res = await _db()[database_config["PROJECT_COLLECTION"]].delete_one({"_id": ObjectId(project_id)})
        if res.deleted_count == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        return {"detail": "Project deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete_project_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


