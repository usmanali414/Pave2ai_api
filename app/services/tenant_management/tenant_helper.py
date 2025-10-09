from app.database.conn import mongo_client
from app.utils.logger_utils import logger
from app.models.tenant.tenant import Tenant, TenantOut, TenantUpdate
from app.services.auth.auth_utils import hash_password
from fastapi import HTTPException, status
from bson import ObjectId
from datetime import datetime
from config import database_config
from typing import Optional


def _db():
    return mongo_client.database


async def create_tenant_helper(payload: Tenant) -> TenantOut:
    try:
        doc = payload.dict()
        doc["password"] = hash_password(doc["password"]) 
        doc["created_at"] = datetime.utcnow()
        doc["updated_at"] = datetime.utcnow()
        doc["deleted_at"] = None
        result = await _db()[database_config["TENANT_COLLECTION"]].insert_one(doc)
        if not result.inserted_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create tenant")
        doc["_id"] = str(result.inserted_id)
        return TenantOut(**doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"create_tenant_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def get_tenants_helper(tenant_id: Optional[str] | None = None, email: Optional[str] | None = None):
    try:
        query = {}
        if tenant_id:
            query["_id"] = ObjectId(tenant_id)
        if email:
            query["email"] = email
        cursor = _db()[database_config["TENANT_COLLECTION"]].find(query)
        items: list[TenantOut] = []
        async for doc in cursor:
            doc["_id"] = str(ObjectId(doc["_id"]))
            items.append(TenantOut(**doc))
        return items
    except Exception as e:
        logger.error(f"get_tenants_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def update_tenant_helper(tenant_id: str, payload: TenantUpdate) -> TenantOut:
    try:
        update_data = {k: v for k, v in payload.dict().items() if v is not None}
        # Prevent duplicate email error when email unchanged: no special handling needed here as we set what's provided.
        if "password" in update_data and update_data["password"]:
            update_data["password"] = hash_password(update_data["password"])
        else:
            update_data.pop("password", None)
        update_data["updated_at"] = datetime.utcnow()
        res = await _db()[database_config["TENANT_COLLECTION"]].update_one({"_id": ObjectId(tenant_id)}, {"$set": update_data})
        if res.matched_count == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")
        doc = await _db()[database_config["TENANT_COLLECTION"]].find_one({"_id": ObjectId(tenant_id)})
        doc["_id"] = str(ObjectId(doc["_id"]))
        return TenantOut(**doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"update_tenant_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def delete_tenant_helper(tenant_id: str):
    try:
        res = await _db()[database_config["TENANT_COLLECTION"]].delete_one({"_id": ObjectId(tenant_id)})
        if res.deleted_count == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")
        return {"detail": "Tenant deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete_tenant_helper error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


