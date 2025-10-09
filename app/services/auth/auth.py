from fastapi import HTTPException, status
from bson import ObjectId
from datetime import datetime

from app.database.conn import mongo_client
from config import database_config
from app.services.auth.auth_utils import hash_password, verify_password, create_access_token, create_refresh_token
from app.models.auth.auth import TokenPair
from app.models.admin.admin import Admin, AdminOut
from app.models.tenant.tenant import Tenant


def _db():
    return mongo_client.database


async def register_tenant(payload: Tenant) -> Tenant:
    # Ensure email uniqueness
    existing = await _db()[database_config["TENANT_COLLECTION"]].find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tenant with this email already exists")

    doc = payload.dict()
    doc["password"] = hash_password(doc["password"])
    doc["created_at"] = datetime.utcnow()
    doc["updated_at"] = datetime.utcnow()
    doc["deleted_at"] = None
    res = await _db()[database_config["TENANT_COLLECTION"]].insert_one(doc)
    if not res.inserted_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to register tenant")

    doc["_id"] = str(res.inserted_id)
    access = create_access_token({"sub": doc["_id"], "scope": "tenant"})
    refresh = create_refresh_token(doc["_id"]) 
    return Tenant(**doc).model_dump(by_alias=True)

async def login_tenant(email: str, password: str) -> TokenPair:
    tenant = await _db()[database_config["TENANT_COLLECTION"]].find_one({"email": email})
    if not tenant or not verify_password(password, tenant.get("password", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    tenant["_id"] = str(tenant["_id"])
    access = create_access_token({"sub": tenant["_id"], "scope": "tenant"})
    refresh = create_refresh_token(tenant["_id"]) 
    return TokenPair(access_token=access, refresh_token=refresh)


async def seed_admin(admin: Admin) -> AdminOut:
    existing = await _db()[database_config["ADMIN_COLLECTION"]].find_one({"email": admin.email})
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Admin with this email already exists")
    doc = admin.dict()
    doc["password"] = hash_password(doc["password"])  
    doc["created_at"] = datetime.utcnow()
    doc["updated_at"] = datetime.utcnow()
    res = await _db()[database_config["ADMIN_COLLECTION"]].insert_one(doc)
    if not res.inserted_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create admin")
    doc["_id"] = str(res.inserted_id)
    return AdminOut(**doc)


async def login_admin(email: str, password: str) -> TokenPair:
    admin = await _db()[database_config["ADMIN_COLLECTION"]].find_one({"email": email})
    if not admin or not verify_password(password, admin.get("password", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    admin["_id"] = str(admin["_id"])
    access = create_access_token({"sub": admin["_id"], "scope": "admin"})
    refresh = create_refresh_token(admin["_id"]) 
    return TokenPair(access_token=access, refresh_token=refresh)


# async def register_user(payload: User) -> User:
#     existing = await _db()[database_config["USER_COLLECTION"]].find_one({"email": payload.email})
#     if existing:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User with this email already exists")
#     # Ensure tenant exists
#     tenant_exists = await _db()[database_config["TENANT_COLLECTION"]].find_one({"_id": ObjectId(payload.tenant_id)})
#     if not tenant_exists:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tenant_id")

#     doc = payload.dict()
#     doc["password"] = hash_password(doc["password"])
#     doc["created_at"] = datetime.utcnow()
#     doc["updated_at"] = datetime.utcnow()
#     doc["deleted_at"] = None
#     res = await _db()[database_config["USER_COLLECTION"]].insert_one(doc)
#     if not res.inserted_id:
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to register user")
#     doc["_id"] = str(res.inserted_id)
#     access = create_access_token({"sub": doc["_id"], "scope": "user", "tenant_id": doc.get("tenant_id")})
#     refresh = create_refresh_token(doc["_id"]) 
#     return UserAuthOut(**UserOut(**doc).model_dump(by_alias=True), token=TokenPair(access_token=access, refresh_token=refresh))


# async def login_user(email: str, password: str) -> UserAuthOut:
#     user = await _db()[database_config["USER_COLLECTION"]].find_one({"email": email})
#     if not user or not verify_password(password, user.get("password", "")):
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
#     user["_id"] = str(user["_id"])
#     access = create_access_token({"sub": user["_id"], "scope": "user", "tenant_id": user.get("tenant_id")})
#     refresh = create_refresh_token(user["_id"]) 
#     return UserAuthOut(**UserOut(**user).model_dump(by_alias=True), token=TokenPair(access_token=access, refresh_token=refresh))


# async def refresh_tokens(refresh_token: str) -> TokenPair:
#     # Minimal: trust refresh token verification at route and re-issue access
#     # Verification is done via auth_utils.verify_refresh_token in route
#     raise NotImplementedError


