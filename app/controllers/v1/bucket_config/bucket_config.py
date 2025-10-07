from fastapi import APIRouter
from typing import List, Optional

from app.models.bucket_config.bucket_config import BucketConfig, BucketConfigOut
from app.services.bucket_config.bucket_config import (
    create_bucket_config_helper,
    get_bucket_configs_helper,
    update_bucket_config_helper, 
    delete_bucket_config_helper
)


router = APIRouter()

# ---------- BUCKET CONFIG CRUD OPERATIONS ----------

@router.post("/bucket-config", response_model=BucketConfigOut)
async def create_bucket_config(payload: BucketConfig):
    """Create a new bucket configuration."""
    return await create_bucket_config_helper(payload)

@router.get("/bucket-config", response_model=List[BucketConfigOut])
async def get_bucket_config(tenant_id: Optional[str] = None, user_id: Optional[str] = None, project_id: Optional[str] = None, bucket_id: Optional[str] = None):
    """Optional: Get bucket configuration by tenant ID, user ID, project ID, or bucket ID."""
    return await get_bucket_configs_helper(tenant_id, user_id, project_id, bucket_id)

@router.put("/bucket-config/{bucket_id}", response_model=BucketConfigOut)
async def update_bucket_config(bucket_id: str, payload: BucketConfig):
    """Update bucket configuration by bucket ID."""
    return await update_bucket_config_helper(bucket_id, payload)

@router.delete("/bucket-config/{bucket_id}", response_model=dict)
async def delete_bucket_config(bucket_id: str):
    """Delete bucket configuration by bucket ID."""
    return await delete_bucket_config_helper(bucket_id)