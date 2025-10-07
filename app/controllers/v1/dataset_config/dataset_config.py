from fastapi import APIRouter, Query
from typing import List, Optional
from app.models.dataset_config.dataset_config import DatasetConfig, DatasetConfigOut, DatasetConfigUpdate
from app.services.dataset_config.dataset_config import (
    create_dataset_config_helper,
    get_dataset_configs_helper,
    update_dataset_config_helper,
    delete_dataset_config_helper
)

router = APIRouter()

@router.post("/dataset-config", response_model=DatasetConfigOut)
async def create_dataset_config(payload: DatasetConfig):
    """Create a new dataset configuration."""
    return await create_dataset_config_helper(payload)

@router.get("/dataset-config", response_model=List[DatasetConfigOut])
async def list_dataset_configs(project_id: Optional[str] = Query(None, description="Project ID to filter dataset configs"), dataset_id: Optional[str] = Query(None, description="Dataset ID to filter dataset configs")):
    """Optional: Get dataset configurations by project ID or dataset ID."""
    return await get_dataset_configs_helper(project_id, dataset_id)

@router.put("/dataset-config/{dataset_id}", response_model=DatasetConfigOut)
async def update_dataset_config(dataset_id: str, payload: DatasetConfigUpdate):
    """Update dataset configuration by dataset ID."""
    return await update_dataset_config_helper(dataset_id, payload)

@router.delete("/dataset-config/{dataset_id}", response_model=dict)
async def delete_dataset_config(dataset_id: str):
    """Delete dataset configuration by dataset ID."""
    return await delete_dataset_config_helper(dataset_id)

