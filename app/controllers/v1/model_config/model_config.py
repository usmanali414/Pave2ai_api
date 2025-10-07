from fastapi import APIRouter, Query
from typing import List, Optional
from app.models.model_config.model_config import ModelConfig, ModelConfigOut, ModelConfigUpdate
from app.services.model_config.model_config import (
    create_model_config_helper,
    get_model_configs_helper,
    update_model_config_helper,
    delete_model_config_helper
)

router = APIRouter()

@router.post("/model-config", response_model=ModelConfigOut)
async def create_model_config(payload: ModelConfig):
    """Create a new model configuration."""
    return await create_model_config_helper(payload)

@router.get("/model-config", response_model=List[ModelConfigOut])
async def list_model_configs(project_id: Optional[str] = Query(None, description="Project ID to filter model configs"), model_id: Optional[str] = Query(None, description="Model ID to filter model configs")):
    """Optional: Get model configurations by project ID or model ID."""
    return await get_model_configs_helper(project_id, model_id)

@router.put("/model-config/{model_id}", response_model=ModelConfigOut)
async def update_model_config(model_id: str, payload: ModelConfigUpdate):
    """Update model configuration by model ID."""
    return await update_model_config_helper(model_id, payload)

@router.delete("/model-config/{model_id}", response_model=dict)
async def delete_model_config(model_id: str):
    """Delete model configuration by model ID."""
    return await delete_model_config_helper(model_id)


