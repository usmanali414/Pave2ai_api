from fastapi import APIRouter, Query
from typing import List, Optional
from app.models.train_config.train_config import TrainConfig, TrainConfigOut, TrainConfigUpdate
from app.services.train_config.train_config import (
    create_train_config_helper,
    get_train_configs_helper,
    update_train_config_helper,
    delete_train_config_helper,
)


router = APIRouter()


@router.post("/train-config", response_model=TrainConfigOut)
async def create_train_config(payload: TrainConfig):
    return await create_train_config_helper(payload)


@router.get("/train-config", response_model=List[TrainConfigOut])
async def list_train_configs(project_id: Optional[str] = Query(None, description="Project ID to filter train configs"), train_config_id: Optional[str] = Query(None, description="Train Config ID to filter")):
    return await get_train_configs_helper(project_id, train_config_id)


@router.put("/train-config/{config_id}", response_model=TrainConfigOut)
async def update_train_config(config_id: str, payload: TrainConfigUpdate):
    return await update_train_config_helper(config_id, payload)


@router.delete("/train-config/{config_id}", response_model=dict)
async def delete_train_config(config_id: str):
    return await delete_train_config_helper(config_id)


