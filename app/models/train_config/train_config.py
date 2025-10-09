from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict


class TrainConfig(BaseModel):
    name: str
    tenant_id: str
    project_id: str
    model_version: str
    metadata: Dict  # expected keys: data_parser, model_name, initial_weights(bool)


class TrainConfigOut(TrainConfig):
    id: str = Field(..., alias="_id")
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)


class TrainConfigUpdate(BaseModel):
    name: Optional[str] = None
    tenant_id: Optional[str] = None
    project_id: Optional[str] = None
    model_version: Optional[str] = None
    metadata: Optional[Dict] = None


