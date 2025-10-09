from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, List

# ---------- Model Configs ----------#

class ModelClass(BaseModel):
    id: str
    name: str
    color_bgr: List[int]
    is_background: Optional[bool] = False
    raw_labels: Optional[List[str]] = []

    # @field_validator("color_bgr")
    # @classmethod
    # def validate_color(cls, v: List[int]):
    #     if len(v) != 3 or any((not isinstance(c, int) or c < 0 or c > 255) for c in v):
    #         raise ValueError("color_bgr must be a list of 3 ints in [0,255]")
    #     return v

class ModelConfig(BaseModel):
    project_id: str
    name: str
    status: str = "active"
    description: Optional[str] = None
    model_path: str
    model_name: str
    metadata: Optional[Dict] = None
    # Avoid warnings for fields starting with "model_"
    model_config = ConfigDict(protected_namespaces=())


class ModelConfigOut(ModelConfig):
    id: str = Field(..., alias="_id")
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)


class ModelConfigUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    metadata: Optional[Dict] = None
    model_config = ConfigDict(protected_namespaces=())