from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, List

# ---------- Dataset Configs ----------#

class DatasetClass(BaseModel):
    id: str
    name: str
    color_bgr: List[int]
    is_background: Optional[bool] = False
    raw_labels: Optional[List[str]] = []

class DatasetConfig(BaseModel):
    project_id: str
    classes: List[DatasetClass]
    labels_dict: Dict[str, int]
    metadata: Optional[Dict] = None


class DatasetConfigOut(DatasetConfig):
    id: str = Field(..., alias="_id")
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)


class DatasetConfigUpdate(BaseModel):
    project_id: Optional[str] = None
    classes: Optional[List[DatasetClass]] = None
    labels_dict: Optional[Dict[str, int]] = None
    metadata: Optional[Dict] = None

