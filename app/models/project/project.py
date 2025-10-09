from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, Dict

# ---------- Project Models ----------#

class Project(BaseModel):
    name: str
    user_id: str
    tenant_id: str
    status: str = "active"
    metadata: Optional[Dict]

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    metadata: Optional[Dict] = None

class ProjectOut(Project):
    id: str = Field(..., alias="_id")
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)