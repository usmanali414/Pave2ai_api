from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, Dict

# ---------- User Models ----------

class User(BaseModel):
    tenant_id: str
    name: Optional[str]
    email: EmailStr
    password: str
    role: str
    status: str = "active"
    metadata: Optional[Dict]


class UserUpdate(BaseModel):
    tenant_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    metadata: Optional[Dict] = None


class UserOut(BaseModel):
    id: str = Field(..., alias="_id")
    tenant_id: str
    name: Optional[str]
    email: EmailStr
    role: str
    status: str = "active"
    metadata: Optional[Dict]
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)