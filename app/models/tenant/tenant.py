from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, Dict

# ---------- Tenant Models ----------

class Tenant(BaseModel):
    name: Optional[str]
    email: EmailStr
    password: str
    company_name: Optional[str]
    website: Optional[str]
    phone: Optional[str]
    status: str = "active"
    metadata: Optional[Dict]


class TenantUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    company_name: Optional[str] = None
    website: Optional[str] = None
    phone: Optional[str] = None
    status: Optional[str] = None
    metadata: Optional[Dict] = None


class TenantOut(BaseModel):
    id: str = Field(..., alias="_id")
    name: Optional[str]
    email: EmailStr
    company_name: Optional[str]
    website: Optional[str]
    phone: Optional[str]
    status: str = "active"
    metadata: Optional[Dict]
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)