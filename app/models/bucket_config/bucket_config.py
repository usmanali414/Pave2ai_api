from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional

# ---------- Bucket Config Models ----------#

class BucketConfig(BaseModel):
    tenant_id: str
    project_id: str
    # Single dict containing all folder paths
    folder_structure: Optional[Dict[str, str]] = None
    status: str = "active"

class BucketConfigOut(BucketConfig):
    id: str = Field(..., alias="_id")
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)