from fastapi import APIRouter, Query, Depends
from typing import List, Optional
from app.models.tenant.tenant import Tenant, TenantOut, TenantUpdate
from app.services.auth.auth_utils import require_scope
from app.services.tenant_management.tenant_helper import (
    create_tenant_helper,
    get_tenants_helper,
    update_tenant_helper,
    delete_tenant_helper,
)


router = APIRouter()

@router.post("/tenant", response_model=TenantOut)
async def create_tenant(payload: Tenant):
    """Create a new tenant."""
    return await create_tenant_helper(payload)

@router.get("/tenant", response_model=List[TenantOut])
async def list_tenants(tenant_id: Optional[str] = Query(None, description="Tenant ID to filter"), email: Optional[str] = Query(None, description="Email to filter")):
    """Optional: List tenants by tenant ID, email, or all tenants."""
    return await get_tenants_helper(tenant_id, email)

@router.put("/tenant/{tenant_id}", response_model=TenantOut)
async def update_tenant(tenant_id: str, payload: TenantUpdate):
    """Update a tenant by tenant ID."""
    return await update_tenant_helper(tenant_id, payload)

@router.delete("/tenant/{tenant_id}", response_model=dict)
async def delete_tenant(tenant_id: str):
    """Delete a tenant by tenant ID."""
    return await delete_tenant_helper(tenant_id)


