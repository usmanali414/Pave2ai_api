from fastapi import APIRouter, Query
from typing import List, Optional

from app.models.project.project import Project, ProjectOut, ProjectUpdate
from app.services.project_management.project import (
    create_project_helper,
    get_projects_helper,
    update_project_helper,
    delete_project_helper,
)


router = APIRouter()


@router.post("/project", response_model=ProjectOut)
async def create_project(payload: Project):
    return await create_project_helper(payload)


@router.get("/project", response_model=List[ProjectOut])
async def list_projects(
    project_id: Optional[str] = Query(None, description="Project ID to filter"),
    tenant_id: Optional[str] = Query(None, description="Tenant ID to filter"),
    user_id: Optional[str] = Query(None, description="User ID to filter"),
):
    return await get_projects_helper(project_id, tenant_id, user_id)


@router.put("/project/{project_id}", response_model=ProjectOut)
async def update_project(project_id: str, payload: ProjectUpdate):
    return await update_project_helper(project_id, payload)


@router.delete("/project/{project_id}", response_model=dict)
async def delete_project(project_id: str):
    return await delete_project_helper(project_id)


