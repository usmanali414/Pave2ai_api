from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from app.models.user.user import User, UserOut, UserUpdate
from app.services.user_management.user_helper import create_user_helper, get_users_helper, update_user_helper, delete_user_helper
from app.services.auth.auth_utils import require_scope


router = APIRouter()

# ---------- USER CRUD OPERATIONS ----------

@router.post("/user", response_model=UserOut)
async def create_user(payload: User):
    """Create a new user."""
    return await create_user_helper(payload)

@router.get("/user", response_model=List[UserOut])
async def get_users(user_id: Optional[str] = Query(None, description="Optional key to filter by"), tenant_id: Optional[str] = Query(None, description="Optional key to filter by")):
    """Optional: Get a user by ID, tenant ID, or all users"""
    return await get_users_helper(user_id, tenant_id)

@router.put("/user/{user_id}", response_model=UserOut)
async def update_user(user_id: str, payload: UserUpdate):
    """Update a user by ID."""
    return await update_user_helper(user_id, payload)

@router.delete("/user/{user_id}", response_model=dict)
async def delete_user(user_id: str):
    """Delete a user by ID."""
    return await delete_user_helper(user_id)