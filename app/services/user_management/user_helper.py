from app.database.conn import mongo_client
from app.utils.logger_utils import logger
from app.models.user.user import User, UserOut, UserUpdate
from app.services.auth.auth_utils import hash_password
from fastapi import HTTPException, status
from bson import ObjectId
from datetime import datetime
from config import database_config
from typing import Optional

# Dependency to get database instance
def get_database(): 
    """Get database instance from MongoDB client."""
    return mongo_client.database

# ---------- USER CRUD OPERATIONS ----------
async def create_user_helper(payload: User):
    logger.info(f"Attempting to create user with email: {payload.email}")
    
    try:
        # Check if user with email already exists
        existing_user = await get_database()[database_config["USER_COLLECTION"]].find_one({"email": payload.email})
        logger.info(f"Existing user: {existing_user}")
        if existing_user:
            logger.warning(f"User with email {payload.email} already exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create user document (user_id will be set to MongoDB's _id after insertion)
        payload_dict = payload.dict()
        payload_dict["created_at"] = datetime.utcnow()
        payload_dict["updated_at"] = datetime.utcnow()
        payload_dict["deleted_at"] = None
        
        # Hash password
        payload_dict["password"] = hash_password(payload_dict["password"]) 
        # Insert into database
        result = await get_database()[database_config["USER_COLLECTION"]].insert_one(payload_dict)
        
        if result.inserted_id:
            logger.info(f"User created successfully with ID: {result.inserted_id}")
            payload_dict["_id"] = str(ObjectId(result.inserted_id))
            logger.info(f"User document: {payload_dict}")
            # Return the API response shape expected by response_model=UserOut
            return UserOut(**payload_dict)
        else:
            logger.error("Failed to create user - no inserted ID returned")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


async def get_users_helper(user_id: Optional[str] | None = None, tenant_id: Optional[str] | None = None):
    """Get users as a list, filtered by optional user_id and/or tenant_id."""
    try:
        query = {}
        if user_id:
            query["_id"] = ObjectId(user_id)
        if tenant_id:
            query["tenant_id"] = tenant_id

        cursor = mongo_client.database[database_config["USER_COLLECTION"]].find(query)
        users: list[UserOut] = []
        async for user in cursor:
            user["_id"] = str(ObjectId(user["_id"]))
            users.append(UserOut(**user))
        return users
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

async def update_user_helper(user_id: str, payload: UserUpdate):
    """Update a user by ID."""
    existing_user = await get_database()[database_config["USER_COLLECTION"]].find_one({"_id": ObjectId(user_id)})
    # Check if email is being updated and if it already exists
    if payload.email and payload.email != existing_user["email"]:
        email_exists = await get_database()[database_config["USER_COLLECTION"]].find_one({"email": payload.email})
        if email_exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
    
    # Prepare update data
    update_data = {k: v for k, v in payload.dict().items() if v is not None}
    if "password" in update_data and update_data["password"]:
        update_data["password"] = hash_password(update_data["password"]) 
    else:
        update_data.pop("password", None)
    update_data["updated_at"] = datetime.utcnow()
    
    # Update user
    result = await get_database()[database_config["USER_COLLECTION"]].update_one(
        {"_id": ObjectId(user_id)},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Return updated customer
    updated_user = await get_database()[database_config["USER_COLLECTION"]].find_one({"_id": ObjectId(user_id)})
    updated_user["_id"] = str(updated_user["_id"])
    return UserOut(**updated_user)


async def delete_user_helper(user_id: str):
    """Delete a user by ID."""  
    try:
        # Check if user exists first
        existing_user = await get_database()[database_config["USER_COLLECTION"]].find_one({"_id": ObjectId(user_id)})
        if not existing_user:
            raise HTTPException(    
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        result = await get_database()[database_config["USER_COLLECTION"]].delete_one({"_id": ObjectId(user_id)})
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user"
            )
        
        logger.info(f"User {user_id} deleted successfully")
        return {"message": "User deleted successfully"}
        # Don't return anything for 204 status code
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )