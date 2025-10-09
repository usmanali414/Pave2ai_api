from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Admin(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str]
    status: str = "active"


class AdminOut(BaseModel):
    id: str = Field(..., alias="_id")
    email: EmailStr
    name: Optional[str]
    status: str = "active"

