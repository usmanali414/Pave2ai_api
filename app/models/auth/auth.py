from pydantic import BaseModel, EmailStr

# ---------- Auth Schemas ----------#

class LoginPayload(BaseModel):
    email: EmailStr
    password: str

class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"