import bcrypt
from datetime import datetime, timedelta, timezone
from jose import jwt
from config import JWT_CONFIG
from fastapi import HTTPException, Depends
from fastapi import status
from jose import JWTError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# --------------------------------------------------------------------
# Config (you can load these from environment variables)
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Password hashing
# --------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


# --------------------------------------------------------------------
# JWT token creation
# --------------------------------------------------------------------
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a short-lived JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=JWT_CONFIG["ACCESS_TOKEN_EXPIRE_MINUTES"]))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_CONFIG["JWT_SECRET_KEY"], algorithm=JWT_CONFIG["JWT_ALGORITHM"])


def create_refresh_token(user_id: str, expires_delta: timedelta | None = None) -> str:
    """Create a long-lived JWT refresh token."""
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(days=JWT_CONFIG["REFRESH_TOKEN_EXPIRE_DAYS"]))
    to_encode = {"sub": user_id, "exp": expire}
    return jwt.encode(to_encode, JWT_CONFIG["JWT_REFRESH_SECRET_KEY"], algorithm=JWT_CONFIG["JWT_ALGORITHM"])


# --------------------------------------------------------------------
# Token verification
# --------------------------------------------------------------------
def verify_access_token(token: str) -> dict:
    """Verify and decode an access token."""
    try:
        payload = jwt.decode(token, JWT_CONFIG["JWT_SECRET_KEY"], algorithms=[JWT_CONFIG["JWT_ALGORITHM"]])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"}, 
        )


security_scheme = HTTPBearer()


def require_scope(required_scope: str):
    def dependency(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
        token = credentials.credentials
        payload = verify_access_token(token)
        scope = payload.get("scope")
        if scope != required_scope:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return payload
    return dependency


def verify_refresh_token(token: str) -> dict:
    """Verify and decode a refresh token."""
    try:
        payload = jwt.decode(token, JWT_CONFIG["JWT_REFRESH_SECRET_KEY"], algorithms=[JWT_CONFIG["JWT_ALGORITHM"]])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )