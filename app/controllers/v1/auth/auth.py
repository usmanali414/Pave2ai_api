from fastapi import APIRouter, Depends, Request, Response, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.models.auth.auth import LoginPayload, TokenPair
from app.services.auth.auth import (
    register_tenant, login_tenant, login_admin
)
from app.services.auth.auth_utils import verify_refresh_token, create_access_token, create_refresh_token
from app.models.tenant.tenant import Tenant
router = APIRouter()
templates = Jinja2Templates(directory="app/views")


@router.post("/auth/tenant/register", response_model=TokenPair)
async def tenant_register(payload: Tenant):
    return await register_tenant(payload)


@router.post("/auth/tenant/login", response_model=TokenPair)
async def tenant_login(payload: LoginPayload):
    return await login_tenant(payload.email, payload.password)
@router.post("/auth/admin/login", response_model=TokenPair)
async def admin_login(payload: LoginPayload):
    return await login_admin(payload.email, payload.password)


@router.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    return templates.TemplateResponse("admin/login.html", {"request": request})


@router.post("/admin/login")
async def admin_login_submit(request: Request, response: Response, email: str = Form(...), password: str = Form(...)):
    token_pair = await login_admin(email, password)
    # Set HttpOnly cookies
    response = RedirectResponse(url="/admin/dashboard", status_code=303)
    response.set_cookie(key="access_token", value=token_pair.access_token, httponly=True, secure=False, samesite="lax")
    response.set_cookie(key="refresh_token", value=token_pair.refresh_token, httponly=True, secure=False, samesite="lax")
    return response


# @router.post("/auth/user/register", response_model=UserAuthOut)
# async def user_register(payload: User):
#     return await register_user(payload)


# @router.post("/auth/user/login", response_model=UserAuthOut)
# async def user_login(payload: LoginPayload):
#     return await login_user(payload.email, payload.password)


@router.post("/auth/refresh", response_model=dict)
async def refresh(payload: dict):
    token = payload.get("refresh_token")
    data = verify_refresh_token(token)
    subject = data.get("sub")
    access = create_access_token({"sub": subject})
    refresh_token = create_refresh_token(subject)
    return {"token": {"access_token": access, "refresh_token": refresh_token, "token_type": "bearer"}}