from fastapi import APIRouter, Request, HTTPException, status, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.services.auth.auth_utils import verify_access_token
from app.models.tenant.tenant import Tenant
from app.models.project.project import Project
from app.models.user.user import User
from app.services.tenant_management.tenant_helper import get_tenants_helper
from app.services.project_management.project import get_projects_helper
from app.services.user_management.user_helper import get_users_helper
from urllib.parse import quote
from app.services.user_management.user_helper import get_users_helper

router = APIRouter()
templates = Jinja2Templates(directory="app/views")


def _require_admin_cookie(request: Request):
    access = request.cookies.get("access_token")
    if not access:
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER)
    payload = verify_access_token(access)
    if payload.get("scope") != "admin":
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER)
    return payload


@router.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    try:
        _require_admin_cookie(request)
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)
    return templates.TemplateResponse("admin/dashboard.html", {"request": request})


@router.get("/admin/customers", response_class=HTMLResponse)
async def admin_customers(request: Request):
    try:
        _require_admin_cookie(request)
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)
    tenants = await get_tenants_helper()
    users = await get_users_helper(None, None)
    projects = await get_projects_helper(None, None, None)
    active_tab = request.query_params.get("tab", "tenants")
    subtab = request.query_params.get("sub", "tenants")
    return templates.TemplateResponse("customer_management/customers.html", {"request": request, "tenants": tenants, "users": users, "projects": projects, "active_tab": active_tab, "subtab": subtab})


# Removed admin tenant create/update/delete: use API endpoints


# Removed admin project create: use API endpoint


# Removed admin configs create: use API endpoints


# Removed admin user create/update/delete: use API endpoints


@router.get("/admin/users", response_class=HTMLResponse)
async def admin_users_html(request: Request, tenant_id: str | None = None):
    try:
        _require_admin_cookie(request)
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)
    users = await get_users_helper(None, tenant_id)
    def label(u):
        try:
            name = getattr(u, 'name', None)
        except Exception:
            name = None
        if name and str(name).strip():
            return name
        # Fallback to email then id
        try:
            email = getattr(u, 'email', None)
        except Exception:
            email = None
        return email or getattr(u, 'id', '')
    options = "".join([f'<option value="{u.id}">{label(u)}</option>' for u in users])
    return HTMLResponse(content=options)


@router.get("/admin/customers/tenant/{tenant_id}", response_class=HTMLResponse)
async def admin_get_tenant(request: Request, tenant_id: str):
    try:
        _require_admin_cookie(request)
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)
    # Reuse helper to fetch single tenant
    items = await get_tenants_helper(tenant_id, None)
    if not items:
        return HTMLResponse(status_code=404, content="")
    t = items[0]
    # Return minimal JSON for modal prefill
    import json
    return HTMLResponse(content=json.dumps({
        "id": t.id,
        "name": getattr(t, 'name', None),
        "email": t.email,
        "company_name": getattr(t, 'company_name', None),
        "website": getattr(t, 'website', None),
        "phone": getattr(t, 'phone', None),
        "status": getattr(t, 'status', 'active')
    }), media_type="application/json")


@router.get("/admin/customers/user/{user_id}", response_class=HTMLResponse)
async def admin_get_user(request: Request, user_id: str):
    try:
        _require_admin_cookie(request)
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)
    items = await get_users_helper(user_id, None)
    if not items:
        return HTMLResponse(status_code=404, content="")
    u = items[0]
    import json
    return HTMLResponse(content=json.dumps({
        "id": u.id,
        "tenant_id": u.tenant_id,
        "name": getattr(u, 'name', None),
        "email": u.email,
        "role": u.role,
        "status": getattr(u, 'status', 'active')
    }), media_type="application/json")


@router.get("/admin/projects", response_class=HTMLResponse)
async def admin_projects(request: Request):
    try:
        _require_admin_cookie(request)
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)
    projects = await get_projects_helper(None, None, None)
    return templates.TemplateResponse("project_management/projects.html", {"request": request, "projects": projects})


@router.get("/admin/projects/{project_id}", response_class=HTMLResponse)
async def admin_get_project(request: Request, project_id: str):
    try:
        _require_admin_cookie(request)
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)
    items = await get_projects_helper(project_id, None, None)
    if not items:
        return HTMLResponse(status_code=404, content="")
    p = items[0]
    import json
    return HTMLResponse(content=json.dumps({
        "id": p.id,
        "name": p.name,
        "tenant_id": p.tenant_id,
        "user_id": p.user_id,
        "status": getattr(p, 'status', 'active')
    }), media_type="application/json")

@router.get("/admin/project-settings", response_class=HTMLResponse)
async def admin_project_settings(request: Request):
    try:
        _require_admin_cookie(request)
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)
    return templates.TemplateResponse("project_management/project_settings.html", {"request": request})


@router.get("/admin/train-settings", response_class=HTMLResponse)
async def admin_train_settings(request: Request):
    try:
        _require_admin_cookie(request)
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)
    sub = request.query_params.get("sub", "create")
    return templates.TemplateResponse("training_management/train_settings.html", {"request": request, "sub": sub})

