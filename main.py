from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.controllers.v1.user_management.user import router as user_router
from app.controllers.v1.bucket_config.bucket_config import router as bucket_config_router
from app.controllers.v1.dataset_config.dataset_config import router as dataset_config_router
from app.controllers.v1.model_config.model_config import router as model_config_router
from app.controllers.v1.tenant_management.tenant import router as tenant_router
from app.controllers.v1.auth.auth import router as auth_router
from app.controllers.v1.project_management.project import router as project_router
from app.controllers.v1.admin.admin import router as admin_router
from app.controllers.v1.train.train import router as train_router
from app.controllers.v1.train_config.train_config import router as train_config_router
from app.database.conn import mongo_client
from app.database.schema import ensure_collections_and_indexes

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    
    # Startup
    print("🚀 Starting up the application...")
    await mongo_client.connect()
    # Ensure DB collections, validators and indexes
    try:
        await ensure_collections_and_indexes()
        print("✅ Ensured DB schema (collections, validators, indexes)")
    except Exception as e:
        print(f"⚠️ Failed to ensure DB schema: {e}")
    
    yield   # 👈 important to allow app to run
    
    # Shutdown
    print("🛑 Shutting down the application...")
    if mongo_client.client:
        await mongo_client.close()

# Create FastAPI application
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router, tags=["Auth"])
app.include_router(tenant_router, tags=["Tenant"])
app.include_router(user_router, tags=["User"])
app.include_router(bucket_config_router, tags=["Bucket Config"])
app.include_router(dataset_config_router, tags=["Dataset Config"])
app.include_router(model_config_router, tags=["Model Config"])
app.include_router(project_router, tags=["Project"])
app.include_router(admin_router, tags=["Admin"])
app.include_router(train_router, tags=["Train"])
app.include_router(train_config_router, tags=["Train Config"])


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/admin/dashboard", status_code=307)