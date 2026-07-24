from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth_middleware import AuthenticationMiddleware
from app.api.camera import router as camera_router
from app.api.memory_routes import router as memory_router
from app.api.platform_routes import router as platform_router
from app.api.routes import router
from app.core.config import get_settings
from app.core.database import init_database

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.environment.casefold() == "production" and settings.auth_secret == "development-only-change-me":
        raise RuntimeError("DIRECTOR_AUTH_SECRET must be changed in production")
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    if settings.auto_create_schema:
        init_database()
    yield


app = FastAPI(title=settings.app_name, version="0.11.0", lifespan=lifespan)
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Upload-Offset"],
)
app.include_router(platform_router, prefix=settings.api_v1_prefix)
app.include_router(router, prefix=settings.api_v1_prefix)
app.include_router(memory_router, prefix=settings.api_v1_prefix)
app.include_router(camera_router, prefix=settings.api_v1_prefix)
