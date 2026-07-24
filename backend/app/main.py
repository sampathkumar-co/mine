from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.account_routes import router as account_router
from app.api.auth_middleware import AuthenticationMiddleware
from app.api.billing_middleware import BillingReservationMiddleware
from app.api.camera import router as camera_router
from app.api.governance_routes import router as governance_router
from app.api.memory_routes import router as memory_router
from app.api.observability_middleware import ObservabilityMiddleware
from app.api.operations_routes import router as operations_router
from app.api.platform_routes import router as platform_router
from app.api.rate_limit_middleware import RateLimitMiddleware
from app.api.routes import router
from app.api.subscription_routes import router as subscription_router
from app.api.system_routes import router as system_router
from app.core.config import get_settings
from app.core.database import init_database, migrate_database
from app.services.audit import AuditMiddleware
from app.services.observability import configure_logging

settings = get_settings()
configure_logging(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.environment.casefold() == "production":
        if settings.auth_secret == "development-only-change-me":
            raise RuntimeError("DIRECTOR_AUTH_SECRET must be changed in production")
        if not settings.auth_required:
            raise RuntimeError("DIRECTOR_AUTH_REQUIRED must remain enabled in production")
        if settings.auto_create_schema:
            raise RuntimeError("DIRECTOR_AUTO_CREATE_SCHEMA must be disabled in production")
        if settings.subscriptions_enabled and (
            not settings.stripe_secret_key or not settings.stripe_webhook_secret
        ):
            raise RuntimeError(
                "Stripe secret and webhook keys are required when subscriptions are enabled"
            )
        if settings.rate_limit_enabled and settings.rate_limit_backend.casefold() != "redis":
            raise RuntimeError("Production rate limiting must use the Redis backend")
        if settings.rate_limit_enabled and not settings.rate_limit_fail_closed:
            raise RuntimeError("Production rate limiting must fail closed")
        if settings.metrics_enabled and not settings.metrics_token:
            raise RuntimeError("DIRECTOR_METRICS_TOKEN is required when production metrics are enabled")
        if not settings.readiness_require_redis:
            raise RuntimeError("Production readiness must require Redis")
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    if settings.object_storage_provider.casefold() == "local":
        Path(settings.object_storage_local_dir).mkdir(parents=True, exist_ok=True)
    if settings.run_migrations_on_startup:
        migrate_database()
    elif settings.auto_create_schema:
        init_database()
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(BillingReservationMiddleware)
app.add_middleware(AuditMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Upload-Offset",
        "X-Request-ID",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    ],
)
app.include_router(system_router, prefix=settings.api_v1_prefix)
app.include_router(account_router, prefix=settings.api_v1_prefix)
app.include_router(subscription_router, prefix=settings.api_v1_prefix)
app.include_router(governance_router, prefix=settings.api_v1_prefix)
app.include_router(operations_router, prefix=settings.api_v1_prefix)
app.include_router(platform_router, prefix=settings.api_v1_prefix)
app.include_router(router, prefix=settings.api_v1_prefix)
app.include_router(memory_router, prefix=settings.api_v1_prefix)
app.include_router(camera_router, prefix=settings.api_v1_prefix)
