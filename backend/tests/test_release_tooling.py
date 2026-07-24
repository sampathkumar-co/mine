from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_doctor = load_module("director_release_doctor", ROOT / "ops" / "release_doctor.py")
release_manifest = load_module("director_release_manifest", ROOT / "ops" / "release_manifest.py")


def production_env() -> dict[str, str]:
    return {
        "DIRECTOR_ENVIRONMENT": "production",
        "DIRECTOR_PUBLIC_APP_URL": "https://director.example.test",
        "DIRECTOR_DOMAIN": "director.example.test",
        "DIRECTOR_CORS_ORIGINS": "https://director.example.test",
        "DIRECTOR_AUTH_REQUIRED": "true",
        "DIRECTOR_AUTH_SECRET": "a-secure-release-secret-with-more-than-32-characters",
        "DIRECTOR_REQUIRE_VERIFIED_EMAIL": "true",
        "DIRECTOR_AUTO_CREATE_SCHEMA": "false",
        "DIRECTOR_DATABASE_URL": "postgresql+psycopg://release:strong-password@postgres/director",
        "DIRECTOR_REDIS_URL": "redis://redis:6379/0",
        "DIRECTOR_RATE_LIMIT_ENABLED": "true",
        "DIRECTOR_RATE_LIMIT_BACKEND": "redis",
        "DIRECTOR_RATE_LIMIT_FAIL_CLOSED": "true",
        "DIRECTOR_TRUST_PROXY_HEADERS": "true",
        "DIRECTOR_READINESS_REQUIRE_REDIS": "true",
        "DIRECTOR_METRICS_ENABLED": "true",
        "DIRECTOR_METRICS_TOKEN": "a-secure-metrics-token-with-sufficient-length",
        "DIRECTOR_EMAIL_PROVIDER": "smtp",
        "DIRECTOR_SMTP_HOST": "smtp.example.test",
        "DIRECTOR_SMTP_USERNAME": "director",
        "DIRECTOR_SMTP_PASSWORD": "smtp-password",
        "DIRECTOR_SMTP_FROM_EMAIL": "deliver@example.test",
        "DIRECTOR_ACME_EMAIL": "operator@example.test",
        "DIRECTOR_OBJECT_STORAGE_PROVIDER": "s3",
        "DIRECTOR_S3_BUCKET": "director-production",
        "DIRECTOR_S3_REGION": "us-east-1",
        "DIRECTOR_SUBSCRIPTIONS_ENABLED": "true",
        "DIRECTOR_BILLING_PROVIDER": "stripe",
        "DIRECTOR_STRIPE_SECRET_KEY": "sk_test_valid_for_configuration_test",
        "DIRECTOR_STRIPE_WEBHOOK_SECRET": "whsec_valid_for_configuration_test",
        "DIRECTOR_BILLING_PLANS_JSON": (
            '{"creator":{"price_id":"price_creator_live"},'
            '"studio":{"price_id":"price_studio_live"}}'
        ),
        "DIRECTOR_REQUIRE_TRANSCRIPTION": "true",
        "DIRECTOR_OPENAI_API_KEY": "provider-key-for-configuration-test",
        "DIRECTOR_BACKUP_DIR": "/mnt/off-host/director-backups",
    }


def test_release_doctor_accepts_complete_production_configuration() -> None:
    checks = release_doctor.local_checks(production_env())
    assert not [item for item in checks if item.status != "pass"]
    assert release_doctor.summary(checks) == {
        "pass": len(checks),
        "fail": 0,
        "pending": 0,
        "total": len(checks),
    }


def test_release_doctor_rejects_unsafe_core_configuration() -> None:
    env = production_env()
    env.update(
        {
            "DIRECTOR_AUTH_REQUIRED": "false",
            "DIRECTOR_AUTO_CREATE_SCHEMA": "true",
            "DIRECTOR_RATE_LIMIT_BACKEND": "memory",
            "DIRECTOR_METRICS_TOKEN": "",
        }
    )
    checks = release_doctor.local_checks(env)
    failed = {item.code for item in checks if item.status == "fail"}
    assert {
        "auth.required",
        "database.auto_create",
        "security.rate_limit",
        "observability.metrics",
    } <= failed


def test_release_doctor_keeps_external_credentials_visible_as_pending() -> None:
    env = production_env()
    env.update(
        {
            "DIRECTOR_EMAIL_PROVIDER": "database",
            "DIRECTOR_OBJECT_STORAGE_PROVIDER": "local",
            "DIRECTOR_SUBSCRIPTIONS_ENABLED": "false",
            "DIRECTOR_REQUIRE_TRANSCRIPTION": "false",
            "DIRECTOR_BACKUP_DIR": "./backups",
        }
    )
    checks = release_doctor.local_checks(env)
    pending = {item.code for item in checks if item.status == "pending"}
    assert {
        "external.smtp",
        "external.object_storage",
        "external.stripe",
        "external.transcription",
        "external.backup_destination",
    } <= pending
    markdown = release_doctor.render_markdown(checks, allow_external_pending=True)
    assert "**Decision:** GO" in markdown
    assert "PENDING" in markdown


def test_release_manifest_is_versioned_and_content_addressed() -> None:
    manifest = release_manifest.build_manifest(ROOT)
    assert manifest["version"] == "1.0.0"
    assert manifest["backend"]["version"] == "1.0.0"
    assert manifest["frontend"]["version"] == "1.0.0"
    assert manifest["source_file_count"] > 50
    assert len(manifest["manifest_sha256"]) == 64
    paths = {item["path"] for item in manifest["source_files"]}
    assert "VERSION" in paths
    assert "ops/release_doctor.py" in paths
