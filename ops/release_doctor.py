#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

Status = Literal["pass", "fail", "pending"]


@dataclass(frozen=True, slots=True)
class Check:
    code: str
    status: Status
    message: str
    remediation: str | None = None
    external: bool = False


PLACEHOLDERS = {
    "",
    "change-me",
    "changeme",
    "development-only-change-me",
    "replace-me",
    "replace-with-at-least-32-random-characters",
    "admin@example.com",
    "price_creator_replace",
    "price_studio_replace",
}
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"Environment file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    for key, value in os.environ.items():
        if key.startswith("DIRECTOR_"):
            values[key] = value
    return values


def truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def placeholder(value: str | None) -> bool:
    normalized = str(value or "").strip()
    if normalized.casefold() in PLACEHOLDERS:
        return True
    return "replace" in normalized.casefold() or normalized.casefold().endswith("example.com")


def _check(
    condition: bool,
    code: str,
    passed: str,
    failed: str,
    remediation: str,
    *,
    external: bool = False,
    pending_when_false: bool = False,
) -> Check:
    if condition:
        return Check(code=code, status="pass", message=passed, external=external)
    status: Status = "pending" if pending_when_false else "fail"
    return Check(
        code=code,
        status=status,
        message=failed,
        remediation=remediation,
        external=external,
    )


def _https_origin(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc) and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
    }


def _price_ids_configured(raw: str) -> bool:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    paid = [payload.get("creator", {}), payload.get("studio", {})]
    return all(
        isinstance(item, dict)
        and str(item.get("price_id") or "").startswith("price_")
        and not placeholder(str(item.get("price_id") or ""))
        for item in paid
    )


def local_checks(env: dict[str, str]) -> list[Check]:
    public_url = env.get("DIRECTOR_PUBLIC_APP_URL", "")
    domain = env.get("DIRECTOR_DOMAIN", "")
    cors_origins = [
        item.strip()
        for item in env.get("DIRECTOR_CORS_ORIGINS", "").split(",")
        if item.strip()
    ]
    auth_secret = env.get("DIRECTOR_AUTH_SECRET", "")
    database_url = env.get("DIRECTOR_DATABASE_URL", "")
    metrics_token = env.get("DIRECTOR_METRICS_TOKEN", "")
    subscriptions = truthy(env.get("DIRECTOR_SUBSCRIPTIONS_ENABLED"))
    require_transcription = truthy(env.get("DIRECTOR_REQUIRE_TRANSCRIPTION"))
    require_verified_email = truthy(env.get("DIRECTOR_REQUIRE_VERIFIED_EMAIL"))

    checks = [
        _check(
            env.get("DIRECTOR_ENVIRONMENT", "").casefold() == "production",
            "runtime.environment",
            "Production environment mode is enabled.",
            "DIRECTOR_ENVIRONMENT is not production.",
            "Set DIRECTOR_ENVIRONMENT=production.",
        ),
        _check(
            _https_origin(public_url),
            "network.public_url",
            "Public application URL uses HTTPS.",
            "Public application URL is not a non-local HTTPS origin.",
            "Set DIRECTOR_PUBLIC_APP_URL to the deployed https:// URL.",
        ),
        _check(
            bool(domain) and domain not in {"localhost", "127.0.0.1"},
            "network.domain",
            "A non-local production domain is configured.",
            "Production domain is missing or local.",
            "Set DIRECTOR_DOMAIN to the public hostname.",
        ),
        _check(
            bool(cors_origins)
            and all(_https_origin(origin) for origin in cors_origins)
            and "*" not in cors_origins,
            "network.cors",
            "CORS is restricted to HTTPS origins.",
            "CORS includes an unsafe, local, wildcard, or non-HTTPS origin.",
            "Set DIRECTOR_CORS_ORIGINS to explicit production HTTPS origins.",
        ),
        _check(
            truthy(env.get("DIRECTOR_AUTH_REQUIRED")),
            "auth.required",
            "Authentication is required.",
            "Authentication is disabled.",
            "Set DIRECTOR_AUTH_REQUIRED=true.",
        ),
        _check(
            len(auth_secret) >= 32 and not placeholder(auth_secret),
            "auth.secret",
            "Authentication signing secret is non-placeholder and sufficiently long.",
            "Authentication signing secret is missing, short, or a placeholder.",
            "Generate a random secret of at least 32 characters.",
        ),
        _check(
            require_verified_email,
            "auth.email_verification",
            "Email verification is required.",
            "Email verification is not required.",
            "Set DIRECTOR_REQUIRE_VERIFIED_EMAIL=true for public launch.",
        ),
        _check(
            not truthy(env.get("DIRECTOR_AUTO_CREATE_SCHEMA")),
            "database.auto_create",
            "Automatic schema creation is disabled.",
            "Automatic schema creation is enabled.",
            "Set DIRECTOR_AUTO_CREATE_SCHEMA=false and run Alembic migrations.",
        ),
        _check(
            database_url.startswith("postgresql+psycopg://")
            and "director:director@" not in database_url,
            "database.url",
            "PostgreSQL is configured without the development credential pair.",
            "Database URL is missing, uses the wrong driver, or retains development credentials.",
            "Set a production PostgreSQL URL with rotated credentials.",
        ),
        _check(
            env.get("DIRECTOR_REDIS_URL", "").startswith(("redis://", "rediss://")),
            "redis.url",
            "Redis URL is configured.",
            "Redis URL is missing or invalid.",
            "Set DIRECTOR_REDIS_URL to the production Redis endpoint.",
        ),
        _check(
            truthy(env.get("DIRECTOR_RATE_LIMIT_ENABLED"))
            and env.get("DIRECTOR_RATE_LIMIT_BACKEND", "").casefold() == "redis"
            and truthy(env.get("DIRECTOR_RATE_LIMIT_FAIL_CLOSED")),
            "security.rate_limit",
            "Redis-backed fail-closed rate limiting is enabled.",
            "Production rate limiting is not Redis-backed and fail-closed.",
            "Enable rate limiting with DIRECTOR_RATE_LIMIT_BACKEND=redis and DIRECTOR_RATE_LIMIT_FAIL_CLOSED=true.",
        ),
        _check(
            truthy(env.get("DIRECTOR_TRUST_PROXY_HEADERS")),
            "network.proxy_headers",
            "Trusted proxy headers are enabled for the HTTPS edge.",
            "Trusted proxy headers are disabled.",
            "Set DIRECTOR_TRUST_PROXY_HEADERS=true behind the supplied Caddy edge.",
        ),
        _check(
            truthy(env.get("DIRECTOR_READINESS_REQUIRE_REDIS")),
            "observability.readiness",
            "Readiness requires Redis.",
            "Readiness does not require Redis.",
            "Set DIRECTOR_READINESS_REQUIRE_REDIS=true.",
        ),
        _check(
            truthy(env.get("DIRECTOR_METRICS_ENABLED"))
            and len(metrics_token) >= 16
            and not placeholder(metrics_token),
            "observability.metrics",
            "Protected metrics are enabled with a non-placeholder token.",
            "Metrics are disabled or the metrics token is missing/weak.",
            "Enable metrics and set a random DIRECTOR_METRICS_TOKEN.",
        ),
    ]

    smtp_ready = (
        env.get("DIRECTOR_EMAIL_PROVIDER", "").casefold() == "smtp"
        and bool(env.get("DIRECTOR_SMTP_HOST"))
        and bool(env.get("DIRECTOR_SMTP_USERNAME"))
        and bool(env.get("DIRECTOR_SMTP_PASSWORD"))
        and bool(EMAIL_PATTERN.fullmatch(env.get("DIRECTOR_SMTP_FROM_EMAIL", "")))
    )
    checks.append(
        _check(
            smtp_ready,
            "external.smtp",
            "SMTP delivery configuration is present.",
            "SMTP delivery configuration is incomplete.",
            "Configure the SMTP provider and complete a deliverability rehearsal.",
            external=True,
            pending_when_false=True,
        )
    )
    checks.append(
        _check(
            bool(EMAIL_PATTERN.fullmatch(env.get("DIRECTOR_ACME_EMAIL", "")))
            and not placeholder(env.get("DIRECTOR_ACME_EMAIL")),
            "external.acme_email",
            "A real ACME contact email is configured.",
            "ACME contact email is missing or a placeholder.",
            "Set DIRECTOR_ACME_EMAIL to an operated mailbox.",
            external=True,
            pending_when_false=True,
        )
    )

    s3_ready = (
        env.get("DIRECTOR_OBJECT_STORAGE_PROVIDER", "").casefold() == "s3"
        and bool(env.get("DIRECTOR_S3_BUCKET"))
        and bool(env.get("DIRECTOR_S3_REGION"))
    )
    checks.append(
        _check(
            s3_ready,
            "external.object_storage",
            "S3-compatible object storage is selected with bucket and region.",
            "Production object storage is not fully configured.",
            "Configure S3-compatible storage and rehearse upload, download, lifecycle, and deletion.",
            external=True,
            pending_when_false=True,
        )
    )

    stripe_ready = (
        subscriptions
        and env.get("DIRECTOR_BILLING_PROVIDER", "").casefold() == "stripe"
        and not placeholder(env.get("DIRECTOR_STRIPE_SECRET_KEY"))
        and not placeholder(env.get("DIRECTOR_STRIPE_WEBHOOK_SECRET"))
        and _price_ids_configured(env.get("DIRECTOR_BILLING_PLANS_JSON", ""))
    )
    checks.append(
        _check(
            stripe_ready,
            "external.stripe",
            "Stripe subscriptions, webhook signing, and paid Price IDs are configured.",
            "Stripe launch configuration is incomplete or subscriptions are disabled.",
            "Configure Stripe test/live credentials, webhook endpoint, portal, and real recurring Price IDs.",
            external=True,
            pending_when_false=True,
        )
    )

    transcription_ready = require_transcription and not placeholder(
        env.get("DIRECTOR_OPENAI_API_KEY")
    )
    checks.append(
        _check(
            transcription_ready,
            "external.transcription",
            "Required transcription has a provider credential.",
            "Required transcription is disabled or lacks a provider credential.",
            "Set DIRECTOR_REQUIRE_TRANSCRIPTION=true and configure the transcription provider credential.",
            external=True,
            pending_when_false=True,
        )
    )

    backup_dir = env.get("DIRECTOR_BACKUP_DIR", "")
    checks.append(
        _check(
            bool(backup_dir)
            and backup_dir not in {"./backups", "backups"}
            and not backup_dir.startswith("/data"),
            "external.backup_destination",
            "Backup destination is distinct from the primary application data path.",
            "Backup destination appears local to the application host or is not configured.",
            "Point DIRECTOR_BACKUP_DIR at encrypted off-host or separately mounted storage and rehearse restore.",
            external=True,
            pending_when_false=True,
        )
    )
    return checks


def _get_json(url: str, *, token: str | None = None) -> tuple[int, object]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=20, context=context) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body)


def remote_checks(base_url: str, metrics_token: str | None) -> list[Check]:
    base = base_url.rstrip("/")
    api = f"{base}/api/v1"
    checks: list[Check] = [
        _check(
            _https_origin(base),
            "remote.tls",
            "Remote endpoint uses a public HTTPS origin.",
            "Remote endpoint is not a public HTTPS origin.",
            "Deploy behind valid TLS before release.",
            external=True,
        )
    ]
    for code, path, expected in (
        ("remote.liveness", "health/live", "alive"),
        ("remote.readiness", "health/ready", "ready"),
    ):
        try:
            status, payload = _get_json(f"{api}/{path}")
            value = payload.get("status") if isinstance(payload, dict) else None
            checks.append(
                _check(
                    status == 200 and value == expected,
                    code,
                    f"{path} returned {expected}.",
                    f"{path} did not return {expected}.",
                    "Inspect service logs and dependency readiness.",
                    external=True,
                )
            )
        except (OSError, ValueError, urllib.error.URLError) as exc:
            checks.append(
                Check(
                    code=code,
                    status="fail",
                    message=f"{path} request failed: {exc}",
                    remediation="Verify DNS, TLS, routing, and service health.",
                    external=True,
                )
            )

    if metrics_token:
        try:
            request = urllib.request.Request(
                f"{api}/metrics",
                headers={"Authorization": f"Bearer {metrics_token}"},
            )
            with urllib.request.urlopen(
                request, timeout=20, context=ssl.create_default_context()
            ) as response:
                body = response.read().decode("utf-8")
            checks.append(
                _check(
                    response.status == 200 and "director_build_info" in body,
                    "remote.metrics",
                    "Protected metrics are reachable with the configured token.",
                    "Protected metrics verification failed.",
                    "Verify metrics routing and token configuration.",
                    external=True,
                )
            )
        except (OSError, urllib.error.URLError) as exc:
            checks.append(
                Check(
                    code="remote.metrics",
                    status="fail",
                    message=f"Metrics request failed: {exc}",
                    remediation="Verify metrics routing and DIRECTOR_METRICS_TOKEN.",
                    external=True,
                )
            )
    return checks


def summary(checks: list[Check]) -> dict[str, int]:
    return {
        "pass": sum(item.status == "pass" for item in checks),
        "fail": sum(item.status == "fail" for item in checks),
        "pending": sum(item.status == "pending" for item in checks),
        "total": len(checks),
    }


def render_markdown(checks: list[Check], *, allow_external_pending: bool) -> str:
    counts = summary(checks)
    decision = "GO"
    if counts["fail"] or (counts["pending"] and not allow_external_pending):
        decision = "NO-GO"
    lines = [
        "# Director OS release doctor",
        "",
        f"**Decision:** {decision}",
        "",
        f"Passed: {counts['pass']}  ",
        f"Failed: {counts['fail']}  ",
        f"Pending external gates: {counts['pending']}",
        "",
        "| Status | Check | Result |",
        "|---|---|---|",
    ]
    symbols = {"pass": "PASS", "fail": "FAIL", "pending": "PENDING"}
    for item in checks:
        message = item.message.replace("|", "\\|")
        lines.append(f"| {symbols[item.status]} | `{item.code}` | {message} |")
        if item.remediation and item.status != "pass":
            remedy = item.remediation.replace("|", "\\|")
            lines.append(f"|  |  | Remedy: {remedy} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Director OS production go/no-go doctor.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--base-url")
    parser.add_argument("--json-output", default="dist/release-doctor.json")
    parser.add_argument("--markdown-output", default="dist/release-doctor.md")
    parser.add_argument(
        "--allow-external-pending",
        action="store_true",
        help="Do not fail solely because credential/infrastructure checks are pending.",
    )
    args = parser.parse_args()

    try:
        env = parse_env_file(Path(args.env_file))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    checks = local_checks(env)
    if args.base_url:
        checks.extend(remote_checks(args.base_url, env.get("DIRECTOR_METRICS_TOKEN")))

    counts = summary(checks)
    payload = {
        "decision": (
            "go"
            if not counts["fail"]
            and (args.allow_external_pending or not counts["pending"])
            else "no-go"
        ),
        "allow_external_pending": args.allow_external_pending,
        "summary": counts,
        "checks": [asdict(item) for item in checks],
    }
    json_path = Path(args.json_output)
    markdown_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_markdown(checks, allow_external_pending=args.allow_external_pending),
        encoding="utf-8",
    )
    print(markdown_path.read_text(encoding="utf-8"))
    return 0 if payload["decision"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
