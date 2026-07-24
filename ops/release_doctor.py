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
    if not path.exists():
        raise FileNotFoundError(f"Environment file not found: {path}")
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if key.startswith("DIRECTOR_")})
    return values


def truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def placeholder(value: str | None) -> bool:
    normalized = str(value or "").strip()
    folded = normalized.casefold()
    return (
        folded in PLACEHOLDERS
        or "replace" in folded
        or folded.endswith("example.com")
        or folded.endswith("example.invalid")
    )


def _result(
    condition: bool,
    *,
    code: str,
    passed: str,
    failed: str,
    remediation: str,
    external: bool = False,
    pending: bool = False,
) -> Check:
    if condition:
        return Check(code, "pass", passed, external=external)
    return Check(
        code,
        "pending" if pending else "fail",
        failed,
        remediation=remediation,
        external=external,
    )


def _https_origin(value: str) -> bool:
    parsed = urlparse(value)
    return (
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and parsed.hostname not in {"localhost", "127.0.0.1"}
    )


def _all_present(env: dict[str, str], keys: tuple[str, ...]) -> bool:
    return all(bool(env.get(key)) and not placeholder(env.get(key)) for key in keys)


def _paid_price_ids_ready(raw: str) -> bool:
    try:
        plans = json.loads(raw)
    except (TypeError, ValueError):
        return False
    return all(
        isinstance(plans.get(key), dict)
        and str(plans[key].get("price_id") or "").startswith("price_")
        and not placeholder(str(plans[key].get("price_id") or ""))
        for key in ("creator", "studio")
    )


def local_checks(env: dict[str, str]) -> list[Check]:
    cors = [item.strip() for item in env.get("DIRECTOR_CORS_ORIGINS", "").split(",") if item.strip()]
    database_url = env.get("DIRECTOR_DATABASE_URL", "")
    core: list[tuple[str, bool, str, str, str]] = [
        (
            "runtime.environment",
            env.get("DIRECTOR_ENVIRONMENT", "").casefold() == "production",
            "Production environment mode is enabled.",
            "DIRECTOR_ENVIRONMENT is not production.",
            "Set DIRECTOR_ENVIRONMENT=production.",
        ),
        (
            "network.public_url",
            _https_origin(env.get("DIRECTOR_PUBLIC_APP_URL", "")),
            "Public application URL uses HTTPS.",
            "Public application URL is not a non-local HTTPS origin.",
            "Set DIRECTOR_PUBLIC_APP_URL to the deployed HTTPS URL.",
        ),
        (
            "network.domain",
            bool(env.get("DIRECTOR_DOMAIN"))
            and env.get("DIRECTOR_DOMAIN") not in {"localhost", "127.0.0.1"},
            "A non-local production domain is configured.",
            "Production domain is missing or local.",
            "Set DIRECTOR_DOMAIN to the public hostname.",
        ),
        (
            "network.cors",
            bool(cors) and "*" not in cors and all(_https_origin(origin) for origin in cors),
            "CORS is restricted to explicit HTTPS origins.",
            "CORS includes an unsafe, wildcard, local, or non-HTTPS origin.",
            "Set DIRECTOR_CORS_ORIGINS to explicit production HTTPS origins.",
        ),
        (
            "network.proxy_headers",
            truthy(env.get("DIRECTOR_TRUST_PROXY_HEADERS")),
            "Trusted proxy headers are enabled.",
            "Trusted proxy headers are disabled.",
            "Set DIRECTOR_TRUST_PROXY_HEADERS=true behind the supplied HTTPS edge.",
        ),
        (
            "auth.required",
            truthy(env.get("DIRECTOR_AUTH_REQUIRED")),
            "Authentication is required.",
            "Authentication is disabled.",
            "Set DIRECTOR_AUTH_REQUIRED=true.",
        ),
        (
            "auth.secret",
            len(env.get("DIRECTOR_AUTH_SECRET", "")) >= 32
            and not placeholder(env.get("DIRECTOR_AUTH_SECRET")),
            "Authentication signing secret is non-placeholder and sufficiently long.",
            "Authentication signing secret is missing, short, or a placeholder.",
            "Generate a random authentication secret of at least 32 characters.",
        ),
        (
            "auth.email_verification",
            truthy(env.get("DIRECTOR_REQUIRE_VERIFIED_EMAIL")),
            "Email verification is required.",
            "Email verification is not required.",
            "Set DIRECTOR_REQUIRE_VERIFIED_EMAIL=true.",
        ),
        (
            "database.auto_create",
            not truthy(env.get("DIRECTOR_AUTO_CREATE_SCHEMA")),
            "Automatic schema creation is disabled.",
            "Automatic schema creation is enabled.",
            "Set DIRECTOR_AUTO_CREATE_SCHEMA=false and use Alembic.",
        ),
        (
            "database.url",
            database_url.startswith("postgresql+psycopg://")
            and "director:director@" not in database_url
            and not placeholder(database_url),
            "PostgreSQL uses a production credential URL.",
            "Database URL is missing, unsafe, or retains placeholder/development credentials.",
            "Set a production PostgreSQL URL with rotated credentials.",
        ),
        (
            "redis.url",
            env.get("DIRECTOR_REDIS_URL", "").startswith(("redis://", "rediss://")),
            "Redis URL is configured.",
            "Redis URL is missing or invalid.",
            "Set DIRECTOR_REDIS_URL to the production Redis endpoint.",
        ),
        (
            "security.rate_limit",
            truthy(env.get("DIRECTOR_RATE_LIMIT_ENABLED"))
            and env.get("DIRECTOR_RATE_LIMIT_BACKEND", "").casefold() == "redis"
            and truthy(env.get("DIRECTOR_RATE_LIMIT_FAIL_CLOSED")),
            "Redis-backed fail-closed rate limiting is enabled.",
            "Production rate limiting is not Redis-backed and fail-closed.",
            "Enable Redis rate limiting and DIRECTOR_RATE_LIMIT_FAIL_CLOSED=true.",
        ),
        (
            "observability.readiness",
            truthy(env.get("DIRECTOR_READINESS_REQUIRE_REDIS")),
            "Readiness requires Redis.",
            "Readiness does not require Redis.",
            "Set DIRECTOR_READINESS_REQUIRE_REDIS=true.",
        ),
        (
            "observability.metrics",
            truthy(env.get("DIRECTOR_METRICS_ENABLED"))
            and len(env.get("DIRECTOR_METRICS_TOKEN", "")) >= 16
            and not placeholder(env.get("DIRECTOR_METRICS_TOKEN")),
            "Protected metrics are enabled with a strong token.",
            "Metrics are disabled or the metrics token is missing, weak, or a placeholder.",
            "Enable metrics and set a random DIRECTOR_METRICS_TOKEN.",
        ),
    ]
    checks = [
        _result(
            condition,
            code=code,
            passed=passed,
            failed=failed,
            remediation=remediation,
        )
        for code, condition, passed, failed, remediation in core
    ]

    smtp_ready = (
        env.get("DIRECTOR_EMAIL_PROVIDER", "").casefold() == "smtp"
        and _all_present(
            env,
            (
                "DIRECTOR_SMTP_HOST",
                "DIRECTOR_SMTP_USERNAME",
                "DIRECTOR_SMTP_PASSWORD",
                "DIRECTOR_SMTP_FROM_EMAIL",
            ),
        )
        and bool(EMAIL_PATTERN.fullmatch(env.get("DIRECTOR_SMTP_FROM_EMAIL", "")))
    )
    external: list[tuple[str, bool, str, str, str]] = [
        (
            "external.smtp",
            smtp_ready,
            "SMTP delivery configuration is present.",
            "SMTP delivery configuration is incomplete or contains placeholders.",
            "Configure SMTP and complete a deliverability rehearsal.",
        ),
        (
            "external.acme_email",
            bool(EMAIL_PATTERN.fullmatch(env.get("DIRECTOR_ACME_EMAIL", "")))
            and not placeholder(env.get("DIRECTOR_ACME_EMAIL")),
            "A real ACME contact email is configured.",
            "ACME contact email is missing or a placeholder.",
            "Set DIRECTOR_ACME_EMAIL to an operated mailbox.",
        ),
        (
            "external.object_storage",
            env.get("DIRECTOR_OBJECT_STORAGE_PROVIDER", "").casefold() == "s3"
            and _all_present(env, ("DIRECTOR_S3_BUCKET", "DIRECTOR_S3_REGION")),
            "S3-compatible object storage is selected with bucket and region.",
            "Production object storage is incomplete or contains placeholders.",
            "Configure and rehearse S3 upload, download, lifecycle, and deletion.",
        ),
        (
            "external.stripe",
            truthy(env.get("DIRECTOR_SUBSCRIPTIONS_ENABLED"))
            and env.get("DIRECTOR_BILLING_PROVIDER", "").casefold() == "stripe"
            and _all_present(env, ("DIRECTOR_STRIPE_SECRET_KEY", "DIRECTOR_STRIPE_WEBHOOK_SECRET"))
            and _paid_price_ids_ready(env.get("DIRECTOR_BILLING_PLANS_JSON", "")),
            "Stripe subscriptions, webhook signing, and paid Price IDs are configured.",
            "Stripe launch configuration is incomplete or contains placeholders.",
            "Configure Stripe keys, portal, webhook endpoint, and recurring Price IDs.",
        ),
        (
            "external.transcription",
            truthy(env.get("DIRECTOR_REQUIRE_TRANSCRIPTION"))
            and not placeholder(env.get("DIRECTOR_OPENAI_API_KEY")),
            "Required transcription has a provider credential.",
            "Required transcription is disabled or lacks a provider credential.",
            "Require transcription and configure the provider credential.",
        ),
        (
            "external.backup_destination",
            bool(env.get("DIRECTOR_BACKUP_DIR"))
            and Path(env["DIRECTOR_BACKUP_DIR"]).is_absolute()
            and not env["DIRECTOR_BACKUP_DIR"].startswith("/data")
            and not placeholder(env["DIRECTOR_BACKUP_DIR"]),
            "Backup destination is distinct from primary application data.",
            "Backup destination is local, relative, missing, or a placeholder.",
            "Use encrypted off-host or separately mounted backup storage and rehearse restore.",
        ),
    ]
    checks.extend(
        _result(
            condition,
            code=code,
            passed=passed,
            failed=failed,
            remediation=remediation,
            external=True,
            pending=True,
        )
        for code, condition, passed, failed, remediation in external
    )
    return checks


def _json_get(url: str) -> tuple[int, object]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(
        request,
        timeout=20,
        context=ssl.create_default_context(),
    ) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def remote_checks(base_url: str, metrics_token: str | None) -> list[Check]:
    base = base_url.rstrip("/")
    api = f"{base}/api/v1"
    checks = [
        _result(
            _https_origin(base),
            code="remote.tls",
            passed="Remote endpoint uses a public HTTPS origin.",
            failed="Remote endpoint is not a public HTTPS origin.",
            remediation="Deploy behind valid TLS before release.",
            external=True,
        )
    ]
    for code, path, expected in (
        ("remote.liveness", "health/live", "alive"),
        ("remote.readiness", "health/ready", "ready"),
    ):
        try:
            status, payload = _json_get(f"{api}/{path}")
            value = payload.get("status") if isinstance(payload, dict) else None
            checks.append(
                _result(
                    status == 200 and value == expected,
                    code=code,
                    passed=f"{path} returned {expected}.",
                    failed=f"{path} did not return {expected}.",
                    remediation="Inspect service logs and dependency readiness.",
                    external=True,
                )
            )
        except (OSError, ValueError, urllib.error.URLError) as exc:
            checks.append(
                Check(
                    code,
                    "fail",
                    f"{path} request failed: {exc}",
                    "Verify DNS, TLS, routing, and service health.",
                    True,
                )
            )

    if metrics_token:
        try:
            request = urllib.request.Request(
                f"{api}/metrics",
                headers={"Authorization": f"Bearer {metrics_token}"},
            )
            with urllib.request.urlopen(
                request,
                timeout=20,
                context=ssl.create_default_context(),
            ) as response:
                body = response.read().decode("utf-8")
            checks.append(
                _result(
                    response.status == 200 and "director_build_info" in body,
                    code="remote.metrics",
                    passed="Protected metrics are reachable.",
                    failed="Protected metrics verification failed.",
                    remediation="Verify metrics routing and token configuration.",
                    external=True,
                )
            )
        except (OSError, urllib.error.URLError) as exc:
            checks.append(
                Check(
                    "remote.metrics",
                    "fail",
                    f"Metrics request failed: {exc}",
                    "Verify metrics routing and DIRECTOR_METRICS_TOKEN.",
                    True,
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


def decision(checks: list[Check], *, allow_external_pending: bool) -> str:
    counts = summary(checks)
    return (
        "go"
        if not counts["fail"] and (allow_external_pending or not counts["pending"])
        else "no-go"
    )


def render_markdown(checks: list[Check], *, allow_external_pending: bool) -> str:
    counts = summary(checks)
    lines = [
        "# Director OS release doctor",
        "",
        f"**Decision:** {decision(checks, allow_external_pending=allow_external_pending).upper()}",
        "",
        f"Passed: {counts['pass']}  ",
        f"Failed: {counts['fail']}  ",
        f"Pending external gates: {counts['pending']}",
        "",
        "| Status | Check | Result |",
        "|---|---|---|",
    ]
    for item in checks:
        message = item.message.replace("|", "\\|")
        lines.append(f"| {item.status.upper()} | `{item.code}` | {message} |")
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
    parser.add_argument("--allow-external-pending", action="store_true")
    args = parser.parse_args()

    try:
        env = parse_env_file(Path(args.env_file))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    checks = local_checks(env)
    if args.base_url:
        checks.extend(remote_checks(args.base_url, env.get("DIRECTOR_METRICS_TOKEN")))

    payload = {
        "decision": decision(checks, allow_external_pending=args.allow_external_pending),
        "allow_external_pending": args.allow_external_pending,
        "summary": summary(checks),
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
