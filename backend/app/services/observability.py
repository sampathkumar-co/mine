from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.core.config import Settings
from app.core.database import engine

VERSION = "0.15.0"
_UUID_SEGMENT = re.compile(r"/[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,36}(?=/|$)")
_NUMBER_SEGMENT = re.compile(r"/\d+(?=/|$)")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "user_id",
            "workspace_id",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def configure_logging(settings: Settings) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter()
        if settings.log_format == "json"
        else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    root.handlers.clear()
    root.addHandler(handler)


def normalize_path(path: str) -> str:
    value = _UUID_SEGMENT.sub("/{id}", path)
    return _NUMBER_SEGMENT.sub("/{number}", value)


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, str], int] = defaultdict(int)
        self._duration_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._duration_count: dict[tuple[str, str], int] = defaultdict(int)
        self._exceptions = 0
        self._active = 0

    def start(self) -> None:
        with self._lock:
            self._active += 1

    def finish(self, method: str, path: str, status_code: int, duration: float) -> None:
        normalized = normalize_path(path)
        status_class = f"{status_code // 100}xx"
        with self._lock:
            self._active = max(0, self._active - 1)
            self._requests[(method, normalized, status_class)] += 1
            self._duration_sum[(method, normalized)] += duration
            self._duration_count[(method, normalized)] += 1

    def exception(self) -> None:
        with self._lock:
            self._exceptions += 1

    def render(self) -> str:
        with self._lock:
            lines = [
                "# HELP director_build_info Director OS build information.",
                "# TYPE director_build_info gauge",
                f'director_build_info{{version="{VERSION}"}} 1',
                "# HELP director_http_active_requests Requests currently executing.",
                "# TYPE director_http_active_requests gauge",
                f"director_http_active_requests {self._active}",
                "# HELP director_http_exceptions_total Unhandled request exceptions.",
                "# TYPE director_http_exceptions_total counter",
                f"director_http_exceptions_total {self._exceptions}",
                "# HELP director_http_requests_total Completed HTTP requests.",
                "# TYPE director_http_requests_total counter",
            ]
            for (method, path, status_class), count in sorted(self._requests.items()):
                lines.append(
                    f'director_http_requests_total{{method="{method}",path="{path}",status_class="{status_class}"}} {count}'
                )
            lines.extend(
                [
                    "# HELP director_http_request_duration_seconds Request duration in seconds.",
                    "# TYPE director_http_request_duration_seconds summary",
                ]
            )
            for (method, path), total in sorted(self._duration_sum.items()):
                count = self._duration_count[(method, path)]
                labels = f'method="{method}",path="{path}"'
                lines.append(f"director_http_request_duration_seconds_sum{{{labels}}} {total:.6f}")
                lines.append(f"director_http_request_duration_seconds_count{{{labels}}} {count}")
            return "\n".join(lines) + "\n"


metrics = MetricsRegistry()


def _directory_status(path_value: str) -> tuple[str, str | None]:
    path = Path(path_value)
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".director-health-", delete=True):
            pass
        return "ok", None
    except OSError as exc:
        return "failed", str(exc)


def readiness_snapshot(settings: Settings) -> tuple[str, dict[str, dict[str, str | None]]]:
    components: dict[str, dict[str, str | None]] = {}
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        components["database"] = {"status": "ok", "detail": None}
    except Exception as exc:
        components["database"] = {"status": "failed", "detail": str(exc)[:300]}

    redis_status = "ok"
    redis_detail = None
    try:
        from redis import Redis

        client = Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=0.35,
            socket_timeout=0.35,
        )
        client.ping()
    except Exception as exc:
        redis_status = "failed" if settings.readiness_require_redis else "degraded"
        redis_detail = str(exc)[:300]
    components["redis"] = {"status": redis_status, "detail": redis_detail}

    for name, value in (
        ("uploads", settings.upload_dir),
        ("outputs", settings.output_dir),
        ("object_storage", settings.object_storage_local_dir),
    ):
        if name == "object_storage" and settings.object_storage_provider.casefold() != "local":
            components[name] = {
                "status": "ok",
                "detail": f"provider={settings.object_storage_provider}",
            }
            continue
        component_status, detail = _directory_status(value)
        components[name] = {"status": component_status, "detail": detail}

    statuses = {item["status"] for item in components.values()}
    if "failed" in statuses:
        overall = "not_ready"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "ready"
    return overall, components


def runtime_identity() -> dict[str, str]:
    return {
        "version": VERSION,
        "pid": str(os.getpid()),
        "started_at": datetime.now(UTC).isoformat(),
    }
