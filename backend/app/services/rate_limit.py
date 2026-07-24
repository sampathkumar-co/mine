from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.core.config import Settings


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_after_seconds: int


class FixedWindowRateLimiter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._memory: dict[tuple[str, int], int] = {}
        self._redis = None
        if settings.rate_limit_backend.casefold() == "redis":
            try:
                from redis import Redis

                self._redis = Redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=0.25,
                    socket_timeout=0.25,
                )
            except Exception:
                self._redis = None

    def check(self, key: str, *, limit: int, window_seconds: int) -> RateLimitDecision:
        now = int(time.time())
        bucket = now // window_seconds
        reset_after = max(1, window_seconds - now % window_seconds)
        count = self._increment(key, bucket, window_seconds)
        return RateLimitDecision(
            allowed=count <= limit,
            limit=limit,
            remaining=max(0, limit - count),
            reset_after_seconds=reset_after,
        )

    def _increment(self, key: str, bucket: int, window_seconds: int) -> int:
        if self._redis is not None:
            redis_key = f"{self.settings.rate_limit_redis_prefix}:{key}:{bucket}"
            try:
                pipeline = self._redis.pipeline()
                pipeline.incr(redis_key)
                pipeline.expire(redis_key, window_seconds + 2)
                count, _ = pipeline.execute()
                return int(count)
            except Exception:
                if self.settings.rate_limit_fail_closed:
                    return 2**31
        with self._lock:
            stale_before = bucket - 2
            for item in [entry for entry in self._memory if entry[1] < stale_before]:
                self._memory.pop(item, None)
            memory_key = (key, bucket)
            count = self._memory.get(memory_key, 0) + 1
            self._memory[memory_key] = count
            return count
