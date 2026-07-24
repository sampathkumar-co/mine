#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Sample:
    status: int
    elapsed_ms: float
    error: str | None = None


def request_once(url: str, timeout: float) -> Sample:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            response.read(256)
            status = response.status
        return Sample(status=status, elapsed_ms=(time.perf_counter() - started) * 1_000)
    except urllib.error.HTTPError as exc:
        return Sample(
            status=exc.code,
            elapsed_ms=(time.perf_counter() - started) * 1_000,
            error=str(exc),
        )
    except Exception as exc:
        return Sample(
            status=0,
            elapsed_ms=(time.perf_counter() - started) * 1_000,
            error=str(exc),
        )


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded HTTP concurrency probe for release qualification.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--p95-ms", type=float, default=750.0)
    parser.add_argument("--minimum-success-rate", type=float, default=1.0)
    args = parser.parse_args()

    total = max(1, args.requests)
    workers = max(1, min(args.concurrency, total, 128))
    started = time.perf_counter()
    samples: list[Sample] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(request_once, args.url, args.timeout) for _ in range(total)]
        for future in as_completed(futures):
            samples.append(future.result())

    elapsed = max(0.001, time.perf_counter() - started)
    successes = [sample for sample in samples if 200 <= sample.status < 300]
    latencies = [sample.elapsed_ms for sample in successes]
    success_rate = len(successes) / len(samples)
    report = {
        "url": args.url,
        "requests": len(samples),
        "concurrency": workers,
        "successes": len(successes),
        "success_rate": round(success_rate, 5),
        "requests_per_second": round(len(samples) / elapsed, 2),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else None,
            "p50": round(percentile(latencies, 0.50), 2) if latencies else None,
            "p95": round(percentile(latencies, 0.95), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "failures": [asdict(sample) for sample in samples if sample.error][:10],
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if success_rate < args.minimum_success_rate:
        return 1
    if percentile(latencies, 0.95) > args.p95_ms:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
