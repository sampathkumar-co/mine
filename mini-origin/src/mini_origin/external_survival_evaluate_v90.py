from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import heapq
import json
from pathlib import Path
from typing import Iterable

from . import external_survival_hash_lock_v90 as lock_v90


PREREGISTRATION = lock_v90.PREREGISTRATION
LOCK_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v90-external-survival-lock-manifest.json"
)


@dataclass(frozen=True)
class SetCoverColumn:
    index: int
    cost: int
    rows: tuple[int, ...]


@dataclass(frozen=True)
class SetCoverInstance:
    name: str
    row_count: int
    columns: tuple[SetCoverColumn, ...]


@dataclass(frozen=True)
class Arc:
    index: int
    tail: int
    head: int
    cost: int


@dataclass(frozen=True)
class ShortestPathInstance:
    name: str
    node_count: int
    arcs: tuple[Arc, ...]


def parse_rail_set_cover(name: str, payload: bytes) -> SetCoverInstance:
    try:
        tokens = [int(token) for token in payload.decode("ascii").split()]
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid OR-Library integer stream for {name}") from exc
    if len(tokens) < 2:
        raise RuntimeError(f"truncated OR-Library instance: {name}")
    row_count, column_count = tokens[0], tokens[1]
    if row_count <= 0 or column_count <= 0:
        raise RuntimeError(f"invalid OR-Library dimensions: {name}")
    position = 2
    columns = []
    for index in range(1, column_count + 1):
        if position + 2 > len(tokens):
            raise RuntimeError(f"truncated OR-Library column header: {name}:{index}")
        cost = tokens[position]
        covered_count = tokens[position + 1]
        position += 2
        if cost < 0 or covered_count <= 0:
            raise RuntimeError(f"invalid OR-Library column: {name}:{index}")
        end = position + covered_count
        if end > len(tokens):
            raise RuntimeError(f"truncated OR-Library column rows: {name}:{index}")
        rows = tuple(sorted(tokens[position:end]))
        position = end
        if len(rows) != len(set(rows)):
            raise RuntimeError(f"duplicate row inside OR-Library column: {name}:{index}")
        if any(row < 1 or row > row_count for row in rows):
            raise RuntimeError(f"row id outside OR-Library dimensions: {name}:{index}")
        columns.append(SetCoverColumn(index=index, cost=cost, rows=rows))
    if position != len(tokens):
        raise RuntimeError(
            f"unexpected trailing OR-Library tokens for {name}: {len(tokens) - position}"
        )
    return SetCoverInstance(name=name, row_count=row_count, columns=tuple(columns))


def quotient_set_cover(instance: SetCoverInstance) -> tuple[tuple[SetCoverColumn, ...], dict[str, int]]:
    groups: dict[tuple[int, ...], list[SetCoverColumn]] = {}
    for column in instance.columns:
        groups.setdefault(column.rows, []).append(column)

    kept = []
    removed = 0
    equal_ties = 0
    strict = 0
    certificate_failures = 0
    duplicate_classes = 0
    for signature in sorted(groups):
        group = groups[signature]
        if len(group) > 1:
            duplicate_classes += 1
        representative = min(group, key=lambda row: (row.cost, row.index))
        kept.append(representative)
        for column in group:
            if column.index == representative.index:
                continue
            removed += 1
            if representative.rows != column.rows or representative.cost > column.cost:
                certificate_failures += 1
            if representative.cost < column.cost:
                strict += 1
            else:
                equal_ties += 1
                if representative.index > column.index:
                    certificate_failures += 1
    kept.sort(key=lambda row: row.index)
    return tuple(kept), {
        "raw_columns": len(instance.columns),
        "distinct_coverage_signatures": len(groups),
        "duplicate_coverage_classes": duplicate_classes,
        "columns_removed_by_frozen_quotient": removed,
        "equal_cost_tie_removals": equal_ties,
        "strict_cost_dominance_removals": strict,
        "local_certificate_failures": certificate_failures,
    }


def parse_dimacs_graph(name: str, compressed_payload: bytes) -> ShortestPathInstance:
    try:
        payload = gzip.decompress(compressed_payload)
    except (OSError, EOFError) as exc:
        raise RuntimeError(f"invalid DIMACS gzip stream: {name}") from exc
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"non-ASCII DIMACS graph: {name}") from exc

    node_count: int | None = None
    declared_arcs: int | None = None
    arcs = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        parts = line.split()
        if parts[0] == "p":
            if len(parts) != 4 or parts[1] != "sp" or node_count is not None:
                raise RuntimeError(f"invalid DIMACS problem line: {name}:{line_number}")
            node_count = int(parts[2])
            declared_arcs = int(parts[3])
            if node_count <= 0 or declared_arcs < 0:
                raise RuntimeError(f"invalid DIMACS dimensions: {name}")
            continue
        if parts[0] == "a":
            if node_count is None or len(parts) != 4:
                raise RuntimeError(f"invalid DIMACS arc line: {name}:{line_number}")
            tail, head, cost = map(int, parts[1:])
            if tail < 1 or tail > node_count or head < 1 or head > node_count:
                raise RuntimeError(f"DIMACS endpoint outside dimensions: {name}:{line_number}")
            if cost < 0:
                raise RuntimeError(f"negative DIMACS arc length outside v0.90: {name}")
            arcs.append(Arc(index=len(arcs), tail=tail, head=head, cost=cost))
            continue
        raise RuntimeError(f"unknown DIMACS record: {name}:{line_number}")

    if node_count is None or declared_arcs is None:
        raise RuntimeError(f"missing DIMACS problem line: {name}")
    if len(arcs) != declared_arcs:
        raise RuntimeError(
            f"DIMACS arc-count mismatch for {name}: parsed {len(arcs)} declared {declared_arcs}"
        )
    return ShortestPathInstance(name=name, node_count=node_count, arcs=tuple(arcs))


def quotient_shortest_path(
    instance: ShortestPathInstance,
) -> tuple[ShortestPathInstance, dict[str, int]]:
    groups: dict[tuple[int, int], list[Arc]] = {}
    for arc in instance.arcs:
        groups.setdefault((arc.tail, arc.head), []).append(arc)

    kept = []
    removed = 0
    equal_ties = 0
    strict = 0
    certificate_failures = 0
    parallel_classes = 0
    for endpoints in sorted(groups):
        group = groups[endpoints]
        if len(group) > 1:
            parallel_classes += 1
        representative = min(group, key=lambda row: (row.cost, row.index))
        kept.append(representative)
        for arc in group:
            if arc.index == representative.index:
                continue
            removed += 1
            if (
                arc.tail != representative.tail
                or arc.head != representative.head
                or representative.cost > arc.cost
            ):
                certificate_failures += 1
            if representative.cost < arc.cost:
                strict += 1
            else:
                equal_ties += 1
                if representative.index > arc.index:
                    certificate_failures += 1
    kept.sort(key=lambda row: row.index)
    quotient = ShortestPathInstance(
        name=instance.name,
        node_count=instance.node_count,
        arcs=tuple(kept),
    )
    return quotient, {
        "nodes": instance.node_count,
        "raw_arcs": len(instance.arcs),
        "quotient_arcs": len(kept),
        "arcs_removed_by_frozen_quotient": removed,
        "parallel_endpoint_classes": parallel_classes,
        "strict_cost_dominance_removals": strict,
        "equal_cost_tie_removals": equal_ties,
        "local_certificate_failures": certificate_failures,
    }


def dijkstra(instance: ShortestPathInstance, source: int) -> list[int | None]:
    if source < 1 or source > instance.node_count:
        raise ValueError("source outside graph")
    adjacency: list[list[tuple[int, int]]] = [
        [] for _ in range(instance.node_count + 1)
    ]
    for arc in instance.arcs:
        adjacency[arc.tail].append((arc.head, arc.cost))
    distances: list[int | None] = [None] * (instance.node_count + 1)
    distances[source] = 0
    queue: list[tuple[int, int]] = [(0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distances[node] != distance:
            continue
        for head, cost in adjacency[node]:
            candidate = distance + cost
            current = distances[head]
            if current is None or candidate < current:
                distances[head] = candidate
                heapq.heappush(queue, (candidate, head))
    return distances


def distance_digest(distances: Iterable[int | None]) -> tuple[str, int]:
    digest = hashlib.sha256()
    reachable = 0
    for node, distance in enumerate(distances):
        if node == 0:
            continue
        if distance is None:
            digest.update(f"{node}:INF\n".encode())
        else:
            reachable += 1
            digest.update(f"{node}:{distance}\n".encode())
    return digest.hexdigest(), reachable


def shortest_path_distance_certificate(
    raw: ShortestPathInstance,
    quotient: ShortestPathInstance,
) -> dict[str, object]:
    sources = (1, max(1, raw.node_count // 2), raw.node_count)
    rows = []
    for source in sources:
        raw_digest, raw_reachable = distance_digest(dijkstra(raw, source))
        quotient_digest, quotient_reachable = distance_digest(dijkstra(quotient, source))
        rows.append({
            "source": source,
            "raw_distance_digest": raw_digest,
            "quotient_distance_digest": quotient_digest,
            "raw_reachable": raw_reachable,
            "quotient_reachable": quotient_reachable,
            "matched": (
                raw_digest == quotient_digest
                and raw_reachable == quotient_reachable
            ),
        })
    return {
        "sources": list(sources),
        "rows": rows,
        "passed": all(bool(row["matched"]) for row in rows),
    }


def _load_lock_manifest() -> dict[str, object]:
    if not LOCK_MANIFEST.exists():
        raise RuntimeError(
            "v0.90 evaluation is locked until campaigns/"
            "v90-external-survival-lock-manifest.json is committed"
        )
    payload = json.loads(LOCK_MANIFEST.read_text(encoding="utf-8"))
    if payload["status"] != "v90_raw_bytes_hash_locked_not_parsed":
        raise RuntimeError("v0.90 lock-manifest status changed")
    if payload["raw_stream_count"] != 6:
        raise RuntimeError("v0.90 lock manifest must contain six streams")
    return payload


def _verified_downloads(
    preregistration: dict[str, object],
    lock_manifest: dict[str, object],
) -> dict[str, bytes]:
    source_rows = lock_v90.selected_sources(preregistration)
    locked_rows = {
        (str(row["family"]), str(row["name"])): row
        for row in lock_manifest["records"]
    }
    if len(locked_rows) != 6:
        raise RuntimeError("v0.90 lock manifest contains duplicate/missing names")
    payloads: dict[str, bytes] = {}
    for source in source_rows:
        key = (source["family"], source["name"])
        locked = locked_rows.get(key)
        if locked is None:
            raise RuntimeError(f"v0.90 source missing from lock manifest: {key}")
        payload, final_url, _headers = lock_v90.download_raw(source["url"])
        if final_url != locked["final_url"]:
            raise RuntimeError(f"v0.90 final URL changed for {key}")
        if len(payload) != int(locked["bytes"]):
            raise RuntimeError(f"v0.90 byte count changed for {key}")
        if hashlib.sha256(payload).hexdigest() != locked["sha256"]:
            raise RuntimeError(f"v0.90 SHA-256 changed for {key}")
        payloads[source["name"]] = payload
    return payloads


def evaluate() -> dict[str, object]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if preregistration["status"] != "preregistered_before_raw_instance_access":
        raise RuntimeError("v0.90 preregistration status changed")
    lock_manifest = _load_lock_manifest()
    payloads = _verified_downloads(preregistration, lock_manifest)

    set_cover_rows = []
    for name in ("rail507", "rail516", "rail582"):
        instance = parse_rail_set_cover(name, payloads[name])
        _kept, stats = quotient_set_cover(instance)
        set_cover_rows.append({"name": name, **stats})

    shortest_path_rows = []
    for name in ("NY-distance", "BAY-distance", "COL-distance"):
        raw = parse_dimacs_graph(name, payloads[name])
        quotient, stats = quotient_shortest_path(raw)
        certificate = shortest_path_distance_certificate(raw, quotient)
        shortest_path_rows.append({
            "name": name,
            **stats,
            "distance_certificate": certificate,
        })

    set_cover_removed = sum(
        int(row["columns_removed_by_frozen_quotient"])
        for row in set_cover_rows
    )
    shortest_path_removed = sum(
        int(row["arcs_removed_by_frozen_quotient"])
        for row in shortest_path_rows
    )
    certificate_failures = (
        sum(int(row["local_certificate_failures"]) for row in set_cover_rows)
        + sum(int(row["local_certificate_failures"]) for row in shortest_path_rows)
    )
    distance_pass = all(
        bool(row["distance_certificate"]["passed"])
        for row in shortest_path_rows
    )
    passed = (
        certificate_failures == 0
        and distance_pass
        and set_cover_removed > 0
        and shortest_path_removed > 0
    )
    return {
        "status": (
            "external_survival_pass_v90"
            if passed
            else "external_survival_rejected_v90"
        ),
        "version": "v0.90",
        "passed": passed,
        "byte_lock_records_digest": lock_manifest["records_digest"],
        "all_six_raw_byte_streams_reverified": True,
        "set_cover": {
            "instances": set_cover_rows,
            "aggregate_removed_columns": set_cover_removed,
        },
        "shortest_path": {
            "instances": shortest_path_rows,
            "aggregate_removed_arcs": shortest_path_removed,
            "all_distance_certificates_match": distance_pass,
        },
        "local_certificate_failures": certificate_failures,
        "claim_boundary": preregistration["claim_boundary"],
        "kill_rule": preregistration["kill_rule"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "passed": result["passed"],
        "set_cover_aggregate_removed_columns": result["set_cover"]["aggregate_removed_columns"],
        "shortest_path_aggregate_removed_arcs": result["shortest_path"]["aggregate_removed_arcs"],
        "all_distance_certificates_match": result["shortest_path"]["all_distance_certificates_match"],
        "local_certificate_failures": result["local_certificate_failures"],
    }, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
