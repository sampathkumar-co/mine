from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import io
import itertools
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

from . import proof_carrying_reduction_synthesis_v91 as v91
from . import tsplib_hash_lock_v92 as lock_v92


PREREGISTRATION = lock_v92.PREREGISTRATION
LOCK_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v92-tsplib-archive-lock-manifest.json"
)
EXPECTED_SPEC = v91.RelationSpec(
    "true",
    "forall_right_exists_left",
    "le",
    "recursive_left_le_right",
)
EXPECTED_SPEC_DIGEST = "7f0f6d05c27c2021055e9b110a0e789ecb0a9e9326ab64f15cb61cd726601375"
EXPECTED_V91_FREEZE_DIGEST = "450c2911ed499890178b88db5451d2ae8ef16a637072758704b671b3151dee75"
PROJECTED_CITIES = 9


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_integral_cost(text: str) -> int:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise RuntimeError(f"invalid TSPLIB XML edge cost: {text!r}") from exc
    integral = value.to_integral_value()
    if value != integral or integral < 0:
        raise RuntimeError(f"v0.92 requires nonnegative integral edge costs: {text!r}")
    return int(integral)


def parse_projected_matrix(name: str, archive: bytes) -> tuple[tuple[int, ...], ...]:
    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as zf:
            files = [row for row in zf.namelist() if not row.endswith("/")]
            xml_files = [row for row in files if row.lower().endswith(".xml")]
            if len(xml_files) != 1:
                raise RuntimeError(
                    f"v0.92 requires exactly one XML file in {name}, found {len(xml_files)}"
                )
            xml_bytes = zf.read(xml_files[0])
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise RuntimeError(f"invalid TSPLIB XML archive: {name}") from exc

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise RuntimeError(f"invalid TSPLIB XML document: {name}") from exc
    graph = next((node for node in root.iter() if _local_name(node.tag) == "graph"), None)
    if graph is None:
        raise RuntimeError(f"TSPLIB XML graph missing: {name}")
    vertices = [node for node in graph if _local_name(node.tag) == "vertex"]
    if len(vertices) < PROJECTED_CITIES:
        raise RuntimeError(
            f"TSPLIB instance {name} has only {len(vertices)} vertices; need {PROJECTED_CITIES}"
        )

    matrix = [[0 if i == j else None for j in range(PROJECTED_CITIES)] for i in range(PROJECTED_CITIES)]
    for tail in range(PROJECTED_CITIES):
        for edge in vertices[tail]:
            if _local_name(edge.tag) != "edge":
                continue
            text = (edge.text or "").strip()
            try:
                head = int(text)
            except ValueError as exc:
                raise RuntimeError(f"invalid TSPLIB XML edge endpoint in {name}: {text!r}") from exc
            if head < 0:
                raise RuntimeError(f"negative TSPLIB XML edge endpoint: {name}:{head}")
            if head >= PROJECTED_CITIES:
                continue
            cost_text = edge.attrib.get("cost")
            if cost_text is None:
                raise RuntimeError(f"missing TSPLIB XML edge cost: {name}:{tail}->{head}")
            cost = parse_integral_cost(cost_text)
            if matrix[tail][head] is not None and tail != head:
                raise RuntimeError(f"duplicate TSPLIB XML edge: {name}:{tail}->{head}")
            matrix[tail][head] = cost

    for tail in range(PROJECTED_CITIES):
        for head in range(PROJECTED_CITIES):
            if tail == head:
                continue
            if matrix[tail][head] is None:
                raise RuntimeError(f"projected TSPLIB edge missing: {name}:{tail}->{head}")
    return tuple(tuple(int(value) for value in row) for row in matrix)  # type: ignore[arg-type]


def matrix_digest(matrix: tuple[tuple[int, ...], ...]) -> str:
    return hashlib.sha256(
        json.dumps(matrix, separators=(",", ":")).encode()
    ).hexdigest()


def held_karp_problem(name: str, matrix: tuple[tuple[int, ...], ...]) -> v91.Problem:
    n = len(matrix)
    if n != PROJECTED_CITIES or any(len(row) != n for row in matrix):
        raise RuntimeError("v0.92 Held-Karp matrix shape changed")
    full = (1 << n) - 1
    terminal = ("terminal",)
    actions_by_state: dict[object, tuple[v91.Action, ...]] = {}
    action_id = 0

    initial = (1, 0)
    for mask in range(1, full + 1):
        if not (mask & 1):
            continue
        lasts = [0] if mask == 1 else [
            last for last in range(1, n) if mask & (1 << last)
        ]
        for last in lasts:
            state = (mask, last)
            rows = []
            if mask == full:
                rows.append(v91.Action(action_id, matrix[last][0], terminal))
                action_id += 1
            else:
                for nxt in range(1, n):
                    if mask & (1 << nxt):
                        continue
                    rows.append(
                        v91.Action(
                            action_id,
                            matrix[last][nxt],
                            (mask | (1 << nxt), nxt),
                        )
                    )
                    action_id += 1
            if rows:
                actions_by_state[state] = tuple(rows)
    return v91.Problem(
        name=f"tsplib-held-karp:{name}",
        initial_state=initial,
        terminal_state=terminal,
        actions_by_state=actions_by_state,
    )


def brute_force_optimum(matrix: tuple[tuple[int, ...], ...]) -> int:
    n = len(matrix)
    best: int | None = None
    for order in itertools.permutations(range(1, n)):
        total = matrix[0][order[0]]
        for left, right in zip(order, order[1:]):
            total += matrix[left][right]
        total += matrix[order[-1]][0]
        if best is None or total < best:
            best = total
    if best is None:
        raise RuntimeError("empty v0.92 brute-force tour set")
    return best


def _load_lock_manifest() -> dict[str, object]:
    if not LOCK_MANIFEST.exists():
        raise RuntimeError("v0.92 evaluation locked until archive manifest is committed")
    payload = json.loads(LOCK_MANIFEST.read_text(encoding="utf-8"))
    if payload["status"] != "v92_tsplib_archives_hash_locked_not_opened":
        raise RuntimeError("v0.92 archive lock status changed")
    if payload["archive_count"] != 3:
        raise RuntimeError("v0.92 archive lock must contain three records")
    return payload


def verified_archives(lock_manifest: dict[str, object]) -> dict[str, bytes]:
    locked = {str(row["name"]): row for row in lock_manifest["records"]}
    archives = {}
    for source in lock_v92.selected_sources():
        row = locked.get(source["name"])
        if row is None:
            raise RuntimeError(f"v0.92 locked archive missing: {source['name']}")
        payload, final_url, _headers = lock_v92.download_raw(source["url"])
        if final_url != row["final_url"]:
            raise RuntimeError(f"v0.92 final URL changed: {source['name']}")
        if len(payload) != int(row["bytes"]):
            raise RuntimeError(f"v0.92 byte count changed: {source['name']}")
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise RuntimeError(f"v0.92 SHA-256 changed: {source['name']}")
        archives[source["name"]] = payload
    return archives


def evaluate() -> dict[str, object]:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg["status"] != "preregistered_before_selected_archive_access":
        raise RuntimeError("v0.92 preregistration status changed")
    frozen = prereg["frozen_v91_rule"]
    if EXPECTED_SPEC.digest() != EXPECTED_SPEC_DIGEST:
        raise RuntimeError("v0.92 compiled v0.91 spec digest changed")
    if frozen["selected_spec_digest"] != EXPECTED_SPEC_DIGEST:
        raise RuntimeError("v0.92 preregistered spec digest changed")
    if frozen["v91_freeze_digest"] != EXPECTED_V91_FREEZE_DIGEST:
        raise RuntimeError("v0.92 preregistered v0.91 freeze digest changed")

    lock_manifest = _load_lock_manifest()
    archives = verified_archives(lock_manifest)
    rows = []
    total_raw = 0
    total_candidate = 0
    positive = 0
    objective_mismatches = 0
    brute_force_mismatches = 0
    certificate_failures = 0
    for name in ("gr21", "gr24", "p43"):
        matrix = parse_projected_matrix(name, archives[name])
        problem = held_karp_problem(name, matrix)
        brute = brute_force_optimum(matrix)
        raw = v91.solve_with_spec(problem, None)
        candidate = v91.solve_with_spec(problem, EXPECTED_SPEC)
        reduction = raw.action_expansions - candidate.action_expansions
        brute_mismatch = brute != raw.objective
        objective_mismatch = candidate.objective != raw.objective
        local_certificate_failure = int(
            candidate.relation_certificates != candidate.actions_pruned
        )
        total_raw += raw.action_expansions
        total_candidate += candidate.action_expansions
        positive += int(reduction > 0)
        objective_mismatches += int(objective_mismatch)
        brute_force_mismatches += int(brute_mismatch)
        certificate_failures += local_certificate_failure
        rows.append({
            "name": name,
            "projected_city_count": PROJECTED_CITIES,
            "matrix_digest": matrix_digest(matrix),
            "brute_force_optimum": brute,
            "raw_held_karp_optimum": raw.objective,
            "candidate_optimum": candidate.objective,
            "brute_force_matches_raw": not brute_mismatch,
            "candidate_matches_raw": not objective_mismatch,
            "raw_action_expansions": raw.action_expansions,
            "candidate_action_expansions": candidate.action_expansions,
            "action_expansion_reduction": reduction,
            "action_expansion_reduction_fraction": (
                reduction / raw.action_expansions if raw.action_expansions else 0.0
            ),
            "actions_pruned": candidate.actions_pruned,
            "relation_certificates": candidate.relation_certificates,
            "local_certificate_failures": local_certificate_failure,
        })

    reduction = total_raw - total_candidate
    fraction = reduction / total_raw if total_raw else 0.0
    passed = (
        brute_force_mismatches == 0
        and objective_mismatches == 0
        and certificate_failures == 0
        and positive >= 2
        and fraction >= 0.05
    )
    return {
        "status": "tsplib_external_holdout_pass_v92" if passed else "tsplib_external_holdout_rejected_v92",
        "version": "v0.92",
        "passed": passed,
        "archive_records_digest": lock_manifest["records_digest"],
        "all_three_archives_reverified": True,
        "selected_spec": EXPECTED_SPEC.payload(),
        "selected_spec_digest": EXPECTED_SPEC_DIGEST,
        "v91_freeze_digest": EXPECTED_V91_FREEZE_DIGEST,
        "instances": rows,
        "summary": {
            "instances": 3,
            "brute_force_mismatches": brute_force_mismatches,
            "candidate_objective_mismatches": objective_mismatches,
            "local_certificate_failures": certificate_failures,
            "instances_with_positive_action_expansion_reduction": positive,
            "raw_action_expansions": total_raw,
            "candidate_action_expansions": total_candidate,
            "aggregate_action_expansion_reduction": reduction,
            "aggregate_action_expansion_reduction_fraction": fraction,
        },
        "claim_boundary": prereg["claim_boundary"],
        "kill_rule": prereg["kill_rule"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "passed": result["passed"],
        "summary": result["summary"],
        "instances": [
            {key: row[key] for key in (
                "name", "brute_force_optimum", "raw_held_karp_optimum",
                "candidate_optimum", "raw_action_expansions",
                "candidate_action_expansions", "action_expansion_reduction",
                "action_expansion_reduction_fraction", "actions_pruned"
            )}
            for row in result["instances"]
        ],
    }, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
