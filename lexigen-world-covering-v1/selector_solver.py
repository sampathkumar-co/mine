from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import math
import random
import re
import time
import urllib.request
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ortools.sat.python import cp_model

SNAPSHOT_URL = "https://zenodo.org/records/19735294/files/coverdata.json?download=1"
SNAPSHOT_MD5 = "b2c626b07f216aac830d344eff5ad523"
SEED_MATERIAL = (
    "32c897005c91865319f1b7da264b6162fc1ff4de|"
    "b2c626b07f216aac830d344eff5ad523|LEXIGEN_WORLD_COVERING_V1"
)
REFERENCE_DATE = datetime(2026, 4, 24, tzinfo=timezone.utc)
TARGET_COUNT = 3
GREEDY_DETERMINISTIC = 8
GREEDY_RANDOMIZED = 24
CP_SAT_SECONDS = 1200.0
CP_SAT_WORKERS = 4
KEY_RE = re.compile(r"^C\((\d+),(\d+),(\d+)\)$")


@dataclass(frozen=True)
class Target:
    name: str
    v: int
    k: int
    t: int
    upper: int
    lower: int
    last_update: str
    gap: int
    candidate_blocks: int
    t_subsets: int
    incidence_edges: int
    opportunity_score: float
    tie_break: str


def download_snapshot() -> bytes:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                SNAPSHOT_URL,
                headers={"User-Agent": "LEXIGEN-world-covering-v1"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                data = response.read()
            digest = hashlib.md5(data).hexdigest()
            if digest != SNAPSHOT_MD5:
                raise RuntimeError(f"snapshot MD5 mismatch: {digest} != {SNAPSHOT_MD5}")
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError("could not download frozen covering snapshot") from last_error


def parse_date(text: str) -> datetime:
    value = (text or "").strip()
    if not value:
        return datetime(1996, 1, 1, tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime(1996, 1, 1, tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def target_from_entry(name: str, entry: dict[str, object]) -> Target | None:
    match = KEY_RE.match(name)
    if not match:
        return None
    v, k, t = map(int, match.groups())
    upper = int(entry["size"])
    lower = int(entry["low_bd"])
    gap = upper - lower
    if not (
        10 <= v <= 22
        and 4 <= k <= min(10, v - 2)
        and 3 <= t <= min(5, k - 1)
        and gap >= 2
        and upper <= 100
        and upper - 1 >= lower
    ):
        return None
    candidate_blocks = math.comb(v, k)
    t_subsets = math.comb(v, t)
    incidence_edges = candidate_blocks * math.comb(k, t)
    if candidate_blocks > 50_000 or t_subsets > 5_000 or incidence_edges > 3_000_000:
        return None

    improvements = entry.get("imps") or []
    last_update = ""
    if isinstance(improvements, list) and improvements:
        row = improvements[0]
        if isinstance(row, list) and len(row) >= 4:
            last_update = str(row[3])
    age_years = max(0.0, (REFERENCE_DATE - parse_date(last_update)).days / 365.25)
    age_factor = 1.0 + min(age_years, 25.0) / 18.0
    gap_factor = float(gap) ** 1.35
    complexity_factor = float(incidence_edges) ** 0.35
    size_factor = (100.0 / float(upper)) ** 0.15
    opportunity_score = gap_factor * age_factor * size_factor / complexity_factor
    tie_break = hashlib.sha256(f"{SEED_MATERIAL}|{name}".encode()).hexdigest()
    return Target(
        name=name,
        v=v,
        k=k,
        t=t,
        upper=upper,
        lower=lower,
        last_update=last_update,
        gap=gap,
        candidate_blocks=candidate_blocks,
        t_subsets=t_subsets,
        incidence_edges=incidence_edges,
        opportunity_score=opportunity_score,
        tie_break=tie_break,
    )


def select_targets(coverdata: dict[str, object]) -> list[Target]:
    eligible: list[Target] = []
    for name, raw in coverdata.items():
        if isinstance(raw, dict):
            target = target_from_entry(name, raw)
            if target is not None:
                eligible.append(target)
    eligible.sort(key=lambda x: (-x.opportunity_score, x.tie_break, x.name))
    selected: list[Target] = []
    pair_counts: dict[tuple[int, int], int] = {}
    for target in eligible:
        pair = (target.k, target.t)
        if pair_counts.get(pair, 0) >= 2:
            continue
        selected.append(target)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if len(selected) == TARGET_COUNT:
            break
    if len(selected) != TARGET_COUNT:
        raise RuntimeError(f"selector found only {len(selected)} eligible targets")
    return selected


@dataclass
class Incidence:
    blocks: list[tuple[int, ...]]
    tsets: list[tuple[int, ...]]
    cover_by_block: list[array]
    blocks_by_t: list[array]


def build_incidence(target: Target) -> Incidence:
    points = range(target.v)
    tsets = list(itertools.combinations(points, target.t))
    t_index = {subset: index for index, subset in enumerate(tsets)}
    blocks: list[tuple[int, ...]] = []
    cover_by_block: list[array] = []
    blocks_by_t = [array("I") for _ in tsets]
    for block_index, block in enumerate(itertools.combinations(points, target.k)):
        covered = array("I", (t_index[s] for s in itertools.combinations(block, target.t)))
        blocks.append(block)
        cover_by_block.append(covered)
        for t_index_value in covered:
            blocks_by_t[t_index_value].append(block_index)
    if len(blocks) != target.candidate_blocks or len(tsets) != target.t_subsets:
        raise RuntimeError("incidence dimensions differ from frozen selector metadata")
    return Incidence(blocks, tsets, cover_by_block, blocks_by_t)


def weighted_choice(rng: random.Random, top: list[tuple[int, float, int]]) -> int:
    weights = [max(1, gain) ** 3 for gain, _, _ in top]
    total = sum(weights)
    draw = rng.randrange(total)
    for weight, (_, _, block_index) in zip(weights, top):
        if draw < weight:
            return block_index
        draw -= weight
    return top[-1][2]


def greedy_cover(
    incidence: Incidence,
    seed: int,
    randomized: bool,
    step_limit: int,
) -> list[int]:
    rng = random.Random(seed)
    uncovered = bytearray(b"\x01") * len(incidence.tsets)
    remaining = len(incidence.tsets)
    gains = [len(covered) for covered in incidence.cover_by_block]
    heap: list[tuple[int, float, int]] = [
        (-gain, rng.random(), index) for index, gain in enumerate(gains)
    ]
    heapq.heapify(heap)
    selected: list[int] = []
    selected_set: set[int] = set()

    def pop_valid() -> tuple[int, float, int] | None:
        while heap:
            neg_gain, tie, block_index = heapq.heappop(heap)
            if block_index not in selected_set and -neg_gain == gains[block_index]:
                return (-neg_gain, tie, block_index)
        return None

    while remaining and len(selected) < step_limit:
        valid_top: list[tuple[int, float, int]] = []
        width = 8 if randomized else 1
        for _ in range(width):
            item = pop_valid()
            if item is None:
                break
            valid_top.append(item)
        if not valid_top or valid_top[0][0] <= 0:
            break
        chosen = weighted_choice(rng, valid_top) if randomized else valid_top[0][2]
        for item in valid_top:
            if item[2] != chosen:
                heapq.heappush(heap, (-item[0], item[1], item[2]))
        selected.append(chosen)
        selected_set.add(chosen)

        newly_covered: list[int] = []
        for t_index_value in incidence.cover_by_block[chosen]:
            if uncovered[t_index_value]:
                uncovered[t_index_value] = 0
                remaining -= 1
                newly_covered.append(t_index_value)
        for t_index_value in newly_covered:
            for block_index in incidence.blocks_by_t[t_index_value]:
                if gains[block_index] > 0 and block_index not in selected_set:
                    gains[block_index] -= 1
                    heapq.heappush(
                        heap, (-gains[block_index], rng.random(), block_index)
                    )
    if remaining:
        return []
    return selected


def coverage_counts(incidence: Incidence, selected: list[int]) -> list[int]:
    counts = [0] * len(incidence.tsets)
    for block_index in selected:
        for t_index_value in incidence.cover_by_block[block_index]:
            counts[t_index_value] += 1
    return counts


def prune_redundant(
    incidence: Incidence,
    selected: list[int],
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    selected = list(dict.fromkeys(selected))
    counts = coverage_counts(incidence, selected)
    changed = True
    while changed:
        changed = False
        order = list(selected)
        rng.shuffle(order)
        order.sort(
            key=lambda block_index: sum(
                1 for t_index_value in incidence.cover_by_block[block_index]
                if counts[t_index_value] == 1
            )
        )
        for block_index in order:
            if block_index not in selected:
                continue
            if all(counts[t_index_value] >= 2 for t_index_value in incidence.cover_by_block[block_index]):
                selected.remove(block_index)
                for t_index_value in incidence.cover_by_block[block_index]:
                    counts[t_index_value] -= 1
                changed = True
    return selected


def two_for_one_reduce(
    incidence: Incidence,
    selected: list[int],
    goal: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    selected = list(selected)
    while len(selected) > goal:
        counts = coverage_counts(incidence, selected)
        selected_set = set(selected)
        pairs = list(itertools.combinations(selected, 2))
        rng.shuffle(pairs)
        replacement: tuple[int, int, int] | None = None
        for first, second in pairs:
            removed_contribution: dict[int, int] = {}
            for t_index_value in incidence.cover_by_block[first]:
                removed_contribution[t_index_value] = removed_contribution.get(t_index_value, 0) + 1
            for t_index_value in incidence.cover_by_block[second]:
                removed_contribution[t_index_value] = removed_contribution.get(t_index_value, 0) + 1
            required = [
                t_index_value
                for t_index_value, amount in removed_contribution.items()
                if counts[t_index_value] - amount <= 0
            ]
            if not required:
                selected = [x for x in selected if x not in (first, second)]
                replacement = (first, second, -1)
                break
            anchor = min(required, key=lambda x: len(incidence.blocks_by_t[x]))
            required_set = set(required)
            candidates = list(incidence.blocks_by_t[anchor])
            rng.shuffle(candidates)
            for candidate in candidates:
                if candidate in selected_set:
                    continue
                covered_required = sum(
                    1 for t_index_value in incidence.cover_by_block[candidate]
                    if t_index_value in required_set
                )
                if covered_required == len(required_set):
                    replacement = (first, second, candidate)
                    break
            if replacement is not None:
                break
        if replacement is None:
            break
        first, second, candidate = replacement
        selected = [x for x in selected if x not in (first, second)]
        if candidate >= 0:
            selected.append(candidate)
        selected = prune_redundant(incidence, selected, seed ^ len(selected))
    return selected


def verify_design(target: Target, blocks: list[tuple[int, ...]]) -> tuple[bool, str]:
    if len(blocks) != len(set(blocks)):
        return False, "duplicate blocks"
    point_set = set(range(target.v))
    for block in blocks:
        if len(block) != target.k or tuple(sorted(block)) != block:
            return False, "malformed block"
        if not set(block).issubset(point_set):
            return False, "point outside universe"
    covered: set[tuple[int, ...]] = set()
    for block in blocks:
        covered.update(itertools.combinations(block, target.t))
    expected = math.comb(target.v, target.t)
    if len(covered) != expected:
        return False, f"covered {len(covered)} of {expected} t-subsets"
    if len(blocks) >= target.upper:
        return False, "not strictly smaller than frozen upper bound"
    return True, "verified"


def cp_sat_search(
    target: Target,
    incidence: Incidence,
    hint: list[int],
    seed: int,
) -> tuple[list[int], str, dict[str, object]]:
    goal = target.upper - 1
    model = cp_model.CpModel()
    variables = [model.new_bool_var(f"b{index}") for index in range(len(incidence.blocks))]
    for containing_blocks in incidence.blocks_by_t:
        model.add(sum(variables[index] for index in containing_blocks) >= 1)
    model.add(sum(variables) <= goal)
    model.add(variables[0] == 1)
    for block_index in hint:
        model.add_hint(variables[block_index], 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = CP_SAT_SECONDS
    solver.parameters.num_search_workers = CP_SAT_WORKERS
    solver.parameters.random_seed = seed & 0x7FFFFFFF
    solver.parameters.log_search_progress = True
    solver.parameters.cp_model_presolve = True
    status = solver.solve(model)
    status_name = solver.status_name(status)
    selected: list[int] = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = [index for index, variable in enumerate(variables) if solver.value(variable)]
    stats = {
        "status": status_name,
        "response_stats": solver.response_stats(),
    }
    return selected, status_name, stats


def solve_target(target: Target, output_dir: Path) -> dict[str, object]:
    started = time.time()
    incidence = build_incidence(target)
    base_seed = int(hashlib.sha256(f"{SEED_MATERIAL}|{target.name}".encode()).hexdigest(), 16)
    goal = target.upper - 1
    best: list[int] = []
    heuristic_runs: list[dict[str, object]] = []
    attempts = GREEDY_DETERMINISTIC + GREEDY_RANDOMIZED
    for attempt in range(attempts):
        randomized = attempt >= GREEDY_DETERMINISTIC
        seed = (base_seed + attempt * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        raw = greedy_cover(
            incidence,
            seed=seed,
            randomized=randomized,
            step_limit=max(target.upper * 2, target.lower + 20),
        )
        reduced: list[int] = []
        if raw:
            reduced = prune_redundant(incidence, raw, seed)
            reduced = two_for_one_reduce(incidence, reduced, goal, seed ^ 0xA5A5A5A5)
        heuristic_runs.append(
            {
                "attempt": attempt,
                "randomized": randomized,
                "raw_blocks": len(raw) if raw else None,
                "reduced_blocks": len(reduced) if reduced else None,
            }
        )
        if reduced and (not best or len(reduced) < len(best)):
            best = reduced
        if best and len(best) <= goal:
            break

    method = "generic_greedy_local_search"
    cp_stats: dict[str, object] | None = None
    cp_status = "not_run"
    selected = best
    if not selected or len(selected) > goal:
        selected, cp_status, cp_stats = cp_sat_search(
            target, incidence, best, base_seed & 0x7FFFFFFF
        )
        method = "cp_sat_feasibility"

    blocks = [incidence.blocks[index] for index in selected] if selected else []
    valid, verification = verify_design(target, blocks) if blocks else (False, "no design returned")
    record_candidate = valid and len(blocks) <= goal
    result = {
        "protocol": "LEXIGEN World Covering Record v1",
        "snapshot_url": SNAPSHOT_URL,
        "snapshot_md5": SNAPSHOT_MD5,
        "target": {
            "name": target.name,
            "v": target.v,
            "k": target.k,
            "t": target.t,
            "frozen_upper_bound": target.upper,
            "frozen_lower_bound": target.lower,
            "last_update": target.last_update,
            "opportunity_score": target.opportunity_score,
            "candidate_blocks": target.candidate_blocks,
            "t_subsets": target.t_subsets,
            "incidence_edges": target.incidence_edges,
        },
        "goal_blocks": goal,
        "method": method,
        "heuristic_best_blocks": len(best) if best else None,
        "heuristic_runs": heuristic_runs,
        "cp_sat_status": cp_status,
        "cp_sat_stats": cp_stats,
        "result_blocks": len(blocks) if blocks else None,
        "valid": valid,
        "verification": verification,
        "record_candidate": record_candidate,
        "blocks_zero_based": [list(block) for block in blocks] if record_candidate else [],
        "elapsed_s": time.time() - started,
    }
    destination = output_dir / f"{target.name.replace('(', '_').replace(')', '').replace(',', '_')}.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    snapshot = download_snapshot()
    coverdata = json.loads(snapshot)
    if not isinstance(coverdata, dict):
        raise TypeError("coverdata root must be a dictionary")
    targets = select_targets(coverdata)
    selection = {
        "snapshot_md5": SNAPSHOT_MD5,
        "seed_material_sha256": hashlib.sha256(SEED_MATERIAL.encode()).hexdigest(),
        "targets": [target.__dict__ for target in targets],
    }
    (args.output / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print("FROZEN_SELECTED_TARGETS")
    print(json.dumps(selection, indent=2), flush=True)

    results = []
    for target in targets:
        print(f"START {target.name} upper={target.upper} lower={target.lower}", flush=True)
        result = solve_target(target, args.output)
        results.append(result)
        print(
            f"FINISH {target.name} valid={result['valid']} "
            f"blocks={result['result_blocks']} record={result['record_candidate']}",
            flush=True,
        )

    root = Path(__file__).resolve().parent
    summary = {
        "protocol": "LEXIGEN World Covering Record v1",
        "snapshot_md5": SNAPSHOT_MD5,
        "selected_count": len(targets),
        "record_candidates": sum(bool(result["record_candidate"]) for result in results),
        "results": results,
        "code_hashes": {
            "selector_solver.py": file_sha256(root / "selector_solver.py"),
            "PROTOCOL.md": file_sha256(root / "PROTOCOL.md"),
            "requirements.txt": file_sha256(root / "requirements.txt"),
        },
    }
    (args.output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
