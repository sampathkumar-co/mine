from __future__ import annotations

import hashlib
import random
import time
from dataclasses import asdict
from pathlib import Path

from ortools.sat.python import cp_model

from common import (
    CP_WORKERS,
    FULL_CP_SECONDS,
    GREEDY_DETERMINISTIC,
    GREEDY_RANDOMIZED,
    REPAIR_RESTARTS,
    REPAIR_SECONDS,
    RESTRICTED_CP_SECONDS,
    SNAPSHOT_MD5,
    SNAPSHOT_URL,
    V3_SEED,
    Incidence,
    Target,
    build_incidence,
    verify_design,
)
from greedy import greedy_cover, normalize_budget, prune
from repair import fixed_budget_repair


def cp_search(
    target: Target,
    incidence: Incidence,
    allowed: list[int] | None,
    hint: list[int],
    seconds: float,
    seed: int,
) -> tuple[list[int], str, dict[str, object]]:
    indices = allowed if allowed is not None else list(range(len(incidence.blocks)))
    index_set = set(indices)
    model = cp_model.CpModel()
    variables = {i: model.new_bool_var(f"b{i}") for i in indices}
    for containing in incidence.blocks_by_t:
        usable = [variables[i] for i in containing if i in index_set]
        if not usable:
            return [], "POOL_INCOMPLETE", {"pool_size": len(indices)}
        model.add(sum(usable) >= 1)
    goal = target.upper - 1
    model.add(sum(variables.values()) <= goal)
    if 0 in variables:
        model.add(variables[0] == 1)
    for i in hint:
        if i in variables:
            model.add_hint(variables[i], 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = CP_WORKERS
    solver.parameters.random_seed = seed & 0x7FFFFFFF
    solver.parameters.cp_model_presolve = True
    solver.parameters.log_search_progress = True
    status = solver.solve(model)
    status_name = solver.status_name(status)
    selected = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = [i for i, var in variables.items() if solver.value(var)]
    return selected, status_name, {
        "pool_size": len(indices),
        "status": status_name,
        "response_stats": solver.response_stats(),
    }


def candidate_pool(incidence: Incidence, hint: list[int], per_subset: int = 12) -> list[int]:
    pool = set(hint)
    pool.add(0)
    global_scores = [len(row) for row in incidence.cover_by_block]
    pool.update(sorted(range(len(global_scores)), key=lambda i: -global_scores[i])[:512])
    for containing in incidence.blocks_by_t:
        ranked = sorted(containing, key=lambda i: (-global_scores[i], i))
        pool.update(ranked[:per_subset])
    return sorted(pool)


def solve_target(target: Target, output_dir: Path) -> dict[str, object]:
    started = time.time()
    incidence = build_incidence(target)
    base_seed = int(hashlib.sha256(f"{V3_SEED}|{target.name}".encode()).hexdigest(), 16)
    goal = target.upper - 1

    greedy_best = []
    greedy_runs = []
    for attempt in range(GREEDY_DETERMINISTIC + GREEDY_RANDOMIZED):
        randomized = attempt >= GREEDY_DETERMINISTIC
        seed = (base_seed + attempt * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        raw = greedy_cover(
            incidence,
            seed,
            randomized,
            max(target.upper * 2, target.lower + 28),
        )
        reduced = prune(incidence, raw, seed) if raw else []
        greedy_runs.append(
            {
                "attempt": attempt,
                "randomized": randomized,
                "raw_blocks": len(raw) if raw else None,
                "reduced_blocks": len(reduced) if reduced else None,
            }
        )
        if reduced and (not greedy_best or len(reduced) < len(greedy_best)):
            greedy_best = reduced
        if greedy_best and len(greedy_best) <= goal:
            break

    selected = []
    method = "none"
    repair_runs = []
    if greedy_best and len(greedy_best) <= goal:
        selected = greedy_best
        method = "generic_greedy"
    else:
        overall_deadline = time.monotonic() + REPAIR_SECONDS
        per_restart = REPAIR_SECONDS / REPAIR_RESTARTS
        for restart in range(REPAIR_RESTARTS):
            deadline = min(overall_deadline, time.monotonic() + per_restart)
            seed = (base_seed ^ (restart * 0xD1B54A32D192ED03)) & ((1 << 64) - 1)
            candidate, stats = fixed_budget_repair(
                incidence, greedy_best, goal, seed, deadline
            )
            blocks = [incidence.blocks[i] for i in candidate]
            valid, _ = verify_design(target, blocks)
            stats.update({"restart": restart, "valid": valid})
            repair_runs.append(stats)
            if valid:
                selected = candidate
                method = "guided_fixed_budget_repair"
                break
            if time.monotonic() >= overall_deadline:
                break

    restricted_status = "not_run"
    restricted_stats = None
    if not selected:
        hint = (
            normalize_budget(
                incidence,
                greedy_best,
                min(goal, max(1, len(greedy_best))),
                random.Random(base_seed),
            )
            if greedy_best
            else []
        )
        pool = candidate_pool(incidence, hint)
        selected, restricted_status, restricted_stats = cp_search(
            target, incidence, pool, hint, RESTRICTED_CP_SECONDS, base_seed
        )
        if selected:
            method = "restricted_cp_sat"

    full_status = "not_run"
    full_stats = None
    if not selected:
        hint = (
            normalize_budget(
                incidence,
                greedy_best,
                min(goal, max(1, len(greedy_best))),
                random.Random(base_seed),
            )
            if greedy_best
            else []
        )
        selected, full_status, full_stats = cp_search(
            target,
            incidence,
            None,
            hint,
            FULL_CP_SECONDS,
            base_seed ^ 0xA5A5A5A5,
        )
        if selected:
            method = "full_cp_sat"

    blocks = [incidence.blocks[i] for i in selected] if selected else []
    valid, verification = (
        verify_design(target, blocks) if blocks else (False, "no design returned")
    )
    record_candidate = valid and len(blocks) <= goal
    result = {
        "protocol": "LEXIGEN World Covering Record v3",
        "snapshot_url": SNAPSHOT_URL,
        "snapshot_md5": SNAPSHOT_MD5,
        "target": asdict(target),
        "goal_blocks": goal,
        "method": method,
        "greedy_best_blocks": len(greedy_best) if greedy_best else None,
        "greedy_runs": greedy_runs,
        "repair_runs": repair_runs,
        "restricted_cp_status": restricted_status,
        "restricted_cp_stats": restricted_stats,
        "full_cp_status": full_status,
        "full_cp_stats": full_stats,
        "result_blocks": len(blocks) if blocks else None,
        "valid": valid,
        "verification": verification,
        "record_candidate": record_candidate,
        "blocks_zero_based": [list(block) for block in blocks] if record_candidate else [],
        "elapsed_s": time.time() - started,
    }
    path = output_dir / f"{target.name.replace('(', '_').replace(')', '').replace(',', '_')}.json"
    path.write_text(__import__("json").dumps(result, indent=2), encoding="utf-8")
    return result
