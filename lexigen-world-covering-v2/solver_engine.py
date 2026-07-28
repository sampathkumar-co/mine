from __future__ import annotations

import hashlib
import random
import time
from dataclasses import asdict
from pathlib import Path

from ortools.sat.python import cp_model

from common import (
    CP_SAT_SECONDS,
    CP_SAT_WORKERS,
    GREEDY_DETERMINISTIC,
    GREEDY_RANDOMIZED,
    LOCAL_RESTARTS,
    LOCAL_SECONDS,
    SEED_MATERIAL,
    SNAPSHOT_MD5,
    SNAPSHOT_URL,
    Incidence,
    Target,
    build_incidence,
    verify_design,
)
from greedy import greedy_cover, normalize_to_budget, prune_redundant
from local_search import stochastic_fixed_budget


def cp_sat_search(
    target: Target, incidence: Incidence, hint: list[int], seed: int
) -> tuple[list[int], str, dict[str, object]]:
    goal = target.upper - 1
    model = cp_model.CpModel()
    variables = [model.new_bool_var(f"b{index}") for index in range(len(incidence.blocks))]
    for containing_blocks in incidence.blocks_by_t:
        model.add(sum(variables[index] for index in containing_blocks) >= 1)
    model.add(sum(variables) <= goal)
    model.add(variables[0] == 1)
    model.minimize(sum(variables))
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
        selected = [i for i, variable in enumerate(variables) if solver.value(variable)]
    return selected, status_name, {
        "status": status_name,
        "objective": solver.objective_value if selected else None,
        "best_bound": solver.best_objective_bound if selected else None,
        "response_stats": solver.response_stats(),
    }


def solve_target(target: Target, output_dir: Path) -> dict[str, object]:
    started = time.time()
    incidence = build_incidence(target)
    base_seed = int(hashlib.sha256(f"{SEED_MATERIAL}|{target.name}".encode()).hexdigest(), 16)
    goal = target.upper - 1

    greedy_best: list[int] = []
    greedy_runs: list[dict[str, object]] = []
    for attempt in range(GREEDY_DETERMINISTIC + GREEDY_RANDOMIZED):
        randomized = attempt >= GREEDY_DETERMINISTIC
        seed = (base_seed + attempt * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        raw = greedy_cover(
            incidence,
            seed=seed,
            randomized=randomized,
            step_limit=max(target.upper * 2, target.lower + 24),
        )
        reduced = prune_redundant(incidence, raw, seed) if raw else []
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

    local_best: list[int] = []
    local_runs: list[dict[str, object]] = []
    if not greedy_best or len(greedy_best) > goal:
        local_deadline = time.monotonic() + LOCAL_SECONDS
        per_restart = LOCAL_SECONDS / LOCAL_RESTARTS
        for restart in range(LOCAL_RESTARTS):
            restart_deadline = min(local_deadline, time.monotonic() + per_restart)
            seed = (base_seed ^ (restart * 0xD1B54A32D192ED03)) & ((1 << 64) - 1)
            candidate, stats = stochastic_fixed_budget(
                incidence, greedy_best, goal, seed, restart_deadline
            )
            candidate_blocks = [incidence.blocks[index] for index in candidate]
            valid, _ = verify_design(target, candidate_blocks)
            stats.update({"restart": restart, "valid": valid})
            local_runs.append(stats)
            if valid:
                local_best = candidate
                break
            if time.monotonic() >= local_deadline:
                break

    selected = greedy_best if greedy_best and len(greedy_best) <= goal else local_best
    method = "generic_greedy" if selected else "cp_sat_minimization"
    cp_status = "not_run"
    cp_stats: dict[str, object] | None = None
    if not selected:
        hint = (
            normalize_to_budget(
                incidence,
                greedy_best,
                min(goal, max(1, len(greedy_best))),
                random.Random(base_seed),
            )
            if greedy_best
            else []
        )
        selected, cp_status, cp_stats = cp_sat_search(target, incidence, hint, base_seed)

    blocks = [incidence.blocks[index] for index in selected] if selected else []
    valid, verification = verify_design(target, blocks) if blocks else (False, "no design returned")
    record_candidate = valid and len(blocks) <= goal
    result = {
        "protocol": "LEXIGEN World Covering Record v2",
        "snapshot_url": SNAPSHOT_URL,
        "snapshot_md5": SNAPSHOT_MD5,
        "target": asdict(target),
        "goal_blocks": goal,
        "method": method,
        "greedy_best_blocks": len(greedy_best) if greedy_best else None,
        "greedy_runs": greedy_runs,
        "local_runs": local_runs,
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
    destination.write_text(__import__("json").dumps(result, indent=2), encoding="utf-8")
    return result
