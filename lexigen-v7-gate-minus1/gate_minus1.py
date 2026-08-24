from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
from pathlib import Path


def apply_atom(name: str, xs: tuple[int, ...]) -> tuple[int, ...]:
    if name == "ABS":
        return tuple(abs(x) for x in xs)
    if name == "CLIP_POS":
        return tuple(max(0, x) for x in xs)
    if name == "CUMSUM":
        out, total = [], 0
        for x in xs:
            total += x
            out.append(total)
        return tuple(out)
    if name == "DIFF":
        return tuple(xs[i + 1] - xs[i] for i in range(len(xs) - 1))
    if name == "NEG":
        return tuple(-x for x in xs)
    if name == "REVERSE":
        return tuple(reversed(xs))
    if name == "SORT":
        return tuple(sorted(xs))
    if name == "UNIQUE":
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return tuple(out)
    raise ValueError(f"unknown atom {name}")


def execute(program: tuple[str, ...], xs: tuple[int, ...]) -> tuple[int, ...]:
    value = xs
    for atom in program:
        value = apply_atom(atom, value)
    return value


def behavior_signature(program: tuple[str, ...], inputs: list[list[int]]) -> str:
    outputs = [list(execute(program, tuple(xs))) for xs in inputs]
    payload = json.dumps(outputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def induce_macros(spec: dict) -> list[dict]:
    family_support: dict[tuple[str, ...], set[str]] = collections.defaultdict(set)
    occurrences: collections.Counter[tuple[str, ...]] = collections.Counter()
    for item in spec["apprenticeship"]:
        family = item["family"]
        program = tuple(item["program"])
        for length in spec["macro_lengths"]:
            for i in range(len(program) - length + 1):
                seq = program[i : i + length]
                family_support[seq].add(family)
                occurrences[seq] += 1

    ranked = []
    for seq, occurrence_count in occurrences.items():
        source_families = sorted(family_support[seq])
        if len(source_families) < spec["minimum_source_families_per_macro"]:
            continue
        mdl_savings = occurrence_count * (len(seq) - 1) - 1
        ranked.append((-mdl_savings, -len(source_families), -occurrence_count, -len(seq), seq, source_families))
    ranked.sort()
    selected = []
    for index, row in enumerate(ranked[: spec["learned_macro_count"]], start=1):
        neg_savings, neg_families, neg_occurrences, _neg_len, seq, source_families = row
        selected.append({
            "id": f"MG-{index:03d}",
            "sequence": list(seq),
            "source_families": source_families,
            "family_count": -neg_families,
            "occurrences": -neg_occurrences,
            "mdl_savings": -neg_savings,
        })
    return selected


def random_library(spec: dict, learned: list[dict]) -> list[dict]:
    rng = random.Random(spec["random_library_seed"])
    learned_sequences = {tuple(x["sequence"]) for x in learned}
    used = set()
    out = []
    atom_choices = list(spec["atoms"])
    lengths = [len(x["sequence"]) for x in learned]
    while len(out) < len(learned):
        length = lengths[len(out)]
        seq = tuple(rng.choice(atom_choices) for _ in range(length))
        if seq in learned_sequences or seq in used:
            continue
        used.add(seq)
        out.append({"id": f"RG-{len(out)+1:03d}", "sequence": list(seq)})
    return out


def token_space(spec: dict, library: list[dict]) -> list[tuple[str, tuple[str, ...]]]:
    tokens = [(f"@{item['id']}", tuple(item["sequence"])) for item in library]
    tokens.extend((atom, (atom,)) for atom in spec["atoms"])
    return sorted(tokens, key=lambda item: item[0])


def search_holdout(spec: dict, oracle: dict, holdout: dict, library: list[dict], removed_macro_id: str | None = None) -> dict:
    if removed_macro_id is not None:
        library = [x for x in library if x["id"] != removed_macro_id]
    tokens = token_space(spec, library)
    max_expanded = int(spec["max_expanded_program_length"])
    budget = int(spec["search_budget_evaluator_calls"])
    seen_expansions: set[tuple[str, ...]] = set()
    evaluator_calls = 0

    for token_depth in range(1, max_expanded + 1):
        for choice_indices in itertools.product(range(len(tokens)), repeat=token_depth):
            expansion = tuple(atom for index in choice_indices for atom in tokens[index][1])
            if len(expansion) > max_expanded or expansion in seen_expansions:
                continue
            seen_expansions.add(expansion)
            evaluator_calls += 1
            if behavior_signature(expansion, oracle["search_inputs"]) == holdout["search_signature_sha256"]:
                validation_sig = behavior_signature(expansion, oracle["validation_inputs"])
                if validation_sig == holdout["validation_signature_sha256"]:
                    return {
                        "found": True,
                        "evaluator_calls": evaluator_calls,
                        "program": list(expansion),
                        "tokens": [tokens[i][0] for i in choice_indices],
                        "validation_passed": True,
                        "budget": budget,
                    }
            if evaluator_calls >= budget:
                return {"found": False, "evaluator_calls": evaluator_calls, "validation_passed": False, "budget": budget}
    return {"found": False, "evaluator_calls": evaluator_calls, "validation_passed": False, "budget": budget}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run(spec: dict, oracle: dict) -> dict:
    learned = induce_macros(spec)
    random_lib = random_library(spec, learned)
    results = []
    removal_failures = learned_over_none = learned_over_random = successful_transfers = 0
    source_family_by_macro = {m["id"]: set(m["source_families"]) for m in learned}

    for holdout in oracle["holdouts"]:
        full = search_holdout(spec, oracle, holdout, learned)
        none = search_holdout(spec, oracle, holdout, [])
        random_result = search_holdout(spec, oracle, holdout, random_lib)
        used_macro_ids = [token[1:] for token in full.get("tokens", []) if token.startswith("@MG-")]
        cross_family_used = [macro_id for macro_id in used_macro_ids if holdout["family"] not in source_family_by_macro[macro_id]]
        transfer_success = bool(full["found"] and cross_family_used)
        successful_transfers += int(transfer_success)
        learned_over_none += int(full["found"] and not none["found"])
        learned_over_random += int(full["found"] and not random_result["found"])

        removals = {}
        specific_removal_failed = False
        for macro_id in sorted(set(cross_family_used)):
            replay = search_holdout(spec, oracle, holdout, learned, removed_macro_id=macro_id)
            removals[macro_id] = replay
            if full["found"] and not replay["found"]:
                specific_removal_failed = True
        removal_failures += int(specific_removal_failed)
        results.append({
            "family": holdout["family"],
            "full": full,
            "no_library": none,
            "random_library": random_result,
            "cross_family_macro_ids_used": cross_family_used,
            "prospective_transfer_success": transfer_success,
            "specific_macro_removal_replays": removals,
            "specific_macro_removal_failure": specific_removal_failed,
        })

    gate = spec["pass_gate"]
    checks = {
        "induced_macros_min": len(learned) >= gate["induced_macros_min"],
        "successful_holdout_transfers_min": successful_transfers >= gate["successful_holdout_transfers_min"],
        "learned_library_successes_over_no_library_min": learned_over_none >= gate["learned_library_successes_over_no_library_min"],
        "learned_library_successes_over_random_library_min": learned_over_random >= gate["learned_library_successes_over_random_library_min"],
        "specific_macro_removal_failures_min": removal_failures >= gate["specific_macro_removal_failures_min"],
        "human_task_specific_solver_hints_zero": gate["human_task_specific_solver_hints"] == 0,
    }
    return {
        "experiment": spec["experiment"],
        "status": "passed" if all(checks.values()) else "failed",
        "spec_sha256": canonical_sha256(spec),
        "oracle_sha256": canonical_sha256(oracle),
        "learned_macros": learned,
        "random_library": random_lib,
        "holdout_results": results,
        "aggregate": {
            "holdout_count": len(oracle["holdouts"]),
            "successful_prospective_transfers": successful_transfers,
            "learned_over_no_library_success_count": learned_over_none,
            "learned_over_random_library_success_count": learned_over_random,
            "specific_macro_removal_failure_count": removal_failures,
        },
        "gate_checks": checks,
        "claim_boundary": "Gate -1 is only an architecture-feasibility result on a committed synthetic DSL. Passing does not establish novelty, external generality, or an AI breakthrough."
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    oracle = json.loads(args.oracle.read_text())
    result = run(spec, oracle)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "GATE_MINUS1_R2_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
