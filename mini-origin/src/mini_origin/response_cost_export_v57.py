from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import average_odt_frontier_v44 as frontier
from . import exact_quotient_certificate_v42 as corpus
from . import response_cost_pareto_v56 as response


def indices(mask: int) -> list[int]:
    result: list[int] = []
    pending = mask
    while pending:
        bit = pending & -pending
        result.append(bit.bit_length() - 1)
        pending ^= bit
    return result


def compact_state(task: object, allowed: int, remaining: int, seed: int) -> dict[str, object]:
    candidates = indices(allowed)
    queries = indices(remaining)
    profile = response.profile_for_task(task, seed)
    label_values = sorted({task.labels[index] for index in candidates})
    label_ids = {value: index for index, value in enumerate(label_values)}
    labels = [label_ids[task.labels[index]] for index in candidates]
    masses = [profile.hypothesis_mass[index] for index in candidates]
    response_maps: list[dict[object, int]] = []
    for query in queries:
        values = sorted({task.rows[index][query] for index in candidates})
        response_maps.append({value: index for index, value in enumerate(values)})
    matrix = [
        [
            response_maps[column][task.rows[candidate][query]]
            for column, query in enumerate(queries)
        ]
        for candidate in candidates
    ]
    costs = [
        [profile.hypothesis_cost_by_query[query][candidate] for query in queries]
        for candidate in candidates
    ]
    base_digest = hashlib.sha256(
        f"{task.name}:{allowed}:{remaining}".encode("utf-8")
    ).hexdigest()
    digest = hashlib.sha256(f"{base_digest}:{seed}:response-cost-v57".encode("utf-8")).hexdigest()
    return {
        "digest": digest,
        "base_digest": base_digest,
        "task": task.name,
        "profile_seed": seed,
        "labels": labels,
        "masses": masses,
        "query_ids": queries,
        "matrix": matrix,
        "costs": costs,
    }


def write_text(states: list[dict[str, object]], path: Path) -> None:
    lines = [f"COUNT {len(states)}"]
    for row in states:
        labels = row["labels"]
        query_ids = row["query_ids"]
        lines.append(
            f"STATE {row['digest']} {row['profile_seed']} {len(labels)} {len(query_ids)}"
        )
        lines.append("LABELS " + " ".join(map(str, labels)))
        lines.append("MASSES " + " ".join(map(str, row["masses"])))
        lines.append("QUERY_IDS " + " ".join(map(str, query_ids)))
        for values in row["matrix"]:
            lines.append("RESPONSES " + " ".join(map(str, values)))
        for values in row["costs"]:
            lines.append("COSTS " + " ".join(map(str, values)))
        lines.append("END")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path, manifest: Path) -> dict[str, object]:
    tasks, verification = corpus.load_all_opened_tasks()
    states: list[dict[str, object]] = []
    base_digests: set[str] = set()
    for task in tasks:
        selected, _ = frontier.select_frontier_states(task)
        for allowed, remaining, _ in selected:
            for seed in response.PROFILE_SEEDS:
                row = compact_state(task, allowed, remaining, seed)
                states.append(row)
                base_digests.add(str(row["base_digest"]))
    states.sort(key=lambda row: str(row["digest"]))
    write_text(states, output)
    payload = {
        "status": "response_cost_compiled_input_frozen",
        "format": "mini-origin-response-cost-state-v1",
        "state_count": len(states),
        "base_state_count": len(base_digests),
        "profile_seeds": list(response.PROFILE_SEEDS),
        "parent_v56_digest": "5d0c281c66200ae378af25e6e6214e6b8b49b9fd144d4060fe29afc0a795a1cc",
        "archive_verification": verification,
        "input_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "state_digests": [row["digest"] for row in states],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.manifest), indent=2))


if __name__ == "__main__":
    main()
