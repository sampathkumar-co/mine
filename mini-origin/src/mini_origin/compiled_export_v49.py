from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import average_odt_frontier_v44 as frontier
from . import cost_prior_quotient_v48 as weighted
from . import exact_quotient_certificate_v42 as corpus


def indices(mask: int) -> list[int]:
    values = []
    pending = mask
    while pending:
        bit = pending & -pending
        values.append(bit.bit_length() - 1)
        pending ^= bit
    return values


def compact_state(
    task: object,
    allowed: int,
    remaining: int,
    seed: int,
) -> dict[str, object]:
    candidates = indices(allowed)
    queries = indices(remaining)
    profile = weighted.profile_for_task(task, seed)
    label_values = sorted({task.labels[index] for index in candidates})
    label_ids = {value: index for index, value in enumerate(label_values)}
    labels = [label_ids[task.labels[index]] for index in candidates]
    masses = [profile.hypothesis_mass[index] for index in candidates]
    costs = [profile.query_cost[query] for query in queries]
    matrix = []
    response_maps = []
    for query in queries:
        values = sorted({task.rows[index][query] for index in candidates})
        response_maps.append({value: index for index, value in enumerate(values)})
    for candidate in candidates:
        matrix.append([
            response_maps[column][task.rows[candidate][query]]
            for column, query in enumerate(queries)
        ])
    base_digest = hashlib.sha256(
        f"{task.name}:{allowed}:{remaining}".encode("utf-8")
    ).hexdigest()
    digest = hashlib.sha256(
        f"{base_digest}:{seed}".encode("utf-8")
    ).hexdigest()
    return {
        "digest": digest,
        "base_digest": base_digest,
        "profile_seed": seed,
        "labels": labels,
        "masses": masses,
        "query_ids": queries,
        "costs": costs,
        "matrix": matrix,
    }


def write_text(states: list[dict[str, object]], path: Path) -> None:
    lines = [f"COUNT {len(states)}"]
    for row in states:
        labels = row["labels"]
        query_ids = row["query_ids"]
        lines.append(
            f"STATE {row['digest']} {row['profile_seed']} "
            f"{len(labels)} {len(query_ids)}"
        )
        lines.append("LABELS " + " ".join(map(str, labels)))
        lines.append("MASSES " + " ".join(map(str, row["masses"])))
        lines.append("QUERY_IDS " + " ".join(map(str, query_ids)))
        lines.append("COSTS " + " ".join(map(str, row["costs"])))
        for values in row["matrix"]:
            lines.append("ROW " + " ".join(map(str, values)))
        lines.append("END")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path, manifest: Path) -> dict[str, object]:
    tasks, verification = corpus.load_all_opened_tasks()
    states = []
    base_digests = set()
    for task in tasks:
        selected, _ = frontier.select_frontier_states(task)
        for allowed, remaining, _ in selected:
            for seed in weighted.PROFILE_SEEDS:
                row = compact_state(task, allowed, remaining, seed)
                base_digests.add(row["base_digest"])
                states.append(row)
    states.sort(key=lambda row: row["digest"])
    write_text(states, output)
    payload = {
        "status": "compiled_input_frozen",
        "format": "mini-origin-weighted-state-v1",
        "state_count": len(states),
        "base_state_count": len(base_digests),
        "profile_seeds": list(weighted.PROFILE_SEEDS),
        "parent_weighted_digest": (
            "cfdbb2d072d426d7b4a684986af23d573976809fc1fa0949e4edb5579b6012b0"
        ),
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
