from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

import numpy as np

from . import attainable_envelope_v38 as v38
from . import exact_tail_v36 as v36
from . import safe_portfolio_v37 as v37
from . import state_policy_v34 as v34


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "external-data" / "uci-v39" / "manifest.json"
SEEDS = (2201, 2202, 2203, 2204, 2205)
CANDIDATE_THRESHOLD = 12
FEATURE_LIMIT = 12
FROZEN_DEVELOPMENT_DIGEST = (
    "8117605201d4a4dba684757b2802e4426cacb0065e878b58708196606bfa28dc"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_and_verify(root: Path) -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["rows"]:
        request = urllib.request.Request(
            entry["url"],
            headers={"User-Agent": "Mini-ORIGIN-v0.39/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        actual_hash = sha256_bytes(payload)
        actual_bytes = len(payload)
        matched = (
            actual_hash == entry["sha256"]
            and actual_bytes == entry["bytes"]
        )
        if not matched:
            raise RuntimeError(
                f"archive commitment mismatch: {entry['name']}"
            )
        (root / entry["archive"]).write_bytes(payload)
        rows.append({
            "name": entry["name"],
            "archive": entry["archive"],
            "expected_sha256": entry["sha256"],
            "actual_sha256": actual_hash,
            "expected_bytes": entry["bytes"],
            "actual_bytes": actual_bytes,
            "matched": matched,
        })
    return {
        "protocol": manifest["protocol"],
        "source": manifest["source"],
        "license": manifest["license"],
        "workflow_run": manifest["workflow_run"],
        "workflow_artifact_digest": manifest[
            "workflow_artifact_digest"
        ],
        "all_hashes_match": all(row["matched"] for row in rows),
        "rows": rows,
    }


def read_member(root: Path, archive: str, member: str) -> list[str]:
    with zipfile.ZipFile(root / archive) as handle:
        text = handle.read(member).decode("utf-8", errors="strict")
    return [line.strip() for line in text.splitlines() if line.strip()]


def comma_fields(line: str) -> tuple[str, ...]:
    return tuple(next(csv.reader(io.StringIO(line))))


def load_hayes_roth(root: Path):
    rows = []
    labels = []
    for line in read_member(root, "hayes-roth.zip", "hayes-roth.data"):
        fields = comma_fields(line)
        if len(fields) != 6:
            raise ValueError("Hayes-Roth training width")
        rows.append(fields[1:5])
        labels.append(fields[5])
    for line in read_member(root, "hayes-roth.zip", "hayes-roth.test"):
        fields = comma_fields(line)
        if len(fields) != 5:
            raise ValueError("Hayes-Roth test width")
        rows.append(fields[:4])
        labels.append(fields[4])
    return v34.base.make_task(
        "hayes-roth",
        ("hobby", "age", "education", "marital"),
        rows,
        labels,
    )


def load_spect(root: Path):
    rows = []
    labels = []
    for member in ("SPECT.train", "SPECT.test"):
        for line in read_member(root, "spect-heart.zip", member):
            fields = comma_fields(line)
            if len(fields) != 23:
                raise ValueError("SPECT width")
            labels.append(fields[0])
            rows.append(fields[1:])
    return v34.base.make_task(
        "spect-heart",
        tuple(f"f{index}" for index in range(1, 23)),
        rows,
        labels,
    )


def load_audiology(root: Path):
    rows = []
    labels = []
    for member in (
        "audiology.standardized.data",
        "audiology.standardized.test",
    ):
        for line in read_member(
            root,
            "audiology-standardized.zip",
            member,
        ):
            fields = comma_fields(line)
            if len(fields) != 71:
                raise ValueError("Audiology width")
            # The second-last field is a unique case identifier, not a
            # diagnostic measurement. The final field is the class label.
            rows.append(fields[:69])
            labels.append(fields[70])
    return v34.base.make_task(
        "audiology-standardized",
        tuple(f"feature-{index}" for index in range(1, 70)),
        rows,
        labels,
    )


def load_dermatology(root: Path):
    rows = []
    labels = []
    for line in read_member(
        root,
        "dermatology.zip",
        "dermatology.data",
    ):
        fields = comma_fields(line)
        if len(fields) != 35:
            raise ValueError("Dermatology width")
        rows.append(fields[:34])
        labels.append(fields[34])
    return v34.base.make_task(
        "dermatology",
        tuple(f"feature-{index}" for index in range(1, 35)),
        rows,
        labels,
    )


def load_tasks(root: Path) -> list[object]:
    return [
        load_hayes_roth(root),
        load_spect(root),
        load_audiology(root),
        load_dermatology(root),
    ]


def permute_task(task: object, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(task.candidate_count)
    return v34.base.make_task(
        task.name,
        task.feature_names,
        (task.rows[int(index)] for index in order),
        (task.labels[int(index)] for index in order),
    )


def evaluate_task(task: object) -> dict[str, object]:
    baseline = v38.choose_attainable(v38.constant_rows(task))
    exact = v36.ExactPlanner(task)
    fallbacks = {
        objective: v37.FallbackPlanner(task, objective)
        for objective in v34.OBJECTIVE_NAMES
    }
    row = v38.compare_task(
        task,
        CANDIDATE_THRESHOLD,
        FEATURE_LIMIT,
        exact,
        fallbacks,
        baseline,
    )
    return {
        **row,
        "candidate_count": task.candidate_count,
        "feature_count": task.query_count,
        "label_count": task.label_count,
        "exact_planner_states": len(exact.cache),
        "fallback_planner_states": {
            objective: len(planner.cache)
            for objective, planner in fallbacks.items()
        },
    }


def run_seed(tasks: list[object], seed: int) -> dict[str, object]:
    rows = {
        task.name: evaluate_task(
            permute_task(task, seed + index * 1009)
        )
        for index, task in enumerate(tasks)
    }
    strict_wins = sum(int(row["strict_win"]) for row in rows.values())
    aggregate_saving = float(sum(
        float(row["total_query_saving"])
        for row in rows.values()
    ))
    exact_uses = sum(
        int(row["candidate"]["exact_query_uses"])
        for row in rows.values()
    )
    gate = (
        all(bool(row["lex_no_harm"]) for row in rows.values())
        and all(
            bool(row["coordinate_certificate"])
            for row in rows.values()
        )
        and strict_wins >= 2
        and aggregate_saving >= 5.0
        and exact_uses > 0
    )
    digest_rows = {
        name: {
            "baseline_objective": row["baseline"]["objective"],
            "candidate_objective": row["candidate"]["objective"],
            "diagnosed_gap": row["diagnosed_gap"],
            "worst_query_gap": row["worst_query_gap"],
            "mean_query_gap": row["mean_query_gap"],
            "total_query_saving": row["total_query_saving"],
            "exact_query_uses": row["candidate"]["exact_query_uses"],
        }
        for name, row in rows.items()
    }
    metrics_digest = hashlib.sha256(
        json.dumps(digest_rows, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "seed": seed,
        "candidate_gate": gate,
        "strict_wins": strict_wins,
        "aggregate_total_query_saving": aggregate_saving,
        "exact_query_uses": exact_uses,
        "metrics_digest": metrics_digest,
        "tasks": rows,
    }


def run() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        verification = download_and_verify(root)
        tasks = load_tasks(root)
        seeds = [run_seed(tasks, seed) for seed in SEEDS]
    passing = sum(int(row["candidate_gate"]) for row in seeds)
    digests = {row["metrics_digest"] for row in seeds}
    gate = (
        verification["all_hashes_match"]
        and len(seeds) == 5
        and passing >= 4
        and len(digests) == 1
    )
    compiler_digest = hashlib.sha256(
        json.dumps(
            {
                "development_digest": FROZEN_DEVELOPMENT_DIGEST,
                "candidate_threshold": CANDIDATE_THRESHOLD,
                "feature_limit": FEATURE_LIMIT,
                "seeds": SEEDS,
                "manifest": verification["rows"],
                "gate": {
                    "passing_seeds": 4,
                    "strict_wins_per_seed": 2,
                    "aggregate_saving_per_seed": 5.0,
                    "permutation_invariance": True,
                },
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": (
            "external_attainable_envelope_candidate"
            if gate
            else "not_yet"
        ),
        "claim_scope": (
            "the frozen attainable safe-envelope compiler is evaluated on four "
            "previously untouched UCI classification domains under five row "
            "permutations; a pass is external transfer evidence, but the "
            "components remain known decision-tree heuristics and exact dynamic "
            "programming rather than a world breakthrough"
        ),
        "external_gate": gate,
        "passing_count": passing,
        "run_count": len(seeds),
        "permutation_digest_count": len(digests),
        "archive_verification": verification,
        "frozen_configuration": {
            "candidate_threshold": CANDIDATE_THRESHOLD,
            "feature_limit": FEATURE_LIMIT,
            "development_digest": FROZEN_DEVELOPMENT_DIGEST,
            "seeds": list(SEEDS),
        },
        "frozen_compiler_digest": compiler_digest,
        "seed_reports": seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "external_gate": report["external_gate"],
        "passing_count": report["passing_count"],
        "permutation_digest_count": report[
            "permutation_digest_count"
        ],
        "compiler_digest": report["frozen_compiler_digest"],
    }, indent=2))


if __name__ == "__main__":
    main()
