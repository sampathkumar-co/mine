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
from . import local_quotient_v40 as v40
from . import safe_portfolio_v37 as v37
from . import state_policy_v34 as v34


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "external-data" / "uci-v41" / "manifest.json"
SEEDS = (2301, 2302, 2303, 2304, 2305)
CANDIDATE_THRESHOLD = 12
FROZEN_PARENT_DIGEST = (
    "3f6fe829f99a27c9c2a4e26756715f8949db87a009169f917544b8f26c19d538"
)
LARGE_DOMAINS = {"chess-kr-vs-kp", "splice-junction"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_and_verify(root: Path) -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["frozen_parent_compiler_digest"] != FROZEN_PARENT_DIGEST:
        raise RuntimeError("parent compiler commitment mismatch")
    rows = []
    for entry in manifest["rows"]:
        request = urllib.request.Request(
            entry["url"],
            headers={"User-Agent": "Mini-ORIGIN-v0.41/1.0"},
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
    return tuple(value.strip() for value in next(csv.reader(io.StringIO(line))))


def load_chess(root: Path):
    rows = []
    labels = []
    for line in read_member(
        root,
        "chess-kr-vs-kp.zip",
        "kr-vs-kp.data",
    ):
        fields = comma_fields(line)
        if len(fields) != 37:
            raise ValueError("Chess row width")
        rows.append(fields[:-1])
        labels.append(fields[-1])
    return v34.base.make_task(
        "chess-kr-vs-kp",
        tuple(f"feature-{index}" for index in range(1, 37)),
        rows,
        labels,
    )


def load_splice(root: Path):
    rows = []
    labels = []
    for line in read_member(root, "splice.zip", "splice.data"):
        fields = comma_fields(line)
        if len(fields) != 3:
            raise ValueError("Splice row width")
        label, _identifier, sequence = fields
        sequence = sequence.replace(" ", "").upper()
        if len(sequence) != 60:
            raise ValueError("Splice sequence width")
        rows.append(tuple(sequence))
        labels.append(label.upper())
    return v34.base.make_task(
        "splice-junction",
        tuple(f"base-{index}" for index in range(-30, 30)),
        rows,
        labels,
    )


def load_hepatitis(root: Path):
    rows = []
    labels = []
    for line in read_member(root, "hepatitis.zip", "hepatitis.data"):
        fields = comma_fields(line)
        if len(fields) != 20:
            raise ValueError("Hepatitis row width")
        labels.append(fields[0])
        # Continuous and missing values remain explicit observed categories;
        # no data-dependent discretisation is introduced after commitment.
        rows.append(fields[1:])
    return v34.base.make_task(
        "hepatitis",
        tuple(f"feature-{index}" for index in range(1, 20)),
        rows,
        labels,
    )


def load_teaching_assistant(root: Path):
    rows = []
    labels = []
    for line in read_member(
        root,
        "teaching-assistant.zip",
        "tae.data",
    ):
        fields = comma_fields(line)
        if len(fields) != 6:
            raise ValueError("Teaching Assistant row width")
        rows.append(fields[:5])
        labels.append(fields[5])
    return v34.base.make_task(
        "teaching-assistant",
        (
            "native-language",
            "instructor",
            "course",
            "semester",
            "class-size",
        ),
        rows,
        labels,
    )


def load_tasks(root: Path) -> list[object]:
    return [
        load_chess(root),
        load_splice(root),
        load_hepatitis(root),
        load_teaching_assistant(root),
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
    exact = v40.LocalQuotientPlanner(task)
    candidates = []
    diagnostics = {}
    for objective in v34.OBJECTIVE_NAMES:
        fallback = v37.FallbackPlanner(task, objective)
        result = v40.evaluate(
            task,
            v40.QuotientPolicy(CANDIDATE_THRESHOLD, objective),
            exact,
            fallback,
        )
        diagnostics[objective] = result
        candidates.append(v38.AttainableRow(
            objective=objective,
            diagnosed_fraction=result.diagnosed_fraction,
            mean_queries=result.mean_queries,
            worst_queries=result.worst_queries,
            candidates=result.candidates,
            exact_query_uses=result.exact_query_uses,
        ))
    candidate = v38.choose_attainable(tuple(candidates))
    selected = diagnostics[candidate.objective]
    diagnosed_gap = (
        candidate.diagnosed_fraction - baseline.diagnosed_fraction
    )
    worst_gap = candidate.worst_queries - baseline.worst_queries
    mean_gap = candidate.mean_queries - baseline.mean_queries
    metric_no_harm = candidate.metric() >= baseline.metric()
    coordinate_certificate = (
        (abs(diagnosed_gap) > 1e-12 or worst_gap <= 0)
        and (
            not (abs(diagnosed_gap) <= 1e-12 and worst_gap == 0)
            or mean_gap <= 1e-12
        )
    )
    return {
        "baseline": baseline.__dict__,
        "candidate": candidate.__dict__,
        "metric_no_harm": metric_no_harm,
        "coordinate_certificate": coordinate_certificate,
        "strict_win": candidate.metric() > baseline.metric(),
        "diagnosed_gap": diagnosed_gap,
        "worst_query_gap": worst_gap,
        "mean_query_gap": mean_gap,
        "total_query_saving": (
            baseline.mean_queries - candidate.mean_queries
        ) * task.candidate_count,
        "selected_diagnostics": selected.__dict__,
        "candidate_count": task.candidate_count,
        "feature_count": task.query_count,
        "label_count": task.label_count,
    }


def run_seed(tasks: list[object], seed: int) -> dict[str, object]:
    rows = {
        task.name: evaluate_task(
            permute_task(task, seed + index * 1009)
        )
        for index, task in enumerate(tasks)
    }
    strict_wins = sum(int(row["strict_win"]) for row in rows.values())
    large_wins = sum(
        int(row["strict_win"])
        for name, row in rows.items()
        if name in LARGE_DOMAINS
    )
    aggregate_saving = float(sum(
        float(row["total_query_saving"])
        for row in rows.values()
    ))
    exact_uses = sum(
        int(row["candidate"]["exact_query_uses"])
        for row in rows.values()
    )
    raw_queries = sum(
        int(row["selected_diagnostics"]["raw_queries_seen"])
        for row in rows.values()
    )
    quotient_queries = sum(
        int(row["selected_diagnostics"]["quotient_queries_seen"])
        for row in rows.values()
    )
    gate = (
        all(bool(row["metric_no_harm"]) for row in rows.values())
        and all(
            bool(row["coordinate_certificate"])
            for row in rows.values()
        )
        and strict_wins >= 2
        and large_wins >= 1
        and aggregate_saving >= 10.0
        and exact_uses > 0
        and quotient_queries < raw_queries
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
    return {
        "seed": seed,
        "candidate_gate": gate,
        "strict_wins": strict_wins,
        "large_domain_strict_wins": large_wins,
        "aggregate_total_query_saving": aggregate_saving,
        "exact_query_uses": exact_uses,
        "raw_queries_seen": raw_queries,
        "quotient_queries_seen": quotient_queries,
        "metrics_digest": hashlib.sha256(
            json.dumps(digest_rows, sort_keys=True).encode("utf-8")
        ).hexdigest(),
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
                "parent_digest": FROZEN_PARENT_DIGEST,
                "candidate_threshold": CANDIDATE_THRESHOLD,
                "seeds": SEEDS,
                "manifest": verification["rows"],
                "gate": {
                    "passing_seeds": 4,
                    "strict_wins_per_seed": 2,
                    "large_domain_wins_per_seed": 1,
                    "aggregate_saving_per_seed": 10.0,
                    "permutation_invariance": True,
                },
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": (
            "external_local_quotient_candidate"
            if gate
            else "not_yet"
        ),
        "claim_scope": (
            "the frozen state-local partition quotient and exact-subtree "
            "compiler is evaluated on four previously untouched UCI domains "
            "under five row permutations; a pass is broad external transfer "
            "evidence, but its components remain exact decision-tree search and "
            "known impurity heuristics rather than a world breakthrough"
        ),
        "external_gate": gate,
        "passing_count": passing,
        "run_count": len(seeds),
        "permutation_digest_count": len(digests),
        "archive_verification": verification,
        "frozen_configuration": {
            "candidate_threshold": CANDIDATE_THRESHOLD,
            "parent_compiler_digest": FROZEN_PARENT_DIGEST,
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
