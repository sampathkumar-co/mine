from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

INTERVENTIONS = {
    "cache": {"alloc": 1.0, "dispatch": 0.2, "memory": 0.1, "loop": 0.0},
    "specialize": {"alloc": 0.1, "dispatch": 1.0, "memory": 0.0, "loop": 0.2},
    "vectorize": {"alloc": 0.0, "dispatch": 0.1, "memory": 0.3, "loop": 1.0},
    "representation": {"alloc": 0.3, "dispatch": 0.0, "memory": 1.0, "loop": 0.2},
}


def noise(*parts: str) -> float:
    h = hashlib.sha256("|".join(parts).encode()).digest()
    x = int.from_bytes(h[:8], "big") / 2**64
    return (x - 0.5) * 0.10


def value(repo: int, sample: int, key: str) -> float:
    h = hashlib.sha256(f"repo={repo}|sample={sample}|{key}".encode()).digest()
    return 0.1 + 0.9 * (int.from_bytes(h[:8], "big") / 2**64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repos", type=int, default=12)
    ap.add_argument("--samples-per-repo", type=int, default=18)
    args = ap.parse_args()
    rows = []
    for repo in range(args.repos):
        family = f"family-{repo % 4}"
        language = "python" if repo % 2 == 0 else "rust"
        env = "cpu-a" if repo % 3 else "cpu-b"
        for sample in range(args.samples_per_repo):
            prog = {
                "alloc": value(repo, sample, "alloc"),
                "dispatch": value(repo, sample, "dispatch"),
                "memory": value(repo, sample, "memory"),
                "loop": value(repo, sample, "loop"),
            }
            for fam, intv in INTERVENTIONS.items():
                match = sum(prog[k] * intv[k] for k in prog)
                mismatch = sum((1.0 - prog[k]) * intv[k] for k in prog)
                structural = 0.42 * match - 0.20 * mismatch
                language_term = 0.04 if (language == "rust" and fam == "vectorize") else 0.0
                env_term = -0.03 if (env == "cpu-b" and fam == "cache") else 0.0
                log_speedup = structural + language_term + env_term - 0.22 + noise(str(repo), str(sample), fam, "speed")
                valid_margin = 0.55 * match - 0.25 * mismatch + noise(str(repo), str(sample), fam, "valid")
                valid = valid_margin > 0.05
                if not valid:
                    log_speedup = min(log_speedup, -0.08 - abs(noise(str(repo), str(sample), fam, "bad")))
                before = 0.8 + 0.4 * sum(prog.values())
                after = before / math.exp(log_speedup)
                rows.append({
                    "record_id": f"r{repo:02d}-s{sample:02d}-{fam}",
                    "repository_id": f"repo-{repo:02d}",
                    "repository_family": family,
                    "language": language,
                    "environment_id": env,
                    "program_features": prog,
                    "intervention_id": f"{fam}-r1",
                    "intervention_family": fam,
                    "intervention_features": intv,
                    "valid": valid,
                    "runtime_before": before,
                    "runtime_after": after,
                    "measurement_repetitions": 7,
                    "provenance": {
                        "kind": "deterministic_synthetic_pipeline_sanity_only",
                        "scientific_transfer_evidence": False,
                    },
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"records": len(rows), "repositories": args.repos, "scientific_transfer_evidence": False}, indent=2))


if __name__ == "__main__":
    main()
