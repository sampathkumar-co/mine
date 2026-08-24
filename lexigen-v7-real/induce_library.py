from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20260824
MIN_FAMILIES = 2
MIN_LENGTH = 2
MAX_LENGTH = 3


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def load_traces(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    roles = set(payload["allowed_role_vocabulary"])
    traces = payload["traces"]
    if len({t["source_family"] for t in traces}) < 4:
        raise RuntimeError("expected four apprenticeship families")
    for trace in traces:
        if not trace["roles"] or any(role not in roles for role in trace["roles"]):
            raise RuntimeError(f"invalid roles in {trace['trace_id']}")
    return payload


def mine(payload: dict) -> list[dict]:
    support: dict[tuple[str, ...], dict[str, object]] = defaultdict(lambda: {"families": set(), "traces": set(), "occurrences": 0})
    for trace in payload["traces"]:
        seq = list(trace["roles"])
        for length in range(MIN_LENGTH, min(MAX_LENGTH, len(seq)) + 1):
            for i in range(len(seq) - length + 1):
                sub = tuple(seq[i:i + length])
                row = support[sub]
                row["families"].add(trace["source_family"])
                row["traces"].add(trace["trace_id"])
                row["occurrences"] += 1
    candidates = []
    for seq, row in support.items():
        families = sorted(row["families"])
        occurrences = int(row["occurrences"])
        if len(families) < MIN_FAMILIES:
            continue
        # Replacing a length-L subsequence with one macro token saves L-1 tokens per use,
        # while defining the macro costs L tokens once. Add one token back because the
        # macro identifier itself is reusable; positive values are retained.
        savings = occurrences * (len(seq) - 1) - (len(seq) - 1)
        if savings <= 0:
            continue
        candidates.append({
            "sequence": list(seq),
            "source_families": families,
            "source_trace_ids": sorted(row["traces"]),
            "family_count": len(families),
            "occurrences": occurrences,
            "mdl_savings": savings,
        })
    candidates.sort(key=lambda r: (-r["mdl_savings"], -r["family_count"], -len(r["sequence"]), r["sequence"]))
    for i, row in enumerate(candidates, 1):
        row["macro_id"] = f"V7M-{i:03d}"
    return candidates


def random_library(payload: dict, learned: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    roles = list(payload["allowed_role_vocabulary"])
    used = {tuple(row["sequence"]) for row in learned}
    out = []
    for i, row in enumerate(learned, 1):
        length = len(row["sequence"])
        for _ in range(1000):
            seq = tuple(rng.choice(roles) for _ in range(length))
            if seq not in used and seq not in {tuple(x["sequence"]) for x in out}:
                break
        else:
            raise RuntimeError("could not construct random-library control")
        out.append({"macro_id": f"V7R-{i:03d}", "sequence": list(seq)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    payload = load_traces(args.traces)
    learned = mine(payload)
    if len(learned) < 3:
        raise RuntimeError(f"architecture precondition failed: expected >=3 induced macros, got {len(learned)}")
    random_control = random_library(payload, learned)
    report = {
        "campaign": "LEXIGEN V7 Real Mechanism-Genesis Pilot R1",
        "stage": "preholdout_real_library_induction",
        "seed": SEED,
        "apprenticeship_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
        "learned_macros": learned,
        "random_library": random_control,
        "learned_macro_count": len(learned),
        "holdout_identity_used": False,
        "holdout_source_opened": False,
        "holdout_payloads_opened": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "library.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
