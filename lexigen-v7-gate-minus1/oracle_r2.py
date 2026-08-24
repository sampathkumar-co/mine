from __future__ import annotations

import argparse
import hashlib
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
    raise ValueError(name)


def execute(program: tuple[str, ...], xs: tuple[int, ...]) -> tuple[int, ...]:
    value = xs
    for atom in program:
        value = apply_atom(atom, value)
    return value


def signature(program: tuple[str, ...], inputs: list[list[int]]) -> str:
    outputs = [list(execute(program, tuple(xs))) for xs in inputs]
    payload = json.dumps(outputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build() -> dict:
    search_rng = random.Random(2026082401)
    search_inputs = [
        [], [0], [1, -1, 1, -1], [3, 2, 1, 0, -1],
        [-3, -3, 0, 2, 2], [2, -1, 0, 2, -1],
    ]
    for _ in range(18):
        n = search_rng.randint(2, 6)
        search_inputs.append([search_rng.randint(-3, 3) for _ in range(n)])

    validation_rng = random.Random(2026082402)
    validation_inputs = [
        [4, -4, 0, 4], [-2, 1, -2, 3, 0], [0, 0, 0], [5, -1, 2, -3, 4],
    ]
    for _ in range(32):
        n = validation_rng.randint(1, 7)
        validation_inputs.append([validation_rng.randint(-5, 5) for _ in range(n)])

    targets = [
        ("combinatorial", ("CLIP_POS", "SORT", "UNIQUE", "REVERSE")),
        ("cryptography", ("DIFF", "ABS", "REVERSE", "CUMSUM")),
        ("linear_algebra", ("REVERSE", "CUMSUM", "SORT", "UNIQUE")),
    ]
    return {
        "oracle_revision": 2,
        "search_inputs": search_inputs,
        "validation_inputs": validation_inputs,
        "holdouts": [
            {
                "family": family,
                "search_signature_sha256": signature(program, search_inputs),
                "validation_signature_sha256": signature(program, validation_inputs),
            }
            for family, program in targets
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build()
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
