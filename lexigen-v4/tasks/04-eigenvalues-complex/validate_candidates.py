from __future__ import annotations

import json
import math
from pathlib import Path
import mpmath as mp
import numpy as np
from candidates import ALL_CANDIDATES

mp.mp.dps = 80


def sortvals(values):
    return sorted((complex(v) for v in values), key=lambda z: (-z.real, -z.imag))


def oracle(a: np.ndarray):
    m = mp.matrix([[mp.mpf(str(float(x))) for x in row] for row in a])
    vals, _ = mp.eig(m)
    return sortvals([complex(float(mp.re(v)), float(mp.im(v))) for v in vals])


def max_rel(candidate, expected):
    if len(candidate) != len(expected):
        return math.inf
    c = sortvals(candidate)
    return max((abs(x - y) / max(abs(y), 1e-12) for x, y in zip(c, expected)), default=0.0)


def cases():
    out = [
        ("single", np.array([[3.25]], dtype=float)),
        ("rotation", np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)),
        ("triangular", np.array([[4.0, 2.0, 1.0], [0.0, -1.0, 3.0], [0.0, 0.0, 0.25]], dtype=float)),
        ("symmetric", np.array([[2.0, -1.0, 0.0, 0.0], [-1.0, 2.0, -1.0, 0.0], [0.0, -1.0, 2.0, -1.0], [0.0, 0.0, -1.0, 2.0]], dtype=float)),
        ("scaled", np.array([[1e4, 2.0, 0.0], [-3.0, -2e-3, 4.0], [0.0, -1.0, 7.0]], dtype=float)),
        ("tiny_eigen", np.diag([5.0, 2.0, 1e-7, -3.0]) + np.array([[0, 0.1, 0, 0], [0, 0, 0.2, 0], [0, 0, 0, 0.1], [0, 0, 0, 0]], dtype=float)),
        ("near_defective", np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0 + 1e-7, 1.0, 0.0], [0.0, 0.0, 1.0 + 2e-7, 1.0], [1e-8, 0.0, 0.0, 1.0 + 3e-7]], dtype=float)),
    ]
    for seed, n in [(11, 4), (29, 5), (47, 6)]:
        out.append((f"random_{n}_{seed}", np.random.default_rng(seed).normal(size=(n, n))))
    return out


def main():
    all_cases = cases()
    expected = {name: oracle(a) for name, a in all_cases}
    results = []
    all_ok = True
    for arm, name, fn in ALL_CANDIDATES:
        passed = 0
        worst = 0.0
        failures = []
        for case, a in all_cases:
            try:
                sol = fn(a.copy())
                err = max_rel(sol, expected[case])
                sorted_ok = sol == sortvals(sol)
                finite = all(np.isfinite(z.real) and np.isfinite(z.imag) for z in sol)
                ok = len(sol) == a.shape[0] and sorted_ok and finite and err <= 1e-6
            except Exception as exc:
                ok = False
                err = math.inf
                failures.append({"case": case, "exception": repr(exc)})
            if ok:
                passed += 1
            else:
                if not failures or failures[-1].get("case") != case:
                    failures.append({"case": case, "max_relative_error": err})
            worst = max(worst, err)
        all_ok &= passed == len(all_cases)
        results.append({"arm": arm, "candidate": name, "passed": passed, "total": len(all_cases), "worst_relative_error": worst, "failures": failures})

    summary = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 4,
        "task": "eigenvalues_complex",
        "revision": 1,
        "status": "passed" if all_ok else "failed",
        "oracle": "mpmath.eig at 80 decimal digits",
        "candidate_count": len(ALL_CANDIDATES),
        "case_count": len(all_cases),
        "checks": len(ALL_CANDIDATES) * len(all_cases),
        "official_training_opened": False,
        "official_test_opened": False,
        "results": results,
    }
    out = Path("synthetic-evidence")
    out.mkdir(exist_ok=True)
    (out / "synthetic-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
