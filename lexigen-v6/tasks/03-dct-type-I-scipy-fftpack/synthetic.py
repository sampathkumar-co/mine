from __future__ import annotations

import hashlib
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'lexigen-v5'))
sys.path.insert(0, str(HERE))

from candidates import (
    build_candidates,
    independent_semantic_certificate,
    official_verifier_accepts,
)

SOURCE_SHA256 = 'd9667553f833e9966df0d6fde154c473f7b33285fd67a28f216f9c3df25d4e11'
ARM_ORDER = (
    'v6_full', 'v6_no_transfer', 'random_search',
    'static_template', 'v5_compatible', 'strong_baseline',
)
EXPECTED_BY_ARM = {
    'v6_full': 6,
    'v6_no_transfer': 6,
    'random_search': 6,
    'static_template': 6,
    'v5_compatible': 6,
    'strong_baseline': 1,
}
DIFFERENT_FAMILY_TRANSFER_IDS = {'TM-BFR-01', 'TM-CAC-01', 'TM-RRR-01'}
SAME_FAMILY_TRANSFER_IDS = {'TM-PBEB-01'}


def problems() -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    sizes = (2, 3, 4, 5, 7, 9, 13, 17)
    for mode in range(3):
        for j, n in enumerate(sizes):
            rng = np.random.default_rng(93000 + mode * 100 + j)
            if mode == 0:
                a = rng.random((n, n), dtype=np.float64)
            elif mode == 1:
                scale = 10.0 ** ((j % 7) - 3)
                a = rng.normal(0.0, scale, size=(n, n)).astype(np.float64)
            else:
                x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
                a = np.outer(np.cos((j + 1) * np.pi * x), np.sin((j + 2) * np.pi * x))
                a += 0.125 * np.outer(x, x[::-1])
                a[j % n, (2 * j + 1) % n] += 3.0 + 0.5 * j
            rows.append(np.ascontiguousarray(a, dtype=np.float64))
    if len(rows) != 24:
        raise RuntimeError('synthetic case count changed')
    return rows


def main() -> None:
    source_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('task-source.py')
    raw = source_path.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    if source_sha != SOURCE_SHA256:
        raise SystemExit(f'task source sha256 mismatch {source_sha}')

    arms = build_candidates(raw.decode('utf-8'))
    counts = {arm: len(arms[arm]) for arm in ARM_ORDER}
    if counts != EXPECTED_BY_ARM:
        raise RuntimeError(f'frozen proposal count mismatch: {counts}')

    meta = []
    candidates = []
    for arm in ARM_ORDER:
        for candidate in arms[arm]:
            tids = list(candidate.transfer_ids)
            row = {
                'name': candidate.name,
                'arm': candidate.arm,
                'implementation_class': candidate.implementation_class,
                'operators': list(candidate.operators),
                'transfer_ids': tids,
                'different_family_transfer_ids': [x for x in tids if x in DIFFERENT_FAMILY_TRANSFER_IDS],
                'same_family_transfer_ids': [x for x in tids if x in SAME_FAMILY_TRANSFER_IDS],
                'learned_template': candidate.learned_template,
                'baseline_id': candidate.baseline_id,
            }
            meta.append(row)
            candidates.append(candidate)

    if len(candidates) != 31 or len({c.name for c in candidates}) != 31:
        raise RuntimeError('expected 31 unique frozen Task3 candidates')

    cases = problems()
    evidence = []
    for candidate in candidates:
        for case_index, problem in enumerate(cases):
            start = time.perf_counter_ns()
            error = None
            semantic_valid = False
            official_valid = False
            relative_scale = float(np.max(np.abs(problem)))
            try:
                solution = candidate.solve(problem)
                semantic_valid = independent_semantic_certificate(problem, solution)
                official_valid = official_verifier_accepts(problem, solution)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            elapsed = time.perf_counter_ns() - start
            evidence.append({
                'candidate': candidate.name,
                'arm': candidate.arm,
                'implementation_class': candidate.implementation_class,
                'case': case_index,
                'shape': list(problem.shape),
                'input_abs_max': relative_scale,
                'semantic_valid': bool(semantic_valid),
                'official_valid': bool(official_valid),
                'elapsed_ns': elapsed,
                'error': error,
            })

    by_candidate = {}
    for m in meta:
        sub = [r for r in evidence if r['candidate'] == m['name']]
        eligible = all(
            bool(r['semantic_valid']) and bool(r['official_valid']) and r['error'] is None
            for r in sub
        )
        by_candidate[m['name']] = {
            **m,
            'cases': len(sub),
            'semantic_valid': sum(bool(r['semantic_valid']) for r in sub),
            'official_valid': sum(bool(r['official_valid']) for r in sub),
            'errors': sum(r['error'] is not None for r in sub),
            'median_elapsed_ns_diagnostic_only': statistics.median(r['elapsed_ns'] for r in sub),
            'eligible': eligible,
        }

    arm_eligible = {
        arm: sum(
            1 for m in meta
            if m['arm'] == arm and by_candidate[m['name']]['eligible']
        )
        for arm in ARM_ORDER
    }
    eligible_names = sorted(name for name, row in by_candidate.items() if row['eligible'])

    evidence_payload = '\n'.join(json.dumps(r, separators=(',', ':')) for r in evidence) + '\n'
    plan_payload = json.dumps(meta, sort_keys=True, separators=(',', ':'))
    summary = {
        'campaign': 'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication',
        'task_index': 3,
        'task': 'dct_type_I_scipy_fftpack',
        'family': 'signal_processing',
        'stage': 'synthetic_r1',
        'case_count': len(cases),
        'candidate_count': len(candidates),
        'row_count': len(evidence),
        'eligible_count': len(eligible_names),
        'eligible_names': eligible_names,
        'arm_candidate_count': counts,
        'arm_eligible_count': arm_eligible,
        'by_candidate': by_candidate,
        'candidate_plan_sha256': hashlib.sha256(plan_payload.encode()).hexdigest(),
        'results_sha256': hashlib.sha256(evidence_payload.encode()).hexdigest(),
        'source_sha256': source_sha,
        'different_family_transfer_ids': sorted(DIFFERENT_FAMILY_TRANSFER_IDS),
        'same_family_transfer_ids_in_engine_but_ineligible_for_causal_credit': sorted(SAME_FAMILY_TRANSFER_IDS),
        'eligibility_uses_timing': False,
        'official_verifier_tolerance': 1e-6,
        'versions': {
            'python': sys.version.split()[0],
            'numpy': importlib.metadata.version('numpy'),
            'scipy': importlib.metadata.version('scipy'),
        },
        'official_train_manifest_opened': False,
        'official_test_manifest_opened': False,
        'official_payloads_opened': 0,
        'public_task_specific_solvers_opened': False,
        'threshold_changes': False,
    }

    Path('synthetic-results.jsonl').write_text(evidence_payload)
    Path('synthetic-candidate-plan.json').write_text(json.dumps(meta, indent=2) + '\n')
    Path('synthetic-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({
        'candidate_count': len(candidates),
        'eligible_count': len(eligible_names),
        'arm_eligible_count': arm_eligible,
        'candidate_plan_sha256': summary['candidate_plan_sha256'],
        'results_sha256': summary['results_sha256'],
        'versions': summary['versions'],
    }, indent=2))

    if any(arm_eligible[arm] < 1 for arm in ARM_ORDER):
        raise SystemExit(4)


if __name__ == '__main__':
    main()
