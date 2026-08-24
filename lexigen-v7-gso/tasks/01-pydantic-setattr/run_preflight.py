from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

IMAGE = 'slimshetty/gso:gso.eval.x86_64.pydantic__pydantic-addf1f9'
CANDIDATES = ['F1', 'F2', 'F3', 'N1', 'N2', 'N3', 'R1', 'R2', 'R3']
TEST_INDEXES = [0, 4, 8, 13]
TIME_RE = re.compile(r'Execution time:\s*([0-9.]+)s')


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def dexec(name: str, script: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return run(['docker', 'exec', name, 'bash', '-lc', script], check=check, capture=capture)


def make_container(name: str) -> None:
    run(['docker', 'rm', '-f', name], check=False, capture=True)
    run(['docker', 'create', '--name', name, IMAGE, 'tail', '-f', '/dev/null'])
    run(['docker', 'start', name])


def inject_tests(name: str, tests_dir: Path) -> None:
    for index in TEST_INDEXES:
        src = (tests_dir / f'gso_test_{index}.py').resolve()
        if not src.is_file():
            raise RuntimeError(f'missing frozen test {src}')
        run(['docker', 'cp', str(src), f'{name}:/gso_test_{index}.py'])


def timing_from_container(name: str, path: str) -> float:
    text = dexec(name, f'cat {path}', capture=True).stdout
    match = TIME_RE.search(text)
    if not match:
        raise RuntimeError(f'no timing in {path}: {text!r}')
    return float(match.group(1))


def imported_file(name: str) -> str:
    return dexec(
        name,
        "cd /testbed && source .venv/bin/activate && "
        "python -c 'import pydantic; print(pydantic.__file__)'",
        capture=True,
    ).stdout.strip()


def imported_root(name: str) -> str:
    return dexec(
        name,
        "cd /testbed && source .venv/bin/activate && "
        "python -c 'import pathlib,pydantic; print(pathlib.Path(pydantic.__file__).resolve().parent.parent)'",
        capture=True,
    ).stdout.strip()


def apply_frozen_candidate(name: str, candidate: str, import_root: str) -> None:
    dexec(name, f'python /tmp/build_candidate.py --root /testbed --candidate {candidate}')
    if import_root.rstrip('/') != '/testbed':
        dexec(name, f"python /tmp/build_candidate.py --root '{import_root}' --candidate {candidate}")
    dexec(
        name,
        "python -m py_compile /testbed/pydantic/main.py "
        "/testbed/pydantic/_internal/_model_construction.py",
    )
    if import_root.rstrip('/') != '/testbed':
        dexec(
            name,
            f"python -m py_compile '{import_root}/pydantic/main.py' "
            f"'{import_root}/pydantic/_internal/_model_construction.py'",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--builder', type=Path, required=True)
    parser.add_argument('--tests-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    refs = out / 'reference'
    refs.mkdir(exist_ok=True)

    run(['docker', 'pull', IMAGE])

    base_name = 'lexigen-v7-pydantic-base'
    make_container(base_name)
    inject_tests(base_name, args.tests_dir)
    imported = imported_file(base_name)
    import_root = imported_root(base_name)
    base_times: dict[str, float] = {}
    for index in TEST_INDEXES:
        prefix = f'/tmp/v7_ref_{index}'
        dexec(
            base_name,
            f"cd /testbed && source .venv/bin/activate && python /gso_test_{index}.py "
            f"/tmp/base_{index}.txt --reference --file_prefix {prefix}",
        )
        base_times[str(index)] = timing_from_container(base_name, f'/tmp/base_{index}.txt')
        run(['docker', 'cp', f'{base_name}:{prefix}_result.json', str(refs / f'{index}_result.json')])
    run(['docker', 'rm', '-f', base_name], check=False)

    rows = []
    for candidate in CANDIDATES:
        name = f'lexigen-v7-pydantic-{candidate.lower()}'
        candidate_dir = out / candidate
        candidate_dir.mkdir(exist_ok=True)
        make_container(name)
        inject_tests(name, args.tests_dir)
        run(['docker', 'cp', str(args.builder.resolve()), f'{name}:/tmp/build_candidate.py'])
        for index in TEST_INDEXES:
            run(['docker', 'cp', str(refs / f'{index}_result.json'), f'{name}:/tmp/v7_ref_{index}_result.json'])

        patch_ok = True
        error = None
        candidate_times: dict[str, float] = {}
        try:
            candidate_import_root = imported_root(name)
            apply_frozen_candidate(name, candidate, candidate_import_root)
            imported_candidate = imported_file(name)
            for index in TEST_INDEXES:
                dexec(
                    name,
                    f"cd /testbed && source .venv/bin/activate && python /gso_test_{index}.py "
                    f"/tmp/candidate_{index}.txt --eqcheck --file_prefix /tmp/v7_ref_{index}",
                )
                candidate_times[str(index)] = timing_from_container(name, f'/tmp/candidate_{index}.txt')
            patch = dexec(
                name,
                'cd /testbed && git diff -- pydantic/main.py pydantic/_internal/_model_construction.py',
                capture=True,
            ).stdout
            if not patch.strip():
                raise RuntimeError('candidate produced empty repository patch')
            (candidate_dir / 'candidate.patch').write_text(patch, encoding='utf-8')
        except Exception as exc:
            patch_ok = False
            imported_candidate = None
            candidate_import_root = None
            error = repr(exc)

        ratios = {
            key: (base_times[key] / candidate_times[key])
            for key in candidate_times
            if candidate_times[key] > 0
        }
        harmonic = 0.0
        if len(ratios) == len(TEST_INDEXES) and all(value > 0 for value in ratios.values()):
            harmonic = len(ratios) / sum(1.0 / value for value in ratios.values())
        row = {
            'candidate_id': candidate,
            'arm': 'v7_full' if candidate.startswith('F') else ('v7_no_library' if candidate.startswith('N') else 'v7_random_library'),
            'all_preflight_equivalence_checks_passed': patch_ok and len(candidate_times) == len(TEST_INDEXES),
            'base_times_seconds': base_times,
            'candidate_times_seconds': candidate_times,
            'speedup_by_test': ratios,
            'harmonic_speedup': harmonic,
            'minimum_speedup': min(ratios.values()) if ratios else 0.0,
            'base_import_path': imported,
            'base_import_root': import_root,
            'candidate_import_path': imported_candidate,
            'candidate_import_root': candidate_import_root,
            'error': error,
            'expert_opt_commit_accessed': False,
            'expert_diff_accessed': False,
            'hints_accessed': False,
            'git_history_inspected': False,
        }
        (candidate_dir / 'result.json').write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
        rows.append(row)
        run(['docker', 'rm', '-f', name], check=False)

    summary = {
        'stage': 'task1_equal_budget_preflight_r3',
        'image': IMAGE,
        'test_indexes': TEST_INDEXES,
        'frozen_tests_injected_from_pinned_dataset': True,
        'base_times_seconds': base_times,
        'base_import_path': imported,
        'base_import_root': import_root,
        'candidates': rows,
        'official_gso_evaluator_used': False,
        'expert_target_timing_accessed': False,
        'expert_opt_commit_accessed': False,
        'expert_diff_accessed': False,
        'hints_accessed': False,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
