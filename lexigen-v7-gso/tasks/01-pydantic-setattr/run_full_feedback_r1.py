from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

IMAGE = 'slimshetty/gso:gso.eval.x86_64.pydantic__pydantic-addf1f9'
CANDIDATES = ['F3', 'N1', 'R3']
TEST_INDEXES = list(range(14))
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


def timing(name: str, path: str) -> float:
    text = dexec(name, f'cat {path}', capture=True).stdout
    match = TIME_RE.search(text)
    if not match:
        raise RuntimeError(f'no timing in {path}: {text!r}')
    return float(match.group(1))


def discover_reference_basename(name: str, prefix: str) -> str:
    prefix_name = Path(prefix).name
    script = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        f"matches=sorted(p.name for p in Path('/tmp').glob('{prefix_name}*') if p.is_file())\n"
        "print('\\n'.join(matches))\n"
        "PY"
    )
    matches = [line.strip() for line in dexec(name, script, capture=True).stdout.splitlines() if line.strip()]
    if len(matches) != 1:
        raise RuntimeError(f'expected exactly one reference file for {prefix_name}, got {matches}')
    return matches[0]


def import_root(name: str) -> str:
    return dexec(
        name,
        "cd /testbed && source .venv/bin/activate && "
        "python -c 'import pathlib,pydantic; print(pathlib.Path(pydantic.__file__).resolve().parent.parent)'",
        capture=True,
    ).stdout.strip()


def apply_candidate(name: str, candidate: str, builder: Path) -> str:
    run(['docker', 'cp', str(builder.resolve()), f'{name}:/tmp/build_candidate.py'])
    root = import_root(name)
    dexec(name, f'python /tmp/build_candidate.py --root /testbed --candidate {candidate}')
    if root.rstrip('/') != '/testbed':
        dexec(name, f"python /tmp/build_candidate.py --root '{root}' --candidate {candidate}")
    dexec(name, 'python -m py_compile /testbed/pydantic/main.py /testbed/pydantic/_internal/_model_construction.py')
    if root.rstrip('/') != '/testbed':
        dexec(name, f"python -m py_compile '{root}/pydantic/main.py' '{root}/pydantic/_internal/_model_construction.py'")
    return root


def harmonic(values: list[float]) -> float:
    if not values or any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--builder', type=Path, required=True)
    ap.add_argument('--tests-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    refs = out / 'reference'
    refs.mkdir(exist_ok=True)

    run(['docker', 'pull', IMAGE])
    base_name = 'lexigen-v7-pydantic-full-base'
    make_container(base_name)
    inject_tests(base_name, args.tests_dir)
    base_times: dict[str, float] = {}
    reference_basenames: dict[str, str] = {}
    for index in TEST_INDEXES:
        prefix = f'/tmp/v7_full_ref_{index}'
        dexec(base_name, f"cd /testbed && source .venv/bin/activate && python /gso_test_{index}.py /tmp/base_{index}.txt --reference --file_prefix {prefix}")
        base_times[str(index)] = timing(base_name, f'/tmp/base_{index}.txt')
        basename = discover_reference_basename(base_name, prefix)
        reference_basenames[str(index)] = basename
        run(['docker', 'cp', f'{base_name}:/tmp/{basename}', str(refs / basename)])
    run(['docker', 'rm', '-f', base_name], check=False)

    rows = []
    for candidate in CANDIDATES:
        name = f'lexigen-v7-pydantic-full-{candidate.lower()}'
        candidate_dir = out / candidate
        candidate_dir.mkdir(exist_ok=True)
        make_container(name)
        inject_tests(name, args.tests_dir)
        for index in TEST_INDEXES:
            basename = reference_basenames[str(index)]
            run(['docker', 'cp', str(refs / basename), f'{name}:/tmp/{basename}'])
        candidate_times: dict[str, float] = {}
        passed = True
        error = None
        try:
            root = apply_candidate(name, candidate, args.builder)
            for index in TEST_INDEXES:
                dexec(name, f"cd /testbed && source .venv/bin/activate && python /gso_test_{index}.py /tmp/candidate_{index}.txt --eqcheck --file_prefix /tmp/v7_full_ref_{index}")
                candidate_times[str(index)] = timing(name, f'/tmp/candidate_{index}.txt')
            patch = dexec(name, 'cd /testbed && git diff -- pydantic/main.py pydantic/_internal/_model_construction.py', capture=True).stdout
            if not patch.strip():
                raise RuntimeError('empty candidate patch')
            (candidate_dir / 'candidate.patch').write_text(patch, encoding='utf-8')
        except Exception as exc:
            passed = False
            root = None
            error = repr(exc)
        speedups = {k: base_times[k] / v for k, v in candidate_times.items() if v > 0}
        row = {
            'candidate_id': candidate,
            'arm': 'v7_full' if candidate == 'F3' else ('v7_no_library' if candidate == 'N1' else 'v7_random_library'),
            'all_14_equivalence_checks_passed': passed and len(candidate_times) == 14,
            'candidate_times_seconds': candidate_times,
            'speedup_by_test': speedups,
            'harmonic_speedup': harmonic(list(speedups.values())),
            'minimum_speedup': min(speedups.values()) if speedups else 0.0,
            'maximum_speedup': max(speedups.values()) if speedups else 0.0,
            'import_root': root,
            'error': error,
        }
        (candidate_dir / 'result.json').write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
        rows.append(row)
        run(['docker', 'rm', '-f', name], check=False)

    report = {
        'stage': 'task1_full_14_test_feedback_round0_r2',
        'image': IMAGE,
        'test_indexes': TEST_INDEXES,
        'base_times_seconds': base_times,
        'reference_basenames': reference_basenames,
        'candidates': rows,
        'revision_slots_consumed_per_arm': 0,
        'official_gso_evaluator_used': False,
        'expert_target_timing_accessed': False,
        'expert_opt_commit_accessed': False,
        'expert_diff_accessed': False,
        'hints_accessed': False,
    }
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
