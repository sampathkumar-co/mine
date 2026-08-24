from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FORBIDDEN_PATH_PARTS = (
    '/.git/', '/.github/', '/tests/', '/test/', '/benchmarks/', '/benchmark/',
    '/.venv/', '/venv/', '/__pycache__/', '/reports/', '/artifacts/',
)
FORBIDDEN_BASENAMES = {
    'eval.sh', 'run_evaluation.py', 'opt_at_k.py', 'test_output.txt', 'report.json'
}
FORBIDDEN_ADDED_PATTERNS = {
    'gso_test': re.compile(r'\bgso_test_', re.I),
    'eqcheck': re.compile(r'\beqcheck\b|--eqcheck', re.I),
    'file_prefix': re.compile(r'\bfile_prefix\b', re.I),
    'reference_flag': re.compile(r'--reference|\breference[_ -]?result\b', re.I),
    'execution_time_text': re.compile(r'Execution time:', re.I),
    'timer_manipulation': re.compile(r'\btime\.(?:perf_counter|process_time|monotonic)\b|\btimeit\b', re.I),
    'environment_access': re.compile(r'\bos\.environ\b|\bgetenv\s*\(', re.I),
    'subprocess_shell': re.compile(r'\bsubprocess\b|\bos\.system\b|\bPopen\s*\(', re.I),
    'network_access': re.compile(r'\brequests\.(?:get|post|put|delete)\b|\burllib\b|\bhttpx\b|\bsocket\b', re.I),
    'git_history': re.compile(r'\bgit\s+(?:show|log|diff|cat-file|rev-parse)\b', re.I),
}


def parse_patch(text: str):
    changed = []
    added = []
    current = None
    for line in text.splitlines():
        if line.startswith('diff --git a/'):
            m = re.match(r'diff --git a/(.+?) b/(.+)$', line)
            if not m:
                raise ValueError(f'bad diff header: {line}')
            a, b = m.groups()
            if a != b:
                raise ValueError(f'rename/path mismatch forbidden: {a} -> {b}')
            current = b
            changed.append(b)
        elif line.startswith('+++ ') or line.startswith('--- '):
            continue
        elif line.startswith('+') and not line.startswith('+++'):
            added.append((current, line[1:]))
    return changed, added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--patch', type=Path, required=True)
    ap.add_argument('--allow-path', action='append', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    text = args.patch.read_text(encoding='utf-8')
    changed, added = parse_patch(text)
    allow = set(args.allow_path)
    violations = []

    if not changed:
        violations.append({'rule': 'nonempty_patch', 'detail': 'patch changes no files'})

    for path in changed:
        norm = '/' + path.replace('\\', '/') + '/'
        if path not in allow:
            violations.append({'rule': 'source_path_allowlist', 'path': path, 'detail': 'path not in frozen allowlist'})
        if any(part in norm for part in FORBIDDEN_PATH_PARTS) or Path(path).name in FORBIDDEN_BASENAMES:
            violations.append({'rule': 'forbidden_path', 'path': path})

    for path, line in added:
        stripped = line.strip()
        if not stripped or stripped.startswith(('#', '//', '/*', '*')):
            continue
        for label, pat in FORBIDDEN_ADDED_PATTERNS.items():
            if pat.search(line):
                violations.append({'rule': label, 'path': path, 'line': line[:500]})

    report = {
        'policy': 'LEXIGEN V7 GSO Frozen Anti-Gaming Policy R1',
        'patch': str(args.patch),
        'allow_paths': sorted(allow),
        'changed_paths': changed,
        'added_executable_line_count': sum(1 for _, line in added if line.strip() and not line.lstrip().startswith(('#','//','/*','*'))),
        'eligible': not violations,
        'violations': violations,
        'expert_diff_inspected': False,
        'hints_inspected': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if violations:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
