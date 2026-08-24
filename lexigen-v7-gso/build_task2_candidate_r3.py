from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_candidate_r1 as r1
import build_task2_candidate_r2 as r2


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one target inside Llama.eval, got {count}")
    return text.replace(old, new, 1)


def transform_r3(text: str) -> str:
    start = text.index("    def eval(self, tokens: Sequence[int]):")
    end = text.index("\n    def ", start + 8)
    prefix, segment, suffix = text[:start], text[start:end], text[end:]
    old = '''        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)\n        for i in range(0, len(tokens), self.n_batch):\n'''
    new = '''        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)\n        set_batch = self._batch.set_batch\n        decode = self._ctx.decode\n        for i in range(0, len(tokens), self.n_batch):\n'''
    segment = replace_once(segment, old, new, "R3-local-bind")
    segment = replace_once(segment, "            self._batch.set_batch(\n", "            set_batch(\n", "R3-set-batch")
    segment = replace_once(segment, "            self._ctx.decode(self._batch)\n", "            decode(self._batch)\n", "R3-decode")
    return prefix + segment + suffix


def write_patch(rel: Path, before: str, after: str, patch_path: Path) -> None:
    diff = difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile=f"a/{rel.as_posix()}", tofile=f"b/{rel.as_posix()}"
    )
    patch_path.write_text("".join(diff), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    root = args.root.resolve()
    if args.candidate == "R3":
        path = root / "llama_cpp/llama.py"
        before = path.read_text(encoding="utf-8")
        after = transform_r3(before)
        path.write_text(after, encoding="utf-8")
    elif args.candidate in {"F2", "N1", "N3"}:
        path = root / "llama_cpp/llama.py"
        before = path.read_text(encoding="utf-8")
        after = r2.transform_eval(before, args.candidate)
        path.write_text(after, encoding="utf-8")
    else:
        path, before, after = r1.task2(root, args.candidate)

    rel = path.relative_to(root)
    patch_path = args.output / f"task2-{args.candidate}.patch"
    write_patch(rel, before, after, patch_path)
    report = {
        "task": 2,
        "candidate": args.candidate,
        "changed_file": str(rel),
        "before_sha256": sha256_bytes(before.encode()),
        "after_sha256": sha256_bytes(after.encode()),
        "patch_sha256": sha256_bytes(patch_path.read_bytes()),
        "candidate_execution_count": 0,
        "candidate_timing_observed": False,
        "repair_scope": "construction locator only; candidate mechanism unchanged",
        "expert_information_used": False,
    }
    (args.output / f"task2-{args.candidate}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
