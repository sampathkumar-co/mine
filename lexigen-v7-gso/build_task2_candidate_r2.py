from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_candidate_r1 as r1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one target inside Llama.eval, got {count}")
    return text.replace(old, new, 1)


def transform_eval(text: str, cid: str) -> str:
    start = text.index("    def eval(self, tokens: Sequence[int]):")
    end = text.index("\n    def ", start + 8)
    prefix, segment, suffix = text[:start], text[start:end], text[end:]
    marker = "        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)\n"

    if cid in {"F2", "N1"}:
        fast = '''        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)
        if len(tokens) == 1:
            n_past = self.n_tokens
            self._batch.set_batch(
                batch=tokens, n_past=n_past, logits_all=self.context_params.logits_all
            )
            self._ctx.decode(self._batch)
            self.input_ids[n_past] = tokens[0]
            if self.context_params.logits_all:
                logits = np.ctypeslib.as_array(
                    self._ctx.get_logits(), shape=(self._n_vocab,)
                )
                self.scores[n_past, :][::] = logits
            self.n_tokens = n_past + 1
            return
'''
        segment = replace_once(segment, marker, fast, cid)
    elif cid == "N3":
        new = marker + "        if not isinstance(tokens, np.ndarray):\n            tokens = np.asarray(tokens, dtype=np.intc)\n"
        segment = replace_once(segment, marker, new, cid)
    else:
        raise KeyError(cid)
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

    if args.candidate not in {"F2", "N1", "N3"}:
        path, before, after = r1.task2(args.root.resolve(), args.candidate)
    else:
        path = args.root.resolve() / "llama_cpp/llama.py"
        before = path.read_text(encoding="utf-8")
        after = transform_eval(before, args.candidate)
        path.write_text(after, encoding="utf-8")

    rel = path.relative_to(args.root.resolve())
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
