from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

TARGET = Path("bindings/python/src/tokenizer.rs")
MARKER = "    /// Decode the given list of ids back to a string\n"

ROBUST = '''    /// Encode a batch with a raw-string fast path and compatible recovery.
    #[pyo3(signature = (input, add_special_tokens = true, is_pretokenized = false))]
    #[pyo3(text_signature = "(self, input, add_special_tokens=True, is_pretokenized=False)")]
    fn encode_batch_fast(
        &self,
        py: Python<'_>,
        input: &PyAny,
        add_special_tokens: bool,
        is_pretokenized: bool,
    ) -> PyResult<Vec<PyEncoding>> {
        if !is_pretokenized {
            if let Ok(raw) = input.extract::<Vec<String>>() {
                let converted: Vec<tk::EncodeInput> = raw
                    .into_iter()
                    .map(|s| tk::EncodeInput::Single(s.into()))
                    .collect();
                return py.allow_threads(|| {
                    ToPyResult(
                        self.tokenizer
                            .encode_batch_char_offsets(converted, add_special_tokens)
                            .map(|encodings| encodings.into_iter().map(|e| e.into()).collect()),
                    )
                    .into()
                });
            }
        }

        let objects = input.extract::<Vec<&PyAny>>()?;
        let converted: Vec<tk::EncodeInput> = objects
            .into_iter()
            .map(|o| {
                let item: tk::EncodeInput = if is_pretokenized {
                    o.extract::<PreTokenizedEncodeInput>()?.into()
                } else {
                    o.extract::<TextEncodeInput>()?.into()
                };
                Ok(item)
            })
            .collect::<PyResult<Vec<tk::EncodeInput>>>()?;
        py.allow_threads(|| {
            ToPyResult(
                self.tokenizer
                    .encode_batch_char_offsets(converted, add_special_tokens)
                    .map(|encodings| encodings.into_iter().map(|e| e.into()).collect()),
            )
            .into()
        })
    }

'''

GENERAL = '''    /// Compatibility batch entry using the existing dynamic conversion path.
    #[pyo3(signature = (input, add_special_tokens = true, is_pretokenized = false))]
    #[pyo3(text_signature = "(self, input, add_special_tokens=True, is_pretokenized=False)")]
    fn encode_batch_fast(
        &self,
        py: Python<'_>,
        input: Vec<&PyAny>,
        add_special_tokens: bool,
        is_pretokenized: bool,
    ) -> PyResult<Vec<PyEncoding>> {
        let converted: Vec<tk::EncodeInput> = input
            .into_iter()
            .map(|o| {
                let item: tk::EncodeInput = if is_pretokenized {
                    o.extract::<PreTokenizedEncodeInput>()?.into()
                } else {
                    o.extract::<TextEncodeInput>()?.into()
                };
                Ok(item)
            })
            .collect::<PyResult<Vec<tk::EncodeInput>>>()?;
        py.allow_threads(|| {
            ToPyResult(
                self.tokenizer
                    .encode_batch_char_offsets(converted, add_special_tokens)
                    .map(|encodings| encodings.into_iter().map(|e| e.into()).collect()),
            )
            .into()
        })
    }

'''

BODIES = {
    "F2R1": ROBUST,
    "N4R1": ROBUST,
    "R4R1": GENERAL,
}

PROVENANCE = {
    "F2R1": {"arm": "v7_full", "source_proposal_id": "F2", "macro_ids": ["V7M-002"]},
    "N4R1": {"arm": "v7_no_library", "source_proposal_id": "N4", "macro_ids": []},
    "R4R1": {"arm": "v7_random_library", "source_proposal_id": "R4", "macro_ids": ["V7R-003"]},
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(root: Path, candidate: str, output: Path) -> dict:
    if candidate not in BODIES:
        raise SystemExit(f"unknown Task3 revision1 candidate: {candidate}")
    path = root / TARGET
    before = path.read_text(encoding="utf-8")
    if before.count(MARKER) != 1:
        raise RuntimeError(f"expected exactly one insertion marker, found {before.count(MARKER)}")
    if "fn encode_batch_fast(" in before:
        raise RuntimeError("base source unexpectedly already contains encode_batch_fast")
    after = before.replace(MARKER, BODIES[candidate] + MARKER, 1)
    path.write_text(after, encoding="utf-8")

    patch = "".join(difflib.unified_diff(
        before.splitlines(True),
        after.splitlines(True),
        fromfile=f"a/{TARGET.as_posix()}",
        tofile=f"b/{TARGET.as_posix()}",
    ))
    output.mkdir(parents=True, exist_ok=True)
    patch_path = output / f"task3-{candidate}.patch"
    patch_path.write_text(patch, encoding="utf-8")
    report = {
        "task": 3,
        "candidate": candidate,
        "target": TARGET.as_posix(),
        "before_sha256": sha256(before.encode()),
        "after_sha256": sha256(after.encode()),
        "patch_sha256": sha256(patch.encode()),
        "provenance": PROVENANCE[candidate],
        "expert_information_used": False,
        "hints_used": False,
        "thresholds_changed": False,
    }
    (output / f"task3-{candidate}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    build(args.root, args.candidate, args.output)


if __name__ == "__main__":
    main()
