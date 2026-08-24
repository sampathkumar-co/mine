from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one replacement target, got {count}")
    return text.replace(old, new, 1)


def write_patch(path: Path, before: str, after: str, patch_path: Path) -> None:
    diff = difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile=f"a/{path.as_posix()}", tofile=f"b/{path.as_posix()}"
    )
    patch_path.write_text("".join(diff), encoding="utf-8")


def task2(root: Path, cid: str) -> tuple[Path, str, str]:
    path = root / "llama_cpp/llama.py"
    before = path.read_text(encoding="utf-8")
    after = before
    if cid == "F1":
        old = '''    def eval(self, tokens: Sequence[int]):
        """Evaluate a list of tokens.

        Args:
            tokens: The list of tokens to evaluate.
        """
        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)
        for i in range(0, len(tokens), self.n_batch):
            batch = tokens[i : min(len(tokens), i + self.n_batch)]
            n_past = self.n_tokens
            n_tokens = len(batch)
            self._batch.set_batch(
                batch=batch, n_past=n_past, logits_all=self.context_params.logits_all
            )
            self._ctx.decode(self._batch)
            # Save tokens
            self.input_ids[n_past : n_past + n_tokens] = batch
            # Save logits
            if self.context_params.logits_all:
                rows = n_tokens
                cols = self._n_vocab
                logits = np.ctypeslib.as_array(
                    self._ctx.get_logits(), shape=(rows * cols,)
                )
                self.scores[n_past : n_past + n_tokens, :].reshape(-1)[::] = logits
            else:
                # rows = 1
                # cols = self._n_vocab
                # logits = np.ctypeslib.as_array(
                #     self._ctx.get_logits(), shape=(rows * cols,)
                # )
                # self.scores[n_past + n_tokens - 1, :].reshape(-1)[::] = logits
                # NOTE: Now that sampling is done inside the sampler, logits are only needed for logprobs which requires logits_all
                pass
            # Update n_tokens
            self.n_tokens += n_tokens
'''
        new = old.replace(
            '        for i in range(0, len(tokens), self.n_batch):\n',
            '        start_n_tokens = self.n_tokens\n        total_new_tokens = len(tokens)\n        if total_new_tokens:\n            self.input_ids[start_n_tokens : start_n_tokens + total_new_tokens] = tokens\n        local_n_tokens = start_n_tokens\n        for i in range(0, total_new_tokens, self.n_batch):\n'
        ).replace(
            '            n_past = self.n_tokens\n', '            n_past = local_n_tokens\n'
        ).replace(
            '            # Save tokens\n            self.input_ids[n_past : n_past + n_tokens] = batch\n',
            '            # Tokens were copied once before the decode loop.\n'
        ).replace(
            '            # Update n_tokens\n            self.n_tokens += n_tokens\n',
            '            local_n_tokens += n_tokens\n        self.n_tokens = local_n_tokens\n'
        )
        after = replace_once(after, old, new, cid)
    elif cid in {"F2", "N1"}:
        marker = '        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)\n'
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
        after = replace_once(after, marker, fast, cid)
    elif cid in {"F4", "N6", "R1"}:
        old = '''        if reset and self.n_tokens > 0:
            longest_prefix = 0
            for a, b in zip(self._input_ids, tokens[:-1]):
                if a == b:
                    longest_prefix += 1
                else:
                    break
'''
        new = '''        if reset and self.n_tokens > 0:
            max_prefix = min(self.n_tokens, max(0, len(tokens) - 1))
            if max_prefix:
                cached = self._input_ids[:max_prefix]
                incoming = np.asarray(tokens[:max_prefix], dtype=np.intc)
                mismatch = np.flatnonzero(cached != incoming)
                longest_prefix = int(mismatch[0]) if mismatch.size else max_prefix
            else:
                longest_prefix = 0
'''
        after = replace_once(after, old, new, cid)
    elif cid == "N3":
        marker = '        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)\n'
        new = marker + '        if not isinstance(tokens, np.ndarray):\n            tokens = np.asarray(tokens, dtype=np.intc)\n'
        after = replace_once(after, marker, new, cid)
    elif cid == "R3":
        old = '''        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)
        for i in range(0, len(tokens), self.n_batch):
'''
        new = '''        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)
        set_batch = self._batch.set_batch
        decode = self._ctx.decode
        for i in range(0, len(tokens), self.n_batch):
'''
        after = replace_once(after, old, new, cid)
        after = replace_once(after, '            self._batch.set_batch(\n', '            set_batch(\n', cid + '-set')
        after = replace_once(after, '            self._ctx.decode(self._batch)\n', '            decode(self._batch)\n', cid + '-decode')
    elif cid == "R6":
        old = '''                tokens_or_none = yield token
                tokens.clear()
                tokens.append(token)
                if tokens_or_none is not None:
                    tokens.extend(tokens_or_none)
'''
        new = '''                tokens_or_none = yield token
                if tokens:
                    tokens[0] = token
                    del tokens[1:]
                else:
                    tokens.append(token)
                if tokens_or_none is not None:
                    tokens.extend(tokens_or_none)
'''
        after = replace_once(after, old, new, cid)
    else:
        raise KeyError((2, cid))
    path.write_text(after, encoding="utf-8")
    return path, before, after


def _tokenizers_fast_method() -> str:
    return '''    /// Encode a raw-string batch through a specialized Python binding.
    #[pyo3(signature = (input, add_special_tokens = true))]
    #[pyo3(text_signature = "(self, input, add_special_tokens=True)")]
    fn encode_batch_fast(
        &self,
        py: Python<'_>,
        input: Vec<String>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<PyEncoding>> {
        let input: Vec<tk::EncodeInput> = input
            .into_iter()
            .map(|s| tk::EncodeInput::Single(s.into()))
            .collect();
        py.allow_threads(|| {
            ToPyResult(
                self.tokenizer
                    .encode_batch_char_offsets(input, add_special_tokens)
                    .map(|encodings| encodings.into_iter().map(|e| e.into()).collect()),
            )
            .into()
        })
    }

'''


def task3(root: Path, cid: str) -> tuple[Path, str, str]:
    path = root / "bindings/python/src/tokenizer.rs"
    before = path.read_text(encoding="utf-8")
    after = before
    direct = {"F1", "F3", "F4", "N1", "N3", "R1", "R5"}
    capacity = {"N6", "R2"}
    if cid in direct:
        marker = '    /// Decode the given list of ids back to a string\n'
        after = replace_once(after, marker, _tokenizers_fast_method() + marker, cid)
    elif cid in capacity:
        old = '''        let input: Vec<tk::EncodeInput> = input
            .into_iter()
            .map(|o| {
                let input: tk::EncodeInput = if is_pretokenized {
                    o.extract::<PreTokenizedEncodeInput>()?.into()
                } else {
                    o.extract::<TextEncodeInput>()?.into()
                };
                Ok(input)
            })
            .collect::<PyResult<Vec<tk::EncodeInput>>>()?;
'''
        new = '''        let mut converted: Vec<tk::EncodeInput> = Vec::with_capacity(input.len());
        for o in input {
            let item: tk::EncodeInput = if is_pretokenized {
                o.extract::<PreTokenizedEncodeInput>()?.into()
            } else {
                o.extract::<TextEncodeInput>()?.into()
            };
            converted.push(item);
        }
        let input = converted;
'''
        after = replace_once(after, old, new, cid)
    else:
        raise KeyError((3, cid))
    path.write_text(after, encoding="utf-8")
    return path, before, after


def task4(root: Path, cid: str) -> tuple[Path, str, str]:
    path = root / "numpy/core/code_generators/generate_umath.py"
    before = path.read_text(encoding="utf-8")
    after = before
    start = after.index("'subtract':")
    end = after.index("'multiply':", start)
    segment = after[start:end]
    old = "          indexed=flts + ints\n"
    if cid in {"F1", "F3", "F6", "N1", "N6", "R1"}:
        repl = "          indexed=flts + ints + cmplx\n"
    elif cid in {"N2", "R2"}:
        repl = "          indexed=flts + ints + 'D'\n"
    elif cid == "R6":
        repl = "          indexed=flts + ints + 'F'\n"
    else:
        raise KeyError((4, cid))
    if segment.count(old) != 1:
        raise RuntimeError(f"Task4 {cid}: subtract indexed target mismatch")
    segment2 = segment.replace(old, repl, 1)
    after = after[:start] + segment2 + after[end:]
    path.write_text(after, encoding="utf-8")
    return path, before, after


def _range_take_method(kind: str) -> str:
    # All variants implement only information independently visible in the base RangeIndex/Index source.
    if kind == "simple":
        body = '''        if kwargs:
            nv.validate_take((), kwargs)
        if is_scalar(indices):
            raise TypeError("Expected indices to be array-like")
        indices = ensure_platform_int(indices)
        allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)
        if allow_fill:
            return super().take(indices, axis=axis, allow_fill=allow_fill, fill_value=fill_value)
        n = len(self)
        if ((indices >= n) | (indices < -n)).any():
            raise IndexError("index out of bounds")
        indices = np.where(indices < 0, indices + n, indices)
        taken = self.start + self.step * indices
        return self._constructor._simple_new(taken.astype(np.int64, copy=False), name=self.name)
'''
    elif kind == "where":
        body = '''        if kwargs:
            nv.validate_take((), kwargs)
        if is_scalar(indices):
            raise TypeError("Expected indices to be array-like")
        indices = ensure_platform_int(indices)
        allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)
        if allow_fill:
            return super().take(indices, axis=axis, allow_fill=allow_fill, fill_value=fill_value)
        n = len(self)
        bad = (indices >= n) | (indices < -n)
        if bad.any():
            raise IndexError("index out of bounds")
        positions = indices.copy()
        positions[positions < 0] += n
        taken = np.multiply(positions, self.step, dtype=np.int64)
        taken += self.start
        return self._constructor._simple_new(taken, name=self.name)
'''
    else:
        raise KeyError(kind)
    return '''    @doc(Index.take)
    def take(
        self,
        indices,
        axis=0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ):
''' + body + '\n'


def task5(root: Path, cid: str) -> tuple[Path, str, str]:
    path = root / "pandas/core/indexes/range.py"
    before = path.read_text(encoding="utf-8")
    after = before
    marker = '    @doc(Index.__iter__)\n    def __iter__(self) -> Iterator[int]:\n'
    if cid in {"F1", "F4", "N1", "N4", "R1", "R2"}:
        method = _range_take_method("simple")
    elif cid in {"F2", "N2", "R6"}:
        method = _range_take_method("where")
    else:
        raise KeyError((5, cid))
    after = replace_once(after, marker, method + marker, cid)
    path.write_text(after, encoding="utf-8")
    return path, before, after


def _whisper_set_replace(after: str, frozen: bool = False) -> str:
    old = '''            timestamp_ids = self.timestamp_ids(time_precision=time_precision)
            token_ids = [token for token in token_ids if token not in timestamp_ids]
'''
    ctor = 'frozenset' if frozen else 'set'
    new = f'''            timestamp_ids = {ctor}(self.timestamp_ids(time_precision=time_precision))
            token_ids = [token for token in token_ids if token not in timestamp_ids]
'''
    return replace_once(after, old, new, "whisper-set")


def task6(root: Path, cid: str) -> tuple[Path, str, str]:
    path = root / "src/transformers/models/whisper/tokenization_whisper.py"
    before = path.read_text(encoding="utf-8")
    after = before
    if cid in {"F1", "N1", "R2"}:
        after = _whisper_set_replace(after, frozen=False)
    elif cid in {"R1"}:
        after = _whisper_set_replace(after, frozen=True)
    elif cid in {"F3", "N2"}:
        marker = '''    def _preprocess_token_ids(
'''
        helper = '''    @lru_cache
    def _timestamp_ids_set(self, time_precision=0.02):
        return frozenset(self.timestamp_ids(time_precision=time_precision))

'''
        after = replace_once(after, marker, helper + marker, cid + '-helper')
        old = '''            timestamp_ids = self.timestamp_ids(time_precision=time_precision)
            token_ids = [token for token in token_ids if token not in timestamp_ids]
'''
        new = '''            timestamp_ids = self._timestamp_ids_set(time_precision=time_precision)
            token_ids = [token for token in token_ids if token not in timestamp_ids]
'''
        after = replace_once(after, old, new, cid + '-use')
    elif cid in {"F4", "N3", "R5"}:
        old = '''            timestamp_ids = self.timestamp_ids(time_precision=time_precision)
            token_ids = [token for token in token_ids if token not in timestamp_ids]
'''
        new = '''            timestamp_ids = frozenset(self.timestamp_ids(time_precision=time_precision))
            if timestamp_ids:
                first_timestamp = min(timestamp_ids)
                last_timestamp = max(timestamp_ids)
                if len(timestamp_ids) == last_timestamp - first_timestamp + 1:
                    token_ids = [token for token in token_ids if token < first_timestamp or token > last_timestamp]
                else:
                    token_ids = [token for token in token_ids if token not in timestamp_ids]
'''
        after = replace_once(after, old, new, cid)
    else:
        raise KeyError((6, cid))
    path.write_text(after, encoding="utf-8")
    return path, before, after


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    dispatch = {2: task2, 3: task3, 4: task4, 5: task5, 6: task6}
    path, before, after = dispatch[args.task](root, args.candidate)
    args.output.mkdir(parents=True, exist_ok=True)
    rel = path.relative_to(root)
    patch_path = args.output / f"task{args.task}-{args.candidate}.patch"
    write_patch(rel, before, after, patch_path)
    report = {
        "task": args.task,
        "candidate": args.candidate,
        "changed_file": str(rel),
        "before_sha256": sha256_bytes(before.encode()),
        "after_sha256": sha256_bytes(after.encode()),
        "patch_sha256": sha256_bytes(patch_path.read_bytes()),
        "expert_information_used": False if args.task != 5 else None,
        "task5_campaign_credit_eligible": False if args.task == 5 else None,
    }
    (args.output / f"task{args.task}-{args.candidate}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
