from __future__ import annotations

from pathlib import Path
import types


_SOURCE = Path(__file__).with_name("real_exact_frontier_v43.py")
_text = _SOURCE.read_text(encoding="utf-8")
_start = _text.index("def expected_elimination_numerator(\n")
_end = _text.index("\ndef gini_approximation_certificate(\n", _start)
_replacement = '''def expected_elimination_numerator(
    task: object,
    allowed: int,
    query: int,
) -> int:
    """Uniform-prior expected eliminations, scaled by |V|."""
    size = allowed.bit_count()
    total = 0
    for mask in task.masks_for(query).values():
        child = allowed & mask
        if not child:
            continue
        bucket_size = child.bit_count()
        total += bucket_size * (size - bucket_size)
    return total
'''
_text = _text[:_start] + _replacement + _text[_end:]
_module = types.ModuleType("mini_origin.real_exact_frontier_v43_fixed")
_module.__file__ = str(_SOURCE)
_module.__package__ = "mini_origin"
exec(compile(_text, str(_SOURCE), "exec"), _module.__dict__)

for _name, _value in _module.__dict__.items():
    if not _name.startswith("__"):
        globals()[_name] = _value


if __name__ == "__main__":
    _module.main()
