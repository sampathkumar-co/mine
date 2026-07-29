from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from runtime_v22 import Grid, canonical, execute

@dataclass(frozen=True)
class Result:
    program: dict | None
    candidates_tested: int
    exact_count: int


def candidates() -> Iterable[dict]:
    yield {"op": "pack_columns"}
    for marker in range(10):
        for paint in range(10):
            if paint != marker:
                yield {"op": "connect_aligned", "marker_colour": marker, "paint_colour": paint}
    for axis in ("vertical", "horizontal"):
        for marker in range(10):
            for yes in range(10):
                for no in range(10):
                    if yes != no:
                        yield {
                            "op": "classify_reflection",
                            "axis": axis,
                            "marker_colour": marker,
                            "equal_colour": yes,
                            "unequal_colour": no,
                        }


def synthesize(examples: list[tuple[Grid, Grid]]) -> Result:
    exact = []
    tested = 0
    for program in candidates():
        tested += 1
        try:
            if all(execute(program, source) == target for source, target in examples):
                exact.append(program)
        except Exception:
            continue
    exact.sort(key=canonical)
    return Result(exact[0] if exact else None, tested, len(exact))
