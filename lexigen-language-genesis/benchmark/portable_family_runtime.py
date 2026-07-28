from __future__ import annotations

from collections.abc import Callable
from typing import Any

PortableState = frozenset[str]


def _key(value: PortableState) -> tuple[str, ...]:
    return tuple(sorted(value))


def run_portable_instance(
    instance: dict[str, Any],
    transition: Callable[[PortableState], PortableState],
    initial: PortableState,
    *,
    limit: int = 5_000,
) -> PortableState:
    """Second implementation written independently from the RIFT-2/3 runtime."""
    stop_rule = str(instance["stop"])
    output_rule = str(instance["finalize"])
    value = initial
    visited: set[tuple[str, ...]] = set()
    history: list[PortableState] = []

    for _ in range(limit):
        history.append(value)
        visited.add(_key(value))
        successor = transition(value)

        if stop_rule == "stable":
            done = successor == value
        elif stop_rule == "repeat":
            done = _key(successor) in visited
        else:
            raise ValueError(f"unsupported stop rule: {stop_rule}")

        if done:
            if output_rule == "current":
                return value
            if output_rule == "next":
                return successor
            if output_rule == "trace_union":
                return frozenset().union(*history)
            if output_rule == "canonical_pair":
                return value if _key(value) <= _key(successor) else successor
            if output_rule == "canonical_max":
                return value if _key(value) >= _key(successor) else successor
            raise ValueError(f"unsupported output rule: {output_rule}")

        value = successor

    raise RuntimeError("portable instance exceeded its execution limit")
