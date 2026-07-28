from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

State = frozenset[str]
Step = Callable[[State], State]


class World(Protocol):
    name: str
    surface: str
    seed: State

    def step(self, state: State) -> State: ...

    def independently_verified_target(self) -> State: ...


@dataclass(frozen=True)
class EdgeWorld:
    name: str
    edges: tuple[tuple[str, str], ...]
    seed: State
    surface: str = "edge"

    def step(self, state: State) -> State:
        added = {dst for src, dst in self.edges if src in state}
        return frozenset(set(state) | added)

    def independently_verified_target(self) -> State:
        adjacency: dict[str, set[str]] = {}
        for src, dst in self.edges:
            adjacency.setdefault(src, set()).add(dst)
        reached = set(self.seed)
        queue = list(sorted(self.seed))
        while queue:
            current = queue.pop(0)
            for nxt in sorted(adjacency.get(current, ())):
                if nxt not in reached:
                    reached.add(nxt)
                    queue.append(nxt)
        return frozenset(reached)


@dataclass(frozen=True)
class RuleWorld:
    name: str
    implications: tuple[tuple[str, str], ...]
    seed: State
    surface: str = "rule"

    def step(self, state: State) -> State:
        consequences = {right for left, right in self.implications if left in state}
        return frozenset(set(state) | consequences)

    def independently_verified_target(self) -> State:
        reached = set(self.seed)
        changed = True
        while changed:
            changed = False
            for premise, consequence in self.implications:
                if premise in reached and consequence not in reached:
                    reached.add(consequence)
                    changed = True
        return frozenset(reached)


@dataclass(frozen=True)
class GridWorld:
    name: str
    open_cells: State
    start: str
    width: int
    height: int
    surface: str = "grid"

    @property
    def seed(self) -> State:
        return frozenset({self.start})

    def _neighbours(self, cell: str) -> Iterable[str]:
        x_text, y_text = cell.split(",")
        x, y = int(x_text), int(y_text)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            candidate = f"{nx},{ny}"
            if 0 <= nx < self.width and 0 <= ny < self.height and candidate in self.open_cells:
                yield candidate

    def step(self, state: State) -> State:
        added = {neighbour for cell in state for neighbour in self._neighbours(cell)}
        return frozenset(set(state) | added)

    def independently_verified_target(self) -> State:
        reached = {self.start}
        stack = [self.start]
        while stack:
            current = stack.pop()
            for neighbour in self._neighbours(current):
                if neighbour not in reached:
                    reached.add(neighbour)
                    stack.append(neighbour)
        return frozenset(reached)


def bounded_unroll_3(step: Step, seed: State) -> State:
    """Frozen starting language: exactly three transition applications."""
    state = seed
    for _ in range(3):
        state = step(state)
    return state


def oracle_least_fixed_point(step: Step, seed: State) -> State:
    """Benchmark-validation oracle, never evidence of autonomous invention."""
    state = seed
    while True:
        updated = step(state)
        if updated == state:
            return state
        if not state.issubset(updated):
            raise ValueError("RIFT-0 requires monotone transitions")
        state = updated


def _chain_edges(rounds: int, prefix: str) -> tuple[tuple[str, str], ...]:
    return tuple((f"{prefix}{i}", f"{prefix}{i + 1}") for i in range(rounds))


def _edge_world(rounds: int, index: int) -> EdgeWorld:
    prefix = f"e{index}_"
    return EdgeWorld(
        name=f"edge-r{rounds}-i{index}",
        edges=_chain_edges(rounds, prefix),
        seed=frozenset({f"{prefix}0"}),
    )


def _rule_world(rounds: int, index: int) -> RuleWorld:
    prefix = f"fact{index}_"
    return RuleWorld(
        name=f"rule-r{rounds}-i{index}",
        implications=_chain_edges(rounds, prefix),
        seed=frozenset({f"{prefix}0"}),
    )


def _grid_world(rounds: int, index: int) -> GridWorld:
    # A one-cell-wide winding-free corridor requires exactly `rounds` expansions.
    cells = frozenset(f"{x},0" for x in range(rounds + 1))
    return GridWorld(
        name=f"grid-r{rounds}-i{index}",
        open_cells=cells,
        start="0,0",
        width=rounds + 1,
        height=1,
    )


def build_cases(rounds_values: Iterable[int], replicas: int = 2) -> list[World]:
    cases: list[World] = []
    for rounds in rounds_values:
        for index in range(replicas):
            cases.extend(
                (
                    _edge_world(rounds, index),
                    _rule_world(rounds, index),
                    _grid_world(rounds, index),
                )
            )
    return cases


def evaluate(solver: Callable[[Step, State], State], cases: list[World]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for case in cases:
        expected = case.independently_verified_target()
        predicted = solver(case.step, case.seed)
        records.append(
            {
                "name": case.name,
                "surface": case.surface,
                "correct": predicted == expected,
                "predicted_size": len(predicted),
                "expected_size": len(expected),
            }
        )
    accuracy = sum(bool(record["correct"]) for record in records) / len(records)
    by_surface = {
        surface: sum(bool(r["correct"]) for r in records if r["surface"] == surface)
        / sum(1 for r in records if r["surface"] == surface)
        for surface in sorted({str(r["surface"]) for r in records})
    }
    return {"accuracy": accuracy, "by_surface": by_surface, "records": records}


def build_report() -> dict[str, object]:
    train_cases = build_cases(range(1, 4), replicas=3)
    transfer_cases = build_cases(range(4, 13), replicas=2)

    report: dict[str, object] = {
        "benchmark": "RIFT-0",
        "version": 1,
        "train_case_count": len(train_cases),
        "transfer_case_count": len(transfer_cases),
        "bounded_train": evaluate(bounded_unroll_3, train_cases),
        "bounded_transfer": evaluate(bounded_unroll_3, transfer_cases),
        "oracle_train": evaluate(oracle_least_fixed_point, train_cases),
        "oracle_transfer": evaluate(oracle_least_fixed_point, transfer_cases),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def validate_report(report: dict[str, object]) -> None:
    bounded_train = report["bounded_train"]
    bounded_transfer = report["bounded_transfer"]
    oracle_train = report["oracle_train"]
    oracle_transfer = report["oracle_transfer"]
    assert isinstance(bounded_train, dict)
    assert isinstance(bounded_transfer, dict)
    assert isinstance(oracle_train, dict)
    assert isinstance(oracle_transfer, dict)
    assert float(bounded_train["accuracy"]) >= 0.95
    assert float(bounded_transfer["accuracy"]) < 0.80
    assert float(oracle_train["accuracy"]) == 1.0
    assert float(oracle_transfer["accuracy"]) == 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("rift0-report.json"))
    args = parser.parse_args()
    report = build_report()
    validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "bounded_train_accuracy": report["bounded_train"]["accuracy"],  # type: ignore[index]
        "bounded_transfer_accuracy": report["bounded_transfer"]["accuracy"],  # type: ignore[index]
        "oracle_transfer_accuracy": report["oracle_transfer"]["accuracy"],  # type: ignore[index]
        "content_sha256": report["content_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
