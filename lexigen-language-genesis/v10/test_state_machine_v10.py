from __future__ import annotations

from portable_runtime_v10 import as_grid as portable_grid
from portable_runtime_v10 import execute_portable
from state_machine_runtime_v10 import as_grid, canonical_json, execute_machine
from state_machine_synthesizer_v10 import synthesize_state_machine


def base_case():
    source = [[7 for _ in range(7)] for _ in range(7)]
    target = [[7 for _ in range(7)] for _ in range(7)]
    source[1][0] = source[1][1] = 3
    target[1][0] = target[1][1] = 3
    source[1][4] = target[1][4] = 8
    source[0][3] = target[0][3] = 8
    for row, col in [(1, 2), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3)]:
        target[row][col] = 3
    return as_grid(source), as_grid(target)


def flip_h(grid):
    return tuple(tuple(reversed(row)) for row in grid)


def flip_v(grid):
    return tuple(reversed(grid))


def transpose(grid):
    return tuple(tuple(grid[row][col] for row in range(len(grid))) for col in range(len(grid[0])))


def examples():
    source, target = base_case()
    return [
        (source, target),
        (flip_h(source), flip_h(target)),
        (flip_v(source), flip_v(target)),
        (transpose(source), transpose(target)),
    ]


def test_v10_synthesizes_sensor_transition() -> None:
    result = synthesize_state_machine(examples())
    assert result.machine is not None
    assert result.machine["seed"]["direction_mode"] == "away_from_boundary"
    assert result.machine["transition"]["turn"] == "away_from_lateral_obstacle"
    for source, target in examples():
        assert execute_machine(result.machine, source) == target


def test_v10_portable_runtime_agrees() -> None:
    result = synthesize_state_machine(examples())
    assert result.machine is not None
    for source, target in examples():
        assert execute_portable(result.machine, portable_grid(source)) == target


def test_v10_machine_has_no_named_ray_task_operator() -> None:
    result = synthesize_state_machine(examples())
    assert result.machine is not None
    text = canonical_json(result.machine)
    assert "bouncing_ray" not in text
    assert "task_6bcdb01e" not in text
    assert result.machine["provenance"]["human_supplied_finished_task_operator"] is False


def test_v10_ablation_fails() -> None:
    assert any(source != target for source, target in examples())
