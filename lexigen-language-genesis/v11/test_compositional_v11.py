from __future__ import annotations

from compositional_runtime_v11 import as_grid, canonical_json, execute_pipeline
from compositional_synthesizer_v11 import synthesize_pipeline
from portable_runtime_v11 import as_grid as portable_grid
from portable_runtime_v11 import execute_portable


def make_input(values):
    height, width = len(values), len(values[0])
    grid = [[0 for _ in range(2 * width + 1)] for _ in range(2 * height + 1)]
    for row in range(height):
        for col in range(width):
            grid[2 * row + 1][2 * col + 1] = values[row][col]
    return as_grid(grid)


def render_expected(values):
    height, width = len(values), len(values[0])
    gap = margin = 2
    tile_h = (26 - 2 * margin - gap * (height - 1)) // height
    tile_w = (26 - 2 * margin - gap * (width - 1)) // width
    canvas = [[0 for _ in range(26)] for _ in range(26)]

    def bounds(row, col):
        r0 = margin + row * (tile_h + gap)
        c0 = margin + col * (tile_w + gap)
        return r0, r0 + tile_h - 1, c0, c0 + tile_w - 1

    horizontal = []
    vertical = []
    for row in range(height):
        for col in range(width - 1):
            if values[row][col] == values[row][col + 1] != 0:
                horizontal.append(((row, col), (row, col + 1)))
    claimed = {point for edge in horizontal for point in edge}
    for row in range(height - 1):
        for col in range(width):
            if values[row][col] == values[row + 1][col] != 0:
                edge = ((row, col), (row + 1, col))
                if edge[0] not in claimed and edge[1] not in claimed:
                    vertical.append(edge)
    for row in range(height):
        for col in range(width):
            colour = values[row][col]
            if colour == 0:
                continue
            r0, r1, c0, c1 = bounds(row, col)
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    canvas[r][c] = colour
    for first, second in horizontal:
        r0, r1, _, c1 = bounds(*first)
        _, _, c2, _ = bounds(*second)
        for r in range(r0, r1 + 1):
            for c in range(c1 + 1, c2):
                canvas[r][c] = values[first[0]][first[1]]
    for first, second in vertical:
        _, r1, c0, c1 = bounds(*first)
        r2, _, _, _ = bounds(*second)
        for r in range(r1 + 1, r2):
            for c in range(c0, c1 + 1):
                canvas[r][c] = values[first[0]][first[1]]
    return as_grid(canvas)


def examples():
    values = [
        [[4, 4], [4, 6]],
        [[3, 6, 3], [3, 9, 6]],
        [[3, 7, 8], [7, 5, 5], [7, 3, 5], [3, 5, 3]],
    ]
    return [(make_input(item), render_expected(item)) for item in values]


def test_v11_synthesizes_five_stage_pipeline() -> None:
    result = synthesize_pipeline(examples())
    assert result.program is not None
    assert [stage["kind"] for stage in result.program["stages"]] == [
        "extract_lattice",
        "allocate_layout",
        "build_relations",
        "apply_precedence",
        "render",
    ]
    assert result.program["stages"][3]["mode"] == "horizontal_then_vertical_unclaimed"


def test_v11_portable_runtime_agrees() -> None:
    result = synthesize_pipeline(examples())
    assert result.program is not None
    for source, target in examples():
        assert execute_pipeline(result.program, source) == target
        assert execute_portable(result.program, portable_grid(source)) == target


def test_v11_requires_equal_colour_relation() -> None:
    result = synthesize_pipeline(examples())
    assert result.program is not None
    assert result.program["stages"][2]["predicate"] == "equal_nonbackground"


def test_v11_artifact_has_no_generator_or_task_name() -> None:
    result = synthesize_pipeline(examples())
    assert result.program is not None
    text = canonical_json(result.program)
    assert "task_33067df9" not in text
    assert "connective_tissue" not in text
    assert result.program["provenance"]["human_supplied_finished_task_operator"] is False
