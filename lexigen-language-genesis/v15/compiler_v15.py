from __future__ import annotations

from typing import Any

from ir_runtime_v15 import AST


def input_node() -> AST:
    return {"op": "input"}


def background(grid: AST | None = None) -> AST:
    return {"op": "background", "grid": grid or input_node()}


def compile_stage(stage: dict[str, Any]) -> AST:
    op = stage["op"]
    source = input_node()
    if op == "recolour":
        return {"op": "recolour", "grid": source, "mapping": dict(stage["mapping"])}
    if op == "move_singleton_towards":
        source_colour = int(stage["source_colour"])
        target_colour = int(stage["target_colour"])
        source_point = {"op": "singleton", "grid": source, "colour": source_colour}
        target_point = {"op": "singleton", "grid": source, "colour": target_colour}
        return {
            "op": "move_point",
            "grid": source,
            "point": source_point,
            "delta": {"op": "unit_step_towards", "source": source_point, "target": target_point},
            "erase": background(source),
            "colour": source_colour,
        }
    if op == "edge_project":
        fill = stage.get("fill_colour")
        fill_node: Any = background(source) if fill is None else int(fill)
        return {"op": "edge_project", "grid": source, "fill": fill_node}
    if op == "decode_regular_linegrid":
        separator: Any
        if stage.get("line_colour") == "structural":
            separator = {"op": "separator_role", "grid": source}
        else:
            separator = int(stage["line_colour"])
        decoded = {"op": "decode_cells", "grid": source, "separator": separator}
        return {"op": "transform", "grid": decoded, "name": str(stage["transform"])}
    if op == "overlay_equal_tiles":
        tiles = {
            "op": "partition",
            "grid": source,
            "rows": int(stage["tile_rows"]),
            "cols": int(stage["tile_cols"]),
        }
        return {
            "op": "overlay",
            "tiles": tiles,
            "order": [int(value) for value in stage["order"]],
            "background": background(source),
        }
    if op == "canonical_rectangular_layers":
        mode = str(stage["object_mode"])
        objects = {"op": "rect_objects", "grid": source, "mode": mode}
        order = {"op": "rect_order", "grid": source, "objects": objects, "mode": mode}
        return {"op": "render_concentric", "objects": objects, "order": order}
    if op == "fill_internal_blank_axis":
        points = {"op": "blank_axis", "grid": source}
        return {
            "op": "paint",
            "grid": source,
            "points": points,
            "colour": int(stage["fill_colour"]),
        }
    if op == "extend_corner_marked_rays":
        motifs = {"op": "corner_motifs", "grid": source}
        points = {"op": "ray_points", "grid": source, "motifs": motifs}
        return {"op": "paint", "grid": source, "points": points, "colour": None}
    raise ValueError(f"unsupported v14 stage: {op}")


def compile_pipeline(pipeline: list[dict[str, Any]]) -> AST:
    if len(pipeline) != 1:
        raise ValueError("v15 development compiler currently expects one v14 stage")
    return compile_stage(pipeline[0])
