from __future__ import annotations

import os
from pathlib import Path

from app.core.config import Settings
from app.director.semantic_overlays import ProductionEditDecisionGraph, VisualOverlay
from app.director.style import ProductionStyle
from app.rendering.ffmpeg import (
    MediaCommandError,
    RenderSource,
    _escape_filter_path,
    _run,
    _vertical_filter,
    render_multiclip_edit_decision_graph,
)


def _source_for_overlay(
    overlay: VisualOverlay,
    sources: list[RenderSource],
) -> RenderSource:
    by_asset_id = {source.asset_id: source for source in sources}
    source = by_asset_id.get(overlay.source_asset_id)
    if source is None:
        raise MediaCommandError(
            f"Overlay references unavailable source asset {overlay.source_asset_id}"
        )
    return source


def render_semantic_production_graph(
    sources: list[RenderSource],
    output_path: str | Path,
    graph: ProductionEditDecisionGraph,
    *,
    caption_path: str | Path | None = None,
    music_path: str | Path | None = None,
    style: ProductionStyle | None = None,
    settings: Settings,
) -> None:
    production_style = style or ProductionStyle()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_path = output.with_name(f"{output.stem}.narration-base{output.suffix}")

    render_multiclip_edit_decision_graph(
        sources,
        base_path,
        graph,
        caption_path=None,
        music_path=music_path,
        style=production_style,
        settings=settings,
    )

    has_captions = caption_path is not None and Path(caption_path).exists()
    if not graph.overlays and not has_captions:
        os.replace(base_path, output)
        return

    command = [settings.ffmpeg_binary, "-y", "-i", str(base_path)]
    overlay_sources: list[RenderSource] = []
    for overlay in graph.overlays:
        source = _source_for_overlay(overlay, sources)
        if source.asset_id not in {item.asset_id for item in overlay_sources}:
            overlay_sources.append(source)
            command.extend(["-i", source.path])
    second_pass_index = {
        source.asset_id: index + 1 for index, source in enumerate(overlay_sources)
    }

    visual_finish = (
        f"eq=contrast={production_style.visual.contrast:.4f}:"
        f"saturation={production_style.visual.saturation:.4f}:"
        f"brightness={production_style.visual.brightness:.4f}"
    )
    filters: list[str] = []
    current_video = "basevideo"
    filters.append("[0:v]setpts=PTS-STARTPTS[basevideo]")

    for index, overlay in enumerate(graph.overlays):
        source = _source_for_overlay(overlay, sources)
        input_index = second_pass_index[source.asset_id]
        duration = overlay.output_end - overlay.output_start
        vertical_filter = _vertical_filter(source.probe, source.subject_center_x)
        alpha_filters = "format=yuva420p"
        if overlay.transition == "overlay_fade" and duration >= 0.4:
            fade_duration = min(0.14, duration / 4)
            fade_out_start = max(0.0, duration - fade_duration)
            alpha_filters += (
                f",fade=t=in:st=0:d={fade_duration:.3f}:alpha=1"
                f",fade=t=out:st={fade_out_start:.3f}:d={fade_duration:.3f}:alpha=1"
            )
        filters.append(
            f"[{input_index}:v]trim=start={overlay.source_start}:end={overlay.source_end},"
            f"setpts=PTS-STARTPTS+{overlay.output_start:.3f}/TB,"
            f"{vertical_filter},{visual_finish},{alpha_filters}[overlay{index}]"
        )
        next_video = f"video{index}"
        filters.append(
            f"[{current_video}][overlay{index}]overlay=0:0:eof_action=pass:"
            f"enable='between(t,{overlay.output_start:.3f},{overlay.output_end:.3f})'"
            f"[{next_video}]"
        )
        current_video = next_video

    if has_captions:
        escaped_caption_path = _escape_filter_path(caption_path)
        filters.append(f"[{current_video}]ass=filename='{escaped_caption_path}'[vout]")
    else:
        filters.append(f"[{current_video}]null[vout]")

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-c:a",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    try:
        _run(command, timeout_seconds=settings.render_timeout_seconds)
    finally:
        base_path.unlink(missing_ok=True)
