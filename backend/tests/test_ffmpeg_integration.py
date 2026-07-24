from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app.core.config import Settings
from app.director.edit_graph import EditDecisionGraph, EditSegment
from app.director.style import ProductionStyle
from app.rendering.ffmpeg import (
    RenderSource,
    probe_media,
    render_multiclip_edit_decision_graph,
    validate_vertical_output,
)

pytestmark = pytest.mark.skipif(
    os.getenv("DIRECTOR_RUN_MEDIA_SMOKE") != "1",
    reason="Set DIRECTOR_RUN_MEDIA_SMOKE=1 to run the real FFmpeg qualification.",
)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)


def _write_ass(path: Path) -> None:
    path.write_text(
        """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,72,&H00FFFFFF,&H00FFFFFF,&H00101010,&H00000000,-1,0,0,0,100,100,0,0,1,4,0,2,80,80,260,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
Dialogue: 0,0:00:00.00,0:00:02.30,Default,,0,0,0,,RELEASE QUALIFICATION
""",
        encoding="utf-8",
    )


def test_real_multiclip_caption_music_and_audio_render(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    assert ffmpeg and ffprobe, "FFmpeg and ffprobe are required for release qualification"

    source_a = tmp_path / "speaker.mp4"
    source_b = tmp_path / "evidence.mp4"
    music = tmp_path / "music.wav"
    captions = tmp_path / "captions.ass"
    output = tmp_path / "qualified.mp4"

    _run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30:duration=2.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2.4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-shortest",
            str(source_a),
        ]
    )
    _run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x244060:size=360x640:rate=30:duration=2.4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source_b),
        ]
    )
    _run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:sample_rate=48000:duration=4",
            "-ac",
            "2",
            str(music),
        ]
    )
    _write_ass(captions)

    settings = Settings(
        ffmpeg_binary=ffmpeg,
        ffprobe_binary=ffprobe,
        render_timeout_seconds=180,
    )
    probe_a = probe_media(source_a, settings)
    probe_b = probe_media(source_b, settings)
    graph = EditDecisionGraph(
        strategy="release_qualification",
        target_duration_seconds=2.4,
        selected_duration_seconds=2.4,
        segments=[
            EditSegment(
                source_asset_id="speaker",
                source_index=0,
                source_start=0,
                source_end=1.2,
                output_start=0,
                output_end=1.2,
                score=0.9,
                confidence=0.9,
                reason="Synthetic narration segment",
            ),
            EditSegment(
                source_asset_id="evidence",
                source_index=1,
                clip_role="evidence",
                source_start=0.2,
                source_end=1.4,
                output_start=1.2,
                output_end=2.4,
                score=0.9,
                confidence=0.9,
                transition="source_change_cut",
                reason="Synthetic silent evidence segment",
            ),
        ],
    )

    render_multiclip_edit_decision_graph(
        [
            RenderSource(asset_id="speaker", path=str(source_a), probe=probe_a),
            RenderSource(asset_id="evidence", path=str(source_b), probe=probe_b),
        ],
        output,
        graph,
        caption_path=captions,
        music_path=music,
        style=ProductionStyle(),
        settings=settings,
    )

    rendered = probe_media(output, settings)
    validate_vertical_output(rendered, expect_audio=True)
    assert 2.1 <= rendered.duration_seconds <= 2.8
    payload = output.read_bytes()
    assert payload.index(b"moov") < payload.index(b"mdat"), "Output is not fast-start optimized"
