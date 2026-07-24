from pathlib import Path

import pytest

from app.core.config import Settings
from app.director.edit_graph import EditSegment
from app.director.rhythm import (
    MusicTimingPlan,
    estimate_rhythm_from_envelope,
    plan_music_timing,
    prepare_aligned_music,
)
from app.director.semantic_overlays import ProductionEditDecisionGraph
from app.sensory.models import MusicProfile


def _pulse_envelope(*, bpm: float, frames_per_second: int, duration_seconds: int) -> list[float]:
    frame_count = frames_per_second * duration_seconds
    interval = round(frames_per_second * 60 / bpm)
    values = [0.04] * frame_count
    for index in range(6, frame_count, interval):
        values[index] = 1.0
        if index + 1 < frame_count:
            values[index + 1] = 0.35
    return values


def test_estimate_rhythm_detects_regular_120_bpm_grid() -> None:
    estimate = estimate_rhythm_from_envelope(
        _pulse_envelope(bpm=120, frames_per_second=50, duration_seconds=24),
        frames_per_second=50,
        duration_seconds=24,
    )

    assert estimate.tempo_bpm == pytest.approx(120, abs=2)
    assert estimate.beat_interval_seconds == pytest.approx(0.5, abs=0.03)
    assert estimate.confidence >= 0.2
    assert len(estimate.beat_times) >= 40
    assert estimate.phrase_times


def test_plan_music_timing_aligns_cuts_and_source_change_phrase() -> None:
    graph = ProductionEditDecisionGraph(
        target_duration_seconds=10,
        selected_duration_seconds=10,
        segments=[
            EditSegment(
                source_asset_id="a",
                source_start=0,
                source_end=1,
                output_start=0,
                output_end=1,
                score=0.8,
                confidence=0.8,
                reason="hook",
            ),
            EditSegment(
                source_asset_id="a",
                source_start=1,
                source_end=8,
                output_start=1,
                output_end=8,
                score=0.8,
                confidence=0.8,
                transition="continuity_cut",
                reason="body",
            ),
            EditSegment(
                source_asset_id="b",
                source_index=1,
                source_start=0,
                source_end=2,
                output_start=8,
                output_end=10,
                score=0.8,
                confidence=0.8,
                transition="source_change_cut",
                reason="close",
            ),
        ],
    )
    profile = MusicProfile(
        asset_id="music",
        filename="track.mp3",
        duration_seconds=90,
        tempo_bpm=120,
        beat_interval_seconds=0.5,
        beat_offset_seconds=0.1,
        rhythm_confidence=0.9,
    )

    plan = plan_music_timing(graph, profile)

    assert plan.usable is True
    assert plan.start_offset_seconds == pytest.approx(0.1, abs=0.001)
    assert plan.beat_aligned_cut_count == 2
    assert plan.phrase_aligned_cut_count >= 1
    assert plan.alignment_score >= 0.9


def test_plan_music_timing_falls_back_for_uncertain_track() -> None:
    graph = ProductionEditDecisionGraph(
        target_duration_seconds=3,
        selected_duration_seconds=3,
        segments=[
            EditSegment(
                source_start=0,
                source_end=3,
                output_start=0,
                output_end=3,
                score=0.7,
                confidence=0.7,
                reason="single segment",
            )
        ],
    )
    profile = MusicProfile(
        asset_id="music",
        filename="ambient.mp3",
        duration_seconds=30,
        tempo_bpm=90,
        beat_interval_seconds=2 / 3,
        rhythm_confidence=0.05,
    )

    plan = plan_music_timing(graph, profile)

    assert plan.usable is False
    assert plan.start_offset_seconds == 0
    assert "confidence" in plan.reason.casefold()


def test_prepare_aligned_music_builds_bounded_ffmpeg_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr("app.director.rhythm.subprocess.run", fake_run)
    plan = MusicTimingPlan(
        usable=True,
        start_offset_seconds=0.375,
        tempo_bpm=128,
        beat_interval_seconds=0.46875,
        total_cut_count=4,
        beat_aligned_cut_count=4,
        phrase_aligned_cut_count=1,
        alignment_score=0.96,
        reason="aligned",
    )
    output = tmp_path / "aligned.m4a"

    prepare_aligned_music(
        tmp_path / "source.mp3",
        output,
        plan=plan,
        duration_seconds=12.5,
        settings=Settings(render_timeout_seconds=90),
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("-ss") + 1] == "0.375"
    assert command[command.index("-t") + 1] == "12.500"
    assert command[-1] == str(output)
    assert captured["kwargs"] == {
        "check": True,
        "capture_output": True,
        "text": True,
        "timeout": 90,
    }
