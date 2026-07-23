from pathlib import Path

from app.director.cleanup import refine_graph_with_word_timings
from app.director.edit_graph import EditDecisionGraph, EditSegment
from app.rendering.captions import build_caption_cues, write_ass_captions
from app.rendering.ffmpeg import MediaProbe, _vertical_filter
from app.sensory.models import TranscriptResult, TranscriptWord


def _graph() -> EditDecisionGraph:
    return EditDecisionGraph(
        target_duration_seconds=5,
        selected_duration_seconds=5,
        segments=[
            EditSegment(
                source_start=0,
                source_end=5,
                output_start=0,
                output_end=5,
                score=0.8,
                confidence=0.9,
                reason="Selected as a clear explanation",
                transcript_text="Hello um world next idea",
            )
        ],
    )


def _transcript() -> TranscriptResult:
    return TranscriptResult(
        text="Hello um world next idea",
        provider="fixture",
        model="fixture",
        words=[
            TranscriptWord(word="Hello", start=0.2, end=0.5),
            TranscriptWord(word="um", start=0.6, end=0.8),
            TranscriptWord(word="world", start=0.9, end=1.2),
            TranscriptWord(word="next", start=2.0, end=2.3),
            TranscriptWord(word="idea", start=2.4, end=2.8),
        ],
    )


def test_word_cleanup_removes_fillers_and_long_silence() -> None:
    refined = refine_graph_with_word_timings(
        _graph(),
        _transcript(),
        silence_threshold_seconds=0.55,
        speech_padding_seconds=0.05,
    )

    assert refined.strategy == "tier1_retention_cleanup_with_word_precision"
    assert len(refined.segments) == 3
    assert refined.selected_duration_seconds < 2
    assert all("um" not in (segment.transcript_text or "").casefold() for segment in refined.segments)
    assert refined.segments[0].output_start == 0
    assert refined.segments[-1].output_end == refined.selected_duration_seconds


def test_caption_cues_follow_retimed_output_and_ass_safe_zone(tmp_path: Path) -> None:
    refined = refine_graph_with_word_timings(_graph(), _transcript())
    cues = build_caption_cues(refined, _transcript(), max_words=4)

    assert cues
    assert cues[0].start >= 0
    assert cues[-1].end <= refined.selected_duration_seconds

    caption_path = tmp_path / "captions.ass"
    count = write_ass_captions(
        caption_path,
        refined,
        _transcript(),
        max_words=4,
        margin_vertical=300,
    )
    content = caption_path.read_text(encoding="utf-8")
    assert count == len(cues)
    assert "Style: Director" in content
    assert ",300,1" in content
    assert r"\fad(60,80)" in content


def test_landscape_crop_tracks_subject_position() -> None:
    probe = MediaProbe(
        duration_seconds=10,
        width=1920,
        height=1080,
        video_codec="h264",
        has_audio=True,
    )

    centred = _vertical_filter(probe, 0.5)
    right_weighted = _vertical_filter(probe, 0.8)

    assert "crop=608:1080:656:0" in centred
    assert "crop=608:1080:1232:0" in right_weighted
    assert centred.endswith("scale=1080:1920:flags=lanczos,setsar=1")
