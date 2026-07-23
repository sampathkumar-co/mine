from app.director.edit_graph import build_tier1_edit_graph
from app.sensory.models import AnalysisBundle, SceneRange, TranscriptResult, TranscriptSegment
from app.sensory.transcription import parse_verbose_transcription


def test_parses_word_and_segment_timestamps() -> None:
    result = parse_verbose_transcription(
        {
            "text": "How to improve your video",
            "language": "en",
            "duration": 2.5,
            "words": [
                {"word": "How", "start": 0.0, "end": 0.3},
                {"word": "video", "start": 2.0, "end": 2.5},
            ],
            "segments": [
                {"text": "How to improve your video", "start": 0.0, "end": 2.5}
            ],
        },
        provider="openai",
        model="whisper-1",
    )

    assert result.duration_seconds == 2.5
    assert result.words[0].word == "How"
    assert result.segments[0].end == 2.5


def test_tier1_graph_prefers_clear_hook_over_filler() -> None:
    transcript = TranscriptResult(
        text="Um like. How to avoid the biggest editing mistake. Here is the proof.",
        provider="test",
        model="fixture",
        segments=[
            TranscriptSegment(start=0, end=4, text="Um like you know", confidence=0.9),
            TranscriptSegment(
                start=4,
                end=10,
                text="How to avoid the biggest editing mistake",
                confidence=0.94,
            ),
            TranscriptSegment(start=10, end=15, text="Here is the proof", confidence=0.9),
        ],
    )
    analysis = AnalysisBundle(
        media={"duration_seconds": 15.0, "has_audio": True},
        transcript=transcript,
        scenes=[SceneRange(start=0, end=15)],
    )

    graph = build_tier1_edit_graph(
        analysis,
        objective="Teach creators how to avoid editing mistakes",
        target_duration_seconds=8,
    )

    assert graph.segments
    assert graph.selected_duration_seconds <= 8.01
    assert any("biggest editing mistake" in (segment.transcript_text or "") for segment in graph.segments)
    assert all(segment.reason for segment in graph.segments)


def test_tier1_graph_falls_back_to_scene_ranges_without_transcript() -> None:
    analysis = AnalysisBundle(
        media={"duration_seconds": 12.0, "has_audio": False},
        scenes=[
            SceneRange(start=0, end=4),
            SceneRange(start=4, end=8),
            SceneRange(start=8, end=12),
        ],
    )

    graph = build_tier1_edit_graph(
        analysis,
        objective="Create a clean product reel",
        target_duration_seconds=6,
    )

    assert graph.segments
    assert graph.selected_duration_seconds == 6
    assert graph.notes
