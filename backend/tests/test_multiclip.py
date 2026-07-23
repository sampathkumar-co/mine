from app.director.cleanup import refine_graph_with_word_timings
from app.director.edit_graph import (
    EditDecisionGraph,
    EditSegment,
    build_multiclip_edit_graph,
)
from app.rendering.captions import build_caption_cues
from app.sensory.models import (
    AnalysisBundle,
    ClipAnalysis,
    SceneRange,
    SubjectFraming,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)
from app.sensory.multiclip import analyze_source_clip, hamming_distance, mark_duplicate_clips


def _transcript(text: str, *, start: float = 0, end: float = 4) -> TranscriptResult:
    words = text.split()
    step = (end - start) / max(len(words), 1)
    return TranscriptResult(
        text=text,
        provider="fixture",
        model="fixture",
        words=[
            TranscriptWord(
                word=word,
                start=start + index * step,
                end=start + (index + 0.8) * step,
            )
            for index, word in enumerate(words)
        ],
        segments=[
            TranscriptSegment(
                start=start,
                end=end,
                text=text,
                confidence=0.92,
            )
        ],
    )


def _clip(
    asset_id: str,
    text: str,
    *,
    filename: str,
    sha256: str,
    role: str = "primary_speech",
    quality: float = 0.8,
) -> ClipAnalysis:
    transcript = _transcript(text)
    return ClipAnalysis(
        asset_id=asset_id,
        filename=filename,
        sha256=sha256,
        media={
            "duration_seconds": 4.0,
            "width": 1920,
            "height": 1080,
            "has_audio": True,
        },
        transcript=transcript,
        scenes=[SceneRange(start=0, end=4)],
        subject_framing=SubjectFraming(normalized_center_x=0.5, confidence=0.7),
        perceptual_hash=f"{int(asset_id[-1], 16):016x}",
        role=role,
        quality_score=quality,
    )


def test_marks_exact_and_near_visual_duplicates() -> None:
    first = _clip("asset-1", "A clear opening", filename="take-1.mp4", sha256="same")
    exact = _clip("asset-2", "A clear opening", filename="take-2.mp4", sha256="same")
    near = _clip("asset-3", "Another take", filename="take-3.mp4", sha256="different")
    near.perceptual_hash = "0000000000000001"
    first.perceptual_hash = "0000000000000000"

    marked = mark_duplicate_clips([first, exact, near], perceptual_distance_threshold=1)

    assert marked[0].role == "primary_speech"
    assert marked[1].duplicate_of_asset_id == "asset-1"
    assert marked[1].role == "rejected"
    assert marked[2].duplicate_of_asset_id == "asset-1"
    assert hamming_distance(first.perceptual_hash, near.perceptual_hash) == 1


def test_source_clip_role_and_quality_detect_evidence() -> None:
    clip = analyze_source_clip(
        asset_id="proof-1",
        filename="dashboard-proof.mp4",
        sha256="abc",
        media={
            "duration_seconds": 5.0,
            "width": 1080,
            "height": 1920,
            "has_audio": False,
        },
        transcript=None,
        scenes=[SceneRange(start=0, end=5)],
        subject_framing=SubjectFraming(),
        perceptual_hash="0123456789abcdef",
    )

    assert clip.role == "evidence"
    assert "proof" in clip.evidence_terms
    assert clip.quality_score > 0.4


def test_multiclip_graph_uses_multiple_assets() -> None:
    first = _clip(
        "asset-1",
        "How to avoid the biggest editing mistake",
        filename="hook.mp4",
        sha256="one",
    )
    second = _clip(
        "asset-2",
        "Here is the proof and the final result",
        filename="proof.mp4",
        sha256="two",
        role="evidence",
    )
    analysis = AnalysisBundle(
        media=first.media,
        transcript=first.transcript,
        scenes=first.scenes,
        source_clips=[first, second],
    )

    graph = build_multiclip_edit_graph(
        analysis,
        objective="Teach creators how to avoid editing mistakes with proof",
        target_duration_seconds=7,
    )

    assert graph.strategy == "tier1_multiclip_story"
    assert graph.selected_duration_seconds == 7
    assert {segment.source_asset_id for segment in graph.segments} == {"asset-1", "asset-2"}
    assert graph.segments[0].output_start == 0
    assert graph.segments[-1].output_end == 7


def test_multisource_cleanup_and_captions_resolve_each_transcript() -> None:
    first_transcript = _transcript("Hello um world")
    second_transcript = _transcript("Proof appears now")
    graph = EditDecisionGraph(
        target_duration_seconds=8,
        selected_duration_seconds=8,
        segments=[
            EditSegment(
                source_asset_id="asset-1",
                source_index=0,
                source_start=0,
                source_end=4,
                output_start=0,
                output_end=4,
                score=0.8,
                confidence=0.9,
                reason="Opening",
                transcript_text="Hello um world",
            ),
            EditSegment(
                source_asset_id="asset-2",
                source_index=1,
                clip_role="evidence",
                source_start=0,
                source_end=4,
                output_start=4,
                output_end=8,
                score=0.75,
                confidence=0.85,
                reason="Proof",
                transcript_text="Proof appears now",
            ),
        ],
    )
    transcripts = {
        "asset-1": first_transcript,
        "asset-2": second_transcript,
    }

    refined = refine_graph_with_word_timings(graph, transcripts)
    cues = build_caption_cues(refined, transcripts, max_words=4)

    assert {segment.source_asset_id for segment in refined.segments} == {"asset-1", "asset-2"}
    assert all("um" not in (segment.transcript_text or "").casefold() for segment in refined.segments)
    assert cues
    assert cues[-1].start >= refined.segments[-1].output_start
