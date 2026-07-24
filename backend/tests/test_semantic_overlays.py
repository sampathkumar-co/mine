from app.director.edit_graph import EditSegment, build_multiclip_edit_graph
from app.director.semantic_overlays import (
    ProductionEditDecisionGraph,
    VisualOverlay,
    enhance_graph_with_semantic_overlays,
)
from app.quality.editorial import review_and_repair_edit_graph
from app.sensory.models import (
    AnalysisBundle,
    ClipAnalysis,
    ContinuityProfile,
    SceneRange,
    SemanticTag,
    SubjectFraming,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)
from app.sensory.semantics import continuity_similarity, extract_text_semantic_tags


def _transcript(text: str, *, duration: float = 5) -> TranscriptResult:
    words = text.split()
    step = duration / max(1, len(words))
    return TranscriptResult(
        text=text,
        provider="fixture",
        model="fixture",
        words=[
            TranscriptWord(
                word=word,
                start=index * step,
                end=(index + 0.8) * step,
            )
            for index, word in enumerate(words)
        ],
        segments=[
            TranscriptSegment(start=0, end=duration, text=text, confidence=0.92)
        ],
    )


def _clip(
    asset_id: str,
    text: str,
    *,
    role: str = "primary_speech",
    tags: list[SemanticTag] | None = None,
    evidence_terms: list[str] | None = None,
    continuity: ContinuityProfile | None = None,
) -> ClipAnalysis:
    transcript = _transcript(text)
    return ClipAnalysis(
        asset_id=asset_id,
        filename=f"{asset_id}.mp4",
        sha256=asset_id,
        media={
            "duration_seconds": 5.0,
            "width": 1920,
            "height": 1080,
            "has_audio": role == "primary_speech",
        },
        transcript=transcript if role == "primary_speech" else None,
        scenes=[SceneRange(start=0, end=5)],
        subject_framing=SubjectFraming(normalized_center_x=0.5, confidence=0.7),
        role=role,
        quality_score=0.9,
        semantic_tags=tags or [],
        evidence_terms=evidence_terms or [],
        continuity=continuity or ContinuityProfile(),
    )


def test_extracts_inspectable_text_semantics() -> None:
    tags = extract_text_semantic_tags(
        "dashboard-proof.mp4",
        _transcript("Revenue increased by 42 percent after the new workflow"),
    )

    labels = {tag.label for tag in tags}
    assert {"screen", "proof", "chart", "before_after", "measurable_claim"} & labels
    assert all(tag.source in {"filename", "transcript"} for tag in tags)


def test_semantic_overlay_preserves_narration_and_matches_evidence() -> None:
    speech = _clip(
        "speech-1",
        "Our revenue result improved after changing the workflow",
        continuity=ContinuityProfile(brightness=0.55, saturation=0.5, subject_center_x=0.5),
    )
    evidence = _clip(
        "evidence-1",
        "",
        role="evidence",
        tags=[
            SemanticTag(
                label="chart",
                confidence=0.9,
                source="filename",
                evidence="revenue result dashboard",
            )
        ],
        evidence_terms=["revenue", "result", "proof"],
        continuity=ContinuityProfile(brightness=0.52, saturation=0.48, subject_center_x=0.5),
    )
    analysis = AnalysisBundle(
        media=speech.media,
        transcript=speech.transcript,
        scenes=speech.scenes,
        source_clips=[speech, evidence],
    )
    base = build_multiclip_edit_graph(
        analysis,
        objective="Show proof of the revenue result",
        target_duration_seconds=5,
    )

    graph = enhance_graph_with_semantic_overlays(
        base,
        analysis,
        objective="Show proof of the revenue result",
        target_duration_seconds=5,
        max_overlays=2,
        minimum_match_score=0.25,
    )

    assert isinstance(graph, ProductionEditDecisionGraph)
    assert {segment.source_asset_id for segment in graph.segments} == {"speech-1"}
    assert graph.overlays
    assert graph.overlays[0].source_asset_id == "evidence-1"
    assert graph.overlays[0].output_end <= graph.selected_duration_seconds
    assert "result" in graph.overlays[0].matched_terms


def test_continuity_similarity_and_transition_policy() -> None:
    close = continuity_similarity(
        ContinuityProfile(brightness=0.5, saturation=0.5, subject_center_x=0.48),
        ContinuityProfile(brightness=0.52, saturation=0.48, subject_center_x=0.5),
    )
    far = continuity_similarity(
        ContinuityProfile(brightness=0.15, saturation=0.1, subject_center_x=0.1),
        ContinuityProfile(brightness=0.9, saturation=0.95, subject_center_x=0.9),
    )

    assert close > 0.8
    assert far < 0.4


def test_editorial_critic_repairs_invalid_overlay_and_enforces_contract() -> None:
    speech = _clip("speech-1", "This section includes approved proof")
    analysis = AnalysisBundle(
        media=speech.media,
        transcript=speech.transcript,
        scenes=speech.scenes,
        source_clips=[speech],
    )
    graph = ProductionEditDecisionGraph(
        target_duration_seconds=5,
        selected_duration_seconds=5,
        segments=[
            EditSegment(
                source_asset_id="speech-1",
                source_index=0,
                source_start=0,
                source_end=5,
                output_start=0,
                output_end=5,
                score=0.8,
                confidence=0.9,
                reason="Narration",
                transcript_text="This section includes approved proof",
            )
        ],
        overlays=[
            VisualOverlay(
                source_asset_id="missing",
                source_index=1,
                source_start=0,
                source_end=2,
                output_start=1,
                output_end=3,
                match_score=0.8,
                reason="Invalid fixture",
            )
        ],
    )

    repaired, report = review_and_repair_edit_graph(
        graph,
        analysis,
        {
            "target_duration_seconds": 5,
            "must_include": ["approved proof"],
            "must_avoid": ["emoji"],
        },
    )

    assert report.passed is True
    assert repaired.overlays == []
    assert report.repairs_applied
    assert repaired.critic_report["score"] == report.score


def test_editorial_critic_blocks_prohibited_selected_content() -> None:
    speech = _clip("speech-1", "Add emoji captions here")
    analysis = AnalysisBundle(
        media=speech.media,
        transcript=speech.transcript,
        scenes=speech.scenes,
        source_clips=[speech],
    )
    graph = ProductionEditDecisionGraph(
        target_duration_seconds=5,
        selected_duration_seconds=5,
        segments=[
            EditSegment(
                source_asset_id="speech-1",
                source_start=0,
                source_end=5,
                output_start=0,
                output_end=5,
                score=0.8,
                confidence=0.9,
                reason="Narration",
                transcript_text="Add emoji captions here",
            )
        ],
    )

    _, report = review_and_repair_edit_graph(
        graph,
        analysis,
        {"target_duration_seconds": 5, "must_avoid": ["emoji captions"]},
    )

    assert report.passed is False
    assert any(issue.code == "prohibited_content_selected" for issue in report.issues)
