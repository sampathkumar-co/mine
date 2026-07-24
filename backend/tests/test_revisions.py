from app.director.edit_graph import EditSegment
from app.director.revision_engine import (
    RevisionEditDecisionGraph,
    apply_graph_revision,
    compare_revision_graphs,
)
from app.director.revisions import LockedRange, parse_revision_intent
from app.director.semantic_overlays import VisualOverlay


def _graph() -> RevisionEditDecisionGraph:
    return RevisionEditDecisionGraph(
        version=1,
        target_duration_seconds=8,
        selected_duration_seconds=8,
        segments=[
            EditSegment(
                source_asset_id="speech-1",
                source_index=0,
                source_start=0,
                source_end=4,
                output_start=0,
                output_end=4,
                score=0.9,
                confidence=0.9,
                reason="Hook",
                transcript_text="This is the opening hook",
            ),
            EditSegment(
                source_asset_id="speech-2",
                source_index=1,
                source_start=0,
                source_end=4,
                output_start=4,
                output_end=8,
                score=0.8,
                confidence=0.85,
                reason="Proof",
                transcript_text="Here is the dashboard proof",
            ),
        ],
        overlays=[
            VisualOverlay(
                source_asset_id="proof-1",
                source_index=2,
                source_start=0,
                source_end=2,
                output_start=4.5,
                output_end=6.5,
                match_score=0.8,
                continuity_score=0.7,
                reason="Dashboard evidence",
                matched_terms=["dashboard", "proof"],
            )
        ],
    )


def test_parses_shorter_revision_and_caption_override() -> None:
    intent = parse_revision_intent(
        "Shorten it to 6 seconds and use all caps captions",
        base_duration_seconds=8,
    )

    assert intent.target_duration_seconds == 6
    assert intent.caption_all_caps is True


def test_caption_only_revision_reuses_narration_master() -> None:
    application = apply_graph_revision(
        _graph(),
        "Use all caps captions and make the captions larger",
        next_version=2,
    )

    assert application.graph.render_overrides["caption_all_caps"] is True
    assert application.graph.render_overrides["caption_size_delta"] == 8
    assert application.render_plan.scope == "component_partial"
    assert application.render_plan.reuse_narration_master is True
    assert application.render_plan.changed_components == ["captions"]


def test_remove_broll_only_changes_overlay_component() -> None:
    application = apply_graph_revision(
        _graph(),
        "Remove B-roll",
        next_version=2,
    )

    assert application.graph.overlays == []
    assert application.render_plan.scope == "component_partial"
    assert application.render_plan.changed_components == ["overlays"]


def test_shortening_changes_narration_and_requires_master_render() -> None:
    application = apply_graph_revision(
        _graph(),
        "Shorten the video to 5 seconds",
        next_version=2,
    )

    assert application.graph.selected_duration_seconds == 5
    assert application.render_plan.scope == "full_master"
    assert "narration" in application.render_plan.changed_components
    assert application.render_plan.reuse_narration_master is False


def test_locked_range_preserves_locked_segment_content() -> None:
    application = apply_graph_revision(
        _graph(),
        "Shorten the video to 3 seconds",
        next_version=2,
        locked_ranges=[LockedRange(start=4, end=8, label="Keep proof")],
    )

    assert any(
        segment.source_asset_id == "speech-2"
        for segment in application.graph.segments
    )
    assert application.graph.selected_duration_seconds > 3


def test_compare_identical_graphs_is_metadata_only() -> None:
    graph = _graph()
    comparison = compare_revision_graphs(graph, graph.model_copy())

    assert comparison.scope == "metadata_only"
    assert comparison.changed_components == []
    assert comparison.reuse_narration_master is True
