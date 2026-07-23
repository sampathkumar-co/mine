from pathlib import Path

from app.director.edit_graph import EditDecisionGraph, EditSegment
from app.director.style import CaptionStyle, compile_production_style, hex_to_ass
from app.rendering.captions import write_ass_captions
from app.sensory.models import (
    MusicProfile,
    ReferenceStyleProfile,
    SceneRange,
    TranscriptResult,
    TranscriptWord,
)
from app.sensory.music import choose_music
from app.sensory.reference import build_reference_profile


def test_reference_profile_classifies_fast_and_slow_pacing() -> None:
    fast = build_reference_profile(
        asset_id="ref-fast",
        duration_seconds=20,
        scenes=[SceneRange(start=index, end=index + 1) for index in range(10)],
        brightness=0.6,
        saturation=0.7,
        motion_energy=0.8,
        sampled_frames=10,
    )
    slow = build_reference_profile(
        asset_id="ref-slow",
        duration_seconds=20,
        scenes=[SceneRange(start=0, end=10), SceneRange(start=10, end=20)],
        brightness=0.4,
        saturation=0.3,
        motion_energy=0.2,
        sampled_frames=8,
    )

    assert fast.pace == "fast"
    assert fast.cuts_per_minute == 27
    assert slow.pace == "slow"
    assert slow.average_shot_seconds == 10


def test_brand_rules_compile_to_safe_bounded_style() -> None:
    reference = ReferenceStyleProfile(pace="slow", saturation=0.25, brightness=0.4)
    style = compile_production_style(
        {
            "objective": "Create a calm luxury property reel",
            "brand_rules": {
                "caption_font": "Inter; DROP TABLE",
                "caption_primary_color": "#12ab34",
                "caption_accent_color": "not-a-color",
                "caption_position": "upper",
                "caption_all_caps": "yes",
                "caption_font_size": 999,
                "music_energy": "calm",
                "music_volume": 4,
            },
        },
        reference,
    )

    assert style.caption.font_name == "Inter DROP TABLE"
    assert style.caption.primary_color == "#12AB34"
    assert style.caption.accent_color == "#FFE700"
    assert style.caption.position == "upper"
    assert style.caption.all_caps is True
    assert style.caption.font_size == 120
    assert style.caption.animation == "fade"
    assert style.music.desired_energy == 0.25
    assert style.music.volume == 0.5
    assert style.source == ["tier1_defaults", "reference_style", "brand_rules"]
    assert hex_to_ass("#12AB34") == "&H0034AB12"


def test_music_selection_matches_desired_energy() -> None:
    profiles = [
        MusicProfile(asset_id="calm", filename="calm-piano.mp3", energy=0.2, duration_seconds=40),
        MusicProfile(asset_id="balanced", filename="balanced.mp3", energy=0.52, duration_seconds=20),
        MusicProfile(asset_id="fast", filename="upbeat-fast.mp3", energy=0.9, duration_seconds=60),
    ]

    selected = choose_music(profiles, desired_energy=0.8)

    assert selected is not None
    assert selected.asset_id == "fast"
    assert choose_music([], desired_energy=0.5) is None


def test_brand_caption_style_is_written_to_ass(tmp_path: Path) -> None:
    graph = EditDecisionGraph(
        target_duration_seconds=2,
        selected_duration_seconds=2,
        segments=[
            EditSegment(
                source_start=0,
                source_end=2,
                output_start=0,
                output_end=2,
                score=0.9,
                confidence=0.9,
                reason="clear hook",
                transcript_text="hello brand",
            )
        ],
    )
    transcript = TranscriptResult(
        text="hello brand",
        provider="fixture",
        model="fixture",
        words=[
            TranscriptWord(word="hello", start=0.1, end=0.5),
            TranscriptWord(word="brand", start=0.6, end=1.0),
        ],
    )
    style = CaptionStyle(
        font_name="Inter",
        primary_color="#12AB34",
        position="upper",
        all_caps=True,
        animation="none",
    )
    output = tmp_path / "brand.ass"

    count = write_ass_captions(output, graph, transcript, style=style)
    content = output.read_text(encoding="utf-8")

    assert count == 1
    assert "Style: Director,Inter" in content
    assert "&H0034AB12" in content
    assert r"{\an8}HELLO BRAND" in content
