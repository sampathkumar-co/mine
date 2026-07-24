from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.database import Base
from app.director.memory import (
    MemorySignal,
    apply_director_memory,
    calculate_performance_score,
    explicit_preference_signals,
    extract_text_memory_signals,
    update_memory_state,
)
from app.models.memory import DirectorMemoryProfile
from app.models.project import Project


def test_repeated_feedback_becomes_eligible_memory() -> None:
    signal = MemorySignal(
        dimension="music.enabled",
        value=False,
        sentiment="positive",
        weight=1.0,
        source="accepted_revision",
        reason="User accepted an edit without music.",
    )
    preferences, negative = update_memory_state({}, {}, [signal, signal])

    assert negative == {}
    assert preferences["music.enabled"]["selected"] is False
    assert preferences["music.enabled"]["evidence_count"] == 2
    assert preferences["music.enabled"]["confidence"] >= 0.67

    application = apply_director_memory(
        {"objective": "Create a professional reel", "brand_rules": {}},
        preferences,
    )
    assert application.contract["brand_rules"]["music_enabled"] is False


def test_explicit_preference_applies_after_one_event() -> None:
    signals = explicit_preference_signals(
        {
            "caption_all_caps": True,
            "caption_size": "large",
            "overlay_density": "sparse",
        }
    )
    preferences, _ = update_memory_state({}, {}, signals)
    application = apply_director_memory(
        {"objective": "Create a launch reel", "brand_rules": {}},
        preferences,
    )

    assert application.contract["brand_rules"]["caption_all_caps"] is True
    assert application.contract["brand_rules"]["caption_font_size"] == 84
    assert application.max_visual_overlays == 2


def test_explicit_contract_overrides_memory() -> None:
    preferences, _ = update_memory_state(
        {},
        {},
        explicit_preference_signals(
            {
                "caption_all_caps": True,
                "music_energy": "energetic",
            }
        ),
    )
    application = apply_director_memory(
        {
            "objective": "Create a calm property reel",
            "brand_rules": {
                "caption_all_caps": False,
                "music_energy": "calm",
            },
        },
        preferences,
    )

    assert application.contract["brand_rules"]["caption_all_caps"] is False
    assert application.contract["brand_rules"]["music_energy"] == "calm"
    assert len(application.skipped) == 2


def test_negative_taste_memory_tracks_rejected_values() -> None:
    rejected = MemorySignal(
        dimension="transitions.style",
        value="soft",
        sentiment="negative",
        weight=1.5,
        source="rejected_revision",
        reason="User rejected soft dissolves.",
    )
    preferences, negative = update_memory_state({}, {}, [rejected])

    assert "selected" not in preferences["transitions.style"]
    assert negative["transitions.style"][0]["value"] == "soft"
    assert negative["transitions.style"][0]["confidence"] >= 0.35


def test_feedback_text_extracts_desired_preferences() -> None:
    signals = extract_text_memory_signals(
        "Please use larger captions, no music, and less B-roll next time."
    )
    values = {(signal.dimension, signal.value) for signal in signals}

    assert ("captions.size", "large") in values
    assert ("music.enabled", False) in values
    assert ("overlays.density", "sparse") in values


def test_performance_score_is_bounded() -> None:
    strong = calculate_performance_score(
        {
            "views": 1_000,
            "average_watch_seconds": 30,
            "completion_rate": 0.9,
            "likes": 200,
            "comments": 30,
            "shares": 100,
            "saves": 120,
            "clicks": 80,
            "conversions": 20,
        },
        video_duration_seconds=30,
    )
    weak = calculate_performance_score(
        {"views": 100, "average_watch_seconds": 0, "completion_rate": 0},
        video_duration_seconds=30,
    )

    assert 0 <= weak <= strong <= 1
    assert strong >= 0.68


def test_project_insert_compiles_selected_memory_profile() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    user_id = uuid4()
    preferences, negative = update_memory_state(
        {},
        {},
        explicit_preference_signals(
            {
                "caption_all_caps": True,
                "music_enabled": False,
                "overlay_density": "sparse",
            }
        ),
    )

    with Session(engine) as db:
        db.add(
            DirectorMemoryProfile(
                user_id=user_id,
                profile_key="founder-brand",
                preferences=preferences,
                negative_preferences=negative,
            )
        )
        db.commit()

        project = Project(
            user_id=user_id,
            contract={
                "objective": "Create a founder update",
                "director_profile_key": "founder-brand",
                "use_director_memory": True,
                "brand_rules": {},
            },
        )
        db.add(project)
        db.commit()
        db.refresh(project)

    assert project.contract["brand_rules"]["caption_all_caps"] is True
    assert project.contract["brand_rules"]["music_enabled"] is False
    assert project.contract["brand_rules"]["max_visual_overlays"] == 2
    assert project.contract["_director_memory_application"]["profile_key"] == "founder-brand"
