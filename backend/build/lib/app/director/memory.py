from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.director.revision_engine import RevisionEditDecisionGraph


class MemorySignal(BaseModel):
    dimension: str
    value: Any
    sentiment: Literal["positive", "negative"] = "positive"
    weight: float = Field(default=1.0, gt=0, le=3)
    source: str
    reason: str
    explicit: bool = False


class DirectorMemoryApplication(BaseModel):
    contract: dict[str, Any]
    captions_enabled: bool | None = None
    max_visual_overlays: int | None = None
    applied: list[dict[str, Any]] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)


ALLOWED_EXPLICIT_PREFERENCES = {
    "captions_enabled": "captions.enabled",
    "caption_all_caps": "captions.all_caps",
    "caption_size": "captions.size",
    "music_enabled": "music.enabled",
    "music_energy": "music.energy",
    "overlay_density": "overlays.density",
    "pace": "pacing.style",
    "transition_style": "transitions.style",
}


def normalise_profile_key(value: str | None) -> str:
    candidate = re.sub(r"[^a-z0-9_-]+", "-", (value or "default").strip().casefold())
    candidate = candidate.strip("-")[:100]
    return candidate or "default"


def _candidate_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _confidence(support: float, opposition: float) -> float:
    total = support + opposition
    if total <= 0:
        return 0.0
    return round(min(0.98, abs(support - opposition) / (total + 0.5)), 3)


def update_memory_state(
    preferences: dict[str, Any] | None,
    negative_preferences: dict[str, Any] | None,
    signals: list[MemorySignal],
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = deepcopy(preferences or {})

    for signal in signals:
        dimension = state.setdefault(signal.dimension, {"candidates": {}})
        candidates = dimension.setdefault("candidates", {})
        key = _candidate_key(signal.value)
        candidate = candidates.setdefault(
            key,
            {
                "value": signal.value,
                "support": 0.0,
                "opposition": 0.0,
                "evidence_count": 0,
                "explicit_count": 0,
                "last_source": signal.source,
                "last_reason": signal.reason,
            },
        )
        if signal.sentiment == "positive":
            candidate["support"] = round(float(candidate.get("support", 0)) + signal.weight, 3)
        else:
            candidate["opposition"] = round(
                float(candidate.get("opposition", 0)) + signal.weight, 3
            )
        candidate["evidence_count"] = int(candidate.get("evidence_count", 0)) + 1
        if signal.explicit:
            candidate["explicit_count"] = int(candidate.get("explicit_count", 0)) + 1
        candidate["last_source"] = signal.source
        candidate["last_reason"] = signal.reason
        candidate["confidence"] = _confidence(
            float(candidate["support"]), float(candidate["opposition"])
        )

    negative: dict[str, Any] = {}
    for dimension_name, dimension in state.items():
        candidates = dimension.get("candidates", {})
        positive_candidates = [
            item
            for item in candidates.values()
            if float(item.get("support", 0)) > float(item.get("opposition", 0))
        ]
        positive_candidates.sort(
            key=lambda item: (
                float(item.get("support", 0)) - float(item.get("opposition", 0)),
                float(item.get("support", 0)),
            ),
            reverse=True,
        )
        if positive_candidates:
            selected = positive_candidates[0]
            dimension["selected"] = selected.get("value")
            dimension["confidence"] = selected.get("confidence", 0.0)
            dimension["evidence_count"] = selected.get("evidence_count", 0)
            dimension["explicit_count"] = selected.get("explicit_count", 0)
            dimension["last_reason"] = selected.get("last_reason")
        else:
            for key in (
                "selected",
                "confidence",
                "evidence_count",
                "explicit_count",
                "last_reason",
            ):
                dimension.pop(key, None)

        disliked = []
        for item in candidates.values():
            opposition = float(item.get("opposition", 0))
            support = float(item.get("support", 0))
            confidence = _confidence(support, opposition)
            if opposition > support and confidence >= 0.35:
                disliked.append(
                    {
                        "value": item.get("value"),
                        "confidence": confidence,
                        "support": support,
                        "opposition": opposition,
                        "evidence_count": item.get("evidence_count", 0),
                        "last_reason": item.get("last_reason"),
                    }
                )
        disliked.sort(key=lambda item: (item["confidence"], item["opposition"]), reverse=True)
        if disliked:
            negative[dimension_name] = disliked

    return state, negative


def _phrase_signal(
    dimension: str,
    value: Any,
    *,
    source: str,
    reason: str,
    weight: float,
) -> MemorySignal:
    return MemorySignal(
        dimension=dimension,
        value=value,
        sentiment="positive",
        weight=weight,
        source=source,
        reason=reason,
        explicit=True,
    )


def extract_text_memory_signals(
    text: str | None,
    *,
    source: str = "feedback_text",
    weight: float = 1.8,
) -> list[MemorySignal]:
    if not text:
        return []
    normalized = " ".join(text.casefold().split())
    signals: list[MemorySignal] = []

    rules: list[tuple[tuple[str, ...], str, Any, str]] = [
        (("no captions", "remove captions", "hide captions", "hate captions"), "captions.enabled", False, "User prefers videos without captions."),
        (("keep captions", "show captions", "add captions", "love captions"), "captions.enabled", True, "User prefers captions."),
        (("all caps captions", "uppercase captions"), "captions.all_caps", True, "User prefers uppercase captions."),
        (("sentence case captions", "normal case captions", "not all caps"), "captions.all_caps", False, "User prefers normal-case captions."),
        (("larger captions", "bigger captions", "captions larger", "captions bigger"), "captions.size", "large", "User prefers larger captions."),
        (("smaller captions", "captions smaller"), "captions.size", "small", "User prefers smaller captions."),
        (("no music", "remove music", "mute music", "hate music"), "music.enabled", False, "User prefers no background music."),
        (("keep music", "add music", "love the music"), "music.enabled", True, "User prefers background music."),
        (("calm music", "softer music"), "music.energy", "calm", "User prefers calmer music."),
        (("energetic music", "more energetic music", "upbeat music"), "music.energy", "energetic", "User prefers energetic music."),
        (("less b-roll", "less broll", "fewer overlays", "minimal b-roll"), "overlays.density", "sparse", "User prefers fewer visual overlays."),
        (("more b-roll", "more broll", "more overlays", "more cutaways"), "overlays.density", "dense", "User prefers more visual overlays."),
        (("faster pace", "make it faster", "tighter pacing", "quick cuts"), "pacing.style", "fast", "User prefers faster pacing."),
        (("slower pace", "slow it down", "more breathing room"), "pacing.style", "slow", "User prefers slower pacing."),
        (("hard cuts", "clean cuts", "no dissolves"), "transitions.style", "hard_cuts", "User prefers hard cuts."),
        (("soft transitions", "more dissolves", "gentle transitions"), "transitions.style", "soft", "User prefers softer transitions."),
    ]
    for phrases, dimension, value, reason in rules:
        if any(phrase in normalized for phrase in phrases):
            signals.append(
                _phrase_signal(
                    dimension,
                    value,
                    source=source,
                    reason=reason,
                    weight=weight,
                )
            )
    return signals


def explicit_preference_signals(preferences: dict[str, Any] | None) -> list[MemorySignal]:
    signals: list[MemorySignal] = []
    for key, value in (preferences or {}).items():
        dimension = ALLOWED_EXPLICIT_PREFERENCES.get(key)
        if dimension is None:
            continue
        signals.append(
            MemorySignal(
                dimension=dimension,
                value=value,
                sentiment="positive",
                weight=2.3,
                source="explicit_preference",
                reason=f"User explicitly set {key} to {value!r}.",
                explicit=True,
            )
        )
    return signals


def _overlay_density(graph: RevisionEditDecisionGraph) -> str:
    duration_minutes = max(graph.selected_duration_seconds / 60, 1 / 60)
    rate = len(graph.overlays) / duration_minutes
    if not graph.overlays:
        return "none"
    if rate < 3:
        return "sparse"
    if rate < 7:
        return "balanced"
    return "dense"


def _pacing_style(graph: RevisionEditDecisionGraph) -> str:
    if not graph.segments:
        return "balanced"
    average = graph.selected_duration_seconds / len(graph.segments)
    if average < 2.6:
        return "fast"
    if average > 5.5:
        return "slow"
    return "balanced"


def _transition_style(graph: RevisionEditDecisionGraph) -> str:
    transitions = [
        str(item.transition).casefold() for item in graph.continuity_decisions
    ] or [str(item.transition).casefold() for item in graph.segments]
    soft = sum("dissolve" in item or "fade" in item for item in transitions)
    return "soft" if soft > len(transitions) / 2 else "hard_cuts"


def graph_memory_signals(
    graph: RevisionEditDecisionGraph,
    production_style: dict[str, Any] | None,
    *,
    sentiment: Literal["positive", "negative"],
    weight: float,
    source: str,
    changed_components: set[str] | None = None,
) -> list[MemorySignal]:
    changed = changed_components or {"narration", "overlays", "captions", "music", "transitions"}
    style = production_style or {}
    caption_style = style.get("caption") if isinstance(style.get("caption"), dict) else {}
    music_style = style.get("music") if isinstance(style.get("music"), dict) else {}
    overrides = graph.render_overrides
    signals: list[MemorySignal] = []

    def add(dimension: str, value: Any, reason: str) -> None:
        signals.append(
            MemorySignal(
                dimension=dimension,
                value=value,
                sentiment=sentiment,
                weight=weight,
                source=source,
                reason=reason,
            )
        )

    if "captions" in changed:
        enabled = bool(overrides.get("captions_enabled", True))
        add("captions.enabled", enabled, "Caption visibility in the evaluated edit.")
        all_caps = bool(overrides.get("caption_all_caps", caption_style.get("all_caps", False)))
        add("captions.all_caps", all_caps, "Caption casing in the evaluated edit.")
        font_size = int(caption_style.get("font_size", 72) or 72) + int(
            overrides.get("caption_size_delta", 0) or 0
        )
        size = "large" if font_size >= 80 else "small" if font_size <= 64 else "medium"
        add("captions.size", size, "Caption size in the evaluated edit.")

    if "music" in changed:
        enabled = bool(overrides.get("music_enabled", music_style.get("enabled", True)))
        add("music.enabled", enabled, "Music presence in the evaluated edit.")
        energy = float(music_style.get("desired_energy", 0.5) or 0.5)
        energy_name = "calm" if energy < 0.4 else "energetic" if energy > 0.68 else "balanced"
        add("music.energy", energy_name, "Music energy in the evaluated edit.")

    if "overlays" in changed:
        add("overlays.density", _overlay_density(graph), "Overlay density in the evaluated edit.")
    if "narration" in changed:
        add("pacing.style", _pacing_style(graph), "Narration pacing in the evaluated edit.")
    if "transitions" in changed or "narration" in changed:
        add("transitions.style", _transition_style(graph), "Transition style in the evaluated edit.")
    return signals


def calculate_performance_score(
    metrics: dict[str, Any],
    *,
    video_duration_seconds: float,
) -> float:
    views = max(1.0, float(metrics.get("views") or 0))
    completion = min(1.0, max(0.0, float(metrics.get("completion_rate") or 0)))
    average_watch = max(0.0, float(metrics.get("average_watch_seconds") or 0))
    watch_ratio = min(1.0, average_watch / max(video_duration_seconds, 1.0))
    engagement = (
        float(metrics.get("likes") or 0)
        + float(metrics.get("comments") or 0) * 1.5
        + float(metrics.get("shares") or 0) * 2.5
        + float(metrics.get("saves") or 0) * 2.5
        + float(metrics.get("clicks") or 0) * 1.5
        + float(metrics.get("conversions") or 0) * 4
    ) / views
    engagement_score = min(1.0, max(0.0, engagement / 0.12))
    score = completion * 0.45 + watch_ratio * 0.3 + engagement_score * 0.25
    return round(min(1.0, max(0.0, score)), 3)


def performance_memory_signals(
    graph: RevisionEditDecisionGraph,
    production_style: dict[str, Any] | None,
    *,
    score: float,
) -> list[MemorySignal]:
    if 0.3 < score < 0.68:
        return []
    sentiment: Literal["positive", "negative"] = "positive" if score >= 0.68 else "negative"
    weight = 0.4 if sentiment == "positive" else 0.25
    return graph_memory_signals(
        graph,
        production_style,
        sentiment=sentiment,
        weight=weight,
        source="performance_signal",
    )


def _eligible_preference(entry: dict[str, Any]) -> bool:
    confidence = float(entry.get("confidence", 0))
    evidence_count = int(entry.get("evidence_count", 0))
    explicit_count = int(entry.get("explicit_count", 0))
    return confidence >= 0.67 and (evidence_count >= 2 or explicit_count >= 1)


def apply_director_memory(
    contract: dict[str, Any],
    preferences: dict[str, Any] | None,
) -> DirectorMemoryApplication:
    updated = deepcopy(contract)
    if not bool(updated.get("use_director_memory", True)):
        return DirectorMemoryApplication(contract=updated, skipped=["Director memory disabled by contract."])

    brand_rules = updated.get("brand_rules")
    if not isinstance(brand_rules, dict):
        brand_rules = {}
    else:
        brand_rules = dict(brand_rules)
    updated["brand_rules"] = brand_rules
    application = DirectorMemoryApplication(contract=updated)

    def selected(dimension: str) -> tuple[Any, dict[str, Any]] | None:
        entry = (preferences or {}).get(dimension)
        if not isinstance(entry, dict) or "selected" not in entry or not _eligible_preference(entry):
            return None
        return entry["selected"], entry

    mappings = {
        "captions.all_caps": ("caption_all_caps", lambda value: bool(value)),
        "captions.size": (
            "caption_font_size",
            lambda value: {"small": 60, "medium": 72, "large": 84}.get(str(value), 72),
        ),
        "music.enabled": ("music_enabled", lambda value: bool(value)),
        "music.energy": ("music_energy", lambda value: str(value)),
    }
    for dimension, (brand_key, transform) in mappings.items():
        match = selected(dimension)
        if match is None:
            continue
        value, entry = match
        if brand_key in brand_rules:
            application.skipped.append(
                f"Explicit brand rule {brand_key} overrides remembered {dimension}."
            )
            continue
        resolved = transform(value)
        brand_rules[brand_key] = resolved
        application.applied.append(
            {
                "dimension": dimension,
                "value": value,
                "render_value": resolved,
                "confidence": entry.get("confidence", 0),
            }
        )

    caption_match = selected("captions.enabled")
    if caption_match is not None:
        value, entry = caption_match
        if "captions_enabled" in brand_rules:
            application.skipped.append(
                "Explicit brand rule captions_enabled overrides remembered caption visibility."
            )
            application.captions_enabled = bool(brand_rules["captions_enabled"])
        else:
            application.captions_enabled = bool(value)
            application.applied.append(
                {
                    "dimension": "captions.enabled",
                    "value": bool(value),
                    "confidence": entry.get("confidence", 0),
                }
            )

    overlay_match = selected("overlays.density")
    if overlay_match is not None:
        value, entry = overlay_match
        application.max_visual_overlays = {
            "none": 0,
            "sparse": 2,
            "balanced": 4,
            "dense": 6,
        }.get(str(value))
        if application.max_visual_overlays is not None:
            application.applied.append(
                {
                    "dimension": "overlays.density",
                    "value": value,
                    "render_value": application.max_visual_overlays,
                    "confidence": entry.get("confidence", 0),
                }
            )
    return application
