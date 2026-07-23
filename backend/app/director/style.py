from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.sensory.models import ReferenceStyleProfile

HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
SAFE_FONT = re.compile(r"[^A-Za-z0-9 _-]")


class CaptionStyle(BaseModel):
    font_name: str = "Arial"
    font_size: int = Field(default=72, ge=36, le=120)
    primary_color: str = "#FFFFFF"
    accent_color: str = "#FFE700"
    outline_color: str = "#101010"
    position: Literal["lower", "center", "upper"] = "lower"
    margin_vertical: int = Field(default=260, ge=80, le=700)
    max_words: int = Field(default=5, ge=1, le=10)
    all_caps: bool = False
    animation: Literal["pop", "fade", "none"] = "pop"


class VisualStyle(BaseModel):
    contrast: float = Field(default=1.03, ge=0.8, le=1.3)
    saturation: float = Field(default=1.04, ge=0.6, le=1.5)
    brightness: float = Field(default=0.005, ge=-0.15, le=0.15)


class MusicStyle(BaseModel):
    enabled: bool = True
    desired_energy: float = Field(default=0.5, ge=0, le=1)
    volume: float = Field(default=0.16, ge=0, le=0.5)
    ducking_threshold: float = Field(default=0.035, ge=0.005, le=0.2)
    fade_seconds: float = Field(default=0.8, ge=0, le=5)


class ProductionStyle(BaseModel):
    caption: CaptionStyle = Field(default_factory=CaptionStyle)
    visual: VisualStyle = Field(default_factory=VisualStyle)
    music: MusicStyle = Field(default_factory=MusicStyle)
    reference_pace: str = "balanced"
    source: list[str] = Field(default_factory=lambda: ["tier1_defaults"])


def _number(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, parsed))


def _integer(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, parsed))


def _color(value: Any, default: str) -> str:
    candidate = str(value or "").strip()
    return candidate.upper() if HEX_COLOR.fullmatch(candidate) else default


def _font(value: Any, default: str = "Arial") -> str:
    candidate = SAFE_FONT.sub("", str(value or "")).strip()[:80]
    return candidate or default


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return default


def _desired_music_energy(objective: str, brand_rules: dict[str, Any]) -> float:
    explicit = str(brand_rules.get("music_energy", "")).strip().casefold()
    mapping = {"calm": 0.25, "balanced": 0.5, "energetic": 0.82}
    if explicit in mapping:
        return mapping[explicit]

    text = objective.casefold()
    if any(term in text for term in {"urgent", "launch", "fitness", "motivation", "exciting"}):
        return 0.78
    if any(term in text for term in {"luxury", "trust", "calm", "professional", "medical"}):
        return 0.32
    return 0.5


def compile_production_style(
    contract: dict[str, Any],
    reference: ReferenceStyleProfile | None,
    *,
    default_caption_margin: int = 260,
    default_music_volume: float = 0.16,
    default_ducking_threshold: float = 0.035,
    default_music_fade_seconds: float = 0.8,
) -> ProductionStyle:
    brand_rules = contract.get("brand_rules") or {}
    if not isinstance(brand_rules, dict):
        brand_rules = {}

    caption_position = str(brand_rules.get("caption_position", "lower")).casefold()
    if caption_position not in {"lower", "center", "upper"}:
        caption_position = "lower"

    animation = str(brand_rules.get("caption_animation", "")).casefold()
    if animation not in {"pop", "fade", "none"}:
        animation = "pop" if reference is None or reference.pace != "slow" else "fade"

    caption = CaptionStyle(
        font_name=_font(brand_rules.get("caption_font")),
        font_size=_integer(brand_rules.get("caption_font_size"), 72, 36, 120),
        primary_color=_color(brand_rules.get("caption_primary_color"), "#FFFFFF"),
        accent_color=_color(brand_rules.get("caption_accent_color"), "#FFE700"),
        outline_color=_color(brand_rules.get("caption_outline_color"), "#101010"),
        position=caption_position,
        margin_vertical=_integer(
            brand_rules.get("caption_margin_vertical"), default_caption_margin, 80, 700
        ),
        max_words=_integer(brand_rules.get("caption_max_words"), 5, 1, 10),
        all_caps=_bool(brand_rules.get("caption_all_caps"), False),
        animation=animation,
    )

    reference_saturation = 0.5 if reference is None else reference.saturation
    reference_brightness = 0.5 if reference is None else reference.brightness
    visual = VisualStyle(
        contrast=_number(brand_rules.get("visual_contrast"), 1.03, 0.8, 1.3),
        saturation=_number(
            brand_rules.get("visual_saturation"),
            0.92 + reference_saturation * 0.24,
            0.6,
            1.5,
        ),
        brightness=_number(
            brand_rules.get("visual_brightness"),
            (reference_brightness - 0.5) * 0.08,
            -0.15,
            0.15,
        ),
    )

    objective = str(contract.get("objective", ""))
    music = MusicStyle(
        enabled=_bool(brand_rules.get("music_enabled"), True),
        desired_energy=_desired_music_energy(objective, brand_rules),
        volume=_number(brand_rules.get("music_volume"), default_music_volume, 0, 0.5),
        ducking_threshold=_number(
            brand_rules.get("music_ducking_threshold"),
            default_ducking_threshold,
            0.005,
            0.2,
        ),
        fade_seconds=_number(
            brand_rules.get("music_fade_seconds"), default_music_fade_seconds, 0, 5
        ),
    )

    source = ["tier1_defaults"]
    if reference is not None:
        source.append("reference_style")
    if brand_rules:
        source.append("brand_rules")

    return ProductionStyle(
        caption=caption,
        visual=visual,
        music=music,
        reference_pace=reference.pace if reference else "balanced",
        source=source,
    )


def hex_to_ass(color: str, *, alpha: str = "00") -> str:
    normalized = _color(color, "#FFFFFF")[1:]
    red, green, blue = normalized[0:2], normalized[2:4], normalized[4:6]
    return f"&H{alpha}{blue}{green}{red}"
