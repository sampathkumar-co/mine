from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptWord(BaseModel):
    word: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class TranscriptSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    confidence: float = Field(default=0.8, ge=0, le=1)


class TranscriptResult(BaseModel):
    text: str
    language: str | None = None
    duration_seconds: float = Field(default=0, ge=0)
    provider: str
    model: str
    words: list[TranscriptWord] = Field(default_factory=list)
    segments: list[TranscriptSegment] = Field(default_factory=list)


class SceneRange(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    confidence: float = Field(default=0.7, ge=0, le=1)


class SubjectFraming(BaseModel):
    normalized_center_x: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=0, ge=0, le=1)
    detector: str = "center_fallback"
    sampled_frames: int = Field(default=0, ge=0)
    detected_frames: int = Field(default=0, ge=0)


class ReferenceStyleProfile(BaseModel):
    source_asset_id: str | None = None
    duration_seconds: float = Field(default=0, ge=0)
    average_shot_seconds: float = Field(default=0, ge=0)
    cuts_per_minute: float = Field(default=0, ge=0)
    brightness: float = Field(default=0.5, ge=0, le=1)
    saturation: float = Field(default=0.5, ge=0, le=1)
    motion_energy: float = Field(default=0.3, ge=0, le=1)
    pace: str = "balanced"
    sampled_frames: int = Field(default=0, ge=0)


class MusicProfile(BaseModel):
    asset_id: str
    filename: str
    duration_seconds: float = Field(default=0, ge=0)
    mean_volume_db: float = -24
    peak_volume_db: float = -6
    energy: float = Field(default=0.5, ge=0, le=1)


class AnalysisBundle(BaseModel):
    media: dict[str, object]
    transcript: TranscriptResult | None = None
    scenes: list[SceneRange] = Field(default_factory=list)
    subject_framing: SubjectFraming = Field(default_factory=SubjectFraming)
    reference_style: ReferenceStyleProfile | None = None
    music_profiles: list[MusicProfile] = Field(default_factory=list)
    production_style: dict[str, object] = Field(default_factory=dict)
