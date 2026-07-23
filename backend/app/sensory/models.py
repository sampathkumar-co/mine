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


class AnalysisBundle(BaseModel):
    media: dict[str, object]
    transcript: TranscriptResult | None = None
    scenes: list[SceneRange] = Field(default_factory=list)
