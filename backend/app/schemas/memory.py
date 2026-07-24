from datetime import datetime
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DirectorMemoryProfileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: UUID
    profile_key: str
    version: int
    evidence_count: int
    performance_sample_count: int
    preferences: dict[str, Any]
    negative_preferences: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ProjectFeedbackCreate(BaseModel):
    revision_version: int | None = Field(default=None, ge=1)
    verdict: Literal["accepted", "rejected", "needs_changes"]
    rating: int | None = Field(default=None, ge=1, le=5)
    feedback_text: str | None = Field(default=None, max_length=5_000)
    explicit_preferences: dict[str, Any] = Field(default_factory=dict)
    dimension_ratings: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def contains_feedback_evidence(self) -> Self:
        if (
            self.rating is None
            and not (self.feedback_text or "").strip()
            and not self.explicit_preferences
            and not self.dimension_ratings
            and self.verdict == "needs_changes"
        ):
            raise ValueError("needs_changes feedback requires a rating, text, or preference")
        invalid = [value for value in self.dimension_ratings.values() if value < 1 or value > 5]
        if invalid:
            raise ValueError("dimension ratings must be between 1 and 5")
        return self


class ProjectFeedbackRecorded(BaseModel):
    evidence_id: UUID
    project_id: UUID
    revision_version: int
    profile_key: str
    profile_version: int
    signal_count: int
    updated_dimensions: list[str]
    message: str = "Feedback recorded and Director Memory updated."


class DirectorMemoryPreferenceUpdate(BaseModel):
    preferences: dict[str, Any] = Field(default_factory=dict)
    avoid_preferences: dict[str, Any] = Field(default_factory=dict)
    note: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def contains_preference(self) -> Self:
        if not self.preferences and not self.avoid_preferences:
            raise ValueError("At least one preferred or avoided setting is required")
        return self


class DirectorMemoryPreferenceUpdated(BaseModel):
    user_id: UUID
    profile_key: str
    profile_version: int
    signal_count: int
    updated_dimensions: list[str]


class ProjectPerformanceCreate(BaseModel):
    revision_version: int | None = Field(default=None, ge=1)
    platform: str = Field(min_length=2, max_length=80)
    impressions: int = Field(default=0, ge=0)
    views: int = Field(default=0, ge=0)
    average_watch_seconds: float = Field(default=0, ge=0)
    completion_rate: float = Field(default=0, ge=0, le=1)
    likes: int = Field(default=0, ge=0)
    comments: int = Field(default=0, ge=0)
    shares: int = Field(default=0, ge=0)
    saves: int = Field(default=0, ge=0)
    clicks: int = Field(default=0, ge=0)
    conversions: int = Field(default=0, ge=0)
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def views_do_not_exceed_impressions(self) -> Self:
        if self.impressions and self.views > self.impressions:
            raise ValueError("views cannot exceed impressions")
        return self

    def metrics_payload(self) -> dict[str, Any]:
        return {
            "impressions": self.impressions,
            "views": self.views,
            "average_watch_seconds": self.average_watch_seconds,
            "completion_rate": self.completion_rate,
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "saves": self.saves,
            "clicks": self.clicks,
            "conversions": self.conversions,
            "metadata": self.metadata,
        }


class ProjectPerformanceRecorded(BaseModel):
    signal_id: UUID
    project_id: UUID
    revision_version: int
    profile_key: str
    normalized_score: float
    memory_signal_count: int
    profile_version: int
    message: str = "Performance signal recorded as low-weight Director Memory evidence."
