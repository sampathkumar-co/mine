from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DIRECTOR_", extra="ignore")

    app_name: str = "Director OS API"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://director:director@postgres:5432/director"
    redis_url: str = "redis://redis:6379/0"
    upload_dir: str = "/data/uploads"
    output_dir: str = "/data/outputs"
    max_upload_bytes: int = Field(default=2_147_483_648, ge=1)
    upload_chunk_bytes: int = Field(default=1_048_576, ge=65_536, le=16_777_216)
    render_timeout_seconds: int = Field(default=3_600, ge=30)
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    scene_detection_threshold: float = Field(default=0.35, ge=0.05, le=0.95)
    minimum_scene_seconds: float = Field(default=0.5, ge=0.1, le=10)
    transcription_provider: str = "openai"
    transcription_model: str = "whisper-1"
    transcription_timeout_seconds: int = Field(default=600, ge=30, le=3_600)
    require_transcription: bool = False
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    enable_word_cleanup: bool = True
    silence_threshold_seconds: float = Field(default=0.55, ge=0.2, le=3)
    speech_padding_seconds: float = Field(default=0.08, ge=0, le=0.5)
    enable_captions: bool = True
    caption_max_words: int = Field(default=5, ge=1, le=10)
    caption_margin_vertical: int = Field(default=260, ge=80, le=700)
    enable_subject_framing: bool = True
    subject_frame_samples: int = Field(default=24, ge=4, le=120)
    enable_reference_style: bool = True
    reference_frame_samples: int = Field(default=24, ge=4, le=120)
    enable_music: bool = True
    music_default_volume: float = Field(default=0.16, ge=0, le=0.5)
    music_ducking_threshold: float = Field(default=0.035, ge=0.005, le=0.2)
    music_fade_seconds: float = Field(default=0.8, ge=0, le=5)
    auto_create_schema: bool = True
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
