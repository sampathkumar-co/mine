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
    auto_create_schema: bool = True
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
