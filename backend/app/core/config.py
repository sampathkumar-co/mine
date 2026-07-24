from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DIRECTOR_", extra="ignore")

    app_name: str = "Director OS API"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    public_app_url: str = "http://localhost:3000"
    database_url: str = "postgresql+psycopg://director:director@postgres:5432/director"
    redis_url: str = "redis://redis:6379/0"
    upload_dir: str = "/data/uploads"
    output_dir: str = "/data/outputs"
    max_upload_bytes: int = Field(default=2_147_483_648, ge=1)
    upload_chunk_bytes: int = Field(default=1_048_576, ge=65_536, le=16_777_216)
    resumable_request_bytes: int = Field(default=16_777_216, ge=65_536, le=67_108_864)
    upload_session_hours: int = Field(default=24, ge=1, le=720)

    auth_required: bool = True
    auth_secret: str = Field(default="development-only-change-me", min_length=16)
    auth_previous_secret: str | None = None
    auth_key_id: str = "primary"
    auth_previous_key_id: str | None = None
    refresh_cookie_name: str = "director_refresh"
    csrf_cookie_name: str = "director_csrf"
    auth_cookie_secure: bool = False
    auth_cookie_samesite: str = "strict"
    auth_session_minutes: int = Field(default=15, ge=5, le=1_440)
    access_token_minutes: int = Field(default=15, ge=5, le=1_440)
    refresh_token_days: int = Field(default=30, ge=1, le=365)
    require_verified_email: bool = False
    email_verification_minutes: int = Field(default=1_440, ge=5, le=10_080)
    password_reset_minutes: int = Field(default=60, ge=5, le=1_440)
    invitation_days: int = Field(default=7, ge=1, le=30)
    delivery_link_minutes: int = Field(default=20, ge=1, le=1_440)

    rate_limit_enabled: bool = True
    rate_limit_backend: str = "memory"
    rate_limit_fail_closed: bool = False
    rate_limit_window_seconds: int = Field(default=60, ge=10, le=3_600)
    auth_rate_limit_requests: int = Field(default=20, ge=1, le=10_000)
    mutation_rate_limit_requests: int = Field(default=120, ge=1, le=100_000)
    webhook_rate_limit_requests: int = Field(default=240, ge=1, le=100_000)
    rate_limit_redis_prefix: str = "director:rate-limit"
    trust_proxy_headers: bool = False

    log_level: str = "INFO"
    log_format: str = "json"
    metrics_enabled: bool = True
    metrics_token: str | None = None
    readiness_require_redis: bool = False

    email_provider: str = "database"
    smtp_host: str | None = None
    smtp_port: int = Field(default=587, ge=1, le=65_535)
    smtp_starttls: bool = True
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from_email: str | None = None
    email_body_retention_hours: int = Field(default=24, ge=0, le=720)
    email_body_encryption_key: str | None = None

    object_storage_provider: str = "local"
    object_storage_local_dir: str = "/data/object-storage"
    multipart_part_bytes: int = Field(default=8_388_608, ge=5_242_880, le=67_108_864)
    multipart_presign_minutes: int = Field(default=15, ge=1, le=1_440)
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    s3_endpoint_url: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None

    starter_credits: float = Field(default=100, ge=0, le=1_000_000)
    max_workspaces_per_user: int = Field(default=3, ge=1, le=100)
    starter_grants_per_user: int = Field(default=1, ge=0, le=1)
    billing_enabled: bool = True
    entitlements_enabled: bool = True
    subscriptions_enabled: bool = False
    billing_provider: str = "stripe"
    billing_plans_json: str = (
        '{"starter":{"name":"Starter","description":"Core autonomous production for small teams.",'
        '"price_id":null,"monthly_credits":0,"max_source_clips":8,'
        '"max_target_duration_seconds":180,"max_members":3,"max_tier":1},'
        '"creator":{"name":"Creator","description":"Longer productions, more sources, and advanced Director tiers.",'
        '"price_id":null,"monthly_credits":250,"max_source_clips":16,'
        '"max_target_duration_seconds":600,"max_members":8,"max_tier":3},'
        '"studio":{"name":"Studio","description":"High-volume production and full Director capabilities.",'
        '"price_id":null,"monthly_credits":1200,"max_source_clips":24,'
        '"max_target_duration_seconds":1800,"max_members":25,"max_tier":6}}'
    )
    stripe_secret_key: str | None = None
    stripe_webhook_secret: str | None = None
    stripe_portal_configuration_id: str | None = None
    audit_retention_days: int = Field(default=365, ge=30, le=3_650)
    data_export_retention_hours: int = Field(default=48, ge=1, le=720)
    workspace_deletion_grace_days: int = Field(default=7, ge=1, le=30)
    workspace_deletion_retry_limit: int = Field(default=10, ge=1, le=100)
    run_migrations_on_startup: bool = False
    auto_create_schema: bool = True

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
    max_source_clips: int = Field(default=8, ge=1, le=24)
    duplicate_hash_distance: int = Field(default=6, ge=0, le=24)
    enable_semantic_overlays: bool = True
    semantic_frame_samples: int = Field(default=10, ge=2, le=48)
    max_visual_overlays: int = Field(default=4, ge=0, le=12)
    minimum_overlay_match_score: float = Field(default=0.3, ge=0, le=1)
    require_editorial_critic_pass: bool = True
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
