from enum import StrEnum


class ProjectStatus(StrEnum):
    CREATED = "created"
    UPLOADING = "uploading"
    READY_TO_QUEUE = "ready_to_queue"
    QUEUED = "queued"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    RENDERING = "rendering"
    QUALITY_CHECK = "quality_check"
    READY = "ready"
    FAILED = "failed"


class AssetKind(StrEnum):
    SOURCE_VIDEO = "source_video"
    REFERENCE_VIDEO = "reference_video"
    LOGO = "logo"
    BRAND_ASSET = "brand_asset"
    MUSIC = "music"
