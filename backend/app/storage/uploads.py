import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from app.core.config import Settings
from app.core.enums import AssetKind

VIDEO_KINDS = {
    AssetKind.SOURCE_VIDEO,
    AssetKind.PICKUP_VIDEO,
    AssetKind.REFERENCE_VIDEO,
}


class UploadTooLargeError(ValueError):
    pass


class UnsupportedAssetError(ValueError):
    pass


class EmptyUploadError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StoredUpload:
    original_filename: str
    stored_filename: str
    content_type: str
    size_bytes: int
    sha256: str
    storage_path: str


def _validate_content_type(kind: AssetKind, content_type: str) -> None:
    normalized = content_type.casefold()
    if kind in VIDEO_KINDS and not normalized.startswith("video/"):
        raise UnsupportedAssetError(f"{kind.value} requires a video content type")
    if kind == AssetKind.LOGO and not normalized.startswith("image/"):
        raise UnsupportedAssetError("logo requires an image content type")
    if kind == AssetKind.BRAND_ASSET and not (
        normalized.startswith("image/") or normalized == "application/pdf"
    ):
        raise UnsupportedAssetError("brand_asset requires an image or PDF content type")
    if kind == AssetKind.MUSIC and not normalized.startswith("audio/"):
        raise UnsupportedAssetError("music requires an audio content type")


async def store_upload(
    upload: UploadFile,
    *,
    project_id: UUID,
    kind: AssetKind,
    settings: Settings,
) -> StoredUpload:
    original_filename = Path(upload.filename or "upload").name[:512]
    content_type = upload.content_type or "application/octet-stream"
    destination: Path | None = None
    partial: Path | None = None

    try:
        _validate_content_type(kind, content_type)
        suffix = Path(original_filename).suffix.lower()
        if len(suffix) > 12 or not suffix.replace(".", "").isalnum():
            suffix = ""

        stored_filename = f"{uuid4().hex}{suffix}"
        project_dir = Path(settings.upload_dir) / str(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        destination = project_dir / stored_filename
        partial = destination.with_suffix(f"{destination.suffix}.part")

        digest = hashlib.sha256()
        size_bytes = 0
        with partial.open("wb") as output:
            while chunk := await upload.read(settings.upload_chunk_bytes):
                size_bytes += len(chunk)
                if size_bytes > settings.max_upload_bytes:
                    raise UploadTooLargeError(
                        f"Upload exceeds {settings.max_upload_bytes} bytes"
                    )
                digest.update(chunk)
                output.write(chunk)

        if size_bytes == 0:
            raise EmptyUploadError("Upload is empty")
        os.replace(partial, destination)
    except Exception:
        if partial is not None:
            partial.unlink(missing_ok=True)
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    return StoredUpload(
        original_filename=original_filename,
        stored_filename=stored_filename,
        content_type=content_type,
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
        storage_path=str(destination),
    )
