from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.enums import AssetKind, ProjectStatus
from app.core.time import is_expired_at
from app.models.operations import (
    MultipartUpload,
    MultipartUploadPart,
    StoredObject,
)
from app.models.project import Project, ProjectAsset


class StorageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PartUploadTarget:
    method: str
    url: str
    headers: dict[str, str]


@dataclass(frozen=True, slots=True)
class CompletedObject:
    provider: str
    bucket: str | None
    object_key: str
    local_path: Path
    checksum_sha256: str


class MultipartStorageAdapter(Protocol):
    def initiate(self, upload: MultipartUpload) -> str: ...

    def part_target(self, upload: MultipartUpload, part_number: int) -> PartUploadTarget: ...

    def store_local_part(
        self,
        upload: MultipartUpload,
        part_number: int,
        payload: bytes,
    ) -> tuple[str, int]: ...

    def complete(
        self,
        upload: MultipartUpload,
        parts: list[MultipartUploadPart],
    ) -> CompletedObject: ...

    def abort(self, upload: MultipartUpload) -> None: ...


class LocalMultipartStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = Path(settings.object_storage_local_dir)

    def _upload_dir(self, upload: MultipartUpload) -> Path:
        return self.root / "multipart" / str(upload.id)

    def _object_path(self, upload: MultipartUpload) -> Path:
        return self.root / upload.object_key

    def initiate(self, upload: MultipartUpload) -> str:
        directory = self._upload_dir(upload)
        directory.mkdir(parents=True, exist_ok=False)
        return str(upload.id)

    def part_target(self, upload: MultipartUpload, part_number: int) -> PartUploadTarget:
        return PartUploadTarget(
            method="PUT",
            url=f"/api/v1/multipart-uploads/{upload.id}/parts/{part_number}",
            headers={"Content-Type": "application/octet-stream"},
        )

    def store_local_part(
        self,
        upload: MultipartUpload,
        part_number: int,
        payload: bytes,
    ) -> tuple[str, int]:
        directory = self._upload_dir(upload)
        if not directory.exists():
            raise StorageError("Multipart upload storage is unavailable")
        destination = directory / f"{part_number:06d}.part"
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
        etag = hashlib.sha256(payload).hexdigest()
        return etag, len(payload)

    def complete(
        self,
        upload: MultipartUpload,
        parts: list[MultipartUploadPart],
    ) -> CompletedObject:
        destination = self._object_path(upload)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".assembling")
        digest = hashlib.sha256()
        total = 0
        with temporary.open("wb") as output:
            for part in sorted(parts, key=lambda item: item.part_number):
                source_path = self._upload_dir(upload) / f"{part.part_number:06d}.part"
                if not source_path.exists():
                    raise StorageError(f"Multipart part {part.part_number} is missing")
                with source_path.open("rb") as source:
                    while chunk := source.read(self.settings.upload_chunk_bytes):
                        output.write(chunk)
                        digest.update(chunk)
                        total += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total != upload.total_bytes:
            temporary.unlink(missing_ok=True)
            raise StorageError(
                f"Completed object has {total} bytes; expected {upload.total_bytes}"
            )
        os.replace(temporary, destination)
        shutil.rmtree(self._upload_dir(upload), ignore_errors=True)
        return CompletedObject(
            provider="local",
            bucket=None,
            object_key=upload.object_key,
            local_path=destination,
            checksum_sha256=digest.hexdigest(),
        )

    def abort(self, upload: MultipartUpload) -> None:
        shutil.rmtree(self._upload_dir(upload), ignore_errors=True)


class S3MultipartStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.s3_bucket:
            raise StorageError("DIRECTOR_S3_BUCKET is required for S3 storage")
        try:
            import boto3
        except ImportError as exc:
            raise StorageError("boto3 is required for S3 multipart storage") from exc
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
        )

    def initiate(self, upload: MultipartUpload) -> str:
        response = self.client.create_multipart_upload(
            Bucket=self.settings.s3_bucket,
            Key=upload.object_key,
            ContentType=upload.content_type,
            Metadata={"director-project-id": str(upload.project_id)},
        )
        return str(response["UploadId"])

    def part_target(self, upload: MultipartUpload, part_number: int) -> PartUploadTarget:
        url = self.client.generate_presigned_url(
            "upload_part",
            Params={
                "Bucket": self.settings.s3_bucket,
                "Key": upload.object_key,
                "UploadId": upload.provider_upload_id,
                "PartNumber": part_number,
            },
            ExpiresIn=self.settings.multipart_presign_minutes * 60,
        )
        return PartUploadTarget(method="PUT", url=url, headers={})

    def store_local_part(
        self,
        upload: MultipartUpload,
        part_number: int,
        payload: bytes,
    ) -> tuple[str, int]:
        response = self.client.upload_part(
            Bucket=self.settings.s3_bucket,
            Key=upload.object_key,
            UploadId=upload.provider_upload_id,
            PartNumber=part_number,
            Body=payload,
        )
        return str(response["ETag"]).strip('"'), len(payload)

    def complete(
        self,
        upload: MultipartUpload,
        parts: list[MultipartUploadPart],
    ) -> CompletedObject:
        ordered = sorted(parts, key=lambda item: item.part_number)
        self.client.complete_multipart_upload(
            Bucket=self.settings.s3_bucket,
            Key=upload.object_key,
            UploadId=upload.provider_upload_id,
            MultipartUpload={
                "Parts": [
                    {"ETag": part.etag, "PartNumber": part.part_number}
                    for part in ordered
                ]
            },
        )
        cache_path = (
            Path(self.settings.upload_dir)
            / str(upload.project_id)
            / "object-cache"
            / Path(upload.object_key).name
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".download")
        self.client.download_file(self.settings.s3_bucket, upload.object_key, str(temporary))
        digest = hashlib.sha256()
        total = 0
        with temporary.open("rb") as source:
            while chunk := source.read(self.settings.upload_chunk_bytes):
                digest.update(chunk)
                total += len(chunk)
        if total != upload.total_bytes:
            temporary.unlink(missing_ok=True)
            raise StorageError(f"Downloaded object has {total} bytes; expected {upload.total_bytes}")
        os.replace(temporary, cache_path)
        return CompletedObject(
            provider="s3",
            bucket=self.settings.s3_bucket,
            object_key=upload.object_key,
            local_path=cache_path,
            checksum_sha256=digest.hexdigest(),
        )

    def abort(self, upload: MultipartUpload) -> None:
        if upload.provider_upload_id:
            self.client.abort_multipart_upload(
                Bucket=self.settings.s3_bucket,
                Key=upload.object_key,
                UploadId=upload.provider_upload_id,
            )


def storage_adapter(settings: Settings, provider: str | None = None) -> MultipartStorageAdapter:
    selected = (provider or settings.object_storage_provider).casefold()
    if selected == "local":
        return LocalMultipartStorage(settings)
    if selected == "s3":
        return S3MultipartStorage(settings)
    raise StorageError(f"Unsupported object storage provider: {selected}")


def safe_object_key(project_id: UUID, original_filename: str) -> str:
    suffix = Path(original_filename).suffix.lower()
    if len(suffix) > 12 or not suffix.replace(".", "").isalnum():
        suffix = ""
    return f"projects/{project_id}/assets/{uuid4().hex}{suffix}"


def validate_part_number(upload: MultipartUpload, part_number: int) -> None:
    maximum_parts = max(1, (upload.total_bytes + upload.part_size - 1) // upload.part_size)
    if part_number < 1 or part_number > maximum_parts:
        raise StorageError(f"Part number must be between 1 and {maximum_parts}")


def expected_part_size(upload: MultipartUpload, part_number: int) -> int:
    maximum_parts = max(1, (upload.total_bytes + upload.part_size - 1) // upload.part_size)
    validate_part_number(upload, part_number)
    if part_number < maximum_parts:
        return upload.part_size
    return upload.total_bytes - upload.part_size * (maximum_parts - 1)


def upsert_part(
    db: Session,
    upload: MultipartUpload,
    *,
    part_number: int,
    etag: str,
    size_bytes: int,
) -> MultipartUploadPart:
    existing = db.scalar(
        select(MultipartUploadPart).where(
            MultipartUploadPart.upload_id == upload.id,
            MultipartUploadPart.part_number == part_number,
        )
    )
    if existing is None:
        existing = MultipartUploadPart(
            upload_id=upload.id,
            part_number=part_number,
            etag=etag,
            size_bytes=size_bytes,
        )
        db.add(existing)
    else:
        existing.etag = etag
        existing.size_bytes = size_bytes
    db.flush()
    return existing


def finalize_upload_asset(
    db: Session,
    upload: MultipartUpload,
    completed: CompletedObject,
) -> ProjectAsset:
    project = db.get(Project, upload.project_id)
    if project is None:
        raise StorageError("Upload project no longer exists")
    asset = ProjectAsset(
        project_id=project.id,
        kind=AssetKind(upload.kind),
        original_filename=upload.original_filename,
        stored_filename=completed.local_path.name,
        content_type=upload.content_type,
        size_bytes=upload.total_bytes,
        sha256=completed.checksum_sha256,
        storage_path=str(completed.local_path),
    )
    db.add(asset)
    db.flush()
    db.add(
        StoredObject(
            asset_id=asset.id,
            provider=completed.provider,
            bucket=completed.bucket,
            object_key=completed.object_key,
            local_cache_path=str(completed.local_path),
            verified=True,
        )
    )
    upload.asset_id = asset.id
    upload.status = "completed"
    upload.error_message = None
    project.status = ProjectStatus.READY_TO_QUEUE
    project.error_message = None
    db.flush()
    return asset


def parts_fingerprint(parts: list[MultipartUploadPart]) -> str:
    payload = [
        {"part_number": item.part_number, "etag": item.etag, "size_bytes": item.size_bytes}
        for item in sorted(parts, key=lambda item: item.part_number)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def cleanup_expired_multipart_upload(
    db: Session,
    upload: MultipartUpload,
    settings: Settings,
) -> None:
    if upload.status not in {"uploading", "failed"}:
        return
    try:
        storage_adapter(settings, upload.provider).abort(upload)
    finally:
        upload.status = "expired"
        upload.error_message = "Upload session expired before completion"
        db.flush()


def is_expired(upload: MultipartUpload) -> bool:
    return is_expired_at(upload.expires_at)
