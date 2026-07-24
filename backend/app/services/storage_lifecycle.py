from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.models.operations import StoredObject
from app.services.storage import StorageError


def delete_stored_object(settings: Settings, stored: StoredObject) -> None:
    if stored.local_cache_path:
        Path(stored.local_cache_path).unlink(missing_ok=True)
    if stored.provider == "local":
        (Path(settings.object_storage_local_dir) / stored.object_key).unlink(missing_ok=True)
        return
    if stored.provider != "s3":
        raise StorageError(f"Unsupported stored object provider: {stored.provider}")
    bucket = stored.bucket or settings.s3_bucket
    if not bucket:
        raise StorageError("S3 bucket is unavailable for object deletion")
    try:
        import boto3
    except ImportError as exc:
        raise StorageError("boto3 is required for S3 object deletion") from exc
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
    )
    client.delete_object(Bucket=bucket, Key=stored.object_key)
