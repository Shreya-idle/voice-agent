from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Protocol

from mutagen import File as MutagenFile


class ObjectStorage(Protocol):
    def upload(self, data: bytes, path: str, content_type: str) -> None:
        ...

    def delete(self, path: str) -> None:
        ...

    def cleanup_expired(self, prefix: str, cutoff: datetime) -> int:
        ...


@dataclass(frozen=True)
class AudioAsset:
    audio_id: str
    storage_path: str
    duration: float
    mime_type: str
    created_at: datetime
    expires_at: datetime


def get_mp3_duration(data: bytes) -> float:
    audio = MutagenFile(BytesIO(data))
    if audio is None or audio.info is None or audio.info.length is None:
        raise ValueError("Generated MP3 has no readable duration")
    return round(float(audio.info.length), 3)


class FirebaseStorage:
    def __init__(self, bucket):
        self.bucket = bucket

    def upload(self, data: bytes, path: str, content_type: str) -> None:
        self.bucket.blob(path).upload_from_string(data, content_type=content_type)

    def delete(self, path: str) -> None:
        self.bucket.blob(path).delete()

    def cleanup_expired(self, prefix: str, cutoff: datetime) -> int:
        removed = 0
        for blob in self.bucket.list_blobs(prefix=prefix):
            created = blob.time_created
            if created and created < cutoff:
                blob.delete()
                removed += 1
        return removed


def create_audio_asset(
    data: bytes,
    user_id: str,
    conversation_id: str,
    audio_id: str,
    retention_hours: int,
) -> AudioAsset:
    created_at = datetime.now(timezone.utc)
    return AudioAsset(
        audio_id=audio_id,
        storage_path=f"audio/{user_id}/{conversation_id}/{audio_id}.mp3",
        duration=get_mp3_duration(data),
        mime_type="audio/mpeg",
        created_at=created_at,
        expires_at=created_at + timedelta(hours=retention_hours),
    )
