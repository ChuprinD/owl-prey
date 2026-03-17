from dataclasses import dataclass


@dataclass(slots=True)
class BlobMetadata:
    blob_name: str
    size_bytes: int
    suffix: str
    top_level_dir: str
    parent_dir: str
    filename: str
    is_video: bool
    last_modified: str | None
    etag: str | None

    # parsed fields
    location_from_name: str | None
    camera_from_name: str | None
    stream: str | None
    start_time: str | None
    end_time: str | None
    duration_seconds: float | None