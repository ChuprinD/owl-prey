from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from owl_prey_detection.data_access.azure_blob import AzureBlobClient
from owl_prey_detection.data_access.schemas import BlobMetadata
from owl_prey_detection.data_access.filename_parser import parse_video_filename


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class BlobInventoryBuilder:
    def __init__(self, azure_client: AzureBlobClient) -> None:
        self._azure_client = azure_client
        
    @staticmethod
    def _blob_to_metadata(blob) -> BlobMetadata:
        blob_name = blob.name
        path = Path(blob_name)
        parts = blob_name.split("/")

        suffix = path.suffix.lower()
        top_level_dir = parts[0] if len(parts) > 1 else ""
        parent_dir = "/".join(parts[:-1]) if len(parts) > 1 else ""
        filename = parts[-1]
        is_video = suffix in VIDEO_EXTENSIONS

        parsed = parse_video_filename(filename)

        if parsed is not None:
            location_from_name = parsed.location
            camera_from_name = parsed.camera
            stream = parsed.stream
            start_time = parsed.start_time.isoformat()
            end_time = parsed.end_time.isoformat()
            duration_seconds = (parsed.end_time - parsed.start_time).total_seconds()
        else:
            location_from_name = None
            camera_from_name = None
            stream = None
            start_time = None
            end_time = None
            duration_seconds = None

        return BlobMetadata(
            blob_name=blob_name,
            size_bytes=int(getattr(blob, "size", 0) or 0),
            suffix=suffix,
            top_level_dir=top_level_dir,
            parent_dir=parent_dir,
            filename=filename,
            is_video=is_video,
            last_modified=str(getattr(blob, "last_modified", None)),
            etag=str(getattr(blob, "etag", None)),

            location_from_name=location_from_name,
            camera_from_name=camera_from_name,
            stream=stream,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
        )

    def build_dataframe(self) -> pd.DataFrame:
        records = []

        for blob in self._azure_client.list_blobs():
            metadata = self._blob_to_metadata(blob)
            records.append(asdict(metadata))

        return pd.DataFrame(records)

    @staticmethod
    def save_to_csv(df: pd.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    @staticmethod
    def summarize(df: pd.DataFrame) -> dict:
        if df.empty:
            return {
                "total_blobs": 0,
                "total_videos": 0,
                "total_size_gb": 0.0,
                "video_size_gb": 0.0,
            }

        video_df = df[df["is_video"] == True]

        return {
            "total_blobs": int(len(df)),
            "total_videos": int(len(video_df)),
            "total_size_gb": float(df["size_bytes"].sum() / (1024 ** 3)),
            "video_size_gb": float(video_df["size_bytes"].sum() / (1024 ** 3)),
        }