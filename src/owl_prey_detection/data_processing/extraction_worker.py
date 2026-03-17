from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from azure.storage.blob import BlobServiceClient

from owl_prey_detection.data_processing.motion import extract_motion_frames


def process_blob_mp(
    blob_name: str,
    connection_string: str,
    container_name: str,
    output_dir: str | Path,
    temp_dir: str | Path,
    skip_frames: int,
    motion_threshold: int,
    min_motion_pixels: int,
    min_sharpness: int,
    jpeg_quality: int = 92,
) -> tuple[str, int, list[dict[str, Any]] | str]:
    """
    Multiprocessing worker function that processes a single blob.
    Downloads the blob, extracts motion frames, saves JPEGs, and returns metadata.
    """
    try:
        # Initialize client inside the process
        client = BlobServiceClient.from_connection_string(connection_string).get_container_client(container_name)
    except Exception as e:
        return blob_name, -1, f"Azure Connection Error: {e}"

    stem = Path(blob_name).stem
    parts = stem.split("_")
    location = parts[0]
    channel = parts[1] if len(parts) > 1 else "ch1"
    dt_str = parts[2] if len(parts) > 2 else ""

    out_dir = Path(output_dir) / location
    tmp_path = Path(temp_dir) / f"{stem}_{os.getpid()}.mp4"
    rows: list[dict[str, Any]] = []

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Download blob
        with open(tmp_path, "wb") as f:
            f.write(client.get_blob_client(blob_name).download_blob().readall())

        # Extract frames
        frames = extract_motion_frames(
            str(tmp_path),
            skip_frames=skip_frames,
            motion_threshold=motion_threshold,
            min_motion_pixels=min_motion_pixels,
            min_sharpness=min_sharpness,
            jpeg_quality=jpeg_quality,
        )

        # Save extracted frames and construct manifest rows
        for jpg_bytes, frame_no, score in frames:
            fname = f"{location}_{channel}_{dt_str}_f{frame_no:05d}_s{score:.0f}.jpg"
            out_path = out_dir / fname
            with open(out_path, "wb") as jf:
                jf.write(jpg_bytes)

            rows.append(
                {
                    "filename": fname,
                    "label": "unlabelled",
                    "location": location,
                    "channel": channel,
                    "dt_start": dt_str,
                    "frame_no": frame_no,
                    "sharpness": round(score, 1),
                    "source_blob": blob_name,
                }
            )

        return blob_name, len(frames), rows

    except Exception as e:
        return blob_name, -1, str(e)
    finally:
        # Cleanup temporary file
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
