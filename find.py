from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath

from owl_prey_detection.data_access.azure_blob import AzureBlobClient


VIDEO_PATTERN = re.compile(
    r"(?P<location>[A-Za-z0-9_-]+)_"
    r"(?P<channel>ch\d+)_"
    r"(?P<start>\d{14})_"
    r"(?P<end>\d{14})"
    r"\.(?P<extension>mp4|avi|mov|mkv)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VideoBlobInfo:
    blob_name: str
    filename: str
    location: str
    channel: str
    start_time: datetime
    end_time: datetime


def parse_video_blob_name(blob_name: str) -> VideoBlobInfo | None:
    filename = PurePosixPath(blob_name).name
    match = VIDEO_PATTERN.search(filename)

    if match is None:
        return None

    start_time = datetime.strptime(match.group("start"), "%Y%m%d%H%M%S")
    end_time = datetime.strptime(match.group("end"), "%Y%m%d%H%M%S")

    return VideoBlobInfo(
        blob_name=blob_name,
        filename=filename,
        location=match.group("location"),
        channel=match.group("channel"),
        start_time=start_time,
        end_time=end_time,
    )


def parse_target_datetime(value: str) -> datetime:
    supported_formats = [
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y%m%d%H%M%S",
    ]

    for fmt in supported_formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass

    raise ValueError(
        "Unsupported datetime format. Use one of:\n"
        "  13-05-2025 04:38:56\n"
        "  2025-05-13 04:38:56\n"
        "  20250513043856"
    )


def find_matching_videos(
    location: str,
    target_time: datetime,
    channel: str | None = None,
    max_scan: int = 200_000,
) -> list[VideoBlobInfo]:
    client = AzureBlobClient()

    location_lower = location.lower()
    channel_lower = channel.lower() if channel is not None else None

    matches: list[VideoBlobInfo] = []
    scanned = 0

    for blob in client.list_blobs():
        scanned += 1

        blob_name = blob.name

        # Fast pre-filter: avoid parsing completely unrelated blobs.
        if location_lower not in blob_name.lower():
            if scanned >= max_scan:
                break
            continue

        info = parse_video_blob_name(blob_name)

        if info is None:
            if scanned >= max_scan:
                break
            continue

        if info.location.lower() != location_lower:
            if scanned >= max_scan:
                break
            continue

        if channel_lower is not None and info.channel.lower() != channel_lower:
            if scanned >= max_scan:
                break
            continue

        if info.start_time <= target_time <= info.end_time:
            matches.append(info)

        if scanned >= max_scan:
            break

    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find full Azure blob name by location and timestamp."
    )

    parser.add_argument(
        "--location",
        required=True,
        help="Location name from video overlay, for example Haarle.",
    )

    parser.add_argument(
        "--time",
        required=True,
        help='Timestamp from video overlay, for example "13-05-2025 04:38:56".',
    )

    parser.add_argument(
        "--channel",
        default=None,
        help="Optional camera channel, for example ch2.",
    )

    parser.add_argument(
        "--max-scan",
        type=int,
        default=200_000,
        help="Maximum number of blobs to scan.",
    )

    args = parser.parse_args()

    target_time = parse_target_datetime(args.time)

    matches = find_matching_videos(
        location=args.location,
        target_time=target_time,
        channel=args.channel,
        max_scan=args.max_scan,
    )

    if not matches:
        print("No matching videos found.")
        print()
        print("Check:")
        print("  1. location spelling, for example Haarle")
        print("  2. channel, for example ch1/ch2")
        print("  3. whether blob timestamps use another timezone")
        print("  4. whether the video is in another Azure container")
        return

    print(f"Found {len(matches)} matching video(s):")
    print()

    for info in matches:
        print(info.blob_name)
        print(f"  location: {info.location}")
        print(f"  channel:  {info.channel}")
        print(f"  start:    {info.start_time}")
        print(f"  end:      {info.end_time}")
        print()

    if len(matches) == 1:
        blob_name = matches[0].blob_name

        print("Use this in config.yaml:")
        print()
        print("input:")
        print("  source: azure_blob")
        print(f'  blob_name: "{blob_name}"')
        print("  download_dir: data/videos/downloaded")
        print("  overwrite_download: false")


if __name__ == "__main__":
    main()