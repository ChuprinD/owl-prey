import argparse
import csv
import re
import sys
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FULL_VIDEO_PATTERN = re.compile(
    r"^(?P<location>.+?)_ch(?P<channel>\d+)(?:_(?P<camera_label>.+?))?_(?P<start>\d{14})_(?P<end>\d{14})(?:_|$)",
    re.IGNORECASE,
)
START_ONLY_PATTERN = re.compile(
    r"^(?P<location>.+?)_ch(?P<channel>\d+)(?:_(?P<camera_label>.+?))?_(?P<start>\d{14})(?:_|$)",
    re.IGNORECASE,
)


class VideoReference:
    def __init__(self, location, channel, start, end, camera_label):
        self.location = location
        self.channel = channel
        self.start = start
        self.end = end
        self.camera_label = camera_label

    @property
    def original_stem(self):
        parts = [self.location, self.channel]

        if self.camera_label:
            parts.append(self.camera_label)

        parts.append(self.start)

        if self.end:
            parts.append(self.end)

        return "_".join(parts)


class TableRow:
    def __init__(
        self,
        row_id,
        blob_name,
        location,
        channel,
        start,
        end,
        camera_label,
    ):
        self.row_id = row_id
        self.blob_name = blob_name
        self.location = location
        self.channel = channel
        self.start = start
        self.end = end
        self.camera_label = camera_label

    @property
    def blob_stem(self):
        return PurePosixPath(self.blob_name).stem


def compact_datetime(value):
    return re.sub(r"\D", "", value)[:14]


def parse_image_video_reference(image_path):
    stem = image_path.stem
    match = FULL_VIDEO_PATTERN.match(stem)

    if match is None:
        match = START_ONLY_PATTERN.match(stem)

    if match is None:
        return None

    return VideoReference(
        location=match["location"],
        channel=f"ch{match['channel']}",
        start=match["start"],
        end=match.groupdict().get("end"),
        camera_label=match.groupdict().get("camera_label") or "",
    )


def iter_dataset_images(dataset_dir):
    images = []

    for split in ("train", "val"):
        images_dir = dataset_dir / split / "images"

        if not images_dir.exists():
            continue

        for image_path in images_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(image_path)

    return sorted(images)


def read_table(table_path):
    rows = []

    with table_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "blob_name",
            "location",
            "channel",
            "start_datetime",
            "end_datetime",
            "camera_label",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise SystemExit(f"Table is missing required columns: {missing}")

        for row_id, row in enumerate(reader, start=1):
            rows.append(
                TableRow(
                    row_id=row_id,
                    blob_name=row["blob_name"],
                    location=row["location"],
                    channel=row["channel"],
                    start=compact_datetime(row["start_datetime"]),
                    end=compact_datetime(row["end_datetime"]),
                    camera_label=row.get("camera_label", ""),
                )
            )

    return rows


def build_indexes(rows):
    exact_by_stem = {}
    by_start = {}

    for row in rows:
        exact_by_stem[row.blob_stem.lower()] = row
        start_key = (row.location.lower(), row.channel.lower(), row.start)
        by_start.setdefault(start_key, []).append(row)

    return exact_by_stem, by_start


def find_matching_rows(
    reference,
    exact_by_stem,
    by_start,
):
    exact_match = exact_by_stem.get(reference.original_stem.lower())

    if exact_match:
        return [exact_match], "matched_exact_video_name"

    start_key = (reference.location.lower(), reference.channel.lower(), reference.start)
    start_matches = by_start.get(start_key, [])

    if len(start_matches) == 1:
        return start_matches, "matched_by_location_channel_start"

    if len(start_matches) > 1:
        return start_matches, "ambiguous_location_channel_start"

    return [], "not_found_in_table"


def candidate_blob_names(reference):
    filename = f"{reference.original_stem}.mp4"

    candidates = [
        f"{reference.location}/{reference.channel}/{filename}",
        f"{reference.location}/{filename}",
        filename,
    ]

    return list(dict.fromkeys(candidates))


def find_azure_fallback_row(
    reference,
    client,
):
    if reference.end is None:
        return None, "not_found_in_table"

    for blob_name in candidate_blob_names(reference):
        if client.blob_exists(blob_name):
            return (
                TableRow(
                    row_id=0,
                    blob_name=blob_name,
                    location=reference.location,
                    channel=reference.channel,
                    start=reference.start,
                    end=reference.end,
                    camera_label=reference.camera_label,
                ),
                "matched_by_azure_candidate_path",
            )

    return None, "not_found_in_table_or_candidate_paths"


def destination_for_blob(blob_name, download_dir, flat):
    blob_path = PurePosixPath(blob_name)

    if flat:
        return download_dir / blob_path.name

    return download_dir.joinpath(*blob_path.parts)


def write_manifest(manifest_rows, manifest_path):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path",
        "extracted_video_name",
        "status",
        "reason",
        "blob_name",
        "table_row_id",
        "download_path",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Azure videos referenced by YOLO dataset image filenames."
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/temp",
        help="Dataset folder containing train/images and val/images.",
    )
    parser.add_argument(
        "--table",
        default="data/blob_video_table_ch2.csv",
        help="Blob video table CSV produced by create_blob_video_table.py.",
    )
    parser.add_argument(
        "--download-dir",
        default="data/videos/downloaded_for_dataset_images",
        help="Directory where matched videos will be saved.",
    )
    parser.add_argument(
        "--manifest",
        default="data/videos/downloaded_for_dataset_images/manifest.csv",
        help="CSV report of matched and unmatched dataset images.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Save all videos directly in download-dir instead of preserving blob folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download again even if the local file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write the manifest and print counts; do not download videos.",
    )
    parser.add_argument(
        "--no-azure-fallback",
        action="store_true",
        help=(
            "Only match against the CSV table. By default, full video names that "
            "are missing from the table are checked against likely Azure blob paths."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    table_path = Path(args.table)
    download_dir = Path(args.download_dir)
    manifest_path = Path(args.manifest)

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    if not table_path.exists():
        raise SystemExit(f"Blob video table not found: {table_path}")

    images = iter_dataset_images(dataset_dir)
    table_rows = read_table(table_path)
    exact_by_stem, by_start = build_indexes(table_rows)
    azure_client = None

    if not args.no_azure_fallback:
        try:
            from owl_prey_detection.data_access.azure_blob import AzureBlobClient
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing Azure dependency. Install project requirements first:\n"
                "  pip install -r requirements.txt"
            ) from exc

        azure_client = AzureBlobClient()

    manifest_rows = []
    blobs_to_download = {}
    reference_cache = {}
    matched_images = 0
    ambiguous_images = 0
    unmatched_images = 0

    for image_path in images:
        reference = parse_image_video_reference(image_path)

        if reference is None:
            unmatched_images += 1
            manifest_rows.append(
                {
                    "image_path": str(image_path),
                    "extracted_video_name": "",
                    "status": "unmatched",
                    "reason": "no_video_timestamp_in_image_name",
                    "blob_name": "",
                    "table_row_id": "",
                    "download_path": "",
                }
            )
            continue

        cache_key = reference.original_stem.lower()
        cached_match = reference_cache.get(cache_key)

        if cached_match is None:
            matches, reason = find_matching_rows(reference, exact_by_stem, by_start)

            if not matches and azure_client is not None:
                fallback_row, reason = find_azure_fallback_row(reference, azure_client)

                if fallback_row is not None:
                    matches = [fallback_row]

            reference_cache[cache_key] = (matches, reason)
        else:
            matches, reason = cached_match

        if len(matches) != 1:
            status = "ambiguous" if matches else "unmatched"
            ambiguous_images += 1 if matches else 0
            unmatched_images += 0 if matches else 1
            manifest_rows.append(
                {
                    "image_path": str(image_path),
                    "extracted_video_name": reference.original_stem,
                    "status": status,
                    "reason": reason,
                    "blob_name": ";".join(row.blob_name for row in matches),
                    "table_row_id": ";".join(str(row.row_id) for row in matches),
                    "download_path": "",
                }
            )
            continue

        matched_images += 1
        row = matches[0]
        destination_path = destination_for_blob(row.blob_name, download_dir, args.flat)
        blobs_to_download[row.blob_name] = destination_path
        manifest_rows.append(
            {
                "image_path": str(image_path),
                "extracted_video_name": reference.original_stem,
                "status": "matched",
                "reason": reason,
                "blob_name": row.blob_name,
                "table_row_id": str(row.row_id),
                "download_path": str(destination_path),
            }
        )

    write_manifest(manifest_rows, manifest_path)

    print(f"Images scanned: {len(images)}")
    print(f"Matched images: {matched_images}")
    print(f"Unmatched images: {unmatched_images}")
    print(f"Ambiguous images: {ambiguous_images}")
    print(f"Unique videos to download: {len(blobs_to_download)}")
    print(f"Manifest: {manifest_path}")

    if args.dry_run:
        print("Dry run enabled; no videos downloaded.")
        return

    if not blobs_to_download:
        print("No matched videos to download.")
        return

    client = azure_client

    if client is None:
        try:
            from owl_prey_detection.data_access.azure_blob import AzureBlobClient
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing Azure dependency. Install project requirements first:\n"
                "  pip install -r requirements.txt"
            ) from exc

        client = AzureBlobClient()
    downloaded = 0
    skipped_existing = 0

    for index, (blob_name, destination_path) in enumerate(blobs_to_download.items(), start=1):
        existed = destination_path.exists()
        client.download_blob(
            blob_name=blob_name,
            destination_path=destination_path,
            overwrite=args.overwrite,
        )

        if existed and not args.overwrite:
            skipped_existing += 1
            print(f"[{index}/{len(blobs_to_download)}] already exists: {destination_path}")
            continue

        downloaded += 1
        print(f"[{index}/{len(blobs_to_download)}] downloaded: {destination_path}")

    print(
        f"Finished. Downloaded {downloaded}; "
        f"already existed {skipped_existing}; selected {len(blobs_to_download)}."
    )


if __name__ == "__main__":
    main()
