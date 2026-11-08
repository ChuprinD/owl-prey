import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

VIDEO_FILENAME_PATTERN = re.compile(
    r"^(?P<filename_location>.+)_ch(?P<channel>\d+)(?:_(?P<camera_label>.+?))?_(?P<start>\d{14})_(?P<end>\d{14})\.mp4$",
    re.IGNORECASE,
)


class VideoRecord:
    def __init__(
        self,
        blob_name,
        location,
        channel,
        date,
        start_datetime,
        end_datetime,
        filename_location,
        camera_label,
    ):
        self.blob_name = blob_name
        self.location = location
        self.channel = channel
        self.date = date
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.filename_location = filename_location
        self.camera_label = camera_label


def parse_blob_video_name(blob_name):
    path = PurePosixPath(blob_name)
    match = VIDEO_FILENAME_PATTERN.match(path.name)

    if match is None:
        return None

    start_datetime = datetime.strptime(match["start"], "%Y%m%d%H%M%S")
    end_datetime = datetime.strptime(match["end"], "%Y%m%d%H%M%S")
    filename_location = match["filename_location"]

    location = path.parts[0] if len(path.parts) > 1 else filename_location

    return VideoRecord(
        blob_name=blob_name,
        location=location,
        channel=f"ch{match['channel']}",
        date=start_datetime.date().isoformat(),
        start_datetime=start_datetime.isoformat(sep=" "),
        end_datetime=end_datetime.isoformat(sep=" "),
        filename_location=filename_location,
        camera_label=match["camera_label"] or "",
    )


def collect_video_records(
    client,
    prefix,
    channel,
):
    records = []
    skipped = []
    normalized_channel = normalize_channel(channel) if channel else None

    for blob in client.list_blobs(name_starts_with=prefix):
        blob_name = blob.name

        if not blob_name.lower().endswith(".mp4"):
            continue

        record = parse_blob_video_name(blob_name)

        if record is None:
            skipped.append(blob_name)
            continue

        if normalized_channel and record.channel != normalized_channel:
            continue

        records.append(record)

    records.sort(key=lambda record: (record.location, record.channel, record.start_datetime))

    return records, skipped


def write_csv(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "blob_name",
        "location",
        "channel",
        "date",
        "start_datetime",
        "end_datetime",
        "filename_location",
        "camera_label",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "blob_name": record.blob_name,
                    "location": record.location,
                    "channel": record.channel,
                    "date": record.date,
                    "start_datetime": record.start_datetime,
                    "end_datetime": record.end_datetime,
                    "filename_location": record.filename_location,
                    "camera_label": record.camera_label,
                }
            )


def normalize_channel(channel):
    channel = channel.strip().lower()

    if channel.startswith("ch"):
        return channel

    return f"ch{channel}"


def print_preview(records, limit):
    if not records:
        print("No matching video blobs found.")
        return

    print("location,channel,date,start_datetime,end_datetime,blob_name")

    for record in records[:limit]:
        print(
            ",".join(
                [
                    record.location,
                    record.channel,
                    record.date,
                    record.start_datetime,
                    record.end_datetime,
                    record.blob_name,
                ]
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a CSV table of Azure Blob videos named like "
            "Vosseveld/Vosseveld_ch2_20240610043904_20240610043955.mp4."
        )
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional blob prefix to scan, for example Vosseveld/.",
    )
    parser.add_argument(
        "--output",
        default="data/blob_video_table.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="Optional channel filter, for example 2 or ch2.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=20,
        help="Number of parsed rows to print after writing the CSV.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from owl_prey_detection.data_access.azure_blob import AzureBlobClient
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Azure dependency. Install project requirements first:\n"
            "  pip install -r requirements.txt"
        ) from exc

    client = AzureBlobClient()
    records, skipped = collect_video_records(client, args.prefix, args.channel)
    output_path = Path(args.output)

    write_csv(records, output_path)

    print(f"Wrote {len(records)} rows to {output_path}")

    if skipped:
        print(f"Skipped {len(skipped)} .mp4 blobs with unexpected names.")
        print("First skipped examples:")

        for blob_name in skipped[:10]:
            print(f"- {blob_name}")

    print_preview(records, args.preview)


if __name__ == "__main__":
    main()
