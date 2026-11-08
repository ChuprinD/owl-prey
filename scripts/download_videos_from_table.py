import argparse
import csv
import sys
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download a row range of videos listed in a blob video table CSV. "
            "Row ids are 1-based data rows, excluding the CSV header."
        )
    )
    parser.add_argument(
        "--csv",
        default="data/blob_video_table_ch2.csv",
        help="Input CSV path with a blob_name column.",
    )
    parser.add_argument(
        "--from-id",
        type=int,
        required=True,
        help="First row id to download, inclusive. The first data row is 1.",
    )
    parser.add_argument(
        "--to-id",
        type=int,
        required=True,
        help="Last row id to download, inclusive.",
    )
    parser.add_argument(
        "--download-dir",
        default="data/videos/downloaded_from_table",
        help="Directory where selected videos will be saved.",
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

    return parser.parse_args()


def validate_args(args):
    if args.from_id < 1:
        raise SystemExit("--from-id must be 1 or higher.")

    if args.to_id < args.from_id:
        raise SystemExit("--to-id must be greater than or equal to --from-id.")


def destination_for_blob(
    blob_name,
    download_dir,
    flat,
):
    blob_path = PurePosixPath(blob_name)

    if flat:
        return download_dir / blob_path.name

    return download_dir.joinpath(*blob_path.parts)


def selected_blob_names(csv_path, from_id, to_id):
    selected = []

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "blob_name" not in reader.fieldnames:
            raise SystemExit(f"CSV must contain a blob_name column: {csv_path}")

        for row_id, row in enumerate(reader, start=1):
            if row_id < from_id:
                continue

            if row_id > to_id:
                break

            blob_name = (row.get("blob_name") or "").strip()

            if not blob_name:
                print(f"Skipping row {row_id}: empty blob_name")
                continue

            selected.append((row_id, blob_name))

    return selected


def main():
    args = parse_args()
    validate_args(args)

    csv_path = Path(args.csv)
    download_dir = Path(args.download_dir)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = selected_blob_names(csv_path, args.from_id, args.to_id)

    if not rows:
        raise SystemExit(
            f"No rows selected from {csv_path} for ids {args.from_id} through {args.to_id}."
        )

    try:
        from owl_prey_detection.data_access.azure_blob import AzureBlobClient
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Azure dependency. Install project requirements first:\n"
            "  pip install -r requirements.txt"
        ) from exc

    client = AzureBlobClient()

    print(f"Downloading {len(rows)} videos from rows {args.from_id} through {args.to_id}.")
    print(f"Destination: {download_dir}")

    downloaded = 0
    skipped_existing = 0

    for row_id, blob_name in rows:
        destination_path = destination_for_blob(
            blob_name=blob_name,
            download_dir=download_dir,
            flat=args.flat,
        )

        existed = destination_path.exists()
        client.download_blob(
            blob_name=blob_name,
            destination_path=destination_path,
            overwrite=args.overwrite,
        )

        if existed and not args.overwrite:
            skipped_existing += 1
            print(f"[{row_id}] already exists: {destination_path}")
            continue

        downloaded += 1
        print(f"[{row_id}] downloaded: {destination_path}")

    print(
        f"Finished. Downloaded {downloaded}; "
        f"already existed {skipped_existing}; selected {len(rows)}."
    )


if __name__ == "__main__":
    main()
