import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select a balanced benchmark set of videos for pipeline/model testing."
        )
    )
    parser.add_argument(
        "--summary",
        default="data/videos/downloaded_for_dataset_images/video_prey_summary.csv",
        help="CSV produced by create_downloaded_video_prey_summary.py.",
    )
    parser.add_argument(
        "--output",
        default="data/videos/benchmark_video_set/benchmark_videos.csv",
        help="Output CSV with selected benchmark videos.",
    )
    parser.add_argument(
        "--video-list",
        default="data/videos/benchmark_video_set/video_paths.txt",
        help="Output TXT with one selected video path per line.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=50,
        help="Approximate number of videos to select.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic selection.",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include rows with empty prey/prey_counts as the 'unknown' class.",
    )
    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help="Keep rows even when download_path does not exist locally.",
    )

    return parser.parse_args()


def parse_prey_counts(value):
    counts = Counter()

    for part in value.split(";"):
        part = part.strip()

        if not part:
            continue

        match = re.match(r"^(?P<name>.+?):(?P<count>\d+)$", part)

        if match is None:
            continue

        counts[match["name"].strip()] += int(match["count"])

    return counts


def primary_prey(row):
    counts = parse_prey_counts(row.get("prey_counts", ""))

    if counts:
        return sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

    prey_value = row.get("prey", "").strip()

    if prey_value:
        return sorted(part.strip() for part in prey_value.split(";") if part.strip())[0]

    return "unknown"


def balanced_quotas(
    availability,
    target_size,
):
    classes = sorted(availability)
    quotas = {class_name: 0 for class_name in classes}

    remaining = target_size

    while remaining > 0:
        progressed = False

        for class_name in classes:
            if quotas[class_name] >= availability[class_name]:
                continue

            quotas[class_name] += 1
            remaining -= 1
            progressed = True

            if remaining == 0:
                break

        if not progressed:
            break

    return quotas


def read_rows(
    summary_path,
    include_unknown,
    allow_missing_files,
):
    with summary_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    usable_rows = []

    for row in rows:
        row["primary_prey"] = primary_prey(row)
        video_path = Path(row.get("download_path", ""))

        if row["primary_prey"] == "unknown" and not include_unknown:
            continue

        if not allow_missing_files and not video_path.exists():
            continue

        usable_rows.append(row)

    return usable_rows


def select_rows(
    rows,
    target_size,
    seed,
):
    rng = random.Random(seed)
    by_class = defaultdict(list)

    for row in rows:
        by_class[row["primary_prey"]].append(row)

    for class_rows in by_class.values():
        rng.shuffle(class_rows)
        class_rows.sort(
            key=lambda row: (
                -int(row.get("image_count") or 0),
                row.get("blob_name", ""),
            )
        )

    availability = {
        class_name: len(class_rows)
        for class_name, class_rows in by_class.items()
    }
    quotas = balanced_quotas(availability, target_size)

    selected = []

    for class_name in sorted(by_class):
        selected.extend(by_class[class_name][: quotas[class_name]])

    rng.shuffle(selected)

    for index, row in enumerate(selected, start=1):
        row["benchmark_id"] = f"benchmark_{index:03d}"

    return selected, quotas


def write_outputs(
    selected_rows,
    output_path,
    video_list_path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_list_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "benchmark_id",
        "primary_prey",
        "blob_name",
        "download_path",
        "image_count",
        "prey",
        "prey_counts",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in selected_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    with video_list_path.open("w", encoding="utf-8") as file:
        for row in selected_rows:
            file.write(f"{row['download_path']}\n")


def print_distribution(rows, label):
    counts = Counter(row["primary_prey"] for row in rows)
    print(label)

    for class_name, count in sorted(counts.items()):
        print(f"  {class_name}: {count}")


def main():
    args = parse_args()

    if args.target_size < 1:
        raise SystemExit("--target-size must be positive.")

    summary_path = Path(args.summary)

    if not summary_path.exists():
        raise SystemExit(f"Summary CSV not found: {summary_path}")

    rows = read_rows(
        summary_path=summary_path,
        include_unknown=args.include_unknown,
        allow_missing_files=args.allow_missing_files,
    )

    if not rows:
        raise SystemExit("No usable video rows found.")

    selected_rows, quotas = select_rows(
        rows=rows,
        target_size=args.target_size,
        seed=args.seed,
    )

    output_path = Path(args.output)
    video_list_path = Path(args.video_list)
    write_outputs(
        selected_rows=selected_rows,
        output_path=output_path,
        video_list_path=video_list_path,
    )

    print(f"Usable videos: {len(rows)}")
    print(f"Target size: {args.target_size}")
    print(f"Selected videos: {len(selected_rows)}")
    print(f"Output CSV: {output_path}")
    print(f"Video list: {video_list_path}")
    print_distribution(selected_rows, "Selected primary_prey distribution:")
    print("Quotas:")

    for class_name, quota in sorted(quotas.items()):
        print(f"  {class_name}: {quota}")


if __name__ == "__main__":
    main()
