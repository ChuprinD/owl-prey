import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import yaml


OWL_CLASS_NAME = "steenuil"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a CSV summary of downloaded videos and prey classes from YOLO labels."
    )
    parser.add_argument(
        "--dataset-yaml",
        default="data/temp/data.yaml",
        help="YOLO data.yaml with class id to class name mapping.",
    )
    parser.add_argument(
        "--manifest",
        default="data/videos/downloaded_for_dataset_images/manifest.csv",
        help="Manifest produced by download_videos_for_dataset_images.py.",
    )
    parser.add_argument(
        "--output",
        default="data/videos/downloaded_for_dataset_images/video_prey_summary.csv",
        help="Output CSV path.",
    )

    return parser.parse_args()


def read_class_names(dataset_yaml):
    with dataset_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data.get("names", {})

    return {int(class_id): str(name) for class_id, name in names.items()}


def label_path_for_image(image_path):
    parts = list(image_path.parts)

    try:
        images_index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Image path does not contain an images folder: {image_path}") from exc

    parts[images_index] = "labels"

    return Path(*parts).with_suffix(".txt")


def read_label_classes(label_path, class_names):
    classes = Counter()

    if not label_path.exists():
        return classes

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            class_id = int(line.split()[0])
            class_name = class_names.get(class_id, f"class_{class_id}")

            if class_name == OWL_CLASS_NAME:
                continue

            classes[class_name] += 1

    return classes


def format_counts(counts):
    return "; ".join(f"{name}:{count}" for name, count in sorted(counts.items()))


def main():
    args = parse_args()
    dataset_yaml = Path(args.dataset_yaml)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    if not dataset_yaml.exists():
        raise SystemExit(f"Dataset YAML not found: {dataset_yaml}")

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    class_names = read_class_names(dataset_yaml)
    video_prey_counts = defaultdict(Counter)
    video_image_counts = Counter()
    video_paths = {}

    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("status") != "matched":
                continue

            download_path = row.get("download_path", "")

            if not download_path or not Path(download_path).exists():
                continue

            blob_name = row["blob_name"]
            image_path = Path(row["image_path"])
            label_path = label_path_for_image(image_path)
            prey_counts = read_label_classes(label_path, class_names)

            video_paths[blob_name] = download_path
            video_image_counts[blob_name] += 1
            video_prey_counts[blob_name].update(prey_counts)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "blob_name",
            "download_path",
            "image_count",
            "prey",
            "prey_counts",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for blob_name in sorted(video_paths):
            counts = video_prey_counts[blob_name]
            writer.writerow(
                {
                    "blob_name": blob_name,
                    "download_path": video_paths[blob_name],
                    "image_count": video_image_counts[blob_name],
                    "prey": "; ".join(sorted(counts)),
                    "prey_counts": format_counts(counts),
                }
            )

    print(f"Wrote {len(video_paths)} video rows to {output_path}")


if __name__ == "__main__":
    main()
