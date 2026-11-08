import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from owl_prey_detection.config import PipelineConfig  # noqa: E402
from owl_prey_detection.pipeline import process_video  # noqa: E402


def normalize_label(value):
    return str(value or "").strip().lower()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the owl-prey pipeline on a fixed video benchmark set."
    )
    parser.add_argument(
        "--benchmark",
        default="data/videos/benchmark_video_set/benchmark_videos.csv",
        help="Benchmark CSV produced by select_balanced_video_benchmark.py.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Base pipeline config.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/video_benchmark",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--run-name",
        default="default",
        help="Name for this model/parameter run.",
    )
    parser.add_argument("--model-path", help="Override config model_path.")
    parser.add_argument("--conf", type=float, help="Override inference.conf.")
    parser.add_argument("--iou", type=float, help="Override inference.iou.")
    parser.add_argument("--imgsz", type=int, help="Override inference.imgsz.")
    parser.add_argument("--max-det", type=int, help="Override inference.max_det.")
    parser.add_argument("--device", help="Override inference.device.")
    parser.add_argument("--batch-size", type=int, help="Override inference.batch_size.")
    parser.add_argument("--frame-stride", type=int, help="Override frame_stride.")
    parser.add_argument("--sample-fps", type=float, help="Override sample_fps.")
    parser.add_argument("--top-k", default=None, help="Override top_k, e.g. 5 or all.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N benchmark videos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected videos and write no pipeline outputs.",
    )

    return parser.parse_args()


def parse_top_k(value):
    if value is None:
        return None

    normalized = value.strip().lower()

    if normalized == "all":
        return "all"

    return int(normalized)


def read_benchmark(path, limit):
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if limit is not None:
        rows = rows[:limit]

    return rows


def apply_overrides(config, args):
    if args.model_path:
        config.model_path = Path(args.model_path)

    if args.conf is not None:
        config.inference_conf = args.conf

    if args.iou is not None:
        config.inference_iou = args.iou

    if args.imgsz is not None:
        config.inference_imgsz = args.imgsz

    if args.max_det is not None:
        config.inference_max_det = args.max_det

    if args.device is not None:
        config.inference_device = args.device

    if args.batch_size is not None:
        config.inference_batch_size = args.batch_size

    if args.frame_stride is not None:
        config.frame_stride = args.frame_stride

    if args.sample_fps is not None:
        config.sample_fps = args.sample_fps

    top_k = parse_top_k(args.top_k)

    if top_k is not None:
        config.top_k = top_k


def metadata_row(
    benchmark_row,
    metadata_path,
    top_review_path,
    run_name,
):
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    prey_class_gt = normalize_label(benchmark_row.get("primary_prey", ""))
    has_prey_gt = bool(prey_class_gt and prey_class_gt != "unknown")
    has_owl_gt = has_prey_gt
    predicted_prey_labels = read_predicted_prey_labels(top_review_path)
    prediction_correct = (
        has_owl_gt
        and has_prey_gt
        and prey_class_gt in predicted_prey_labels
    )

    return {
        "run_name": run_name,
        "benchmark_id": benchmark_row.get("benchmark_id", ""),
        "primary_prey": benchmark_row.get("primary_prey", ""),
        "has_owl_gt": has_owl_gt,
        "has_prey_gt": has_prey_gt,
        "prey_class_gt": prey_class_gt,
        "prediction_correct": prediction_correct,
        "predicted_prey_labels": "; ".join(sorted(predicted_prey_labels)),
        "blob_name": benchmark_row.get("blob_name", ""),
        "video_path": metadata["video_path"],
        "status": "ok",
        "target_frames_found": metadata["target_frames_found"],
        "top_frames_saved": metadata["top_frames_saved"],
        "frames_processed": metadata["frames_processed"],
        "total_detections": metadata["total_detections"],
        "video_duration_sec": metadata["video_duration_sec"],
        "model_runtime_sec": metadata["model_runtime_sec"],
        "total_runtime_sec": metadata["total_runtime_sec"],
        "processing_fps": metadata["processing_fps"],
        "conf": metadata["inference"]["conf"],
        "iou": metadata["inference"]["iou"],
        "imgsz": metadata["inference"]["imgsz"],
        "max_det": metadata["inference"]["max_det"],
        "device": metadata["inference"]["device"],
        "batch_size": metadata["inference"]["batch_size"],
        "frame_stride": metadata["frame_stride"],
        "sample_fps": metadata["sample_fps"],
        "video_output_dir": str(metadata_path.parent),
        "error": "",
    }


def error_row(
    benchmark_row,
    run_name,
    error,
):
    return {
        "run_name": run_name,
        "benchmark_id": benchmark_row.get("benchmark_id", ""),
        "primary_prey": benchmark_row.get("primary_prey", ""),
        "has_owl_gt": "",
        "has_prey_gt": "",
        "prey_class_gt": normalize_label(benchmark_row.get("primary_prey", "")),
        "prediction_correct": "",
        "predicted_prey_labels": "",
        "blob_name": benchmark_row.get("blob_name", ""),
        "video_path": benchmark_row.get("download_path", ""),
        "status": "error",
        "target_frames_found": "",
        "top_frames_saved": "",
        "frames_processed": "",
        "total_detections": "",
        "video_duration_sec": "",
        "model_runtime_sec": "",
        "total_runtime_sec": "",
        "processing_fps": "",
        "conf": "",
        "iou": "",
        "imgsz": "",
        "max_det": "",
        "device": "",
        "batch_size": "",
        "frame_stride": "",
        "sample_fps": "",
        "video_output_dir": "",
        "error": str(error),
    }


def read_predicted_prey_labels(top_review_path):
    if not top_review_path.exists():
        return set()

    with top_review_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return {
            normalize_label(row.get("prey_label", ""))
            for row in reader
            if normalize_label(row.get("prey_label", ""))
        }


def write_summary(rows, path):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_run_config(config, path):
    config_dict = vars(config).copy()

    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config_dict, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    benchmark_path = Path(args.benchmark)

    if not benchmark_path.exists():
        raise SystemExit(f"Benchmark CSV not found: {benchmark_path}")

    rows = read_benchmark(benchmark_path, args.limit)

    if args.dry_run:
        print(f"Videos selected: {len(rows)}")

        for row in rows:
            print(
                f"{row.get('benchmark_id', '')}\t"
                f"{row.get('primary_prey', '')}\t"
                f"{row.get('download_path', '')}"
            )

        return

    base_output_dir = Path(args.output_dir) / args.run_name
    summary_rows = []
    wrote_run_config = False

    for index, benchmark_row in enumerate(rows, start=1):
        benchmark_id = benchmark_row.get("benchmark_id") or f"video_{index:03d}"
        video_path = Path(benchmark_row["download_path"])
        video_output_dir = base_output_dir / benchmark_id

        print(f"[{index}/{len(rows)}] {benchmark_id}: {video_path}")

        try:
            config = PipelineConfig.from_yaml(args.config)
            apply_overrides(config, args)
            config.output_dir = video_output_dir

            if not wrote_run_config:
                run_config = PipelineConfig.from_yaml(args.config)
                apply_overrides(run_config, args)
                run_config.output_dir = base_output_dir
                write_run_config(
                    config=run_config,
                    path=base_output_dir / "effective_config.json",
                )
                wrote_run_config = True

            outputs = process_video(
                video_path=video_path,
                config=config,
            )

            summary_rows.append(
                metadata_row(
                    benchmark_row=benchmark_row,
                    metadata_path=outputs["metadata"],
                    top_review_path=outputs["top_for_review"],
                    run_name=args.run_name,
                )
            )
        except Exception as exc:
            summary_rows.append(
                error_row(
                    benchmark_row=benchmark_row,
                    run_name=args.run_name,
                    error=exc,
                )
            )
            print(f"  ERROR: {exc}")

    summary_path = base_output_dir / "benchmark_summary.csv"
    write_summary(summary_rows, summary_path)
    print(f"Benchmark summary: {summary_path}")


if __name__ == "__main__":
    main()
