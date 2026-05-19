from __future__ import annotations

import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import pandas as pd
from tqdm import tqdm

from owl_prey_detection.config import PipelineConfig
from owl_prey_detection.detector import YoloDetector
from owl_prey_detection.draw import draw_detections
from owl_prey_detection.frame_logic import (
    compute_frame_score,
    frame_has_required_groups,
    summarize_detection_groups,
)
from owl_prey_detection.ranking import select_top_k_frames
from owl_prey_detection.video_io import (
    FrameItem,
    get_video_metadata,
    iter_video_frames,
)


def _safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _frame_filename(
    video_stem: str,
    rank: int,
    frame_index: int,
    timestamp_sec: float,
) -> str:
    return (
        f"top_{rank:02d}_"
        f"{video_stem}_"
        f"f{frame_index:08d}_"
        f"t{timestamp_sec:010.3f}.jpg"
    )


def _empty_group_summary() -> dict[str, Any]:
    return {
        "present": False,
        "score": 0.0,
        "label": "",
        "count": 0,
    }


def _expected_processed_frames(
    frame_count: int,
    frame_stride: int,
) -> int:
    if frame_count <= 0:
        return 0

    return (frame_count + frame_stride - 1) // frame_stride


def _process_detections_for_frame(
    video_path: Path,
    frame_item: FrameItem,
    detections: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    config: PipelineConfig,
) -> bool:
    group_summary = summarize_detection_groups(
        detections=detections,
        class_groups=config.class_groups,
    )

    is_target_frame = frame_has_required_groups(
        group_summary=group_summary,
        required_groups=config.required_groups,
    )

    if not is_target_frame:
        return False

    frame_score = compute_frame_score(
        group_summary=group_summary,
        required_groups=config.required_groups,
    )

    owl_summary = group_summary.get("owl", _empty_group_summary())
    prey_summary = group_summary.get("prey", _empty_group_summary())

    candidate_rows.append(
        {
            "video_path": str(video_path),
            "frame_index": frame_item.frame_index,
            "timestamp_sec": frame_item.timestamp_sec,
            "is_target_frame": True,
            "frame_score": frame_score,
            "has_owl": bool(owl_summary["present"]),
            "owl_score": float(owl_summary["score"]),
            "owl_label": str(owl_summary["label"]),
            "owl_count": int(owl_summary["count"]),
            "has_prey": bool(prey_summary["present"]),
            "prey_score": float(prey_summary["score"]),
            "prey_label": str(prey_summary["label"]),
            "prey_count": int(prey_summary["count"]),
            "num_detections": len(detections),
            "detections_json": json.dumps(detections),
            "_image_bgr": frame_item.image_bgr,
        }
    )

    return True


def _process_batch(
    video_path: Path,
    batch_items: list[FrameItem],
    batch_detections: list[list[dict[str, Any]]],
    candidate_rows: list[dict[str, Any]],
    config: PipelineConfig,
) -> int:
    total_detections = 0

    for frame_item, detections in zip(batch_items, batch_detections):
        total_detections += len(detections)

        _process_detections_for_frame(
            video_path=video_path,
            frame_item=frame_item,
            detections=detections,
            candidate_rows=candidate_rows,
            config=config,
        )

    return total_detections


def process_video(
    video_path: str | Path,
    config: PipelineConfig,
) -> dict[str, Path]:
    config.validate()

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    total_start_time = perf_counter()
    video_metadata = get_video_metadata(video_path)

    output_dir = config.output_dir
    review_dir = output_dir / "review_top_frames"

    output_dir.mkdir(parents=True, exist_ok=True)

    if review_dir.exists():
        shutil.rmtree(review_dir)

    review_dir.mkdir(parents=True, exist_ok=True)

    detector = YoloDetector(
        model_path=config.model_path,
        conf=config.inference_conf,
        iou=config.inference_iou,
        imgsz=config.inference_imgsz,
        max_det=config.inference_max_det,
        device=config.inference_device,
        augment=config.inference_augment,
    )

    video_stem = _safe_stem(video_path)

    candidate_rows: list[dict[str, Any]] = []

    frame_extraction_runtime_sec = 0.0
    model_runtime_sec = 0.0
    frames_processed = 0
    total_detections = 0

    expected_frames_to_process = _expected_processed_frames(
        frame_count=video_metadata.frame_count,
        frame_stride=config.frame_stride,
    )

    frame_iterator = iter_video_frames(
        video_path=video_path,
        frame_stride=config.frame_stride,
        sample_fps=config.sample_fps,
    )

    progress = tqdm(
        frame_iterator,
        total=expected_frames_to_process,
        desc="Processing video",
        unit="frame",
    )

    batch_items: list[FrameItem] = []

    for frame_item in progress:
        frames_processed += 1

        if hasattr(frame_item, "read_runtime_sec"):
            frame_extraction_runtime_sec += frame_item.read_runtime_sec

        batch_items.append(frame_item)

        if len(batch_items) < config.inference_batch_size:
            continue

        model_start_time = perf_counter()

        batch_detections = detector.predict_batch(
            [item.image_bgr for item in batch_items]
        )

        model_runtime_sec += perf_counter() - model_start_time

        total_detections += _process_batch(
            video_path=video_path,
            batch_items=batch_items,
            batch_detections=batch_detections,
            candidate_rows=candidate_rows,
            config=config,
        )

        progress.set_postfix(
            {
                "targets": len(candidate_rows),
                "detections": total_detections,
            }
        )

        batch_items = []

    if batch_items:
        model_start_time = perf_counter()

        batch_detections = detector.predict_batch(
            [item.image_bgr for item in batch_items]
        )

        model_runtime_sec += perf_counter() - model_start_time

        total_detections += _process_batch(
            video_path=video_path,
            batch_items=batch_items,
            batch_detections=batch_detections,
            candidate_rows=candidate_rows,
            config=config,
        )

        progress.set_postfix(
            {
                "targets": len(candidate_rows),
                "detections": total_detections,
            }
        )

    candidates = pd.DataFrame(candidate_rows)

    if candidates.empty:
        top_frames = candidates.copy()
    else:
        top_frames = select_top_k_frames(
            frame_predictions=candidates.drop(columns=["_image_bgr"]),
            top_k=config.top_k,
            min_gap_seconds=config.min_gap_seconds,
        )

    top_rows: list[dict[str, Any]] = []

    for index, row in top_frames.iterrows():
        frame_index = int(row["frame_index"])
        timestamp_sec = float(row["timestamp_sec"])
        rank = index + 1

        source_match = candidates[candidates["frame_index"] == frame_index]

        if source_match.empty:
            continue

        source_row = source_match.iloc[0]
        image_bgr = source_row["_image_bgr"]
        detections = json.loads(str(source_row["detections_json"]))

        filename = _frame_filename(
            video_stem=video_stem,
            rank=rank,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )

        image_path = review_dir / filename

        annotated_image = draw_detections(
            image_bgr=image_bgr,
            detections=detections,
        )

        cv2.imwrite(str(image_path), annotated_image)

        row_dict = row.to_dict()
        row_dict["review_rank"] = rank
        row_dict["review_image_path"] = str(image_path)

        top_rows.append(row_dict)

    top_review = pd.DataFrame(top_rows)

    top_csv = output_dir / f"top{config.top_k}_for_review.csv"
    metadata_json = output_dir / "run_metadata.json"

    top_review.to_csv(top_csv, index=False)

    total_runtime_sec = perf_counter() - total_start_time

    metadata = {
        "video_path": str(video_path),
        "model_path": str(config.model_path),
        "video_duration_sec": video_metadata.duration_sec,
        "video_fps": video_metadata.fps,
        "video_frame_count": video_metadata.frame_count,
        "video_width": video_metadata.width,
        "video_height": video_metadata.height,
        "frames_processed": frames_processed,
        "target_frames_found": int(len(candidates)),
        "top_frames_saved": int(len(top_review)),
        "total_detections": total_detections,
        "frame_extraction_runtime_sec": frame_extraction_runtime_sec,
        "model_runtime_sec": model_runtime_sec,
        "total_runtime_sec": total_runtime_sec,
        "processing_fps": frames_processed / total_runtime_sec
        if total_runtime_sec > 0
        else 0.0,
        "inference": {
            "conf": config.inference_conf,
            "iou": config.inference_iou,
            "imgsz": config.inference_imgsz,
            "max_det": config.inference_max_det,
            "device": config.inference_device,
            "augment": config.inference_augment,
            "batch_size": config.inference_batch_size,
        },
        "frame_stride": config.frame_stride,
        "sample_fps": config.sample_fps,
        "class_groups": config.class_groups,
        "required_groups": config.required_groups,
        "top_k": config.top_k,
        "min_gap_seconds": config.min_gap_seconds,
    }

    metadata_json.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "top_for_review": top_csv,
        "metadata": metadata_json,
        "review_dir": review_dir,
    }