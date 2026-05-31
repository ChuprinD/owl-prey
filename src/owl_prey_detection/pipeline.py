import json
import shutil
from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from owl_prey_detection.detector import YoloDetector
from owl_prey_detection.frame_logic import (
    compute_frame_score,
    filter_detections_by_group_rules,
    frame_has_required_groups,
    summarize_detection_groups,
)
from owl_prey_detection.review import (
    save_top_frames,
    select_top_frames,
    top_k_filename,
)
from owl_prey_detection.video_io import (
    count_interval_frames,
    get_video_metadata,
    iter_video_frames,
    iter_video_frames_in_intervals,
    resolve_stride,
)


def _empty_group_summary():
    return {
        "present": False,
        "score": 0.0,
        "label": "",
        "count": 0,
    }


def _expected_processed_frames(frame_count, stride):
    if frame_count <= 0:
        return 0

    return (frame_count + stride - 1) // stride


def _create_detector(config):
    return YoloDetector(
        model_path=config.model_path,
        conf=config.inference_conf,
        iou=config.inference_iou,
        imgsz=config.inference_imgsz,
        max_det=config.inference_max_det,
        device=config.inference_device,
        augment=config.inference_augment,
    )


def _process_detections_for_frame(
    video_path,
    frame_item,
    detections,
    candidate_rows,
    config,
):
    group_summary = summarize_detection_groups(
        detections=detections,
        class_groups=config.class_groups,
    )

    if not frame_has_required_groups(group_summary, config.required_groups):
        return

    owl_summary = group_summary.get("owl", _empty_group_summary())
    prey_summary = group_summary.get("prey", _empty_group_summary())

    candidate_rows.append(
        {
            "video_path": str(video_path),
            "frame_index": frame_item.frame_index,
            "timestamp_sec": frame_item.timestamp_sec,
            "is_target_frame": True,
            "frame_score": compute_frame_score(
                group_summary,
                config.required_groups,
            ),
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


def _process_batch(
    video_path,
    batch_items,
    detector,
    candidate_rows,
    config,
):
    start_time = perf_counter()
    batch_detections = detector.predict_batch(
        [item.image_bgr for item in batch_items]
    )
    model_runtime_sec = perf_counter() - start_time
    total_detections = 0

    for frame_item, raw_detections in zip(batch_items, batch_detections):
        detections = filter_detections_by_group_rules(
            detections=raw_detections,
            class_groups=config.class_groups,
            min_conf_by_group=config.postprocess_min_conf_by_group,
        )
        total_detections += len(detections)
        _process_detections_for_frame(
            video_path=video_path,
            frame_item=frame_item,
            detections=detections,
            candidate_rows=candidate_rows,
            config=config,
        )

    return model_runtime_sec, total_detections


def _scan_frames(
    video_path,
    frame_iterator,
    expected_frames,
    config,
):
    candidates = []
    detector = None
    batch_items = []
    total_detections = 0
    frames_processed = 0
    frame_extraction_runtime_sec = 0.0
    model_runtime_sec = 0.0
    progress = tqdm(
        frame_iterator,
        total=expected_frames,
        desc="Processing video",
        unit="frame",
    )

    for frame_item in progress:
        frames_processed += 1
        frame_extraction_runtime_sec += frame_item.read_runtime_sec
        batch_items.append(frame_item)

        if len(batch_items) < config.inference_batch_size:
            continue

        if detector is None:
            detector = _create_detector(config)

        runtime_sec, detections_count = _process_batch(
            video_path=video_path,
            batch_items=batch_items,
            detector=detector,
            candidate_rows=candidates,
            config=config,
        )
        model_runtime_sec += runtime_sec
        total_detections += detections_count
        progress.set_postfix(
            targets=len(candidates),
            detections=total_detections,
        )
        batch_items = []

    if batch_items:
        if detector is None:
            detector = _create_detector(config)

        runtime_sec, detections_count = _process_batch(
            video_path=video_path,
            batch_items=batch_items,
            detector=detector,
            candidate_rows=candidates,
            config=config,
        )
        model_runtime_sec += runtime_sec
        total_detections += detections_count

    return (
        pd.DataFrame(candidates),
        frames_processed,
        total_detections,
        frame_extraction_runtime_sec,
        model_runtime_sec,
    )


def _inference_metadata(config):
    return {
        "conf": config.inference_conf,
        "iou": config.inference_iou,
        "imgsz": config.inference_imgsz,
        "max_det": config.inference_max_det,
        "device": config.inference_device,
        "augment": config.inference_augment,
        "batch_size": config.inference_batch_size,
    }


def process_video(video_path, config):
    config.validate()
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    total_start_time = perf_counter()
    video_metadata = get_video_metadata(video_path)
    mode = "audio_guided" if config.audio_enabled else "visual"
    output_dir = config.output_dir

    review_dir = output_dir / "review_top_frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    if review_dir.exists():
        shutil.rmtree(review_dir)

    review_dir.mkdir(parents=True)
    stride = resolve_stride(
        video_fps=video_metadata.fps,
        frame_stride=config.frame_stride,
        sample_fps=config.sample_fps,
    )
    audio_metadata = {}

    if config.audio_enabled:
        from owl_prey_detection.audio import analyze_audio

        audio_scan = analyze_audio(
            video_path=video_path,
            output_dir=output_dir,
            threshold_db=config.audio_threshold_db,
            min_event_seconds=config.audio_min_event_seconds,
            padding_seconds=config.audio_padding_seconds,
        )
        frame_iterator = iter_video_frames_in_intervals(
            video_path=video_path,
            intervals=audio_scan.intervals,
            frame_stride=config.frame_stride,
            sample_fps=config.sample_fps,
        )
        expected_frames = count_interval_frames(
            intervals=audio_scan.intervals,
            fps=video_metadata.fps,
            frame_count=video_metadata.frame_count,
            stride=stride,
        )
        audio_metadata = {
            "audio_duration_sec": audio_scan.duration_sec,
            "audio_analysis_runtime_sec": audio_scan.analysis_runtime_sec,
            "acoustic_events_detected": audio_scan.acoustic_events_detected,
            "merged_segments_count": len(audio_scan.intervals),
            "early_exit_triggered": not audio_scan.intervals,
            "audio": {
                "enabled": True,
                "threshold_db": config.audio_threshold_db,
                "padding_seconds": config.audio_padding_seconds,
                "min_event_seconds": config.audio_min_event_seconds,
                "output_dir": str(output_dir),
                "wav_path": str(audio_scan.output_wav),
            },
        }
    else:
        frame_iterator = iter_video_frames(
            video_path=video_path,
            frame_stride=config.frame_stride,
            sample_fps=config.sample_fps,
        )
        expected_frames = _expected_processed_frames(
            frame_count=video_metadata.frame_count,
            stride=stride,
        )

    (
        candidates,
        frames_processed,
        total_detections,
        frame_runtime,
        model_runtime,
    ) = _scan_frames(
        video_path=video_path,
        frame_iterator=frame_iterator,
        expected_frames=expected_frames,
        config=config,
    )
    top_frames = select_top_frames(
        candidates=candidates,
        top_k=config.top_k,
        min_gap_seconds=config.min_gap_seconds,
    )
    top_review = save_top_frames(
        candidates=candidates,
        top_frames=top_frames,
        review_dir=review_dir,
        video_path=video_path,
    )
    top_csv = output_dir / top_k_filename(config.top_k)
    metadata_json = output_dir / "run_metadata.json"
    top_review.to_csv(top_csv, index=False)
    total_runtime_sec = perf_counter() - total_start_time
    metadata = {
        "pipeline_mode": mode,
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
        "frame_extraction_runtime_sec": frame_runtime,
        "model_runtime_sec": model_runtime,
        "total_runtime_sec": total_runtime_sec,
        "processing_fps": (
            frames_processed / total_runtime_sec
            if total_runtime_sec > 0
            else 0.0
        ),
        "inference": _inference_metadata(config),
        "postprocess": {
            "min_conf_by_group": config.postprocess_min_conf_by_group,
        },
        "frame_stride": config.frame_stride,
        "sample_fps": config.sample_fps,
        "class_groups": config.class_groups,
        "required_groups": config.required_groups,
        "top_k": config.top_k,
        "min_gap_seconds": config.min_gap_seconds,
        **audio_metadata,
    }

    if config.audio_enabled:
        metadata["frames_scanned"] = frames_processed

    metadata_json.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "top_for_review": top_csv,
        "metadata": metadata_json,
        "review_dir": review_dir,
    }
