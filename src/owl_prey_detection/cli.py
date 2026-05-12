from __future__ import annotations

import argparse
import json

from owl_prey_detection.config import PipelineConfig
from owl_prey_detection.data_access.video_source import resolve_input_video
from owl_prey_detection.pipeline import process_video


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"

    if minutes > 0:
        return f"{minutes}m {secs:.1f}s"

    return f"{secs:.1f}s"

def main() -> None:

    config = PipelineConfig.from_yaml("config.yaml")
    config.validate()

    video_path = resolve_input_video(config)

    outputs = process_video(
        video_path=video_path,
        config=config,
    )

    with outputs["metadata"].open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    print("Pipeline finished.")
    print(f"Video duration: {format_seconds(metadata['video_duration_sec'])}")
    print(f"Model time: {format_seconds(metadata['model_runtime_sec'])}")
    print(f"Total time: {format_seconds(metadata['total_runtime_sec'])}")
    print(f"Frames processed: {metadata['frames_processed']}")
    print(f"Target frames found: {metadata['target_frames_found']}")
    print(f"Top frames saved: {metadata['top_frames_saved']}")
    print(f"Top-5 CSV: {outputs['top5_for_review']}")
    print(f"Review frames: {outputs['review_dir']}")


if __name__ == "__main__":
    main()