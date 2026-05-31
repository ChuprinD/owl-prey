import json

from owl_prey_detection.config import PipelineConfig
from owl_prey_detection.data_access.video_source import resolve_input_video
from owl_prey_detection.pipeline import process_video


def format_seconds(seconds):
    seconds = float(seconds)

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"

    if minutes > 0:
        return f"{minutes}m {secs:.1f}s"

    return f"{secs:.1f}s"


def main():
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
    print(f"Mode: {metadata.get('pipeline_mode', 'visual')}")
    print(f"Video duration: {format_seconds(metadata['video_duration_sec'])}")
    if "audio_analysis_runtime_sec" in metadata:
        print(
            "Audio analysis time: "
            f"{format_seconds(metadata['audio_analysis_runtime_sec'])}"
        )
    print(f"Model time: {format_seconds(metadata['model_runtime_sec'])}")
    print(f"Total time: {format_seconds(metadata['total_runtime_sec'])}")
    print(
        "Frames processed: "
        f"{metadata.get('frames_processed', metadata.get('frames_scanned', 0))}"
    )
    print(f"Target frames found: {metadata['target_frames_found']}")
    print(f"Top frames saved: {metadata['top_frames_saved']}")
    print(f"Review CSV: {outputs['top_for_review']}")
    print(f"Review frames: {outputs['review_dir']}")


if __name__ == "__main__":
    main()
