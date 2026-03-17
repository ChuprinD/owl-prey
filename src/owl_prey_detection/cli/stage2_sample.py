from __future__ import annotations

from pprint import pprint

import pandas as pd

from owl_prey_detection.config.data_config import load_data_config
from owl_prey_detection.data_processing.cleaning import clean_video_metadata
from owl_prey_detection.data_processing.sampling import (
    SamplingConfig,
    select_balanced_pilot_sample,
)
from owl_prey_detection.utils.paths import (
    BLOB_INDEX_DIR,
    METADATA_TABLES_DIR,
    ensure_directories,
)


def main() -> None:
    print("Stage 2: pilot dataset sampling started")

    ensure_directories()

    data_config = load_data_config()

    sampling_cfg = data_config["sampling"]
    cleaning_cfg = data_config["cleaning"]

    input_path = BLOB_INDEX_DIR / "video_index.csv"
    clean_output_path = METADATA_TABLES_DIR / "clean_video_index.csv"
    sample_output_path = METADATA_TABLES_DIR / "pilot_sample.csv"

    print(f"Reading input metadata: {input_path}")
    video_df = pd.read_csv(input_path)

    print("Cleaning metadata...")
    clean_df = clean_video_metadata(
        video_df,
        min_duration_seconds=cleaning_cfg["min_duration_seconds"],
    )
    clean_df.to_csv(clean_output_path, index=False)

    sampling_config = SamplingConfig(
        max_total=sampling_cfg["max_total"],
        max_per_location_camera=sampling_cfg["max_per_location_camera"],
        max_per_time_bucket=sampling_cfg["max_per_time_bucket"],
        min_gap_minutes=sampling_cfg["min_gap_minutes"],
        random_seed=sampling_cfg["random_seed"],
    )

    print("Selecting balanced pilot sample...")
    sample_df = select_balanced_pilot_sample(clean_df, config=sampling_config)
    sample_df.to_csv(sample_output_path, index=False)

    summary = {
        "input_videos": int(len(video_df)),
        "clean_videos": int(len(clean_df)),
        "pilot_sample_videos": int(len(sample_df)),
        "unique_locations": int(sample_df["location_from_name"].nunique())
        if not sample_df.empty
        else 0,
        "unique_cameras": int(sample_df["camera_from_name"].nunique())
        if not sample_df.empty
        else 0,
    }

    print("Stage 2 completed successfully")
    print(f"Clean metadata saved to: {clean_output_path}")
    print(f"Pilot sample saved to: {sample_output_path}")
    pprint(summary)


if __name__ == "__main__":
    main()