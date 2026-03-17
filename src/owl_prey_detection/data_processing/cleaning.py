from __future__ import annotations

import pandas as pd


def clean_video_metadata(
    df: pd.DataFrame,
    min_duration_seconds: float = 5.0,
) -> pd.DataFrame:
    cleaned_df = df.copy()

    cleaned_df = cleaned_df[cleaned_df["is_video"] == True]
    cleaned_df = cleaned_df.dropna(
        subset=[
            "blob_name",
            "filename",
            "location_from_name",
            "camera_from_name",
            "start_time",
            "end_time",
            "duration_seconds",
        ]
    )

    cleaned_df["start_time"] = pd.to_datetime(cleaned_df["start_time"], errors="coerce")
    cleaned_df["end_time"] = pd.to_datetime(cleaned_df["end_time"], errors="coerce")

    cleaned_df = cleaned_df.dropna(subset=["start_time", "end_time"])
    cleaned_df = cleaned_df[cleaned_df["duration_seconds"] >= min_duration_seconds]
    cleaned_df = cleaned_df.drop_duplicates(subset=["blob_name"])

    return cleaned_df.reset_index(drop=True)