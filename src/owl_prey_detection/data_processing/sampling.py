from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


TIME_BUCKET_ORDER = ["night", "morning", "afternoon", "evening"]


@dataclass(slots=True)
class SamplingConfig:
    max_total: int = 120
    max_per_location_camera: int = 12
    max_per_time_bucket: int = 3
    min_gap_minutes: int = 60
    random_seed: int = 42


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched_df = df.copy()

    enriched_df["start_time"] = pd.to_datetime(enriched_df["start_time"], errors="coerce")
    enriched_df["end_time"] = pd.to_datetime(enriched_df["end_time"], errors="coerce")

    enriched_df = enriched_df.dropna(subset=["start_time", "end_time"])

    enriched_df["date"] = enriched_df["start_time"].dt.date
    enriched_df["hour"] = enriched_df["start_time"].dt.hour
    enriched_df["time_bucket"] = enriched_df["hour"].apply(_hour_to_bucket)

    return enriched_df


def _hour_to_bucket(hour: int) -> str:
    if 0 <= hour <= 5:
        return "night"
    if 6 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 17:
        return "afternoon"
    return "evening"


def select_balanced_pilot_sample(
    df: pd.DataFrame,
    config: SamplingConfig,
) -> pd.DataFrame:
    working_df = add_time_features(df)
    working_df = working_df.sort_values(
        by=["location_from_name", "camera_from_name", "start_time", "blob_name"]
    ).reset_index(drop=True)

    selected_indices: list[int] = []

    location_camera_groups = list(
        working_df.groupby(["location_from_name", "camera_from_name"], sort=True)
    )

    for (location, camera), group_df in location_camera_groups:
        group_selected_indices = _sample_within_location_camera_group(
            group_df=group_df,
            max_per_location_camera=config.max_per_location_camera,
            max_per_time_bucket=config.max_per_time_bucket,
            min_gap_minutes=config.min_gap_minutes,
        )

        selected_indices.extend(group_selected_indices)

    if not selected_indices:
        return working_df.iloc[0:0].copy()

    selected_df = working_df.loc[selected_indices].copy()
    selected_df = selected_df.sort_values("start_time").reset_index(drop=True)

    if len(selected_df) > config.max_total:
        selected_df = _downsample_globally(
            df=selected_df,
            max_total=config.max_total,
            random_seed=config.random_seed,
        )

    return selected_df.reset_index(drop=True)


def _sample_within_location_camera_group(
    group_df: pd.DataFrame,
    max_per_location_camera: int,
    max_per_time_bucket: int,
    min_gap_minutes: int,
) -> list[int]:
    selected_indices: list[int] = []
    bucket_counts: dict[str, int] = {bucket: 0 for bucket in TIME_BUCKET_ORDER}

    # first pass: enforce coverage across time buckets
    for bucket in TIME_BUCKET_ORDER:
        bucket_df = group_df[group_df["time_bucket"] == bucket].sort_values("start_time")

        for row in bucket_df.itertuples():
            if bucket_counts[bucket] >= max_per_time_bucket:
                break

            if len(selected_indices) >= max_per_location_camera:
                break

            if _is_far_enough(
                candidate_time=row.start_time,
                selected_times=group_df.loc[selected_indices, "start_time"]
                if selected_indices
                else pd.Series(dtype="datetime64[ns]"),
                min_gap_minutes=min_gap_minutes,
            ):
                selected_indices.append(row.Index)
                bucket_counts[bucket] += 1

    # second pass: fill remaining slots, preserving time gap
    if len(selected_indices) < max_per_location_camera:
        remaining_df = group_df.drop(index=selected_indices).sort_values("start_time")

        for row in remaining_df.itertuples():
            if len(selected_indices) >= max_per_location_camera:
                break

            if _is_far_enough(
                candidate_time=row.start_time,
                selected_times=group_df.loc[selected_indices, "start_time"]
                if selected_indices
                else pd.Series(dtype="datetime64[ns]"),
                min_gap_minutes=min_gap_minutes,
            ):
                selected_indices.append(row.Index)

    return selected_indices


def _is_far_enough(
    candidate_time: pd.Timestamp,
    selected_times: pd.Series,
    min_gap_minutes: int,
) -> bool:
    if selected_times.empty:
        return True

    min_gap = pd.Timedelta(minutes=min_gap_minutes)
    time_differences = (selected_times - candidate_time).abs()

    return bool((time_differences >= min_gap).all())


def _downsample_globally(
    df: pd.DataFrame,
    max_total: int,
    random_seed: int,
) -> pd.DataFrame:
    if len(df) <= max_total:
        return df.copy()

    # balanced trimming: shuffle within strata and keep fair representation
    shuffled_df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    selected_rows: list[dict[str, Any]] = []
    per_group_counts: dict[tuple[str, str], int] = {}

    group_sizes = (
        shuffled_df.groupby(["location_from_name", "camera_from_name"])
        .size()
        .sort_values(ascending=False)
    )

    total_groups = max(len(group_sizes), 1)
    base_quota = max_total // total_groups
    remainder = max_total % total_groups

    quotas: dict[tuple[str, str], int] = {}
    for index, ((location, camera), _) in enumerate(group_sizes.items()):
        quota = base_quota + (1 if index < remainder else 0)
        quotas[(location, camera)] = max(1, quota)

    for row in shuffled_df.itertuples(index=False):
        key = (row.location_from_name, row.camera_from_name)
        current_count = per_group_counts.get(key, 0)

        if current_count < quotas[key]:
            selected_rows.append(row._asdict())
            per_group_counts[key] = current_count + 1

        if len(selected_rows) >= max_total:
            break

    selected_df = pd.DataFrame(selected_rows)
    selected_df = selected_df.sort_values("start_time").reset_index(drop=True)

    return selected_df