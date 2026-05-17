from __future__ import annotations

import pandas as pd


def select_top_k_frames(frame_predictions: pd.DataFrame, top_k: int = 5, min_gap_seconds: float = 1.0,) -> pd.DataFrame:
    if frame_predictions.empty:
        return frame_predictions.copy()

    target_frames = frame_predictions[
        frame_predictions["is_target_frame"] == True
    ].copy()

    if target_frames.empty:
        return target_frames

    target_frames = target_frames.sort_values(
        "frame_score",
        ascending=False,
    )

    chosen_rows = []
    chosen_times: list[float] = []

    for _, row in target_frames.iterrows():
        timestamp = float(row["timestamp_sec"])

        is_far_enough = all(
            abs(timestamp - selected_time) >= min_gap_seconds
            for selected_time in chosen_times
        )

        if is_far_enough:
            chosen_rows.append(row)
            chosen_times.append(timestamp)

        if len(chosen_rows) >= top_k:
            break

    if len(chosen_rows) < top_k:
        chosen_indices = {int(row.name) for row in chosen_rows}

        for index, row in target_frames.iterrows():
            if int(index) not in chosen_indices:
                chosen_rows.append(row)

            if len(chosen_rows) >= top_k:
                break

    result = pd.DataFrame(chosen_rows)

    return result.sort_values(
        "frame_score",
        ascending=False,
    ).reset_index(drop=True)