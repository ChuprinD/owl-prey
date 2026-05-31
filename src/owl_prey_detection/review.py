import json

import cv2
import pandas as pd


def _safe_stem(path):
    return path.stem.replace(" ", "_")


def _frame_filename(video_stem, rank, frame_index, timestamp_sec):
    return (
        f"top_{rank:02d}_"
        f"{video_stem}_"
        f"f{frame_index:08d}_"
        f"t{timestamp_sec:010.3f}.jpg"
    )


def top_k_filename(top_k):
    if top_k == "all":
        return "top_all_for_review.csv"

    return f"top{top_k}_for_review.csv"


def draw_detections(image_bgr, detections):
    annotated = image_bgr.copy()

    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection["xyxy"]]
        label = str(detection["label"])
        confidence = float(detection["confidence"])
        text = f"{label} {confidence:.2f}"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            text,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def select_top_k_frames(
    frame_predictions,
    top_k=5,
    min_gap_seconds=1.0,
):
    if frame_predictions.empty:
        return frame_predictions.copy()

    target_frames = frame_predictions[frame_predictions["is_target_frame"]].copy()

    if target_frames.empty:
        return target_frames

    target_frames = target_frames.sort_values(
        "frame_score",
        ascending=False,
    )
    chosen_rows = []
    chosen_times = []

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

    return pd.DataFrame(chosen_rows).sort_values(
        "frame_score",
        ascending=False,
    ).reset_index(drop=True)


def select_top_frames(candidates, top_k, min_gap_seconds):
    if candidates.empty:
        return candidates.copy()

    predictions = candidates.drop(columns=["_image_bgr"])

    if top_k == "all":
        return predictions.sort_values(
            "frame_score",
            ascending=False,
        ).reset_index(drop=True)

    return select_top_k_frames(
        frame_predictions=predictions,
        top_k=top_k,
        min_gap_seconds=min_gap_seconds,
    )


def save_top_frames(candidates, top_frames, review_dir, video_path):
    rows_by_frame = {
        int(row["frame_index"]): row
        for _, row in candidates.iterrows()
    }
    top_rows = []

    for index, row in top_frames.iterrows():
        frame_index = int(row["frame_index"])
        timestamp_sec = float(row["timestamp_sec"])
        rank = index + 1
        source_row = rows_by_frame.get(frame_index)

        if source_row is None:
            continue

        image_path = review_dir / _frame_filename(
            video_stem=_safe_stem(video_path),
            rank=rank,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )
        annotated_image = draw_detections(
            image_bgr=source_row["_image_bgr"],
            detections=json.loads(str(source_row["detections_json"])),
        )
        cv2.imwrite(str(image_path), annotated_image)
        output_row = row.to_dict()
        output_row["review_rank"] = rank
        output_row["review_image_path"] = str(image_path)
        top_rows.append(output_row)

    if top_rows:
        return pd.DataFrame(top_rows)

    return pd.DataFrame(
        columns=[
            "review_rank",
            "video_path",
            "frame_index",
            "timestamp_sec",
            "frame_score",
            "owl_score",
            "prey_score",
            "owl_label",
            "prey_label",
            "review_image_path",
        ]
    )
