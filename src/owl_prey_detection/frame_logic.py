from __future__ import annotations

from typing import Any


def normalize_label(label: str) -> str:
    return label.lower().strip()


def summarize_detection_groups(
    detections: list[dict[str, Any]],
    class_groups: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    normalized_groups = {
        group_name: {normalize_label(label) for label in labels}
        for group_name, labels in class_groups.items()
    }

    summary: dict[str, dict[str, Any]] = {}

    for group_name in class_groups:
        summary[group_name] = {
            "present": False,
            "score": 0.0,
            "label": "",
            "count": 0,
        }

    for detection in detections:
        detection_label = normalize_label(str(detection["label"]))
        confidence = float(detection["confidence"])

        for group_name, group_labels in normalized_groups.items():
            if detection_label not in group_labels:
                continue

            summary[group_name]["present"] = True
            summary[group_name]["count"] += 1

            if confidence > float(summary[group_name]["score"]):
                summary[group_name]["score"] = confidence
                summary[group_name]["label"] = str(detection["label"])

    return summary


def frame_has_required_groups(
    group_summary: dict[str, dict[str, Any]],
    required_groups: list[str],
) -> bool:
    return all(
        bool(group_summary.get(group_name, {}).get("present", False))
        for group_name in required_groups
    )


def compute_frame_score(
    group_summary: dict[str, dict[str, Any]],
    required_groups: list[str],
) -> float:
    if not frame_has_required_groups(group_summary, required_groups):
        return 0.0

    scores = [
        float(group_summary[group_name]["score"])
        for group_name in required_groups
    ]

    return min(scores)