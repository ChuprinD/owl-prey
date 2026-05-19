from __future__ import annotations

from typing import Any


def normalize_label(label: str) -> str:
    return label.lower().strip()


def _normalized_group_labels(
    class_groups: dict[str, list[str]],
) -> dict[str, set[str]]:
    return {
        group_name: {normalize_label(label) for label in labels}
        for group_name, labels in class_groups.items()
    }


def _detection_groups(
    detection: dict[str, Any],
    normalized_groups: dict[str, set[str]],
) -> list[str]:
    detection_label = normalize_label(str(detection["label"]))

    return [
        group_name
        for group_name, group_labels in normalized_groups.items()
        if detection_label in group_labels
    ]


def _bbox_area_ratio(
    detection: dict[str, Any],
    image_width: int,
    image_height: int,
) -> float:
    if image_width <= 0 or image_height <= 0:
        return 0.0

    x1, y1, x2, y2 = [float(value) for value in detection["xyxy"]]
    bbox_width = max(0.0, x2 - x1)
    bbox_height = max(0.0, y2 - y1)

    return (bbox_width * bbox_height) / float(image_width * image_height)


def filter_detections_by_group_rules(
    detections: list[dict[str, Any]],
    class_groups: dict[str, list[str]],
    image_width: int,
    image_height: int,
    min_conf_by_group: dict[str, float] | None = None,
    max_area_ratio_by_group: dict[str, float] | None = None,
    min_area_ratio_by_group: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    min_conf_by_group = min_conf_by_group or {}
    max_area_ratio_by_group = max_area_ratio_by_group or {}
    min_area_ratio_by_group = min_area_ratio_by_group or {}

    normalized_groups = _normalized_group_labels(class_groups)
    filtered: list[dict[str, Any]] = []

    for detection in detections:
        matching_groups = _detection_groups(
            detection=detection,
            normalized_groups=normalized_groups,
        )

        if not matching_groups:
            filtered.append(detection)
            continue

        confidence = float(detection["confidence"])
        area_ratio = _bbox_area_ratio(
            detection=detection,
            image_width=image_width,
            image_height=image_height,
        )

        should_keep = True

        for group_name in matching_groups:
            min_conf = min_conf_by_group.get(group_name)
            max_area_ratio = max_area_ratio_by_group.get(group_name)
            min_area_ratio = min_area_ratio_by_group.get(group_name)

            if min_conf is not None and confidence < min_conf:
                should_keep = False
                break

            if max_area_ratio is not None and area_ratio > max_area_ratio:
                should_keep = False
                break

            if min_area_ratio is not None and area_ratio < min_area_ratio:
                should_keep = False
                break

        if should_keep:
            filtered.append(detection)

    return filtered


def summarize_detection_groups(
    detections: list[dict[str, Any]],
    class_groups: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    normalized_groups = _normalized_group_labels(class_groups)

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
