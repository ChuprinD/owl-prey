def normalize_label(label):
    return label.lower().strip()


def _normalized_group_labels(class_groups):
    return {
        group_name: {normalize_label(label) for label in labels}
        for group_name, labels in class_groups.items()
    }


def _detection_groups(detection, normalized_groups):
    detection_label = normalize_label(str(detection["label"]))

    return [
        group_name
        for group_name, group_labels in normalized_groups.items()
        if detection_label in group_labels
    ]


def filter_detections_by_group_rules(
    detections,
    class_groups,
    min_conf_by_group=None,
):
    min_conf_by_group = min_conf_by_group or {}
    normalized_groups = _normalized_group_labels(class_groups)
    filtered = []

    for detection in detections:
        matching_groups = _detection_groups(
            detection=detection,
            normalized_groups=normalized_groups,
        )

        if not matching_groups:
            filtered.append(detection)
            continue

        confidence = float(detection["confidence"])
        should_keep = True

        for group_name in matching_groups:
            min_conf = min_conf_by_group.get(group_name)

            if min_conf is not None and confidence < min_conf:
                should_keep = False
                break

        if should_keep:
            filtered.append(detection)

    return filtered


def summarize_detection_groups(detections, class_groups):
    normalized_groups = _normalized_group_labels(class_groups)

    summary = {}

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


def frame_has_required_groups(group_summary, required_groups):
    return all(
        bool(group_summary.get(group_name, {}).get("present", False))
        for group_name in required_groups
    )


def compute_frame_score(group_summary, required_groups):
    if not frame_has_required_groups(group_summary, required_groups):
        return 0.0

    scores = [
        float(group_summary[group_name]["score"])
        for group_name in required_groups
    ]

    return min(scores)
