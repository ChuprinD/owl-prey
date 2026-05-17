from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st


REVIEW_CSV_PATTERN = re.compile(r"top(?P<top_k>\d+)_for_review\.csv$")


def find_review_csv(run_path: Path) -> Path | None:
    metadata_path = run_path / "run_metadata.json"

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            top_k = int(metadata["top_k"])
            expected_path = run_path / f"top{top_k}_for_review.csv"

            if expected_path.exists():
                return expected_path
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass

    candidates = [
        path
        for path in run_path.glob("top*_for_review.csv")
        if REVIEW_CSV_PATTERN.fullmatch(path.name)
    ]

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


st.set_page_config(
    page_title="Owl Prey Review",
    layout="wide",
)

st.title("Owl Prey Human-in-the-Loop Review")

run_dir = st.sidebar.text_input(
    "Run directory",
    value="runs/demo",
)

run_path = Path(run_dir)
review_csv_path = find_review_csv(run_path)
validated_path = run_path / "validated_predictions.csv"

if review_csv_path is None:
    st.error(
        f"Could not find top{{K}}_for_review.csv in {run_path}. "
        "Run the pipeline first."
    )
    st.stop()

st.sidebar.caption(f"Review CSV: {review_csv_path.name}")

df = pd.read_csv(review_csv_path)

if df.empty:
    st.warning("No target frames were found.")
    st.stop()

validated_rows = []

for _, row in df.iterrows():
    rank = int(row.get("review_rank", 0))
    timestamp = float(row["timestamp_sec"])
    prey_label = str(row.get("prey_label", ""))
    prey_score = float(row.get("prey_score", 0.0))
    image_path = str(row.get("review_image_path", ""))

    st.divider()

    st.subheader(
        f"Candidate {rank}: "
        f"t={timestamp:.2f}s, "
        f"prey={prey_label}, "
        f"confidence={prey_score:.3f}"
    )

    if image_path and Path(image_path).exists():
        st.image(image_path, use_container_width=True)
    else:
        st.warning(f"Image file missing: {image_path}")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        decision = st.radio(
            "Decision",
            options=[
                "accept",
                "correct",
                "reject_no_prey",
                "uncertain",
            ],
            key=f"decision_{rank}",
        )

    with col2:
        corrected_label = st.text_input(
            "Correct prey label",
            value=prey_label,
            key=f"label_{rank}",
        )

    with col3:
        notes = st.text_area(
            "Notes",
            value="",
            key=f"notes_{rank}",
        )

    row_dict = row.to_dict()

    row_dict.update(
        {
            "human_decision": decision,
            "human_corrected_label": corrected_label,
            "human_notes": notes,
        }
    )

    validated_rows.append(row_dict)

if st.button("Save validation"):
    validated = pd.DataFrame(validated_rows)
    validated.to_csv(validated_path, index=False)

    st.success(f"Saved: {validated_path}")
