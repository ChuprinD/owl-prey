# Owl Prey Detection

Pipeline for finding likely owl-with-prey moments in nest-box videos. The project uses a trained Ultralytics YOLO model, scans video frames from either a local file or Azure Blob Storage, ranks frames that contain the configured detection groups, and writes review artifacts for human validation.

## What This Project Does

- Downloads or opens a configured video.
- Runs YOLO object detection on sampled frames.
- Groups model labels into semantic groups such as `owl` and `prey`.
- Keeps frames that contain all `required_groups`.
- Ranks candidate frames by the weakest required-group confidence score.
- Saves annotated top frames, a CSV for review, and run metadata.
- Provides a Streamlit app for human-in-the-loop validation.

## Repository Structure

```text
.
|-- app_review.py                       # Streamlit review UI
|-- config.yaml                         # Main pipeline configuration
|-- models/
|   `-- best.pt                         # Trained YOLO weights
|-- src/owl_prey_detection/
|   |-- cli.py                          # Pipeline entry point
|   |-- config.py                       # YAML config loading and validation
|   |-- detector.py                     # Ultralytics YOLO wrapper
|   |-- draw.py                         # Detection annotation rendering
|   |-- frame_logic.py                  # Grouping and target-frame logic
|   |-- pipeline.py                     # End-to-end video processing
|   |-- ranking.py                      # Top-K frame selection
|   |-- video_io.py                     # OpenCV video reading
|   `-- data_access/                    # Azure Blob access and input resolution
|-- data/                               # Local data, downloads, indexes, examples
|-- docs/                               # Project notes and architecture guide
|-- notebooks/                          # EDA and data preparation notebooks
`-- runs/                               # Pipeline outputs
```

## Requirements

- Python 3.11 or newer is recommended.
- A trained YOLO model at `models/best.pt`.
- For Azure input: a valid Azure Storage connection string and container name.
- For GPU inference: a compatible PyTorch/CUDA installation. The pinned `requirements.txt` can also run on CPU, but inference will be slower.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
pip install --no-build-isolation -e .
```

## Azure Credentials

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```env
AZURE_CONNECTION_STRING=your_connection_string_here
AZURE_CONTAINER_NAME=rawdata
```

`.env` is ignored by Git and should contain local secrets only.

## Configuration

The pipeline reads `config.yaml`.

Important fields:

```yaml
model_path: models/best.pt
output_dir: runs/demo

input:
  source: azure_blob        # local or azure_blob
  blob_name: "Vosseveld/Vosseveld_ch2_20240525013955_20240525014035.mp4"
  video_path: data/videos/example.mp4
  download_dir: data/videos/downloaded
  overwrite_download: false

frame_stride: 1
sample_fps: null

inference:
  conf: 0.5
  iou: 0.6
  imgsz: 640
  max_det: 10
  device: null
  augment: false
  batch_size: 32

required_groups:
  - owl
  - prey

top_k: 3
min_gap_seconds: 1.0
```

Input modes:

- `input.source: azure_blob` downloads `input.blob_name` into `input.download_dir` unless it already exists and `overwrite_download` is `false`.
- `input.source: local` reads `input.video_path` directly.

Sampling controls:

- `frame_stride` processes every Nth frame.
- `sample_fps` overrides `frame_stride` by deriving a stride from the video FPS.

Detection controls:

- `class_groups` maps YOLO labels to logical groups.
- `required_groups` defines which groups must be present for a frame to become a candidate.
- `top_k` controls how many annotated frames are saved for review.
- `min_gap_seconds` avoids selecting near-duplicate top frames when possible.

## Run the Pipeline

From the repository root:

```powershell
python -m owl_prey_detection.cli
```

The CLI always reads `config.yaml`, validates the model and input settings, resolves the video, runs inference, and prints a short runtime summary.

Generated outputs are written under `output_dir`, for example `runs/demo`:

```text
runs/demo/
|-- run_metadata.json
|-- top3_for_review.csv
`-- review_top_frames/
    `-- top_*.jpg
```

The CSV contains frame timestamps, owl/prey labels and confidence scores, detection JSON, and the path to each annotated review image.

## Review Predictions

Start the Streamlit app:

```powershell
streamlit run app_review.py
```

In the sidebar, set the run directory, such as `runs/demo`. The app shows each candidate image and lets a reviewer mark it as accepted, corrected, rejected, or uncertain. Clicking **Save validation** writes:

```text
runs/demo/validated_predictions.csv
```

## Output Files

- `top{K}_for_review.csv`: ranked candidate frames for human review.
- `review_top_frames/`: annotated JPEG images for the selected top frames.
- `run_metadata.json`: video metadata, runtime statistics, inference settings, frame counts, and selection settings.
- `validated_predictions.csv`: reviewer decisions from the Streamlit app.

## Development Notes

- Core reusable code lives in `src/owl_prey_detection/`.
- Notebooks are for exploration, inventory, and dataset preparation, not production pipeline logic.
- Azure secrets are read only through `src/owl_prey_detection/data_access/secret_access.py`.
- `data/` and `.env` are ignored by Git.

## Common Issues

### `ModuleNotFoundError: No module named 'owl_prey_detection'`

Install the project in editable mode from the repository root:

```powershell
pip install --no-build-isolation -e .
```

### `Environment variable AZURE_CONNECTION_STRING is not set`

Create `.env` from `.env.example` and fill in valid Azure credentials.

### `Model file not found`

Check that `models/best.pt` exists or update `model_path` in `config.yaml`.
