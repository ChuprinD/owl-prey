# Model Registry

This directory contains the history of YOLO models trained for this project.

## Available Models

### Master Model Comparison

| Model | Architecture | Params | F1 Score | mAP50 | mAP50-95 | Notes / Evaluation Split |
|---|---|---|---|---|---|---|
| **V2s** | YOLO12s | `9.26M` | `0.9863` | `0.9950` | `0.9431` | Production analysis (CH1-CH4) |
| **V2n** | YOLO12n | `2.57M` | `0.8524` | `0.8940` | `0.7459` | Edge deployment (CH1-CH2) |
| **V2n26** | YOLO26n | `2.51M` | `0.8682` | `0.8906` | `0.7328` | Alternative Nano (CH1-CH2) |
| **V1n** | YOLO12n | `~2.57M` | `~0.75 - 0.80` | `-` | `-` | Estimated (CH2 only, lacks testing) |
| **Leiden Old (Mapped)** | YOLOv8n | `3.01M` | `0.6642` | `0.6759` | `0.5664` | CH2-only mapped split |
| **Leiden Old (Mapped)** | YOLOv8n | `3.01M` | `0.5679` | `0.6005` | `0.4488` | Full test split (CH1-CH2) |

> [!TIP]
> **YOLO26 CPU Optimization**: YOLO26 is designed specifically to run efficiently on **CPUs** by introducing a native end-to-end NMS-free head, removing the standard Non-Maximum Suppression (NMS) post-processing bottleneck.
> * **Performance**: While YOLO26n (`2.51M` parameters) achieves similar accuracy to YOLO12n (`2.57M` parameters), it executes faster on CPU.
> * **Speed**: In direct CPU benchmarks, YOLO26n achieves a **~15-20% latency reduction** (Avg `43.4 ms` vs `52.1 ms` for the full pipeline), and can run up to multiple times faster in dense scenarios where NMS would normally bottleneck CPU performance.

---

### 1. V2 Small YOLO12s
* **Path**: `models/v2_small_yolo12s`
* **Architecture**: YOLO12s (261 epochs)
* **Classes**: 8 consolidated classes (includes `ei` at ID 5)
* **Channel Coverage**: CH1, CH2, CH3, CH4
* **Notes**: The most capable model, highly recommended for production analysis.

### 2. V2 Nano YOLO12n
* **Path**: `models/v2_nano_yolo12n`
* **Architecture**: YOLO12n
* **Classes**: 7 consolidated classes
* **Channel Coverage**: CH1, CH2
* **Notes**: Recommended for lightweight edge deployment.

### 3. V2 Nano YOLO26n
* **Path**: `models/v2_nano_yolo26n`
* **Architecture**: YOLO26n
* **Classes**: 7 consolidated classes
* **Channel Coverage**: CH1, CH2
* **Notes**: Alternative Nano model. Slightly lower mAP but better F1 than YOLO12n.

### 4. V1 Nano
* **Path**: `models/v1_nano`
* **Architecture**: YOLO12n
* **Classes**: 14 granular classes
* **Channel Coverage**: CH2 only
* **Notes**: First iteration, explicitly separates mouse species and insects. Achieved this mAP roughly without proper rigorous testing; it was quickly fed a small batch of new annotations, so its true generalizability is largely unknown.

### 5. Leiden Old
* **Path**: `models/leiden_old`
* **Architecture**: YOLOv8n
* **Classes**: 8 granular classes
* **Notes**: Original baseline.
  > [!NOTE]
  > **Class Mapping Evaluation**: When evaluating `leiden_old` against V2 datasets, we mapped its predictions to the V2 consolidated categories:
  > * Rodents (`veldmuis` [7], `bosmuis` [0], `spitsmuis` [6]) &rarr; `knaagdier` [0]
  > * Larvae (`rups` [5], `larve` [1]) &rarr; `larve` [1]
  > * Beetles (`meikever` [2]) &rarr; `kever` [2]
  > * Moths (`nachtvlinder` [3]) &rarr; `nachtvlinder` [3]
  > * Earthworms (`regenworm` [4]) &rarr; `regenworm` [4]

---

## Model Tuning & Optimal Thresholds

To optimize the F1-score during inference, confidence thresholds must be tuned.

### General Recommendations
* **V2 Models (Nano & Small)**: Prey: `0.40`, Owl: `0.70`
* **V1 Models**: Prey: `0.25`, Owl: `0.70`

### YOLO26n (v2n26) Optimal Class Thresholds

| Class | Best Conf | F0.75 | Precision | Recall |
|---|---|---|---|---|
| knaagdier | 0.45 | 0.9288 | 0.9551 | 0.8854 |
| larve | 0.35 | 0.8442 | 0.9095 | 0.7486 |
| kever | 0.40 | 0.8452 | 0.8352 | 0.8636 |
| nachtvlinder | 0.50 | 0.8013 | 0.9167 | 0.6548 |
| regenworm | 0.50 | 0.8605 | 0.8761 | 0.8341 |
| steenuil | 0.65 | 0.9577 | 0.9670 | 0.9415 |
| vogel | 0.35 | 0.3788 | 0.8333 | 0.1923 |

### YOLO12n (v2n) Optimal Class Thresholds

| Class | Best Conf | F0.75 | Precision | Recall |
|---|---|---|---|---|
| knaagdier | 0.55 | 0.9657 | 0.9889 | 0.9271 |
| larve | 0.40 | 0.8853 | 0.9130 | 0.8400 |
| kever | 0.60 | 0.8713 | 0.8824 | 0.8523 |
| nachtvlinder | 0.75 | 0.7794 | 0.8983 | 0.6310 |
| regenworm | 0.70 | 0.8982 | 0.9585 | 0.8079 |
| steenuil | 0.70 | 0.9680 | 0.9732 | 0.9587 |
| vogel | 0.40 | 0.7911 | 1.0000 | 0.5769 |

---

## Dataset Channel Coverage

* **V1**: Contains only **CH2** (second camera channel).
* **V2 Nano (v2n)**: Trained on **CH1** and **CH2** (first and second channels).
* **V2 Small (v2s)**: Fully trained on **CH1**, **CH2**, **CH3**, and **CH4**.
  > [!NOTE]
  > For classes other than owls on CH3 and CH4 (e.g. prey, beetles, larvae), model behavior is not fully validated yet. Additional annotation and training data are required for reliable multi-class detection on CH3/CH4.

---

## Adding New Models
To add a new model to the Workbench, create a new folder in `models/`, place `best.pt` inside it, and add a `manifest.json` file.
