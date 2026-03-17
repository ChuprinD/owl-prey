# Owl Prey Detection — Project Guide

This document defines the architecture, development rules, and team workflow.

---

## 1. Project goal

The project builds an AI pipeline for analyzing owl nest-box video data.

Main objectives:
- access large-scale video data from Azure Blob Storage;
- work without downloading the full archive (metadata-first approach);
- extract and process frames;
- build datasets for object detection;
- train and evaluate models;
- run inference and support human validation workflows.

---

## 2. High-level pipeline

The system is organized as a sequence of independent stages.

Each stage:
- reads input artifacts;
- performs a single task;
- writes output artifacts.

Pipeline stages:

1. Inventory  
   Build metadata index of all blobs (no full download)

2. Sampling  
   Select representative subset of videos

3. Frame extraction  
   Extract frames from selected videos

4. Dataset building  
   Prepare YOLO-compatible dataset

5. Modeling  
   Train, evaluate, compute metrics

6. Application  
   Run inference, generate outputs, prepare human review

---

## 3. Project structure and responsibilities

### `configs/`
Configuration files (YAML):
- paths
- data settings
- training parameters
- inference parameters

---

### `notebooks/`
Used only for:
- EDA
- visualization
- debugging

Not allowed:
- core pipeline logic
- secrets
- hardcoded paths

---

### `src/owl_prey_detection/`

Core source code.

#### `config/`
Environment and config loading.

#### `data_access/`
External data interaction:
- Azure Blob Storage
- metadata extraction

#### `data_processing/`
Data preparation:
- sampling
- frame extraction
- filtering
- splitting

#### `annotation/`
Annotation validation and dataset construction.

#### `features/`
Optional feature engineering.

#### `modeling/`
Training, prediction, metrics, error analysis.

#### `application/`
Inference pipeline and reporting.

#### `utils/`
Shared helpers:
- paths
- logging
- IO
- visualization

#### `cli/`
Entry points for pipeline stages.

---

### `data/`

- `raw/` — downloaded subset only
- `interim/` — intermediate artifacts
- `processed/` — model-ready datasets

---

### `models/`
Saved models and checkpoints.

---

### `logs/`
Execution logs.

---

### `tests/`
Unit tests.

---

## 4. Core design principles

1. Secrets are never stored in code  
   Use `.env` only

2. No hardcoded paths  
   Use `paths.yaml` + path utilities

3. One module = one responsibility  

4. Each stage is independently runnable  

5. Each stage produces explicit outputs  

6. Avoid recomputation  
   Reuse saved artifacts

---

## 5. Developer workflow

### Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## 6. Environment variables and secrets

Sensitive data (e.g. Azure credentials) must NOT be stored in code or configuration files.

### Setup

Each developer must create a `.env` file in the project root:

.env

Add the following variables:

AZURE_CONNECTION_STRING=your_connection_string_here
AZURE_CONTAINER_NAME=your_container_name_here

Replace the values with your own credentials.

---

### Security rules

- `.env` must be included in `.gitignore`
- Never commit real credentials to Git
- Do not hardcode secrets in Python files
- Do not store secrets in YAML configs (`configs/*.yaml`)
- Do not share `.env` files between developers

---

### How secrets are used

All environment variables must be accessed through a centralized configuration layer.

Direct access like this is allowed only inside config modules:

import os

connection_string = os.getenv("AZURE_CONNECTION_STRING")
container_name = os.getenv("AZURE_CONTAINER_NAME")

Other parts of the codebase must NOT call `os.getenv` directly.

---

### .env.example

The repository should include a template file:

.env.example

Example:

AZURE_CONNECTION_STRING=your_connection_string_here
AZURE_CONTAINER_NAME=your_container_name_here

This allows new developers to quickly understand required variables without exposing secrets.

---

### Summary

- .env = local secrets (never committed)
- .env.example = template (committed)
- YAML configs = no secrets
- Code = no hardcoded credentials