# Data Pipeline

The data pipeline consists of two main stages: Frame Extraction and Annotation.

## 1. Frame Extraction
Using `notebooks/01_frame_extraction.ipynb`, the pipeline connects to Azure Blob Storage and processes videos using multiprocessing. It identifies frames containing motion (using structural similarity index) and saves them locally.

## 2. Workbench Annotation
The Workbench application (`app/main.py`) provides tools for dataset creation:
* **Scout Mode**: Uses a pre-trained model to pre-annotate new data.
* **Re-annotate Mode**: Allows reviewing and modifying existing annotations in the dataset, adding new classes dynamically.

## Supported Classes & Dataset Evolution

The annotation classes evolved significantly across the dataset versions to consolidate highly specific classes with low representation:

### 1. V1 Dataset (14 Granular Classes)
Initially trained on CH2 only, this dataset used a highly granular schema. Due to sparse samples in specific bins, it suffered from classification performance issues:
* **Rodents**: Separated into `bosmuis` (wood mouse), `spitsmuis` (shrew), `veldmuis` (field vole), and `woelrat` (water vole).
* **Insects/Invertebrates**: Separated into `larve` (larva), `loopkever` (ground beetle), `meikever` (cockchafer), `nachtvlinder` (moth), `regenworm` (earthworm), `spin` (spider), and `sprinkhaan` (grasshopper).
* **Birds & Owls**: Separated into `steenuil` (little owl), `vogel` (other birds), and `vogel_prooi` (bird prey).

### 2. V2 Nano Dataset (7 Consolidated Classes)
To solve data scarcity, V2 Nano grouped similar species together and focused exclusively on CH1 & CH2 (no nest box interior, so no eggs):
* **Rodents**: All grouped under `knaagdier`.
* **Insects**: Grouped `loopkever`, `meikever`, and `sprinkhaan` under a unified `kever` (beetle) class. Omitted sparse classes like `spin` (spider).
* **Birds**: Merged `vogel_prooi` into `vogel`.
* **Summary Schema**: `0: knaagdier`, `1: larve`, `2: kever`, `3: nachtvlinder`, `4: regenworm`, `5: steenuil`, `6: vogel`.

### 3. V2 Small Dataset (8 Consolidated Classes + Egg)
V2 Small expands channel coverage to include CH3 & CH4 (nest box interior) for owl tracking. Because nest boxes contain eggs, the `ei` (egg) class is introduced:
* **Egg Addition**: Adds `ei` (egg) at index 5 (ch3 only).
* **Shifted Indexes**: Pushes `steenuil` to index 6 and `vogel` to index 7.
* **Summary Schema**: `0: knaagdier`, `1: larve`, `2: kever`, `3: nachtvlinder`, `4: regenworm`, `5: ei`, `6: steenuil`, `7: vogel`.

