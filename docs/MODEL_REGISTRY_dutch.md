# Modelregister

Deze map bevat de geschiedenis van YOLO-modellen die voor dit project zijn getraind.

## Beschikbare Modellen

### Hoofdvergelijking Modellen

| Model | Architectuur | Params | F1-Score | mAP50 | mAP50-95 | Opmerkingen / Evaluatie Split |
|---|---|---|---|---|---|---|
| **V2 Small YOLO12s** | YOLO12s | `9.26M` | `0.9863` | `0.9950` | `0.9431` | Productieanalyse (CH1-CH4) |
| **V2 Nano YOLO12n** | YOLO12n | `2.57M` | `0.8524` | `0.8940` | `0.7459` | Edge-implementatie (CH1-CH2) |
| **V2 Nano YOLO26n** | YOLO26n | `2.51M` | `0.8682` | `0.8906` | `0.7328` | Alternatief Nano-model (CH1-CH2) |
| **V1 Nano** | YOLO12n | `~2.57M` | `~0.75 - 0.80` | `-` | `-` | Geschat (alleen CH2, mist rigorous testen) |
| **Leiden Old (Gekoppeld)** | YOLOv8n | `3.01M` | `0.6642` | `0.6759` | `0.5664` | Alleen CH2 (Gekoppeld) |
| **Leiden Old (Gekoppeld)** | YOLOv8n | `3.01M` | `0.5679` | `0.6005` | `0.4488` | Volledige testset (CH1 & CH2) |

> [!TIP]
> **YOLO26 CPU-Optimalisatie**: YOLO26 is specifiek ontworpen om efficiënt op **CPUs** te draaien door de introductie van een native end-to-end NMS-vrije detectiekop, waarmee de standaard Non-Maximum Suppression (NMS) post-processing bottleneck wordt geëlimineerd.
> * **Prestaties**: Hoewel YOLO26n (`2.51M` parameters) een vergelijkbare nauwkeurigheid behaalt als YOLO12n (`2.57M` parameters), voert het sneller uit op de CPU.
> * **Snelheid**: In directe CPU-benchmarks behaalt YOLO26n een **latentiereductie van ~15-20%** (gemiddeld `43.4 ms` vs `52.1 ms` voor de volledige pipeline). Het kan tot meerdere malen sneller draaien in drukke scènes waarin NMS normaal gesproken de CPU-prestaties zou blokkeren.

---

### 1. V2 Small YOLO12s
* **Pad**: `models/v2_small_yolo12s`
* **Architectuur**: YOLO12s (261 epochs)
* **Klassen**: 8 geconsolideerde klassen (inclusief `ei` op ID 5)
* **Kanaaldekking**: CH1, CH2, CH3, CH4
* **Opmerkingen**: Het meest capabele model, ten zeerste aanbevolen voor productieanalyse.

### 2. V2 Nano YOLO12n
* **Pad**: `models/v2_nano_yolo12n`
* **Architectuur**: YOLO12n
* **Klassen**: 7 geconsolideerde klassen
* **Kanaaldekking**: CH1, CH2
* **Opmerkingen**: Aanbevolen voor lichte edge-implementaties.

### 3. V2 Nano YOLO26n
* **Pad**: `models/v2_nano_yolo26n`
* **Architectuur**: YOLO26n
* **Klassen**: 7 geconsolideerde klassen
* **Kanaaldekking**: CH1, CH2
* **Opmerkingen**: Alternatief Nano model. Iets lagere mAP maar betere F1 dan YOLO12n.

### 4. V1 Nano
* **Pad**: `models/v1_nano`
* **Architectuur**: YOLO12n
* **Klassen**: 14 gedetailleerde klassen
* **Kanaaldekking**: Alleen CH2
* **Opmerkingen**: Eerste iteratie, met expliciet onderscheid tussen muizensoorten en insecten. Behaalde deze mAP grofweg zonder grondige tests; het kreeg snel een kleine set nieuwe annotaties gevoed, waardoor de ware generaliseerbaarheid grotendeels onbekend is.

### 5. Leiden Old
* **Pad**: `models/leiden_old`
* **Architectuur**: YOLOv8n
* **Klassen**: 8 gedetailleerde klassen
* **Opmerkingen**: Originele baseline.
  > [!NOTE]
  > **Klassekoppeling Evaluatie (Class Mapping)**: Bij het evalueren van `leiden_old` tegen V2-datasets hebben we de voorspellingen gekoppeld aan de geconsolideerde V2-categorieën:
  > * Knaagdieren (`veldmuis` [7], `bosmuis` [0], `spitsmuis` [6]) &rarr; `knaagdier` [0]
  > * Larven (`rups` [5], `larve` [1]) &rarr; `larve` [1]
  > * Kevers (`meikever` [2]) &rarr; `kever` [2]
  > * Nachtvlinders (`nachtvlinder` [3]) &rarr; `nachtvlinder` [3]
  > * Regenwormen (`regenworm` [4]) &rarr; `regenworm` [4]

---

## Modeltuning & Optimale Drempelwaarden

Om de F1-score tijdens inferentie te optimaliseren, moeten de betrouwbaarheidsdrempels worden afgestemd.

### Algemene Aanbevelingen
* **V2 Modellen (Nano & Small)**: Prooi: `0.40`, Uil: `0.70`
* **V1 Modellen**: Prooi: `0.25`, Uil: `0.70`

### YOLO26n (v2n26) Optimale Drempelwaarden

| Klasse | Best Conf | F0.75 | Precision | Recall |
|---|---|---|---|---|
| knaagdier | 0.45 | 0.9288 | 0.9551 | 0.8854 |
| larve | 0.35 | 0.8442 | 0.9095 | 0.7486 |
| kever | 0.40 | 0.8452 | 0.8352 | 0.8636 |
| nachtvlinder | 0.50 | 0.8013 | 0.9167 | 0.6548 |
| regenworm | 0.50 | 0.8605 | 0.8761 | 0.8341 |
| steenuil | 0.65 | 0.9577 | 0.9670 | 0.9415 |
| vogel | 0.35 | 0.3788 | 0.8333 | 0.1923 |

### YOLO12n (v2n) Optimale Drempelwaarden (F0.75)

| Klasse | Best Conf | F0.75 | Precision | Recall |
|---|---|---|---|---|
| knaagdier | 0.55 | 0.9657 | 0.9889 | 0.9271 |
| larve | 0.40 | 0.8853 | 0.9130 | 0.8400 |
| kever | 0.60 | 0.8713 | 0.8824 | 0.8523 |
| nachtvlinder | 0.75 | 0.7794 | 0.8983 | 0.6310 |
| regenworm | 0.70 | 0.8982 | 0.9585 | 0.8079 |
| steenuil | 0.70 | 0.9680 | 0.9732 | 0.9587 |
| vogel | 0.40 | 0.7911 | 1.0000 | 0.5769 |

---

## Kanaaldekking van Datasets

* **V1**: Bevat alleen **CH2** (tweede camerakanaal).
* **V2 Nano (v2n)**: Getraind op **CH1** en **CH2** (eerste en tweede kanaal).
* **V2 Small (v2s)**: Volledig getraind op **CH1**, **CH2**, **CH3**, en **CH4**.
  > [!NOTE]
  > Voor andere klassen dan uilen op CH3 en CH4 (zoals prooien, kevers, larven), is het modelgedrag nog niet volledig gevalideerd. Aanvullende annotaties en trainingsgegevens zijn vereist voor betrouwbare multi-klasse detectie op CH3/CH4.

---

## Nieuwe Modellen Toevoegen
Om een nieuw model toe te voegen aan de Workbench, maakt u een nieuwe map aan in `models/`, plaatst u `best.pt` daarin, en voegt u een `manifest.json` bestand toe.
