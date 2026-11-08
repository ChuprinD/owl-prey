# Data Pijplijn

De data pijplijn bestaat uit twee hoofdfasen: Frame Extractie en Annotatie.

## 1. Frame Extractie
Via `notebooks/01_frame_extraction.ipynb` maakt de pijplijn verbinding met Azure Blob Storage en verwerkt het video's met behulp van multiprocessing. Het identificeert frames met beweging en slaat deze lokaal op.

## 2. Workbench Annotatie
De Workbench applicatie (`app/main.py`) biedt tools voor dataset creatie:
* **Scout Modus**: Gebruikt een voorgetraind model om nieuwe data voor te annoteren.
* **Re-annotate Modus**: Maakt het mogelijk om bestaande annotaties te beoordelen en te wijzigen, en dynamisch nieuwe klassen toe te voegen.

## Ondersteunde Klassen & Evolutie van de Dataset

De annotatieklassen zijn aanzienlijk geëvolueerd tussen de verschillende datasetversies om zeer specifieke klassen met een lage vertegenwoordiging samen te voegen:

### 1. V1 Dataset (14 Gedetailleerde Klassen)
In eerste instantie getraind op alleen CH2, maakte deze dataset gebruik van een zeer gedetailleerd schema. Door een tekort aan voorbeelden in specifieke groepen leed het model onder classificatieprestatieproblemen:
* **Knaagdieren**: Opgesplitst in `bosmuis`, `spitsmuis`, `veldmuis` en `woelrat`.
* **Insecten/Ongewervelden**: Opgesplitst in `larve`, `loopkever`, `meikever`, `nachtvlinder`, `regenworm`, `spin` en `sprinkhaan`.
* **Vogels & Uilen**: Opgesplitst in `steenuil`, `vogel` (overige vogels) en `vogel_prooi` (vogel als prooi).

### 2. V2 Nano Dataset (7 Geconsolideerde Klassen)
Om het datatekort op te lossen, heeft V2 Nano vergelijkbare soorten samengevoegd en zich uitsluitend gericht op CH1 & CH2 (geen nestkastinterieur, dus geen eieren):
* **Knaagdieren**: Allemaal samengevoegd onder `knaagdier`.
* **Insecten**: `loopkever`, `meikever` en `sprinkhaan` zijn samengevoegd onder een universele klasse `kever`. Zeer schaarse klassen zoals `spin` zijn weggelaten.
* **Vogels**: `vogel_prooi` is samengevoegd met `vogel`.
* **Overzicht Schema**: `0: knaagdier`, `1: larve`, `2: kever`, `3: nachtvlinder`, `4: regenworm`, `5: steenuil`, `6: vogel`.

### 3. V2 Small Dataset (8 Geconsolideerde Klassen + Ei)
V2 Small breidt de kanaaldekking uit met CH3 & CH4 (nestkastinterieur) voor het volgen van uilen. Omdat nestkasten eieren bevatten, is de klasse `ei` geïntroduceerd:
* **Toevoeging Ei**: Voegt `ei` toe op index 5.
* **Verschoven Indexen**: Verschuift `steenuil` naar index 6 en `vogel` naar index 7.
* **Overzicht Schema**: `0: knaagdier`, `1: larve`, `2: kever`, `3: nachtvlinder`, `4: regenworm`, `5: ei`, `6: steenuil`, `7: vogel`.

