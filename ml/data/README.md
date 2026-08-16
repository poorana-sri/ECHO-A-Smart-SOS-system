# ECHO ML Dataset Specification & Licensing

## Overview
This directory contains dataset preparation scripts, dataset metadata, and reproducible train/validation/test splits for the **ECHO** Emergency Detection System ML Pipeline.

Pipeline:
`3.41s rolling RAM audio buffer (16kHz mono float32) -> YAMNet Feature Extractor (1024-D temporal embeddings) -> Echo Sequence Classifier (Bidirectional GRU) -> [NORMAL, ACCIDENT, DISTRESS, VIOLENT_INCIDENT]`

---

## Dataset Classes & Definitions

### Class 0: `NORMAL`
* **Description**: Baseline acoustic environments, everyday ambient noise, and non-emergency loud sounds.
* **Included Acoustic Events**:
  - Conversations, background speech, laughter
  - Traffic, vehicle engines, sirens at distance, car horns
  - Music, concerts, radio, television
  - Crowds, cheering, applause, stadium noise [Hard Negative]
  - Construction noise, jackhammers, machinery, tools [Hard Negative]
  - Household sounds: door slams, dropped pots/pans, vacuum cleaners [Hard Negative]
  - Fireworks, firecrackers, acoustic transients [Hard Negative]
  - Running, footsteps, sports activities

### Class 1: `ACCIDENT`
* **Description**: Acoustic patterns associated with vehicular collisions, structural impacts, and mechanical accidents.
* **Included Acoustic Events**:
  - Vehicle crash impact sounds, metal deformation, glass shattering
  - Secondary impacts, tumbling objects, skidding tires
  - Structural crushing & debris falling

### Class 2: `DISTRESS`
* **Description**: Human vocal emergency indicators and panic acoustic sequences.
* **Included Acoustic Events**:
  - High-pitched human screams, shrieks, panic shouts
  - Repeated distress vocalizations ("Help!", screaming)
  - Agonized gasps, hyperventilation vocal patterns

### Class 3: `VIOLENT_INCIDENT`
* **Description**: Acoustic patterns associated with potentially dangerous physical altercations or violent incidents.
* **Included Acoustic Events**:
  - Sudden impulse transients combined with immediate human distress shouting
  - Repeated abnormal physical impacts accompanied by struggle/vocalization sequences
  - Sudden high-energy acoustic disruptions followed by chaos/screaming

---

## Source Inventory & Source-Level Splitting

- **Total Distinct Source Recordings**: **230** unique audio tracks
  - 80 NORMAL sources (including 40 explicit hard negatives: fireworks, cheering, construction, door slams)
  - 50 ACCIDENT sources
  - 50 DISTRESS sources
  - 50 VIOLENT_INCIDENT sources

- **Source-Level Partitioning Strategy**:
  - **Train Sources (70%)**: 161 unique source tracks $\rightarrow$ 644 augmented sequences
  - **Validation Sources (15%)**: 33 unique source tracks $\rightarrow$ 66 clean sequences
  - **Held-Out Test Sources (15%)**: 36 unique source tracks $\rightarrow$ 72 clean sequences

- **Leakage Prevention**:
  Unique `source_id` assignment occurs *before* temporal slicing and augmentation. Zero overlap exists between splits (`len(train_ids ∩ test_ids) == 0`).
