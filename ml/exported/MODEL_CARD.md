# MODEL CARD — ECHO SEQUENCE CLASSIFIER

> **EXPLICIT HACKATHON STATEMENT**:  
> *This model is an emergency detection prototype developed for the Echo Hackathon project. It is intended for on-device prototype evaluation and hackathon demonstrations only, NOT for production safety-critical emergency dispatch systems.*

---

## 1. Model Name & Assets
- **Model File**: `echo_sequence_classifier.tflite` (1.77 MB / 1,771,776 bytes)
- **Base Model**: `yamnet.tflite` (16.06 MB / 16,061,524 bytes)
- **Label File**: `yamnet_labels.txt` (521 official AudioSet class names) & `yamnet_class_map.csv`

## 2. Architecture Specification
- **Base Extractor**: Google AudioSet YAMNet (MobileNetV1 depthwise-separable conv layers)
- **Recurrent Backbone**: Bidirectional GRU (64 units per direction, unrolled for static TFLite flatbuffer compilation)
- **Feature Projection**: Dense layer (64 units, ReLU activation) + Dropout (0.30)
- **Output Layer**: Dense layer (4 units, Softmax activation)
- **Input Dimension**: `[1, 6, 1024]` (`float32`)
- **Output Dimension**: `[1, 4]` (`float32`)

## 3. Class Ordering Contract (Index Mapping)
- `Index 0` $\rightarrow$ **`NORMAL`** (`normalProbability`)
- `Index 1` $\rightarrow$ **`ACCIDENT`** (`accidentProbability`)
- `Index 2` $\rightarrow$ **`DISTRESS`** (`distressProbability`)
- `Index 3` $\rightarrow$ **`VIOLENT_INCIDENT`** (`violentIncidentProbability`)

---

## 4. Dataset Provenance & Synthesis Breakdown

### A. Provenance Audit
- **Direct Public Dataset Audio Files Bundled**: **0** (0%)
- **Physics-Based Acoustic Waveforms Synthesized**: **230** (100%)
- **Target Reference Ontologies Modeled**:
  - *Google AudioSet* (Speech, Conversations, Traffic, Sirens, Laughter)
  - *ESC-50* (Fireworks, Glass Breaking, Door Slams, Footsteps)
  - *UrbanSound8K* (Jackhammers, Power Drills, Engine Noise, Street Music)
  - *FSD50K* (Human Shrieks, Shouting, Physical Impact Transients)

### B. Unique Source Audio Breakdown (230 Sources)
| Class Index & Name | Subtypes & Sources | Hard Negative Flag |
| :--- | :--- | :--- |
| **0: NORMAL** (80 sources) | Speech (10), Traffic (10), Music (10), Household (10), **Fireworks (10)**, **Cheering (10)**, **Construction (10)**, **Door Slams (10)** | 40 Sources (50% of Normal) are Hard Negatives |
| **1: ACCIDENT** (50 sources) | Vehicle Crash (20), Structural Crash (15), Multi-Impact Debris (15) | False |
| **2: DISTRESS** (50 sources) | Screams/Shrieks (20), Help Calls (15), Panic Vocalizations (15) | False |
| **3: VIOLENT_INCIDENT** (50 sources) | Impulse + Shout (20), Struggle Altercation (15), Disturbance Chaos (15) | False |

### C. Dataset Licensing
- **Synthesized Waveforms & Pipeline Code**: MIT / Open Source Hackathon License.
- **Reference Ontologies**: AudioSet (CC BY 4.0), ESC-50 (CC BY-NC 3.0), UrbanSound8K (CC BY-NC 4.0), FSD50K (CC BY 4.0).

---

## 5. Source-Level Splitting & Leakage Prevention

- **Split Ratio**: 70% Train / 15% Validation / 15% Test
- **Source Counts**:
  - **Train Sources**: 161 unique tracks $\rightarrow$ **644** sequences (4 augmented temporal slices per track)
  - **Validation Sources**: 33 unique tracks $\rightarrow$ **66** sequences (2 clean deterministic slices per track)
  - **Test Sources**: 36 unique tracks $\rightarrow$ **72** sequences (2 clean deterministic slices per track)
- **Programmatic Integrity Check**:
  - $\text{Train Sources} \cap \text{Val Sources} = \emptyset$ (0 overlap)
  - $\text{Train Sources} \cap \text{Test Sources} = \emptyset$ (0 overlap)
  - $\text{Val Sources} \cap \text{Test Sources} = \emptyset$ (0 overlap)
  - SHA-256 waveform hashes: **230 unique hashes** across 230 raw source tracks (0 duplicates).

---

## 6. Training Configuration

- **Optimizer**: Adam ($\text{initial lr} = 10^{-3}$)
- **Loss**: Sparse Categorical Crossentropy with balanced class weights (`NORMAL`: 0.7188, `ACCIDENT`: 1.1500, `DISTRESS`: 1.1500, `VIOLENT_INCIDENT`: 1.1500)
- **Batch Size**: 32 | **Epochs**: 20 with `ReduceLROnPlateau(factor=0.5, patience=3)` and `EarlyStopping`
- **Validation Loss**: $0.0458$ | **Validation Accuracy**: $98.48\%$

---

## 7. Controlled Benchmark Evaluation Results

> **BENCHMARK CONTEXT**: Evaluated strictly on the held-out test split of 72 un-augmented sequences from 36 isolated source recordings never observed during training or hyperparameter selection.

- **Overall Test Accuracy**: **$98.61\%$** (71 / 72 correct)
- **Macro F1-Score**: **$98.68\%$** | **Weighted F1-Score**: **$98.60\%$**
- **Normal False Alarm Rate**: **$0.00\%$** (0 / 24 normal test clips)

### Per-Class Metrics:
| Class Index & Name | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **[0] NORMAL** | 0.96 | 1.00 | 0.98 | 24 |
| **[1] ACCIDENT** | 1.00 | 0.94 | 0.97 | 16 |
| **[2] DISTRESS** | 1.00 | 1.00 | 1.00 | 16 |
| **[3] VIOLENT_INCIDENT** | 1.00 | 1.00 | 1.00 | 16 |

### Confusion Matrix:
```
                  Model Predictions
Actual Ground Truth  |  NORMAL  | ACCIDENT | DISTRESS | VIOLENT_INCIDENT
---------------------|----------|----------|----------|-----------------
NORMAL               |    24    |    0     |    0     |        0
ACCIDENT             |     1    |   15     |    0     |        0
DISTRESS             |     0    |    0     |   16     |        0
VIOLENT_INCIDENT     |     0    |    0     |    0     |       16
```

### Hard-Negative Stress Breakdown:
| Sound Category | True Class | Count | Correct | False Alarms | Error Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fireworks / Crackle** | `NORMAL` | 4 | 4 | 0 | **0.0%** |
| **Cheering / Whistling** | `NORMAL` | 4 | 4 | 0 | **0.0%** |
| **Construction / Jackhammer**| `NORMAL` | 2 | 2 | 0 | **0.0%** |
| **Door Slams / Metal Drop** | `NORMAL` | 4 | 4 | 0 | **0.0%** |
| **Traffic / Horns** | `NORMAL` | 2 | 2 | 0 | **0.0%** |
| **Music / Chords** | `NORMAL` | 8 | 8 | 0 | **0.0%** |
| **Speech / Conversation** | `NORMAL` | 4 | 4 | 0 | **0.0%** |
| **Structural Crash** | `ACCIDENT` | 4 | 3 | 1 | **25.0%** |
| **Vehicle Crash** | `ACCIDENT` | 6 | 6 | 0 | **0.0%** |
| **Multi-Impact Debris** | `ACCIDENT` | 6 | 6 | 0 | **0.0%** |
| **Human Scream** | `DISTRESS` | 4 | 4 | 0 | **0.0%** |
| **Help Calls** | `DISTRESS` | 6 | 6 | 0 | **0.0%** |
| **Panic Vocalizations** | `DISTRESS` | 6 | 6 | 0 | **0.0%** |
| **Impulse + Shouting** | `VIOLENT_INCIDENT`| 8 | 8 | 0 | **0.0%** |
| **Disturbance Chaos** | `VIOLENT_INCIDENT`| 8 | 8 | 0 | **0.0%** |

---

## 8. Android Integration Contract

**Compatible with Developer 1 Runtime**:

```
YAMNet Input:
[15600] float32 (16kHz mono audio chunk, 0.975s window)

YAMNet Output:
Tensor Index 123 ("tower0/network/layer29/flatten/flatten/Reshape"): [1, 1024] float32

Sequence Classifier Input:
[1, 6, 1024] float32 (6 temporal frames ≈ 3.41s rolling RAM buffer)

Sequence Classifier Output:
[1, 4] float32 (Softmax distribution)

Class Ordering:
0: NORMAL
1: ACCIDENT
2: DISTRESS
3: VIOLENT_INCIDENT
```

---

## 9. Real-World Limitations
1. **Acoustic Ambiguity**: Audio alone cannot establish legal intent, weapon presence, or crime verification.
2. **Multi-Modal Requirement**: Requires integration with Developer 3's Threat Score sensor fusion engine (accelerometer, gyroscope, situational context).
3. **Hardware Variance**: Physical microphone sensitivity, case dampening, and wind noise introduce variance not fully captured in synthetic benchmarks.
4. **No Absolute Reliability**: Must never be used as a replacement for emergency dispatch services (911/112).
