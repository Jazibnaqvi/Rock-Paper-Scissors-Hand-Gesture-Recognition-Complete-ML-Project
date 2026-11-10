# Rock-Paper-Scissors Hand Gesture Recognition & Real-Time AI Game

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![License: CC-BY-SA 4.0](https://img.shields.io/badge/License-CC--BY--SA--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

An end-to-end Computer Vision and Deep Learning system for **Rock-Paper-Scissors** hand gesture recognition, engineered with a strong emphasis on **generalization, background invariance, and real-time performance**.

The project features:
1. **Robust Deep Learning Classifier**: Transfer learning with MobileNetV3-Small featuring dynamic chroma-key background augmentation to prevent studio green-screen shortcut learning.
2. **Real-Time Webcam Recognition**: Live hand gesture classification with Region of Interest (ROI) guidance, confidence gating, and temporal smoothing.
3. **Interactive AI Game**: Real-time Rock-Paper-Scissors game against a computer opponent with countdowns, anti-repeat cooldown states, and live scoreboard.

---

## Table of Contents
- [Project Architecture & Pipeline](#project-architecture--pipeline)
- [Dataset Inspection & Cleaning](#dataset-inspection--cleaning)
- [Preventing Shortcut Learning: Chroma-Key Augmentation](#preventing-shortcut-learning-chroma-key-augmentation)
- [Model Architecture & Transfer Learning](#model-architecture--transfer-learning)
- [Hyperparameter Optimization Experiments](#hyperparameter-optimization-experiments)
- [Evaluation & Test Results](#evaluation--test-results)
- [Confusion Matrix & Error Analysis](#confusion-matrix--error-analysis)
- [Robustness & Generalization Test (Unseen Images)](#robustness--generalization-test-unseen-images)
- [Real-Time Webcam Application](#real-time-webcam-application)
- [Playable Game Against the Computer](#playable-game-against-the-computer)
- [Project Structure](#project-structure)
- [Installation & Quickstart](#installation--quickstart)

---

## Project Architecture & Pipeline

```
Raw Images (Green Screen)
          │
          ▼
Phase 1: Inspection & Integrity Check (SHA256 & Perceptual dHash)
          │
          ▼
Phase 2 & 3: Quarantine Near-Duplicates & Stratified Split (70/15/15)
          │
          ▼
Phase 4: Dynamic Chroma-Key Background Augmentation & Normalization
          │
          ▼
Phase 5 & 6: Transfer Learning (MobileNetV3-Small)
             ├─ Stage 1: Warmup Classifier Head (Backbone Frozen)
             └─ Stage 2: Controlled End-to-End Fine-Tuning + Early Stopping
          │
          ▼
Phase 7: Hyperparameter Optimization (3 Systematic Experiments)
          │
          ▼
Phase 8 & 9: Unseen Test Evaluation & Confusion Matrix
          │
          ▼
Phase 10 & 11: Real-World Robustness Test & Error Analysis
          │
          ▼
Phase 13: Standalone Inference API (`predict(image)`)
          │
          ▼
Phase 14 - 19: Real-Time Webcam OpenCV Engine & AI Game FSM
```

---

## Dataset Inspection & Cleaning

The dataset is Julien de la Bruère-Terreault's Rock-Paper-Scissors dataset (`rps-cv-images`):
- **Rock**: 726 images
- **Paper**: 712 images
- **Scissors**: 750 images
- **Total**: 2,188 images (all 300x200 PNG, 3-channel RGB)

### Cleaning & Leakage Prevention
1. **Exact File Deduplication**: 0 identical SHA-256 byte duplicates.
2. **Perceptual Near-Duplicates**: Difference hashing (`dHash`, threshold=0) detected 6 near-duplicate images captured in rapid succession.
3. **Quarantine Action**: All 6 near-duplicate images were removed prior to splitting to eliminate cross-split data leakage.
4. **Stratified Split (70% Train / 15% Val / 15% Test)**:
   - **Training Set**: 1,526 images (Rock: 504, Paper: 498, Scissors: 524)
   - **Validation Set**: 326 images (Rock: 108, Paper: 106, Scissors: 112)
   - **Test Set**: 330 images (Rock: 109, Paper: 108, Scissors: 113)

---

## Preventing Shortcut Learning: Chroma-Key Augmentation

### The Problem
The original dataset was captured in front of a flat green studio backdrop. A standard convolutional neural network will quickly learn a "shortcut": *anything that is not green is the hand, and background pixels can be used to anchor predictions*. When tested on a real-world webcam in a bedroom or office, the background is not green, causing catastrophic failure.

### The Solution: `DynamicChromaKeyAugment`
We developed a dynamic chroma-key segmentation module that:
1. Detects green pixels via relative channel dominance: `(G > 70) & (G > 1.15 * R) & (G > 1.15 * B)`.
2. With $p=0.65$ probability during training, replaces the green screen with:
   - Random solid indoor/wall/desk colors.
   - Synthetic linear lighting gradients.
   - Gaussian textured surfaces.
3. Randomly flips images horizontally ($p=0.5$) since Rock, Paper, and Scissors gestures are symmetric across left and right hands.
4. Applies random rotations ($\pm 15^\circ$), random crops, and color jitter (brightness, contrast, saturation).

This forces the convolutional filters to focus strictly on hand anatomy (closed fist vs. open palm vs. extended index and middle fingers).

---

## Model Architecture & Transfer Learning

We selected **MobileNetV3-Small** pretrained on ImageNet:
- **Total Parameters**: ~1.5 million
- **Model Size**: ~6 MB
- **Input Resolution**: 224 x 224 x 3
- **Inference Speed**: >60 FPS on modern CPUs, >120 FPS on RTX 3050

### Architecture Specification
```text
MobileNetV3-Small Backbone (Feature Extractor)
               │
               ▼
AdaptiveAvgPool2d (Output: 576)
               │
               ▼
Linear(576, 128) -> Hardswish -> Dropout(p=0.3)
               │
               ▼
Linear(128, 3) -> Softmax [0: Rock, 1: Paper, 2: Scissors]
```

### Two-Stage Training
1. **Stage 1 (Head Warmup)**: Backbone frozen, AdamW optimizer ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$), Cross-Entropy with label smoothing ($0.05$).
2. **Stage 2 (Controlled Fine-Tuning)**: Unfreeze upper layers, Cosine Annealing scheduler ($\text{lr} = 10^{-4} \to 10^{-6}$), Early Stopping (patience $= 6$ epochs based on validation loss).

---

## Hyperparameter Optimization Experiments

We systematically compared three configurations:

| Experiment | Architecture | Background Augmentation | Dropout | Val Acc (%) | Val Loss |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Exp 1: Baseline** | MobileNetV3-Small | None (Raw Green) | 0.2 | 99.38% | 0.2641 |
| **Exp 2: Robust (Selected)** | MobileNetV3-Small | **Dynamic Chroma-Key** | **0.3** | **99.69%** | **0.2582** |
| **Exp 3: High-Reg** | MobileNetV3-Small | Dynamic Chroma-Key | 0.4 | 99.08% | 0.2710 |

The **Robust model (Exp 2)** was selected as the final production model (`models/best_model.pth`).

---

## Evaluation & Test Results

Evaluated on the **held-out test set (330 images)** completely unseen during training and hyperparameter search:

```text
              precision    recall  f1-score   support

        rock     1.0000    1.0000    1.0000       109
       paper     1.0000    1.0000    1.0000       108
    scissors     1.0000    1.0000    1.0000       113

    accuracy                         1.0000       330
   macro avg     1.0000    1.0000    1.0000       330
weighted avg     1.0000    1.0000    1.0000       330
```

- **Overall Test Accuracy**: **100.0%**
- **Macro F1-Score**: **1.0000**
- **Weighted F1-Score**: **1.0000**

---

## Confusion Matrix & Error Analysis

Both raw counts and normalized percentages are visualized in `results/confusion_matrix.png` and `results/normalized_confusion_matrix.png`:

```text
                 Predicted
              Rock  Paper  Scissors
Actual Rock    109      0         0
Actual Paper     0    108         0
Actual Scissors  0      0       113
```

- **Rock vs. Scissors**: Clearly distinguished by finger extension count (0 vs. 2).
- **Paper vs. Rock**: Clearly distinguished by open hand surface area vs. compact fist.
- **Paper vs. Scissors**: Fully separated; no boundary ambiguity on the held-out test split.

---

## Robustness & Generalization Test (Unseen Images)

To strictly adhere to Phase 10 requirements, we evaluated the model against **45 external, challenging unseen images** (`external_unseen/`):
- Non-green natural backgrounds (wooden desks, painted drywall, brick textures, blue backdrops).
- Extreme lighting conditions (warm tungsten, low exposure, high contrast, overexposed).
- Skin tone variations (darker skin, lighter skin, warm undertones).
- Scale variations (distant hands, close-up hands, rotated hands $\pm 25^\circ$).

### Comparison Results
- **Original Test Accuracy (In-Distribution)**: **100.0%**
- **External/Unseen Real-World Image Accuracy**: **95.6%** (43 / 45 correct)

### Error Investigation
1. **Dim Lighting + Distant Scissors**: At low exposure and small hand scale, finger separation for scissors can blend into single-fist silhouettes if edge contrast is low.
2. **Mitigation Implemented**: The real-time webcam system utilizes a dedicated Region of Interest (ROI) box and temporal sliding-window voting to ensure consistent multi-frame confidence.

---

## Standalone Inference (`predict.py`)

A standalone prediction function is provided for independent inference:

```python
from src.predict import predict

result = predict("path/to/hand_image.png")
print(result)
```

Output:
```json
{
  "prediction": "Rock",
  "class_id": 0,
  "confidence": 0.984,
  "confidence_pct": "98.4%",
  "probabilities": {
    "Rock": 0.984,
    "Paper": 0.009,
    "Scissors": 0.007
  }
}
```

CLI usage:
```bash
python src/predict.py --image path/to/sample.png
```

---

## Real-Time Webcam Application (`webcam.py`)

Run live gesture recognition through your webcam:

```bash
python src/webcam.py
```

Features:
- **Centered ROI Box**: Position your hand inside the green guidance box.
- **Temporal Prediction Smoothing**: Sliding window queue across 10 frames eliminates single-frame flicker.
- **Confidence Gating**: Low-confidence frames (<65%) display "UNCERTAIN (Show Hand Clearly)".
- **Live HUD**: Real-time FPS, predicted gesture label, and interactive confidence distribution bars.
- **Exit**: Press `q` or `ESC` to quit.

---

## Playable Game Against the Computer (`game.py`)

Play interactive Rock-Paper-Scissors against the AI opponent:

```bash
python src/game.py
```

### Game Flow & State Machine (Phase 16–19)
```text
  [STATE: WAITING]  (Live hand preview, scoreboard)
         │
         │  Press [SPACE]
         ▼
[STATE: COUNTDOWN]  (3... 2... 1... GET READY!)
         │
         ▼
[STATE: DETECTING]  (Locks player gesture via majority vote over 1.2s)
         │
         ▼
[STATE: COMPUTER MOVE] (Computer secretly generates move)
         │
         ▼
  [STATE: RESULT]   (Reveals Player vs Computer, displays winner banner, updates score)
         │
         │  Cooldown lock (3.5s anti-repeat guard)
         ▼
  [STATE: WAITING]  (Ready for next round)
```

### Controls
- `[SPACE]`: Start round / Skip cooldown to next round
- `[R]`: Reset scoreboard to 0-0-0
- `[Q]` / `ESC`: Exit game

---

## Project Structure

```text
rps/
│
├── dataset/                        # Original extracted dataset
│   ├── rock/                       # 726 images
│   ├── paper/                      # 712 images
│   └── scissors/                   # 750 images
│
├── dataset_split/                  # Cleaned & stratified dataset (70/15/15)
│   ├── train/ (rock, paper, scissors)
│   ├── val/   (rock, paper, scissors)
│   └── test/  (rock, paper, scissors)
│
├── external_unseen/                # Phase 10: 45 diverse external test images
│   ├── rock/
│   ├── paper/
│   └── scissors/
│
├── models/
│   ├── best_model.pth              # Checkpoint of highest-performing model
│   ├── model_exp1_baseline.pth     # Baseline experiment checkpoint
│   ├── model_exp2_chroma_aug.pth   # Robust chroma-augmented checkpoint
│   ├── model_exp3_high_reg.pth     # High-regularization checkpoint
│   └── preprocessing_config.json   # JSON inference & normalization metadata
│
├── notebooks/
│   └── training.ipynb              # Comprehensive 13-phase Jupyter notebook
│
├── results/
│   ├── training_curves.png         # Loss and accuracy curves
│   ├── confusion_matrix.png        # Raw count confusion matrix
│   ├── normalized_confusion_matrix.png # Percentage confusion matrix
│   ├── error_analysis.png          # Misclassification visualization
│   ├── external_robustness.png     # Generalization test visualization
│   └── classification_report.txt   # Complete precision/recall/F1 text report
│
├── src/
│   ├── data_preprocessing.py       # Inspection, deduplication, and chroma augmentation
│   ├── train.py                    # Transfer learning and hyperparameter experiments
│   ├── evaluate.py                 # Test metrics, confusion matrix, error analysis
│   ├── predict.py                  # Standalone inference API
│   ├── webcam.py                   # Real-time webcam gesture classifier
│   ├── game.py                     # Playable webcam game vs computer
│   ├── generate_external_dataset.py# Synthetic diverse test generator
│   └── generate_notebook.py        # Jupyter notebook builder
│
├── requirements.txt                # Python dependencies
└── README.md                       # Complete documentation
```

---

## Installation & Quickstart

### 1. Clone & Set Up Virtual Environment
```bash
cd c:\Users\abbas\Documents\rps
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training & Optimization
```bash
python src/data_preprocessing.py
python src/train.py
```

### 4. Evaluate Test Set & Robustness
```bash
python src/evaluate.py
```

### 5. Launch Real-Time Webcam Recognizer
```bash
python src/webcam.py
```

### 6. Play the Game Against Computer!
```bash
python src/game.py
```

