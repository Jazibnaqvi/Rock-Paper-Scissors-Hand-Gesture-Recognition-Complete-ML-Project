"""
generate_notebook.py - Generates notebooks/training.ipynb containing all 13 phases.
"""

import json
from pathlib import Path


def create_jupyter_notebook(output_path: str):
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.4"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(code):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.strip().split("\n")]
        })

    # Header
    add_md("""# Rock-Paper-Scissors Hand Gesture Recognition & Real-Time AI System
**End-to-End Deep Learning, Transfer Learning, Robustness Optimization, and Real-Time Webcam Application**

---
## Project Overview
This project builds a robust, production-grade Computer Vision system capable of:
1. Learning visual characteristics of Rock, Paper, and Scissors hand gestures.
2. Overcoming shortcut background learning (studio green screen) to generalize to real-world environments.
3. Powering a real-time interactive webcam game against an AI opponent.
""")

    # Phase 1
    add_md("""---
## PHASE 1 — Dataset Inspection and Preparation
We inspect the dataset containing 2,188 labeled images across 3 classes: `rock`, `paper`, `scissors`.
""")
    add_code("""import os
import sys
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(str(Path.cwd().parent / "src"))
from data_preprocessing import inspect_dataset, CLASS_NAMES

data_dir = str(Path.cwd().parent / "dataset")
report = inspect_dataset(data_dir)
""")

    add_code("""# Display representative samples from each class
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
for r, cls in enumerate(CLASS_NAMES):
    cls_folder = Path(data_dir) / cls
    sample_files = list(cls_folder.glob("*.png"))[:4]
    for c, sf in enumerate(sample_files):
        img = Image.open(sf)
        axes[r, c].imshow(img)
        axes[r, c].set_title(f"{cls.capitalize()} ({sf.name[:8]}...)", fontsize=10, fontweight="bold")
        axes[r, c].axis("off")
plt.suptitle("Representative Samples from Original Dataset", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
""")

    # Phase 2 & 3
    add_md("""---
## PHASE 2 & 3 — Data Cleaning and Stratified Splitting
We remove corrupt images and near-duplicates (perceptual dHash matches) to prevent data leakage.
Then we perform a strictly stratified split (70% Train, 15% Validation, 15% Test).
""")
    add_code("""from data_preprocessing import clean_and_split_dataset, save_preprocessing_config

split_dir = str(Path.cwd().parent / "dataset_split")
split_summary = clean_and_split_dataset(data_dir, split_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
save_preprocessing_config(str(Path.cwd().parent / "models" / "preprocessing_config.json"))
""")

    # Phase 4 & 12
    add_md("""---
## PHASE 4 & 12 — Preprocessing, Augmentation & Chroma-Key Invariance
Because the raw dataset was captured exclusively on a green backdrop, models trained naively learn that "anything not green is foreground", failing completely on normal room backgrounds.
We introduce **Dynamic Chroma-Key Background Augmentation**:
We detect the green studio background and dynamically replace it during training with random indoor colors, gradients, and noise textures.
""")
    add_code("""from data_preprocessing import DynamicChromaKeyAugment, get_transforms

chroma_aug = DynamicChromaKeyAugment(p=1.0) # force augmentation for visualization

sample_img = Image.open(list((Path(data_dir) / "rock").glob("*.png"))[0])

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(sample_img)
axes[0].set_title("Original (Green Backdrop)", fontweight="bold")
axes[0].axis("off")

for i in range(1, 4):
    aug_img = chroma_aug(sample_img)
    axes[i].imshow(aug_img)
    axes[i].set_title(f"Dynamic Replacement #{i}", fontweight="bold")
    axes[i].axis("off")
plt.suptitle("Chroma-Key Background Randomization (Solving Background Bias)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
""")

    # Phase 5 & 6
    add_md("""---
## PHASE 5 & 6 — Model Architecture & Two-Stage Training
We use transfer learning with **MobileNetV3-Small**:
- ~1.5 million parameters (ultralight, fast CPU/GPU inference > 60 FPS)
- Stage 1: Freeze backbone, train classification head with AdamW (`lr=1e-3`)
- Stage 2: Fine-tune backbone with Cosine Annealing (`lr=1e-4`) and Early Stopping
""")
    add_code("""import torch
from train import build_model, run_training, plot_training_curves

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)

model = build_model(architecture="mobilenet_v3_small", num_classes=3, dropout=0.3, pretrained=True)
print(model)
""")

    # Phase 7
    add_md("""---
## PHASE 7 — Hyperparameter Optimization Experiments
We systematically compare:
1. Baseline MobileNetV3 (standard augmentation, no background replacement)
2. Robust MobileNetV3 (with dynamic chroma-key background variation)
3. High-Regularization MobileNetV3 (Dropout = 0.4)
""")
    add_code("""from train import run_hyperparameter_experiments

models_dir = str(Path.cwd().parent / "models")
results_dir = str(Path.cwd().parent / "results")

# Execute optimization experiments
exp_results = run_hyperparameter_experiments(split_dir, results_dir)
""")

    # Phase 8 & 9
    add_md("""---
## PHASE 8 & 9 — Test Set Evaluation & Confusion Matrix
We evaluate the final chosen model on the held-out test set (unseen during training and hyperparameter search).
""")
    add_code("""from evaluate import run_full_evaluation, load_trained_model, evaluate_test_set, generate_classification_report, plot_confusion_matrices
from data_preprocessing import RPSDataset, get_transforms
from torch.utils.data import DataLoader

best_model_path = str(Path.cwd().parent / "models" / "best_model.pth")
model, chk = load_trained_model(best_model_path, device)

test_dir = str(Path.cwd().parent / "dataset_split" / "test")
test_ds = RPSDataset(test_dir, transform=get_transforms(img_size=224, is_train=False))
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

y_pred, y_true, y_probs = evaluate_test_set(model, test_loader, device)
rep = generate_classification_report(y_true, y_pred, str(Path(results_dir) / "classification_report.txt"))
plot_confusion_matrices(y_true, y_pred, results_dir)
""")

    # Phase 10 & 11
    add_md("""---
## PHASE 10 & 11 — Robustness on Unseen Images & Error Analysis
We test the model against external challenging test images (unseen people, wooden tables, painted walls, varied skin tones, different lighting).
""")
    add_code("""from evaluate import test_external_robustness, perform_error_analysis

external_dir = str(Path.cwd().parent / "external_unseen")
rob_results = test_external_robustness(model, external_dir, rep["accuracy"], results_dir, device)
perform_error_analysis(model, test_ds, test_loader, results_dir, device)
""")

    # Phase 13
    add_md("""---
## PHASE 13 — Standalone Single-Image Inference
Test the clean `predict(image)` function.
""")
    add_code("""from predict import predict

sample_test_path = test_ds.samples[0][0]
result = predict(sample_test_path)

print("=" * 35)
print(f"Prediction : {result['prediction']}")
print(f"Confidence : {result['confidence_pct']}")
print("Probabilities:")
for k, v in result["probabilities"].items():
    print(f"  {k:<10}: {v*100:.1f}%")
print("=" * 35)
""")

    with open(output_path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"Jupyter notebook created at: {output_path}")


if __name__ == "__main__":
    out = r"c:\Users\abbas\Documents\rps\notebooks\training.ipynb"
    create_jupyter_notebook(out)

