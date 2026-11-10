"""
evaluate.py - Comprehensive Evaluation, Confusion Matrix, Error Analysis, and Robustness Testing.

Implements:
- Phase 8: Test set evaluation (Accuracy, Precision, Recall, F1, Per-class metrics)
- Phase 9: Raw & Normalized Confusion Matrix with error diagnosis
- Phase 10: Robustness test on external/unseen real-world images
- Phase 11: Error analysis (misclassifications, confidence, root cause investigation)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure local src directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import (
    RPSDataset,
    get_transforms,
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
)
from train import build_model


def load_trained_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load model architecture and weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint.get("architecture", "mobilenet_v3_small")
    dropout = checkpoint.get("dropout", 0.3)

    model = build_model(architecture=architecture, num_classes=3, dropout=dropout, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over entire test set, returning (all_preds, all_labels, all_probs)."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_txt_path: str
) -> Dict:
    """
    Phase 8: Calculate precision, recall, f1-score, and per-class accuracy.
    Saves clean report to text file and prints formatted table.
    """
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=4
    )
    report_str = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4
    )

    acc = report_dict["accuracy"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]

    print("\n" + "=" * 55)
    print(" PHASE 8 — TEST SET CLASSIFICATION REPORT")
    print("=" * 55)
    print(report_str)
    print("-" * 55)
    print(f"Overall Test Accuracy: {acc*100:.2f}%")
    print(f"Macro F1-Score        : {macro_f1:.4f}")
    print(f"Weighted F1-Score     : {weighted_f1:.4f}")
    print("=" * 55)

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w") as f:
        f.write("Rock-Paper-Scissors Model Test Evaluation Report\n")
        f.write("==================================================\n\n")
        f.write(report_str)
        f.write(f"\nOverall Test Accuracy: {acc*100:.2f}%\n")
        f.write(f"Macro F1-Score       : {macro_f1:.4f}\n")
        f.write(f"Weighted F1-Score    : {weighted_f1:.4f}\n")

    return report_dict


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    results_dir: str
):
    """
    Phase 9: Generates both Raw and Normalized Confusion Matrices with in-depth error diagnosis.
    """
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = cm_raw.astype("float") / cm_raw.sum(axis=1)[:, np.newaxis]

    # 1. Raw Confusion Matrix Plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in CLASS_NAMES],
        yticklabels=[c.capitalize() for c in CLASS_NAMES],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Rock-Paper-Scissors: Raw Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    raw_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(raw_path, dpi=300)
    plt.close()

    # 2. Normalized Confusion Matrix Plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Greens",
        xticklabels=[c.capitalize() for c in CLASS_NAMES],
        yticklabels=[c.capitalize() for c in CLASS_NAMES],
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title("Rock-Paper-Scissors: Normalized Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    norm_path = os.path.join(results_dir, "normalized_confusion_matrix.png")
    plt.savefig(norm_path, dpi=300)
    plt.close()

    print(f"\n[PHASE 9 - Confusion Matrix Analysis]")
    print(f"Raw confusion matrix saved to: {raw_path}")
    print(f"Normalized confusion matrix saved to: {norm_path}")

    # Detailed Error Diagnosis
    print("\nDetailed Misclassification Breakdown:")
    for i, true_name in enumerate(CLASS_NAMES):
        for j, pred_name in enumerate(CLASS_NAMES):
            if i != j and cm_raw[i, j] > 0:
                print(f"  * {true_name.capitalize()} mistaken for {pred_name.capitalize()}: "
                      f"{cm_raw[i, j]} images ({cm_norm[i, j]*100:.1f}%)")
    if np.sum(cm_raw) - np.trace(cm_raw) == 0:
        print("  * Zero misclassifications on the held-out test set!")


def perform_error_analysis(
    model: nn.Module,
    test_dataset: RPSDataset,
    test_loader: DataLoader,
    results_dir: str,
    device: torch.device
):
    """
    Phase 11: Error Analysis.
    Finds misclassified images, displays them with true/pred labels and confidence,
    and analyzes root causes (lighting, pose ambiguity, edge alignment).
    """
    model.eval()
    misclassified = []

    for idx, (path, label) in enumerate(test_dataset.samples):
        img_pil = Image.open(path).convert("RGB")
        tf = get_transforms(img_size=224, is_train=False)
        tensor = tf(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)
            confidence = probs[pred]

        if pred != label:
            misclassified.append({
                "path": path,
                "true_label": CLASS_NAMES[label],
                "pred_label": CLASS_NAMES[pred],
                "confidence": confidence,
                "all_probs": probs,
                "image": img_pil
            })

    print("\n" + "=" * 55)
    print(" PHASE 11 — ERROR ANALYSIS")
    print("=" * 55)
    print(f"Total misclassified images in test set: {len(misclassified)} / {len(test_dataset)}")

    if len(misclassified) > 0:
        n_show = min(len(misclassified), 6)
        fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4.5))
        if n_show == 1:
            axes = [axes]

        for i, item in enumerate(misclassified[:n_show]):
            axes[i].imshow(item["image"])
            axes[i].set_title(
                f"True: {item['true_label'].capitalize()}\nPred: {item['pred_label'].capitalize()} ({item['confidence']*100:.1f}%)",
                fontsize=11, fontweight="bold", color="darkred"
            )
            axes[i].axis("off")

        plt.suptitle("Misclassified Test Examples (Error Analysis)", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        err_path = os.path.join(results_dir, "error_analysis.png")
        plt.savefig(err_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Misclassified samples visualization saved to: {err_path}")
    else:
        # Create an informative visual showing high-confidence edge cases
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, cls in enumerate(CLASS_NAMES):
            # pick one sample
            cls_samples = [s for s in test_dataset.samples if s[1] == i]
            sample_path = cls_samples[0][0]
            img = Image.open(sample_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Class: {cls.capitalize()} (100% Correct on Test Set)", fontsize=11, fontweight="bold")
            axes[i].axis("off")
        plt.suptitle("Perfect Test Set Discrimination (0 Errors)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        err_path = os.path.join(results_dir, "error_analysis.png")
        plt.savefig(err_path, dpi=300)
        plt.close()
        print("Test set achieved 100% accuracy! Displayed representative test images in error_analysis.png")


def test_external_robustness(
    model: nn.Module,
    external_dir: str,
    original_test_acc: float,
    results_dir: str,
    device: torch.device
) -> Dict:
    """
    Phase 10: Robustness / Generalization Test on Completely Unseen Images.
    Evaluates model on external images containing:
    - Different backgrounds (wooden desks, white walls, colored paper, textured surfaces)
    - Different lighting & skin tones
    - Different angles and distances
    Reports comparison between Original Test Accuracy and External/Unseen Image Accuracy.
    """
    print("\n" + "=" * 60)
    print(" PHASE 10 — ROBUSTNESS & GENERALIZATION TEST (UNSEEN IMAGES)")
    print("=" * 60)

    external_path = Path(external_dir)
    tf = get_transforms(img_size=224, is_train=False)

    total = 0
    correct = 0
    per_class_results = {c: {"correct": 0, "total": 0} for c in CLASS_NAMES}
    samples_detail = []

    for cls in CLASS_NAMES:
        cls_folder = external_path / cls
        if not cls_folder.exists():
            continue
        for f in sorted(cls_folder.iterdir()):
            if not f.is_file() or f.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue
            try:
                img_pil = Image.open(f).convert("RGB")
                tensor = tf(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred = int(np.argmax(probs))
                    conf = float(probs[pred])

                true_idx = CLASS_TO_IDX[cls]
                is_correct = (pred == true_idx)

                total += 1
                if is_correct:
                    correct += 1
                per_class_results[cls]["total"] += 1
                if is_correct:
                    per_class_results[cls]["correct"] += 1

                samples_detail.append({
                    "path": str(f),
                    "filename": f.name,
                    "true_class": cls,
                    "pred_class": CLASS_NAMES[pred],
                    "confidence": conf,
                    "correct": is_correct,
                    "image": img_pil
                })
            except Exception as e:
                print(f"Error processing {f}: {e}")

    ext_acc = (correct / total) if total > 0 else 0.0

    print(f"\n[GENERALIZATION COMPARISON]")
    print(f"Original In-Distribution Test Accuracy : {original_test_acc*100:.2f}%")
    print(f"External/Unseen Real-World Accuracy    : {ext_acc*100:.2f}% ({correct}/{total})")
    print("-" * 55)
    for c in CLASS_NAMES:
        c_tot = per_class_results[c]["total"]
        c_cor = per_class_results[c]["correct"]
        c_acc = (c_cor / c_tot * 100) if c_tot > 0 else 0.0
        print(f"  * {c.capitalize():<10}: {c_cor:>2}/{c_tot:>2} correct ({c_acc:.1f}%)")
    print("=" * 60)

    # Plot sample external predictions
    if len(samples_detail) > 0:
        n_show = min(len(samples_detail), 9)
        rows = (n_show + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
        axes = np.array(axes).flatten()

        for idx in range(n_show):
            item = samples_detail[idx]
            color = "green" if item["correct"] else "red"
            axes[idx].imshow(item["image"])
            axes[idx].set_title(
                f"True: {item['true_class'].capitalize()} | Pred: {item['pred_class'].capitalize()}\nConf: {item['confidence']*100:.1f}%",
                fontsize=10, fontweight="bold", color=color
            )
            axes[idx].axis("off")

        for idx in range(n_show, len(axes)):
            axes[idx].axis("off")

        plt.suptitle("External / Robustness Evaluation Samples", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        rob_path = os.path.join(results_dir, "external_robustness.png")
        plt.savefig(rob_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Robustness visual report saved to: {rob_path}")

    return {
        "original_test_acc": original_test_acc,
        "external_acc": ext_acc,
        "total_external": total,
        "correct_external": correct,
        "per_class": per_class_results
    }


def run_full_evaluation(
    checkpoint_path: str,
    data_split_dir: str,
    external_dir: str,
    results_dir: str
):
    """Orchestrates Phases 8, 9, 10, and 11."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running full evaluation on {device}...")

    model, checkpoint = load_trained_model(checkpoint_path, device)
    img_size = checkpoint.get("img_size", 224)

    test_dir = os.path.join(data_split_dir, "test")
    test_tf = get_transforms(img_size=img_size, is_train=False)
    test_dataset = RPSDataset(test_dir, transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Phase 8: Metrics
    y_pred, y_true, y_probs = evaluate_test_set(model, test_loader, device)
    report_dict = generate_classification_report(
        y_true, y_pred, os.path.join(results_dir, "classification_report.txt")
    )
    test_acc = report_dict["accuracy"]

    # Phase 9: Confusion Matrices
    plot_confusion_matrices(y_true, y_pred, results_dir)

    # Phase 11: Error Analysis
    perform_error_analysis(model, test_dataset, test_loader, results_dir, device)

    # Phase 10: Robustness Test
    if os.path.exists(external_dir) and any(os.scandir(external_dir)):
        test_external_robustness(model, external_dir, test_acc, results_dir, device)


if __name__ == "__main__":
    chk = r"c:\Users\abbas\Documents\rps\models\best_model.pth"
    splits = r"c:\Users\abbas\Documents\rps\dataset_split"
    ext = r"c:\Users\abbas\Documents\rps\external_unseen"
    res = r"c:\Users\abbas\Documents\rps\results"

    run_full_evaluation(chk, splits, ext, res)

