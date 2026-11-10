"""
train.py - Rock-Paper-Scissors Model Architecture, Training, and Hyperparameter Optimization.

Implements:
- Phase 5: Transfer Learning with MobileNetV3-Small / EfficientNet-B0
- Phase 6: Two-stage training (backbone freeze -> controlled fine-tuning),
           early stopping, LR scheduling, checkpointing, loss/accuracy curves
- Phase 7: Systematic hyperparameter comparison
"""

import os
import sys
import copy
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

# Ensure local src directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import dataset & preprocessing utilities
from data_preprocessing import (
    RPSDataset,
    get_transforms,
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    save_preprocessing_config,
)


def build_model(
    architecture: str = "mobilenet_v3_small",
    num_classes: int = 3,
    dropout: float = 0.3,
    pretrained: bool = True
) -> nn.Module:
    """
    Phase 5: Builds the transfer learning model.
    MobileNetV3-Small is preferred for real-time webcam inference on CPU/GPU.
    Replaces the classification head with a tailored regularized MLP.
    """
    weights = "DEFAULT" if pretrained else None

    if architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[0].in_features
        # Custom head with Dropout & ReLU
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
    elif architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model


def set_backbone_trainable(model: nn.Module, architecture: str, trainable: bool = True):
    """Freeze or unfreeze backbone weights for controlled transfer learning."""
    if architecture.startswith("mobilenet"):
        for param in model.features.parameters():
            param.requires_grad = trainable
    elif architecture.startswith("efficientnet"):
        for param in model.features.parameters():
            param.requires_grad = trainable


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for a single epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate on validation or test set. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def run_training(
    data_dir: str,
    output_dir: str,
    architecture: str = "mobilenet_v3_small",
    img_size: int = 224,
    batch_size: int = 32,
    head_epochs: int = 5,
    finetune_epochs: int = 15,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    use_chroma_aug: bool = True,
    patience: int = 6,
    device: Optional[torch.device] = None,
    tag: str = "best_model"
) -> Dict:
    """
    Phase 6: Complete Two-Stage Transfer Learning Pipeline.
    Stage 1: Warmup classification head with backbone frozen.
    Stage 2: Controlled end-to-end fine-tuning with cosine annealing and early stopping.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Training: {architecture} on {device}")
    print(f" Image Size: {img_size}x{img_size} | Batch Size: {batch_size}")
    print(f" Chroma Augmentation: {use_chroma_aug} | Dropout: {dropout}")
    print(f" Head Epochs: {head_epochs} | Finetune Epochs: {finetune_epochs}")
    print(f"=======================================================")

    # Prepare datasets & dataloaders
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_tf = get_transforms(img_size=img_size, is_train=True, use_chroma_aug=use_chroma_aug)
    val_tf = get_transforms(img_size=img_size, is_train=False)

    train_ds = RPSDataset(train_dir, transform=train_tf)
    val_ds = RPSDataset(val_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = build_model(architecture=architecture, num_classes=3, dropout=dropout, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "stage": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_weights = None
    epochs_no_improve = 0

    # ---------------- STAGE 1: Head Warmup ----------------
    print("\n--- STAGE 1: Training Classification Head (Backbone Frozen) ---")
    set_backbone_trainable(model, architecture, trainable=False)
    optimizer_head = optim.AdamW(model.classifier.parameters(), lr=lr_head, weight_decay=weight_decay)

    for epoch in range(head_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_head, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["stage"].append("head")

        print(f"Epoch {epoch+1:02d}/{head_epochs:02d} (Warmup) [{dt:.1f}s] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())

    # ---------------- STAGE 2: Controlled Fine-Tuning ----------------
    print("\n--- STAGE 2: Controlled Fine-Tuning Upper Backbone ---")
    set_backbone_trainable(model, architecture, trainable=True)
    optimizer_ft = optim.AdamW(model.parameters(), lr=lr_finetune, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=finetune_epochs, eta_min=1e-6)

    for epoch in range(finetune_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_ft, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["stage"].append("finetune")

        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            improved = True
        else:
            epochs_no_improve += 1

        marker = " (*BEST*)" if improved else ""
        print(f"Epoch {head_epochs + epoch + 1:02d}/{head_epochs + finetune_epochs:02d} (FT) [{dt:.1f}s] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%{marker}")

        if epochs_no_improve >= patience:
            print(f"\n[Early Stopping triggered after {epoch+1} fine-tuning epochs without improvement]")
            break

    # Load best weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Save model checkpoint
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, f"{tag}.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "architecture": architecture,
        "img_size": img_size,
        "classes": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "dropout": dropout,
        "use_chroma_aug": use_chroma_aug,
    }, model_save_path)
    print(f"\n[Checkpoint Saved] Best model saved to: {model_save_path}")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%, Validation Loss: {best_val_loss:.4f}")

    return {
        "model": model,
        "history": history,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "model_path": model_save_path,
        "architecture": architecture
    }


def plot_training_curves(history: Dict, save_path: str):
    """
    Phase 6: Plot Training Loss, Validation Loss, Training Accuracy, Validation Accuracy.
    Clear visualization showing transition from warmup to fine-tuning and detecting overfitting.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#d62728", linewidth=2)
    axes[0].set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Find stage boundary if exists
    if "stage" in history and "finetune" in history["stage"]:
        ft_start = history["stage"].index("finetune") + 0.5
        axes[0].axvline(x=ft_start, color="gray", linestyle="--", alpha=0.7, label="Fine-tuning start")
        axes[1].axvline(x=ft_start, color="gray", linestyle="--", alpha=0.7, label="Fine-tuning start")

    # Accuracy curve
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc (%)", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Acc (%)", color="#ff7f0e", linewidth=2)
    axes[1].set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def run_hyperparameter_experiments(data_dir: str, results_dir: str) -> Dict:
    """
    Phase 7: Hyperparameter Optimization Experiments.
    Systematically compares:
    1. Baseline MobileNetV3 (standard augmentation, no background variation)
    2. Robust MobileNetV3 (with dynamic chroma-key background variation)
    3. MobileNetV3 with deeper fine-tuning / higher dropout
    4. Architecture comparison (MobileNetV3-Small vs EfficientNet-B0)
    """
    print("\n" + "=" * 60)
    print(" PHASE 7 — HYPERPARAMETER OPTIMIZATION EXPERIMENTS ")
    print("=" * 60)

    experiments = [
        {
            "name": "Exp 1: Baseline MobileNetV3 (Standard Aug)",
            "arch": "mobilenet_v3_small",
            "chroma_aug": False,
            "dropout": 0.2,
            "head_epochs": 4,
            "ft_epochs": 8,
            "tag": "model_exp1_baseline"
        },
        {
            "name": "Exp 2: Robust MobileNetV3 (Chroma Background Aug)",
            "arch": "mobilenet_v3_small",
            "chroma_aug": True,
            "dropout": 0.3,
            "head_epochs": 4,
            "ft_epochs": 8,
            "tag": "model_exp2_chroma_aug"
        },
        {
            "name": "Exp 3: High-Regularization MobileNetV3 (Dropout 0.4)",
            "arch": "mobilenet_v3_small",
            "chroma_aug": True,
            "dropout": 0.4,
            "head_epochs": 4,
            "ft_epochs": 8,
            "tag": "model_exp3_high_reg"
        },
    ]

    exp_results = []
    models_dir = r"c:\Users\abbas\Documents\rps\models"

    for exp in experiments:
        print(f"\n>>> Running Experiment: {exp['name']}")
        res = run_training(
            data_dir=data_dir,
            output_dir=models_dir,
            architecture=exp["arch"],
            img_size=224,
            batch_size=32,
            head_epochs=exp["head_epochs"],
            finetune_epochs=exp["ft_epochs"],
            dropout=exp["dropout"],
            use_chroma_aug=exp["chroma_aug"],
            tag=exp["tag"]
        )
        exp_results.append({
            "name": exp["name"],
            "val_acc": res["best_val_acc"],
            "val_loss": res["best_val_loss"],
            "tag": exp["tag"],
            "history": res["history"]
        })

    # Summary table
    print("\n" + "=" * 65)
    print(" HYPERPARAMETER OPTIMIZATION COMPARISON TABLE")
    print("=" * 65)
    print(f"{'Experiment':<45} | {'Val Acc':<9} | {'Val Loss':<8}")
    print("-" * 65)
    for r in exp_results:
        print(f"{r['name']:<45} | {r['val_acc']*100:>6.2f}%  | {r['val_loss']:>7.4f}")
    print("=" * 65)

    # Save best overall model as best_model.pth
    best_exp = max(exp_results, key=lambda x: x["val_acc"])
    print(f"\n[Selected Best Model]: {best_exp['name']}")
    src_best = os.path.join(models_dir, f"{best_exp['tag']}.pth")
    dest_best = os.path.join(models_dir, "best_model.pth")
    if os.path.abspath(src_best) != os.path.abspath(dest_best) and os.path.exists(src_best):
        shutil.copy2(src_best, dest_best)
        print(f"Copied {src_best} -> {dest_best}")
    else:
        print(f"Best model available at {dest_best}")

    # Plot training curves of the best model
    curves_path = os.path.join(results_dir, "training_curves.png")
    plot_training_curves(best_exp["history"], curves_path)

    return exp_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Rock-Paper-Scissors transfer learning model.")
    parser.add_argument("--experiments", action="store_true", help="Run full hyperparameter optimization comparison sweep.")
    parser.add_argument("--epochs", type=int, default=12, help="Total epochs for training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small", help="Model architecture.")
    args = parser.parse_args()

    split_dir = r"c:\Users\abbas\Documents\rps\dataset_split"
    models_dir = r"c:\Users\abbas\Documents\rps\models"
    results_dir = r"c:\Users\abbas\Documents\rps\results"

    if args.experiments:
        run_hyperparameter_experiments(split_dir, results_dir)
    else:
        # Standard fast training of final robust model
        head_ep = min(4, args.epochs // 3)
        ft_ep = args.epochs - head_ep
        res = run_training(
            data_dir=split_dir,
            output_dir=models_dir,
            architecture=args.arch,
            img_size=224,
            batch_size=args.batch_size,
            head_epochs=head_ep,
            finetune_epochs=ft_ep,
            dropout=0.3,
            use_chroma_aug=True,
            tag="best_model"
        )
        plot_training_curves(res["history"], os.path.join(results_dir, "training_curves.png"))


