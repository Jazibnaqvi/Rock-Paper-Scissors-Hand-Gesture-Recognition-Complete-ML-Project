"""
data_preprocessing.py - Rock-Paper-Scissors Dataset Inspection, Cleaning, and Preprocessing.

Handles:
- Dataset inspection, integrity validation, and statistics reporting (Phase 1)
- Data cleaning (deduplication, corruption check, format validation) (Phase 2)
- Stratified Train / Validation / Test split preventing data leakage (Phase 3)
- Realistic augmentation with chroma-key background variation (Phase 4 & 12)
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


CLASS_NAMES = ["rock", "paper", "scissors"]
CLASS_TO_IDX = {"rock": 0, "paper": 1, "scissors": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# ImageNet standard normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) to detect perceptual/near-duplicate images."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def inspect_dataset(data_dir: str) -> Dict:
    """
    Phase 1: Deep inspection of dataset.
    Returns dictionary with all metadata, statistics, and integrity reports.
    """
    data_path = Path(data_dir)
    report = {
        "classes": {},
        "total_images": 0,
        "dimensions": {},
        "formats": {},
        "modes": {},
        "corrupt_files": [],
        "exact_duplicates": [],
        "near_duplicates": [],
    }

    seen_hashes = {}
    seen_dhashes = {}

    for cls in CLASS_NAMES:
        cls_dir = data_path / cls
        if not cls_dir.exists():
            continue

        files = sorted([f for f in cls_dir.iterdir() if f.is_file()])
        report["classes"][cls] = len(files)
        report["total_images"] += len(files)

        for file_path in files:
            str_path = str(file_path)

            # 1. Exact SHA-256 Hash
            try:
                sha = compute_sha256(str_path)
                if sha in seen_hashes:
                    report["exact_duplicates"].append((str_path, seen_hashes[sha]))
                else:
                    seen_hashes[sha] = str_path
            except Exception as e:
                report["corrupt_files"].append((str_path, f"Read error: {e}"))
                continue

            # 2. Image integrity & metadata
            try:
                with Image.open(str_path) as img:
                    img.verify()
                with Image.open(str_path) as img:
                    dim_str = f"{img.width}x{img.height}"
                    report["dimensions"][dim_str] = report["dimensions"].get(dim_str, 0) + 1
                    report["formats"][img.format] = report["formats"].get(img.format, 0) + 1
                    report["modes"][img.mode] = report["modes"].get(img.mode, 0) + 1

                    dh = compute_dhash(img)
                    if dh in seen_dhashes:
                        report["near_duplicates"].append((str_path, seen_dhashes[dh]))
                    else:
                        seen_dhashes[dh] = str_path
            except Exception as e:
                report["corrupt_files"].append((str_path, f"Image verify error: {e}"))

    # Print nicely formatted summary matching Phase 1 specs
    print("=" * 45)
    print("Dataset Summary")
    print("-" * 45)
    for cls in CLASS_NAMES:
        cnt = report["classes"].get(cls, 0)
        print(f"{cls.capitalize():<10}: {cnt:>5} images")
    print("-" * 45)
    print(f"{'Total':<10}: {report['total_images']:>5} images")
    print("=" * 45)
    print(f"Dimensions observed : {report['dimensions']}")
    print(f"Image formats       : {report['formats']}")
    print(f"Color modes         : {report['modes']}")
    print(f"Corrupt images      : {len(report['corrupt_files'])}")
    print(f"Exact duplicates    : {len(report['exact_duplicates'])}")
    print(f"Near-duplicates     : {len(report['near_duplicates'])}")

    return report


def clean_and_split_dataset(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Phase 2 & 3:
    1. Removes corrupt images and redundant near-duplicates to eliminate data leakage.
    2. Performs stratified train/val/test split (70% / 15% / 15%).
    3. Writes splits to output_dir/train, output_dir/val, output_dir/test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Splits must sum to 1.0"
    np.random.seed(seed)

    data_path = Path(data_dir)
    out_path = Path(output_dir)

    # Detect duplicates & near-duplicates
    seen_dhashes = {}
    valid_records = []
    removed_records = []

    for cls in CLASS_NAMES:
        cls_dir = data_path / cls
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.iterdir()):
            if not f.is_file():
                continue
            str_path = str(f)
            try:
                with Image.open(str_path) as img:
                    img.verify()
                with Image.open(str_path) as img:
                    dh = compute_dhash(img)
                    if dh in seen_dhashes:
                        removed_records.append({
                            "path": str_path,
                            "reason": f"Near-duplicate of {seen_dhashes[dh]}",
                            "class": cls
                        })
                    else:
                        seen_dhashes[dh] = str_path
                        valid_records.append({"path": str_path, "class": cls, "filename": f.name})
            except Exception as e:
                removed_records.append({
                    "path": str_path,
                    "reason": f"Corrupt image: {e}",
                    "class": cls
                })

    print(f"\n[PHASE 2 - Data Cleaning Report]")
    print(f"Total raw files analyzed: {len(valid_records) + len(removed_records)}")
    print(f"Files quarantined/removed: {len(removed_records)}")
    for r in removed_records:
        print(f"  - {Path(r['path']).name} ({r['class']}): {r['reason']}")
    print(f"Clean valid files remaining: {len(valid_records)}")

    # Stratified split per class
    splits = {"train": [], "val": [], "test": []}

    for cls in CLASS_NAMES:
        cls_items = [r for r in valid_records if r["class"] == cls]
        n_total = len(cls_items)
        indices = np.random.permutation(n_total)

        n_train = int(np.floor(train_ratio * n_total))
        n_val = int(np.floor(val_ratio * n_total))
        n_test = n_total - (n_train + n_val)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        for i in train_idx:
            splits["train"].append(cls_items[i])
        for i in val_idx:
            splits["val"].append(cls_items[i])
        for i in test_idx:
            splits["test"].append(cls_items[i])

    # Copy files into output structure
    if out_path.exists():
        shutil.rmtree(out_path)

    for split_name in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            (out_path / split_name / cls).mkdir(parents=True, exist_ok=True)

        for item in splits[split_name]:
            dest = out_path / split_name / item["class"] / item["filename"]
            shutil.copy2(item["path"], dest)

    summary = {
        "train": {c: len([x for x in splits["train"] if x["class"] == c]) for c in CLASS_NAMES},
        "val": {c: len([x for x in splits["val"] if x["class"] == c]) for c in CLASS_NAMES},
        "test": {c: len([x for x in splits["test"] if x["class"] == c]) for c in CLASS_NAMES},
    }

    print(f"\n[PHASE 3 - Stratified Dataset Split Summary]")
    print(f"{'Split':<10} | {'Rock':<8} | {'Paper':<8} | {'Scissors':<8} | {'Total':<8}")
    print("-" * 52)
    for s in ["train", "val", "test"]:
        r_c = summary[s]["rock"]
        p_c = summary[s]["paper"]
        s_c = summary[s]["scissors"]
        tot = r_c + p_c + s_c
        pct = (tot / len(valid_records)) * 100
        print(f"{s.capitalize():<10} | {r_c:<8} | {p_c:<8} | {s_c:<8} | {tot:<4} ({pct:.1f}%)")
    print("-" * 52)

    return summary


class DynamicChromaKeyAugment:
    """
    Phase 4 & 12: Background variation via Chroma-Key augmentation.
    Detects green screen backdrop and dynamically replaces it with random colors,
    synthetic gradients, textures, or noise.
    Forces model to rely on hand anatomy rather than memorizing the green studio background!
    """
    def __init__(self, p: float = 0.65):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img

        # Convert to numpy array
        arr = np.array(img).astype(np.float32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

        # Green chroma detection: green channel significantly dominates red and blue
        is_green = (g > 70) & (g > r * 1.15) & (g > b * 1.15)

        if not np.any(is_green):
            return img

        # Choose random synthetic replacement background:
        mode = np.random.choice(["solid", "gradient", "noise"])
        h, w = arr.shape[:2]

        if mode == "solid":
            # Neutral indoor wall / desk color
            bg_color = np.random.randint(40, 240, size=(1, 1, 3)).astype(np.float32)
            bg = np.broadcast_to(bg_color, (h, w, 3)).copy()
        elif mode == "gradient":
            c1 = np.random.randint(40, 220, size=3).astype(np.float32)
            c2 = np.random.randint(40, 220, size=3).astype(np.float32)
            alpha = np.linspace(0, 1, h)[:, None, None].astype(np.float32)
            bg = (1 - alpha) * c1 + alpha * c2
        else: # noise
            base = np.random.randint(80, 200, size=(1, 1, 3)).astype(np.float32)
            noise = np.random.randn(h, w, 3).astype(np.float32) * 25.0
            bg = np.clip(base + noise, 0, 255)

        # Soft blending on mask boundary
        mask_3d = np.repeat(is_green[..., None], 3, axis=-1)
        arr = np.where(mask_3d, bg, arr)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def get_transforms(
    img_size: int = 224,
    is_train: bool = True,
    use_chroma_aug: bool = True
) -> T.Compose:
    """
    Phase 4: Data Preprocessing and Augmentation Pipeline.
    - Training: Dynamic chroma background replacement, random rotation (-15 to 15 deg),
      RandomResizedCrop (0.8 to 1.0), ColorJitter, RandomHorizontalFlip, Normalization.
    - Evaluation: Resize, CenterCrop, Normalization.
    """
    if is_train:
        t_list = []
        if use_chroma_aug:
            t_list.append(DynamicChromaKeyAugment(p=0.65))

        t_list.extend([
            T.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),  # Left hand vs Right hand invariance
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return T.Compose(t_list)
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class RPSDataset(Dataset):
    """PyTorch Dataset for Rock-Paper-Scissors images."""
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        for cls_name, cls_idx in CLASS_TO_IDX.items():
            cls_folder = self.root_dir / cls_name
            if cls_folder.exists():
                for f in sorted(cls_folder.iterdir()):
                    if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        self.samples.append((str(f), cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label


def save_preprocessing_config(filepath: str, img_size: int = 224):
    """Save preprocessing configuration to JSON for independent inference."""
    config = {
        "img_size": img_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "classes": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {str(k): v for k, v in IDX_TO_CLASS.items()},
    }
    with open(filepath, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Preprocessing config saved to {filepath}")


if __name__ == "__main__":
    raw_dir = r"c:\Users\abbas\Documents\rps\dataset"
    split_dir = r"c:\Users\abbas\Documents\rps\dataset_split"

    print("Step 1: Inspecting raw dataset...")
    inspect_dataset(raw_dir)

    print("\nStep 2: Cleaning and Stratified Splitting...")
    clean_and_split_dataset(raw_dir, split_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    save_preprocessing_config(r"c:\Users\abbas\Documents\rps\models\preprocessing_config.json")

