"""
predict.py - Standalone Inference Module for Rock-Paper-Scissors Classification.

Implements Phase 13:
- predict(image) -> returns class prediction, confidence percentage, and class probabilities
- Supports file paths, PIL Images, and OpenCV NumPy arrays
- Usable independently without requiring training dependencies
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Union, Dict, Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "preprocessing_config.json")


class RPSPredictor:
    """Standalone predictor for Rock-Paper-Scissors classification."""
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, config_path: str = DEFAULT_CONFIG_PATH, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.architecture = checkpoint.get("architecture", "mobilenet_v3_small")
        self.classes = checkpoint.get("classes", ["rock", "paper", "scissors"])
        self.class_to_idx = checkpoint.get("class_to_idx", {"rock": 0, "paper": 1, "scissors": 2})
        self.idx_to_class = {int(k): v for k, v in checkpoint.get("idx_to_class", {0: "rock", 1: "paper", 2: "scissors"}).items()}
        self.img_size = checkpoint.get("img_size", 224)

        # Build model architecture
        if self.architecture == "mobilenet_v3_small":
            self.model = models.mobilenet_v3_small(weights=None)
            in_f = self.model.classifier[0].in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_f, 128),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(128, len(self.classes))
            )
        elif self.architecture == "mobilenet_v3_large":
            self.model = models.mobilenet_v3_large(weights=None)
            in_f = self.model.classifier[0].in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_f, 256),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, len(self.classes))
            )
        elif self.architecture == "efficientnet_b0":
            self.model = models.efficientnet_b0(weights=None)
            in_f = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_f, 128),
                nn.SiLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(128, len(self.classes))
            )
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Image transform
        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Run inference on a single image.
        image can be:
        - str/Path: path to image file
        - PIL Image
        - np.ndarray: OpenCV BGR or RGB array
        """
        # 1. Convert input to PIL Image in RGB
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            # If 3 channels, assuming OpenCV BGR format by default
            if image.ndim == 3 and image.shape[2] == 3:
                # Convert BGR to RGB
                rgb_arr = image[..., ::-1]
                pil_img = Image.fromarray(rgb_arr)
            else:
                pil_img = Image.fromarray(image).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 2. Preprocess & Tensor conversion
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # 3. Model inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

        pred_class = self.idx_to_class[pred_idx].capitalize()

        prob_dist = {
            self.idx_to_class[i].capitalize(): float(probs[i])
            for i in range(len(self.classes))
        }

        return {
            "prediction": pred_class,
            "class_id": pred_idx,
            "confidence": confidence,
            "confidence_pct": f"{confidence * 100:.1f}%",
            "probabilities": prob_dist
        }


# Global cached predictor for predict() function
_global_predictor = None


def predict(image: Union[str, Path, Image.Image, np.ndarray], model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """
    Standard simple inference function required by Phase 13:
    predict(image) -> returns prediction and confidence
    """
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = RPSPredictor(model_path=model_path)
    return _global_predictor.predict(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Rock-Paper-Scissors gesture classifier.")
    parser.add_argument("--image", type=str, required=True, help="Path to hand gesture image file.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to model checkpoint .pth")
    args = parser.parse_args()

    predictor = RPSPredictor(model_path=args.model)
    res = predictor.predict(args.image)

    print("\n" + "=" * 40)
    print(f"Prediction : {res['prediction']}")
    print(f"Confidence : {res['confidence_pct']}")
    print("-" * 40)
    print("Class Probabilities:")
    for cls_name, prob in res["probabilities"].items():
        print(f"  {cls_name:<10}: {prob * 100:>5.1f}%")
    print("=" * 40)

