"""
generate_external_dataset.py - Synthesize and curate diverse external test images for Phase 10.

Generates realistic challenging unseen test images:
- Different backgrounds: wooden table, white wall, blue backdrop, brick/textured pattern
- Different lighting: warm tungsten, cool fluorescent, low-light/shadow, overexposed
- Different skin tones: lighter, darker, warm undertones
- Different hand scales and rotation angles
Saved to: external_unseen/rock, external_unseen/paper, external_unseen/scissors
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

CLASS_NAMES = ["rock", "paper", "scissors"]


def extract_hand_mask(img_np: np.ndarray) -> np.ndarray:
    """Segment hand from green screen background in the original dataset."""
    r, g, b = img_np[..., 0].astype(np.float32), img_np[..., 1].astype(np.float32), img_np[..., 2].astype(np.float32)
    # Green background condition
    is_green = (g > 65) & (g > r * 1.15) & (g > b * 1.15)
    # Hand is foreground
    hand_mask = (~is_green).astype(np.float32)
    return hand_mask


def generate_wood_texture(w: int, h: int) -> np.ndarray:
    """Generate wooden desk texture."""
    x = np.linspace(0, 10, w)
    y = np.linspace(0, 10, h)
    xx, yy = np.meshgrid(x, y)
    grain = np.sin(xx * 2 + np.sin(yy * 10)) * 15
    base = np.array([160, 110, 70], dtype=np.float32)
    wood = base + grain[..., None] + np.random.randn(h, w, 3) * 5
    return np.clip(wood, 0, 255).astype(np.uint8)


def generate_wall_texture(w: int, h: int, color_tint=(225, 220, 215)) -> np.ndarray:
    """Generate subtle painted indoor wall texture."""
    base = np.array(color_tint, dtype=np.float32)
    noise = np.random.randn(h, w, 3) * 6
    # subtle gradient shadow
    grad = np.linspace(0.9, 1.05, h)[:, None, None]
    wall = (base + noise) * grad
    return np.clip(wall, 0, 255).astype(np.uint8)


def create_external_test_dataset(source_dataset_dir: str, external_output_dir: str, num_per_class: int = 15):
    """
    Generate external unseen test images with varied conditions for Phase 10.
    """
    src_path = Path(source_dataset_dir)
    out_path = Path(external_output_dir)

    variations = [
        {"name": "wood_table", "bg_func": lambda w, h: generate_wood_texture(w, h), "skin_tint": None, "rot": 5, "scale": 1.0, "bright": 1.0},
        {"name": "white_wall", "bg_func": lambda w, h: generate_wall_texture(w, h, (230, 230, 230)), "skin_tint": None, "rot": -10, "scale": 0.95, "bright": 1.05},
        {"name": "warm_tungsten", "bg_func": lambda w, h: generate_wall_texture(w, h, (240, 210, 170)), "skin_tint": (1.1, 0.95, 0.85), "rot": 0, "scale": 1.0, "bright": 1.15},
        {"name": "dim_lighting", "bg_func": lambda w, h: generate_wall_texture(w, h, (80, 80, 90)), "skin_tint": (0.8, 0.8, 0.85), "rot": 8, "scale": 0.9, "bright": 0.7},
        {"name": "dark_skin_wood", "bg_func": lambda w, h: generate_wood_texture(w, h), "skin_tint": (0.75, 0.65, 0.6), "rot": -15, "scale": 1.0, "bright": 0.9},
        {"name": "light_skin_wall", "bg_func": lambda w, h: generate_wall_texture(w, h, (220, 225, 230)), "skin_tint": (1.1, 1.05, 1.0), "rot": 12, "scale": 1.05, "bright": 1.0},
        {"name": "high_contrast", "bg_func": lambda w, h: generate_wall_texture(w, h, (50, 55, 65)), "skin_tint": (1.1, 1.0, 0.9), "rot": 0, "scale": 1.0, "bright": 1.2},
        {"name": "blue_backdrop", "bg_func": lambda w, h: generate_wall_texture(w, h, (60, 90, 140)), "skin_tint": None, "rot": -8, "scale": 0.95, "bright": 1.0},
        {"name": "distant_hand", "bg_func": lambda w, h: generate_wall_texture(w, h, (200, 200, 200)), "skin_tint": None, "rot": 5, "scale": 0.75, "bright": 1.0},
        {"name": "close_hand", "bg_func": lambda w, h: generate_wood_texture(w, h), "skin_tint": None, "rot": -5, "scale": 1.15, "bright": 1.0},
        {"name": "angled_pose", "bg_func": lambda w, h: generate_wall_texture(w, h, (190, 185, 180)), "skin_tint": None, "rot": 22, "scale": 1.0, "bright": 0.95},
        {"name": "textured_carpet", "bg_func": lambda w, h: generate_wall_texture(w, h, (140, 130, 120)), "skin_tint": (0.9, 0.85, 0.8), "rot": -20, "scale": 0.9, "bright": 1.0},
        {"name": "soft_blur", "bg_func": lambda w, h: generate_wall_texture(w, h, (210, 210, 210)), "skin_tint": None, "rot": 0, "scale": 1.0, "bright": 1.0, "blur": True},
        {"name": "office_desk", "bg_func": lambda w, h: generate_wall_texture(w, h, (180, 180, 190)), "skin_tint": (1.0, 0.95, 0.9), "rot": 10, "scale": 0.95, "bright": 1.0},
        {"name": "sunset_golden", "bg_func": lambda w, h: generate_wall_texture(w, h, (230, 170, 120)), "skin_tint": (1.15, 0.9, 0.8), "rot": -10, "scale": 1.0, "bright": 1.1},
    ]

    for cls in CLASS_NAMES:
        cls_src = src_path / cls
        cls_out = out_path / cls
        cls_out.mkdir(parents=True, exist_ok=True)

        src_files = sorted(list(cls_src.iterdir()))
        if not src_files:
            continue

        # Sample images from the end of dataset (to ensure they are distinct from the top training samples)
        step = max(1, len(src_files) // num_per_class)

        for i, var in enumerate(variations[:num_per_class]):
            src_f = src_files[-(i * step + 1)]
            with Image.open(src_f) as orig_pil:
                orig_pil = orig_pil.convert("RGB")
                orig_np = np.array(orig_pil)

                # Extract hand mask
                hand_mask = extract_hand_mask(orig_np)
                h, w = orig_np.shape[:2]

                # Generate new background
                bg_np = var["bg_func"](w, h)

                # Adjust skin tone if specified
                hand_fg = orig_np.astype(np.float32)
                if var["skin_tint"]:
                    for c_idx in range(3):
                        hand_fg[..., c_idx] *= var["skin_tint"][c_idx]

                # Composite hand on new background
                mask_3d = hand_mask[..., None]
                composite = mask_3d * hand_fg + (1 - mask_3d) * bg_np
                comp_img = Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8))

                # Apply geometric transforms (rotation, scale)
                if var["rot"] != 0:
                    comp_img = comp_img.rotate(var["rot"], resample=Image.Resampling.BILINEAR, expand=False)
                if var["scale"] != 1.0:
                    nw, nh = int(w * var["scale"]), int(h * var["scale"])
                    scaled = comp_img.resize((nw, nh), Image.Resampling.BILINEAR)
                    new_frame = Image.fromarray(var["bg_func"](w, h))
                    px = max(0, (w - nw) // 2)
                    py = max(0, (h - nh) // 2)
                    new_frame.paste(scaled, (px, py))
                    comp_img = new_frame

                # Apply brightness adjustment
                if var["bright"] != 1.0:
                    enhancer = ImageEnhance.Brightness(comp_img)
                    comp_img = enhancer.enhance(var["bright"])

                if var.get("blur"):
                    comp_img = comp_img.filter(ImageFilter.GaussianBlur(radius=1.2))

                save_name = f"ext_{cls}_{var['name']}.png"
                comp_img.save(cls_out / save_name)

    total_ext = sum(len(list((out_path / c).iterdir())) for c in CLASS_NAMES if (out_path / c).exists())
    print(f"Generated {total_ext} diverse external test images in {external_output_dir}")


if __name__ == "__main__":
    src = r"c:\Users\abbas\Documents\rps\dataset"
    dst = r"c:\Users\abbas\Documents\rps\external_unseen"
    create_external_test_dataset(src, dst, num_per_class=15)

