"""
webcam.py - Real-Time Webcam Rock-Paper-Scissors Gesture Recognition.

Implements Phase 14 & 15:
- OpenCV webcam capture loop
- Dedicated Region of Interest (ROI) box with corner styling
- Confidence threshold filtering
- Temporal prediction smoothing (sliding window queue)
- Real-time FPS, predicted gesture, and confidence visualization
- Dynamically scales to any webcam resolution (640x480, 1280x720, etc.)
"""

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

# Ensure local src directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import predictor
from predict import RPSPredictor, DEFAULT_MODEL_PATH


def draw_styled_box(img, pt1, pt2, color=(0, 255, 128), thickness=2):
    """Draw a smooth rounded rectangle / ROI guide with corner accents."""
    x1, y1 = pt1
    x2, y2 = pt2
    d = min(25, (x2 - x1) // 4, (y2 - y1) // 4)
    # Corners
    cv2.line(img, (x1, y1), (x1 + d, y1), color, thickness + 1)
    cv2.line(img, (x1, y1), (x1, y1 + d), color, thickness + 1)
    cv2.line(img, (x2, y1), (x2 - d, y1), color, thickness + 1)
    cv2.line(img, (x2, y1), (x2, y1 + d), color, thickness + 1)
    cv2.line(img, (x1, y2), (x1 + d, y2), color, thickness + 1)
    cv2.line(img, (x1, y2), (x1, y2 - d), color, thickness + 1)
    cv2.line(img, (x2, y2), (x2 - d, y2), color, thickness + 1)
    cv2.line(img, (x2, y2), (x2, y2 - d), color, thickness + 1)
    cv2.rectangle(img, pt1, pt2, color, 1)


def draw_hud(
    frame: np.ndarray,
    gesture: str,
    confidence: float,
    fps: float,
    roi_coords: Tuple[int, int, int, int],
    smoothed_probs: np.ndarray,
    threshold: float = 0.65
):
    """Draw clean, informative HUD overlay on the webcam frame."""
    h, w = frame.shape[:2]
    rx1, ry1, rx2, ry2 = roi_coords

    # Top banner
    top_h = int(h * 0.14)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Title & FPS
    cv2.putText(frame, "Rock-Paper-Scissors Live Recognition", (18, int(top_h * 0.45)),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 85, int(top_h * 0.45)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (180, 200, 200), 1, cv2.LINE_AA)

    # Instructions
    cv2.putText(frame, "Place hand gesture inside guide box | Press 'q' to exit", (18, int(top_h * 0.82)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 180, 190), 1, cv2.LINE_AA)

    # Draw ROI guide box
    box_color = (0, 255, 120) if confidence >= threshold else (0, 180, 255)
    draw_styled_box(frame, (rx1, ry1), (rx2, ry2), color=box_color, thickness=2)
    cv2.putText(frame, "HAND ROI", (rx1 + 5, ry1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

    # Bottom results panel
    bot_h = int(h * 0.17)
    panel_y = h - bot_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    if confidence >= threshold:
        label_text = f"Gesture: {gesture.upper()}"
        conf_text = f"Confidence: {confidence * 100:.1f}%"
        color = (0, 255, 120) if gesture.lower() == "rock" else (
            (255, 200, 0) if gesture.lower() == "paper" else (0, 140, 255)
        )
    else:
        label_text = "Gesture: UNCERTAIN"
        conf_text = f"Confidence: {confidence * 100:.1f}% (<{int(threshold*100)}%)"
        color = (120, 140, 160)

    cv2.putText(frame, label_text, (20, panel_y + int(bot_h * 0.45)),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(frame, conf_text, (20, panel_y + int(bot_h * 0.82)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    # Mini confidence bar graph for Rock / Paper / Scissors
    bar_x = w - 190
    bar_w = 120
    classes = ["R", "P", "S"]
    colors = [(0, 255, 120), (255, 200, 0), (0, 140, 255)]
    for i, cname in enumerate(classes):
        prob = smoothed_probs[i] if smoothed_probs is not None and len(smoothed_probs) == 3 else 0.0
        by = panel_y + 12 + i * 19
        cv2.putText(frame, f"{cname}:", (bar_x - 18, by + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w, by + 10), (50, 50, 60), -1)
        fill_w = int(bar_w * np.clip(prob, 0.0, 1.0))
        cv2.rectangle(frame, (bar_x, by), (bar_x + fill_w, by + 10), colors[i], -1)


def run_webcam(
    model_path: str = DEFAULT_MODEL_PATH,
    camera_id: int = 0,
    confidence_threshold: float = 0.65,
    smoothing_window: int = 10,
    roi_size: int = 260
):
    """
    Phase 14 & 15:
    Runs real-time webcam gesture classification with ROI, confidence gating,
    and temporal sliding window voting.
    """
    print(f"[Webcam] Loading model from {model_path}...")
    predictor = RPSPredictor(model_path=model_path)
    print(f"[Webcam] Model successfully loaded on device: {predictor.device}")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Error] Could not open camera {camera_id}. Please verify webcam connection.")
        return

    # Buffer for temporal prediction smoothing
    prob_buffer = deque(maxlen=smoothing_window)
    prev_time = time.time()
    fps = 0.0

    window_name = "Rock-Paper-Scissors Hand Gesture Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[Webcam Active] Showing live video stream. Press 'q' or 'ESC' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror frame for natural mirroring
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Calculate centered ROI
            effective_roi = min(roi_size, int(min(h, w) * 0.55))
            rx1 = max(0, (w - effective_roi) // 2)
            ry1 = max(0, (h - effective_roi) // 2)
            rx2 = min(w, rx1 + effective_roi)
            ry2 = min(h, ry1 + effective_roi)

            roi = frame[ry1:ry2, rx1:rx2]

            res = predictor.predict(roi)
            current_probs = np.array([
                res["probabilities"]["Rock"],
                res["probabilities"]["Paper"],
                res["probabilities"]["Scissors"]
            ])

            prob_buffer.append(current_probs)
            smoothed_probs = np.mean(prob_buffer, axis=0)
            smoothed_idx = int(np.argmax(smoothed_probs))
            smoothed_confidence = float(smoothed_probs[smoothed_idx])
            smoothed_gesture = predictor.classes[smoothed_idx].capitalize()

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            draw_hud(
                frame=frame,
                gesture=smoothed_gesture,
                confidence=smoothed_confidence,
                fps=fps,
                roi_coords=(rx1, ry1, rx2, ry2),
                smoothed_probs=smoothed_probs,
                threshold=confidence_threshold
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Webcam] Stream closed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live webcam Rock-Paper-Scissors gesture classifier.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--smoothing", type=int, default=10)
    args = parser.parse_args()

    run_webcam(
        model_path=args.model,
        camera_id=args.camera,
        confidence_threshold=args.threshold,
        smoothing_window=args.smoothing
    )
