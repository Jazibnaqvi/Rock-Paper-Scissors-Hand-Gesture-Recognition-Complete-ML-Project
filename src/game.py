"""
game.py - Interactive Rock-Paper-Scissors Webcam Game Against Computer Opponent.

Implements Phases 16, 17, 18, and 19:
- Game flow: Waiting -> Countdown (3,2,1) -> Hand Detection -> Computer Move -> Winner Resolution -> Anti-repeat Cooldown
- Robust finite state machine (FSM) preventing accidental double rounds
- Computer AI with hidden move revealed upon player gesture confirmation
- Polished OpenCV interface with score tracking, badges, and animated round results
- Dynamically scales to any webcam resolution (640x480, 1280x720, etc.)
"""

import os
import sys
import time
import random
from collections import deque
from typing import Tuple, Optional

import cv2
import numpy as np

# Ensure local src directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import RPSPredictor, DEFAULT_MODEL_PATH


# Game States
STATE_WAITING = "WAITING"
STATE_COUNTDOWN = "COUNTDOWN"
STATE_DETECTING = "DETECTING"
STATE_RESULT = "RESULT"

# Move definitions
MOVES = ["Rock", "Paper", "Scissors"]


def determine_winner(player_move: str, computer_move: str) -> Tuple[str, str]:
    """
    Standard Rock-Paper-Scissors rules:
    Rock beats Scissors
    Scissors beats Paper
    Paper beats Rock
    """
    p = player_move.capitalize()
    c = computer_move.capitalize()

    if p == c:
        return "draw", f"Both chose {p} - It's a DRAW!"

    if (p == "Rock" and c == "Scissors") or \
       (p == "Scissors" and c == "Paper") or \
       (p == "Paper" and c == "Rock"):
        return "player", f"{p} beats {c} - YOU WIN!"
    else:
        return "computer", f"{c} beats {p} - COMPUTER WINS!"


def put_centered_text(img, text, y, font, scale, color, thickness=2):
    """Utility to draw horizontally centered text at a given y coordinate."""
    w = img.shape[1]
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (w - text_w) // 2)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


class RPSGame:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        camera_id: int = 0,
        roi_size: int = 260,
        countdown_seconds: float = 3.0,
        detection_duration: float = 1.2,
        result_display_seconds: float = 3.5,
        min_confidence: float = 0.65
    ):
        self.camera_id = camera_id
        self.roi_size = roi_size
        self.countdown_seconds = countdown_seconds
        self.detection_duration = detection_duration
        self.result_display_seconds = result_display_seconds
        self.min_confidence = min_confidence

        print(f"[Game] Initializing gesture classifier from {model_path}...")
        self.predictor = RPSPredictor(model_path=model_path)
        print(f"[Game] Classifier ready on {self.predictor.device}.")

        # Scores
        self.score_player = 0
        self.score_computer = 0
        self.score_draws = 0

        # State tracking
        self.state = STATE_WAITING
        self.state_start_time = time.time()

        # Detection buffers
        self.detection_votes = []
        self.confirmed_player_move = None
        self.confirmed_confidence = 0.0
        self.computer_move = None
        self.winner = None
        self.result_message = ""

        # Live smoothed prediction queue
        self.live_probs_queue = deque(maxlen=8)

    def reset_scores(self):
        self.score_player = 0
        self.score_computer = 0
        self.score_draws = 0

    def start_round(self):
        if self.state == STATE_WAITING:
            self.state = STATE_COUNTDOWN
            self.state_start_time = time.time()
            self.detection_votes.clear()
            self.confirmed_player_move = None
            self.confirmed_confidence = 0.0
            self.computer_move = None
            self.winner = None
            self.result_message = ""

    def update(self, roi_crop: np.ndarray):
        """Update game state machine based on timer and ROI inference."""
        now = time.time()
        elapsed = now - self.state_start_time

        # Run live inference for continuous feedback
        pred_res = self.predictor.predict(roi_crop)
        curr_probs = np.array([
            pred_res["probabilities"]["Rock"],
            pred_res["probabilities"]["Paper"],
            pred_res["probabilities"]["Scissors"]
        ])
        self.live_probs_queue.append(curr_probs)
        smoothed = np.mean(self.live_probs_queue, axis=0)
        live_idx = int(np.argmax(smoothed))
        live_gesture = MOVES[live_idx]
        live_conf = float(smoothed[live_idx])

        if self.state == STATE_COUNTDOWN:
            if elapsed >= self.countdown_seconds:
                self.state = STATE_DETECTING
                self.state_start_time = time.time()
                self.detection_votes.clear()

        elif self.state == STATE_DETECTING:
            if live_conf >= self.min_confidence:
                self.detection_votes.append((live_gesture, live_conf))

            if elapsed >= self.detection_duration:
                if len(self.detection_votes) > 0:
                    gestures = [v[0] for v in self.detection_votes]
                    voted_gesture = max(set(gestures), key=gestures.count)
                    avg_conf = float(np.mean([v[1] for v in self.detection_votes if v[0] == voted_gesture]))
                    self.confirmed_player_move = voted_gesture
                    self.confirmed_confidence = avg_conf
                else:
                    self.confirmed_player_move = live_gesture
                    self.confirmed_confidence = live_conf

                # Computer secretly chooses move NOW
                self.computer_move = random.choice(MOVES)

                # Determine winner
                self.winner, self.result_message = determine_winner(
                    self.confirmed_player_move, self.computer_move
                )

                if self.winner == "player":
                    self.score_player += 1
                elif self.winner == "computer":
                    self.score_computer += 1
                else:
                    self.score_draws += 1

                # Enter RESULT state with anti-repeat cooldown lock
                self.state = STATE_RESULT
                self.state_start_time = time.time()

        elif self.state == STATE_RESULT:
            if elapsed >= self.result_display_seconds:
                self.state = STATE_WAITING
                self.state_start_time = time.time()

        return live_gesture, live_conf

    def render(self, frame: np.ndarray, live_gesture: str, live_conf: float, fps: float):
        """Draw complete polished game UI overlay (Phase 18)."""
        h, w = frame.shape[:2]

        # Calculate centered ROI
        effective_roi = min(self.roi_size, int(min(h, w) * 0.55))
        rx1 = max(0, (w - effective_roi) // 2)
        ry1 = max(0, (h - effective_roi) // 2)
        rx2 = min(w, rx1 + effective_roi)
        ry2 = min(h, ry1 + effective_roi)

        # 1. TOP HEADER & SCOREBOARD
        top_h = int(h * 0.16)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, top_h), (15, 20, 28), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Title
        put_centered_text(frame, "ROCK - PAPER - SCISSORS", int(top_h * 0.42),
                          cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

        # Score Badges
        score_str = f"YOU: {self.score_player}   |   COMPUTER: {self.score_computer}   |   DRAWS: {self.score_draws}"
        put_centered_text(frame, score_str, int(top_h * 0.85),
                          cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 240, 255), 2)

        # FPS counter
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 75, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 180, 180), 1, cv2.LINE_AA)

        # 2. ROI GUIDE BOX
        roi_color = (0, 255, 120) if self.state != STATE_RESULT else (80, 100, 120)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roi_color, 2)
        cv2.putText(frame, "PLACE HAND HERE", (rx1 + 8, ry1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, roi_color, 1, cv2.LINE_AA)

        # 3. STATE OVERLAYS
        now = time.time()
        elapsed = now - self.state_start_time

        if self.state == STATE_WAITING:
            bot_h = int(h * 0.16)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - bot_h), (w, h), (15, 20, 28), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            put_centered_text(frame, "PRESS [SPACE] TO START ROUND  |  [R] RESET  |  [Q] QUIT",
                              h - int(bot_h * 0.55), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 200), 2)
            put_centered_text(frame, f"Live Hand Preview: {live_gesture.upper()} ({live_conf*100:.1f}%)",
                              h - int(bot_h * 0.18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 220), 1)

        elif self.state == STATE_COUNTDOWN:
            remaining = max(0.0, self.countdown_seconds - elapsed)
            count_val = int(np.ceil(remaining))
            overlay = frame.copy()
            cv2.rectangle(overlay, (rx1 - 10, ry1 - 10), (rx2 + 10, ry2 + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            msg = str(count_val) if count_val > 0 else "READY!"
            put_centered_text(frame, msg, (ry1 + ry2) // 2 + 25,
                              cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 255, 255), 4)
            put_centered_text(frame, "GET YOUR HAND READY!", ry2 + 35,
                              cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

        elif self.state == STATE_DETECTING:
            put_centered_text(frame, "SHOW YOUR GESTURE NOW!", ry1 - 15,
                              cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
            pct = min(1.0, elapsed / self.detection_duration)
            bar_w = int((rx2 - rx1) * pct)
            cv2.rectangle(frame, (rx1, ry2 + 8), (rx1 + bar_w, ry2 + 20), (0, 255, 120), -1)

        elif self.state == STATE_RESULT:
            overlay = frame.copy()
            pad_x = int(w * 0.05)
            pad_y_top = int(h * 0.20)
            pad_y_bot = int(h * 0.88)
            cv2.rectangle(overlay, (pad_x, pad_y_top), (w - pad_x, pad_y_bot), (12, 16, 22), -1)
            cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

            res_color = (0, 255, 120) if self.winner == "player" else (
                (0, 100, 255) if self.winner == "computer" else (0, 220, 255)
            )
            cv2.rectangle(frame, (pad_x, pad_y_top), (w - pad_x, pad_y_bot), res_color, 2)

            col1_x = pad_x + int(w * 0.06)
            col2_x = w - pad_x - int(w * 0.35)

            # Player Move
            cv2.putText(frame, "YOU CHOSE", (col1_x, pad_y_top + 38),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{self.confirmed_player_move.upper()}", (col1_x, pad_y_top + 80),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 255, 120), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {self.confirmed_confidence * 100:.1f}%", (col1_x, pad_y_top + 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 180), 1, cv2.LINE_AA)

            # Center VS
            put_centered_text(frame, "VS", pad_y_top + 75,
                              cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

            # Computer Move
            cv2.putText(frame, "COMPUTER CHOSE", (col2_x, pad_y_top + 38),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{self.computer_move.upper()}", (col2_x, pad_y_top + 80),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Computer AI Move", (col2_x, pad_y_top + 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 180), 1, cv2.LINE_AA)

            # Winner Announcement
            banner_title = "YOU WIN!" if self.winner == "player" else (
                "COMPUTER WINS!" if self.winner == "computer" else "IT'S A DRAW!"
            )
            put_centered_text(frame, banner_title, pad_y_top + 175,
                              cv2.FONT_HERSHEY_DUPLEX, 1.15, res_color, 2)
            put_centered_text(frame, self.result_message, pad_y_top + 210,
                              cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)

            rem_cd = max(0.0, self.result_display_seconds - elapsed)
            put_centered_text(frame, f"Next round starts in {rem_cd:.1f}s (or press [SPACE])...",
                              pad_y_bot - 15, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 180), 1)

    def play(self):
        """Main game loop."""
        print(f"[Game] Opening camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"[Error] Failed to open camera {self.camera_id}. Please connect a webcam.")
            return

        window_name = "Rock-Paper-Scissors: Computer Challenge"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        prev_t = time.time()
        fps = 0.0

        print("\n=======================================================")
        print(" ROCK-PAPER-SCISSORS GAME ACTIVE!")
        print(" Controls:")
        print("   [SPACE] : Start Round / Skip Result Cooldown")
        print("   [R]     : Reset Scoreboard")
        print("   [Q]/ESC : Exit Game")
        print("=======================================================\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                effective_roi = min(self.roi_size, int(min(h, w) * 0.55))
                rx1 = max(0, (w - effective_roi) // 2)
                ry1 = max(0, (h - effective_roi) // 2)
                rx2 = min(w, rx1 + effective_roi)
                ry2 = min(h, ry1 + effective_roi)
                roi = frame[ry1:ry2, rx1:rx2]

                now = time.time()
                dt = now - prev_t
                prev_t = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                live_gesture, live_conf = self.update(roi)
                self.render(frame, live_gesture, live_conf, fps)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break
                elif key == 32:  # Spacebar
                    if self.state == STATE_WAITING:
                        self.start_round()
                    elif self.state == STATE_RESULT:
                        self.state = STATE_WAITING
                        self.state_start_time = time.time()
                elif key == ord('r') or key == ord('R'):
                    self.reset_scores()
                    print("[Game] Scoreboard reset to 0-0-0.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n[Game] Thank you for playing Rock-Paper-Scissors!")
            print(f"Final Score -> You: {self.score_player} | Computer: {self.score_computer} | Draws: {self.score_draws}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rock-Paper-Scissors Webcam Game Against Computer Opponent.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--roi-size", type=int, default=260)
    args = parser.parse_args()

    game = RPSGame(model_path=args.model, camera_id=args.camera, roi_size=args.roi_size)
    game.play()
