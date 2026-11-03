"""Deterministic 720p, 6 FPS ground-truth lecture scenarios; no downloaded assets."""
from pathlib import Path
import sys

import cv2
import numpy as np

SCENARIOS = ("progressive", "occlusion", "partial_erase", "rewrite", "cut", "motion_only", "long_hold", "never_visible")
FPS = 6


def board(lines=2, alternate=False, white=False):
    im = np.full((720, 1280, 3), (232, 234, 235) if white else (42, 56, 45), np.uint8)
    color = (25, 25, 25) if white else (230, 235, 228)
    cv2.putText(im, "B: y = sin(x)" if alternate else "A: y = x^2 + 1", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3, cv2.LINE_AA)
    if lines >= 2:
        cv2.putText(im, "dy/dx = 2x", (180, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3, cv2.LINE_AA)
    if lines >= 3:
        cv2.putText(im, "x = 2 : y = 5", (180, 550), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3, cv2.LINE_AA)
    if lines == 0:
        im[:] = (232, 234, 235) if white else (42, 56, 45)
    return im


def render(name, t, white=False):
    if name == "progressive":
        n = min(3, int(t // 3))
        truth = board(n, white=white)
        label = str(n)
    elif name == "partial_erase":
        n = 2 if t < 6 else 1
        truth, label = board(n, white=white), str(n)
    elif name == "rewrite":
        truth, label = board(1, alternate=t >= 6, white=white), "B" if t >= 6 else "A"
    elif name == "cut":
        truth, label = board(2, white=t >= 6), "scene2" if t >= 6 else "scene1"
    else:
        truth, label = board(2, white=white), "same"
    image = truth.copy()
    if name in ("occlusion", "motion_only", "never_visible"):
        if name == "never_visible":
            x = 180
        elif name == "occlusion":
            # Both observations have an occluder; different board regions are exposed.
            x = 180 if int(t // 2) % 2 == 0 else 350
        else:
            x = int(60 + (t*100) % 950)
        width = 150 if name == "occlusion" else 220
        cv2.rectangle(image, (x, 150), (x+width, 710), (115, 101, 85), -1)
        cv2.circle(image, (x+width//2, 130), 60, (160, 160, 190), -1)
    return image, truth, label


def generate(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in SCENARIOS:
        path = root / (name+".mp4")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
        if not writer.isOpened():
            raise RuntimeError("MP4 writer unavailable")
        duration = 40 if name == "long_hold" else 12
        for i in range(duration*FPS):
            writer.write(render(name, i/FPS)[0])
        writer.release()
        paths[name] = path
    srt = root / "example.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:04,000\n함수를 작성합니다.\n\n2\n00:00:04,000 --> 00:00:12,000\n원래 발언을 보존합니다. <script>alert(1)</script>\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    generate(sys.argv[1] if len(sys.argv) > 1 else "experiments/inputs")
