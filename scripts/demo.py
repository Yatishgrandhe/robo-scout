"""Render a scripted practice clip and independent synthetic detector fixtures."""
import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WIDTH, HEIGHT = 960, 540
COLORS = [(53, 208, 246), (57, 53, 232), (234, 145, 46)]


def court(width, height, rng, variation=False):
    yy, xx = np.mgrid[:height, :width]
    light = (12 - 10 * ((xx / width - .48) ** 2 + (yy / height - .42) ** 2))
    base = np.array([38, 42, 43]) + (rng.integers(-15, 18, 3) if variation else 0)
    frame = np.clip(base + light[:, :, None] + rng.normal(0, 1.5, (height, width, 1)), 0, 255).astype(np.uint8)
    for i in range(1, 7):
        cv.line(frame, (i * width // 7, 0), (i * width // 7, height), (48, 52, 53), 1)
    for i in range(1, 5):
        cv.line(frame, (0, i * height // 5), (width, i * height // 5), (48, 52, 53), 1)
    cv.rectangle(frame, (width // 40, height // 23), (width * 39 // 40, height * 22 // 23), (103, 108, 107), 2)
    return frame


def ball(frame, x, y, r, color):
    cv.ellipse(frame, (x + 2, y + 3), (r + 1, max(2, r // 2)), 0, 0, 360, (24, 27, 29), -1, cv.LINE_AA)
    cv.circle(frame, (x, y), r, tuple(int(c * .68) for c in color), -1, cv.LINE_AA)
    cv.circle(frame, (x - 1, y - 1), max(1, r - 1), color, -1, cv.LINE_AA)
    if r >= 6:
        cv.circle(frame, (x - r // 3, y - r // 3), max(1, r // 4), tuple(min(255, c + 39) for c in color), -1, cv.LINE_AA)


def robot(frame, x, y, w, h, red=True):
    x, y = int(x), int(y)
    left, top = x - w // 2, y - h // 2
    right, bottom = left + w, top + h
    color = (50, 65, 185) if red else (166, 113, 45)
    cv.rectangle(frame, (left + 4, top + 7), (right + 6, bottom + 7), (19, 22, 23), -1)
    cv.rectangle(frame, (left, top), (right, bottom), color, -1)
    cv.rectangle(frame, (left + 3, top + 3), (right - 3, bottom - 3), (58, 63, 64), -1)
    cv.rectangle(frame, (left + 8, top + 8), (right - 8, bottom - 8), (101, 111, 110), 2)
    cv.line(frame, (left + 9, top + 9), (right - 8, bottom - 8), (132, 142, 138), 2)
    cv.line(frame, (left + 9, bottom - 8), (right - 8, top + 9), (118, 125, 121), 2)
    cv.circle(frame, (x, y), max(3, min(w, h) // 5), (29, 32, 34), -1)
    for wx in [left + w // 4, right - w // 4]:
        for wy in [top, bottom]:
            cv.rectangle(frame, (wx - 4, wy - 3), (wx + 4, wy + 3), (22, 24, 25), -1)


def target_position(time):
    return round(458 + 92 * math.cos(.37 * time + .9)), round(270 + 53 * math.sin(.42 * time + .5))


def dataset():
    folder = DATA / "synthetic"
    for split, count, seed in [("train", 160, 72), ("val", 36, 191)]:
        rng = np.random.default_rng(seed)
        for kind in ("images", "labels"):
            (folder / kind / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            w, h = 448, 320
            frame = court(w, h, rng, True)
            for _ in range(7):
                x, y = rng.integers(12, [w - 35, h - 35])
                color = tuple(int(c) for c in rng.integers(35, 230, size=3))
                cv.rectangle(frame, (x, y), (x + int(rng.integers(4, 26)), y + int(rng.integers(3, 13))), color, -1)
            rows = []
            for _ in range(int(rng.integers(0, 4))):
                x, y = rng.integers(48, [w - 48, h - 48])
                rw, rh = rng.integers(28, 65, size=2)
                robot(frame, x, y, rw, rh, bool(rng.integers(0, 2)))
                rows.append((3, x / w, y / h, rw / w, rh / h))
            for _ in range(int(rng.integers(0, 18))):
                x, y = rng.integers(14, [w - 14, h - 14])
                r = int(rng.integers(4, 13))
                category = int(rng.integers(0, 3))
                ball(frame, x, y, r, COLORS[category])
                rows.append((category, x / w, y / h, 2 * r / w, 2 * r / h))
            if i % 4 == 0:
                frame = cv.GaussianBlur(frame, (3, 3), .7)
            cv.imwrite(str(folder / "images" / split / f"{i:04}.jpg"), frame)
            (folder / "labels" / split / f"{i:04}.txt").write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows))
    (folder / "dataset.yaml").write_text(
        f"path: {json.dumps(str(folder))}\ntrain: images/train\nval: images/val\n"
        "names: [pollen, red_nectar, blue_nectar, robot]\n")
    return folder / "dataset.yaml"


def demo():
    from backend.rules import event
    folder = DATA / "demo"
    folder.mkdir(parents=True, exist_ok=True)
    events = [
        event("hive_tip", 4.35, "Hive tips to the opposite cell", "§10.5.1", status="confirmed"),
        event("penalty", 7.7, "Early Nectar placed in the Flower", "G410", severity="major"),
        event("hive_tip", 13.3, "Second Hive Tip", "§10.5.1", status="confirmed"),
        event("cell_ball", 21.7, "Yellow Pollen remains in the Cell", "§10.5.1", status="confirmed", count=2),
        event("garden_ball", 21.8, "Pollen remains in the Garden", "§10.5.3", status="confirmed", count=3),
        event("flower_ball", 21.9, "Nectar remains in the owned Flower", "§10.5.2", status="confirmed"),
    ]
    rng = np.random.default_rng(117)
    floor = court(WIDTH, HEIGHT, rng)
    for x, y, name, color in [(104, 104, "RED GARDEN", (62, 60, 105)), (826, 395, "BLUE GARDEN", (113, 84, 48))]:
        cv.rectangle(floor, (x - 63, y - 36), (x + 67, y + 42), color, -1)
        cv.putText(floor, name, (x - 60, y + 61), cv.FONT_HERSHEY_SIMPLEX, .38, (152, 153, 148), 1, cv.LINE_AA)
    for x, y in [(213, 179), (734, 141), (227, 406), (747, 373)]:
        cv.circle(floor, (x + 4, y + 5), 32, (28, 32, 33), -1, cv.LINE_AA)
        cv.circle(floor, (x, y), 31, (133, 139, 115), 4, cv.LINE_AA)
        cv.circle(floor, (x, y), 24, (35, 39, 36), -1, cv.LINE_AA)
        cv.line(floor, (x - 17, y), (x + 17, y), (75, 81, 72), 1, cv.LINE_AA)
        cv.line(floor, (x, y - 17), (x, y + 17), (75, 81, 72), 1, cv.LINE_AA)
    trajectories = [(2.9, 4.35, 470, 160, 0), (6.3, 7.7, 213, 179, 1),
                    (11.8, 13.3, 490, 159, 0), (16.1, 18.2, 107, 103, 0)]
    frames = []
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
                             "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", "24", "-i", "-", "-an", "-c:v",
                             "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags",
                             "+faststart", str(folder / "practice.mp4")], stdin=subprocess.PIPE)
    try:
        for i in range(22 * 24):
            t = i / 24
            image = floor.copy()
            tipped = (4.35 <= t < 13.3)
            for hx in [480, 604]:
                cv.rectangle(image, (hx - 51, 126), (hx + 52, 204), (28, 32, 33), -1)
                cv.rectangle(image, (hx - 50, 124), (hx + 50, 200), (105, 113, 109), 2)
                slope = -20 if tipped and hx == 480 else 20
                cv.line(image, (hx - 41, 161 + slope), (hx + 41, 161 - slope), (172, 185, 175), 9, cv.LINE_AA)
                cv.circle(image, (hx, 161), 10, (40, 45, 46), -1, cv.LINE_AA)
            for x, y in [(305, 175), (573, 341), (681, 278), (351, 439), (820, 239), (113, 122), (88, 94)]:
                ball(image, x, y, 7, COLORS[0])
            target_x, target_y = target_position(t)
            robot(image, round(696 + 70 * math.sin(t * .4)), round(279 + 62 * math.cos(t * .3)), 79, 60, False)
            robot(image, target_x, target_y, 82, 61)
            observations = []
            for start, end, x1, y1, kind in trajectories:
                if t >= start:
                    x0, y0 = target_position(start)
                    fraction = min(1, (t - start) / (end - start))
                    x = round(x0 + (x1 - x0) * fraction)
                    y = round(y0 + (y1 - y0) * fraction - 39 * math.sin(math.pi * fraction))
                    ball(image, x, y, 8, COLORS[kind])
                    observations.append({"class": ["pollen", "red_nectar"][kind], "box": [(x-8)/WIDTH, (y-8)/HEIGHT, (x+8)/WIDTH, (y+8)/HEIGHT]})
            if i % 4 == 0:
                frames.append({"time": round(t, 3), "robot": [(target_x-43)/WIDTH, (target_y-34)/HEIGHT, (target_x+43)/WIDTH, (target_y+34)/HEIGHT], "balls": observations})
            if i == 0:
                cv.imwrite(str(folder / "poster.jpg"), image)
            proc.stdin.write(image.tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code:
        raise RuntimeError("Demo video encoding failed")
    (folder / "result.json").write_text(json.dumps({"events": events, "frames": frames, "coverage": 1, "mode": "scripted", "segmentation": False, "warnings": []}))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("asset", choices=["dataset", "video"])
    options = cli.parse_args()
    print(dataset() if options.asset == "dataset" else demo())
