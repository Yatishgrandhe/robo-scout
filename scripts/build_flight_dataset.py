"""Build a ball detector dataset that includes balls in flight.

The FTC ball detector was trained on sampled frames, where balls mostly rest on
the floor, so it rarely detects a blurred ball in the air. Reviewers confirmed
launches in data/match-labels/<video>/ (Task B), and each confirmed launch
carries its flight path. Every --every-th flight point becomes a ball box in
that frame. Other balls in the frame are labeled by the current detector so they
are not learned as background. Whole videos in --val are held out.

    python -m scripts.build_flight_dataset --val Y6uT5ziHRJY 2EDHhlj1vQk kwYU2RU7V14
    python -m scripts.train data/flight-dataset/dataset.yaml --weights models/ftc.pt --size 960 --batch 8 --name ball-flight
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["pollen", "red_nectar", "blue_nectar"]


def flights(labels):
    """{video: {frame time: [(class, x, y)]}} from confirmed launches."""
    found = defaultdict(lambda: defaultdict(list))
    for stats in sorted(labels.glob("*/stats.json")):
        for row in json.loads(stats.read_text())["launches"]:
            if not row.get("launch") or row.get("kind") not in NAMES:
                continue
            for time, x, y in row.get("flight") or ():
                found[stats.parent.name][round(time, 3)].append((NAMES.index(row["kind"]), x, y))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=ROOT / "data/match-labels")
    parser.add_argument("--videos", type=Path, default=ROOT / "data/videos")
    parser.add_argument("--out", type=Path, default=ROOT / "data/flight-dataset")
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--every", type=int, default=2, help="Use every n-th flight point")
    parser.add_argument("--teacher", type=Path, default=ROOT / "models/ftc.pt")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    from ultralytics import YOLO
    teacher = YOLO(args.teacher)
    shutil.rmtree(args.out, ignore_errors=True)
    counts = defaultdict(int)
    for video, by_time in sorted(flights(args.labels).items()):
        split = "val" if video in args.val else "train"
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        cap = cv.VideoCapture(str(args.videos / f"{video}.mp4"))
        for n, time in enumerate(sorted(by_time)):
            if n % args.every:
                continue
            cap.set(cv.CAP_PROP_POS_MSEC, time * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            result = teacher.predict(frame, imgsz=960, conf=.4, device=args.device, verbose=False)[0]
            boxes = [(int(c), (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)
                     for x0, y0, x1, y1, _, c in result.boxes.data.cpu().numpy() if result.names[int(c)] in NAMES]
            # Size a flying ball like the resting balls the teacher found, stretched for motion blur.
            size = float(np.median([b[3] for b in boxes])) if boxes else 12 / w
            for cls, x, y in by_time[time]:
                boxes = [b for b in boxes if abs(b[1] - x) > size or abs(b[2] - y) > size]
                boxes.append((cls, x, y, size * 1.1, size * w / h * 1.3))
                counts[(split, "flight")] += 1
            stem = f"{video}-{time:08.3f}"
            cv.imwrite(str(args.out / "images" / split / f"{stem}.jpg"), frame)
            (args.out / "labels" / split / f"{stem}.txt").write_text(
                "".join(f"{c} {x:.5f} {y:.5f} {bw:.5f} {bh:.5f}\n" for c, x, y, bw, bh in boxes))
            counts[(split, "frames")] += 1
        print(video, split, dict(counts), flush=True)
    (args.out / "dataset.yaml").write_text(
        f'path: "{args.out}"\ntrain: images/train\nval: images/val\nnames: {json.dumps(NAMES)}\n')
    print("done", dict(counts))


if __name__ == "__main__":
    main()
