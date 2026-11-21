"""Train a classifier that tells one team's robot from the others.

Input is a review folder made from robot detections on sampled match frames:
frames.json lists each frame (video, time, boxes in 960-pixel coordinates);
labels-*.json (whole frames) and strip-labels-*.json or strip2-labels-*.json
(enlarged crops, answered only when sure) are independent reviews, each mapping
a review image to the team's box number or null. A frame is an anchor when a
crop review names a box and no review names a different one. From each anchor the robot is followed with the robot detector, forward
and backward, only while exactly one detection clearly continues its box.
Every followed box is a "team" example; other robots in the same frames are
"other" examples. Videos passed to --val are held out for validation.

    python -m scripts.train_identity /path/to/review-folder --team 26691 --val Y6uT5ziHRJY

The result is models/identity-<team>.pt. Robot Scout uses it for that team when
a robot detector (models/robot.pt) is present.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2 as cv
import numpy as np

from backend.vision import iou, robot_crop

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 960


def read_labels(folder, pattern):
    labels = {}
    for path in sorted(folder.glob(pattern)):
        labels.update(json.loads(path.read_text()))
    return labels


def anchors(folder):
    """Frames where a strict review names the team's box and no review names another."""
    reviews = [read_labels(folder, pattern) for pattern in ("strip-labels-*.json", "strip2-labels-*.json", "labels-*.json")]
    rows, agreed, conflicts = [], 0, 0
    for row in json.loads((folder / "frames.json").read_text()):
        named = [review.get(row["review"]) for review in reviews]
        strict = {int(value) for value in named[:2] if value is not None}
        if not strict:
            continue
        chosen, other = min(strict), named[2]
        if len(strict) > 1 or (other is not None and int(other) != chosen) or not 0 <= chosen < len(row["boxes"]):
            conflicts += 1
            continue
        agreed += other is not None
        rows.append({**row, "box": row["boxes"][int(chosen)]})
    print(f"{len(rows)} anchors ({agreed} confirmed by both reviews), {conflicts} conflicting frames dropped")
    return rows


class Video:
    def __init__(self, path, detector, device):
        self.cap = cv.VideoCapture(str(path))
        self.detector, self.device = detector, device
        self.duration = self.cap.get(cv.CAP_PROP_FRAME_COUNT) / max(self.cap.get(cv.CAP_PROP_FPS), 1)

    def frame(self, time):
        self.cap.set(cv.CAP_PROP_POS_MSEC, time * 1000)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv.resize(frame, (WIDTH, round(frame.shape[0] * WIDTH / frame.shape[1])), interpolation=cv.INTER_AREA)

    def robots(self, frame):
        result = self.detector.predict(frame, imgsz=640, conf=.35, device=self.device, verbose=False)[0]
        return [np.array(box) for *box, _, _ in result.boxes.data.cpu().numpy()]


def follow(video, anchor, step, span):
    """Frames (time, image, team box, other boxes) where the anchored robot is unambiguous."""
    samples = []
    for direction in (1, -1):
        box, time = np.array(anchor["box"]), anchor["time"]
        for _ in range(int(span / step) + 1):
            frame = video.frame(time)
            if frame is None:
                break
            robots = video.robots(frame)
            overlaps = sorted(((iou(other, box), index) for index, other in enumerate(robots)), reverse=True)
            if not overlaps or overlaps[0][0] < .4 or (len(overlaps) > 1 and overlaps[1][0] > .15):
                break
            team = robots[overlaps[0][1]]
            area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
            if not .6 <= area(team) / area(box) <= 1.6:
                break
            others = [other for index, other in enumerate(robots) if index != overlaps[0][1] and iou(other, team) < .1]
            if direction == 1 or time != anchor["time"]:
                samples.append((time, frame, team, others))
            box, time = team, time + direction * step
            if not 0 <= time <= video.duration:
                break
    return samples


def build(folder, videos, val, out, device, step=.2, span=6., every=2):
    from ultralytics import YOLO
    detector = YOLO(ROOT / "models/robot.pt")
    shutil.rmtree(out, ignore_errors=True)
    counts, seen, cache = {}, set(), {}
    for anchor in anchors(folder):
        video = cache.setdefault(anchor["video"], Video(videos / f"{anchor['video']}.mp4", detector, device))
        split = "val" if anchor["video"] in val else "train"
        for index, (time, frame, team, others) in enumerate(follow(video, anchor, step, span)):
            key = (anchor["video"], round(time, 1))
            if index % every or key in seen:
                continue
            seen.add(key)
            h, w = frame.shape[:2]
            for kind, boxes in (("team", [team]), ("other", others)):
                target = out / split / kind
                target.mkdir(parents=True, exist_ok=True)
                for number, box in enumerate(boxes):
                    name = f"{anchor['video']}-{time:07.2f}-{number}.jpg"
                    cv.imwrite(str(target / name), robot_crop(frame, box / (w, h, w, h)))
                    counts[(split, kind)] = counts.get((split, kind), 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--team", required=True)
    parser.add_argument("--val", nargs="+", required=True, help="Video ids held out for validation")
    parser.add_argument("--videos", type=Path, default=ROOT / "data/videos")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    out = ROOT / "data" / f"identity-{args.team}"
    counts = build(args.folder, args.videos, set(args.val), out, args.device)
    print({f"{split}/{kind}": n for (split, kind), n in sorted(counts.items())})
    from ultralytics import YOLO
    model = YOLO("yolo26n-cls.pt")
    model.train(data=str(out), imgsz=128, epochs=args.epochs, batch=32, device=args.device, workers=0,
                project=str(ROOT / "runs"), name=f"identity-{args.team}", exist_ok=True, fliplr=.5, hsv_h=.01)
    best = ROOT / "runs" / f"identity-{args.team}" / "weights" / "best.pt"
    shutil.copy(best, ROOT / "models" / f"identity-{args.team}.pt")
    print(f"Saved models/identity-{args.team}.pt")


if __name__ == "__main__":
    main()
