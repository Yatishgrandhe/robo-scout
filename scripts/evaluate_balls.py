"""Compare the trained ball detector with the color baseline on held-out videos.

Reviewed labels come from color proposals, so they miss balls the color rule
never found. Model detections that match no label are cut onto review sheets
(extras) so a reviewer can count how many are real balls the baseline missed.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2 as cv
import numpy as np
from ultralytics import YOLO

from backend.vision import color_detections
from scripts.build_ball_dataset import COLUMNS, NAMES, PER_SHEET, crop_tile

ROOT = Path(__file__).resolve().parents[1]


def load_labels(path):
    rows = []
    for line in path.read_text().splitlines():
        cls, x, y, w, h = map(float, line.split())
        rows.append((NAMES[int(cls)], np.array((x - w / 2, y - h / 2, x + w / 2, y + h / 2))))
    return rows


def match(predicted, truth, threshold):
    """Greedy same-class matching on center distance relative to ball size."""
    used, hits = set(), []
    for index, (kind, box, _) in enumerate(sorted(predicted, key=lambda row: -row[2])):
        center = (box[:2] + box[2:]) / 2
        best, best_distance = None, None
        for j, (truth_kind, truth_box) in enumerate(truth):
            if j in used or truth_kind != kind:
                continue
            size = max(truth_box[2] - truth_box[0], truth_box[3] - truth_box[1])
            distance = np.linalg.norm(center - (truth_box[:2] + truth_box[2:]) / 2) / max(size, 1e-6)
            if distance <= threshold and (best_distance is None or distance < best_distance):
                best, best_distance = j, distance
        if best is not None:
            used.add(best)
            hits.append(index)
    return len(used), [row for i, row in enumerate(sorted(predicted, key=lambda row: -row[2])) if i not in hits]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--out", default="evaluation", help="Output folder inside the dataset")
    parser.add_argument("--threshold", type=float, default=.6, help="Max center distance in ball diameters")
    args = parser.parse_args()
    model = YOLO(args.weights)
    images = sorted((args.dataset / "images/val").glob("*.jpg"))
    stats = {name: Counter() for name in ("model", "color")}
    extras, tiles = [], []
    for image_path in images:
        frame = cv.imread(str(image_path))
        truth = load_labels(args.dataset / "labels/val" / f"{image_path.stem}.txt")
        h, w = frame.shape[:2]
        result = model.predict(frame, imgsz=args.size, conf=args.conf, device="cpu", verbose=False)[0]
        predicted = [(result.names[int(c)], np.array(b) / (w, h, w, h), float(s))
                     for *b, s, c in result.boxes.data.cpu().numpy() if result.names[int(c)] in NAMES]
        for name, rows in (("model", predicted), ("color", color_detections(frame))):
            found, unmatched = match(rows, truth, args.threshold)
            stats[name].update(truth=len(truth), found=found, predicted=len(rows), unmatched=len(unmatched))
            if name == "model":
                for kind, box, score in unmatched:
                    tiles.append(crop_tile(frame, box, kind, len(extras)))
                    extras.append({"id": len(extras), "image": image_path.stem, "kind": kind,
                                   "box": box.round(5).tolist(), "score": round(score, 3)})
    report = {}
    for name, counter in stats.items():
        recall = counter["found"] / max(1, counter["truth"])
        precision_vs_labels = counter["found"] / max(1, counter["predicted"])
        report[name] = {**counter, "recall": round(recall, 3), "precision_vs_labels": round(precision_vs_labels, 3)}
    out = args.dataset / args.out
    out.mkdir(exist_ok=True)
    for start in range(0, len(tiles), PER_SHEET):
        block = tiles[start:start + PER_SHEET]
        block += [np.zeros_like(tiles[0])] * (-len(block) % COLUMNS)
        sheet = np.vstack([np.hstack(block[i:i + COLUMNS]) for i in range(0, len(block), COLUMNS)])
        cv.imwrite(str(out / f"extras-{start // PER_SHEET:04d}.jpg"), sheet)
    (out / "extras.json").write_text(json.dumps(extras))
    report["validation_images"] = len(images)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
