"""Build a YOLO game-piece dataset from match videos with reviewable proposals.

Frames are sampled from each video and HSV/geometry proposals from
backend.vision.color_detections become draft labels. Every proposal is also
cut into a numbered crop on a review sheet, so a reviewer can reject the ones
that are not Pollen or Nectar. Run with --apply-review to drop rejected boxes.
Whole videos are held out for validation so the score reflects unseen matches.
"""
import argparse
import json
import random
from pathlib import Path

import cv2 as cv
import numpy as np

from backend.vision import color_detections

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["pollen", "red_nectar", "blue_nectar"]
COLORS = {"pollen": (0, 220, 255), "red_nectar": (40, 40, 255), "blue_nectar": (255, 140, 30)}
TILE, COLUMNS, PER_SHEET = 96, 10, 60


def sample_frames(video, every, width):
    cap = cv.VideoCapture(str(video))
    fps = cap.get(cv.CAP_PROP_FPS) or 30
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    time = min(8.0, duration * .05)
    while time < duration - 3:
        cap.set(cv.CAP_PROP_POS_FRAMES, round(time * fps))
        ok, frame = cap.read()
        if not ok:
            break
        height = round(frame.shape[0] * width / frame.shape[1] / 2) * 2
        yield time, cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
        time += every


def crop_tile(frame, box, kind, number):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box * (w, h, w, h)
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 3.2 + 16
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    tile = cv.resize(frame[b:d, a:c], (TILE, TILE), interpolation=cv.INTER_CUBIC)
    scale = TILE / max(1, c - a), TILE / max(1, d - b)
    cv.rectangle(tile, (round((x0 - a) * scale[0]), round((y0 - b) * scale[1])),
                 (round((x1 - a) * scale[0]), round((y1 - b) * scale[1])), COLORS[kind], 1)
    cv.rectangle(tile, (0, 0), (TILE - 1, 14), (0, 0, 0), -1)
    cv.putText(tile, f"{number} {kind[0].upper()}", (2, 11), cv.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv.LINE_AA)
    return tile


def build(args):
    manifest = json.loads((args.videos / "manifest.json").read_text())
    videos = sorted(key for key, value in manifest.items() if value.get("file"))
    if len(videos) < 3:
        raise SystemExit("Download at least three videos so whole matches can be held out.")
    random.Random(408).shuffle(videos)
    held_out = set(videos[:args.val_videos])
    out = args.output
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(exist_ok=True)
    proposals, tiles = [], []
    rng = random.Random(7)
    for vid in videos:
        split = "val" if vid in held_out else "train"
        kept = 0
        for time, frame in sample_frames(args.videos / manifest[vid]["file"], args.every, args.width):
            found = color_detections(frame)
            if not found and rng.random() > args.empty_share:
                continue
            key = f"{vid}-{time:07.2f}"
            cv.imwrite(str(out / "images" / split / f"{key}.jpg"), frame, [cv.IMWRITE_JPEG_QUALITY, 92])
            for kind, box, _ in found:
                number = len(proposals)
                proposals.append({"id": number, "key": key, "split": split, "kind": kind,
                                  "box": [round(float(v), 6) for v in box]})
                tiles.append(crop_tile(frame, box, kind, number))
            kept += 1
        print(f"{vid} ({split}): {kept} frames", flush=True)
    sheets = []
    for start in range(0, len(tiles), PER_SHEET):
        block = tiles[start:start + PER_SHEET]
        block += [np.zeros_like(tiles[0])] * (-len(block) % COLUMNS)
        sheet = np.vstack([np.hstack(block[i:i + COLUMNS]) for i in range(0, len(block), COLUMNS)])
        name = f"sheet-{start // PER_SHEET:04d}.jpg"
        cv.imwrite(str(out / "review" / name), sheet, [cv.IMWRITE_JPEG_QUALITY, 90])
        sheets.append({"sheet": name, "first": start, "last": min(start + PER_SHEET, len(tiles)) - 1})
    (out / "proposals.json").write_text(json.dumps(proposals))
    (out / "review" / "sheets.json").write_text(json.dumps(sheets, indent=1))
    (out / "split.json").write_text(json.dumps({"val": sorted(held_out), "train": sorted(set(videos) - held_out)}, indent=1))
    write_labels(out, proposals, set())
    print(f"{len(proposals)} proposals on {len(sheets)} review sheets; validation videos: {sorted(held_out)}")


def write_labels(out, proposals, rejected):
    by_key = {}
    for image in (out / "images").glob("*/*.jpg"):
        by_key[image.stem] = (image.parent.name, [])
    for row in proposals:
        if row["id"] in rejected:
            continue
        x0, y0, x1, y1 = row["box"]
        by_key[row["key"]][1].append(f"{NAMES.index(row['kind'])} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} "
                                     f"{x1 - x0:.6f} {y1 - y0:.6f}\n")
    for key, (split, lines) in by_key.items():
        (out / "labels" / split / f"{key}.txt").write_text("".join(lines))
    (out / "dataset.yaml").write_text(f"path: {json.dumps(str(out.resolve()))}\ntrain: images/train\n"
                                      f"val: images/val\nnames: {json.dumps(NAMES)}\n")
    kept = len(proposals) - len(rejected)
    print(f"Labels written: {kept} boxes kept, {len(rejected)} rejected")


def apply_review(args):
    out = args.output
    proposals = json.loads((out / "proposals.json").read_text())
    rejected = set()
    for path in sorted((out / "review").glob("decisions-*.json")):
        decision = json.loads(path.read_text())
        rejected.update(int(number) for number in decision.get("reject", []))
        for number, kind in decision.get("relabel", {}).items():
            if kind in NAMES:
                proposals[int(number)]["kind"] = kind
    write_labels(out, proposals, rejected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=ROOT / "data/videos")
    parser.add_argument("--output", type=Path, default=ROOT / "data/ball-dataset")
    parser.add_argument("--every", type=float, default=2.0, help="Seconds between sampled frames")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--val-videos", type=int, default=2)
    parser.add_argument("--empty-share", type=float, default=.12, help="Share of frames without proposals to keep")
    parser.add_argument("--apply-review", action="store_true")
    args = parser.parse_args()
    apply_review(args) if args.apply_review else build(args)


if __name__ == "__main__":
    main()
