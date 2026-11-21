"""Build a YOLO robot dataset from match videos with reviewable motion proposals.

The camera is fixed for most of a match, so a per-video median frame shows the
empty field. Large regions that differ from it become robot proposals. Each
sampled frame is written to review/frame-NNNN.jpg with numbered boxes, and a
reviewer writes review/verdict-*.json:

    {"ok": [box ids], "drop": ["frame key", ...]}

A dropped frame has a robot without a correct box. An unlabeled robot would
teach the detector that robots are background, so a dropped frame contributes
only native-scale crops around its confirmed boxes.
Whole videos are held out for validation, matching the ball dataset.

    python -m scripts.build_robot_dataset
    python -m scripts.build_robot_dataset --apply-review
"""
import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAL_VIDEOS = ["2EDHhlj1vQk", "Y6uT5ziHRJY", "kwYU2RU7V14"]


def grab(cap, fps, time, width):
    cap.set(cv.CAP_PROP_POS_FRAMES, round(time * fps))
    ok, frame = cap.read()
    if not ok:
        return None
    return cv.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1] / 2) * 2), interpolation=cv.INTER_AREA)


def proposals(frame, background, threshold=38):
    """Robot-sized regions that differ from the empty-field background, as normalized boxes."""
    h, w = frame.shape[:2]
    diff = cv.absdiff(cv.GaussianBlur(frame, (5, 5), 0), cv.GaussianBlur(background, (5, 5), 0)).max(axis=2)
    mask = (diff > threshold).astype(np.uint8) * 255
    if np.count_nonzero(mask) > .35 * mask.size:
        return []  # A cut or reframe, not robots on a known field.
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)))
    _, _, stats, _ = cv.connectedComponentsWithStats(mask)
    boxes = []
    for x, y, bw, bh, area in stats[1:]:
        if .0025 * w * h <= bw * bh <= .06 * w * h and .45 < bw / bh < 2.8 and area >= .35 * bw * bh:
            boxes.append([round(x / w, 5), round(y / h, 5), round((x + bw) / w, 5), round((y + bh) / h, 5)])
    return boxes


def review_image(frame, rows):
    image = frame.copy()
    h, w = image.shape[:2]
    for row in rows:
        x0, y0, x1, y1 = (np.array(row["box"]) * (w, h, w, h)).round().astype(int)
        cv.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = str(row["id"])
        cv.rectangle(image, (x0, max(0, y0 - 18)), (x0 + 11 * len(label) + 6, y0), (0, 0, 0), -1)
        cv.putText(image, label, (x0 + 3, y0 - 4), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv.LINE_AA)
    return image


def build(args):
    manifest = json.loads((args.videos / "manifest.json").read_text())
    out = args.output
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(exist_ok=True)
    rows, frames = [], []
    for video_id, entry in sorted(manifest.items()):
        if not entry.get("file") or not (args.videos / entry["file"]).is_file():
            continue
        split = "val" if video_id in VAL_VIDEOS else "train"
        cap = cv.VideoCapture(str(args.videos / entry["file"]))
        fps = cap.get(cv.CAP_PROP_FPS) or 30
        duration = cap.get(cv.CAP_PROP_FRAME_COUNT) / fps
        stack = [f for f in (grab(cap, fps, t, args.width) for t in np.linspace(duration * .05, duration * .9, 45)) if f is not None]
        if len(stack) < 10:
            continue
        background = np.median(np.stack(stack), axis=0).astype(np.uint8)
        time = duration * .05
        while time < duration * .9:
            frame = grab(cap, fps, time, args.width)
            boxes = proposals(frame, background) if frame is not None else []
            if 1 <= len(boxes) <= args.max_boxes:
                key = f"{video_id}-{time:07.2f}"
                cv.imwrite(str(out / "images" / split / f"{key}.jpg"), frame, [cv.IMWRITE_JPEG_QUALITY, 92])
                mine = [{"id": len(rows) + i, "key": key, "split": split, "box": box} for i, box in enumerate(boxes)]
                rows += mine
                frames.append({"key": key, "split": split, "review": f"frame-{len(frames):04d}.jpg",
                               "boxes": [row["id"] for row in mine]})
                cv.imwrite(str(out / "review" / frames[-1]["review"]), review_image(frame, mine), [cv.IMWRITE_JPEG_QUALITY, 88])
            time += args.every
        cap.release()
        print(f"{video_id}: {sum(f['key'].startswith(video_id) for f in frames)} frames", flush=True)
    (out / "proposals.json").write_text(json.dumps(rows))
    (out / "frames.json").write_text(json.dumps(frames, indent=1))
    print(f"{len(frames)} frames, {len(rows)} proposals in {out / 'review'}")


def crop_window(box, w, h, size, shift):
    """A size x size window around a box, shifted so the robot is not always centered."""
    cx, cy = (box[0] + box[2]) / 2 * w + shift[0], (box[1] + box[3]) / 2 * h + shift[1]
    x0 = int(np.clip(cx - size / 2, 0, max(0, w - size)))
    y0 = int(np.clip(cy - size / 2, 0, max(0, h - size)))
    return x0, y0, min(w, x0 + size), min(h, y0 + size)


def apply_review(args):
    """Write a YOLO dataset: full frames whose robots are all boxed, plus crops around every confirmed box.

    Crops at native scale rarely contain an unboxed robot, so boxes confirmed on
    otherwise incomplete frames still train the detector. A confirmed box cut by
    the crop edge is clipped; one mostly outside is grayed out so it is neither
    a label nor background.
    """
    out = args.output
    rows = json.loads((out / "proposals.json").read_text())
    frames = json.loads((out / "frames.json").read_text())
    ok, drop = set(), set()
    for path in sorted((out / "review").glob("verdict-*.json")):
        verdict = json.loads(path.read_text())
        ok.update(int(number) for number in verdict.get("ok", []))
        drop.update(verdict.get("drop", []))
    dest = out / "yolo"
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            folder = dest / kind / split
            folder.mkdir(parents=True, exist_ok=True)
            for old in folder.iterdir():
                old.unlink()
    rng = np.random.default_rng(0)
    counts = {"frames": 0, "crops": 0, "boxes": 0}

    def write(split, name, image, boxes):
        cv.imwrite(str(dest / "images" / split / f"{name}.jpg"), image, [cv.IMWRITE_JPEG_QUALITY, 92])
        (dest / "labels" / split / f"{name}.txt").write_text("".join(
            f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}\n" for x0, y0, x1, y1 in boxes))

    for frame in frames:
        boxes = [np.array(row["box"]) for row in rows if row["key"] == frame["key"] and row["id"] in ok]
        if not boxes:
            continue
        image = cv.imread(str(out / "images" / frame["split"] / f"{frame['key']}.jpg"))
        if image is None:
            continue
        h, w = image.shape[:2]
        counts["boxes"] += len(boxes)
        if frame["key"] not in drop:
            write(frame["split"], frame["key"], image, boxes)
            counts["frames"] += 1
            continue
        for index, box in enumerate(boxes):
            x0, y0, x1, y1 = crop_window(box, w, h, args.crop, rng.uniform(-.25, .25, 2) * args.crop)
            crop = image[y0:y1, x0:x1].copy()
            cw, ch = x1 - x0, y1 - y0
            labels = []
            for other in boxes:
                px = other * (w, h, w, h) - (x0, y0, x0, y0)
                clipped = np.clip(px, 0, (cw, ch, cw, ch))
                area = max(1e-6, (px[2] - px[0]) * (px[3] - px[1]))
                inside = max(0, clipped[2] - clipped[0]) * max(0, clipped[3] - clipped[1]) / area
                if inside >= .6:
                    labels.append(clipped / (cw, ch, cw, ch))
                elif inside > 0:
                    a, b, c, d = clipped.round().astype(int)
                    crop[b:d, a:c] = 114
            write(frame["split"], f"{frame['key']}-c{index}", crop, labels)
            counts["crops"] += 1
    (dest / "dataset.yaml").write_text(f"path: {json.dumps(str(dest.resolve()))}\ntrain: images/train\n"
                                       f"val: images/val\nnames: [\"robot\"]\n")
    print(f"{counts['boxes']} confirmed boxes: {counts['frames']} complete frames, {counts['crops']} crops in {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, default=ROOT / "data/videos")
    parser.add_argument("--output", type=Path, default=ROOT / "data/robot-dataset")
    parser.add_argument("--every", type=float, default=7.0, help="Seconds between sampled frames")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--max-boxes", type=int, default=10)
    parser.add_argument("--crop", type=int, default=384, help="Crop size in pixels around confirmed boxes")
    parser.add_argument("--apply-review", action="store_true")
    args = parser.parse_args()
    apply_review(args) if args.apply_review else build(args)


if __name__ == "__main__":
    main()
