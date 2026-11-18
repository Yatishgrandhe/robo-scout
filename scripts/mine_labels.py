"""Use a trained detector to find label errors, then rebuild labels after review.

Draft labels come from color proposals, so balls the color rule never found
are unlabeled and teach the detector that real balls are background.

Default mode (after a reviewed dataset exists): every confident detection
that matches no label is cut onto numbered review sheets, and confirmed balls
are added on top of the reviewed color labels.

--from-proposals mode (for a fresh, unreviewed draft): a color proposal that
the model confirms with the same class at --auto-conf or higher is accepted
without review on the training split. Every other color proposal, every model
detection with no proposal, and everything in --review-all splits goes on
review sheets, so held-out labels never depend on the model's judgment.

    python -m scripts.mine_labels data/ball-dataset models/ball-v1.pt [--from-proposals]
    # reviewers write data/ball-dataset/mined/verdict-*.json:
    #   {"ok": [ids], "relabel": {"id": "red_nectar"}}
    python -m scripts.mine_labels data/ball-dataset --apply
"""
import argparse
import json
from argparse import Namespace
from collections import Counter
from pathlib import Path

import cv2 as cv
import numpy as np

from scripts.build_ball_dataset import COLUMNS, NAMES, PER_SHEET, apply_review, crop_tile


def load_boxes(path):
    rows = []
    for line in path.read_text().splitlines() if path.is_file() else []:
        cls, x, y, w, h = map(float, line.split())
        rows.append((NAMES[int(cls)], np.array((x - w / 2, y - h / 2, x + w / 2, y + h / 2))))
    return rows


def near(box, other, threshold):
    size = max(other[2] - other[0], other[3] - other[1], 1e-6)
    return np.linalg.norm((box[:2] + box[2:]) / 2 - (other[:2] + other[2:]) / 2) / size <= threshold


def unlabeled(predicted, truth, threshold):
    """Detections whose center is not within `threshold` diameters of any label, of any class."""
    rows = []
    for kind, box, score in predicted:
        if any(near(box, truth_box, threshold) for _, truth_box in truth):
            continue
        if all(np.linalg.norm(box - other) > 1e-3 for _, other, _ in rows):
            rows.append((kind, box, score))
    return rows


def triage(proposals, predicted, threshold, auto_conf, review_all, model_conf=0., drop_unseen=False):
    """Split color proposals and model detections into auto-accepted, to-review and dropped rows.

    With drop_unseen (never on review_all splits), a color proposal with no model
    detection nearby, at any predicted score, is dropped instead of reviewed.
    """
    auto, review, used, dropped = [], [], set(), 0
    for kind, box in proposals:
        best = None
        for index, (model_kind, model_box, score) in enumerate(predicted):
            if near(model_box, box, threshold) and (best is None or score > predicted[best][2]):
                best = index
        if best is not None:
            used.add(best)
        confirmed = best is not None and predicted[best][0] == kind and predicted[best][2] >= auto_conf
        if confirmed and not review_all:
            auto.append(("color", kind, box, predicted[best][2]))
        elif best is None and drop_unseen and not review_all:
            dropped += 1
        else:
            review.append(("color", kind, box, None if best is None else predicted[best][2]))
    for index, (kind, box, score) in enumerate(predicted):
        if index not in used and score >= model_conf and not any(near(box, b, threshold) for _, b in proposals):
            review.append(("model", kind, box, score))
    return auto, review, dropped


def write_sheets(out, tiles):
    sheets = []
    for start in range(0, len(tiles), PER_SHEET):
        block = tiles[start:start + PER_SHEET]
        block += [np.zeros_like(tiles[0])] * (-len(block) % COLUMNS)
        sheet = np.vstack([np.hstack(block[i:i + COLUMNS]) for i in range(0, len(block), COLUMNS)])
        name = f"sheet-{start // PER_SHEET:04d}.jpg"
        cv.imwrite(str(out / name), sheet, [cv.IMWRITE_JPEG_QUALITY, 90])
        sheets.append({"sheet": name, "first": start, "last": min(start + PER_SHEET, len(tiles)) - 1})
    (out / "sheets.json").write_text(json.dumps(sheets, indent=1))
    return sheets


def mine(args):
    from ultralytics import YOLO
    model = YOLO(args.weights)
    out = args.dataset / "mined"
    out.mkdir(exist_ok=True)
    for old in list(out.glob("sheet-*.jpg")) + [out / "auto.json"]:
        old.unlink(missing_ok=True)
    by_key = {}
    if args.from_proposals:
        for row in json.loads((args.dataset / "proposals.json").read_text()):
            by_key.setdefault(row["key"], []).append((row["kind"], np.array(row["box"])))
    candidates, auto_rows, tiles, counts = [], [], [], Counter()
    for split in args.splits:
        images = sorted((args.dataset / "images" / split).glob("*.jpg"))
        for index, image_path in enumerate(images):
            frame = cv.imread(str(image_path))
            h, w = frame.shape[:2]
            floor = min(args.conf, args.drop_below) if args.from_proposals and args.drop_below else args.conf
            result = model.predict(frame, imgsz=args.size, conf=floor, device=args.device, verbose=False)[0]
            predicted = [(result.names[int(c)], np.array(b) / (w, h, w, h), float(s))
                         for *b, s, c in result.boxes.data.cpu().numpy() if result.names[int(c)] in NAMES]
            if args.from_proposals:
                auto, review, dropped = triage(by_key.get(image_path.stem, []), predicted, args.threshold,
                                               args.auto_conf, split in args.review_all, args.conf,
                                               drop_unseen=bool(args.drop_below))
                counts[f"{split}/dropped"] += dropped
            else:
                truth = load_boxes(args.dataset / "labels" / split / f"{image_path.stem}.txt")
                auto, review = [], [("model", k, b, s) for k, b, s in unlabeled(predicted, truth, args.threshold)]
            for source, kind, box, score in auto:
                auto_rows.append({"split": split, "key": image_path.stem, "kind": kind,
                                  "box": [round(float(v), 6) for v in box]})
            for source, kind, box, score in review:
                tiles.append(crop_tile(frame, box, kind, len(candidates)))
                candidates.append({"id": len(candidates), "split": split, "key": image_path.stem, "kind": kind,
                                   "box": [round(float(v), 6) for v in box], "source": source,
                                   "score": None if score is None else round(score, 3)})
                counts[f"{split}/{source}"] += 1
            counts[f"{split}/auto"] += len(auto)
            if index % 200 == 0:
                print(f"{split}: {index}/{len(images)} images, {len(candidates)} to review", flush=True)
    sheets = write_sheets(out, tiles) if tiles else []
    (out / "candidates.json").write_text(json.dumps(candidates))
    if args.from_proposals:
        (out / "auto.json").write_text(json.dumps(auto_rows))
    print(dict(counts))
    print(f"{len(candidates)} detections to review on {len(sheets)} sheets, {len(auto_rows)} auto-accepted, in {out}")


def apply(args):
    """Rebuild labels: reviewed color labels (or auto-accepted rows), plus every confirmed candidate."""
    out = args.dataset / "mined"
    auto_path = out / "auto.json"
    if auto_path.is_file():
        for image in (args.dataset / "images").glob("*/*.jpg"):
            (args.dataset / "labels" / image.parent.name / f"{image.stem}.txt").write_text("")
        rows = json.loads(auto_path.read_text())
    else:
        apply_review(Namespace(output=args.dataset))
        rows = []
    candidates = json.loads((out / "candidates.json").read_text())
    confirmed, relabel = set(), {}
    for path in sorted(out.glob("verdict-*.json")):
        verdict = json.loads(path.read_text())
        confirmed.update(int(number) for number in verdict.get("ok", []))
        for number, kind in verdict.get("relabel", {}).items():
            if kind in NAMES:
                relabel[int(number)] = kind
                confirmed.add(int(number))
    reviewed = [{**row, "kind": relabel.get(row["id"], row["kind"])} for row in candidates if row["id"] in confirmed]
    for row in rows + reviewed:
        x0, y0, x1, y1 = row["box"]
        label = args.dataset / "labels" / row["split"] / f"{row['key']}.txt"
        with label.open("a") as handle:
            handle.write(f"{NAMES.index(row['kind'])} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} "
                         f"{x1 - x0:.6f} {y1 - y0:.6f}\n")
    (args.dataset / "dataset.yaml").write_text(f"path: {json.dumps(str(args.dataset.resolve()))}\ntrain: images/train\n"
                                               f"val: images/val\nnames: {json.dumps(NAMES)}\n")
    print(f"Labels: {len(rows)} auto-accepted, {len(reviewed)} confirmed of {len(candidates)} reviewed "
          f"({len(relabel)} relabeled)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("weights", type=Path, nargs="?")
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=.6, help="Center distance in ball diameters")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--from-proposals", action="store_true", help="Triage unreviewed color proposals")
    parser.add_argument("--auto-conf", type=float, default=.5, help="Model score that confirms a color proposal")
    parser.add_argument("--review-all", nargs="*", default=["val"], help="Splits where nothing is auto-accepted")
    parser.add_argument("--drop-below", type=float, default=0.,
                        help="Outside --review-all splits, drop color proposals the model scores under this")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        apply(args)
    elif args.weights:
        mine(args)
    else:
        parser.error("pass weights to mine, or --apply after review")


if __name__ == "__main__":
    main()
