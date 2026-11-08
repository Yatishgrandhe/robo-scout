"""Prepare local BIOBUZZ YOLO data from checked robot boxes and ball proposals."""
import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from backend.vision import RobotTrack, color_detections

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 640, 402
TOP, BOTTOM = 20, 296
NAMES = ["pollen", "red_nectar", "blue_nectar", "robot_21229", "robot_19049"]
# Field, scoring structures and playable airspace in the two static camera views.
POLYGONS = {
    "1": [(2, 292), (151, 56), (225, 56), (239, 20), (432, 20),
          (461, 65), (510, 65), (639, 252), (639, 294)],
    "2": [(0, 282), (85, 205), (165, 171), (246, 168), (244, 25),
          (421, 22), (430, 170), (476, 174), (638, 249), (639, 295)],
}


def frame_at(cap, fps, time):
    cap.set(cv.CAP_PROP_POS_FRAMES, max(0, round(time * fps)))
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Could not decode the recording at {time:.2f} seconds")
    return cv.resize(frame, (WIDTH, HEIGHT), interpolation=cv.INTER_AREA)


def label(category, box):
    a, b, c, d = box
    a, c = max(0, a), min(WIDTH, c)
    b, d = max(TOP, b) - TOP, min(BOTTOM, d) - TOP
    if c - a < 3 or d - b < 3:
        return ""
    h = BOTTOM - TOP
    return f"{category} {(a+c)/(2*WIDTH):.6f} {(b+d)/(2*h):.6f} {(c-a)/WIDTH:.6f} {(d-b)/h:.6f}\n"


def write_example(output, split, key, frame, box, robot_class, polygon):
    labels = [label(robot_class, box)]
    for kind, candidate, _ in color_detections(frame):
        a, b, c, d = candidate * (WIDTH, HEIGHT, WIDTH, HEIGHT)
        x, y = float((a+c)/2), float((b+d)/2)
        if cv.pointPolygonTest(polygon, (x, y), False) < 0:
            continue
        # Match broadcasts use bright wheels. Exclude proposals inside the
        # selected chassis to keep wheels out of the ball class.
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            continue
        labels.append(label(NAMES.index(kind), (a, b, c, d)))
    cv.imwrite(str(output / "images" / split / f"{key}.jpg"), frame[TOP:BOTTOM])
    (output / "labels" / split / f"{key}.txt").write_text("".join(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=ROOT / "data/recordings")
    parser.add_argument("--output", type=Path, default=ROOT / "data/biobuzz-recordings")
    args = parser.parse_args()
    output = args.output.resolve()
    labels = json.loads((args.recordings / "labels.json").read_text())
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (output / kind / split).mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    for match, spec in labels.items():
        cap = cv.VideoCapture(str(args.recordings / f"Match-{match}.mp4"))
        if not cap.isOpened():
            raise ValueError(f"Missing decoded match {match}")
        fps = cap.get(cv.CAP_PROP_FPS)
        polygon = np.array(POLYGONS[match], np.int32)
        robot_class = NAMES.index(f"robot_{spec['team']}")
        try:
            for timestamp, points in spec["boxes"].items():
                time = float(timestamp)
                split = "val" if time in spec["validation"] else "train"
                frame = frame_at(cap, fps, time)
                seed = np.array(points) / (WIDTH, HEIGHT, WIDTH, HEIGHT)
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                offsets = [0] if split == "val" else [0, -.10, .10, -.24, .24]
                for index, offset in enumerate(offsets):
                    candidate = frame if offset == 0 else frame_at(cap, fps, time + offset)
                    box = np.array(points)
                    if offset:
                        track = RobotTrack(gray, seed)
                        tracked = track.advance(cv.cvtColor(candidate, cv.COLOR_BGR2GRAY), [])
                        if tracked is None or np.max(np.abs(tracked - seed)) > .045:
                            continue
                        box = tracked * (WIDTH, HEIGHT, WIDTH, HEIGHT)
                    write_example(output, split, f"{match}-{timestamp}-{index}", candidate, box, robot_class, polygon)
                    counts[split] += 1
        finally:
            cap.release()
    yaml = output / "dataset.yaml"
    yaml.write_text(f"path: {json.dumps(str(output))}\ntrain: images/train\nval: images/val\nnames: {json.dumps(NAMES)}\n")
    print(f"{yaml}\nReal recording samples: {counts['train']} train, {counts['val']} validation")


if __name__ == "__main__":
    main()
