"""Train models/identity.pt, a classifier with one class per team, from reviewed robot tracks.

Reads data/match-labels/<video>/tracks.json and answers-tracks.json (made by
scripts.label_matches and reviewers). Every track a reviewer named becomes
crops of that team, sampled every --every seconds. Videos passed to --val are
held out for validation. Teams seen in fewer than --min-tracks tracks are skipped.

    python -m scripts.train_teams --val Y6uT5ziHRJY 2EDHhlj1vQk kwYU2RU7V14
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2 as cv
import numpy as np

from backend.vision import robot_crop

ROOT = Path(__file__).resolve().parents[1]


def build(labels, videos, val, out, every, min_tracks):
    named = {}
    for folder in sorted(path for path in labels.iterdir() if (path / "tracks.json").is_file()):
        answers = {}
        for path in folder.glob("answers-*.json"):
            answers.update(json.loads(path.read_text()).get("tracks", {}))
        for key, team in answers.items():
            if team:
                named.setdefault(folder.name, {})[key] = str(team)
    tracks_per_team = Counter(team for teams in named.values() for team in teams.values())
    keep = {team for team, count in tracks_per_team.items() if count >= min_tracks}
    shutil.rmtree(out, ignore_errors=True)
    counts = Counter()
    for video, teams in named.items():
        data = json.loads((labels / video / "tracks.json").read_text())
        cap = cv.VideoCapture(str(videos / f"{video}.mp4"))
        split = "val" if video in val else "train"
        for key, team in teams.items():
            if team not in keep or key not in data["tracks"]:
                continue
            target = out / split / team
            target.mkdir(parents=True, exist_ok=True)
            last = -1e9
            for time, box in data["tracks"][key]:
                if time - last < every:
                    continue
                last = time
                cap.set(cv.CAP_PROP_POS_MSEC, time * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])), interpolation=cv.INTER_AREA)
                cv.imwrite(str(target / f"{video}-{key}-{time:07.2f}.jpg"), robot_crop(frame, np.array(box)))
                counts[(split, team)] += 1
    # Classification needs the same class folders in both splits.
    for team in keep:
        for split in ("train", "val"):
            (out / split / team).mkdir(parents=True, exist_ok=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=ROOT / "data/match-labels")
    parser.add_argument("--videos", type=Path, default=ROOT / "data/videos")
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--every", type=float, default=.5)
    parser.add_argument("--min-tracks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--weights", default="yolo26s-cls.pt")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    out = ROOT / "data/identity-teams"
    counts = build(args.labels, args.videos, set(args.val), out, args.every, args.min_tracks)
    for team in sorted({team for _, team in counts}):
        print(f"{team}: train {counts[('train', team)]}, val {counts[('val', team)]}")
    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.train(data=str(out), imgsz=args.size, epochs=args.epochs, batch=64, device=args.device, workers=0,
                project=str(ROOT / "runs"), name="identity-teams", exist_ok=True, fliplr=.5, hsv_h=.01)
    shutil.copy(ROOT / "runs/identity-teams/weights/best.pt", ROOT / "models/identity.pt")
    print("Saved models/identity.pt")


if __name__ == "__main__":
    main()
