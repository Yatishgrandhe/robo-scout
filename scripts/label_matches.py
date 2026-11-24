"""Build per-robot match statistics from public match video with reviewable evidence.

For each video:
  track       follow every robot with models/robot.pt at 10 samples per second
  candidates  scan the full frame rate for launches from any tracked robot
  loose       add short or slow rising paths near robots for review
  render      cut review images: one strip per robot track, one per launch
  apply       combine reviewer answers into per-team shots and Hive entries

Reviewers answer in <out>/<video>/answers-*.json:
  {"tracks": {"<track id>": "<team number>" | null},
   "launches": {"<launch id>": {"launch": true|false, "entered": true|false|null}}}

    python -m scripts.label_matches track candidates render --videos K5YZrcz5puc ...
    python -m scripts.label_matches apply
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.launches import scan_launches
from backend.vision import iou

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 960


def read_frame(cap, time):
    cap.set(cv.CAP_PROP_POS_MSEC, time * 1000)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv.resize(frame, (WIDTH, round(frame.shape[0] * WIDTH / frame.shape[1])), interpolation=cv.INTER_AREA)


def track(video, out, device, step=.1, max_gap=1.5):
    """Link robot detections into tracks by box overlap; a track ends after max_gap seconds unseen."""
    from ultralytics import YOLO
    detector = YOLO(ROOT / "models/robot.pt")
    cap = cv.VideoCapture(str(video))
    fps = cap.get(cv.CAP_PROP_FPS) or 30
    duration = cap.get(cv.CAP_PROP_FRAME_COUNT) / fps
    every = max(1, round(fps * step))
    tracks, active, times, index = {}, {}, [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if index % every:
            index += 1
            continue
        ok, frame = cap.retrieve()
        time = index / fps
        index += 1
        if not ok:
            continue
        frame = cv.resize(frame, (WIDTH, round(frame.shape[0] * WIDTH / frame.shape[1])), interpolation=cv.INTER_AREA)
        h, w = frame.shape[:2]
        result = detector.predict(frame, imgsz=640, conf=.35, device=device, verbose=False)[0]
        boxes = [np.array(b) / (w, h, w, h) for *b, _, _ in result.boxes.data.cpu().numpy()]
        times.append(round(time, 3))
        live = [key for key, (last, _) in active.items() if time - last <= max_gap]
        matched = set()
        if live and boxes:
            cost = np.array([[1 - iou(active[key][1], box) for box in boxes] for key in live])
            for row, col in zip(*linear_sum_assignment(cost)):
                if cost[row, col] < .75:
                    key = live[row]
                    active[key] = (time, boxes[col])
                    tracks[key].append([round(time, 3), boxes[col].round(5).tolist()])
                    matched.add(col)
        for col, box in enumerate(boxes):
            if col not in matched:
                key = str(len(tracks))
                tracks[key] = [[round(time, 3), box.round(5).tolist()]]
                active[key] = (time, box)
    tracks = {key: rows for key, rows in tracks.items() if len(rows) >= 10}
    (out / "tracks.json").write_text(json.dumps({"duration": duration, "times": times, "tracks": tracks}))
    print(f"{video.stem}: {len(tracks)} robot tracks over {duration:.0f}s", flush=True)


def candidates(video, out):
    data = json.loads((out / "tracks.json").read_text())
    by_time = {time: {} for time in data["times"]}
    for key, rows in data["tracks"].items():
        for time, box in rows:
            by_time[time][key] = box
    keys = list(data["tracks"])
    frames = [{"time": time, "robot": None, "robots": {key: robots.get(key) for key in keys}} for time, robots in by_time.items()]
    seeds = [SimpleNamespace(team=key, seed_time=0.) for key in keys]
    config = SimpleNamespace(team=keys[0], seed_time=0., end_time=data["duration"] - .1, field=None, robots=seeds[1:])
    rows = scan_launches(video, frames, config)
    launches = [{"id": index, "track": row["team"], "time": row["time"], "kind": row["kind"],
                 "confidence": row["confidence"], "flight": row.get("flight") or row["samples"]}
                for index, row in enumerate(rows)]
    (out / "launches.json").write_text(json.dumps(launches))
    print(f"{video.stem}: {len(launches)} launch candidates", flush=True)


def loose_candidates(video, out):
    """Add short or slow rising ball paths near any robot that the launch scanner rejected.

    Reviewers decide which are shots, so this favors finding every launch.
    """
    from backend.launches import _associate, _blobs
    data = json.loads((out / "tracks.json").read_text())
    launches = json.loads((out / "launches.json").read_text())
    launches = [row for row in launches if not row.get("loose")]
    boxes = {}
    for key, rows in data["tracks"].items():
        for time, box in rows:
            boxes.setdefault(round(time, 1), []).append((key, np.array(box)))
    cap = cv.VideoCapture(str(video))
    fps = cap.get(cv.CAP_PROP_FPS) or 30
    stale, done, active, previous, index = max(.075, 2.25 / fps), [], [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        time, index = index / fps, index + 1
        frame = cv.resize(frame, (WIDTH, round(frame.shape[0] * WIDTH / frame.shape[1])), interpolation=cv.INTER_AREA)
        h, w = frame.shape[:2]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        alive = []
        for trail in active:
            (alive if time - trail.observations[-1][0] <= stale and time - trail.observations[0][0] <= 1.3 else done).append(trail)
        active = alive
        if previous is not None:
            motion = cv.absdiff(gray, previous)
            if np.count_nonzero(motion > 22) > .28 * motion.size:
                done += active
                active = []
            elif np.count_nonzero(motion > 10) >= 9:
                _associate(active, _blobs(cv.cvtColor(frame, cv.COLOR_BGR2HSV), motion, np.zeros(2), w, h), time, h)
        previous = gray
    done += active
    added = 0
    for trail in done:
        points = [(t, blob.xy / (w, h), blob.diameter) for t, blob in trail.observations if t - trail.observations[0][0] <= .4]
        if len(points) < 3:
            continue
        (t0, start, diameter), (t2, third, _) = points[0], points[2]
        rise = start[1] - min(p[1] for _, p, _ in points)
        if t2 <= t0 or (third[1] - start[1]) / (t2 - t0) > -.10 or rise * h < max(10, diameter):
            continue
        near = boxes.get(round(t0, 1), [])
        scored = []
        for key, box in near:
            bw, bh = box[2] - box[0], box[3] - box[1]
            if box[0] - .6 * bw <= start[0] <= box[2] + .6 * bw and box[1] - .25 * bh <= start[1] <= box[3]:
                scored.append((np.linalg.norm(start - ((box[:2] + box[2:]) / 2)), key))
        if not scored:
            continue
        if any(abs(row["time"] - t0) < .12 and np.linalg.norm(np.array(row["flight"][0][1:]) - start) < .03 for row in launches):
            continue
        flight = [[round(t, 3), round(float(b.xy[0] / w), 5), round(float(b.xy[1] / h), 5)] for t, b in trail.observations]
        launches.append({"id": len(launches), "track": min(scored)[1], "time": round(t0, 3), "kind": trail.kind,
                         "confidence": 0., "flight": flight, "loose": True})
        added += 1
    for index, row in enumerate(sorted(launches, key=lambda row: row["time"])):
        row["id"] = index
    (out / "launches.json").write_text(json.dumps(sorted(launches, key=lambda row: row["time"])))
    print(f"{video.stem}: {added} loose candidates added, {len(launches)} total", flush=True)


def tile(frame, box, size, label=None, points=()):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (np.array(box) * (w, h, w, h)).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    crop = frame[y0:y1, x0:x1].copy()
    for px, py in points:
        cv.circle(crop, (int(px * w - x0), int(py * h - y0)), 3, (0, 255, 255), -1)
    crop = cv.resize(crop, size, interpolation=cv.INTER_CUBIC)
    if label:
        cv.rectangle(crop, (0, 0), (8 + 9 * len(label), 20), (0, 0, 0), -1)
        cv.putText(crop, label, (4, 15), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv.LINE_AA)
    return crop


def render(video, out):
    data = json.loads((out / "tracks.json").read_text())
    launches = json.loads((out / "launches.json").read_text())
    cap = cv.VideoCapture(str(video))
    (out / "tracks").mkdir(exist_ok=True)
    (out / "launches").mkdir(exist_ok=True)
    for key, rows in data["tracks"].items():
        picks = [rows[i] for i in np.linspace(0, len(rows) - 1, 6).round().astype(int)]
        tiles = []
        for time, box in picks:
            frame = read_frame(cap, time)
            if frame is None:
                continue
            x0, y0, x1, y1 = box
            pad = max(x1 - x0, y1 - y0) * .15
            tiles.append(tile(frame, (x0 - pad, y0 - pad, x1 + pad, y1 + pad), (200, 200), f"{time:.0f}s"))
        if tiles:
            cv.imwrite(str(out / "tracks" / f"{key}.jpg"), np.hstack(tiles), [cv.IMWRITE_JPEG_QUALITY, 90])
    for row in launches:
        flight = np.array(row["flight"])
        robot = dict((round(t, 1), b) for t, b in data["tracks"][row["track"]])
        box = robot.get(round(row["time"], 1)) or min(data["tracks"][row["track"]], key=lambda r: abs(r[0] - row["time"]))[1]
        xs = np.concatenate([flight[:, 1], [box[0], box[2]]])
        ys = np.concatenate([flight[:, 2], [box[1], box[3]]])
        span = max(xs.max() - xs.min(), (ys.max() - ys.min()) * .5625, .12)
        cx, top, bottom = (xs.min() + xs.max()) / 2, max(0., ys.min() - .12), min(1., ys.max() + .03)
        region = (cx - span * .7, top, cx + span * .7, bottom)
        tiles = []
        for offset in np.linspace(-.05, max(.6, flight[-1, 0] - row["time"] + .5), 9):
            time = row["time"] + offset
            frame = read_frame(cap, time)
            if frame is None:
                continue
            seen = [(x, y) for t, x, y in flight if t <= time + .01]
            tiles.append(tile(frame, region, (180, 220), f"{offset:+.2f}", seen[-6:]))
        # Second row: where the ball lands. Centered on the highest tracked
        # point, from the top of the flight until well after it should land.
        top_index = int(np.argmin(flight[:, 2]))
        apex_time, apex_x, apex_y = flight[top_index]
        landing = (apex_x - .17, 0., apex_x + .17, min(1., apex_y + .12))
        width = len(tiles) * 180
        entry = []
        for time in np.linspace(apex_time, row["time"] + 1.8, 6):
            frame = read_frame(cap, time)
            if frame is not None:
                entry.append(tile(frame, landing, (width // 6, 260), f"+{time - row['time']:.2f}"))
        frame = read_frame(cap, row["time"])
        if frame is None or not tiles:
            continue
        if entry:
            entry_row = np.hstack(entry)
            tiles_row = np.hstack(tiles)
            entry_row = cv.resize(entry_row, (tiles_row.shape[1], entry_row.shape[0]))
        context = frame.copy()
        h, w = context.shape[:2]
        x0, y0, x1, y1 = (np.array(box) * (w, h, w, h)).astype(int)
        cv.rectangle(context, (x0, y0), (x1, y1), (0, 255, 0), 2)
        rx0, ry0, rx1, ry1 = (np.array(region) * (w, h, w, h)).astype(int)
        cv.rectangle(context, (rx0, ry0), (rx1, ry1), (255, 0, 255), 1)
        context = cv.resize(context, (len(tiles) * 180, round(len(tiles) * 180 * h / w)))
        rows_out = [np.hstack(tiles)] + ([entry_row] if entry else []) + [context]
        cv.imwrite(str(out / "launches" / f"{row['id']:03d}.jpg"), np.vstack(rows_out), [cv.IMWRITE_JPEG_QUALITY, 88])
    print(f"{video.stem}: rendered {len(data['tracks'])} tracks, {len(launches)} launches", flush=True)


BURST = .5


def apply(root):
    """Per-team shots and Hive entries from reviewed answers, for every labeled video."""
    summary = {}
    for out in sorted(path for path in root.iterdir() if (path / "launches.json").is_file()):
        teams, verdicts, shooters = {}, {}, {}
        for path in sorted(out.glob("answers-*.json")):
            answer = json.loads(path.read_text())
            teams.update({str(int(k)): v for k, v in answer.get("tracks", {}).items()})
            verdicts.update({str(int(k)): v for k, v in answer.get("launches", {}).items()})
            shooters.update({str(int(k)): v for k, v in answer.get("shooters", {}).items()})
        stats, rows, confirmed = {}, [], {}
        for row in json.loads((out / "launches.json").read_text()):
            verdict = verdicts.get(str(row["id"]))
            team = shooters.get(str(row["id"])) or teams.get(str(row["track"]))
            team = str(team).strip().lstrip("0") if team else None
            rows.append({**row, "team": team, **(verdict or {})})
            if verdict and verdict.get("launch") and team:
                confirmed.setdefault(team, []).append((row["time"], verdict))
        for team, shots in confirmed.items():
            # Candidates of one robot within BURST seconds are one volley: count its distinct balls once.
            shots.sort(key=lambda item: item[0])
            bursts = [[shots[0]]]
            for item in shots[1:]:
                (bursts[-1].append if item[0] - bursts[-1][-1][0] <= BURST else lambda x: bursts.append([x]))(item)
            entry = stats.setdefault(team, {"shots": 0, "entered": 0, "unsure_entry": 0, "times": []})
            for burst in bursts:
                balls = max(len(burst), 1 + max(int(v.get("others") or 0) for _, v in burst))
                entered = sum(v.get("entered") is True for _, v in burst)
                entry["shots"] += balls
                entry["entered"] += min(entered, balls)
                entry["unsure_entry"] += min(sum(v.get("entered") is None for _, v in burst), balls - min(entered, balls))
                entry["times"].append(round(burst[0][0], 2))
        reviewed = sum(str(row["id"]) in verdicts for row in rows)
        (out / "stats.json").write_text(json.dumps({"teams": stats, "launches": rows}, indent=1))
        summary[out.name] = {"reviewed": reviewed, "candidates": len(rows),
                             "teams": {team: {k: v for k, v in s.items() if k != "times"} for team, s in stats.items()}}
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=["track", "candidates", "loose", "render", "apply"])
    parser.add_argument("--videos", nargs="*", default=[])
    parser.add_argument("--source", type=Path, default=ROOT / "data/videos")
    parser.add_argument("--out", type=Path, default=ROOT / "data/match-labels")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if "apply" in args.stages:
        apply(args.out)
        return
    for name in args.videos:
        video = args.source / f"{name}.mp4"
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        if "track" in args.stages:
            track(video, out, args.device)
        if "candidates" in args.stages:
            candidates(video, out)
        if "loose" in args.stages:
            loose_candidates(video, out)
        if "render" in args.stages:
            render(video, out)


if __name__ == "__main__":
    main()
