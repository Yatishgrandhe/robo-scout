"""Shots from a trained ball detector: per-frame boxes linked into ballistic flights.

The color trail scanner (launches.py) loses balls against clutter and behind the
Hive frame. This pass runs a detector trained on balls in flight
(models/ball-flight.pt, see scripts/build_flight_dataset.py), links detections
into tracklets, and keeps tracklets that a ballistic model explains: x linear and
y quadratic in time, fitted with RANSAC, rising steeply from robot height.
"""
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

BALLS = ("pollen", "red_nectar", "blue_nectar")
MIN_INLIERS = 5
MIN_RISE = .08          # frame heights
MIN_SPEED = .4          # frame heights per second at release
MAX_RESIDUAL = .006     # frame heights
LOWEST_START = .28      # releases start below this fraction of the frame height


def _tracklets(frames, gate=.035):
    """Link [(time, [(kind, x, y, score)])] into tracklets with constant-velocity gating."""
    active, done = [], []
    for time, detections in frames:
        alive = []
        for track in active:
            (alive if time - track[-1][0] <= .1 else done).append(track)
        active = alive
        if detections and active:
            cost = np.full((len(active), len(detections)), 1e4)
            for i, track in enumerate(active):
                t1, kind, x1, y1, _ = track[-1]
                vx = vy = 0.
                if len(track) > 1:
                    t0, _, x0, y0, _ = track[-2]
                    vx, vy = (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)
                dt = time - t1
                for j, (k, x, y, _) in enumerate(detections):
                    distance = np.hypot(x - (x1 + vx * dt), y - (y1 + vy * dt))
                    if k == kind and distance < gate + .45 * dt:
                        cost[i, j] = distance
            used = set()
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e4:
                    active[i].append((time, *detections[j]))
                    used.add(j)
            detections = [d for j, d in enumerate(detections) if j not in used]
        active += [[(time, *d)] for d in detections]
    return done + active


def _ballistic(track, rng):
    """RANSAC fit of x = a + b t and y = c + d t + e t^2 (image y points down, so e > 0)."""
    t = np.array([p[0] for p in track]) - track[0][0]
    x = np.array([p[2] for p in track])
    y = np.array([p[3] for p in track])
    best = None
    for _ in range(60):
        pick = rng.choice(len(t), size=3, replace=False)
        if np.ptp(t[pick]) < .03:
            continue
        cy = np.polyfit(t[pick], y[pick], 2)
        cx = np.polyfit(t[pick], x[pick], 1)
        residual = np.hypot(np.polyval(cy, t) - y, (np.polyval(cx, t) - x) * .5)
        inliers = residual < MAX_RESIDUAL
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < MIN_INLIERS:
        return None
    cy = np.polyfit(t[best], y[best], 2)
    cx = np.polyfit(t[best], x[best], 1)
    if cy[0] <= 0:
        return None
    first = int(np.argmax(best))
    speed = -(2 * cy[0] * t[first] + cy[1])
    apex_t = -cy[1] / (2 * cy[0])
    top = np.polyval(cy, np.clip(apex_t, t[best].min(), t[best].max()))
    rise = y[first] - top
    if speed < MIN_SPEED or rise < MIN_RISE or y[first] < LOWEST_START:
        return None
    return first, best, speed, rise, cx[0]


def scan_flights(video: Path, model, device, start, end, stride=2, progress=None):
    """Return launch candidates (same shape as launches.scan_launches) from detector flights."""
    capture = cv.VideoCapture(str(video))
    fps = capture.get(cv.CAP_PROP_FPS)
    if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
        return []
    frames, batch = [], []

    def flush():
        if not batch:
            return
        results = model.predict([image for _, image in batch], imgsz=960, conf=.12, device=device, verbose=False)
        for (time, image), result in zip(batch, results):
            h, w = image.shape[:2]
            frames.append((time, [(result.names[int(c)], float(x0 + x1) / 2 / w, float(y0 + y1) / 2 / h, float(s))
                                  for x0, y0, x1, y1, s, c in result.boxes.data.cpu().numpy()
                                  if result.names[int(c)] in BALLS]))
        batch.clear()

    try:
        first = max(0, round(start * fps))
        capture.set(cv.CAP_PROP_POS_FRAMES, first)
        index = first
        while True:
            if not capture.grab():
                break
            time = index / fps
            index += 1
            if time > end:
                break
            if (index - first) % stride:
                continue
            ok, image = capture.retrieve()
            if not ok:
                break
            batch.append((time, image))
            if len(batch) == 16:
                flush()
                if progress is not None:
                    progress((time - start) / max(1e-6, end - start))
        flush()
    finally:
        capture.release()
    # A ball resting on the floor or in a Cell is not a flight; drop static detections early.
    rng = np.random.default_rng(408)
    found = []
    for track in _tracklets(frames):
        if len(track) < MIN_INLIERS or np.ptp([p[3] for p in track]) < MIN_RISE:
            continue
        fit = _ballistic(track, rng)
        if fit is None:
            continue
        first, inliers, speed, rise, vx = fit
        points = [p for p, keep in zip(track, inliers) if keep]
        release = track[first]
        found.append({
            "time": round(float(release[0]), 3),
            "kind": release[1],
            "confidence": round(float(min(.95, .5 + rise)), 3),
            "source_error": 9.0,
            "free": True,
            "detector": True,
            "release_norm": [release[2], release[3]],
            "velocity_norm": [float(vx), -float(speed)],
            "samples": [[round(p[0], 3), round(p[2], 5), round(p[3], 5)] for p in points
                        if p[3] <= release[3] + .01],
            "flight": [[round(p[0], 3), round(p[2], 5), round(p[3], 5)] for p in points],
        })
    return found


def merge(primary, extra):
    """Add detector flights that no color-scanner launch already explains."""
    merged = list(primary)
    for flight in extra:
        start = np.array(flight["flight"][0][1:])
        if any(row["kind"] == flight["kind"] and abs(row["time"] - flight["time"]) < .2
               and np.linalg.norm(np.array(row["samples"][0][1:]) - start) < .05 for row in primary):
            continue
        merged.append(flight)
    return sorted(merged, key=lambda row: row["time"])
