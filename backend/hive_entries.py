"""Reviewable Hive entries from a descending ball path and its launch."""
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Arrival:
    time: float
    path: list[tuple[float, float, float]]


def arrival(path, box, lip_fraction=.54):
    """Require a descent into the opening followed by visible deceleration."""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    lip = y0 + lip_fraction * height
    for i in range(1, len(path) - 1):
        before, crossing = path[i - 1:i + 1]
        time, x, y = crossing
        if not before[2] < lip <= y or time - before[0] > .17:
            continue
        if not x0 + .06 * width < x < x1 - .06 * width or min(p[2] for p in path[:i]) > lip - .16 * height:
            continue
        after = [point for point in path[i:] if point[0] - time <= .60]
        if any(px < x0 or px > x1 or py > y1 + .11 * height for _, px, py in after):
            continue
        settled = [point for point in after if lip <= point[2] <= y1]
        if not settled or settled[-1][0] - settled[0][0] < .18:
            continue
        recent = [point for point in settled if settled[-1][0] - point[0] <= .24]
        if (len(recent) >= 3 and recent[-1][0] - recent[0][0] >= .13
                and np.ptp(np.asarray(recent)[:, 2]) <= .20 * height):
            return Arrival(time, path)
    return None


def mouth_crop(box, width, height):
    x0, y0, x1, y1 = box
    dx, dy = x1 - x0, y1 - y0
    return (max(0, round((x0 - .32 * dx) * width)), max(0, round((y0 - .72 * dy) * height)),
            min(width, round((x1 + .32 * dx) * width)), min(height, round((y1 + .56 * dy) * height)))


def pollen_centers(image, detections):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    yellow = cv.inRange(hsv, (16, 94, 118), (42, 255, 255))
    contours, _ = cv.findContours(yellow, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)
        if 5 <= w <= 31 and 5 <= h <= 36 and .40 <= w / h <= 2.0 and .48 * w * h <= area <= 550:
            centers.append((x + w / 2, y + h / 2))
    # Color preserves small, low-confidence detections and the trained
    # detector separates touching pieces when they form one color contour.
    for x0, y0, x1, y1, confidence, _ in detections:
        left, top = max(0, int(x0)), max(0, int(y0))
        right, bottom = min(image.shape[1], round(x1)), min(image.shape[0], round(y1))
        crop = yellow[top:bottom, left:right]
        if not crop.size or cv.countNonZero(crop) < crop.size * (.28 if confidence < .20 else .16):
            continue
        x, y = float((x0 + x1) / 2), float((y0 + y1) / 2)
        if all((x - px) ** 2 + (y - py) ** 2 > 9 ** 2 for px, py in centers):
            centers.append((x, y))
    return centers


def associate(active, points, time, box):
    if not points:
        return
    size = np.asarray(box[2:]) - box[:2]
    cost = np.full((len(active), len(points)), 1e4)
    for index, track in enumerate(active):
        last_time, *last = track[-1]
        dt = time - last_time
        predicted = np.asarray(last)
        if len(track) > 1:
            earlier_time, *earlier = track[-2]
            predicted = predicted + (predicted - earlier) * min(1.6, dt / (last_time - earlier_time))
        distances = np.linalg.norm((np.asarray(points) - predicted) / size, axis=1)
        limit = .18 + 4 * dt
        cost[index, distances <= limit] = distances[distances <= limit]
    matched = set()
    for index, point in zip(*linear_sum_assignment(cost)):
        if cost[index, point] < 1e4:
            active[index].append((time, *points[point]))
            matched.add(point)
    active.extend([(time, *point)] for i, point in enumerate(points) if i not in matched)


def scan_mouths(video: Path, cells, weights: Path | None, start, end, device, progress=None):
    """Detect Pollen in high-detail Cell crops, including between known launches."""
    model = None
    pollen = []
    if weights is not None:
        from ultralytics import YOLO

        model = YOLO(weights)
        pollen = [key for key, name in model.names.items() if name == "pollen"]
        if not pollen:
            raise ValueError("The Hive checkpoint needs a pollen class.")
    cap = cv.VideoCapture(str(video))
    try:
        width, height, fps = (cap.get(prop) for prop in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))
        if not cap.isOpened() or min(width, height, fps) <= 0:
            return {}
        crops = {team: mouth_crop(box, width, height) for team, box in cells.items()}
        crops = {team: crop for team, crop in crops.items() if min(crop[2] - crop[0], crop[3] - crop[1]) >= 8}
        observations = {team: [] for team in crops}
        if not crops:
            return observations
        first, last = round(start * fps), round(end * fps)
        cap.set(cv.CAP_PROP_POS_FRAMES, first)
        stride = max(1, round(fps / 15))
        batch = []

        def detect():
            images = [item[2] for item in batch]
            if model is None:
                detections = [()] * len(images)
            else:
                predictions = model.predict(images, imgsz=320, conf=.075, classes=pollen, max_det=25, device=device, verbose=False)
                detections = [prediction.boxes.data.cpu().numpy() for prediction in predictions]
            for (team, time, image), boxes in zip(batch, detections):
                left, top, right, bottom = crops[team]
                centers = pollen_centers(image, boxes)
                points = [(float((left + x / 256 * (right - left)) / width),
                           float((top + y / 240 * (bottom - top)) / height)) for x, y in centers]
                observations[team].append((time, points))
            batch.clear()

        for index in range(first, last + 1):
            if not cap.grab():
                break
            if (index - first) % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            for team, (x0, y0, x1, y1) in crops.items():
                crop = cv.resize(frame[y0:y1, x0:x1], (256, 240), interpolation=cv.INTER_AREA)
                batch.append((team, index / fps, crop))
            if len(batch) >= 16:
                detect()
                if progress:
                    progress((index - first) / max(1, last - first))
        if batch:
            detect()
        return observations
    finally:
        cap.release()


def find_arrivals(observations, box):
    active, finished = [], []
    for time, points in observations:
        still_active = []
        for track in active:
            if time - track[-1][0] <= .18 and time - track[0][0] <= 1.8:
                still_active.append(track)
            else:
                finished.append(track)
        active = still_active
        associate(active, points, time, box)
    return [found for track in finished + active if (found := arrival(track, box) or arrival(track, box, .40)) is not None]


def overlap_quality(mouth_path, flight):
    a, b = np.asarray(mouth_path), np.asarray(flight)
    gap = abs(a[:, None, 0] - b[None, :, 0])
    distance = np.linalg.norm(a[:, None, 1:] - b[None, :, 1:], axis=2)
    close = (gap <= .063) & (distance <= .019)
    return min(np.count_nonzero(np.any(close, axis=0)), np.count_nonzero(np.any(close, axis=1)))


def projected_quality(launch, hit, box):
    """Bridge a short gap when the Cell frame hides the middle of a flight."""
    ascent = launch["samples"]
    if len(ascent) < 3:
        return 0.
    start, last = ascent[0], ascent[-1]
    gap = hit.time - last[0]
    if not .03 <= gap <= .84 or last[2] > box[3] + .19:
        return 0.
    x = next(point[1] for point in hit.path if point[0] == hit.time)
    velocity = (last[1] - start[1]) / max(.03, last[0] - start[0])
    projected = last[1] + velocity * min(gap, .27)
    error = abs(projected - x)
    return 1 - error / .038 if error < .038 else 0.


def verified_candidates(launches, cells, alliances, observations=None):
    """Return one review suggestion per uniquely matched flight."""
    direct = {}
    for i, launch in enumerate(launches):
        alliance = alliances.get(launch.get("team"))
        if alliance in cells:
            found = arrival(launch["flight"], cells[alliance])
            if found is not None:
                direct[i] = found
    matched = set(direct)
    matches = {}
    for alliance, rows in (observations or {}).items():
        for found in sorted(find_arrivals(rows, cells[alliance]), key=lambda hit: hit.time):
            exact, projected = [], []
            for i, launch in enumerate(launches):
                if (i in matched or launch["kind"] != "pollen" or alliances.get(launch.get("team")) != alliance
                        or not .18 <= found.time - launch["time"] <= 1.55):
                    continue
                score = overlap_quality(found.path, launch["flight"])
                if score >= 2:
                    exact.append((score, i))
                elif (confidence := projected_quality(launch, found, cells[alliance])) > .25:
                    projected.append((confidence, i))
            hits = sorted(exact or projected, reverse=True)
            teams = {launches[index].get("team") for _, index in hits}
            if hits and (len(teams) == 1 or exact and hits[0][0] > hits[1][0] + 1):
                _, index = hits[0]
                matches[index] = found
                matched.add(index)
    return {**direct, **matches}
