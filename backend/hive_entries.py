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


def settled_entry(path, box):
    """A ball that comes into the Cell from outside, from any side, and then rests inside it.

    Low side views show a ball rising into the opening and settling rather than
    dropping past the lip. Resting (little motion for .18 s) separates a ball in
    the Cell from one flying past in front of it.
    """
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0

    def inside(point):
        return x0 + .04 * width < point[1] < x1 - .04 * width and y0 <= point[2] <= y1

    for i in range(1, len(path)):
        before, crossing = path[i - 1], path[i]
        if inside(before) or not inside(crossing) or crossing[0] - before[0] > .17:
            continue
        stay = []
        for point in path[i:]:
            if not inside(point):
                break
            stay.append(point)
        # A piece can roll on inside the Cell, so any still window after entry counts.
        for start in stay:
            rest = np.asarray([point for point in stay if 0 <= point[0] - start[0] <= .22])
            if (len(rest) >= 3 and rest[-1, 0] - rest[0, 0] >= .13
                    and np.ptp(rest[:, 2]) <= .15 * height and np.ptp(rest[:, 1]) <= .15 * width):
                return Arrival(crossing[0], path)
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


def scan_mouths(video: Path, cells, weights: Path | None, start, end, device, progress=None, rate=30):
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
        stride = max(1, round(fps / rate))
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


def resting_entries(observations, box, rest=.4, near=.01, empty=1.5):
    """New balls that come to rest in the Cell, one per resting position until the Cell empties.

    A ball often drops onto the Cell floor in a frame or two, so no crossing is
    tracked. Each distinct position where a ball stays `rest` seconds counts once;
    when the Cell shows no ball for `empty` seconds (a TIP) the positions reset.
    """
    x0, y0, x1, y1 = box
    candidates, resting, found, last_seen = [], [], [], None
    for time, points in observations:
        points = [(x, y) for x, y in points if x0 <= x <= x1 and y0 <= y <= y1]
        if points:
            last_seen = time
        if resting and last_seen is not None and time - last_seen > empty:
            resting = []
        for x, y in points:
            match = next((c for c in candidates if abs(c[0] - x) < near and abs(c[1] - y) < near
                          and time - c[3] < .2), None)
            if match is None:
                candidates.append([x, y, time, time, [(time, x, y)]])
            else:
                match[0], match[1], match[3] = x, y, time
                match[4].append((time, x, y))
        for c in candidates:
            if c[3] - c[2] >= rest and not any(abs(r[0] - c[0]) < near and abs(r[1] - c[1]) < near for r in resting):
                resting.append((c[0], c[1]))
                found.append(Arrival(c[2], c[4]))
        candidates = [c for c in candidates if time - c[3] < .2]
    return found


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
    tracked = [found for track in finished + active
               if (found := arrival(track, box) or arrival(track, box, .40) or settled_entry(track, box)) is not None]
    width = box[2] - box[0]
    # A resting ball already explained by a tracked entry (same time and place) is not a second entry.
    extra = [hit for hit in resting_entries(observations, box)
             if not any(abs(hit.time - other.time) < .8 and abs(hit.path[-1][1] - other.path[-1][1]) < .15 * width
                        for other in tracked)]
    return tracked + extra


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
    """Return {launch index: arrival} for linked entries and {("unlinked", n): arrival} for the rest.

    Cell keys are an alliance or "alliance:2" for its second Cell position.
    """
    side = lambda key: key.split(":")[0]
    direct = {}
    for i, launch in enumerate(launches):
        alliance = alliances.get(launch.get("team"))
        for key, box in cells.items():
            if side(key) == alliance and i not in direct and (found := arrival(launch["flight"], box)) is not None:
                direct[i] = found
    matched = set(direct)
    matches, unlinked = {}, {}
    seen = [hit.time for hit in direct.values()]
    for key, rows in (observations or {}).items():
        alliance, box = side(key), cells[key]
        for found in sorted(find_arrivals(rows, box), key=lambda hit: hit.time):
            if any(abs(found.time - other) < .12 for other in seen):
                continue
            seen.append(found.time)
            window = [i for i, launch in enumerate(launches)
                      if alliances.get(launch.get("team")) == alliance and .18 <= found.time - launch["time"] <= 1.55]
            exact, projected = [], []
            for i in window:
                if i in matched:
                    continue
                score = overlap_quality(found.path, launches[i]["flight"])
                if score >= 2:
                    exact.append((score, i))
                elif (confidence := projected_quality(launches[i], found, box)) > .25:
                    projected.append((confidence, i))
            hits = sorted(exact or projected, reverse=True)
            teams = {launches[index].get("team") for _, index in hits}
            if len(teams) > 1 and not (exact and hits[0][0] > hits[1][0] + 1):
                # Two robots fit equally: count the entry but credit nobody.
                unlinked[("unlinked", len(unlinked))] = found
                continue
            if not hits:
                # No path fits: credit the latest unmatched launch from that alliance, if any.
                hits = [(0, i) for i in sorted((i for i in window if i not in matched),
                                               key=lambda i: launches[i]["time"], reverse=True)[:1]]
            if hits:
                index = hits[0][1]
                matches[index] = found
                matched.add(index)
            else:
                unlinked[("unlinked", len(unlinked))] = found
    return {**direct, **matches, **unlinked}
