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


SEARCH_BLUE_NECTAR = False
# Balls come to rest on the Cell floor. Resting spots above it (roof corners,
# frame bars, the crowd behind) are ignored. Tuned on 18 reviewed matches.
FLOOR = .55    # resting balls sit below this fraction of the Cell box height
MARGIN = .06   # and this fraction away from its sides


def on_floor(box, x, y):
    x0, y0, x1, y1 = box
    return x0 + MARGIN * (x1 - x0) <= x <= x1 - MARGIN * (x1 - x0) and y0 + FLOOR * (y1 - y0) <= y <= y1


def pollen_centers(image, detections, nectar=None):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    yellow = cv.inRange(hsv, (16, 94, 118), (42, 255, 255))
    contours, _ = cv.findContours(yellow, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)
        if 5 <= w <= 31 and 5 <= h <= 36 and .40 <= w / h <= 2.0 and .48 * w * h <= area <= 550:
            centers.append((x + w / 2, y + h / 2))
    # Nectar has the Cell's own color, so thin frame lines are opened away and
    # only solid, round blobs of ball size remain.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    # Red Nectar is not searched: the red Cell's own frame passes the red test.
    # Blue Nectar is off by default: across 18 reviewed matches the blue frame and
    # blue clothing behind the Cell added more false entries than real ones.
    nectar_ranges = {"blue": ((96, 110, 50), (126, 255, 255))} if SEARCH_BLUE_NECTAR else {}
    for ranges in [nectar_ranges[nectar]] if nectar in nectar_ranges else []:
        mask = cv.inRange(hsv, ranges[0], ranges[1])
        if len(ranges) > 2:
            mask |= cv.inRange(hsv, ranges[2], ranges[3])
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            area = cv.contourArea(contour)
            if 12 <= w <= 40 and 12 <= h <= 40 and .6 <= w / h <= 1.6 and .55 * w * h <= area:
                center = (x + w / 2, y + h / 2)
                if all((center[0] - px) ** 2 + (center[1] - py) ** 2 > 9 ** 2 for px, py in centers):
                    centers.append(center)
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
                centers = pollen_centers(image, boxes, team.split(":")[0])
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


def occupancy(observations, box, near=.014, memory=1.0, rest=.4):
    """[(time, balls resting in the Cell)], counting a spot seen within `memory` seconds.

    Detections flicker behind the Cell mesh; remembering each resting spot for a
    second keeps the count steady, so only a real arrival raises it.
    """
    spots, timeline = [], []
    for time, points in observations:
        for x, y in points:
            if not on_floor(box, x, y):
                continue
            spot = next((s for s in spots if abs(s[0] - x) < near and abs(s[1] - y) < near), None)
            if spot is None:
                spots.append([x, y, time, time])
            else:
                spot[0], spot[1], spot[3] = x, y, time
        spots = [s for s in spots if time - s[3] <= memory]
        timeline.append((time, sum(s[3] - s[2] >= rest for s in spots)))
    return timeline


def adds_a_ball(timeline, when):
    """A real entry leaves more balls resting in the Cell than at any moment shortly before it."""
    before = [n for t, n in timeline if when - 1.2 <= t <= when - .2]
    after = [n for t, n in timeline if when + .6 <= t <= when + 1.2]
    return not before or not after or np.median(after) > np.median(before)


def resting_entries(observations, box, rest=.4, near=.014, empty=1.5, baseline=1.5):
    """New balls that come to rest in the Cell, one per resting position until the Cell empties.

    A ball often drops onto the Cell floor in a frame or two, so no crossing is
    tracked. Each distinct position where a ball stays `rest` seconds counts once;
    when the Cell shows no ball for `empty` seconds (a TIP) the positions reset.
    """
    x0, y0, x1, y1 = box
    candidates, resting, found, last_seen = [], [], [], None
    # Whatever rests in the Cell when the scan starts (frame parts, staged pieces) is not an entry.
    start = observations[0][0] if observations else 0
    fixed = []
    for time, points in observations:
        if time - start > baseline:
            break
        fixed += [(x, y) for x, y in points if on_floor(box, x, y)]
    fixed = [p for p in fixed if sum(abs(p[0] - q[0]) < near and abs(p[1] - q[1]) < near for q in fixed) >= 10]
    for time, points in observations:
        points = [(x, y) for x, y in points if on_floor(box, x, y)
                  and not any(abs(x - fx) < near and abs(y - fy) < near for fx, fy in fixed)]
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
    # A ball nudged to a new spot is not an entry: the Cell must hold more balls than before.
    timeline = occupancy([(t, [p for p in pts if not any(abs(p[0] - fx) < near and abs(p[1] - fy) < near
                                                          for fx, fy in fixed)]) for t, pts in observations], box, near)
    return [hit for hit in found if adds_a_ball(timeline, hit.time)]


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
    timeline = occupancy(observations, box)
    tracked = [found for track in finished + active
               if (found := arrival(track, box) or arrival(track, box, .40) or settled_entry(track, box)) is not None
               and adds_a_ball(timeline, found.time)]
    width = box[2] - box[0]
    # A resting ball already explained by a tracked entry (same time and place) is not a second entry.
    extra = [hit for hit in resting_entries(observations, box)
             if not any(abs(hit.time - other.time) < .8 and abs(hit.path[-1][1] - other.path[-1][1]) < .15 * width
                        for other in tracked)]
    merged = []
    for hit in sorted(tracked + extra, key=lambda hit: hit.time):
        # Two detections of one ball within half a second at the same spot are one entry.
        if any(hit.time - other.time < .5 and abs(hit.path[-1][1] - other.path[-1][1]) < .15 * width
               for other in merged):
            continue
        merged.append(hit)
    return merged


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
                unlinked[("unlinked", len(unlinked), key)] = found
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
                unlinked[("unlinked", len(unlinked), key)] = found
    return {**direct, **matches, **unlinked}


def shooter_below(time, cell, teams, boxes_at, lead=(.3, .6, .9)):
    """The alliance robot under a Cell just before a ball rests in it, or None if unclear.

    Robots parked at the Hive base shoot almost straight up behind the Hive
    frame, where the flight is hidden; the robot underneath is the shooter.
    """
    x0, y0, x1, y1 = cell
    middle, width = (x0 + x1) / 2, x1 - x0
    scored = []
    for team in teams:
        distances = []
        for delay in lead:
            box = boxes_at(team, time - delay)
            if box is not None and box[1] >= y1 - .02:
                distances.append(abs((box[0] + box[2]) / 2 - middle) / width)
        if distances:
            scored.append((min(distances), team))
    scored.sort()
    if not scored or scored[0][0] > 1.5 or len(scored) > 1 and scored[1][0] - scored[0][0] < .3:
        return None
    return scored[0][1]
