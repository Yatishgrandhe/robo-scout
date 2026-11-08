"""Find reviewable launch candidates at the video frame rate."""

from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment


COLORS = {
    "pollen": [((15, 58, 114), (44, 255, 255))],
    "red_nectar": [((0, 160, 100), (10, 255, 255)), ((169, 160, 100), (179, 255, 255))],
    "blue_nectar": [((90, 105, 105), (135, 255, 255))],
}


@dataclass
class Blob:
    kind: str
    xy: np.ndarray
    diameter: float
    streak: bool


@dataclass
class Trail:
    kind: str
    observations: list[tuple[float, Blob]] = field(default_factory=list)


class RobotBoxes:
    def __init__(self, frames, width, height, team=None):
        self.times = [float(frame["time"]) for frame in frames]
        scale = np.array([width, height, width, height])
        boxes = [frame.get("robot") if team is None else frame.get("robots", {}).get(team) for frame in frames]
        self.boxes = [np.asarray(box) * scale if box is not None else None for box in boxes]

    def at(self, time):
        index = bisect_left(self.times, time)
        if index < len(self.times) and abs(self.times[index] - time) < 0.002:
            return self.boxes[index]
        if 0 < index < len(self.times):
            t0, t1 = self.times[index - 1:index + 1]
            before, after = self.boxes[index - 1:index + 1]
            if before is not None and after is not None and 0 < t1 - t0 < 0.22:
                if np.linalg.norm(before[:2] - after[:2]) <= max(30, (before[2] - before[0]) * 1.25):
                    return before + (after - before) * ((time - t0) / (t1 - t0))
        if index == len(self.times) and time - self.times[-1] < 0.06:
            return self.boxes[-1]
        return None


def _blobs(hsv, motion, offset, width, height):
    result = []
    for kind, ranges in COLORS.items():
        color = cv.inRange(hsv, *ranges[0])
        for bounds in ranges[1:]:
            color |= cv.inRange(hsv, *bounds)
        contours, _ = cv.findContours(color, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if not 2 <= w <= width * 0.06 or not 2 <= h <= height * 0.075:
                continue
            area = cv.contourArea(contour)
            if area < 4 or area > width * height * 0.0011 or area < 0.16 * w * h:
                continue
            if max(w, h) > 8 * min(w, h):
                continue
            residual = motion[y:y + h, x:x + w][color[y:y + h, x:x + w] > 0]
            if not residual.size or np.mean(residual) < 8 or np.count_nonzero(residual > 13) < 0.23 * residual.size:
                continue
            m = cv.moments(contour)
            xy = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) + offset
            result.append(Blob(kind, xy, float(min(w, h)), max(w, h) >= 1.65 * min(w, h)))
    return result


def _associate(active, blobs, time, height):
    """Associate on predicted motion so closely spaced shots can keep separate IDs."""
    if not active or not blobs:
        active.extend(Trail(blob.kind, [(time, blob)]) for blob in blobs)
        return
    cost = np.full((len(active), len(blobs)), 1e5)
    points = np.asarray([blob.xy for blob in blobs])
    kinds = np.asarray([blob.kind for blob in blobs])
    for index, trail in enumerate(active):
        previous_time, last = trail.observations[-1]
        dt = time - previous_time
        if dt <= 0:
            continue
        predicted = last.xy
        base = max(8, 1.45 * last.diameter)
        if len(trail.observations) > 1:
            t0, earlier = trail.observations[max(0, len(trail.observations) - 3)]
            if previous_time > t0:
                predicted = predicted + (last.xy - earlier.xy) * (dt / (previous_time - t0))
            gate = base + height * dt * 0.26
        else:
            gate = base + height * dt * 0.98
        error = np.linalg.norm(points - predicted, axis=1)
        displacement = np.linalg.norm(points - last.xy, axis=1)
        valid = (kinds == trail.kind) & (error <= gate) & (displacement <= base + 1.55 * height * dt)
        cost[index, valid] = error[valid] / gate
    matched = set()
    for index, blob_index in zip(*linear_sum_assignment(cost)):
        if cost[index, blob_index] < 1e5:
            active[index].observations.append((time, blobs[blob_index]))
            matched.add(blob_index)
    active.extend(Trail(blob.kind, [(time, blob)]) for index, blob in enumerate(blobs) if index not in matched)


def _source(time, point, velocity, robot, width, height):
    """Back-project a short distance to the upper body of the robot."""
    for delay in np.linspace(0, 0.17, 13):
        when = time - delay
        box = robot.at(when)
        if box is None:
            continue
        x, y = point - velocity * delay
        left, top, right, bottom = box
        body_height = bottom - top
        x_pad = max(width * 0.015, 0.12 * (right - left))
        if left - x_pad <= x <= right + x_pad and top - height * 0.020 <= y <= top + body_height * 0.75:
            return when, box, delay
    return None


def _candidate(trail, robot, width, height):
    observations = trail.observations
    if len(observations) < 3:
        return None
    # A rise late in an existing track is commonly a bounce or a ball
    # already outside the robot. Look only at the initial track.
    for start in range(min(3, len(observations) - 2)):
        segment = observations[start:start + 3]
        times = np.asarray([time for time, _ in segment])
        points = np.asarray([blob.xy for _, blob in segment])
        if times[2] - times[0] < 0.024:
            continue
        velocity = (points[2] - points[0]) / (times[2] - times[0])
        if velocity[1] > -0.18 * height:
            continue
        time, first = segment[0]
        origin = _source(time, first.xy, velocity, robot, width, height)
        if origin is None:
            continue
        when, source_box, delay = origin
        upward = []
        lowest_y = first.xy[1]
        for t, blob in observations[start:]:
            if t - time > 0.65 or blob.xy[1] > lowest_y + height * 0.012:
                break
            lowest_y = min(lowest_y, blob.xy[1])
            upward.append((t, blob))
        if len(upward) < 3:
            continue
        rise = first.xy[1] - lowest_y
        if rise < max(height * 0.028, 1.6 * first.diameter) or lowest_y > source_box[1] - height * 0.020:
            continue
        last_time, last_blob = upward[-1]
        last_robot = robot.at(last_time)
        if last_robot is not None and last_blob.xy[1] > last_robot[1] - height * 0.014:
            continue
        if abs(points[1, 1] - (points[0, 1] + velocity[1] * (times[1] - times[0]))) > height * 0.012:
            continue
        confidence = min(0.98, 0.48 + min(0.20, rise / height) + min(0.2, 0.031 * len(upward))
                         + (0.045 if any(blob.streak for _, blob in upward) else 0) - delay * 0.45)
        return {
            "time": round(float(when), 3),
            "kind": trail.kind,
            "confidence": round(float(confidence), 3),
            "source_error": round(float(
                abs((first.xy - velocity * delay)[0] - (source_box[0] + source_box[2]) / 2)
                / max(1, source_box[2] - source_box[0]) + delay * 1.5
            ), 3),
            "samples": [
                [round(t, 3), round(float(blob.xy[0] / width), 5), round(float(blob.xy[1] / height), 5)]
                for t, blob in upward
            ],
            "flight": [
                [round(t, 3), round(float(blob.xy[0] / width), 5), round(float(blob.xy[1] / height), 5)]
                for t, blob in observations[start:]
            ],
        }
    return None


def _stitch_fragments(candidates, width, height):
    kept = []
    scale = np.array([width, height])
    for candidate in sorted(candidates, key=lambda item: item["samples"][0][0]):
        samples = np.asarray(candidate["samples"])
        for prior in reversed(kept):
            cross_team = prior.get("team") != candidate.get("team")
            if prior["kind"] != candidate["kind"] or abs(prior["time"] - candidate["time"]) > (.42 if cross_team else .085):
                continue
            before = np.asarray(prior["samples"])
            gap = samples[0, 0] - before[-1, 0]
            if not (-0.008 if cross_team else -0.025) <= gap <= (.14 if cross_team else .075) or samples[-1, 0] <= before[-1, 0]:
                continue
            approach = (before[-1, 1:] - before[-3, 1:]) / (before[-1, 0] - before[-3, 0])
            departure = (samples[2, 1:] - samples[0, 1:]) / (samples[2, 0] - samples[0, 0])
            direction = (approach * scale) @ (departure * scale)
            speed_product = np.linalg.norm(approach * scale) * np.linalg.norm(departure * scale)
            if direction < 0.72 * speed_product:
                continue
            displacement = (samples[0, 1:] - before[-1, 1:]) * scale
            if cross_team:
                # A continuation cannot start lower in the image than the
                # last rising sample. Ballistic rise may slow as it approaches
                # the apex, but a horizontally displaced trail is separate.
                predicted = approach * scale * gap
                if (displacement[1] > height * .006 or abs(displacement[0] - predicted[0]) > width * .010
                        or abs(displacement[1] - predicted[1]) > height * .034):
                    continue
            elif max(np.linalg.norm(displacement - velocity * scale * gap)
                     for velocity in (approach, departure)) > max(6, 0.022 * height):
                continue
            prior["samples"] = prior["samples"] + [
                sample for sample in candidate["samples"] if sample[0] > before[-1, 0]
            ]
            prior["confidence"] = max(prior["confidence"], candidate["confidence"])
            if "flight" in prior and "flight" in candidate:
                prior["flight"].extend(sample for sample in candidate["flight"] if sample[0] > prior["flight"][-1][0])
            break
        else:
            kept.append(candidate)
    return kept


def _deduplicate(candidates, width, height):
    kept = []
    stitched = _stitch_fragments(candidates, width, height)
    for candidate in sorted(stitched, key=lambda item: item["confidence"], reverse=True):
        samples = np.array(candidate["samples"])
        duplicate = False
        for prior in kept:
            if prior["kind"] != candidate["kind"] or abs(prior["time"] - candidate["time"]) > 0.17:
                continue
            other = np.array(prior["samples"])
            dt = np.abs(samples[:, None, 0] - other[None, :, 0])
            distance = np.linalg.norm(
                (samples[:, None, 1:] - other[None, :, 1:]) * [width, height], axis=2
            )
            close = (dt < 0.016) & (distance < max(5, 0.012 * height))
            if np.count_nonzero(np.any(close, axis=1)) >= 2:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda item: item["time"])


def _bounds(field, width, height):
    if field is None:
        return 0, 0, width, height
    x0, y0, x1, y1 = field
    # Retain air above the marked floor.
    return max(0, round((x0 - 0.04) * width)), max(0, round((y0 - 0.16) * height)), min(
        width, round((x1 + 0.04) * width)
    ), min(height, round((y1 + 0.035) * height))


def scan_launches(video: Path, frames: list[dict], config, progress=None) -> list[dict]:
    """Scan every decodable frame for visible releases from the tracked robot."""
    if not frames:
        return []
    capture = cv.VideoCapture(str(video))
    if not capture.isOpened():
        return []
    try:
        fps = capture.get(cv.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            return []
        full_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        full_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        if min(full_width, full_height) <= 0:
            return []
        other_robots = getattr(config, "robots", [])
        scale = min(1, (1536 if other_robots else 960) / max(full_width, full_height))
        width, height = round(full_width * scale), round(full_height * scale)
        targets = {seed.team: RobotBoxes(frames, width, height, seed.team) for seed in (config, *other_robots)} if other_robots else {
            None: RobotBoxes(frames, width, height)
        }

        def attributed(trail):
            hits = []
            for team, robot in targets.items():
                candidate = _candidate(trail, robot, width, height)
                if candidate is not None:
                    if team is not None:
                        candidate["team"] = team
                    hits.append(candidate)
            hits.sort(key=lambda hit: (hit["source_error"], -hit["confidence"]))
            return hits[0] if hits and (len(hits) == 1 or hits[1]["source_error"] - hits[0]["source_error"] >= .16) else None

        start_time = float(min(seed.seed_time for seed in (config, *other_robots)))
        end_time = config.end_time if config.end_time is not None else frames[-1]["time"]
        if start_time >= end_time:
            return []
        # A frame index is approximate for variable-rate screen recordings.
        # Seek slightly early, then respect the decoded timestamps.
        first = max(0, round((start_time - 3) * fps))
        if first:
            capture.set(cv.CAP_PROP_POS_FRAMES, first)
        x0, y0, x1, y1 = _bounds(getattr(config, "field", None), width, height)
        if x1 - x0 < 5 or y1 - y0 < 5:
            return []
        active, candidates = [], []
        previous_gray = None
        previous_time = None
        stale_after = max(0.075, 2.25 / fps)
        offset = np.array([x0, y0])
        index = first
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            time = index / fps
            pts = capture.get(cv.CAP_PROP_POS_MSEC) / 1000
            if np.isfinite(pts) and pts >= 0 and (previous_time is None or pts > previous_time):
                time = pts
            elif previous_time is not None:
                time = previous_time + 1 / fps
            previous_time = time
            index += 1
            if time > end_time + 0.5 / fps:
                break
            if time < start_time - 0.08:
                continue
            frame = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)[y0:y1, x0:x1]
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            alive = []
            for trail in active:
                if time - trail.observations[-1][0] <= stale_after and time - trail.observations[0][0] <= 1.3:
                    alive.append(trail)
                else:
                    candidate = attributed(trail)
                    if candidate is not None:
                        candidates.append(candidate)
            active = alive
            if previous_gray is not None:
                motion = cv.absdiff(gray, previous_gray)
                if np.count_nonzero(motion > 22) > 0.28 * motion.size:
                    active = []  # A camera cut cannot establish a physical release.
                elif np.count_nonzero(motion > 10) >= 9:
                    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                    _associate(active, _blobs(hsv, motion, offset, width, height), time, height)
            previous_gray = gray
            if progress is not None and (index - first) % max(1, round(fps)) == 0:
                fraction = (time - start_time) / (end_time - start_time)
                progress(max(0, min(1, fraction)), "Reviewing ball launches")
        for trail in active:
            candidate = attributed(trail)
            if candidate is not None:
                candidates.append(candidate)
        return _deduplicate(candidates, width, height)
    finally:
        capture.release()
