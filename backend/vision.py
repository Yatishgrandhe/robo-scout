"""Single-camera visual evidence extraction. Rule calls remain reviewable."""
import math
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

from .camera import CameraRegistration
from .defense import defensive_ratings
from .field_detection import detect_field
from .robot_recovery import AppearanceRecovery
from .rules import event
from .track_refinement import refine_roster

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
BALLS = {"pollen", "red_nectar", "blue_nectar"}
FINAL_ZONES = {"cell": ("cell_ball", "§10.5.1"),
               "flower": ("flower_ball", "§10.5.2"), "garden": ("garden_ball", "§10.5.3")}


def center(box):
    return np.array(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))


def rect(box, width, height):
    return np.rint(np.array(box) * (width, height, width, height)).astype(int)


def inside(point, box, pad=0):
    return box[0] - pad <= point[0] <= box[2] + pad and box[1] - pad <= point[1] <= box[3] + pad


def iou(a, b):
    area = lambda z: max(0, z[2] - z[0]) * max(0, z[3] - z[1])
    overlap = area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    return overlap / max(area(a) + area(b) - overlap, 1e-8)


class RobotTrack:
    """Follow the user-seeded robot and leave gaps if its appearance cannot be found."""

    def __init__(self, gray, box, contour=None):
        self.gray = gray
        self.box = box
        h, w = gray.shape
        x0, y0, x1, y1 = rect(box, w, h)
        self.template = gray[y0:y1, x0:x1].copy()
        self.contour = contour
        self.points = self.features(gray)
        self.misses = 0

    def features(self, gray):
        if self.box is None:
            return None
        mask = np.zeros(gray.shape, np.uint8)
        if self.contour is not None:
            polygon = np.rint(np.array(self.contour) * gray.shape[::-1]).astype(np.int32)
            cv.fillPoly(mask, [polygon], 255)
        else:
            x0, y0, x1, y1 = rect(self.box, *gray.shape[::-1])
            cv.rectangle(mask, (x0, y0), (x1 - 1, y1 - 1), 255, -1)
        return cv.goodFeaturesToTrack(gray, maxCorners=55, qualityLevel=.015, minDistance=4, mask=mask)

    def advance(self, gray, boxes):
        h, w = gray.shape
        estimate = None
        if self.points is not None and self.misses == 0:
            moving, ok, _ = cv.calcOpticalFlowPyrLK(self.gray, gray, self.points, None)
            if moving is not None:
                back, ok_back, _ = cv.calcOpticalFlowPyrLK(gray, self.gray, moving, None)
                if back is not None:
                    moving = moving.reshape(-1, 2)
                    back = back.reshape(-1, 2)
                    original = self.points.reshape(-1, 2)
                    valid = (ok.ravel() == 1) & (ok_back.ravel() == 1)
                    valid &= np.linalg.norm(back - original, axis=1) < 1.6
                    if valid.sum() >= 5:
                        delta = np.median(moving[valid] - original[valid], axis=0)
                        if np.linalg.norm(delta / (w, h)) < .10:
                            estimate = np.clip(self.box + np.tile(delta / (w, h), 2), 0, 1)
        # A detector may refine an active track, but proximity alone cannot reacquire identity.
        if estimate is not None and boxes:
            best = max(boxes, key=lambda box: iou(box, estimate))
            if iou(best, estimate) > .52:
                estimate = .70 * estimate + .30 * best
        if estimate is None and self.template.size and self.misses < 16:
            x0, y0, x1, y1 = rect(self.box, w, h)
            pad = int(min(w, h) * (.055 + min(self.misses, 8) * .01))
            left, top = max(0, x0 - pad), max(0, y0 - pad)
            crop = gray[top:min(h, y1 + pad), left:min(w, x1 + pad)]
            if all(a >= b for a, b in zip(crop.shape, self.template.shape)):
                _, quality, _, (x, y) = cv.minMaxLoc(cv.matchTemplate(crop, self.template, cv.TM_CCOEFF_NORMED))
                th, tw = self.template.shape
                if quality > .69:
                    estimate = np.array((left + x, top + y, left + x + tw, top + y + th)) / (w, h, w, h)
        self.gray = gray
        if estimate is None:
            self.misses += 1
            self.points = None
            return None
        if self.contour is not None:
            shift = center(estimate) - center(self.box)
            self.contour = np.clip(np.array(self.contour) + shift, 0, 1)
        self.box = estimate
        self.misses = 0
        self.points = self.features(gray)
        return estimate


class TeamTrack:
    def __init__(self, frame, gray, box, field, contour=None, nano_params=None):
        self.tracker = RobotTrack(gray, box, contour)
        self.recovery = AppearanceRecovery(frame, box, field)
        self.supported = box
        self.unvalidated = 0
        self.identity_frames = 0
        self.recovery_frames = 0
        self.nano_params = nano_params
        self.nano = None
        self.nano_reference = box
        self.reset_nano(frame, box)

    def reset_nano(self, frame, box):
        if self.nano_params is None:
            return
        h, w = frame.shape[:2]
        x, y, right, bottom = rect(box, w, h)
        try:
            self.nano = cv.TrackerNano_create(self.nano_params)
            self.nano.init(frame, (int(x), int(y), int(right - x), int(bottom - y)))
            self.nano_reference = box
        except cv.error:
            self.nano = None

    def nano_box(self, frame):
        if self.nano is None:
            return None
        ok, (x, y, width, height) = self.nano.update(frame)
        if not ok or self.nano.getTrackingScore() < .68:
            return None
        h, w = frame.shape[:2]
        box = np.clip(np.array((x, y, x + width, y + height)) / (w, h, w, h), 0, 1)
        size = box[2:] - box[:2]
        reference = self.nano_reference[2:] - self.nano_reference[:2]
        prior = self.supported
        if np.any(size < .38 * reference) or np.any(size > 1.95 * reference) or np.linalg.norm(
            (center(box) - center(prior)) / np.maximum(prior[2:] - prior[:2], .001)
        ) > .88:
            return None
        return box

    def advance(self, frame, gray, generic, named, moving_camera, transform, anchor=None):
        if anchor is not None:
            self.tracker = RobotTrack(gray, anchor)
            self.supported, self.unvalidated = anchor, 0
            self.recovery.find(frame, anchor, background_transform=transform)
            self.reset_nano(frame, anchor)
            return anchor
        robot = self.tracker.advance(gray, generic)
        if named:
            best, confidence = max(named, key=lambda candidate: candidate[1])
            agrees = robot is not None and iou(robot, best) > .30
            if confidence >= .48 or confidence >= .30 and agrees:
                self.identity_frames += 1
                self.supported, self.unvalidated = best, 0
                if not agrees or iou(self.tracker.box, best) < .62:
                    self.tracker = RobotTrack(gray, best)
                    self.reset_nano(frame, best)
                return best
        nano = self.nano_box(frame)
        if nano is not None:
            self.supported, self.unvalidated = nano, 0
            # Keep the feature track close without rebuilding its template
            # on a different robot after an occlusion.
            if robot is None or iou(robot, nano) < .45:
                self.tracker = RobotTrack(gray, nano)
            return nano
        broad = robot is None or self.unvalidated > 5
        appearance = self.recovery.find(frame, self.supported if broad or moving_camera else robot,
                                        lost=broad, background_transform=transform)
        if appearance is not None:
            overlaps = robot is not None and iou(robot, appearance) > .28
            if overlaps or broad:
                self.supported, self.unvalidated = appearance, 0
                if not overlaps:
                    self.tracker = RobotTrack(gray, appearance)
                    self.recovery_frames += 1
                    self.reset_nano(frame, appearance)
                    return appearance
                return robot
        self.unvalidated += 1
        if self.nano_params is not None and self.unvalidated > 3:
            return None
        return None if moving_camera and self.recovery.hue is not None and self.unvalidated > 12 else robot


def color_detections(frame):
    """Geometry-gated HSV proposals supplement a generic sports-ball detector."""
    h, w = frame.shape[:2]
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    minimum = max(5, round(h * .009) ** 2)
    maximum = (h * .073) ** 2
    ranges = {"pollen": [(19, 40)], "red_nectar": [(0, 9), (170, 179)], "blue_nectar": [(91, 132)]}
    found = []
    for kind, hues in ranges.items():
        mask = np.zeros((h, w), np.uint8)
        for lo, hi in hues:
            mask |= cv.inRange(hsv, (lo, 105, 85), (hi, 255, 255))
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv.contourArea(contour)
            if not minimum <= area <= maximum:
                continue
            x, y, bw, bh = cv.boundingRect(contour)
            perimeter = cv.arcLength(contour, True)
            roundness = 4 * math.pi * area / max(1, perimeter * perimeter)
            if min(bw, bh) < 4 or not .62 <= bw / bh <= 1.62 or roundness < .60:
                continue
            found.append((kind, np.array((x, y, x + bw, y + bh)) / (w, h, w, h), .43))
    return found


@dataclass
class BallTrack:
    id: int
    kind: str
    samples: list = field(default_factory=list)
    release: float | None = None
    release_team: str | None = None
    arrivals: dict = field(default_factory=dict)

    def update(self, now, box, robot, zones):
        point = center(box)
        if isinstance(robot, dict):
            nearby = [team for team, target in robot.items() if target is not None and inside(point, target, .015)]
            close = nearby[0] if len(nearby) == 1 else None
            clear = not nearby
        else:
            close = robot is not None and inside(point, robot, .025)
            clear = not close
        if clear and self.release is None and self.samples:
            recent = [sample for sample in self.samples[-12:] if now - sample[0] <= .8 and sample[2]]
            unique = {sample[2] for sample in recent}
            if len(recent) >= (2 if isinstance(robot, dict) else 1) and len(unique) == 1 and np.linalg.norm(
                point - center(recent[-1][1])
            ) > .025:
                self.release = now
                if isinstance(robot, dict):
                    self.release_team = recent[-1][2]
        self.samples.append((now, box, close))
        for index, zone in enumerate(zones):
            if not inside(point, zone.box):
                self.arrivals.pop(index, None)
            elif index not in self.arrivals:
                self.arrivals[index] = now

    def forecast(self, now):
        time, box, _ = self.samples[-1]
        point = center(box)
        if len(self.samples) > 1:
            prev_t, prev_box, _ = self.samples[-2]
            delta = (point - center(prev_box)) / max(.01, time - prev_t)
            point += np.clip(delta * (now - time), -.1, .1)
        return point


def match_tracks(tracks, observations, time, robot, zones):
    active = [track for track in tracks if time - track.samples[-1][0] <= .60]
    matched = set()
    if active and observations:
        cost = np.full((len(active), len(observations)), 10.0)
        for i, track in enumerate(active):
            expected = track.forecast(time)
            for j, (kind, box, _) in enumerate(observations):
                if track.kind == kind:
                    cost[i, j] = np.linalg.norm(expected - center(box))
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] <= .085:
                active[i].update(time, observations[j][1], robot, zones)
                matched.add(j)
    for j, (kind, box, _) in enumerate(observations):
        if j not in matched:
            track = BallTrack(len(tracks) + 1, kind)
            track.update(time, box, robot, zones)
            tracks.append(track)


def early_nectar(tracks, now, zones, config, flagged):
    if config.start_time is None or not any(zone.kind == "flower" for zone in zones):
        return []
    events = []
    for track in tracks:
        if track.release is None or track.samples[-1][0] != now or track.kind == "pollen":
            continue
        for zi, entered in track.arrivals.items():
            key = track.id, zi
            if (zones[zi].kind == "flower" and key not in flagged and config.start_time <= entered < config.start_time + 98
                    and now - entered >= .25 and track.release <= entered):
                flagged.add(key)
                row = event("penalty", entered, "Possible early Nectar in Flower", "G410",
                            severity="major", source="vision",
                            evidence="Target-associated Nectar entered the marked Flower before the last 60 seconds. Verify placement and the field timer.")
                events.append({**row, "team": track.release_team} if track.release_team is not None else row)
    return events


def connected_launch(track, launches, entered, used):
    """Match a retained piece to a single flight with overlapping observations."""
    samples = np.array([(time, *center(box)) for time, box, _ in track.samples if entered - 1.4 <= time <= entered + .35])
    if len(samples) < 2:
        return None
    matches = []
    for i, launch in enumerate(launches):
        if i in used or launch["kind"] != track.kind or not 0 <= entered - launch["time"] <= 1.5:
            continue
        path = np.asarray(launch["flight"])
        gap = np.abs(samples[:, None, 0] - path[None, :, 0])
        distance = np.linalg.norm(samples[:, None, 1:] - path[None, :, 1:], axis=2)
        close = (gap <= .072) & (distance <= .017)
        if np.count_nonzero(np.any(close, axis=1)) >= 2 and np.count_nonzero(np.any(close, axis=0)) >= 2:
            matches.append((i, launch))
    return matches[0] if len(matches) == 1 else None


def nectar_pickups(tracks, frames, shape, transforms, config):
    """Find loose Nectar that then moves with the selected robot."""
    frame_index = {round(frame["time"], 3): index for index, frame in enumerate(frames)}
    scale = np.array(shape)
    events = []
    for track in tracks:
        if track.kind == "pollen":
            continue
        loose, held = [], []
        for time, box, _ in track.samples:
            index = frame_index.get(round(time, 3))
            robot = frames[index]["robot"] if index is not None else None
            if robot is None:
                loose.clear()
                held.clear()
                continue
            robot = np.asarray(robot)
            point = center(box)
            other_boxes = frames[index].get("robots", {})
            if any(team != config.team and other is not None and inside(point, other, .006)
                   for team, other in other_boxes.items()):
                loose.clear()
                held.clear()
                continue
            relative = (point - robot[:2]) / np.maximum(robot[2:] - robot[:2], .001)
            positions = np.stack((point, center(robot))) * scale
            if transforms:
                transform = transforms[index]
                positions = positions @ transform[:, :2].T + transform[:, 2]
            position, robot_position = positions / max(shape)
            if np.any(relative < .02) or np.any(relative > .98):
                if held:
                    loose.clear()
                    held.clear()
                if not inside(point, robot, .006):
                    loose = [entry for entry in loose if time - entry[0] <= .40]
                    loose.append((time, position))
                continue
            if len(loose) < 2 or held and time - held[-1][0] > .21:
                held.clear()
                continue
            if not held and (time - loose[-1][0] > .28 or np.linalg.norm(loose[0][1] - loose[-1][1]) > .028):
                loose.clear()
                continue
            held.append((time, position, robot_position, relative))
            first = held[0]
            if time - first[0] > 1.2:
                held.clear()
                loose.clear()
                continue
            if len(held) < 4 or time - first[0] < .27:
                continue
            movement = robot_position - first[2]
            if (np.linalg.norm(movement) < .021 or np.linalg.norm(position - first[1] - movement) > .013
                    or max(np.linalg.norm(entry[3] - first[3]) for entry in held) > .35):
                continue
            when = first[0]
            held.clear()
            loose.clear()
            if config.start_time is not None and when < config.start_time:
                continue
            opponent = track.kind != f"{config.alliance}_nectar"
            detail = "A previously loose piece entered the robot outline and moved with it. Verify the intake and the piece color."
            events.append(event("nectar_pickup", when, f"Possible {'opponent' if opponent else 'own'} Nectar pickup",
                                "G408" if opponent else "", source="vision", piece=track.kind, evidence=detail))
            if opponent:
                events.append(event("penalty", when, "Possible opponent Nectar control", "G408",
                                    severity="warning", source="vision",
                                    evidence="Verify that the robot controlled this opponent-colored piece during play. G408 calls for a warning and can carry a yellow card for strategic control."))
    return events


def hive_angle(gray, zone):
    h, w = gray.shape
    x0, y0, x1, y1 = rect(zone.box, w, h)
    crop = gray[y0:y1, x0:x1]
    if min(crop.shape) < 12:
        return 0
    edges = cv.Canny(crop, 55, 150)
    length = max(12, int(crop.shape[1] * .28))
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, max(12, length // 2),
                          minLineLength=length, maxLineGap=6)
    evidence = 0
    if lines is not None:
        for x, y, a, b in lines.reshape(-1, 4):
            dx, dy = int(a) - int(x), int(b) - int(y)
            if dx and .18 < abs(dy / dx) < 1.6:
                evidence += abs(dx) * np.sign(dx * dy)
    return int(np.sign(evidence))


def hive_color(frame, zone):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = rect(zone.box, w, h)
    hsv = cv.cvtColor(frame[y0:y1, x0:x1], cv.COLOR_BGR2HSV)
    if zone.alliance == "red":
        mask = cv.inRange(hsv, (0, 75, 45), (12, 255, 255))
        mask |= cv.inRange(hsv, (168, 75, 45), (179, 255, 255))
    else:
        mask = cv.inRange(hsv, (91, 75, 45), (134, 255, 255))
    return cv.GaussianBlur(mask.astype(np.float32) / 255, (7, 7), 1.5)


def hive_reference(cap, fps, shape, zone, registration=None):
    samples = []
    for time in zone.state_frames:
        cap.set(cv.CAP_PROP_POS_FRAMES, round(time * fps))
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv.resize(frame, shape)
        if registration is not None:
            frame = registration.stabilize(frame, reference_sample=True)
            if frame is None:
                return None
        samples.append(hive_color(frame, zone))
    difference = samples[0] - samples[1]
    weight = np.abs(difference)
    separation = float(np.sum(difference * difference))
    return (*samples, weight, separation) if separation >= 8 else None


def hive_pose(frame, zone, reference):
    first, second, weight, separation = reference
    state = hive_color(frame, zone)
    distance = float(np.sum(weight * (np.abs(state - second) - np.abs(state - first))))
    if abs(distance) < .22 * separation:
        return 0
    return 1 if distance > 0 else -1


def analyze(video: Path, config, progress, launch_video: Path | None = None, hive_weights: Path | None = None):
    import torch
    from ultralytics import SAM, YOLO

    progress(.015, "Opening the match video")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    weights = MODELS / ({"generic": "yolo26n.pt", "synthetic": "synthetic.pt", "ftc": "ftc.pt"}[config.model])
    warnings = []
    cap = cv.VideoCapture(str(video))
    try:
        fps = cap.get(cv.CAP_PROP_FPS)
        total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0 or total <= 0:
            raise ValueError("OpenCV could not read the normalized video.")
        effective = {}
        manual_zones = [zone for zone in config.zones if not zone.automatic]
        marked_cells = {zone.alliance for zone in manual_zones if zone.kind == "cell"}
        if config.auto_zones and len(marked_cells) < 2 and len(manual_zones) < 32:
            progress(.016, "Finding scoring zones")
            found = detect_field(cap, fps, config, (total - 1) / fps)
            if found is not None:
                from .main import Zone

                proposals, end_time = found
                if end_time is not None and end_time <= max(seed.seed_time for seed in (config, *config.robots)) + .1:
                    end_time = config.end_time
                detected = [Zone(**zone) for zone in proposals if zone["alliance"] not in marked_cells]
                zones = (manual_zones + detected)[:32]
                config = config.model_copy(update={"zones": zones, "end_time": end_time})
                effective = {"zones": [zone.model_dump() for zone in zones], "end_time": end_time}
                warnings.append("Pitlane detected visible Hive Cell zones. Review the boundaries and mark any visible Flowers or Gardens you want to score.")
        roster = {seed.team: seed for seed in (config, *config.robots)}
        multi = bool(config.robots)
        nano_params = None
        backbone, head = (MODELS / f"nanotrack_{part}_sim.onnx" for part in ("backbone", "head"))
        if multi and hasattr(cv, "TrackerNano_create") and backbone.is_file() and head.is_file():
            cv.setNumThreads(2)
            nano_params = cv.TrackerNano_Params()
            nano_params.backbone, nano_params.neckhead = str(backbone), str(head)
        refine = bool(nano_params is not None and config.anchors and not config.camera_motion)
        start = round(min(seed.seed_time for seed in roster.values()) * fps)
        stop = min(total - 1, round(config.end_time * fps)) if config.end_time is not None else total - 1
        cap.set(cv.CAP_PROP_POS_FRAMES, start)
        ok, first = cap.read()
        if not ok or stop <= start:
            raise ValueError("Choose an earlier robot frame and an end time after it.")
        height, width = first.shape[:2]
        field = config.field or (0, 0, 1, 1)
        extent = max(width * (field[2] - field[0]), height * (field[3] - field[1]))
        scale = min(1, 960 / extent, 1500 / max(width, height))
        shape = (round(width * scale), round(height * scale))
        first = cv.resize(first, shape)
        registration = CameraRegistration(first, config.field) if config.camera_motion else None
        progress(.02, "Segmenting the selected robot" if config.sam else "Loading the detector")
        contour = None
        if config.sam and start == round(config.seed_time * fps):
            if not (MODELS / "sam2.1_t.pt").is_file():
                warnings.append("SAM weights unavailable. The tracker used the selected rectangle.")
            else:
                segmenter = None
                try:
                    segmenter = SAM(MODELS / "sam2.1_t.pt")
                    prompt = [rect(config.robot, *shape).tolist()]
                    # CPU keeps this one-shot operation independent of MPS attention kernels.
                    prediction = segmenter.predict(first, bboxes=prompt, device="cpu", imgsz=640, verbose=False)[0]
                    if prediction.masks is not None and len(prediction.masks.xyn):
                        outline = prediction.masks.xyn[0]
                        if len(outline) >= 4:
                            contour = cv.approxPolyDP(outline.astype(np.float32), .003, True).reshape(-1, 2).tolist()
                    if contour is None:
                        warnings.append("SAM did not find a usable outline. The tracker used the selected rectangle.")
                except (RuntimeError, ValueError, cv.error) as exc:
                    warnings.append(f"SAM outline unavailable ({type(exc).__name__}). The tracker used the selected rectangle.")
                finally:
                    del segmenter
        if config.model == "synthetic":
            warnings.append("This detector was fine-tuned on synthetic footage. Match accuracy is unmeasured.")
        if config.end_time is None:
            warnings.append("Match end was unset. Final ball candidates were skipped.")
        if config.start_time is None:
            warnings.append("Match start was unset. Early Nectar candidates were skipped.")
        elif config.seed_time > config.start_time + .5:
            warnings.append("The robot was marked after match start. Actions before the marked frame were skipped.")
        if not config.zones:
            warnings.append("No scoring zones were drawn. Add events in review or rerun with calibration.")
        progress(.04, "Loading YOLO weights")
        model = YOLO(weights)
        target_classes = {f"robot_{team.strip()}": team for team in roster}
        has_identity_model = f"robot_{config.team.strip()}" in model.names.values()
        identity_teams = [name.removeprefix("robot_") for name in model.names.values() if name.startswith("robot_")]
        if identity_teams and not has_identity_model:
            warnings.append(f"Identity classes cover teams {', '.join(identity_teams)}. This run used the selected robot's visual track.")
        step = max(1, round(fps / config.sample_fps))
        sample_at = lambda time: start + round((time * fps - start) / step) * step
        seeds = {team: sample_at(seed.seed_time) for team, seed in roster.items()}
        anchors = {(sample_at(anchor.time), anchor.team): np.array(anchor.box) for anchor in config.anchors
                   if anchor.team in roster and anchor.time > roster[anchor.team].seed_time}
        x0, y0, x1, y1 = rect(field, *shape)
        detection_origin = np.array((x0, y0, x0, y0))
        detection_scale = np.array((x1 - x0, y1 - y0, x1 - x0, y1 - y0))
        frame_scale = np.array((*shape, *shape))
        references = {}
        for zi, zone in enumerate(config.zones):
            if zone.kind == "hive" and zone.state_frames is not None:
                reference = hive_reference(cap, fps, shape, zone, registration)
                if reference is not None:
                    references[zi] = reference
                else:
                    warnings.append("A Hive reference could not be aligned. Verify its estimated flips in the footage.")
        if any(zone.state_frames is not None for zone in config.zones):
            cap.set(cv.CAP_PROP_POS_FRAMES, start + 1)
        tracks, frames, flips, transforms = [], [], [], []
        hive_states = {}
        events = []
        flagged = set()
        trackers = {}
        for index in range(start, stop + 1):
            if index == start:
                frame = first
            else:
                if not cap.grab():
                    break
                if (index - start) % step:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                frame = cv.resize(frame, shape)
            now = index / fps
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if registration is not None and len(frames) % 3 == 0:
                registration.update(frame)
            background_transform = registration.transform if registration is not None else None
            zones = registration.boxes(config.zones) if registration is not None else config.zones
            field_image = frame[y0:y1, x0:x1]
            proposals = [(kind, (box * detection_scale + detection_origin) / frame_scale, score)
                         for kind, box, score in color_detections(field_image)]
            prediction = model.predict(field_image, imgsz=768 if config.camera_motion else 640,
                                       conf=.17, max_det=100, device=device, verbose=False)[0]
            observed, robots, targets = [], [], {team: [] for team in roster}
            for *coords, score, cls in prediction.boxes.data.cpu().numpy():
                kind = prediction.names[int(cls)]
                box = np.clip((np.array(coords) + detection_origin) / frame_scale, 0, 1)
                if kind in target_classes:
                    targets[target_classes[kind]].append((box, float(score)))
                elif kind == "robot":
                    robots.append(box)
                elif kind in BALLS:
                    observed.append((kind, box, float(score)))
                elif kind == "sports ball":
                    overlaps = [candidate for candidate in proposals if inside(center(candidate[1]), box, .012)]
                    if overlaps:
                        kind, tight, _ = max(overlaps, key=lambda candidate: iou(box, candidate[1]))
                        observed.append((kind, tight, float(score)))
            for proposal in proposals:
                if not any(np.linalg.norm(center(proposal[1]) - center(found[1])) < .012 for found in observed):
                    observed.append(proposal)
            boxes, prior = {}, {}
            for team, seed in roster.items():
                if index == seeds[team]:
                    box = np.array(seed.robot)
                    trackers[team] = TeamTrack(frame, gray, box, config.field,
                                               contour if team == config.team else None, nano_params)
                    boxes[team] = box
                elif team in trackers:
                    tracking = trackers[team]
                    if multi:
                        prior[team] = (copy(tracking.tracker), tracking.supported)
                    boxes[team] = tracking.advance(frame, gray, robots, targets[team], config.camera_motion,
                                                   background_transform, anchors.get((index, team)))
                else:
                    boxes[team] = None
            if multi:
                collisions = {team for team, box in boxes.items() if box is not None and any(
                    other != team and other_box is not None and iou(box, other_box) > .50
                    for other, other_box in boxes.items()
                )}
                for team in collisions:
                    if team in prior and (index, team) not in anchors:
                        trackers[team].tracker, trackers[team].supported = prior[team]
                        boxes[team] = None
            robot = boxes[config.team]
            if not refine:
                match_tracks(tracks, observed, now, boxes if multi else robot, zones)
                events.extend(early_nectar(tracks, now, zones, config, flagged))
            reference_frame = registration.stabilize(frame) if registration is not None else frame
            reference_gray = cv.cvtColor(reference_frame, cv.COLOR_BGR2GRAY) if registration is not None else gray
            for zi, zone in enumerate(config.zones):
                if zone.kind == "hive":
                    reference = references.get(zi)
                    angle = hive_pose(reference_frame, zone, reference) if reference is not None else hive_angle(reference_gray, zone)
                    stable, last, streak = hive_states.get(zi, (0, 0, 0))
                    streak = streak + 1 if angle and angle == last else int(bool(angle))
                    hold = .45 if reference is not None else .24
                    if streak >= max(2, round(config.sample_fps * hold)) and angle != stable:
                        if stable:
                            flips.append((now, zi))
                        stable = angle
                    hive_states[zi] = stable, angle, streak
            frames.append({"time": round(now, 3), "robot": robot.round(4).tolist() if robot is not None else None,
                           "mask": np.round(trackers[config.team].tracker.contour, 4).tolist() if robot is not None and trackers[config.team].tracker.contour is not None else None,
                           **({"robots": {team: box.round(4).tolist() if box is not None else None
                                          for team, box in boxes.items()},
                               "reanchored": [team for team in roster if (index, team) in anchors]} if multi else {}),
                           **({"zones": [np.round(zone.box, 4).tolist() for zone in zones]} if registration is not None else {}),
                           "balls": [{"class": k, "box": b.round(4).tolist(), "confidence": round(c, 2)}
                                     for k, b, c in observed[:65]]})
            if registration is not None:
                transforms.append(registration.transform)
            if len(frames) % max(1, config.sample_fps) == 0:
                progress(.05 + (.50 if refine else .67 if config.scan_shots else .90) * (index - start) / (stop - start),
                         "Tracking the robot and game pieces")
        if not frames:
            raise ValueError("The selected interval has no readable video frames.")
        if refine:
            cap.release()
            progress(.56, "Checking team tracks against keyframes")
            try:
                refine_roster(video, frames, roster, config.anchors, shape, nano_params, field,
                              lambda value: progress(.56 + .15 * value, "Checking team tracks against keyframes"))
            except cv.error:
                warnings.append("Bidirectional tracking was unavailable. Using the forward tracks.")
            for frame in frames:
                observations = [(ball["class"], np.asarray(ball["box"]), ball["confidence"]) for ball in frame["balls"]]
                match_tracks(tracks, observations, frame["time"], frame["robots"], config.zones)
                events.extend(early_nectar(tracks, frame["time"], config.zones, config, flagged))
        final_time = frames[-1]["time"]
        if multi:
            for team, seed in roster.items():
                team_frames = [{**frame, "robot": frame["robots"][team]} for frame in frames]
                events.extend({**row, "team": team} for row in nectar_pickups(
                    tracks, team_frames, shape, transforms, config.model_copy(update={"team": team, "alliance": seed.alliance})))
        else:
            events.extend(nectar_pickups(tracks, frames, shape, transforms, config))
        launches = []
        cells = {}
        if config.scan_shots and not config.camera_motion:
            for alliance in ("red", "blue"):
                marked = [zone.box for zone in config.zones if zone.kind == "cell" and zone.alliance == alliance]
                if len(marked) == 1:
                    cells[alliance] = marked[0]
        scan_hive = bool(cells and hive_weights)
        if config.scan_shots:
            from .launches import scan_launches

            cap.release()
            progress(.73, "Finding launches at video frame rate")
            launches = scan_launches(launch_video or video, frames, config,
                                    lambda value, stage: progress(.73 + (.17 if scan_hive else .24) * value, stage))
            for launch in launches:
                kind = launch["kind"]
                row = event(f"{kind}_launch", launch["time"], f"Possible {kind.replace('_', ' ').title()} launch",
                            "", source="vision",
                            evidence="The fast ball pass found a rising trajectory leaving the tracked robot. Check the launch and its color.")
                row["trajectory"] = launch["samples"]
                if multi:
                    row["team"] = launch["team"]
                events.append(row)
        if cells:
            from .hive_entries import scan_mouths, verified_candidates

            observations = {}
            if scan_hive:
                progress(.91, "Following Pollen into the Hive openings")
                try:
                    observations = scan_mouths(launch_video or video, cells, hive_weights, frames[0]["time"],
                                               frames[-1]["time"], device,
                                               lambda value: progress(.91 + .07 * value, "Following Pollen into the Hive openings"))
                except (OSError, RuntimeError, ValueError, cv.error):
                    warnings.append("The match-specific Hive detector was unavailable. Used launch paths for entry suggestions.")
            alliances = {team: seed.alliance for team, seed in roster.items()} if multi else {None: config.alliance}
            for i, hit in verified_candidates(launches, cells, alliances, observations).items():
                launch = launches[i]
                piece = launch["kind"]
                row = event("cell_entry", hit.time, f"Possible {piece.replace('_', ' ').title()} Hive entry", "",
                            source="vision", piece=piece,
                            evidence="The ball descended into the opening and slowed in the marked Cell. Check the release and whether it stayed inside.")
                row["launch_time"] = launch["time"]
                row["trajectory"] = hit.path
                events.append({**row, "team": launch["team"]} if multi else row)
        end_cells = {}
        used_launches = set()
        if config.end_time is not None and abs(final_time - config.end_time) <= max(.35, 2 / config.sample_fps):
            for track in tracks:
                if final_time - track.samples[-1][0] > max(.28, 1.5 / config.sample_fps):
                    continue
                cells = [(zi, entered) for zi, entered in track.arrivals.items()
                         if config.zones[zi].kind == "cell" and final_time - entered >= .4]
                if cells:
                    cell = config.zones[min(cells, key=lambda hit: hit[1])[0]].alliance
                    visible = end_cells.setdefault(cell, {"pollen": 0, "nectar": 0})
                    visible["pollen" if track.kind == "pollen" else "nectar"] += 1
                if track.release is None and not (multi and launches):
                    continue
                possible = [(zi, entered) for zi, entered in track.arrivals.items() if
                            config.zones[zi].kind in FINAL_ZONES and (track.release is None or entered >= track.release) and
                            final_time - entered >= .4]
                if not possible:
                    continue
                # Each physical track can contribute at most one final ball.
                zi, entered = min(possible, key=lambda hit: ((config.zones[hit[0]].box[2] - config.zones[hit[0]].box[0]) *
                                                      (config.zones[hit[0]].box[3] - config.zones[hit[0]].box[1])))
                zone = config.zones[zi]
                team = track.release_team if multi else config.team
                link = connected_launch(track, launches, entered, used_launches) if multi and zone.kind == "cell" else None
                if link:
                    launch_index, launch = link
                    if team is not None and team != launch["team"]:
                        continue
                    team = launch["team"]
                elif track.release is None:
                    continue
                if team not in roster or zone.alliance != roster[team].alliance:
                    continue
                if zone.kind == "cell" and any(
                    track.arrivals[zi] < when - .6 and iou(zone.box, config.zones[hive_index].box) > 0
                    for when, hive_index in flips
                ):
                    continue
                kind, rule = FINAL_ZONES[zone.kind]
                detail = {"cell": "Check that this was the upward-facing Cell after settling.",
                          "flower": "Check the scoring volume and which alliance owns the Flower.",
                          "garden": "Check that the ball overlaps the alliance Garden."}[zone.kind]
                row = event(kind, final_time, f"Possible final {track.kind.replace('_', ' ').title()} in {zone.kind.title()}",
                            rule, source="vision", piece=track.kind,
                            evidence=f"The ball track moved away from the selected robot and remained in the marked zone. {detail}")
                if link:
                    used_launches.add(launch_index)
                    row["evidence"] = f"The flight at {launch['time']:.2f}s matches the retained piece’s path. {detail}"
                    row["launch_time"] = launch["time"]
                events.append({**row, "team": team} if multi else row)
        coverage = sum(frame["robot"] is not None for frame in frames) / len(frames)
        defense = defensive_ratings(frames, roster, shape, transforms, config.start_time) if multi else {}
        if coverage < .85:
            warnings.append(f"Robot visible in {coverage:.0%} of sampled frames. Review gaps for missed events.")
        return {"events": sorted(events, key=lambda row: row["time"]), "frames": frames,
                "coverage": round(coverage, 3), "mode": config.model, "device": device,
                "segmentation": contour is not None, "identity_model": has_identity_model,
                "identity_detections": sum(track.identity_frames for track in trackers.values()),
                "appearance_recoveries": sum(track.recovery_frames for track in trackers.values()),
                "effective_setup": effective,
                "defense": defense,
                "end_cells": end_cells,
                "team_coverage": {team: round(sum(frame["robots"][team] is not None for frame in frames) / len(frames), 3)
                                  for team in roster} if multi else {},
                "shot_scan": config.scan_shots, "entry_scan": bool(cells), "pickup_scan": True, "launch_count": len(launches), "hive_changes": [
                    {"time": round(time, 3), "alliance": config.zones[zone].alliance,
                     "reference_calibrated": zone in references} for time, zone in flips],
                "warnings": warnings}
    finally:
        cap.release()
