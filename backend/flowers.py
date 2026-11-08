"""BIOBUZZ TU08 Flower evidence from a calibrated side view of the scoring volume."""

from dataclasses import dataclass

from .rules import NECTAR_COLORS, event

FLOWER_OPEN_OFFSET = 98  # 30 seconds AUTO, 8 transition, first 60 of TELEOP.
MAX_FLOWER_GAP = .32
BALL_HUES = {"pollen": [(19, 40)], "red_nectar": [(0, 9), (170, 179)], "blue_nectar": [(91, 132)]}


def merge_flower_detections(observations, proposals):
    """Keep established boxes and add distinct crop detections."""
    import numpy as np

    accepted = []
    for proposal in proposals:
        _, box, _ = proposal
        point = (box[:2] + box[2:]) / 2
        duplicate = False
        for _, existing, _ in (*observations, *accepted):
            midpoint = (existing[:2] + existing[2:]) / 2
            proximity = max(.004, .65 * max(box[2] - box[0], existing[2] - existing[0]))
            intersection = np.prod(np.maximum(0, np.minimum(box[2:], existing[2:]) - np.maximum(box[:2], existing[:2])))
            smaller = min(np.prod(box[2:] - box[:2]), np.prod(existing[2:] - existing[:2]))
            if np.linalg.norm(point - midpoint) < proximity or (smaller > 0 and intersection >= .65 * smaller):
                duplicate = True
                break
        if not duplicate:
            accepted.append(proposal)
    return accepted + observations


def trained_flower_detections(frame, zones, model, device):
    """Run the optional local model on Flower crops and keep color-supported proposals."""
    import cv2 as cv
    import numpy as np

    height, width = frame.shape[:2]
    scale = np.array((width, height, width, height))
    crops, origins = [], []
    for zone in zones:
        if zone.kind != "flower":
            continue
        x0, y0, x1, y1 = np.rint(np.asarray(zone.box) * scale).astype(int)
        bore = x1 - x0
        if bore < 9 or y1 - y0 < 2 * bore:
            continue
        left, top = max(0, x0 - bore), max(0, y0 - bore)
        crop = frame[top:min(height, y1 + bore), left:min(width, x1 + bore)]
        crops.append(crop)
        origins.append((left, top, x0, x1, bore))
    if not crops:
        return []
    predictions = model.predict(crops, imgsz=320, conf=.025, max_det=32, device=device, verbose=False)
    detections = []
    for prediction, crop, (left, top, inner_left, inner_right, bore) in zip(predictions, crops, origins):
        hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
        for *coordinates, score, cls in prediction.boxes.data.cpu().numpy():
            kind = prediction.names[int(cls)]
            if kind not in BALL_HUES or (kind == "pollen" and score < .25):
                continue
            x0, y0, x1, y1 = np.rint(coordinates).astype(int)
            if min(x1 - x0, y1 - y0) < 3 or max(x1 - x0, y1 - y0) > 1.2 * bore:
                continue
            pad = 0 if score < .10 else .25 * bore
            if not inner_left - pad <= left + (x0 + x1) / 2 <= inner_right + pad:
                continue
            patch = hsv[max(0, y0):y1, max(0, x0):x1]
            if not patch.size:
                continue
            mask = np.zeros(patch.shape[:2], np.uint8)
            for lo, hi in BALL_HUES[kind]:
                mask |= cv.inRange(patch, (lo, 65, 65), (hi, 255, 255))
            # A dim Nectar candidate needs stronger color evidence from the same crop.
            color_support = .40 if score < .25 else .20
            if np.count_nonzero(mask) < color_support * mask.size:
                continue
            box = np.clip((np.array(coordinates) + (left, top, left, top)) / scale, 0, 1)
            detections.append((kind, box, float(score)))
    return detections


def flower_detections(frame, zones):
    """Find separate color peaks in tight tube crops, including touching balls in a stack."""
    import cv2 as cv
    import numpy as np

    height, width = frame.shape[:2]
    detections = []
    for zone in zones:
        if zone.kind != "flower":
            continue
        x0, y0, x1, y1 = np.rint(np.asarray(zone.box) * (width, height, width, height)).astype(int)
        bore = x1 - x0
        if bore < 9 or y1 - y0 < 2 * bore:
            continue
        left, right = max(0, x0 - bore // 7), min(width, x1 + bore // 7)
        top, bottom = max(0, y0 - bore), min(height, y1 + bore // 2)
        hsv = cv.cvtColor(frame[top:bottom, left:right], cv.COLOR_BGR2HSV)
        for kind, hues in BALL_HUES.items():
            radius = bore * (.35 if kind == "pollen" else .45)
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in hues:
                mask |= cv.inRange(hsv, (lo, 105, 85), (hi, 255, 255))
            distance = cv.distanceTransform(mask, cv.DIST_L2, 5)
            extent = 2 * max(1, round(radius * .8)) + 1
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (extent, extent))
            peaks = ((distance >= max(1.8, radius * .51)) & (distance == cv.dilate(distance, kernel))).astype(np.uint8)
            _, _, _, centers = cv.connectedComponentsWithStats(peaks)
            accepted = []
            for x, y in sorted(centers[1:], key=lambda point: -distance[round(point[1]), round(point[0])]):
                if any(abs(y - prior_y) < 1.3 * radius for _, prior_y in accepted):
                    continue
                accepted.append((x, y))
            _, labels, components, _ = cv.connectedComponentsWithStats(mask)
            for label, (x, y, w, h, _) in enumerate(components[1:], 1):
                # Touching balls can produce a single wide peak. Estimate separate centers along a narrow stack.
                count = round(h / (2 * radius))
                if count < 2 or not radius <= w <= 2.5 * radius:
                    continue
                current = [(cx, cy) for cx, cy in accepted if labels[round(cy), round(cx)] == label]
                if not current or len(current) >= count:
                    continue
                accepted = [point for point in accepted if point not in current]
                for i in range(count):
                    row = min(y + h - 1, round(y + (i + .5) * h / count))
                    columns = np.flatnonzero(labels[row] == label)
                    if len(columns) >= radius * .55:
                        accepted.append((float(np.median(columns)), row))
            for x, y in accepted:
                box = np.clip(np.array((left + x - radius, top + y - radius, left + x + radius, top + y + radius))
                              / (width, height, width, height), 0, 1)
                detections.append((kind, box, .46))
    return detections


def flower_overlap(ball, volume):
    """Check projected ball/volume overlap using the ellipse inscribed in the detection box."""
    x0, y0, x1, y1 = ball
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    if rx <= 0 or ry <= 0:
        return False
    cx, cy = x0 + rx, y0 + ry
    dx = max(volume[0] - cx, 0, cx - volume[2]) / rx
    dy = max(volume[1] - cy, 0, cy - volume[3]) / ry
    return dx * dx + dy * dy < 1


def flower_at(ball, volumes):
    cx, cy = (ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2
    return min((index for index, box in volumes.items() if flower_overlap(ball, box)), default=None,
               key=lambda index: (cx - (volumes[index][0] + volumes[index][2]) / 2) ** 2
               + (cy - (volumes[index][1] + volumes[index][3]) / 2) ** 2)


def from_top(previous, previous_volume, current, volume):
    height = previous[3] - previous[1]
    before, after = previous[3] - previous_volume[1], current[3] - volume[1]
    if before > .15 * height or after - before < .20 * height:
        return False
    fraction = max(0, min(1, -before / (after - before)))
    previous_x = ((previous[0] + previous[2]) / 2 - previous_volume[0]) / (previous_volume[2] - previous_volume[0])
    current_x = ((current[0] + current[2]) / 2 - volume[0]) / (volume[2] - volume[0])
    return 0 <= previous_x + fraction * (current_x - previous_x) <= 1


@dataclass
class FlowerVisit:
    zone: int
    entered: float
    seen: int
    top: bool
    actor: str | None


def observe_flower(track, now, box, volumes, primary):
    index = flower_at(box, volumes)
    prior = track.samples[-1] if track.samples else None
    continuous = prior is not None and now - prior[0] <= MAX_FLOWER_GAP
    if index is None:
        track.flower_visit = None
    elif continuous and track.flower_visit and track.flower_visit.zone == index:
        track.flower_visit.seen += 1
    else:
        top = continuous and index in track.flower_volumes and from_top(prior[1], track.flower_volumes[index], box, volumes[index])
        actor = None
        if top:
            nearby = [sample[2] for sample in track.samples[-9:] if now - sample[0] <= .55 and sample[2]]
            if len(nearby) >= 2 and len(set(nearby)) == 1:
                actor = primary if nearby[0] is True else nearby[0]
            if track.release is not None and 0 <= now - track.release <= 1.4:
                release_team = track.release_team if track.release_team is not None else primary
                actor = release_team if actor is None or release_team == actor else None
        track.flower_visit = FlowerVisit(index, now, 1, bool(top), actor)
    track.flower_volumes = volumes


def early_flower_nectar(track, now, start_time, flagged):
    visit = track.flower_visit
    if (start_time is None or track.kind not in NECTAR_COLORS or visit is None or not visit.top
            or visit.actor is None or visit.seen < 2 or track.samples[-1][0] != now
            or not start_time <= visit.entered < start_time + FLOWER_OPEN_OFFSET):
        return None
    key = track.id, visit.zone
    if key in flagged:
        return None
    flagged.add(key)
    row = event("penalty", visit.entered, "Possible early Nectar in Flower", "G410", severity="major", source="vision",
                evidence="Tracked Nectar crossed the top into the Flower volume. Check the placement and the primary field timer.")
    return {**row, "team": visit.actor} if visit.actor else row


def nectar_at_edge(tracks, top):
    nectar = [(track.kind.split("_")[0], track.samples[-1][1]) for track in tracks if track.kind in NECTAR_COLORS]
    nectar.sort(key=lambda entry: (entry[1][1] + entry[1][3]) / 2, reverse=not top)
    if not nectar:
        return None
    color, box = nectar[0]
    y = (box[1] + box[3]) / 2
    for other, candidate in nectar[1:]:
        if other != color and abs(y - (candidate[1] + candidate[3]) / 2) <= .4 * min(box[3] - box[1], candidate[3] - candidate[1]):
            return None
    return color


def settled_in_flower(track, final_time, max_stale):
    visit = track.flower_visit
    return (visit is not None and visit.seen >= 3 and final_time - visit.entered >= .4
            and final_time - track.samples[-1][0] <= max_stale)


def final_flowers(tracks, zones, final_time, max_stale, roster):
    end, events = [], []
    for index, zone in enumerate(zones):
        if zone.kind != "flower":
            continue
        contents = [track for track in tracks if settled_in_flower(track, final_time, max_stale)
                    and track.flower_visit.zone == index]
        owner = nectar_at_edge(contents, top=True)
        bottom = nectar_at_edge(contents, top=False)
        pieces = {kind: sum(track.kind == kind for track in contents) for kind in ("pollen", "red_nectar", "blue_nectar")}
        end.append({"zone": index, "owner": owner, "bottom_nectar": bottom, "pieces": pieces})
        for track in contents:
            visit = track.flower_visit
            if not visit.top or visit.actor not in roster:
                continue
            label = track.kind.replace("_", " ").title()
            credit = f"The topmost visible Nectar suggests {owner} owns this Flower." if owner else "Flower ownership is unclear from this view."
            row = event("flower_ball", final_time, f"Possible final {label} in Flower", "§10.5.2", piece=track.kind, source="vision",
                        evidence=f"The track crossed the top and remained partly in the marked volume after settling. {credit} Check depth and attribution.")
            events.append({**row, "team": visit.actor, "flower_zone": index, "flower_owner": owner})
    return end, events
