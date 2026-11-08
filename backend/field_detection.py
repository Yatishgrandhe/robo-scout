"""Suggest visible BIOBUZZ Cell envelopes in a fixed broadcast view."""
import cv2 as cv
import numpy as np


def read_at(cap, fps, time):
    cap.set(cv.CAP_PROP_POS_FRAMES, round(time * fps))
    ok, frame = cap.read()
    return frame if ok else None


def roof_candidates(hsv, color):
    h, w = hsv.shape[:2]
    ranges = ((0, 13), (167, 179)) if color == "red" else ((92, 137),)
    mask = np.zeros((h, w), np.uint8)
    for lo, hi in ranges:
        mask |= cv.inRange(hsv, (lo, 110, 60), (hi, 255, 255))
    mask[:round(h * .055)] = 0
    mask[round(h * .42):] = 0
    mask[:, :round(w * .27)] = 0
    mask[:, round(w * .73):] = 0
    lines = cv.HoughLinesP(mask, 1, np.pi / 180, max(16, w // 67),
                           minLineLength=round(w * .023), maxLineGap=round(w * .011))
    if lines is None:
        return []
    sides = {False: [], True: []}
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        dx = x1 - x0
        if dx < .022 * w:
            continue
        slope = (y1 - y0) / dx
        if .33 < abs(slope) < .85:
            sides[slope > 0].append((x0, y0, x1, y1, slope))
    roofs = []
    for x0, y0, x1, y1, m in sides[False]:
        for x2, y2, x3, y3, n in sides[True]:
            x = (y2 - n * x2 - y0 + m * x0) / (m - n)
            y = y0 + m * (x - x0)
            width = x3 - x0
            if (not .057 * w <= width <= .152 * w or not .30 * w < x < .70 * w
                    or not .06 * h < y < .31 * h or abs(x - x1) > .025 * w
                    or abs(x - x2) > .025 * w or not .31 < (x - x0) / width < .69
                    or abs(y0 - y3) > .031 * h):
                continue
            score = width / w - .40 * y / h - abs(x - x1) / w - abs(x - x2) / w
            roofs.append((score, float(x0), float(y), float(x3), float(width)))
    return sorted(roofs, reverse=True)[:12]


def scene_signature(frame):
    h, w = frame.shape[:2]
    area = frame[round(.09 * h):round(.72 * h), round(.31 * w):round(.69 * w)]
    gray = cv.cvtColor(cv.resize(area, (88, 72), interpolation=cv.INTER_AREA), cv.COLOR_BGR2GRAY)
    return gray


def detect_field(cap, fps, config, duration):
    # A median of the opening seconds keeps fixed structure edges and removes
    # moving people and game pieces. Require a matching pair before suggesting.
    base = config.seed_time
    views = [read_at(cap, fps, min(duration, base + delta)) for delta in (0, .6, 1.2, 2.0, 3.0)]
    views = [frame for frame in views if frame is not None]
    if len(views) < 3:
        return None
    reference = np.median(views, axis=0).astype(np.uint8)
    hsv = cv.cvtColor(reference, cv.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    pairs = []
    for red in roof_candidates(hsv, "red"):
        for blue in roof_candidates(hsv, "blue"):
            left, right = sorted((red, blue), key=lambda roof: roof[1])
            if (abs(red[2] - blue[2]) < .05 * h and .63 < red[4] / blue[4] < 1.60
                    and -.018 * w < right[1] - left[3] < .063 * w):
                pairs.append((red[0] + blue[0], red, blue))
    if not pairs:
        return None
    _, red, blue = max(pairs)
    zones = []
    for color, roof in (("red", red), ("blue", blue)):
        _, left, top, right, width = roof
        box = ((left + width * .04) / w, (top + width * .28) / h,
               (right - width * .04) / w, (top + width * .94) / h)
        if box[3] > 1:
            return None
        zones.append({"kind": "cell", "alliance": color, "box": box, "automatic": True})
    end = config.end_time if config.end_time is not None else duration
    signature = scene_signature(reference)
    for back in np.arange(0, min(12, end - base), .25):
        frame = read_at(cap, fps, end - back)
        if frame is None:
            continue
        similarity = float(cv.matchTemplate(scene_signature(frame), signature, cv.TM_CCOEFF_NORMED)[0, 0])
        if similarity >= .57:
            if config.end_time is not None or back >= .75 and end - back - base >= 140:
                return zones, float(round(end - back, 3))
            return zones, None
    return zones, config.end_time
