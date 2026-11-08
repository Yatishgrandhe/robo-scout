"""Refine team tracks between human-marked frames in both time directions."""

import cv2 as cv
import numpy as np


def overlap(left, right):
    common = np.maximum(0, np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2])).prod()
    return common / max(1e-8, np.prod(left[2:] - left[:2]) + np.prod(right[2:] - right[:2]) - common)


def sampled_frames(cap, indices, shape, reverse):
    # Decode batches in display order, then walk the batch backwards. Keeping
    # the full match in memory would take several GB.
    batch = 32
    starts = range(0, len(indices), batch)
    for start in reversed(starts) if reverse else starts:
        pictures = []
        frame_index = indices[start]
        cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_index))
        for index in range(start, min(start + batch, len(indices))):
            while frame_index < indices[index]:
                if not cap.grab():
                    return
                frame_index += 1
            ok, image = cap.read()
            frame_index += 1
            if not ok:
                return
            pictures.append((index, cv.resize(image, shape, interpolation=cv.INTER_AREA)))
        yield from reversed(pictures) if reverse else pictures


def track_direction(video, times, shape, marks, params, field, reverse, progress):
    found = {team: {} for team in marks}
    trackers = {}
    scale = np.array((*shape, *shape))
    cap = cv.VideoCapture(str(video))
    try:
        indices = np.rint(times * cap.get(cv.CAP_PROP_FPS)).astype(int)
        for step, (index, image) in enumerate(sampled_frames(cap, indices, shape, reverse)):
            for team, anchors in marks.items():
                if index in anchors:
                    box = anchors[index]
                    x, y, right, bottom = np.rint(box * scale).astype(int)
                    tracker = cv.TrackerNano_create(params)
                    tracker.init(image, (int(x), int(y), int(right - x), int(bottom - y)))
                    trackers[team] = (tracker, index, box)
                    found[team][index] = (box, 1., 0.)
                elif team in trackers:
                    tracker, origin, anchor = trackers[team]
                    ok, (x, y, w, h) = tracker.update(image)
                    score = tracker.getTrackingScore()
                    if not ok or score < .64:
                        continue
                    box = np.clip(np.array([x, y, x + w, y + h]) / scale, 0, 1)
                    width, height = (box[2:] - box[:2]) / (anchor[2:] - anchor[:2])
                    center = (box[2:] + box[:2]) / 2
                    if (.38 <= width <= 2 and .38 <= height <= 2 and field[0] <= center[0] <= field[2]
                            and field[1] <= center[1] <= field[3]):
                        found[team][index] = (box, score, abs(times[index] - times[origin]))
            if step % 10 == 0:
                progress(step / len(times))
    finally:
        cap.release()
    return found


def merge_box(before, after, previous):
    if before is not None and after is not None:
        left, right = before[0], after[0]
        if overlap(left, right) >= .26:
            w1, w2 = (hit[1] / (.6 + hit[2]) for hit in (before, after))
            return (left * w1 + right * w2) / (w1 + w2)
        close = min((before, after), key=lambda hit: hit[2])
        if close[2] < .8:
            return close[0]
        if previous is not None:
            matches = [hit[0] for hit in (before, after) if overlap(hit[0], previous) >= .38]
            if len(matches) == 1:
                return matches[0]
        return None
    hit = before if before is not None else after
    if hit is None:
        return previous
    if hit[2] < 1.4 or previous is None or overlap(hit[0], previous) >= .22:
        return hit[0]
    return None


def refine_roster(video, frames, roster, anchors, shape, params, field, progress):
    times = np.array([frame["time"] for frame in frames])
    marks = {team: {int(np.argmin(abs(times - seed.seed_time))): np.asarray(seed.robot)}
             for team, seed in roster.items()}
    for anchor in anchors:
        if anchor.team in marks and times[0] <= anchor.time <= times[-1]:
            marks[anchor.team][int(np.argmin(abs(times - anchor.time)))] = np.asarray(anchor.box)
    if not any(len(items) > 1 for items in marks.values()):
        return
    forward = track_direction(video, times, shape, marks, params, field, False, lambda value: progress(value / 2))
    backward = track_direction(video, times, shape, marks, params, field, True, lambda value: progress(.5 + value / 2))
    primary = next(iter(roster))
    for i, frame in enumerate(frames):
        boxes = {}
        for team, old in frame["robots"].items():
            previous = np.asarray(old) if old else None
            boxes[team] = marks[team].get(i)
            if boxes[team] is None:
                if times[i] < roster[team].seed_time:
                    continue
                boxes[team] = merge_box(forward[team].get(i), backward[team].get(i), previous)
        frame["robots"] = {
            team: box.round(4).tolist() if box is not None and (i in marks[team] or not any(
                other != team and other_box is not None and overlap(box, other_box) > .49
                for other, other_box in boxes.items()
            )) else None
            for team, box in boxes.items()
        }
        if frame["robot"] != frame["robots"][primary]:
            frame["mask"] = None
        frame["robot"] = frame["robots"][primary]
