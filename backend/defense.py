"""Estimate sustained opponent pressure from jointly visible robot tracks."""

LANE_CREDIT = .5


def pressure_rating(pressure_seconds, observed_seconds, coverage):
    if observed_seconds < 10 or coverage < .35:
        return None
    return round(100 * min(1, max(0, pressure_seconds / observed_seconds)), 1)


def defense_rating(evidence):
    return pressure_rating(evidence["pressure_seconds"] + LANE_CREDIT * evidence.get("lane_seconds", 0),
                           evidence["observed_seconds"], evidence["coverage"])


def add_lane_defense(results, frames, alliances, shape, start_time, transforms=None):
    """Look for a defender in an opponent's approach when that opponent slows or turns."""
    import numpy as np

    if len(frames) < 4 or not results:
        return
    times = np.asarray([frame["time"] for frame in frames])
    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        return
    stride = max(1, round(.45 / dt))
    minimum = max(2, round(.32 / dt))
    if len(frames) <= 2 * stride:
        return
    before = np.arange(stride, len(frames) - stride)
    earlier, after = before - stride, before + stride
    continuous = np.r_[0, np.cumsum(np.diff(times) > 1.65 * dt)]
    eligible = continuous[earlier] == continuous[after]
    if start_time is not None:
        eligible &= (times[earlier] >= start_time + 38) & (times[after] <= start_time + 158)
    positions, widths, visible_runs = {}, {}, {}
    for team in alliances:
        boxes = np.asarray([(frame.get("robots") or {}).get(team) or [np.nan] * 4 for frame in frames]) * (*shape, *shape)
        feet = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
        if transforms is not None and len(transforms):
            affine = np.asarray(transforms)
            feet = np.einsum("nij,nj->ni", affine[:, :, :2], feet) + affine[:, :, 2]
        positions[team] = feet
        widths[team] = boxes[:, 2] - boxes[:, 0]
        visible_runs[team] = np.r_[0, np.cumsum([
            not np.isfinite(feet[i]).all() or team in frame.get("reanchored", ()) for i, frame in enumerate(frames)
        ])]

    for team, evidence in results.items():
        lanes = []
        active = np.zeros(len(times), bool)
        direct = np.zeros(len(times), bool)
        for episode in evidence["episodes"]:
            direct |= (times >= episode["start"]) & (times < episode["end"] - .001)
        for other, alliance in alliances.items():
            if alliance == alliances.get(team) or team not in positions:
                continue
            own, rival = positions[team], positions[other]
            width = (widths[team][before] + widths[other][before]) / 2
            incoming = (rival[before] - rival[earlier]) / width[:, None]
            outgoing = (rival[after] - rival[before]) / width[:, None]
            speed = np.linalg.norm(incoming, axis=1)
            heading = incoming / np.maximum(speed[:, None], .001)
            offset = (own[before] - rival[before]) / width[:, None]
            along = (offset * heading).sum(axis=1)
            sideways = np.linalg.norm(offset - heading * along[:, None], axis=1)
            advance = (outgoing * heading).sum(axis=1)
            retreat = np.linalg.norm(own[after] - rival[after], axis=1) / width
            own_motion = np.linalg.norm(own[after] - own[earlier], axis=1) / width
            own_run, rival_run = visible_runs[team], visible_runs[other]
            clear = ((own_run[earlier] == own_run[after + 1]) & (rival_run[earlier] == rival_run[after + 1]))
            blocks = (eligible & clear & (speed >= .30) & (speed <= 2.1) & (along >= .60) & (along <= 1.85)
                      & (sideways <= .42) & (advance <= .65 * speed) & (retreat <= 2.25))
            edges = np.flatnonzero(np.diff(np.r_[False, blocks, False]))
            for first, last in zip(edges[::2], edges[1::2]):
                if last - first < minimum:
                    continue
                start, end = before[first], after[last - 1]
                if np.all(direct[start:end + 1]):
                    continue
                active[start:end + 1] = True
                action = "lane_hold" if np.median(own_motion[first:last]) <= .45 else "lane_block"
                lanes.append({"start": float(times[start]), "end": round(float(times[end] + dt), 3),
                              "opponent": other, "kind": action})
        evidence["lane_episodes"] = sorted(lanes, key=lambda episode: episode["start"])
        evidence["lane_seconds"] = round(float(np.count_nonzero(active & ~direct)) * dt, 1)
        evidence["rating"] = defense_rating(evidence)


def defensive_ratings(frames, roster, shape, transforms, start_time):
    import numpy as np

    if len(roster) < 2 or len(frames) < 3:
        return {}
    times = np.array([frame["time"] for frame in frames])
    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        return {}
    baseline = max(2, round(.5 / dt))
    hold = max(2, round(.6 / dt))
    first = (start_time + 38) if start_time is not None else times[0]
    live = times >= first
    positions, widths = {}, {}
    resets = {}
    for team in roster:
        boxes = np.array([frame["robots"][team] or [np.nan] * 4 for frame in frames]) * (*shape, *shape)
        ground = np.stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]), axis=1)
        if transforms:
            affine = np.asarray(transforms)
            ground = np.einsum("nij,nj->ni", affine[:, :, :2], ground) + affine[:, :, 2]
        positions[team] = ground
        widths[team] = boxes[:, 2] - boxes[:, 0]
        resets[team] = np.r_[0, np.cumsum([team in frame.get("reanchored", ()) for frame in frames])]
    results = {}
    for team, seed in roster.items():
        visible = np.zeros(len(times), bool)
        pressure = np.zeros(len(times), bool)
        peers = np.full(len(times), "", dtype=object)
        for other, opponent in roster.items():
            if seed.alliance == opponent.alliance:
                continue
            own, rival = positions[team], positions[other]
            width = (widths[team] + widths[other]) / 2
            distance = np.linalg.norm(own - rival, axis=1) / width
            visible |= np.isfinite(distance) & live
            trigger = -100
            for i in range(2 * baseline, len(times)):
                j, k = i - baseline, i - 2 * baseline
                if (not live[i] or not np.isfinite(distance[i]) or distance[i] > 1.14
                        or resets[team][i + 1] != resets[team][k + 1]
                        or resets[other][i + 1] != resets[other][k + 1]):
                    trigger = -100
                    continue
                rival_before = np.linalg.norm(rival[j] - rival[k]) / width[i]
                rival_after = np.linalg.norm(rival[i] - rival[j]) / width[i]
                defender_motion = own[i] - own[j]
                toward = rival[j] - own[j]
                advance = np.dot(defender_motion, toward) / max(1, np.linalg.norm(toward)) / width[i]
                if rival_before > .35 and rival_after < .60 * rival_before and advance > .18 and distance[j] > distance[i] + .16:
                    trigger = i
                if i - trigger <= max(hold, round(1.8 / dt)) and rival_after <= .40:
                    pressure[i] = True
                    peers[i] = other
        edges = np.flatnonzero(np.diff(np.r_[False, pressure, False]))
        episodes = [{"start": float(times[a]), "end": round(float(times[b - 1] + dt), 3),
                     "opponent": str(peers[a])}
                    for a, b in zip(edges[::2], edges[1::2]) if b - a >= hold]
        observed = float(visible.sum()) * dt
        coverage = round(observed / max(dt, float(live.sum()) * dt), 3)
        seconds = round(sum(item["end"] - item["start"] for item in episodes), 1)
        observed = round(observed, 1)
        results[team] = {"rating": pressure_rating(seconds, observed, coverage),
                         "pressure_seconds": seconds, "observed_seconds": observed,
                         "coverage": coverage, "episodes": episodes}
    add_lane_defense(results, frames, {team: seed.alliance for team, seed in roster.items()}, shape, start_time, transforms)
    return results
