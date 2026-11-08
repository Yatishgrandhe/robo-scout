"""Estimate sustained opponent pressure from jointly visible robot tracks."""

FULL_RATING_PRESSURE_SHARE = .10


def pressure_rating(pressure_seconds, observed_seconds, coverage):
    if pressure_seconds <= 0 or observed_seconds < 10 or coverage < .35:
        return None
    target = observed_seconds * FULL_RATING_PRESSURE_SHARE
    return max(1, round(100 * min(1, pressure_seconds / target)))


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
    return results
