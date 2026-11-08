"""Scouting observations derived from confirmed match events."""

from itertools import groupby

from .rules import SHOT_EVENTS

QUICK_SWITCH_SECONDS = 12


def robot_comment(counts, entries, launches):
    activity = []
    for name, piece in (("Pollen", counts["pollen"]), ("Nectar", counts["nectar"])):
        if piece["shot"]:
            activity.append(f'{piece["shot"]} {name} shots' if piece["shot"] > 1 else f"1 {name} shot")
        if piece["entered"]:
            activity.append(f'{piece["entered"]} {name} Hive entries' if piece["entered"] > 1 else f"1 {name} Hive entry")
        elif piece["scored"]:
            activity.append(f'{piece["scored"]} {name} retained at match end')
    if not activity:
        return {"comment": "Start with a clear scoring attempt to establish how this robot sets up and releases.", "clip": None}

    activity_summary = activity[0]
    if len(activity) > 1:
        conjunction = ", and " if len(activity) > 2 else " and "
        activity_summary = ", ".join(activity[:-1]) + conjunction + activity[-1]
    comment = f"Reviewed footage shows {activity_summary}."
    quick_entries = next(((first, second) for first, second in zip(entries, entries[1:])
                          if first.get("launch_id") and second.get("launch_id")
                          and first["launch_id"] != second["launch_id"]
                          and 0 < second["time"] - first["time"] <= 2), None)
    clip = max(entries, key=lambda row: (row["count"], bool(row.get("launch_id"))), default=None)
    if counts["nectar"]["shot"] and not counts["nectar"]["entered"]:
        comment += " The Nectar attempts give you a separate shooting pattern to study. Follow the highlighted shot to check its landing."
        clip = next((row for row in launches if row["kind"] != "pollen_launch"), clip)
    elif clip and clip["count"] > 1:
        comment += f' The highlighted {clip["count"]}-piece volley gives you a scoring setup to look for on later attempts.'
    elif quick_entries:
        clip = quick_entries[0]
        comment += " The highlighted pair of entries shows how quickly this robot could reset between successful shots."
    elif clip:
        comment += " Use the highlighted entry to compare the approach and release with other attempts."
    elif launches:
        clip = launches[0]
        comment += " Follow the highlighted release to check its aim and where the piece lands."
    return {"comment": comment, "clip": {"time": clip["time"], "title": clip["title"]} if clip else None}


def alliance_fluidity(teams, entries, launches):
    totals = dict.fromkeys(teams, 0)
    for team, row in entries:
        totals[team] += row["count"]
    moments = [(time, {team for team, _ in rows}) for time, rows in groupby(entries, key=lambda item: item[1]["time"])]
    transitions = [(start, end, next(iter(before)), next(iter(after)))
                   for (start, before), (end, after) in zip(moments, moments[1:]) if len(before) == len(after) == 1]
    switches = [{"from_team": before, "to_team": after, "start": start, "end": end}
                for start, end, before, after in transitions if before != after and end - start <= QUICK_SWITCH_SECONDS]
    rating = None
    if len(transitions) >= 2 and all(totals[team] or launches[team] for team in teams):
        balance = len(teams) * min(totals.values()) / sum(totals.values())
        rating = round(50 * (balance + len(switches) / len(transitions)))
    return {"rating": rating, "entries": totals, "transitions": len(transitions), "switches": switches,
            "switch_window": QUICK_SWITCH_SECONDS}


def scouting_report(roster, primary, rows, summaries):
    entries = {team: [] for team in roster}
    launches = {team: [] for team in roster}
    alliance_entries = {"red": [], "blue": []}
    for row in rows:
        team = row.get("team", primary)
        if team not in roster or row["status"] != "confirmed":
            continue
        if row["kind"] == "cell_entry" and row["piece"]:
            entries[team].append(row)
            alliance = roster[team]
            if alliance in alliance_entries:
                alliance_entries[alliance].append((team, row))
        elif row["kind"] in SHOT_EVENTS:
            launches[team].append(row)
    alliances = {}
    for alliance in alliance_entries:
        teams = [team for team in roster if roster[team] == alliance]
        if len(teams) > 1:
            alliances[alliance] = alliance_fluidity(teams, alliance_entries[alliance], launches)
    return {"robots": {team: robot_comment(summaries[team]["confirmed"], entries[team], launches[team]) for team in roster},
            "alliances": alliances}
