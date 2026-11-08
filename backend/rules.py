"""BIOBUZZ TU08 game-piece and referee-call accounting."""
import re
from uuid import uuid4

MANUAL = "https://ftc-resources.firstinspires.org/ftc/archive/2027/game/manual"
BALL_EVENTS = {"cell_ball", "flower_ball", "garden_ball"}
PIECE_EVENTS = BALL_EVENTS | {"cell_entry"}
SHOT_EVENTS = {"pollen_launch": "pollen", "red_nectar_launch": "red_nectar",
               "blue_nectar_launch": "blue_nectar", "nectar_launch": "nectar"}
REVIEW_EVENTS = PIECE_EVENTS | SHOT_EVENTS.keys() | {"nectar_pickup", "penalty", "note"}
EVENT_KINDS = REVIEW_EVENTS | {"hive_tip", "bottom_nectar", "leave", "park"}
FOUL_POINTS = {"warning": 0, "minor": 5, "major": 20}
NECTAR_COLORS = {"red_nectar", "blue_nectar"}


def event(kind, time, title, rule, *, status="suggested", count=1, severity=None,
          evidence=None, source="demo", card=None, piece=None):
    return {"id": uuid4().hex, "kind": kind, "time": round(time, 3), "title": title, "rule": rule,
            "status": status, "count": count, "severity": severity, "source": source,
            "card": card, "evidence": evidence or "", **({"piece": piece} if piece else {})}


def event_piece(row):
    if row.get("piece"):
        return row["piece"]
    if row["kind"] in SHOT_EVENTS:
        return SHOT_EVENTS[row["kind"]]
    if row["kind"] in BALL_EVENTS:
        # Older reviews stored the piece in the title when it was known.
        title = row["title"].casefold()
        if re.search(r"\bpollen\b", title) and not re.search(r"\bnectar\b", title):
            return "pollen"
        if re.search(r"\bnectar\b", title) and not re.search(r"\bpollen\b", title):
            for color in ("red", "blue"):
                if re.search(rf"\b{color}\s+nectar\b", title):
                    return f"{color}_nectar"
            return "nectar"
    return None


def penalty_points(row):
    return FOUL_POINTS.get(row.get("severity"), 0) * row["count"]


def counts(rows, alliance):
    tally = {"pollen": {"shot": 0, "entered": 0, "scored": 0}, "nectar": {"shot": 0, "entered": 0, "scored": 0},
             "unclassified": 0, "pickups": {"own": 0, "opponent": 0, "unassigned": 0},
             "penalty_points": 0, "warnings": 0, "cards": 0}
    for row in rows:
        kind, count = row["kind"], row["count"]
        piece = event_piece(row)
        if kind in SHOT_EVENTS:
            tally["pollen" if piece == "pollen" else "nectar"]["shot"] += count
        elif kind == "cell_entry" and piece is not None:
            tally["pollen" if piece == "pollen" else "nectar"]["entered"] += count
        elif kind in BALL_EVENTS:
            if piece is None:
                tally["unclassified"] += count
            else:
                tally["pollen" if piece == "pollen" else "nectar"]["scored"] += count
        elif kind == "nectar_pickup":
            color = "unassigned" if alliance is None else "own" if piece == f"{alliance}_nectar" else "opponent"
            tally["pickups"][color] += count
        elif kind == "penalty":
            tally["penalty_points"] += penalty_points(row)
            tally["warnings"] += count if row["severity"] == "warning" else 0
            tally["cards"] += int(bool(row.get("card")))
    return tally


def summarize(events, alliance=None):
    confirmed = [row for row in events if row["status"] == "confirmed"]
    suggested = [row for row in events if row["status"] == "suggested" and row["kind"] in REVIEW_EVENTS]
    return {"confirmed": counts(confirmed, alliance), "suggested": counts(suggested, alliance),
            "pending": len(suggested)}
