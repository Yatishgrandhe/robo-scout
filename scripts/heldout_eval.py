"""Run the app on a held-out recording with every labeled robot, then compare per-team counts with the labels.

Seeds each robot from its first reviewer-named track in data/match-labels/<video>/, starts the analysis
through the running server, waits for it, and prints shots and Hive entries per team: app vs. labels.

    python -m scripts.heldout_eval <recording id> <video id>
"""
import argparse
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = "http://127.0.0.1:8765"


def call(path, body=None):
    request = urllib.request.Request(SERVER + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def seeds(video):
    labels = ROOT / "data/match-labels" / video
    tracks = json.loads((labels / "tracks.json").read_text())["tracks"]
    named = {}
    for path in sorted(labels.glob("answers-*.json")):
        named.update({str(int(k)): v for k, v in json.loads(path.read_text()).get("tracks", {}).items() if v})
    first = {}
    for key, team in named.items():
        if key in tracks and tracks[key]:
            time_, box = tracks[key][len(tracks[key]) // 4]
            if str(team) not in first or time_ < first[str(team)][0]:
                first[str(team)] = (time_, box)
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match")
    parser.add_argument("video")
    args = parser.parse_args()
    meta = json.loads((ROOT / "data/matches" / args.match / "meta.json").read_text())
    config = dict(meta["config"])
    rosters = json.loads((ROOT / "data/match-labels/teams.json").read_text())[args.video]
    alliance = {team: side for side, teams in rosters.items() if side in ("red", "blue") for team in teams}
    found = seeds(args.video)
    config["robots"] = [{"team": team, "alliance": alliance.get(team, "red"), "robot": box, "seed_time": round(t, 2)}
                        for team, (t, box) in sorted(found.items()) if team != config["team"] and team in alliance][:5]
    print("primary", config["team"], "others", [(r["team"], r["seed_time"]) for r in config["robots"]])
    call(f"/api/matches/{args.match}/analyze", config)
    while True:
        time.sleep(15)
        state = json.loads((ROOT / "data/matches" / args.match / "meta.json").read_text())
        if state.get("state") != "processing":
            break
        print(f"  {state.get('stage')} {round(100 * (state.get('progress') or 0))}%", flush=True)
    # result.json holds this run's own detections; events.json also keeps earlier review decisions.
    events = json.loads((ROOT / "data/matches" / args.match / "result.json").read_text())["events"]
    app = Counter((event.get("team", config["team"]), "shot" if event["kind"].endswith("launch") else event["kind"])
                  for event in events)
    truth = json.loads((ROOT / "data/match-labels" / args.video / "stats.json").read_text())["teams"]
    print(f"\n{'team':>7} {'app shots':>9} {'label shots':>11} {'label entered':>13}  other app events")
    for team in sorted(set(alliance) | set(truth)):
        other = {kind: n for (t, kind), n in app.items() if t == team and kind != "shot"}
        label = truth.get(team, {})
        print(f"{team:>7} {app[(team, 'shot')]:>9} {label.get('shots', 0):>11} {label.get('entered', 0):>13}  {other}")
    print(f"\nOpen it: {SERVER}/?match={args.match}")


if __name__ == "__main__":
    main()
