"""Usage: python scripts/tune_flights.py <cache dir> [sweep]

Score detector flights against reviewed launches in every cached video and sweep linker settings."""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backend.ball_flights as bf
R = Path(__file__).resolve().parents[1]
DET = Path(sys.argv[1]); VAL = {"Y6uT5ziHRJY", "2EDHhlj1vQk", "kwYU2RU7V14"}
TOL = .35

def team_boxes(video):
    labels = R / "data/match-labels" / video
    tracks = json.loads((labels / "tracks.json").read_text())["tracks"]
    names = {}
    for p in sorted(labels.glob("answers-*.json")):
        names.update({str(int(k)): str(v).strip().lstrip("0") for k, v in json.loads(p.read_text()).get("tracks", {}).items() if v})
    out = {}
    for k, team in names.items():
        for t, box in tracks.get(k, []):
            out.setdefault(team, []).append((t, box))
    return {t: sorted(v) for t, v in out.items()}

def credit(flight, boxes):
    t, (x, y) = flight["time"], flight["release_norm"]
    best = None
    for team, rows in boxes.items():
        times = [r[0] for r in rows]
        i = int(np.searchsorted(times, t))
        for j in (i - 1, i):
            if 0 <= j < len(rows) and abs(rows[j][0] - t) < .3:
                x0, y0, x1, y1 = rows[j][1]
                d = np.hypot(max(x0 - x, 0, x - x1), max(y0 - .03 - y, 0, y - y1)) / max(.01, x1 - x0)
                if best is None or d < best[0]: best = (d, team)
    return best[1] if best and best[0] <= 1.0 else None

data = {}
for f in sorted(DET.glob("*.json")):
    v = f.stem
    stats = json.loads((R / "data/match-labels" / v / "stats.json").read_text())["launches"]
    pos = [(l["time"], l.get("team")) for l in stats if l.get("launch")]
    neg = [l["time"] for l in stats if l.get("launch") is False]
    frames = [(t, [tuple(d) for d in ds]) for t, ds in json.loads(f.read_text())]
    data[v] = (frames, pos, neg, team_boxes(v))
print("videos", len(data), "val", sorted(VAL & data.keys()))

def score(params, verbose=False):
    for k, val in params.items(): setattr(bf, k, val)
    tot = {"train": [0, 0, 0, 0, 0, 0], "val": [0, 0, 0, 0, 0, 0]}  # tp, labeled, known_fp, found, team_ok, team_checked
    for v, (frames, pos, neg, boxes) in data.items():
        flights = bf.link_flights(frames)
        used, tp, fp, tok, tchk = set(), 0, 0, 0, 0
        for fl in flights:
            m = [i for i, (t, team) in enumerate(pos) if i not in used and abs(t - fl["time"]) < TOL]
            if m:
                i = min(m, key=lambda i: abs(pos[i][0] - fl["time"])); used.add(i); tp += 1
                if pos[i][1]:
                    tchk += 1; tok += credit(fl, boxes) == pos[i][1]
            elif any(abs(t - fl["time"]) < TOL for t in neg):
                fp += 1
        s = tot["val" if v in VAL else "train"]
        for k, val in enumerate((tp, len(pos), fp, len(flights), tok, tchk)): s[k] += val
        if verbose: print(f"  {v:12s} labeled {len(pos):3d} found {len(flights):3d} hit {tp:3d} known-false {fp:3d} team {tok}/{tchk}")
    return tot

base = dict(MIN_INLIERS=bf.MIN_INLIERS, MIN_RISE=bf.MIN_RISE, MIN_SPEED=bf.MIN_SPEED, MAX_RESIDUAL=bf.MAX_RESIDUAL, LOWEST_START=bf.LOWEST_START, MAX_GAP=.1, STATIC=0)
def show(name, t):
    for split, (tp, n, fp, found, tok, tchk) in t.items():
        print(f"{name} {split}: recall {tp}/{n}={tp/max(1,n):.2f} found {found} known-false {fp} precision~{tp/max(1,tp+fp):.2f} team {tok}/{tchk}={tok/max(1,tchk):.2f}")
show("current", score({**base, "GATE": .035}))
if len(sys.argv) > 2:
    results = []
    for rise, speed, inl, gap, gate in itertools.product((.03, .05), (.4,), (4,), (.15,), (.07, .1, .14)):
        p = dict(MIN_INLIERS=inl, MIN_RISE=rise, MIN_SPEED=speed, MAX_RESIDUAL=.013, LOWEST_START=.15, MAX_GAP=gap, STATIC=.006, GATE=gate)
        t = score(p)["train"]
        f1 = 2 * t[0] / max(1, t[1] + t[0] + t[2])  # recall vs labels, penalize known-false
        results.append((f1, p, t))
    results.sort(key=lambda r: -r[0])
    for f1, p, t in results[:5]: print(round(f1, 3), p, t)
    best = results[0][1]; print("BEST", best); show("best", score(best, verbose=True))
