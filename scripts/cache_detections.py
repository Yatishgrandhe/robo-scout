"""Cache in-flight detector output for every labeled video: python scripts/cache_detections.py <out dir>"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from backend.ball_flights import detect_frames
R = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1])
model = YOLO(R / "models/ball-flight.pt")
for stats in sorted((R / "data/match-labels").glob("*/stats.json")):
    video = stats.parent.name
    target = out / f"{video}.json"
    if target.exists():
        continue
    launches = [l["time"] for l in json.loads(stats.read_text())["launches"]]
    end = (max(launches) + 5) if launches else 200
    frames = detect_frames(R / "data/videos" / f"{video}.mp4", model, "mps", 0, end, stride=1)
    target.write_text(json.dumps(frames))
    print(video, len(frames), flush=True)
