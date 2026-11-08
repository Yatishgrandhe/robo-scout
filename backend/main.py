"""Loopback-only match review API and production web entry point."""
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import monotonic, sleep
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .rules import BALL_EVENTS, EVENT_KINDS, FOUL_POINTS, MANUAL, NECTAR_COLORS, PIECE_EVENTS, SHOT_EVENTS, event, event_piece, penalty_points, summarize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MATCHES = DATA / "matches"
MATCHES.mkdir(parents=True, exist_ok=True)
WEIGHTS = {"generic": "yolo26n.pt", "synthetic": "synthetic.pt", "ftc": "ftc.pt"}
BOX = tuple[float, float, float, float]
UPLOAD_LIMIT = 2 * 1024**3
ANALYSIS_IDLE_TIMEOUT = 120
app = FastAPI(title="Robo Scout")
logger = logging.getLogger("uvicorn.error")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
lock = Lock()
workers = ThreadPoolExecutor(max_workers=1)
active = None


def valid_box(value):
    x0, y0, x1, y1 = value
    if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1 or min(x1 - x0, y1 - y0) < .012:
        raise ValueError("Draw a rectangle inside the video, at least 1.2% wide and tall.")
    return value


class Zone(BaseModel):
    kind: Literal["hive", "cell", "garden", "flower"]
    alliance: Literal["red", "blue"]
    box: BOX
    state_frames: tuple[float, float] | None = None
    automatic: bool = False

    _box = field_validator("box")(valid_box)

    @field_validator("state_frames")
    @classmethod
    def check_state_frames(cls, value):
        if value is not None and (value[0] == value[1] or not all(0 <= time <= 3600 for time in value)):
            raise ValueError("Choose two different Hive reference times within the video.")
        return value


class Anchor(BaseModel):
    time: float = Field(ge=0, le=3600)
    box: BOX
    team: str = Field(default="", max_length=24)

    _box = field_validator("box")(valid_box)


class RobotSeed(BaseModel):
    team: str = Field(default="", max_length=24)
    alliance: Literal["red", "blue"] = "red"
    robot: BOX
    seed_time: float = Field(ge=0, le=3600)

    _box = field_validator("robot")(valid_box)


class Analysis(RobotSeed):
    start_time: float | None = Field(default=None, ge=0, le=3600)
    end_time: float | None = Field(default=None, ge=0, le=3600)
    zones: list[Zone] = Field(default_factory=list, max_length=32)
    field: BOX | None = None
    anchors: list[Anchor] = Field(default_factory=list, max_length=128)
    robots: list[RobotSeed] = Field(default_factory=list, max_length=5)
    model: Literal["generic", "synthetic", "ftc"] = "generic"
    sample_fps: Literal[6, 10, 15] = 10
    sam: bool = True
    camera_motion: bool = False
    scan_shots: bool = True

    auto_zones: bool = True

    @field_validator("field")
    @classmethod
    def check_field(cls, value):
        return valid_box(value) if value is not None else None


Piece = Literal["pollen", "red_nectar", "blue_nectar", "nectar"]


class ManualEvent(BaseModel):
    kind: str
    time: float = Field(ge=0, le=3600)
    title: str = Field(min_length=1, max_length=120)
    rule: str = Field(default="", max_length=32)
    count: int = Field(default=1, ge=1, le=60)
    severity: Literal["warning", "minor", "major"] | None = None
    card: Literal["yellow", "red"] | None = None
    piece: Piece | None = None
    team: str | None = Field(default=None, max_length=24)
    launch_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    evidence: str = Field(default="", max_length=400)


class Decision(BaseModel):
    status: Literal["suggested", "confirmed", "dismissed"]
    piece: Piece | None = None


class FolderName(BaseModel):
    name: str


class RecordingUpdate(BaseModel):
    name: str | None = None
    folder_id: str | None = None


def folder_for(match):
    if match == "demo":
        return DATA / "demo"
    if re.fullmatch(r"[0-9a-f]{32}", match) is None:
        raise HTTPException(404, "Match not found.")
    return MATCHES / match


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    tmp = path.with_name(f".{path.name}.{uuid4().hex}")
    try:
        tmp.write_text(json.dumps(payload, allow_nan=False))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def library_folders():
    try:
        return read(DATA / "library-folders.json")
    except FileNotFoundError:
        return {}


def library_name(value, limit):
    name = (value or "").strip()
    if not 1 <= len(name) <= limit:
        raise HTTPException(422, f"Enter a name between 1 and {limit} characters.")
    if re.search(r"[/\\\x00-\x1f\x7f]", name) or name in (".", ".."):
        raise HTTPException(422, "Use a name without slashes or control characters.")
    return name


def save_folder(folder_id, data, *, create=False):
    name = library_name(data.name, 80)
    with lock:
        folders = library_folders()
        if not create and folder_id not in folders:
            raise HTTPException(404, "Folder not found.")
        if name.casefold() in {"all recordings", "unfiled", "practice simulation"} or any(
            key != folder_id and other.casefold() == name.casefold() for key, other in folders.items()
        ):
            raise HTTPException(409, "A folder with that name already exists.")
        folders[folder_id] = name
        write(DATA / "library-folders.json", folders)
    return {"id": folder_id, "name": name}


def metadata(match):
    folder = folder_for(match)
    if match == "demo" and (folder / "result.json").is_file():
        return {"id": match, "name": "BIOBUZZ practice simulation", "duration": 22,
                "width": 960, "height": 540, "video": "/api/matches/demo/video",
                "poster": "/api/matches/demo/poster", "demo": True, "config": None,
                "state": "complete", "progress": 1, "stage": "Scripted example"}
    if not (folder / "meta.json").is_file():
        raise HTTPException(404, "Match not found.")
    meta = read(folder / "meta.json")
    if meta["state"] == "processing" and match != active:
        meta.update(state="error", stage="Analysis was interrupted. Run it again.")
    return meta


def rows_for(match):
    folder = folder_for(match)
    path = folder / "events.json"
    if path.is_file():
        return read(path)
    result = folder / "result.json"
    return read(result)["events"] if result.is_file() else []


def report(match):
    meta = metadata(match)
    rows = [{**row, "piece": event_piece(row), **({"penalty_points": penalty_points(row)} if row["kind"] == "penalty" else {})}
            for row in sorted(rows_for(match), key=lambda row: row["time"])]
    config = meta.get("config") or {}
    primary = config.get("team", "")
    summaries = {
        seed["team"]: summarize([row for row in rows if row.get("team", primary) == seed["team"]], seed.get("alliance"))
        for seed in ([config, *config.get("robots", [])] if config else []) if seed.get("team")
    }
    return {**meta, "events": rows, "team_summaries": summaries,
            "summary": summarize([row for row in rows if row.get("team", primary) == primary], config.get("alliance")),
            "manual": MANUAL, "ruleset": "BIOBUZZ TU08"}


def media_folder(meta):
    return folder_for(meta.get("media_id", meta["id"]))


def already_logged(candidate, existing):
    if candidate["kind"] != existing["kind"] or candidate.get("team") != existing.get("team"):
        return False
    difference = abs(candidate["time"] - existing["time"])
    if candidate["kind"] == "hive_tip" and existing["source"] == "manual":
        return difference < 2.5
    if candidate["kind"].endswith("_launch"):
        return difference < .075
    if candidate["kind"] == "cell_entry":
        if event_piece(candidate) != event_piece(existing):
            return False
        if "launch_time" in candidate and "launch_time" in existing:
            return abs(candidate["launch_time"] - existing["launch_time"]) < .12
        return difference < .12
    if candidate["kind"] in BALL_EVENTS and difference < .6:
        if "launch_time" in candidate and "launch_time" in existing:
            return abs(candidate["launch_time"] - existing["launch_time"]) < .24
        return candidate["title"] == existing["title"] or (
            existing["source"] == "manual" and event_piece(candidate) == event_piece(existing)
        )
    if candidate["kind"] == "nectar_pickup":
        return difference < .45 and event_piece(candidate) == event_piece(existing)
    return difference < .5 and candidate["title"] == existing["title"]


@app.get("/api/options")
def options():
    return {"models": {key: (ROOT / "models" / name).is_file() for key, name in WEIGHTS.items()},
            "sam": (ROOT / "models/sam2.1_t.pt").is_file()}


@app.get("/api/matches")
def list_matches():
    reports = []
    for path in MATCHES.glob("*/meta.json"):
        match = path.parent.name
        meta = metadata(match)
        fouls = [row for row in rows_for(match) if row["kind"] == "penalty" and row["status"] == "confirmed"]
        reports.append({**meta, "penalty_points": sum(map(penalty_points, fouls))})
    return sorted(reports, key=lambda row: row.get("updated", 0), reverse=True)


@app.get("/api/library/folders")
def list_folders():
    return [{"id": key, "name": name} for key, name in library_folders().items()]


@app.post("/api/library/folders", status_code=201)
def create_folder(data: FolderName):
    return save_folder(uuid4().hex, data, create=True)


@app.patch("/api/library/folders/{folder_id}")
def rename_folder(folder_id: str, data: FolderName):
    return save_folder(folder_id, data)


@app.get("/api/matches/{match}")
def get_match(match: str):
    return report(match)


@app.patch("/api/matches/{match}")
def update_recording(match: str, data: RecordingUpdate):
    if match == "demo":
        raise HTTPException(409, "The built-in simulation stays in the library.")
    with lock:
        path = folder_for(match) / "meta.json"
        if not path.is_file():
            raise HTTPException(404, "Match not found.")
        # Keep the stored analysis state intact during library edits.
        meta = read(path)
        if "name" in data.model_fields_set:
            meta["name"] = library_name(data.name, 180)
        if "folder_id" in data.model_fields_set:
            if data.folder_id is not None and data.folder_id not in library_folders():
                raise HTTPException(404, "Folder not found.")
            meta["folder_id"] = data.folder_id
        write(path, meta)
        return metadata(match)


@app.post("/api/matches/{match}/robot-reviews", status_code=201)
def new_robot_review(match: str):
    from time import time

    with lock:
        original = metadata(match)
        if original.get("demo"):
            raise HTTPException(409, "Import a recording to review another robot.")
        source = metadata(original.get("media_id", match))
        config = original.get("config") or original.get("field_preset") or {}
        review = uuid4().hex
        folder = folder_for(review)
        folder.mkdir()
        meta = {
            "id": review, "name": f"Robot review · {source['name']}"[:180],
            "duration": source["duration"], "width": source["width"], "height": source["height"],
            "video": f"/api/matches/{review}/video", "poster": f"/api/matches/{review}/poster",
            "media_id": source["id"], "folder_id": original.get("folder_id"), "demo": False,
            "config": None, "state": "setup", "progress": 0, "stage": "Choose the next robot", "updated": time(),
            "field_preset": {
                "start_time": config.get("start_time"), "end_time": config.get("end_time"), "field": config.get("field"),
                "zones": [] if config.get("camera_motion") else config.get("zones", []),
                "camera_motion": bool(config.get("camera_motion")),
            },
        }
        try:
            write(folder / "meta.json", meta)
        except OSError:
            folder.rmdir()
            raise
        return report(review)


@app.get("/api/matches/{match}/frames")
def frames(match: str):
    metadata(match)
    path = folder_for(match) / "result.json"
    if not path.is_file():
        return {"frames": [], "warnings": [], "coverage": None, "segmentation": False}
    result = read(path)
    return {key: value for key, value in result.items() if key != "events"}


@app.get("/api/matches/{match}/video")
def video(match: str, download: bool = False):
    meta = metadata(match)
    filename = f"{meta['name']}.mp4" if download else None
    return FileResponse(media_folder(meta) / "practice.mp4", media_type="video/mp4", filename=filename)


@app.get("/api/matches/{match}/poster")
def poster(match: str):
    return FileResponse(media_folder(metadata(match)) / "poster.jpg", media_type="image/jpeg")


def saved_field_setup(digest, duration, width, height):
    for path in sorted(MATCHES.glob("*/meta.json")):
        meta = read(path)
        config = meta.get("config") or {}
        if (digest in meta.get("video_digests", ()) and meta.get("final_score")
                and meta.get("width") == width and meta.get("height") == height
                and abs(meta["duration"] - duration) < .12 and config.get("zones")
                and config.get("start_time") is not None
                and config.get("end_time") is not None and config["end_time"] <= duration):
            return {
                "final_score": meta["final_score"],
                "field_preset": {key: config.get(key) for key in ("zones", "start_time", "end_time", "field")},
            }
    return {}


@app.post("/api/matches", status_code=201)
def upload(video: UploadFile = File(...)):
    from time import time

    match = uuid4().hex
    folder = folder_for(match)
    folder.mkdir()
    source = folder / "upload"
    normalized = folder / "practice.mp4"
    try:
        size = 0
        digest = sha256()
        with source.open("wb") as out:
            for chunk in iter(lambda: video.file.read(1024 * 1024), b""):
                size += len(chunk)
                if size > UPLOAD_LIMIT:
                    raise HTTPException(413, "Choose a video under 2 GB.")
                digest.update(chunk)
                out.write(chunk)
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                "-show_entries", "format=duration:stream=width,height,avg_frame_rate", "-of", "json", str(source)],
                               capture_output=True, text=True, timeout=20)
        info = json.loads(probe.stdout) if probe.returncode == 0 else {}
        duration = float(info.get("format", {}).get("duration") or 0)
        if not info.get("streams") or not math.isfinite(duration) or not 0 < duration <= 3600:
            raise HTTPException(422, "Choose a readable match video under one hour.")
        try:
            native_fps = float(Fraction(info["streams"][0].get("avg_frame_rate", "30")))
        except (ValueError, ZeroDivisionError):
            native_fps = 30
        output_fps = min(60, max(12, round(native_fps)))
        convert = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", "0:v:0",
                                  "-map", "0:a:0?", "-vf", f"scale=w='min(iw,1280)':h=-2,fps={output_fps}",
                                  "-c:v", "libx264", "-threads", "3", "-preset", "veryfast", "-crf", "21",
                                  "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(normalized)],
                                 capture_output=True, timeout=1800)
        if convert.returncode:
            raise HTTPException(422, "This video could not be converted for playback.")
        source.unlink()
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(normalized), "-frames:v", "1",
                        str(folder / "poster.jpg")], check=True, capture_output=True, timeout=30)
        normalized_info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                          "stream=width,height:format=duration", "-of", "json", str(normalized)],
                                         capture_output=True, text=True, check=True, timeout=20)
        dimensions = json.loads(normalized_info.stdout)
        width, height = (dimensions["streams"][0][k] for k in ("width", "height"))
        duration = float(dimensions["format"]["duration"])
        source_digest = digest.hexdigest()
        with normalized.open("rb") as playback:
            playback_digest = file_digest(playback, "sha256").hexdigest()
        write(folder / "meta.json", {"id": match, "name": Path(video.filename or "Match video").stem[:180],
              "duration": duration, "width": width, "height": height, "video": f"/api/matches/{match}/video",
              "poster": f"/api/matches/{match}/poster", "demo": False, "config": None,
              "state": "setup", "stage": "Ready to calibrate", "progress": 0, "updated": time(),
              "video_digests": [source_digest, playback_digest],
              **saved_field_setup(source_digest, duration, width, height)})
        return report(match)
    except HTTPException:
        shutil.rmtree(folder)
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        shutil.rmtree(folder)
        raise HTTPException(422, "Video import failed. Check the format and available disk space.") from exc
    finally:
        video.file.close()


def run_model_worker(source, config, progress, launch_source=None, hive_weights=None):
    # A blocked model or file read must not leave the job active indefinitely.
    with TemporaryDirectory(prefix="pitlane-analysis-") as directory:
        job = Path(directory)
        write(job / "config.json", config.model_dump())
        status_path = job / "progress.json"
        command = [sys.executable, "-m", "backend.analysis_worker", str(source), str(job)]
        if launch_source is not None or hive_weights is not None:
            command.append(str(launch_source or source))
        if hive_weights is not None:
            command.append(str(hive_weights))
        with subprocess.Popen(command, cwd=ROOT) as process:
            latest = None
            deadline = monotonic() + ANALYSIS_IDLE_TIMEOUT
            try:
                while process.poll() is None:
                    if status_path.is_file():
                        status = read(status_path)
                        if status != latest:
                            latest = status
                            deadline = monotonic() + ANALYSIS_IDLE_TIMEOUT
                            progress(*status)
                    if monotonic() >= deadline:
                        stage = latest[1] if latest else "starting the model worker"
                        raise RuntimeError(f"Analysis stalled while {stage.lower()}. Run it again from setup.")
                    sleep(.25)
                if process.returncode:
                    error_path = job / "error.json"
                    message = read(error_path) if error_path.is_file() else "The model worker stopped unexpectedly. Run analysis again."
                    raise RuntimeError(message)
                return read(job / "result.json")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()


def run_analysis(match, config):
    global active
    folder = folder_for(match)

    def progress(amount, stage):
        with lock:
            meta = metadata(match)
            write(folder / "meta.json", {**meta, "progress": round(amount, 3), "stage": stage})

    try:
        media = media_folder(metadata(match))
        launch_source = media / "native-fps.mp4"
        hive_weights = media / "hive-pollen.pt"
        result = run_model_worker(media / "practice.mp4", config, progress,
                                  launch_source if launch_source.is_file() else None,
                                  hive_weights if hive_weights.is_file() else None)
        with lock:
            previous = [row for row in rows_for(match) if row["status"] != "suggested" or row["source"] != "vision"]
            new_rows = [row for row in result["events"] if not any(already_logged(
                row, {**old, "team": old.get("team", config.team)} if config.robots else old
            ) for old in previous)]
            effective = result.pop("effective_setup", {})
            write(folder / "result.json", result)
            write(folder / "events.json", previous + new_rows)
            meta = metadata(match)
            write(folder / "meta.json", {**meta, "config": {**meta["config"], **effective},
                                         "state": "complete", "stage": "Ready for review", "progress": 1})
    except Exception as exc:
        logger.exception("Match analysis failed")
        with lock:
            meta = metadata(match)
            write(folder / "meta.json", {**meta, "state": "error", "stage": str(exc)[:240] or "Analysis failed."})
    finally:
        with lock:
            active = None


@app.post("/api/matches/{match}/analyze", status_code=202)
def analyze_match(match: str, config: Analysis):
    global active
    if match == "demo":
        raise HTTPException(409, "Import a video to run the detector.")
    with lock:
        meta = metadata(match)
        if active is not None:
            raise HTTPException(409, "One analysis is already running. Wait for it to finish.")
        if not (ROOT / "models" / WEIGHTS[config.model]).is_file():
            raise HTTPException(422, "The selected detector weights are unavailable.")
        if config.robots:
            roster = [config, *config.robots]
            teams = [seed.team for seed in roster]
            if len(set(teams)) != len(teams) or any(not re.fullmatch(r"[0-9]{1,6}", team) for team in teams):
                raise HTTPException(422, "Mark each robot with a different numeric team number.")
            if any(seed.seed_time >= (config.end_time or meta["duration"]) - .1 for seed in roster):
                raise HTTPException(422, "Mark every robot before the end of the match.")
        previous = meta.get("config") or {}
        if previous and (previous.get("alliance") != config.alliance or
                         previous.get("team") and previous["team"] != config.team) and any(
                             row["status"] == "confirmed" for row in rows_for(match)
                         ):
            raise HTTPException(409, "Use Review another robot to keep the existing review separate.")
        if config.seed_time >= meta["duration"] or (
            config.end_time is not None and not config.seed_time + .1 < config.end_time <= meta["duration"]
        ) or (config.start_time is not None and config.start_time >= (config.end_time or meta["duration"])):
            raise HTTPException(422, "Check the robot frame, match start, and settled end time.")
        write(folder_for(match) / "meta.json", {**meta, "config": config.model_dump(), "state": "processing",
                                               "stage": "Preparing models", "progress": 0})
        active = match
        workers.submit(run_analysis, match, config)
        return report(match)


@app.post("/api/matches/{match}/events", status_code=201)
def add_event(match: str, data: ManualEvent):
    meta = metadata(match)
    config = meta.get("config") or {}
    if data.team is not None and data.team not in {seed.get("team") for seed in [config, *config.get("robots", [])]}:
        raise HTTPException(422, "Select a marked team for this event.")
    if (data.kind not in EVENT_KINDS or data.time > meta["duration"] or not data.title.strip()
            or (data.kind == "penalty") != (data.severity in FOUL_POINTS)
            or (data.card is not None and data.kind != "penalty")
            or (data.kind == "nectar_pickup" and data.piece not in NECTAR_COLORS)
            or (data.kind == "cell_entry" and data.piece is None)
            or (data.piece is not None and data.kind not in PIECE_EVENTS | {"nectar_pickup"})):
        raise HTTPException(422, "Check the event category, time, and penalty details.")
    if data.kind == "penalty":
        rule = data.rule.strip().upper()
        if rule == "G408" and data.severity != "warning":
            raise HTTPException(422, "G408 is a warning. Add a yellow card if the referee called strategic control.")
        if rule == "G410" and data.severity != "major":
            raise HTTPException(422, "G410 is a major foul per early Nectar.")
    with lock:
        events = rows_for(match)
        link = {}
        if data.launch_id is not None:
            launch = next((row for row in events if row["id"] == data.launch_id), None)
            primary = config.get("team", "")
            if (data.kind not in PIECE_EVENTS or launch is None or launch["kind"] not in SHOT_EVENTS
                    or launch["status"] != "confirmed" or event_piece(launch) != data.piece
                    or (data.team or primary) != launch.get("team", primary) or launch["time"] > data.time):
                raise HTTPException(422, "Link a reviewed launch of the same piece by the same team.")
            previous = linked_count(events, data.launch_id, data.kind)
            if previous + data.count > launch["count"]:
                raise HTTPException(422, "Those launched pieces are already linked to a scoring event.")
            link = {"launch_id": launch["id"], "launch_time": launch["time"]}
        row = event(data.kind, data.time, data.title.strip(), data.rule.strip(), status="confirmed",
                    count=data.count, severity=data.severity, evidence=data.evidence.strip(),
                    source="manual", card=data.card, piece=data.piece)
        events.append({**row, **link, **({"team": data.team} if data.team is not None else {})})
        write(folder_for(match) / "events.json", events)
        return report(match)


def linked_count(rows, launch_id, kind, exclude=None):
    return sum(row["count"] for row in rows if row["id"] != exclude and row.get("launch_id") == launch_id
               and row["status"] != "dismissed" and (row["kind"] == "cell_entry") == (kind == "cell_entry"))


@app.patch("/api/matches/{match}/events/{row_id}")
def decide(match: str, row_id: str, decision: Decision):
    metadata(match)
    with lock:
        rows = rows_for(match)
        for row in rows:
            if row["id"] == row_id:
                if row.get("launch_id") and decision.status == "confirmed":
                    launch = next((other for other in rows if other["id"] == row["launch_id"]), None)
                    if launch is None or launch["status"] != "confirmed" or event_piece(launch) != (decision.piece or event_piece(row)):
                        raise HTTPException(422, "Confirm the matching launch before counting this scored piece.")
                    linked = linked_count(rows, row["launch_id"], row["kind"], exclude=row_id)
                    if linked + row["count"] > launch["count"]:
                        raise HTTPException(422, "Those launched pieces are already linked to a scoring event.")
                if decision.piece is not None:
                    if row["kind"] not in PIECE_EVENTS:
                        raise HTTPException(422, "Only a scored piece can be classified here.")
                    row["piece"] = decision.piece
                row["status"] = decision.status
                if row["kind"] in SHOT_EVENTS and decision.status != "confirmed":
                    for linked in rows:
                        if linked.get("launch_id") == row_id:
                            linked["status"] = "dismissed"
                write(folder_for(match) / "events.json", rows)
                return report(match)
    raise HTTPException(404, "Event not found.")


dist = Path(os.environ.get("PITLANE_UI_DIST", ROOT / "ui/dist"))
if (dist / "index.html").is_file():
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
