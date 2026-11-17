"""Search YouTube for FTC BIOBUZZ match footage and download a match-length set.

Each video is saved as data/videos/<youtube-id>.mp4 and listed in
data/videos/manifest.json with its title, channel, length and source URL.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUERIES = [
    "FTC BIOBUZZ match",
    "FTC BIOBUZZ qualification match",
    "FTC BIOBUZZ league meet",
    "FTC BIOBUZZ robot reveal",
    "FIRST Tech Challenge BIOBUZZ 2026",
    "FTC BIOBUZZ scrimmage",
    "FTC BIOBUZZ playoff",
    "FTC BIOBUZZ hive pollen",
]


def search(query, limit):
    command = [sys.executable, "-m", "yt_dlp", "--flat-playlist", "--dump-json", "--no-warnings",
               f"ytsearch{limit}:{query}"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    rows = []
    for line in result.stdout.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if result.returncode and not rows:
        print(f"Search failed for {query!r}: {result.stderr.strip()[-300:]}", file=sys.stderr)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/videos")
    parser.add_argument("--per-query", type=int, default=25)
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--min-seconds", type=int, default=140)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--ids", default="", help="Comma-separated YouTube IDs to keep, after reviewing the search")
    args = parser.parse_args()
    keep = {value.strip() for value in args.ids.split(",") if value.strip()}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    candidates = {}
    for query in QUERIES:
        for row in search(query, args.per_query):
            duration = row.get("duration") or 0
            title = row.get("title") or ""
            if not args.min_seconds <= duration <= args.max_seconds:
                continue
            if "biobuzz" not in title.lower() and "bio buzz" not in title.lower():
                continue
            if keep and row["id"] not in keep:
                continue
            candidates[row["id"]] = {"title": title, "channel": row.get("channel") or row.get("uploader"),
                                     "duration": duration, "url": f"https://www.youtube.com/watch?v={row['id']}"}
    print(f"{len(candidates)} BIOBUZZ candidates found")
    for vid, info in candidates.items():
        print(f"{vid}\t{info['duration']:>4}s\t{info['channel']}\t{info['title']}")
    if args.search_only:
        return

    for vid, info in candidates.items():
        if len([key for key, value in manifest.items() if value.get("file")]) >= args.max_videos:
            break
        target = args.output / f"{vid}.mp4"
        if target.is_file():
            manifest.setdefault(vid, info)["file"] = target.name
            continue
        command = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-playlist",
                   "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]",
                   "--merge-output-format", "mp4", "-o", str(args.output / f"{vid}.%(ext)s"), info["url"]]
        result = subprocess.run(command, capture_output=True, text=True, timeout=1200)
        if result.returncode == 0 and target.is_file():
            manifest[vid] = {**info, "file": target.name}
            print(f"Downloaded {vid}: {info['title']}")
        else:
            print(f"Skipped {vid}: {result.stderr.strip()[-200:]}", file=sys.stderr)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"{sum(1 for value in manifest.values() if value.get('file'))} videos ready in {args.output}")


if __name__ == "__main__":
    main()
