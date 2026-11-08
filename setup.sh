#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

for tool in uv npm ffmpeg ffprobe curl; do
  if ! command -v "$tool" >/dev/null; then
    printf 'Missing required tool: %s\n' "$tool" >&2
    exit 1
  fi
done

runtime_dir="${PITLANE_RUNTIME_DIR:-$HOME/Library/Application Support/Pitlane/runtime}"
ui_dist="${PITLANE_UI_DIST:-$(dirname "$runtime_dir")/ui-build/dist}"
export PYTHONPYCACHEPREFIX="$runtime_dir/bytecode"
mkdir -p "$(dirname "$runtime_dir")"
uv venv --allow-existing --python 3.12 "$runtime_dir"
uv pip install --python "$runtime_dir/bin/python" -r requirements.txt
mkdir -p "$ui_dist"
ui_dist="$(cd "$ui_dist" && pwd)"
(
  cd ui
  npm ci
  npm run build -- --outDir "$ui_dist"
)

mkdir -p models
fetch_weights() {
  local name="$1"
  if [[ ! -s "models/$name" ]]; then
    curl --fail --location --retry 2 "https://github.com/ultralytics/assets/releases/download/v8.4.0/$name" \
      --output "models/$name.partial"
    mv "models/$name.partial" "models/$name"
  fi
}
fetch_weights yolo26n.pt
fetch_weights sam2.1_t.pt
for asset in nanotrack_backbone_sim.onnx nanotrack_head_sim.onnx; do
  if [[ ! -s "models/$asset" ]]; then
    curl --fail --location --retry 2 \
      "https://raw.githubusercontent.com/HonglinChu/SiamTrackers/248663fde6bf7c40190cf10ee396d5662919ecd3/NanoTrack/models/nanotrackv2/$asset" \
      --output "models/$asset.partial"
    mv "models/$asset.partial" "models/$asset"
  fi
done

if [[ ! -s data/demo/result.json ]]; then
  "$runtime_dir/bin/python" -m scripts.demo video
fi
