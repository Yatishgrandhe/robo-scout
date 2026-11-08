#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

runtime_dir="${PITLANE_RUNTIME_DIR:-$HOME/Library/Application Support/Pitlane/runtime}"
ui_dist="${PITLANE_UI_DIST:-$(dirname "$runtime_dir")/ui-build/dist}"
if [[ ! -f "$ui_dist/index.html" && -z "${PITLANE_UI_DIST:-}" ]]; then
  ui_dist="$PWD/ui/dist"
fi
if [[ ! -x "$runtime_dir/bin/python" || ! -s "$ui_dist/index.html" ]]; then
  printf 'Run ./setup.sh in this folder before opening Robo Scout.\n'
  exit 1
fi
url=http://127.0.0.1:8765
if curl -fsS "$url/api/options" >/dev/null 2>&1; then
  open "$url"
  exit 0
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPYCACHEPREFIX="$runtime_dir/bytecode"
export PITLANE_UI_DIST="$ui_dist"
"$runtime_dir/bin/python" -m uvicorn backend.main:app --host 127.0.0.1 --port 8765 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
for attempt in {1..30}; do
  if curl -fsS "$url/api/options" >/dev/null 2>&1; then
    open "$url"
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    exit 1
  fi
  sleep .2
done
wait "$server_pid"
